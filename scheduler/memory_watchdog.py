"""
Memory Leak Prevention Watchdog (5.1)
Monitors GPU memory for zombie containers and enforces cleanup
"""
import logging
import subprocess
import threading
import time
import json
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GPUMemoryUsage:
    """GPU memory usage info"""
    pid: int
    used_memory_mb: float
    timestamp: str


@dataclass
class ZombieContainer:
    """Zombie container tracking"""
    container_id: int
    pid: int
    used_memory_mb: float
    first_detected: float
    grace_period_seconds: int = 60


class MemoryWatchdog:
    """
    Requirement 5.1: Memory Leak Prevention
    Polls nvidia-smi every 30 seconds, detects zombies, force-kills after grace period
    """

    def __init__(self, state_tracker, memory_manager, grace_period: int = 60):
        self.state_tracker = state_tracker
        self.memory_manager = memory_manager
        self.grace_period_seconds = grace_period
        self.running = False
        self.watchdog_thread = None

        # Track zombie containers
        self.zombie_containers: Dict[int, ZombieContainer] = {}
        self.force_kill_log = []
        self.poll_interval = 30  # seconds
        self.lock = threading.Lock()

        logger.info(f"Memory Watchdog initialized (grace_period={grace_period}s, poll_interval={self.poll_interval}s)")

    def start(self):
        """Start watchdog thread"""
        if self.running:
            logger.warning("Watchdog already running")
            return

        self.running = True
        self.watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self.watchdog_thread.start()
        logger.info("Memory Watchdog started")

    def stop(self):
        """Stop watchdog thread"""
        self.running = False
        if self.watchdog_thread:
            self.watchdog_thread.join(timeout=5)
        logger.info("Memory Watchdog stopped")

    def _watchdog_loop(self):
        """Main watchdog loop - polls every 30 seconds"""
        while self.running:
            try:
                self._poll_gpu_memory()
                self._check_for_zombies()
                time.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"Watchdog error: {e}")

    def _poll_gpu_memory(self) -> List[GPUMemoryUsage]:
        """
        Requirement 5.1: Poll nvidia-smi every 30 seconds
        Queries: pid, used_memory
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,nounits,noheader"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                logger.warning("nvidia-smi query failed")
                return []

            gpu_processes = []
            timestamp = datetime.now().isoformat()

            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue

                try:
                    parts = line.split(',')
                    pid = int(parts[0].strip())
                    used_memory_mb = float(parts[1].strip())

                    gpu_processes.append(GPUMemoryUsage(
                        pid=pid,
                        used_memory_mb=used_memory_mb,
                        timestamp=timestamp
                    ))
                except (ValueError, IndexError):
                    continue

            logger.debug(f"GPU memory poll: {len(gpu_processes)} processes using GPU")
            return gpu_processes

        except Exception as e:
            logger.error(f"Error polling GPU memory: {e}")
            return []

    def _check_for_zombies(self):
        """
        Requirement 5.1: Detect containers that should have exited but still hold GPU memory
        Force-terminate after grace period (60 seconds)
        Log all force-kills with full state dump
        """
        try:
            gpu_processes = self._poll_gpu_memory()
            gpu_pids = {p.pid for p in gpu_processes}

            with self.lock:
                # Check completed containers
                for container_id, container in list(self.state_tracker.containers.items()):
                    if container.state.value in ['completed', 'failed']:
                        # Container should not be holding GPU memory
                        if container.process_id and container.process_id in gpu_pids:
                            if container_id not in self.zombie_containers:
                                logger.warning(
                                    f"🧟 ZOMBIE DETECTED: Container {container_id} (PID {container.process_id}) "
                                    f"still holds GPU memory after completion"
                                )
                                self.zombie_containers[container_id] = ZombieContainer(
                                    container_id=container_id,
                                    pid=container.process_id,
                                    used_memory_mb=next(
                                        (p.used_memory_mb for p in gpu_processes if p.pid == container.process_id),
                                        0
                                    ),
                                    first_detected=time.time(),
                                    grace_period_seconds=self.grace_period_seconds
                                )
                            else:
                                # Check if grace period exceeded
                                zombie = self.zombie_containers[container_id]
                                elapsed = time.time() - zombie.first_detected

                                if elapsed > zombie.grace_period_seconds:
                                    logger.error(
                                        f"FORCE-KILLING ZOMBIE: Container {container_id} (PID {zombie.pid}) "
                                        f"after {self.grace_period_seconds}s grace period"
                                    )
                                    self._force_kill_container(container_id, zombie)
                        else:
                            # Container not in GPU memory - cleanup
                            if container_id in self.zombie_containers:
                                del self.zombie_containers[container_id]

        except Exception as e:
            logger.error(f"Error checking for zombies: {e}")

    def _force_kill_container(self, container_id: int, zombie: ZombieContainer):
        """
        Requirement 5.1: Force-terminate zombie container
        Log all force-kills with full state dump
        """
        try:
            # Get full state dump before kill
            container = self.state_tracker.get_container_info(container_id)
            state_dump = {
                "timestamp": datetime.now().isoformat(),
                "container_id": container_id,
                "action": "force_kill",
                "reason": "GPU memory leak - zombie process",
                "grace_period_seconds": self.grace_period_seconds,
                "pid": zombie.pid,
                "gpu_memory_held_mb": zombie.used_memory_mb,
                "container_state": {
                    "state": container.state.value if container else None,
                    "memory_mb": container.memory_mb if container else None,
                    "duration": container.duration_seconds if container else None,
                    "state_transitions": [
                        {
                            "timestamp": t.timestamp,
                            "from_state": t.from_state,
                            "to_state": t.to_state,
                            "message": t.message
                        }
                        for t in container.state_transitions
                    ] if container else []
                }
            }

            # Log full state dump
            logger.error(f"FORCE-KILL STATE DUMP:\n{json.dumps(state_dump, indent=2)}")
            self.force_kill_log.append(state_dump)

            # Actually kill the process
            try:
                subprocess.run(["kill", "-9", str(zombie.pid)], timeout=5)
                logger.error(f"Force-killed PID {zombie.pid}")
            except Exception as e:
                logger.error(f"Failed to kill PID {zombie.pid}: {e}")

            # Clean up tracking
            del self.zombie_containers[container_id]

        except Exception as e:
            logger.error(f"Error force-killing container {container_id}: {e}")

    def get_zombie_containers(self) -> List[dict]:
        """Get current list of zombie containers"""
        with self.lock:
            return [
                {
                    "container_id": z.container_id,
                    "pid": z.pid,
                    "gpu_memory_mb": z.used_memory_mb,
                    "seconds_detected": time.time() - z.first_detected,
                    "grace_period_seconds": z.grace_period_seconds
                }
                for z in self.zombie_containers.values()
            ]

    def get_force_kill_log(self) -> List[dict]:
        """Get log of all force-killed containers"""
        with self.lock:
            return list(self.force_kill_log)
