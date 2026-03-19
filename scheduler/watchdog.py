"""
Watchdog Process - GPU Memory Leak Prevention (Requirement 5.1)

Implements:
- Polls nvidia-smi every 30 seconds
- Detects zombie containers still holding GPU memory
- Force-terminates after 60-second grace period
- Logs all force-kills with full state dump
"""
import logging
import subprocess
import time
import threading
import os
import signal
from typing import Dict, Set
from datetime import datetime

logger = logging.getLogger(__name__)

# Import ContainerState at module level to avoid repeated imports
try:
    from .state_tracker import ContainerState
except ImportError:
    # Handle case where watchdog is imported in non-package context
    ContainerState = None


class GPUWatchdog:
    """Monitors GPU memory and force-terminates zombie containers"""

    def __init__(
        self,
        poll_interval_seconds: int = 30,
        grace_period_seconds: int = 60,
        state_tracker=None
    ):
        """
        Initialize watchdog

        Args:
            poll_interval_seconds: How often to poll nvidia-smi (default 30s)
            grace_period_seconds: Grace period before force-kill (default 60s)
            state_tracker: Reference to StateTracker for getting expected containers
        """
        self.poll_interval = poll_interval_seconds
        self.grace_period = grace_period_seconds
        self.state_tracker = state_tracker

        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        # Track containers that should have exited
        # Format: {pid: (timestamp_last_seen, expected_exit_time)}
        self.zombie_suspects: Dict[int, float] = {}

        logger.info(f"Watchdog initialized: poll={poll_interval_seconds}s, grace={grace_period_seconds}s")

    def start(self):
        """Start watchdog in background thread"""
        with self.lock:
            if self.running:
                logger.warning("Watchdog already running")
                return

            self.running = True
            self.thread = threading.Thread(target=self._watch_loop, daemon=True)
            self.thread.start()
            logger.info("Watchdog started")

    def stop(self):
        """Stop watchdog gracefully"""
        with self.lock:
            if not self.running:
                return
            self.running = False

        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Watchdog stopped")

    def _watch_loop(self):
        """Main watchdog loop - runs in background thread"""
        while self.running:
            try:
                self._check_gpu_memory()
                time.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"Watchdog error: {e}", exc_info=True)
                time.sleep(self.poll_interval)

    def _check_gpu_memory(self):
        """
        Poll nvidia-smi to check for zombie containers
        Requirement 5.1: Polls every 30 seconds
        """
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,nounits,noheader"
                ],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                logger.debug("nvidia-smi not available or no GPU processes")
                return

            # Parse nvidia-smi output
            gpu_pids: Set[int] = set()
            gpu_memory: Dict[int, float] = {}

            if result.stdout.strip():
                for line in result.stdout.strip().split('\n'):
                    if not line.strip():
                        continue
                    try:
                        parts = line.split(',')
                        pid = int(parts[0].strip())
                        memory_mb = float(parts[1].strip())
                        gpu_pids.add(pid)
                        gpu_memory[pid] = memory_mb
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Could not parse nvidia-smi line: {line}")
                        continue

            # Check for containers that should have exited but still hold memory
            self._detect_zombies(gpu_pids, gpu_memory)

        except subprocess.TimeoutExpired:
            logger.warning("nvidia-smi check timed out")
        except Exception as e:
            logger.error(f"Error checking GPU memory: {e}")

    def _detect_zombies(self, current_gpu_pids: Set[int], gpu_memory: Dict[int, float]):
        """
        Detect zombie containers (should have exited but still hold GPU memory)
        Requirement 5.1: Force-terminate after 60-second grace period
        """
        current_time = time.time()

        if not self.state_tracker or ContainerState is None:
            return

        # Get all containers that should have completed
        completed_containers = []
        with self.state_tracker.lock:
            for container_info in self.state_tracker.containers.values():
                # Container is completed if it reached COMPLETED or FAILED state
                if container_info.state in [ContainerState.COMPLETED, ContainerState.FAILED]:
                    completed_containers.append(container_info)

        # Check if any completed containers still hold GPU memory
        for container in completed_containers:
            if not container.process_id:
                continue

            pid = container.process_id

            # If this PID is still in GPU processes list
            if pid in current_gpu_pids:
                memory_mb = gpu_memory.get(pid, 0)

                # Check if we've seen this zombie before
                if pid not in self.zombie_suspects:
                    # First detection
                    self.zombie_suspects[pid] = current_time
                    logger.warning(
                        f"ZOMBIE DETECTED: Container {container.container_id} (PID {pid}) "
                        f"completed but still holds {memory_mb:.1f}MB GPU memory. "
                        f"Grace period: {self.grace_period}s"
                    )
                else:
                    # Check if grace period exceeded
                    time_held = current_time - self.zombie_suspects[pid]
                    if time_held >= self.grace_period:
                        # Force-terminate the zombie
                        self._force_terminate_zombie(
                            container,
                            pid,
                            memory_mb,
                            time_held
                        )
                        del self.zombie_suspects[pid]
                    else:
                        remaining = self.grace_period - time_held
                        logger.debug(
                            f"Zombie container {container.container_id} (PID {pid}) "
                            f"held for {time_held:.1f}s, "
                            f"{remaining:.1f}s until force-kill"
                        )
            else:
                # Zombie was cleaned up naturally
                if pid in self.zombie_suspects:
                    logger.info(f"Zombie PID {pid} cleaned up naturally")
                    del self.zombie_suspects[pid]

    def _force_terminate_zombie(
        self,
        container,
        pid: int,
        memory_mb: float,
        time_held: float
    ):
        """
        Force-terminate zombie container
        Requirement 5.1: Force-kill after grace period with full state dump
        SAGEMAKER FIX: Be more aggressive to ensure process termination
        """
        logger.critical(
            f"FORCE-TERMINATING ZOMBIE: Container {container.container_id} (PID {pid}) "
            f"held GPU memory for {time_held:.1f}s, using {memory_mb:.1f}MB. "
            f"Grace period {self.grace_period}s exceeded."
        )

        # Log full state dump
        state_dump = {
            "container_id": container.container_id,
            "pid": pid,
            "state": container.state.value if hasattr(container.state, 'value') else str(container.state),
            "memory_allocated_mb": container.memory_mb,
            "memory_held_gpu_mb": memory_mb,
            "time_held_seconds": time_held,
            "grace_period_seconds": self.grace_period,
            "timestamp": datetime.now().isoformat(),
        }

        logger.critical(f"STATE DUMP: {state_dump}")

        try:
            # Try SIGTERM first (graceful)
            logger.warning(f"Attempting SIGTERM on PID {pid}")
            os.kill(pid, signal.SIGTERM)

            # Wait briefly for graceful shutdown
            time.sleep(1)

            # Check if process still exists
            try:
                os.kill(pid, 0)  # Signal 0 checks if process exists
                # Still alive, escalate to SIGKILL
                logger.critical(f"Process {pid} not terminated by SIGTERM, sending SIGKILL")
                os.kill(pid, signal.SIGKILL)

                # Wait for SIGKILL to take effect
                time.sleep(0.5)

                # Verify it's dead
                try:
                    os.kill(pid, 0)
                    logger.error(f"Process {pid} survived SIGKILL (anomaly)")
                except ProcessLookupError:
                    logger.info(f"Process {pid} terminated successfully (SIGKILL)")

            except ProcessLookupError:
                # Process already terminated
                logger.info(f"Process {pid} terminated (SIGTERM worked)")

        except ProcessLookupError:
            logger.info(f"Process {pid} not found (already terminated)")
        except PermissionError:
            logger.error(f"Permission denied to terminate PID {pid} - may need elevated privileges")
        except Exception as e:
            logger.error(f"Error terminating PID {pid}: {e}", exc_info=True)

    def get_stats(self) -> dict:
        """Get watchdog statistics"""
        with self.lock:
            return {
                "running": self.running,
                "poll_interval_seconds": self.poll_interval,
                "grace_period_seconds": self.grace_period,
                "zombie_suspects_count": len(self.zombie_suspects),
            }
