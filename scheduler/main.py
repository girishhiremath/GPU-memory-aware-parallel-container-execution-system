"""
Main Scheduler - Non-blocking scheduler that runs in main thread
Manages container launching and memory allocation
"""
import logging
import time
import json
import threading
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

import sys
import os

# Support both direct execution and module import
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from memory_manager import MemoryManager
    from state_tracker import StateTracker, ContainerState, SystemState
    from container_runner import ContainerRunner, ContainerRunConfig
    from csv_reporter import CSVReporter
    from watchdog import GPUWatchdog
    from log_setup import setup_logging, log_config_summary, log_dynamic_reset_config
    from log_setup import log_container_launch, log_container_queued, log_container_from_queue
    from log_setup import log_container_completed, log_system_event
    # Import config loader
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config_loader import ConfigLoader
else:
    from .memory_manager import MemoryManager
    from .state_tracker import StateTracker, ContainerState, SystemState
    from .container_runner import ContainerRunner, ContainerRunConfig
    from .log_setup import setup_logging, log_config_summary, log_dynamic_reset_config
    from .log_setup import log_container_launch, log_container_queued, log_container_from_queue
    from .log_setup import log_container_completed, log_system_event
    from .csv_reporter import CSVReporter
    from .watchdog import GPUWatchdog
    # Import config loader
    from config_loader import ConfigLoader

logger = logging.getLogger(__name__)


@dataclass
class SchedulerConfig:
    """
    Scheduler configuration

    QUICK RUN MODE (for testing):
    Use these values for ~5 minute test run:
    - total_gpu_memory_mb=4096 (GPU memory)
    - container_duration_seconds=10 (quick test)
    - step_interval_seconds=5 (faster scheduling)
    - max_concurrent_containers=2 (fewer containers)
    - memory_multiplier=1.5 (memory growth)
    - base_memory_mb=128 (start smaller)
    - simulation_duration_hours=0.1 (6 minutes total)
    """
    total_gpu_memory_mb: float = 4096          # Total GPU memory available (MB)
    container_duration_seconds: int = 600      # How long each container runs (10 minutes)
    step_interval_seconds: int = 5             # How often scheduler checks
    max_concurrent_containers: int = 3         # Max containers running at once
    memory_multiplier: float = 1.5             # Memory growth factor (Container N = Container N-1 * 1.5)
    base_memory_mb: float = 862                # First container memory
    simulation_duration_hours: float = 24      # Total run time (24 hours)
    worker_script: str = "worker/worker.py"    # Worker script path


class Scheduler:
    """Main scheduler - runs in non-blocking mode"""

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.memory_manager = MemoryManager(config.total_gpu_memory_mb)
        self.state_tracker = StateTracker(config.max_concurrent_containers)
        self.container_runner = ContainerRunner(config.max_concurrent_containers)
        self.csv_reporter = CSVReporter()

        # Initialize watchdog for memory leak prevention (Requirement 5.1)
        self.watchdog = GPUWatchdog(
            poll_interval_seconds=30,
            grace_period_seconds=60,
            state_tracker=self.state_tracker
        )

        self.running = False
        self.start_time = time.time()
        self.end_time = self.start_time + (config.simulation_duration_hours * 3600)
        self.last_launch_time = 0
        self.lock = threading.Lock()

        # Ready queue for containers waiting for slot or memory
        self.ready_queue = []  # FIFO queue of containers waiting to launch
        self.next_container_id = 1  # Track next container to create

        # Calculate dynamic reset values
        import math
        base = config.base_memory_mb
        mult = config.memory_multiplier
        gpu = config.total_gpu_memory_mb

        # Find the container index that would first exceed GPU memory alone
        self.max_container_index = int(math.floor(1 + math.log(gpu / base) / math.log(mult))) if base > 0 and base <= gpu and mult > 1 else 1

        # Find max simultaneous containers
        self.max_simultaneous = 1
        total_memory = 0
        for i in range(min(self.max_container_index, config.max_concurrent_containers)):
            mem = base * (mult ** i)
            if total_memory + mem <= gpu:
                total_memory += mem
                self.max_simultaneous = i + 1
            else:
                break

        # Calculate cycle memory (sum of containers in one cycle)
        self.cycle_memory = total_memory

        logger.info("=" * 80)
        logger.info("DYNAMIC RESET CONFIGURATION (computed at startup)")
        logger.info(f"  Base Memory: {base} MB")
        logger.info(f"  Memory Multiplier: {mult}x")
        logger.info(f"  Total GPU Memory: {gpu} MB")
        logger.info(f"  Dynamic Reset Point: Container {self.max_container_index}")
        logger.info(f"    → Would require {base * (mult ** (self.max_container_index - 1)):.1f} MB")
        logger.info(f"    → Exceeds {gpu} MB GPU")
        logger.info(f"  Max Simultaneous Containers: {self.max_simultaneous}")
        logger.info(f"  Cycle Memory Usage: {self.cycle_memory:.1f} MB ({self.cycle_memory / gpu * 100:.2f}% of GPU)")
        logger.info(f"  Reset Formula: container_index % {self.max_simultaneous} + 1")
        logger.info("=" * 80)

        # Pass dynamic values to reporter so all CSV reports use them
        self.csv_reporter.set_dynamic_reset_info(
            max_container_index=self.max_container_index,
            max_simultaneous=self.max_simultaneous,
            cycle_memory=self.cycle_memory,
        )

        # Setup callbacks
        self._setup_callbacks()

        logger.info("Scheduler initialized")
        logger.info(f"Config: {config}")

    def _setup_callbacks(self):
        """Setup callbacks for container runner"""
        self.container_runner.set_callbacks(
            on_start=self._on_container_start,
            on_complete=self._on_container_complete,
            on_error=self._on_container_error
        )

    def _on_container_start(self, container_id: int):
        """Called when container starts — record ALL state transitions"""
        self.state_tracker.update_container_state(container_id, ContainerState.STARTING)
        self.csv_reporter.record_state_transition(container_id, 'STARTING', time.time())

        self.state_tracker.update_container_state(container_id, ContainerState.ALLOCATING_MEMORY)
        self.csv_reporter.record_state_transition(container_id, 'ALLOCATING_MEMORY', time.time())

        self.state_tracker.update_container_state(container_id, ContainerState.RUNNING)
        self.csv_reporter.record_state_transition(container_id, 'RUNNING', time.time())

        logger.info(f"Container {container_id} started")

    def _on_container_complete(self, container_id: int, success: bool):
        """Called when container completes — record end-of-life transitions and process queue"""
        self.csv_reporter.record_state_transition(container_id, 'RELEASING_MEMORY', time.time())

        if success:
            self.csv_reporter.record_state_transition(container_id, 'COMPLETED', time.time())
            self.state_tracker.reset_consecutive_oom_failures()
        else:
            self.csv_reporter.record_state_transition(container_id, 'FAILED', time.time())

        self.state_tracker.mark_container_completed(container_id, success)
        self.csv_reporter.record_container_completion(container_id, success)

        container = self.state_tracker.get_container_info(container_id)
        if container and container.memory_block_id:
            self.memory_manager.release(container.memory_block_id)
            logger.info(f"Released memory for container {container_id}")

        # CRITICAL: After freeing memory, check if any queued containers can now launch
        self._process_queue()

    def _on_container_error(self, container_id: int, error: str):
        """Called when container encounters error"""
        logger.error(f"Container {container_id} error: {error}")
        self.state_tracker.mark_container_completed(container_id, False)

        # Release memory
        container = self.state_tracker.get_container_info(container_id)
        if container and container.memory_block_id:
            self.memory_manager.release(container.memory_block_id)

    def start(self):
        """Start the scheduler (non-blocking)"""
        with self.lock:
            if self.running:
                logger.warning("Scheduler already running")
                return

            self.running = True
            self.state_tracker.set_system_state(SystemState.RUNNING)

            # Start watchdog (Requirement 5.1)
            self.watchdog.start()

            logger.info("Scheduler started")

    def stop(self):
        """Stop the scheduler gracefully"""
        # Stop watchdog first (Requirement 5.1)
        self.watchdog.stop()

        # Generate CSV reports FIRST (before checking if running)
        # This ensures reports are created even if stop() is called multiple times
        report_directory = None
        try:
            reports = self.generate_reports()
            if reports:
                report_directory = reports.get('report_directory')
        except Exception as e:
            logger.error(f"Error generating reports during stop: {e}")

        # Save scheduler report (non-blocking attempt)
        # This is done asynchronously to avoid deadlock with container runner threads
        try:
            if report_directory:
                report_path = os.path.join(report_directory, "scheduler_report.json")
            else:
                report_path = "scheduler_report.json"

            # Create report dict without holding any locks
            report = {
                "config": {
                    "total_gpu_memory_mb": self.config.total_gpu_memory_mb,
                    "container_duration_seconds": self.config.container_duration_seconds,
                    "step_interval_seconds": self.config.step_interval_seconds,
                    "max_concurrent_containers": self.config.max_concurrent_containers,
                    "memory_multiplier": self.config.memory_multiplier,
                    "base_memory_mb": self.config.base_memory_mb,
                    "simulation_duration_hours": self.config.simulation_duration_hours,
                },
            }

            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to save scheduler report: {e}")

        with self.lock:
            if not self.running:
                # Already stopped, but we've generated reports anyway
                logger.info("Scheduler already stopped")
                return

            self.running = False
            self.state_tracker.set_system_state(SystemState.SHUTDOWN)
            logger.info("Scheduler stopping...")
            self.state_tracker.set_system_state(SystemState.SHUTDOWN)
            logger.info("Scheduler stopping...")

        # Shutdown container runner
        self.container_runner.shutdown(wait=False)

        # Final cleanup
        self.memory_manager.cleanup()
        logger.info("Scheduler stopped")

    def _check_container_completion(self):
        """Check for containers that have naturally completed"""
        # Note: Container completion is actually handled by container_runner callbacks
        # when the subprocess naturally finishes. This method can be used for additional
        # monitoring or timeout scenarios if needed in the future.
        pass

    def _queue_container(self, container_id, memory_mb, cycle_position, reason):
        """Add container to ready queue with status (WAITING_SLOT or WAITING_MEMORY)"""
        cycle_number = (container_id - 1) // self.max_container_index + 1
        self.ready_queue.append({
            'id': container_id,
            'memory': memory_mb,
            'cycle_position': cycle_position,
            'cycle_number': cycle_number,
            'reason': reason,
            'queued_at': time.time()
        })
        self.csv_reporter.record_queue_event(container_id, memory_mb, reason, cycle_position)
        logger.info(f"Container {container_id} QUEUED ({reason}): "
                    f"type {cycle_position+1}/{self.max_container_index}, "
                    f"needs {memory_mb:.0f} MB, "
                    f"free={self.memory_manager.get_available_memory_mb():.0f} MB")

    def _process_queue(self):
        """Check if any queued containers can now launch (called after every completion)"""
        while self.ready_queue:
            next_c = self.ready_queue[0]
            can_slot = self.state_tracker.can_launch_container()
            can_mem = next_c['memory'] <= self.memory_manager.get_available_memory_mb()

            if can_slot and can_mem:
                c = self.ready_queue.pop(0)
                wait_time = time.time() - c['queued_at']
                logger.info(f"Container {c['id']} LAUNCHED from queue "
                            f"(type {c['cycle_position']+1}, waited {wait_time:.1f}s)")
                self._launch_queued_container(c['id'], c['memory'], c['cycle_position'])
            else:
                break  # FIFO: if head can't launch, don't try others

    def _launch_queued_container(self, container_id, container_memory, cycle_position):
        """Launch a container that was in the ready queue"""
        memory_block_id = self.memory_manager.allocate(container_memory, container_id)
        if memory_block_id is None:
            logger.error(f"Failed to allocate memory for queued container {container_id}")
            return False

        # Register container
        actual_container_id = self.state_tracker.register_container(
            memory_mb=container_memory,
            memory_block_id=memory_block_id,
            duration_seconds=self.config.container_duration_seconds
        )

        # Register with CSV reporter
        self.csv_reporter.register_container(actual_container_id, container_memory, self.config.container_duration_seconds)

        # Create run config
        worker_path = self.config.worker_script
        if not os.path.isabs(worker_path):
            if os.path.exists(worker_path):
                worker_path = os.path.abspath(worker_path)
            else:
                worker_path = os.path.join("/app", worker_path)

        run_config = ContainerRunConfig(
            container_id=actual_container_id,
            memory_mb=container_memory,
            duration_seconds=self.config.container_duration_seconds,
            worker_path=worker_path
        )

        # Launch container
        self.container_runner.run_container(run_config)
        logger.info(f"Launched queued container {actual_container_id}: {container_memory}MB")
        return True

    def _try_launch_container(self):
        """Try to launch a container - process queue first, then create new if queue empty"""
        with self.lock:
            # Priority: process queued containers FIRST (FIFO fairness)
            if self.ready_queue:
                self._process_queue()
                return  # Processed queue (or head is still blocked)

            # Queue is empty — try to create a new container
            # Check if we can launch a new one
            if not self.state_tracker.can_launch_container():
                return False

            # Calculate memory for next container using DYNAMIC RESET
            # Cycles through ALL container types (max_container_index)
            # This includes C4 (2909 MB) which runs with fewer parallels
            cycle_position = (self.next_container_id - 1) % self.max_container_index
            container_memory = self.config.base_memory_mb * (self.config.memory_multiplier ** cycle_position)

            logger.debug(
                f"Container {self.next_container_id}: cycle_position={cycle_position}/{self.max_container_index}, "
                f"memory={container_memory}MB (dynamic reset)"
            )

            # Dual check: slot AND memory must both be available
            can_launch_slot = self.state_tracker.can_launch_container()
            available_memory = self.memory_manager.get_available_memory_mb()
            can_launch_memory = container_memory <= available_memory

            if not can_launch_slot:
                # All slots full — queue this container
                self._queue_container(self.next_container_id, container_memory, cycle_position, 'WAITING_SLOT')
                self.next_container_id += 1
                return False

            if not can_launch_memory:
                # Slot available but not enough memory — queue this container
                logger.info(f"Container {self.next_container_id}: QUEUED (WAITING_MEMORY) "
                           f"need {container_memory:.0f} MB, free={available_memory:.0f} MB")
                self._queue_container(self.next_container_id, container_memory, cycle_position, 'WAITING_MEMORY')
                self.next_container_id += 1
                return False

            # Both conditions pass — launch immediately
            memory_block_id = self.memory_manager.allocate(container_memory, self.next_container_id)
            if memory_block_id is None:
                logger.error(f"Failed to allocate memory for container {self.next_container_id}")
                return False

            # Register container
            container_id = self.state_tracker.register_container(
                memory_mb=container_memory,
                memory_block_id=memory_block_id,
                duration_seconds=self.config.container_duration_seconds
            )

            # Register with CSV reporter
            self.csv_reporter.register_container(container_id, container_memory, self.config.container_duration_seconds)

            # Create run config with absolute path for worker script
            worker_path = self.config.worker_script

            if not os.path.isabs(worker_path):
                if os.path.exists(worker_path):
                    worker_path = os.path.abspath(worker_path)
                else:
                    worker_path = os.path.join("/app", worker_path)

            run_config = ContainerRunConfig(
                container_id=container_id,
                memory_mb=container_memory,
                duration_seconds=self.config.container_duration_seconds,
                worker_path=worker_path
            )

            # Launch container
            self.container_runner.run_container(run_config)
            logger.info(f"Launched container {container_id}: type {cycle_position+1}/{self.max_container_index}, {container_memory}MB")
            self.next_container_id += 1
            return True

    def step(self):
        """Execute one scheduling step (call this from main loop)"""
        if not self.running:
            return

        current_time = time.time()
        elapsed_time = current_time - self.start_time

        # Check if simulation should end
        if elapsed_time > self.config.simulation_duration_hours * 3600:
            logger.info("Simulation duration reached")
            self.running = False
            return

        # Record memory snapshot
        running = self.state_tracker.get_running_containers()
        allocated = self.memory_manager.get_allocated_memory_mb()
        remaining = self.config.total_gpu_memory_mb - allocated
        self.csv_reporter.record_memory_snapshot(
            timestamp=current_time,
            active_containers=len(running),
            total_memory_mb=allocated,
            remaining_memory_mb=remaining
        )

        # Launch containers at regular intervals
        # Per assignment: Container N should launch every step_interval_seconds
        # Memory constraints and reset policy are checked in _try_launch_container()
        if current_time - self.last_launch_time >= self.config.step_interval_seconds:
            self._try_launch_container()
            self.last_launch_time = current_time

    def get_stats(self) -> dict:
        """Get current scheduler statistics"""
        return {
            "timestamp": datetime.now().isoformat(),
            "system_stats": self.state_tracker.get_system_stats(),
            "memory_stats": self.memory_manager.get_stats(),
            "running": self.running,
        }

    def save_report(self, filepath: str):
        """Save scheduler report"""
        report = {
            "config": {
                "total_gpu_memory_mb": self.config.total_gpu_memory_mb,
                "container_duration_seconds": self.config.container_duration_seconds,
                "step_interval_seconds": self.config.step_interval_seconds,
                "max_concurrent_containers": self.config.max_concurrent_containers,
                "memory_multiplier": self.config.memory_multiplier,
                "base_memory_mb": self.config.base_memory_mb,
                "simulation_duration_hours": self.config.simulation_duration_hours,
            },
            "final_stats": self.get_stats(),
            "event_log": self.state_tracker.get_event_log(),
        }

        try:
            with open(filepath, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save report to {filepath}: {e}", exc_info=True)

    def generate_reports(self):
        """Generate CSV reports"""
        try:
            reports = self.csv_reporter.generate_all_reports()
            logger.info(f"CSV reports generated in: {reports['report_directory']}")
            for report_name, filepath in reports['files'].items():
                logger.info(f"  - {report_name}: {filepath}")
            return reports
        except Exception as e:
            logger.error(f"Error generating CSV reports: {e}")
            return None


def main():
    """Main function - loads config from config.ini and runs scheduler"""
    # Initialize logging to file and console
    logger_local, log_path = setup_logging()
    logger_local.info("Scheduler logging initialized")

    try:
        # Load configuration from config.ini
        config_loader = ConfigLoader()

        # Get scheduler config from config.ini
        cfg = config_loader.get_scheduler_config()

        # Log configuration summary
        log_config_summary(logger_local, cfg)

        # Create SchedulerConfig dataclass from config file
        config = SchedulerConfig(
            total_gpu_memory_mb=cfg.total_gpu_memory_mb,
            container_duration_seconds=cfg.container_duration_seconds,
            step_interval_seconds=cfg.step_interval_seconds,
            max_concurrent_containers=cfg.max_concurrent_containers,
            memory_multiplier=cfg.memory_multiplier,
            base_memory_mb=cfg.base_memory_mb,
            simulation_duration_hours=cfg.simulation_duration_hours,
            worker_script=cfg.worker_script,
        )

        logger.info("Starting scheduler with config from config.ini...")
        logger.info(f"Log file: {log_path}")

        scheduler = Scheduler(config)
        scheduler.start()

        try:
            # Main loop - non-blocking
            while scheduler.running:
                scheduler.step()
                time.sleep(0.01)  # Small sleep to prevent CPU spinning
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            logger.info("Shutting down scheduler...")
            scheduler.stop()  # Now includes report generation and save_report() inside
            logger.info("Scheduler completed successfully!")
            logger.info(f"Full logs available at: {log_path}")

    except FileNotFoundError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
