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
    # Import config loader
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config_loader import ConfigLoader
else:
    from .memory_manager import MemoryManager
    from .state_tracker import StateTracker, ContainerState, SystemState
    from .container_runner import ContainerRunner, ContainerRunConfig
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
    container_duration_seconds: int = 10       # How long each container runs (REDUCED from 600)
    step_interval_seconds: int = 5             # How often scheduler checks (REDUCED from 30)
    max_concurrent_containers: int = 2         # Max containers running at once (REDUCED from 3)
    memory_multiplier: float = 1.5             # Memory growth factor (Container N = Container N-1 * 1.5)
    base_memory_mb: float = 128                # First container memory (REDUCED from 256)
    simulation_duration_hours: float = 0.1     # Total run time (REDUCED from 24 hours)
    memory_multiplier_reset_interval: int = 3  # Reset memory multiplier every N containers (Part 2.3)
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
        """Called when container starts"""
        # Follow the state machine: CREATED → STARTING → ALLOCATING_MEMORY → RUNNING
        self.state_tracker.update_container_state(container_id, ContainerState.STARTING)
        self.state_tracker.update_container_state(container_id, ContainerState.ALLOCATING_MEMORY)
        self.state_tracker.update_container_state(container_id, ContainerState.RUNNING)

        # Get container info for memory reporting
        container = self.state_tracker.get_container_info(container_id)
        if container:
            self.csv_reporter.record_state_transition(container_id, 'RUNNING', time.time())

        logger.info(f"Container {container_id} started")

    def _on_container_complete(self, container_id: int, success: bool):
        """Called when container completes (either successfully or with error)"""
        self.state_tracker.mark_container_completed(container_id, success)
        self.csv_reporter.record_container_completion(container_id, success)

        # Reset consecutive OOM counter on successful completion (Requirement 5.2)
        if success:
            self.state_tracker.reset_consecutive_oom_failures()

        # Release memory
        container = self.state_tracker.get_container_info(container_id)
        if container and container.memory_block_id:
            self.memory_manager.release(container.memory_block_id)
            logger.info(f"Released memory for container {container_id}")

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

    def _try_launch_container(self):
        """Try to launch a container"""
        with self.lock:
            # Check if we can launch
            if not self.state_tracker.can_launch_container():
                return False

            # Calculate memory for next container with periodic reset to prevent starvation
            # Part 2.3 Implementation: Memory cap policy using reset interval
            next_container_id = len(self.state_tracker.containers) + 1
            reset_interval = getattr(self.config, 'memory_multiplier_reset_interval', 3)

            # Periodic reset: every N containers, restart from base memory
            # This prevents unbounded growth from exceeding the 4096MB limit
            cycle_position = (next_container_id - 1) % reset_interval
            container_memory = self.config.base_memory_mb * (self.config.memory_multiplier ** cycle_position)

            logger.debug(
                f"Container {next_container_id}: cycle_position={cycle_position}, "
                f"reset_interval={reset_interval}, memory={container_memory}MB (Part 2.3)"
            )

            # Check memory availability
            if container_memory > self.memory_manager.get_available_memory_mb():
                available_memory = self.memory_manager.get_available_memory_mb()
                logger.error(f"OOM FAILURE - Container memory allocation rejected:")
                logger.error(f"Requested Memory: {container_memory}MB")
                logger.error(f"Available GPU Memory: {available_memory:.2f}MB")
                logger.error(f"Deficit: {container_memory - available_memory:.2f}MB")
                logger.info(f"Retry: Waiting for running containers to complete and free memory")

                # Track consecutive OOM failures (Requirement 5.2)
                self.state_tracker.increment_consecutive_oom_failures()

                # Check if should trigger scheduler reset
                if self.state_tracker.should_trigger_scheduler_reset():
                    logger.critical(f"⚠️ SCHEDULER RESET TRIGGERED: 3 consecutive OOM failures detected")
                    self.state_tracker.record_oom_event()
                    return False

                self.state_tracker.record_oom_event()
                return False

            # Allocate memory
            memory_block_id = self.memory_manager.allocate(container_memory, len(self.state_tracker.containers) + 1)
            if memory_block_id is None:
                logger.error("Failed to allocate memory")
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

            # ALWAYS convert to absolute path
            # This works for both Docker and local execution
            if not os.path.isabs(worker_path):
                # First, check if file exists relative to current directory
                if os.path.exists(worker_path):
                    # Local execution - convert relative to absolute
                    worker_path = os.path.abspath(worker_path)
                else:
                    # Docker execution - file doesn't exist relative to cwd
                    # So construct absolute Docker path
                    worker_path = os.path.join("/app", worker_path)

            logger.info(f"[PATH] Resolved worker path: {worker_path}")

            run_config = ContainerRunConfig(
                container_id=container_id,
                memory_mb=container_memory,
                duration_seconds=self.config.container_duration_seconds,
                worker_path=worker_path
            )

            # Launch container
            self.container_runner.run_container(run_config)
            logger.info(f"Launched container {container_id}: {container_memory}MB")
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
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # Load configuration from config.ini
        config_loader = ConfigLoader()
        config_loader.print_config()

        # Get scheduler config from config.ini
        cfg = config_loader.get_scheduler_config()

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

    except FileNotFoundError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
