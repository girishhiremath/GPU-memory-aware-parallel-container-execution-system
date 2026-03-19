"""
Container Runner - Manages subprocess/thread pool for running containers
"""
import logging
import subprocess
import threading
import time
import os
from typing import Optional, Dict, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ContainerRunConfig:
    """Configuration for running a container"""
    container_id: int
    memory_mb: float
    duration_seconds: int
    worker_path: str = "worker/worker.py"
    python_path: str = "python"


class ContainerRunner:
    """Manages launching and monitoring containers"""

    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running_processes: Dict[int, subprocess.Popen] = {}
        self.process_lock = threading.Lock()
        self.callbacks = {
            "on_start": None,
            "on_complete": None,
            "on_error": None,
        }

    def set_callbacks(
        self,
        on_start: Optional[Callable] = None,
        on_complete: Optional[Callable] = None,
        on_error: Optional[Callable] = None
    ):
        """Set callback functions"""
        if on_start:
            self.callbacks["on_start"] = on_start
        if on_complete:
            self.callbacks["on_complete"] = on_complete
        if on_error:
            self.callbacks["on_error"] = on_error

    def run_container(self, config: ContainerRunConfig) -> int:
        """
        Run a container in a thread
        Returns container_id
        """
        future = self.executor.submit(self._run_container_process, config)
        return config.container_id

    def _run_container_process(self, config: ContainerRunConfig):
        """Actually run the container process (runs in thread)"""
        container_id = config.container_id

        try:
            # Prepare environment variables
            env = os.environ.copy()
            env["CONTAINER_ID"] = str(config.container_id)
            env["DURATION_SEC"] = str(config.duration_seconds)
            env["MEMORY_MB"] = str(config.memory_mb)
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONPATH"] = "/app"  # For module imports in Docker

            # Build command - worker_path should already be absolute from scheduler
            cmd = [config.python_path, config.worker_path]

            logger.info(f"Starting container {container_id}: {' '.join(cmd)}")

            # Call on_start callback
            if self.callbacks["on_start"]:
                self.callbacks["on_start"](container_id)

            # Determine working directory - use /app if it exists (Docker), otherwise None (local)
            cwd = None
            if os.path.exists("/app"):
                cwd = "/app"

            # Launch process
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd
            )

            # Register process
            with self.process_lock:
                self.running_processes[container_id] = process

            # Wait for completion
            stdout, stderr = process.communicate()

            # Log output - show stderr if process failed
            if stdout:
                logger.debug(f"Container {container_id} stdout:\n{stdout}")
            if stderr:
                if process.returncode != 0:
                    logger.error(f"Container {container_id} stderr:\n{stderr}")
                else:
                    logger.debug(f"Container {container_id} stderr:\n{stderr}")

            # Check return code
            if process.returncode == 0:
                logger.info(f"Container {container_id} completed successfully")
                if self.callbacks["on_complete"]:
                    self.callbacks["on_complete"](container_id, True)
            else:
                logger.error(f"Container {container_id} failed with code {process.returncode}")
                if self.callbacks["on_complete"]:
                    self.callbacks["on_complete"](container_id, False)

        except Exception as e:
            logger.error(f"Error running container {container_id}: {e}", exc_info=True)
            if self.callbacks["on_error"]:
                self.callbacks["on_error"](container_id, str(e))

        finally:
            # Clean up process reference
            with self.process_lock:
                if container_id in self.running_processes:
                    del self.running_processes[container_id]

    def stop_container(self, container_id: int, timeout: int = 5) -> bool:
        """
        Stop a running container
        Returns True if stopped, False otherwise
        """
        with self.process_lock:
            process = self.running_processes.get(container_id)
            if not process:
                logger.warning(f"Container {container_id} not found")
                return False

            try:
                process.terminate()
                try:
                    process.wait(timeout=timeout)
                    logger.info(f"Container {container_id} terminated")
                    return True
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    logger.warning(f"Container {container_id} killed forcefully")
                    return True
            except Exception as e:
                logger.error(f"Error stopping container {container_id}: {e}")
                return False

    def get_running_containers(self) -> Dict[int, subprocess.Popen]:
        """Get all running containers"""
        with self.process_lock:
            return dict(self.running_processes)

    def is_running(self, container_id: int) -> bool:
        """Check if container is running"""
        with self.process_lock:
            process = self.running_processes.get(container_id)
            return process is not None and process.poll() is None

    def shutdown(self, wait: bool = True, timeout: int = 30):
        """Shutdown the runner and stop all containers"""
        logger.info("Shutting down container runner")

        # Stop all running containers
        with self.process_lock:
            for container_id, process in list(self.running_processes.items()):
                try:
                    process.terminate()
                except Exception as e:
                    logger.warning(f"Error terminating container {container_id}: {e}")

        # Shutdown executor
        self.executor.shutdown(wait=wait)
        logger.info("Container runner shutdown complete")
