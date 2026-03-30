#!/usr/bin/env python
"""
Worker - GPU Memory Management Worker
Implements Container Lifecycle Contract (4.3):
CREATED → STARTING → ALLOCATING_MEMORY → RUNNING → RELEASING_MEMORY → COMPLETED
                                              ↘ FAILED (OOM or timeout)

Requirements:
- Each state transition logged with timestamp
- Reported to scheduler via shared state file
- Recoverable (on ALLOCATING_MEMORY failure, scheduler reclaims memory slot)
"""
import os
import sys
import time
import logging
import signal
import subprocess
import json
from pathlib import Path
from enum import Enum

# Try to import PyTorch for GPU operations
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class LifecycleState(Enum):
    """Container lifecycle states as per requirement 4.3"""
    CREATED = "created"
    STARTING = "starting"
    ALLOCATING_MEMORY = "allocating_memory"
    RUNNING = "running"
    RELEASING_MEMORY = "releasing_memory"
    COMPLETED = "completed"
    FAILED = "failed"


class GPUMemoryWorker:
    """
    GPU Memory Worker that allocates and holds GPU memory
    Implements lifecycle contract from 4.3
    """

    def __init__(self):
        # Read environment variables as per requirement 4.2
        self.container_id = os.environ.get("CONTAINER_ID", "")
        self.memory_mb = float(os.environ.get("MEMORY_MB", "0"))
        self.duration_sec = int(os.environ.get("DURATION_SEC", "600"))

        if not self.container_id or self.memory_mb <= 0:
            logger.error("CONTAINER_ID and MEMORY_MB are required")
            self._report_lifecycle_event(LifecycleState.FAILED, "Invalid environment variables")
            sys.exit(1)

        self.start_time = time.time()
        self.running = True
        self.gpu_memory_tensors = []
        self.memory_allocated_mb = 0
        self.shared_state_file = f"/tmp/container_{self.container_id}_state.json"

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - WORKER - %(levelname)s - %(message)s'
        )

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        logger.info(f"Worker initialized for container {self.container_id}")
        logger.info(f"Requirements: Memory={self.memory_mb}MB, Duration={self.duration_sec}s")

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Worker {self.container_id} received shutdown signal")
        self.running = False

    def _report_lifecycle_event(
        self,
        state: LifecycleState,
        message: str = "",
        success: bool = True
    ):
        """
        Requirement 4.3: Report lifecycle state with timestamp
        Scheduler reads this file to track container state
        """
        from datetime import datetime

        state_data = {
            "container_id": self.container_id,
            "lifecycle_state": state.value,
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "success": success,
            "memory_requested_mb": self.memory_mb,
            "memory_allocated_mb": self.memory_allocated_mb,
            "gpu_memory_used_mb": self._get_gpu_memory_used(),
            "elapsed_seconds": int(time.time() - self.start_time),
        }

        try:
            with open(self.shared_state_file, 'w') as f:
                json.dump(state_data, f)
            logger.debug(f"Reported lifecycle: {state.value}")
        except Exception as e:
            logger.warning(f"Could not write state file: {e}")

    def _get_gpu_memory_used(self) -> float:
        """Get current GPU memory usage in MB using nvidia-smi"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception as e:
            logger.debug(f"Could not get GPU memory: {e}")
        return 0.0

    def allocate_gpu_memory(self) -> bool:
        """
        Requirement 4.3: ALLOCATING_MEMORY state
        Exact memory allocation, logged with timestamp
        Recoverable: on failure, scheduler reclaims memory slot
        """
        try:
            logger.info(f"ALLOCATING_MEMORY: {self.memory_mb}MB")
            self._report_lifecycle_event(LifecycleState.ALLOCATING_MEMORY, "Allocating memory")

            # Calculate exact float32 elements needed
            elements_needed = int((self.memory_mb * 1024 * 1024) / 4)

            # Allocate tensor on GPU if available, otherwise CPU
            if TORCH_AVAILABLE:
                device = 'cuda'
                tensor = torch.zeros(elements_needed, dtype=torch.float32, device=device)
                logger.info(f"Allocated {self.memory_mb}MB on GPU")
            else:
                device = 'cpu'
                tensor = torch.zeros(elements_needed, dtype=torch.float32, device=device)
                logger.info(f"Allocated {self.memory_mb}MB on CPU (GPU not available)")
            self.gpu_memory_tensors.append(tensor)
            self.memory_allocated_mb += self.memory_mb

            logger.info(f"Allocated {self.memory_mb}MB on GPU")
            self._report_lifecycle_event(
                LifecycleState.ALLOCATING_MEMORY,
                f"Successfully allocated {self.memory_mb}MB",
                True
            )
            return True

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"OOM: Could not allocate {self.memory_mb}MB")
                self._report_lifecycle_event(
                    LifecycleState.FAILED,
                    "Out of GPU memory (OOM)",
                    False
                )
                return False
            else:
                logger.error(f"GPU allocation failed: {e}")
                self._report_lifecycle_event(
                    LifecycleState.FAILED,
                    f"GPU allocation error: {str(e)}",
                    False
                )
                return False

    def run_workload(self) -> bool:
        """
        Requirement 4.3: RUNNING state
        Run for exactly DURATION_SEC, logged with timestamps

        CRITICAL FIX: Don't create new tensors during workload!
        This causes memory fragmentation and excessive allocation
        """
        logger.info(f"RUNNING: Starting {self.duration_sec}s workload")
        self._report_lifecycle_event(LifecycleState.RUNNING, "Starting workload")

        elapsed = 0
        iterations = 0

        while self.running and elapsed < self.duration_sec:
            try:
                # Perform GPU operations WITHOUT creating new tensors
                if TORCH_AVAILABLE and self.gpu_memory_tensors:
                    with torch.no_grad():
                        # CRITICAL: In-place operations only to avoid tensor creation
                        # Don't do: tensor = tensor * 0.9999 + 0.0001 (creates new tensor!)
                        # Do: tensor *= 0.9999; tensor += 0.0001 (in-place)

                        # Compute sum (GPU busy, doesn't allocate new tensor)
                        _ = torch.sum(self.gpu_memory_tensors[0])

                        # In-place operations on existing tensor
                        # This doesn't create a new tensor, just modifies the existing one
                        self.gpu_memory_tensors[0].mul_(0.9999)
                        self.gpu_memory_tensors[0].add_(0.0001)

                iterations += 1
                elapsed = int(time.time() - self.start_time)

                # Log progress and report every 10 seconds
                if elapsed % 10 == 0 and elapsed > 0:
                    logger.info(f"Working: {elapsed}s / {self.duration_sec}s")
                    self._report_lifecycle_event(
                        LifecycleState.RUNNING,
                        f"Running: {elapsed}s / {self.duration_sec}s"
                    )

                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error during workload: {e}")
                self._report_lifecycle_event(
                    LifecycleState.FAILED,
                    f"Workload error: {str(e)}",
                    False
                )
                return False

        logger.info(f"RELEASING_MEMORY: Completed {elapsed}s workload")
        return True

    def release_gpu_memory(self) -> bool:
        """
        Requirement 4.3: RELEASING_MEMORY state
        Release all GPU memory before exit, logged with timestamp
        CRITICAL: Ensure complete memory cleanup for SageMaker
        """
        logger.info("Releasing GPU memory (AGGRESSIVE cleanup)...")
        self._report_lifecycle_event(LifecycleState.RELEASING_MEMORY, "Releasing GPU memory")

        try:
            # Step 1: Explicitly delete all tensors
            logger.info(f"Step 1: Clearing {len(self.gpu_memory_tensors)} tensor(s)")
            for i, tensor in enumerate(self.gpu_memory_tensors):
                del tensor
                logger.debug(f"  Deleted tensor {i}")
            self.gpu_memory_tensors.clear()
            self.memory_allocated_mb = 0

            # Step 2: Force garbage collection
            import gc
            logger.info("Step 2: Running garbage collection")
            gc.collect()
            gc.collect()  # Run twice to ensure cleanup
            gc.collect()  # Run three times for safety

            # Step 3: CUDA cleanup (most critical for SageMaker)
            if TORCH_AVAILABLE:
                logger.info("Step 3: CUDA cleanup")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                logger.info("CUDA cache cleared, GPU synchronized, peak memory reset")

            # Step 4: Verify memory is released
            logger.info("Step 4: Verifying memory release")
            time.sleep(1)  # Give GPU time to actually release
            gpu_memory_after = self._get_gpu_memory_used()
            logger.info(f"GPU memory used after release: {gpu_memory_after}MB")

            if gpu_memory_after < 100:  # Less than 100MB residual
                logger.info("Memory successfully released")
            else:
                logger.warning(f"Residual GPU memory: {gpu_memory_after}MB (may be system overhead)")

            self._report_lifecycle_event(
                LifecycleState.RELEASING_MEMORY,
                f"Memory released, GPU using {gpu_memory_after}MB",
                True
            )
            return True

        except Exception as e:
            logger.error(f"Error releasing GPU memory: {e}", exc_info=True)
            self._report_lifecycle_event(
                LifecycleState.FAILED,
                f"Memory release error: {str(e)}",
                False
            )
            return False

    def run(self) -> int:
        """
        Main execution flow implementing lifecycle contract (4.3)
        CREATED → STARTING → ALLOCATING_MEMORY → RUNNING → RELEASING_MEMORY → COMPLETED
                                              ↘ FAILED
        """
        try:
            # STARTING state
            logger.info("STARTING: Initializing worker")
            self._report_lifecycle_event(LifecycleState.STARTING, "Worker starting")

            # ALLOCATING_MEMORY state (recoverable)
            if not self.allocate_gpu_memory():
                logger.error("Failed to allocate GPU memory")
                # Scheduler will see FAILED state and reclaim memory slot (requirement 4.3)
                return 1

            # RUNNING state
            if not self.run_workload():
                logger.error("Workload failed")
                self._report_lifecycle_event(
                    LifecycleState.FAILED,
                    "Workload execution failed",
                    False
                )
                return 2

            # RELEASING_MEMORY state
            if not self.release_gpu_memory():
                logger.error("Failed to release GPU memory")
                return 2

            # COMPLETED state
            logger.info("COMPLETED: Container finished successfully")
            self._report_lifecycle_event(
                LifecycleState.COMPLETED,
                "Container completed successfully",
                True
            )
            return 0

        except Exception as e:
            logger.error(f"Fatal error: {e}")
            self._report_lifecycle_event(
                LifecycleState.FAILED,
                f"Fatal error: {str(e)}",
                False
            )
            try:
                self.release_gpu_memory()
            except:
                pass
            return 2

        finally:
            # Clean up state file
            try:
                Path(self.shared_state_file).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Could not clean up state file: {e}")


def main():
    """Main entry point"""
    worker = GPUMemoryWorker()
    exit_code = worker.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()


