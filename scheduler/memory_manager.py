"""
GPU Memory Manager - Abstraction for memory block allocation and release
"""
import logging
from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum
import threading

logger = logging.getLogger(__name__)

# Check for PyTorch and GPU availability
try:
    import torch
    TORCH_AVAILABLE = True
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
except ImportError:
    TORCH_AVAILABLE = False
    GPU_AVAILABLE = False
    logger.info("PyTorch not available, will use CPU memory")


class MemoryBlockStrategy(Enum):
    FIXED_POOL = "fixed_pool"
    DYNAMIC_ALLOCATION = "dynamic_allocation"
    MEMORY_MAPPED = "memory_mapped"
    NUMPY_ARRAY = "numpy_array"


@dataclass
class MemoryBlock:
    """Represents a single memory block allocation"""
    block_id: int
    size_bytes: int
    allocated: bool
    container_id: Optional[int] = None
    allocation_time: Optional[float] = None
    release_time: Optional[float] = None


class MemoryManager:
    """Manages GPU/CPU memory allocation and release"""

    def __init__(self, total_memory_mb: float):
        self.total_memory_mb = total_memory_mb
        self.total_memory_bytes = int(total_memory_mb * 1024 * 1024)
        self.allocated_memory = 0
        self.peak_memory = 0
        self.blocks: Dict[int, MemoryBlock] = {}
        self.block_counter = 0
        self.allocation_history = []
        self.lock = threading.Lock()
        self.use_gpu = GPU_AVAILABLE and TORCH_AVAILABLE

        if self.use_gpu:
            logger.info("PyTorch GPU available - worker processes will handle GPU allocation")

    def allocate(self, size_mb: float, container_id: int, retry_count: int = 0) -> Optional[int]:
        """
        Allocate memory block
        Requirement 5.2: OOM Handling - tracks retry count
        Returns block_id on success, None on failure

        CRITICAL: Scheduler only TRACKS memory, doesn't allocate GPU
        Worker process does the actual GPU allocation
        """
        with self.lock:
            size_bytes = int(size_mb * 1024 * 1024)

            if self.allocated_memory + size_bytes > self.total_memory_bytes:
                available_mb = (self.total_memory_bytes - self.allocated_memory) / (1024 * 1024)
                logger.warning(f"Insufficient memory: requested {size_mb}MB, available {available_mb:.2f}MB (retry={retry_count})")
                return None

            # NOTE: Do NOT allocate GPU memory here!
            # The worker process will do the actual GPU allocation
            # This manager just tracks LOGICAL allocations for scheduling decisions

            self.block_counter += 1
            block_id = self.block_counter

            block = MemoryBlock(
                block_id=block_id,
                size_bytes=size_bytes,
                allocated=True,
                container_id=container_id
            )

            self.blocks[block_id] = block
            self.allocated_memory += size_bytes
            self.peak_memory = max(self.peak_memory, self.allocated_memory)

            logger.debug(f"Tracked allocation {size_mb}MB for container {container_id} (Block {block_id}) - GPU allocation done by worker")
            return block_id

    def release(self, block_id: int) -> bool:
        """
        Release memory block
        Returns True on success, False on failure

        CRITICAL: Only updates tracking, worker frees actual GPU memory
        """
        with self.lock:
            if block_id not in self.blocks:
                logger.error(f"Block {block_id} not found")
                return False

            block = self.blocks[block_id]
            if not block.allocated:
                logger.warning(f"Block {block_id} already released")
                return False

            self.allocated_memory -= block.size_bytes

            # NOTE: Do NOT try to release GPU memory here
            # Worker process already released it in its cleanup
            # This manager just updates LOGICAL tracking

            block.allocated = False
            logger.debug(f"Released tracking for {block.size_bytes / (1024*1024):.2f}MB for container {block.container_id}")
            return True

    def _allocate_gpu(self, size_mb: float) -> bool:
        """
        DEPRECATED: Scheduler no longer allocates GPU memory
        Worker process handles actual GPU allocation
        """
        logger.debug("_allocate_gpu() called but scheduler doesn't allocate GPU - worker does")
        return True

    def _release_gpu(self, size_bytes: int):
        """
        DEPRECATED: Scheduler no longer releases GPU memory
        Worker process handles GPU cleanup
        """
        logger.debug("_release_gpu() called but scheduler doesn't manage GPU - worker does")

    def get_allocated_memory_mb(self) -> float:
        """Get currently allocated memory in MB"""
        with self.lock:
            return self.allocated_memory / (1024 * 1024)

    def get_available_memory_mb(self) -> float:
        """Get available memory in MB"""
        with self.lock:
            return (self.total_memory_bytes - self.allocated_memory) / (1024 * 1024)

    def get_utilization_percent(self) -> float:
        """Get memory utilization percentage"""
        with self.lock:
            return (self.allocated_memory / self.total_memory_bytes) * 100

    def get_peak_memory_mb(self) -> float:
        """Get peak memory usage in MB"""
        with self.lock:
            return self.peak_memory / (1024 * 1024)

    def get_stats(self) -> dict:
        """Get comprehensive memory statistics"""
        with self.lock:
            stats = {
                "total_memory_mb": self.total_memory_bytes / (1024 * 1024),
                "allocated_memory_mb": self.get_allocated_memory_mb(),
                "available_memory_mb": self.get_available_memory_mb(),
                "utilization_percent": self.get_utilization_percent(),
                "peak_memory_mb": self.get_peak_memory_mb(),
                "total_blocks": len(self.blocks),
                "allocated_blocks": sum(1 for b in self.blocks.values() if b.allocated),
                "device": "GPU (PyTorch)" if self.use_gpu else "CPU",
            }

            if self.use_gpu:
                try:
                    gpu_memory = torch.cuda.memory_stats()
                    stats["gpu_allocated_mb"] = gpu_memory.get("allocated_bytes.all.current", 0) / (1024 * 1024)
                    stats["gpu_reserved_mb"] = gpu_memory.get("reserved_bytes.all.current", 0) / (1024 * 1024)
                except Exception as e:
                    logger.warning(f"Failed to get GPU stats: {e}")

            return stats

    def cleanup(self):
        """Clean up resources (now just placeholder)"""
        logger.info("Scheduler memory manager cleanup - GPU cleanup handled by worker processes")

    def verify_container_released(self, container_id: int) -> bool:
        """Verify all memory blocks for a container are released"""
        with self.lock:
            for block in self.blocks.values():
                if block.container_id == container_id and block.allocated:
                    return False
            return True
