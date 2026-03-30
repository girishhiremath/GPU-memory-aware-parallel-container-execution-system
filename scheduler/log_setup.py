"""
Logging configuration for GPU scheduler
Creates dated log files in logs/ directory
"""
import logging
import os
from datetime import datetime


def setup_logging():
    """
    Configure logging to:
    1. Write to console (INFO level)
    2. Write to dated file in logs/ directory (DEBUG level for detailed tracking)
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Generate log filename with date and time
    now = datetime.now()
    log_filename = f"scheduler_{now.strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_dir, log_filename)

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove any existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler (INFO level) - for terminal output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # File handler (DEBUG level) - for detailed file logging
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s,%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)

    # Create logger for main module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_path}")

    return logger, log_path


def log_config_summary(logger, config):
    """Log configuration summary at startup"""
    logger.info("=" * 80)
    logger.info("CONFIGURATION SUMMARY")
    logger.info("=" * 80)
    logger.info("")
    logger.info("[SCHEDULER]")
    logger.info(f"  GPU Memory: {config.total_gpu_memory_mb}MB")
    logger.info(f"  Container Duration: {config.container_duration_seconds}s")
    logger.info(f"  Step Interval: {config.step_interval_seconds}s")
    logger.info(f"  Max Concurrent: {config.max_concurrent_containers}")
    logger.info(f"  Memory Multiplier: {config.memory_multiplier}x")
    logger.info(f"  Base Memory: {config.base_memory_mb}MB")
    logger.info(f"  Simulation Duration: {config.simulation_duration_hours}h")
    logger.info("")
    logger.info("[REPORTS]")
    logger.info(f"  Directory: reports")
    if hasattr(config, 'num_containers_to_analyze'):
        logger.info(f"  Containers to Analyze: {config.num_containers_to_analyze}")
    logger.info("")
    logger.info("[ADVANCED]")
    logger.info(f"  OOM Retry Count: 3")
    logger.info(f"  Watchdog Poll Interval: 30s")
    logger.info("")
    logger.info("=" * 80)


def log_dynamic_reset_config(logger, base_memory, multiplier, gpu_memory,
                            max_container_index, max_simultaneous, cycle_memory, gpu_util):
    """Log dynamic reset configuration"""
    logger.info("=" * 80)
    logger.info("DYNAMIC RESET CONFIGURATION (computed at startup)")
    logger.info("=" * 80)
    logger.info(f"  Base Memory: {base_memory} MB")
    logger.info(f"  Memory Multiplier: {multiplier}x")
    logger.info(f"  Total GPU Memory: {gpu_memory} MB")
    logger.info(f"  Dynamic Reset Point: Container {max_container_index}")

    # Calculate what memory would be needed at reset point
    reset_memory = base_memory * (multiplier ** (max_container_index - 1))
    logger.info(f"    → Would require {reset_memory:.1f} MB")
    logger.info(f"    → Exceeds {gpu_memory} MB GPU")

    logger.info(f"  Max Simultaneous Containers: {max_simultaneous}")
    logger.info(f"  Cycle Memory Usage: {cycle_memory:.1f} MB ({gpu_util:.2f}% of GPU)")
    logger.info(f"  Reset Formula: container_index % {max_simultaneous} + 1")
    logger.info("=" * 80)


def log_container_launch(logger, container_id, container_type, memory_mb, cycle_pos=None):
    """Log container launch"""
    logger.info(f"Launched container {container_id}: type {container_type}, {memory_mb}MB")
    if cycle_pos is not None:
        logger.debug(f"  Container {container_id} cycle position: {cycle_pos}")


def log_container_queued(logger, container_id, container_type, memory_mb, reason, free_memory=None):
    """Log when container is queued due to constraints"""
    if free_memory:
        logger.info(f"Container {container_id}: QUEUED ({reason}) need {memory_mb:.0f} MB, free={free_memory:.0f} MB")
    else:
        logger.info(f"Container {container_id}: QUEUED ({reason}) type {container_type}, needs {memory_mb:.0f} MB")


def log_container_from_queue(logger, container_id, container_type, wait_time):
    """Log when container is launched from queue"""
    logger.info(f"Container {container_id} LAUNCHED from queue (type {container_type}, waited {wait_time:.1f}s)")


def log_container_completed(logger, container_id):
    """Log container completion"""
    logger.info(f"Released memory for container {container_id}")


def log_system_event(logger, message):
    """Log general system event"""
    logger.info(message)
