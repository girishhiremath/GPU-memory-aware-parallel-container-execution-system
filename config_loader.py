#!/usr/bin/env python3
"""
Configuration loader for GPU Scheduler
Reads config.ini and provides configuration objects
"""
import configparser
import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SchedulerConfig:
    """Scheduler configuration from config.ini"""
    total_gpu_memory_mb: float
    container_duration_seconds: int
    step_interval_seconds: int
    max_concurrent_containers: int
    memory_multiplier: float
    base_memory_mb: float
    simulation_duration_hours: float
    worker_script: str
    memory_multiplier_reset_interval: int = 3  # Starvation prevention


@dataclass
class WorkerConfig:
    """Worker configuration from config.ini"""
    python_path: str
    default_timeout_seconds: int
    memory_allocation_mode: str


@dataclass
class ReportsConfig:
    """Reports configuration from config.ini"""
    reports_directory: str
    generate_on_shutdown: bool
    report_formats: str
    include_memory_timeline: bool
    include_state_transitions: bool
    num_containers_to_analyze: int


@dataclass
class MemoryManagerConfig:
    """Memory manager configuration from config.ini"""
    enable_pytorch_gpu: bool
    enable_cpu_fallback: bool
    memory_check_interval_seconds: int
    memory_threshold_percent: float


@dataclass
class LoggingConfig:
    """Logging configuration from config.ini"""
    log_level: str
    log_format: str
    log_to_file: bool
    log_file: str


@dataclass
class AdvancedConfig:
    """Advanced configuration from config.ini"""
    watchdog_poll_interval_seconds: int
    zombie_grace_period_seconds: int
    oom_retry_count: int
    oom_retry_reset_after_successes: int
    max_queue_size: int


class ConfigLoader:
    """Load and parse configuration from config.ini"""

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize config loader

        Args:
            config_file: Path to config.ini (default: config.ini in project root)
        """
        if config_file is None:
            # Try to find config.ini in project root
            project_root = Path(__file__).parent.parent
            config_file = project_root / "config.ini"
            if not config_file.exists():
                config_file = Path("config.ini")

        self.config_file = Path(config_file)
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")

        self.parser = configparser.ConfigParser()
        self.parser.read(self.config_file)
        logger.info(f"Loaded configuration from: {self.config_file}")

    def get_scheduler_config(self) -> SchedulerConfig:
        """Get scheduler configuration"""
        section = "SCHEDULER"
        return SchedulerConfig(
            total_gpu_memory_mb=self.parser.getfloat(section, "total_gpu_memory_mb"),
            container_duration_seconds=self.parser.getint(section, "container_duration_seconds"),
            step_interval_seconds=self.parser.getint(section, "step_interval_seconds"),
            max_concurrent_containers=self.parser.getint(section, "max_concurrent_containers"),
            memory_multiplier=self.parser.getfloat(section, "memory_multiplier"),
            base_memory_mb=self.parser.getfloat(section, "base_memory_mb"),
            simulation_duration_hours=self.parser.getfloat(section, "simulation_duration_hours"),
            worker_script=self.parser.get(section, "worker_script"),
            memory_multiplier_reset_interval=self.parser.getint(section, "memory_multiplier_reset_interval") if self.parser.has_option(section, "memory_multiplier_reset_interval") else 3,
        )

    def get_worker_config(self) -> WorkerConfig:
        """Get worker configuration"""
        section = "WORKER"
        return WorkerConfig(
            python_path=self.parser.get(section, "python_path"),
            default_timeout_seconds=self.parser.getint(section, "default_timeout_seconds"),
            memory_allocation_mode=self.parser.get(section, "memory_allocation_mode"),
        )

    def get_reports_config(self) -> ReportsConfig:
        """Get reports configuration"""
        section = "REPORTS"
        return ReportsConfig(
            reports_directory=self.parser.get(section, "reports_directory"),
            generate_on_shutdown=self.parser.getboolean(section, "generate_on_shutdown"),
            report_formats=self.parser.get(section, "report_formats"),
            include_memory_timeline=self.parser.getboolean(section, "include_memory_timeline"),
            include_state_transitions=self.parser.getboolean(section, "include_state_transitions"),
            num_containers_to_analyze=self.parser.getint(section, "num_containers_to_analyze"),
        )

    def get_memory_manager_config(self) -> MemoryManagerConfig:
        """Get memory manager configuration"""
        section = "MEMORY_MANAGER"
        return MemoryManagerConfig(
            enable_pytorch_gpu=self.parser.getboolean(section, "enable_pytorch_gpu"),
            enable_cpu_fallback=self.parser.getboolean(section, "enable_cpu_fallback"),
            memory_check_interval_seconds=self.parser.getint(section, "memory_check_interval_seconds"),
            memory_threshold_percent=self.parser.getfloat(section, "memory_threshold_percent"),
        )

    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration"""
        section = "LOGGING"
        return LoggingConfig(
            log_level=self.parser.get(section, "log_level"),
            log_format=self.parser.get(section, "log_format"),
            log_to_file=self.parser.getboolean(section, "log_to_file"),
            log_file=self.parser.get(section, "log_file"),
        )

    def get_advanced_config(self) -> AdvancedConfig:
        """Get advanced configuration"""
        section = "ADVANCED"
        return AdvancedConfig(
            watchdog_poll_interval_seconds=self.parser.getint(section, "watchdog_poll_interval_seconds"),
            zombie_grace_period_seconds=self.parser.getint(section, "zombie_grace_period_seconds"),
            oom_retry_count=self.parser.getint(section, "oom_retry_count"),
            oom_retry_reset_after_successes=self.parser.getint(section, "oom_retry_reset_after_successes"),
            max_queue_size=self.parser.getint(section, "max_queue_size"),
        )

    def get_all_configs(self):
        """Get all configurations as a dictionary"""
        return {
            "scheduler": self.get_scheduler_config(),
            "worker": self.get_worker_config(),
            "reports": self.get_reports_config(),
            "memory_manager": self.get_memory_manager_config(),
            "logging": self.get_logging_config(),
            "advanced": self.get_advanced_config(),
        }

    def print_config(self):
        """Print current configuration"""
        print("\n" + "="*80)
        print("CONFIGURATION SUMMARY")
        print("="*80)

        scheduler = self.get_scheduler_config()
        print(f"\n[SCHEDULER]")
        print(f"  GPU Memory: {scheduler.total_gpu_memory_mb}MB")
        print(f"  Container Duration: {scheduler.container_duration_seconds}s")
        print(f"  Step Interval: {scheduler.step_interval_seconds}s")
        print(f"  Max Concurrent: {scheduler.max_concurrent_containers}")
        print(f"  Memory Multiplier: {scheduler.memory_multiplier}x")
        print(f"  Base Memory: {scheduler.base_memory_mb}MB")
        print(f"  Simulation Duration: {scheduler.simulation_duration_hours}h")

        reports = self.get_reports_config()
        print(f"\n[REPORTS]")
        print(f"  Directory: {reports.reports_directory}")
        print(f"  Containers to Analyze: {reports.num_containers_to_analyze}")

        advanced = self.get_advanced_config()
        print(f"\n[ADVANCED]")
        print(f"  OOM Retry Count: {advanced.oom_retry_count}")
        print(f"  Watchdog Poll Interval: {advanced.watchdog_poll_interval_seconds}s")

        print("\n" + "="*80)


if __name__ == "__main__":
    # Test config loading
    loader = ConfigLoader()
    loader.print_config()
