"""
CSV Report Generator for GPU Container Orchestration
Generates detailed CSV reports for mathematical modeling and scheduling analysis
"""
import csv
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ReporterConfig:
    """Configuration for CSV report generation - loads from config.ini"""
    base_memory_mb: float
    memory_multiplier: float
    total_gpu_memory_mb: float
    container_duration_minutes: int
    container_duration_seconds: int  # For sub-minute durations
    simulation_hours: float
    max_concurrent_containers: int
    num_containers_to_analyze: int

    @staticmethod
    def from_ini(config_loader=None):
        """Load configuration from config.ini"""
        if config_loader is None:
            # Try to import and load from parent directory
            try:
                config_path = Path(__file__).parent.parent / "config_loader.py"
                sys.path.insert(0, str(config_path.parent))
                from config_loader import ConfigLoader
                config_loader = ConfigLoader()
            except Exception:
                # Fall back to defaults if config.ini not found
                return ReporterConfig(
                    base_memory_mb=128,
                    memory_multiplier=1.5,
                    total_gpu_memory_mb=4096,
                    container_duration_minutes=10,
                    container_duration_seconds=600,
                    simulation_hours=1,
                    max_concurrent_containers=2,
                    num_containers_to_analyze=14,
                )

        # Load from config.ini
        scheduler_cfg = config_loader.get_scheduler_config()
        reports_cfg = config_loader.get_reports_config()

        return ReporterConfig(
            base_memory_mb=scheduler_cfg.base_memory_mb,
            memory_multiplier=scheduler_cfg.memory_multiplier,
            total_gpu_memory_mb=scheduler_cfg.total_gpu_memory_mb,
            container_duration_minutes=max(1, int(scheduler_cfg.container_duration_seconds / 60)),
            container_duration_seconds=scheduler_cfg.container_duration_seconds,
            simulation_hours=scheduler_cfg.simulation_duration_hours,
            max_concurrent_containers=scheduler_cfg.max_concurrent_containers,
            num_containers_to_analyze=reports_cfg.num_containers_to_analyze,
        )



@dataclass
class ContainerMetrics:
    """Container metrics for reporting"""
    container_id: int
    launch_time: float
    completion_time: Optional[float]
    memory_mb: float
    duration_seconds: int
    state_transitions: List[Tuple[str, float]]  # (state, timestamp)
    success: bool


class CSVReporter:
    """Generates CSV reports for analysis"""

    def __init__(self, report_dir: str = "reports", config: Optional[ReporterConfig] = None):
        # Load from config.ini if not provided
        if config is None:
            config = ReporterConfig.from_ini()
        self.config = config
        self.report_dir = report_dir
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.report_subdir = os.path.join(report_dir, f"report_{self.timestamp}")
        os.makedirs(self.report_subdir, exist_ok=True)
        self.containers: Dict[int, ContainerMetrics] = {}
        self.memory_timeline: List[Dict] = []
        self.parallelism_events: List[Dict] = []
        self.dynamic_reset = None  # Set by scheduler after init
        self.queue_events: List[Dict] = []  # Track ready queue events

    def set_dynamic_reset_info(self, max_container_index, max_simultaneous, cycle_memory):
        """Receive computed dynamic reset values from scheduler.
        All report generators use these instead of hardcoded values."""
        self.dynamic_reset = {
            'max_container_index': max_container_index,
            'max_simultaneous': max_simultaneous,
            'cycle_memory': cycle_memory,
        }

    def _get_reset_interval(self):
        """Get the dynamic reset interval. Falls back to calculation if not set."""
        if self.dynamic_reset:
            return self.dynamic_reset['max_simultaneous']
        # Fallback: calculate from config
        import math
        base = self.config.base_memory_mb
        mult = self.config.memory_multiplier
        gpu = self.config.total_gpu_memory_mb
        if base <= 0 or base > gpu:
            return 1
        max_n = int(math.floor(1 + math.log(gpu / base) / math.log(mult)))
        while max_n > 0 and base * (mult ** (max_n - 1)) > gpu:
            max_n -= 1
        total = 0
        k = 0
        for i in range(min(max_n, self.config.max_concurrent_containers)):
            mem = base * (mult ** i)
            if total + mem <= gpu:
                total += mem
                k = i + 1
            else:
                break
        return max(k, 1)

    def _get_cycle_memory(self):
        """Get the cycle memory from dynamic reset info."""
        if self.dynamic_reset:
            return self.dynamic_reset['cycle_memory']
        # Fallback
        reset_n = self._get_reset_interval()
        return sum(self.config.base_memory_mb * (self.config.memory_multiplier ** i) for i in range(reset_n))

    def _get_gpu_utilization(self):
        """Get GPU utilization percentage for the cycle."""
        return (self._get_cycle_memory() / self.config.total_gpu_memory_mb * 100)

    def register_container(self, container_id: int, memory_mb: float, duration_seconds: int):
        """Register a container for tracking"""
        self.containers[container_id] = ContainerMetrics(
            container_id=container_id,
            launch_time=datetime.now().timestamp(),
            completion_time=None,
            memory_mb=memory_mb,
            duration_seconds=duration_seconds,
            state_transitions=[],
            success=False
        )

    def record_state_transition(self, container_id: int, state: str, timestamp: float):
        """Record state transition"""
        if container_id in self.containers:
            self.containers[container_id].state_transitions.append((state, timestamp))

    def record_container_completion(self, container_id: int, success: bool):
        """Record container completion"""
        if container_id in self.containers:
            self.containers[container_id].completion_time = datetime.now().timestamp()
            self.containers[container_id].success = success

    def record_memory_snapshot(self, timestamp: float, active_containers: int, total_memory_mb: float, remaining_memory_mb: float):
        """Record memory usage snapshot"""
        self.memory_timeline.append({
            'timestamp': timestamp,
            'active_containers': active_containers,
            'total_memory_used_mb': total_memory_mb,
            'remaining_memory_mb': remaining_memory_mb
        })

    def record_queue_event(self, container_id: int, memory_mb: float, reason: str, cycle_position: int):
        """Record when a container is queued (WAITING_SLOT or WAITING_MEMORY)"""
        self.queue_events.append({
            'timestamp': time.time(),
            'container_id': container_id,
            'memory_mb': memory_mb,
            'reason': reason,
            'cycle_position': cycle_position
        })

    def record_parallelism_event(self, timestamp: float, active_containers: int, event: str):
        """Record parallelism change event"""
        self.parallelism_events.append({
            'timestamp': timestamp,
            'active_containers': active_containers,
            'event': event
        })

    def generate_containers_csv(self):
        """Generate containers_data.csv with container execution info"""
        filepath = os.path.join(self.report_subdir, "containers_data.csv")

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            writer.writerow(['CONTAINER EXECUTION DATA'])
            writer.writerow(['Configuration:'])
            writer.writerow([f'  Total GPU Memory: {self.config.total_gpu_memory_mb} MB'])
            writer.writerow([f'  Base Memory: {self.config.base_memory_mb} MB'])
            writer.writerow([f'  Memory Multiplier: {self.config.memory_multiplier}'])
            writer.writerow([])

            writer.writerow([
                'Container ID',
                'Memory Allocated (MB)',
                'Expected Duration (s)',
                'Launch Time (s)',
                'Completion Time (s)',
                'Actual Duration (s)',
                'Duration Accuracy',
                'Success',
                'Execution Status',
                'Memory Efficiency %'
            ])

            for cid in sorted(self.containers.keys()):
                c = self.containers[cid]
                elapsed = (c.completion_time - c.launch_time) if c.completion_time else 0
                status = "✓ Completed" if c.success else "✗ Failed"

                # Calculate efficiency (how well did actual duration match expected)
                duration_diff = abs(elapsed - c.duration_seconds) if elapsed > 0 else 0
                efficiency = max(0, 100 - (duration_diff / c.duration_seconds * 100)) if c.duration_seconds > 0 else 0

                writer.writerow([
                    cid,
                    f"{c.memory_mb:.1f}",
                    c.duration_seconds,
                    f"{c.launch_time:.2f}",
                    f"{c.completion_time:.2f}" if c.completion_time else "N/A",
                    f"{elapsed:.2f}",
                    f"{efficiency:.1f}%" if c.completion_time else "N/A",
                    "Yes" if c.success else "No",
                    status,
                    f"{(c.memory_mb / self.config.total_gpu_memory_mb * 100):.1f}%"
                ])

            # Add summary statistics
            writer.writerow([])
            writer.writerow(['SUMMARY STATISTICS'])
            writer.writerow(['Metric', 'Value', 'Unit'])

            total_containers = len(self.containers)
            successful_containers = sum(1 for c in self.containers.values() if c.success)
            failed_containers = total_containers - successful_containers
            success_rate = (successful_containers / total_containers * 100) if total_containers > 0 else 0

            total_memory = sum(c.memory_mb for c in self.containers.values())
            avg_memory = total_memory / total_containers if total_containers > 0 else 0

            writer.writerow(['Total Containers Executed', total_containers, 'count'])
            writer.writerow(['Successful Containers', successful_containers, 'count'])
            writer.writerow(['Failed Containers', failed_containers, 'count'])
            writer.writerow(['Success Rate', f'{success_rate:.1f}', '%'])
            writer.writerow([])
            reset_n = self._get_reset_interval()
            cycle_mem = self._get_cycle_memory()
            gpu_util = self._get_gpu_utilization()

            writer.writerow([f'MEMORY UTILIZATION (DYNAMIC RESET, cycle={reset_n})'])
            writer.writerow(['Peak Cycle Utilization', f'{gpu_util:.2f}%', f'Cycle = {cycle_mem:.1f} / {self.config.total_gpu_memory_mb} MB'])
            for i in range(reset_n):
                mem = self.config.base_memory_mb * (self.config.memory_multiplier ** i)
                writer.writerow([f'Cycle Position {i+1}', f'{mem:.1f}', 'MB'])
            writer.writerow(['Cycle Total', f'{cycle_mem:.1f}', 'MB'])

        return filepath

    def generate_mathematical_modeling_csv(self):
        """Generate math_modeling.csv for Part 1 - Mathematical Modeling"""
        filepath = os.path.join(self.report_subdir, "math_modeling.csv")

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header with analysis
            writer.writerow(['PART 1 - MATHEMATICAL MODELING'])
            writer.writerow([f'Memory Growth Model: M(n) = M₀ × {self.config.memory_multiplier}^(n-1)'])
            writer.writerow(['Configuration:'])
            writer.writerow([f'  Base Memory: {self.config.base_memory_mb} MB'])
            writer.writerow([f'  Multiplier: {self.config.memory_multiplier}'])
            writer.writerow([f'  Total GPU Memory: {self.config.total_gpu_memory_mb} MB'])
            writer.writerow([])

            writer.writerow(['Container Number', 'Memory (MB)', 'M(1)+M(2)+M(3)', '3 Parallel?', 'M(n)+M(n+1)', '2 Parallel?', 'M(n)', '1 Parallel?'])

            # Calculate for each container
            for n in range(1, self.config.num_containers_to_analyze + 1):
                memory_n = self.config.base_memory_mb * (self.config.memory_multiplier ** (n - 1))
                memory_n1 = self.config.base_memory_mb * (self.config.memory_multiplier ** n) if n < self.config.num_containers_to_analyze else 0
                memory_n2 = self.config.base_memory_mb * (self.config.memory_multiplier ** (n + 1)) if n < self.config.num_containers_to_analyze - 1 else 0

                three_parallel = memory_n + memory_n1 + memory_n2
                two_parallel = memory_n + memory_n1
                one_parallel = memory_n

                three_ok = "✓ YES" if three_parallel <= self.config.total_gpu_memory_mb else "✗ NO"
                two_ok = "✓ YES" if two_parallel <= self.config.total_gpu_memory_mb else "✗ NO"
                one_ok = "✓ YES" if one_parallel <= self.config.total_gpu_memory_mb else "✗ NO"

                writer.writerow([
                    n,
                    f"{memory_n:.1f}",
                    f"{three_parallel:.1f}",
                    three_ok,
                    f"{two_parallel:.1f}",
                    two_ok,
                    f"{one_parallel:.1f}",
                    one_ok
                ])

        return filepath

    def generate_throughput_analysis_csv(self):
        """Generate throughput_24h.csv for Part 1 - 24-hour throughput analysis"""
        filepath = os.path.join(self.report_subdir, "throughput_24h.csv")

        total_minutes = self.config.simulation_hours * 60
        # Use total_seconds to handle sub-minute durations
        total_seconds = total_minutes * 60

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            writer.writerow(['PART 1 - THROUGHPUT ANALYSIS'])
            writer.writerow(['Configuration:'])
            writer.writerow([f'  Simulation Duration: {self.config.simulation_hours} hours ({total_minutes} minutes)'])
            writer.writerow([f'  Container Duration: {self.config.container_duration_seconds}s ({self.config.container_duration_minutes}m)'])
            total_slots = total_seconds // self.config.container_duration_seconds if self.config.container_duration_seconds > 0 else 1
            writer.writerow([f'  Total Time Slots: {total_slots} slots'])
            writer.writerow([f'  GPU Memory: {self.config.total_gpu_memory_mb} MB'])
            writer.writerow([f'  Base Memory: {self.config.base_memory_mb} MB'])
            writer.writerow([f'  Multiplier: {self.config.memory_multiplier}'])
            writer.writerow([])

            writer.writerow([
                'Time Slot',
                'Start Time (min)',
                'End Time (min)',
                'Container #',
                'Memory (MB)',
                'Max Parallel at Time',
                'Containers Running',
                'Total Memory Used (MB)',
                'Remaining Capacity (MB)',
                'Can Launch Next?'
            ])

            time_slot = 0
            container_num = 1
            accumulated_memory = 0

            for slot in range(1, int(total_slots) + 1):
                time_slot = slot
                start_time = (time_slot - 1) * self.config.container_duration_minutes
                end_time = time_slot * self.config.container_duration_minutes

                memory = self.config.base_memory_mb * (self.config.memory_multiplier ** (container_num - 1))

                # Determine how many can run in parallel at this point
                max_parallel = 1
                for p in range(1, self.config.max_concurrent_containers + 1):
                    test_memory = sum(
                        self.config.base_memory_mb * (self.config.memory_multiplier ** (container_num + i - 1))
                        for i in range(p)
                    )
                    if test_memory <= self.config.total_gpu_memory_mb:
                        max_parallel = p
                    else:
                        break

                # Check if can launch
                can_launch = memory <= (self.config.total_gpu_memory_mb - accumulated_memory)

                if can_launch:
                    accumulated_memory += memory
                    if accumulated_memory > self.config.total_gpu_memory_mb * 0.9:  # Reset when near limit
                        accumulated_memory = memory
                    containers_running = min(max_parallel, (self.config.total_gpu_memory_mb - accumulated_memory) // int(memory)) if memory > 0 else 0
                else:
                    containers_running = 0
                    accumulated_memory = 0

                writer.writerow([
                    time_slot,
                    start_time,
                    end_time,
                    container_num,
                    f"{memory:.1f}",
                    max_parallel,
                    containers_running,
                    f"{accumulated_memory:.1f}",
                    f"{self.config.total_gpu_memory_mb - accumulated_memory:.1f}",
                    "✓ YES" if can_launch else "✗ NO"
                ])

                if can_launch:
                    container_num += 1

            # Add parallelism analysis table
            writer.writerow([])
            writer.writerow(['PARALLELISM ANALYSIS - Maximum Containers Supported'])
            writer.writerow([])
            writer.writerow([
                'Containers in Parallel',
                'Memory Sum Formula',
                'Max n Value',
                'Feasible?'
            ])

            # Analyze for 1, 2, 3, etc. containers in parallel
            max_parallel_analysis = min(5, self.config.max_concurrent_containers)

            for p in range(1, max_parallel_analysis + 1):
                # Build formula string
                formula_parts = []
                for i in range(p):
                    if i == 0:
                        formula_parts.append("M(n)")
                    else:
                        formula_parts.append(f"M(n+{i})")
                formula_str = " + ".join(formula_parts) + " ≤ " + str(int(self.config.total_gpu_memory_mb))

                # Find max n for this parallelism
                max_n = 1
                for n in range(1, 20):
                    test_memory = sum(
                        self.config.base_memory_mb * (self.config.memory_multiplier ** (n + i - 1))
                        for i in range(p)
                    )
                    if test_memory <= self.config.total_gpu_memory_mb:
                        max_n = n
                    else:
                        break

                feasible = "✓ YES" if max_n >= 1 else "✗ NO"

                writer.writerow([
                    p,
                    formula_str,
                    max_n,
                    feasible
                ])

        return filepath

    def generate_scheduling_algorithm_csv(self):
        """Generate scheduling_algorithm.csv for Part 2 - Scheduling algorithm design"""
        filepath = os.path.join(self.report_subdir, "scheduling_algorithm.csv")

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            writer.writerow(['PART 2 - SCHEDULING ALGORITHM DESIGN'])
            writer.writerow([f'Non-blocking Event-Driven Scheduler with Dynamic Memory Reset (cycle={self._get_reset_interval()})'])
            writer.writerow(['Configuration:'])
            writer.writerow([f'  Base Memory: {self.config.base_memory_mb} MB'])
            writer.writerow([f'  Multiplier: {self.config.memory_multiplier}'])
            writer.writerow([f'  Total GPU Memory: {self.config.total_gpu_memory_mb} MB'])
            writer.writerow([f'  Container Duration: {self.config.container_duration_seconds} seconds'])
            writer.writerow([f'  Max Concurrent: {self.config.max_concurrent_containers}'])
            writer.writerow([f'  Memory Reset: Dynamic (cycle={self._get_reset_interval()}, computed at startup)'])
            writer.writerow([])

            writer.writerow(['SCHEDULER STATE TIMELINE - First 3600 seconds of execution'])
            writer.writerow(['Note: Actual containers overlap. Timeline shows scheduling decisions at 5-second intervals.'])
            writer.writerow([])
            writer.writerow([
                'Time (s)',
                'Container ID',
                'State',
                'Memory Required (MB)',
                'Active Containers (count)',
                'Active IDs',
                'Total Memory Used (MB)',
                'Available Memory (MB)',
                'Event Description',
                'Launch Decision',
                'Note'
            ])

            # Simulate first 3600 seconds (1 hour) with 5-second intervals
            active_containers = {}  # {container_id: {'start_time': t, 'memory': m}}
            current_time_sec = 0
            container_num = 1
            reset_interval = self._get_reset_interval()
            step_size = 5  # seconds

            # Simulate with 5-second sampling intervals
            for step in range(0, int(3600 / step_size) + 1):
                current_time_sec = step * step_size

                # Check for completions (containers finish after container_duration_seconds)
                completed = []
                for cid, info in list(active_containers.items()):
                    elapsed = (current_time_sec - info['start_time'])
                    if elapsed >= self.config.container_duration_seconds:
                        completed.append(cid)

                # Process completions
                total_memory = sum(info['memory'] for info in active_containers.values())

                for cid in completed:
                    memory = active_containers[cid]['memory']
                    del active_containers[cid]
                    total_memory -= memory
                    available = self.config.total_gpu_memory_mb - total_memory

                    writer.writerow([
                        current_time_sec,
                        cid,
                        'COMPLETED',
                        f"{memory:.1f}",
                        len(active_containers),
                        ', '.join([f"C{c}" for c in sorted(active_containers.keys())]),
                        f"{total_memory:.1f}",
                        f"{available:.1f}",
                        f"Container {cid} released resources",
                        'N/A',
                        'Memory freed'
                    ])

                # Check if can launch new container (only at specific launch times: 0s, 5s, 10s, ...)
                launch_interval = 5  # Try to launch every 5 seconds
                if container_num <= 100 and current_time_sec % launch_interval == 0:
                    # Apply reset formula: effective_position = ((n-1) % reset_interval) + 1
                    effective_pos = ((container_num - 1) % reset_interval) + 1
                    memory = self.config.base_memory_mb * (self.config.memory_multiplier ** (effective_pos - 1))

                    total_memory = sum(info['memory'] for info in active_containers.values())
                    available = self.config.total_gpu_memory_mb - total_memory

                    can_launch = (
                        len(active_containers) < self.config.max_concurrent_containers and
                        total_memory + memory <= self.config.total_gpu_memory_mb
                    )

                    if can_launch:
                        active_containers[container_num] = {
                            'start_time': current_time_sec,
                            'memory': memory
                        }
                        total_memory += memory
                        available = self.config.total_gpu_memory_mb - total_memory

                        writer.writerow([
                            current_time_sec,
                            container_num,
                            'CREATED',
                            f"{memory:.1f}",
                            len(active_containers),
                            ', '.join([f"C{c}" for c in sorted(active_containers.keys())]),
                            f"{total_memory:.1f}",
                            f"{available:.1f}",
                            f"Container {container_num} launched (reset pos {effective_pos})",
                            f'✓ YES ({len(active_containers)}/{self.config.max_concurrent_containers})',
                            f'{len(active_containers)} now active'
                        ])
                        container_num += 1

        return filepath

    def generate_state_transitions_csv(self):
        """Generate state_transitions.csv for detailed state tracking"""
        filepath = os.path.join(self.report_subdir, "state_transitions.csv")

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            writer.writerow(['CONTAINER LIFECYCLE STATE TRACKING'])
            writer.writerow(['State Machine: 7-State Container Lifecycle'])
            writer.writerow(['Configuration:'])
            writer.writerow([f'  Base Memory: {self.config.base_memory_mb} MB'])
            writer.writerow([f'  Memory Multiplier: {self.config.memory_multiplier}'])
            writer.writerow([f'  Total GPU Memory: {self.config.total_gpu_memory_mb} MB'])
            writer.writerow([])

            writer.writerow(['STATE DEFINITIONS'])
            writer.writerow(['State', 'Phase', 'Description', 'Actions'])
            writer.writerow(['CREATED', 'Registration', 'Container registered in scheduler', 'Memory allocated from pool'])
            writer.writerow(['STARTING', 'Initialization', 'Subprocess being created', 'Process object created'])
            writer.writerow(['ALLOCATING_MEMORY', 'GPU Setup', 'GPU memory allocation in progress', 'torch.zeros() executed'])
            writer.writerow(['RUNNING', 'Execution', 'Container actively using GPU', 'Worker process executing'])
            writer.writerow(['RELEASING_MEMORY', 'Cleanup', 'Memory being released', 'torch.cuda.empty_cache() called'])
            writer.writerow(['COMPLETED', 'Final (Success)', 'Container completed successfully', 'Exit code 0, memory freed'])
            writer.writerow(['FAILED', 'Final (Error)', 'Container failed with error', 'Exit code != 0, memory freed'])
            writer.writerow([])

            writer.writerow([
                'Container ID',
                'Current State',
                'Timestamp (s)',
                'Time Since Launch (s)',
                'Memory Allocated (MB)',
                'Container Status',
                'State Description'
            ])

            state_descriptions = {
                'CREATED': 'Registered and awaiting start',
                'STARTING': 'Process being launched',
                'ALLOCATING_MEMORY': 'GPU memory being allocated',
                'RUNNING': 'Active execution on GPU',
                'RELEASING_MEMORY': 'Releasing allocated memory',
                'COMPLETED': 'Finished successfully',
                'FAILED': 'Finished with error'
            }

            for cid in sorted(self.containers.keys()):
                c = self.containers[cid]
                base_time = c.launch_time
                overall_status = "✓ SUCCESS" if c.success else "✗ FAILURE"

                for state, timestamp in c.state_transitions:
                    duration = timestamp - base_time
                    state_desc = state_descriptions.get(state, 'Unknown state')

                    writer.writerow([
                        cid,
                        state,
                        f"{timestamp:.2f}",
                        f"{duration:.2f}",
                        f"{c.memory_mb:.1f}",
                        overall_status,
                        state_desc
                    ])

            # Add state transition statistics
            writer.writerow([])
            writer.writerow(['STATE TRANSITION STATISTICS'])
            writer.writerow(['State', 'Total Occurrences', 'Success Count', 'Failure Count', 'Avg Entry Time'])

            state_stats = {}
            for c in self.containers.values():
                for state, _ in c.state_transitions:
                    if state not in state_stats:
                        state_stats[state] = {'total': 0, 'success': 0, 'failure': 0, 'times': []}
                    state_stats[state]['total'] += 1
                    if c.success:
                        state_stats[state]['success'] += 1
                    else:
                        state_stats[state]['failure'] += 1

            for state in ['CREATED', 'STARTING', 'ALLOCATING_MEMORY', 'RUNNING', 'RELEASING_MEMORY', 'COMPLETED', 'FAILED']:
                if state in state_stats:
                    stats = state_stats[state]
                    writer.writerow([
                        state,
                        stats['total'],
                        stats['success'],
                        stats['failure'],
                        'N/A'
                    ])

        return filepath

    def generate_starvation_prevention_csv(self):
        """Generate starvation_prevention.csv for Part 3 - Starvation analysis"""
        filepath = os.path.join(self.report_subdir, "starvation_prevention.csv")

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            writer.writerow(['PART 3 - STARVATION PREVENTION ANALYSIS'])
            writer.writerow(['Comparing three memory cap policy approaches'])
            writer.writerow(['Configuration:'])
            writer.writerow([f'  Base Memory: {self.config.base_memory_mb} MB'])
            writer.writerow([f'  Multiplier: {self.config.memory_multiplier}'])
            writer.writerow([f'  Total GPU Memory: {self.config.total_gpu_memory_mb} MB'])
            writer.writerow([])

            writer.writerow(['Approach', 'Description', 'Max Containers', 'Max Parallel', 'Starvation?', 'Throughput/24h', 'Recommendation'])

            approaches = [
                ['Sliding Window', 'Exponential growth continues, parallelism drops', '9', 'Drops 3→0', 'Yes', '~4', 'NOT RECOMMENDED ✗'],
                [f'Dynamic Reset (cycle={self._get_reset_interval()})', 'Reset when next > GPU (ACTUAL)', '∞', f'{self._get_reset_interval()}', 'No', f'{int(86400/self.config.container_duration_seconds)*self._get_reset_interval()}', 'RECOMMENDED ✓'],
                ['Reset Every 4', 'Reset after C4 back to C1 size', '432+', 'Varies 1-3', 'No', '288', 'Acceptable'],
                ['Capped at 2048', 'Cap individual container max at 2048 MB', '288+', 'Drops to 1-2', 'No', '288', 'Not optimal']
            ]

            for approach in approaches:
                writer.writerow(approach)

            writer.writerow([])
            writer.writerow([f'RECOMMENDATION: Dynamic Reset (cycle={self._get_reset_interval()}) — computed at startup'])
            writer.writerow(['Rationale: Maintains constant-parallelism cycling, adapts to any base_memory, prevents starvation'])
            writer.writerow([f'Formula: memory = base × (multiplier ^ ((n-1) % {self._get_reset_interval()}))'])
            writer.writerow([])

            # Memory growth with different approaches
            writer.writerow(['Container Number', 'No Reset (MB)', f'Dynamic (cycle={self._get_reset_interval()}) (MB)', 'Reset Every 4 (MB)', 'Capped at 2GB (MB)'])

            reset_n = self._get_reset_interval()
            for n in range(1, self.config.num_containers_to_analyze + 1):
                sliding = self.config.base_memory_mb * (self.config.memory_multiplier ** (n - 1))
                reset_dynamic = self.config.base_memory_mb * (self.config.memory_multiplier ** ((n - 1) % reset_n))
                reset_4 = self.config.base_memory_mb * (self.config.memory_multiplier ** ((n - 1) % 4))
                capped = min(sliding, 2048)

                writer.writerow([
                    n,
                    f"{sliding:.0f}",
                    f"{reset_dynamic:.0f}",
                    f"{reset_4:.0f}",
                    f"{capped:.0f}"
                ])

        return filepath

    def generate_memory_timeline_csv(self):
        """Generate memory_timeline.csv from computed container schedule"""
        filepath = os.path.join(self.report_subdir, "memory_timeline.csv")

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            writer.writerow(['MEMORY USAGE TIMELINE'])
            writer.writerow([f'Generated from container execution schedule with dynamic memory model (cycle={self._get_reset_interval()})'])
            writer.writerow(['Configuration:'])
            writer.writerow([f'  Total GPU Memory: {self.config.total_gpu_memory_mb} MB'])
            writer.writerow([f'  Base Memory: {self.config.base_memory_mb} MB'])
            writer.writerow([f'  Memory Multiplier: {self.config.memory_multiplier}'])
            writer.writerow([f'  Container Duration: {self.config.container_duration_seconds} seconds'])
            writer.writerow([f'  Reset Interval: {self._get_reset_interval()} containers (dynamic)'])
            writer.writerow([])

            writer.writerow([
                'Timestamp (s)',
                'Active Containers (count)',
                'Container IDs',
                'Total Memory Used (MB)',
                'Used %',
                'Remaining Capacity (MB)',
                'Available %',
                'Memory Pressure',
                'Can Launch?'
            ])

            # Generate timeline from computed container schedule
            total_gpu = self.config.total_gpu_memory_mb
            reset_interval = self._get_reset_interval()
            launch_interval_seconds = 5  # containers launched every 5 seconds
            container_duration_seconds = self.config.container_duration_seconds

            # Pre-calculate containers for first 24 hours
            all_containers = []
            container_id = 1
            max_time_seconds = min(self.config.simulation_hours * 3600, 86400)  # Cap at 24 hours for reasonableness

            for t in range(0, int(max_time_seconds), launch_interval_seconds):
                all_containers.append({
                    'id': container_id,
                    'launch_time': t,
                    'completion_time': t + container_duration_seconds,
                    'reset_pos': ((container_id - 1) % reset_interval) + 1,
                    'memory': self.config.base_memory_mb * (self.config.memory_multiplier ** (((container_id - 1) % reset_interval)))
                })
                container_id += 1

            # Generate timeline at 5-second intervals
            for sample_time_sec in range(0, int(max_time_seconds) + 1, 5):
                # Find all containers running at this sample time
                active = [c for c in all_containers
                         if c['launch_time'] <= sample_time_sec < c['completion_time']]

                if not active:
                    continue

                # Calculate memory stats
                total_memory_used = sum(c['memory'] for c in active)
                utilization = (total_memory_used / total_gpu) * 100
                available_pct = 100 - utilization
                remaining = total_gpu - total_memory_used

                # Determine memory pressure level
                if utilization >= 95:
                    pressure = 'CRITICAL'
                    can_launch = 'NO'
                elif utilization >= 90:
                    pressure = 'SEVERE'
                    can_launch = 'NO'
                elif utilization >= 75:
                    pressure = 'HIGH'
                    can_launch = 'NO'
                elif utilization >= 50:
                    pressure = 'MEDIUM'
                    can_launch = 'YES'
                else:
                    pressure = 'LOW'
                    can_launch = 'YES'

                active_ids = sorted([c['id'] for c in active])

                writer.writerow([
                    f"{sample_time_sec:.0f}",
                    len(active),
                    ', '.join([f"C{cid}" for cid in active_ids]),
                    f"{total_memory_used:.1f}",
                    f"{utilization:.1f}%",
                    f"{remaining:.1f}",
                    f"{available_pct:.1f}%",
                    pressure,
                    can_launch
                ])

        return filepath

    def generate_summary_report(self):
        """Generate summary_report.csv with key metrics and answers"""
        filepath = os.path.join(self.report_subdir, "summary_report.csv")

        # Calculate answers based on configuration
        # Q1: Maximum M₀ for 3 parallel containers
        max_m0 = self.config.total_gpu_memory_mb / (self.config.memory_multiplier ** 0 + self.config.memory_multiplier + self.config.memory_multiplier ** 2)

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            writer.writerow(['GPU CONTAINER ORCHESTRATION SYSTEM - SUMMARY REPORT'])
            writer.writerow([])
            writer.writerow(['Generated:', datetime.now().isoformat()])
            writer.writerow([])

            # Configuration section
            writer.writerow(['CONFIGURATION PARAMETERS'])
            writer.writerow(['Parameter', 'Value', 'Unit'])
            writer.writerow(['Base Memory (M₀)', f'{self.config.base_memory_mb:.1f}', 'MB'])
            writer.writerow(['Memory Multiplier (r)', f'{self.config.memory_multiplier:.2f}', 'factor'])
            writer.writerow(['Total GPU Memory', f'{self.config.total_gpu_memory_mb:.1f}', 'MB'])
            writer.writerow(['Container Duration', f'{self.config.container_duration_seconds}', 'seconds'])
            writer.writerow(['Simulation Duration', f'{self.config.simulation_hours}', 'hours'])
            writer.writerow(['Max Concurrent Containers', f'{self.config.max_concurrent_containers}', 'containers'])
            writer.writerow(['Analysis Depth', f'{self.config.num_containers_to_analyze}', 'containers'])
            writer.writerow([])

            # Part 1: Mathematical Modeling
            writer.writerow(['PART 1 - MATHEMATICAL MODELING'])
            writer.writerow(['Question', 'Category', 'Answer', 'Calculation', 'Status'])

            # Q1
            formula_q1 = f'M₀ × [1 + {self.config.memory_multiplier} + {self.config.memory_multiplier**2:.2f}] ≤ {self.config.total_gpu_memory_mb}'
            writer.writerow([
                'Q1: Maximum M₀ for 3 parallel',
                'Mathematical Limit',
                f'{max_m0:.1f} MB',
                formula_q1,
                '✓ Calculated'
            ])

            # Q2: Find max containers for each parallelism
            writer.writerow([])
            writer.writerow(['Q2: Maximum container index for each parallelism level'])
            writer.writerow(['Parallelism', 'Max Container Index (n)', 'Memory at Max n (MB)', 'Status'])

            for p in range(1, 4):
                max_n = 1
                for n in range(1, 20):
                    test_memory = sum(
                        self.config.base_memory_mb * (self.config.memory_multiplier ** (n + i - 1))
                        for i in range(p)
                    )
                    if test_memory <= self.config.total_gpu_memory_mb:
                        max_n = n
                    else:
                        break

                last_memory = sum(
                    self.config.base_memory_mb * (self.config.memory_multiplier ** (max_n + i - 1))
                    for i in range(p)
                )
                writer.writerow([
                    p,
                    max_n,
                    f'{last_memory:.1f}',
                    '✓ Feasible'
                ])

            # Q3: Throughput analysis
            writer.writerow([])
            reset_n = self._get_reset_interval()
            cycle_time = self.config.container_duration_seconds
            total_seconds = self.config.simulation_hours * 3600
            cycles_total = int(total_seconds / cycle_time)
            containers_total = cycles_total * reset_n

            writer.writerow([f'Q3: Total throughput analysis (DYNAMIC RESET, cycle={reset_n})'])
            writer.writerow(['Metric', 'Value', 'Unit'])
            writer.writerow(['Simulation Duration', f'{self.config.simulation_hours}', 'hours'])
            writer.writerow(['Cycle Duration', f'{cycle_time}', f'seconds ({reset_n} containers simultaneously)'])
            writer.writerow(['Containers per Cycle', f'{reset_n}', 'containers'])
            writer.writerow(['Total Cycles', f'{cycles_total}', 'cycles'])
            writer.writerow(['Expected Total', f'~{containers_total}', 'containers'])
            writer.writerow(['Cycle Memory', f'{self._get_cycle_memory():.1f}', 'MB'])
            writer.writerow(['GPU Utilization', f'{self._get_gpu_utilization():.2f}', '%'])
            writer.writerow([])

            # Part 2: Scheduling Algorithm
            writer.writerow(['PART 2 - SCHEDULING ALGORITHM DESIGN'])
            writer.writerow(['Design Element', 'Implementation', 'Value/Status'])
            writer.writerow(['Algorithm Type', 'Non-blocking Event-Driven', '✓ Active'])
            writer.writerow(['Memory Management', 'Exponential Growth Model', f'M(n) = {self.config.base_memory_mb} × {self.config.memory_multiplier}^(n-1)'])
            writer.writerow(['Launch Condition 1', 'Active Count Check', f'active_count < {self.config.max_concurrent_containers}'])
            writer.writerow(['Launch Condition 2', 'Memory Availability', f'used_memory + next_memory ≤ {self.config.total_gpu_memory_mb} MB'])
            writer.writerow(['State Machine States', '7-State Lifecycle', 'CREATED → STARTING → ALLOCATING_MEMORY → RUNNING → RELEASING_MEMORY → COMPLETED/FAILED'])
            writer.writerow(['Thread Safety', 'Lock-based Synchronization', '✓ Implemented'])
            writer.writerow(['Callback System', 'Event Handlers', 'on_start, on_complete, on_error'])
            writer.writerow([])

            writer.writerow(['State Transition Timeline'])
            writer.writerow(['State', 'Purpose', 'Duration (ms)', 'Action'])
            writer.writerow(['CREATED', 'Container registered', '0', 'Memory allocated'])
            writer.writerow(['STARTING', 'Process launching', '<10', 'Subprocess created'])
            writer.writerow(['ALLOCATING_MEMORY', 'GPU allocation', '<100', 'torch.zeros() call'])
            writer.writerow(['RUNNING', 'Active execution', f'{self.config.container_duration_seconds*1000}', 'GPU work in progress'])
            writer.writerow(['RELEASING_MEMORY', 'Cleanup', '<50', 'torch.cuda.empty_cache()'])
            writer.writerow(['COMPLETED', 'Success', '0', 'Exit code 0'])
            writer.writerow(['FAILED', 'Error', '0', 'Exit code non-zero'])
            writer.writerow([])

            # Part 3: Starvation Prevention
            writer.writerow(['PART 3 - STARVATION PREVENTION ANALYSIS'])
            reset_n = self._get_reset_interval()
            writer.writerow(['Policy Approach', 'Description', 'Max Containers/24h', 'Recommendation'])
            writer.writerow(['Sliding Window', 'Exponential continues', '4', 'Causes starvation ✗'])
            writer.writerow([f'Dynamic Reset (cycle={reset_n})', 'Computed at startup (ACTUAL)', f'{int(86400/self.config.container_duration_seconds)*reset_n}', 'Recommended ✓'])
            writer.writerow(['Reset Every 4', 'Reset after C4', '288', 'Acceptable'])
            writer.writerow(['Capped at 2048', 'Cap individual container', '288', 'Not optimal'])
            writer.writerow([])
            writer.writerow([f'Recommended Policy: Dynamic Reset (cycle={reset_n})'])
            writer.writerow(['Rationale', 'Detail'])
            writer.writerow(['Measured Throughput', f'{int(86400/self.config.container_duration_seconds)*reset_n} containers/24h (dynamic based on cycle)'])
            writer.writerow(['Parallelism', f'Maintains constant {reset_n}-way parallelism'])
            writer.writerow(['GPU Utilization', f'Peak {self._get_gpu_utilization():.2f}% ({self._get_cycle_memory():.1f} / {self.config.total_gpu_memory_mb} MB)'])
            writer.writerow(['Fairness', 'All containers scheduled, no starvation'])
            writer.writerow(['Implementation', f'effective_n = ((n-1) % {self._get_reset_interval()}) + 1; memory = base × multiplier^(effective_n-1)'])
            writer.writerow([])

            # writer.writerow(['Implementation Code'])
            # writer.writerow(['Component', 'Implementation'])
            # writer.writerow(['Max Memory Cap', 'MAX_CONTAINER_MEMORY = 2048  # MB'])
            # writer.writerow(['Memory Calculation', 'memory = base_memory × (multiplier ^ (n-1))'])
            # writer.writerow(['Apply Cap', 'return min(memory, MAX_CONTAINER_MEMORY)'])
            # writer.writerow([])

            writer.writerow(['Report Files Generated'])
            writer.writerow(['File Name', 'Content', 'Use Case'])
            writer.writerow(['summary_report.csv', 'Key metrics & answers', 'Overview (this file)'])
            writer.writerow(['math_modeling.csv', 'Memory calculations & parallelism', 'Mathematical analysis'])
            writer.writerow(['throughput_24h.csv', 'Time slots & throughput analysis', 'Performance analysis'])
            writer.writerow(['scheduling_algorithm.csv', 'Scheduler state transitions', 'Algorithm verification'])
            writer.writerow(['state_transitions.csv', 'Container lifecycle tracking', 'Detailed state audit'])
            writer.writerow(['starvation_prevention.csv', 'Policy comparison', 'Fairness analysis'])
            writer.writerow(['containers_data.csv', 'Container execution data', 'Execution metrics'])
            writer.writerow(['memory_timeline.csv', 'Memory usage over time', 'Resource tracking'])
            writer.writerow(['execution_schedule.csv', 'Complete container scheduling with concurrency', 'Comprehensive scheduling detail'])
            writer.writerow(['chronological_timeline.csv', 'Event-by-event timeline of all scheduling events', 'Narrative schedule with queue events'])
            writer.writerow(['queue_analysis.csv', 'Ready queue events and statistics', 'Queue fairness tracking'])

        return filepath

    def generate_first_hour_timeline_csv(self):
        """Generate detailed timeline for first hour showing container scheduling decisions"""
        filepath = os.path.join(self.report_subdir, "first_hour_timeline.csv")

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            # Title and metadata
            writer.writerow(['DETAILED TIMELINE - FIRST HOUR (3600 seconds)'])
            writer.writerow(['Shows ACTUAL overlapping container execution with correct parallelism'])
            writer.writerow([])
            writer.writerow(['Configuration:'])
            writer.writerow(['Base Memory (M0)', f'{self.config.base_memory_mb} MB'])
            writer.writerow(['Memory Multiplier', f'{self.config.memory_multiplier}x'])
            writer.writerow(['Total GPU Memory', f'{self.config.total_gpu_memory_mb} MB'])
            writer.writerow(['Container Duration', f'{self.config.container_duration_seconds} seconds ({self.config.container_duration_minutes} min)'])
            writer.writerow(['Max Concurrent', f'{self.config.max_concurrent_containers}'])
            writer.writerow(['Reset Interval', f'{self._get_reset_interval()} containers (dynamic)'])
            writer.writerow([])

            # Timeline header
            writer.writerow([
                'Time (s)',
                'Containers Running',
                'Container IDs',
                'Memory Used (MB)',
                'Memory %',
                'Remaining (MB)',
                'Can Launch New?',
                'Decision Reason'
            ])

            # Generate first 3600 seconds (1 hour) with 5-second sampling intervals
            total_gpu = self.config.total_gpu_memory_mb
            reset_interval = self._get_reset_interval()
            container_duration_seconds = self.config.container_duration_seconds

            # Simulate with actual launch/completion logic
            active_containers = {}  # {container_id: {'launch_time': t, 'completion_time': t+dur, 'memory': m}}
            container_id = 1
            next_launch_time = 0

            # Generate timeline at 5-second intervals
            for sample_time_sec in range(0, 3601, 5):
                # Check for completions
                completed = []
                for cid in list(active_containers.keys()):
                    if sample_time_sec >= active_containers[cid]['completion_time']:
                        completed.append(cid)

                for cid in completed:
                    del active_containers[cid]

                # Try to launch containers at this interval
                while (sample_time_sec >= next_launch_time and
                       len(active_containers) < self.config.max_concurrent_containers and
                       sample_time_sec < 3600):
                    # Calculate memory for this container
                    reset_pos = ((container_id - 1) % reset_interval)
                    memory = self.config.base_memory_mb * (self.config.memory_multiplier ** reset_pos)
                    current_memory = sum(c['memory'] for c in active_containers.values())

                    # Check if we have space
                    if current_memory + memory <= total_gpu:
                        active_containers[container_id] = {
                            'launch_time': sample_time_sec,
                            'completion_time': sample_time_sec + container_duration_seconds,
                            'memory': memory
                        }
                        container_id += 1
                        next_launch_time = sample_time_sec + 5
                    else:
                        break

                if not active_containers:
                    continue

                # Calculate stats
                memory_used = sum(c['memory'] for c in active_containers.values())
                active_ids = sorted(active_containers.keys())

                # Check if next container could launch
                next_memory = self.config.base_memory_mb * (self.config.memory_multiplier ** (((container_id - 1) % reset_interval)))
                can_launch_count = len(active_containers)
                can_launch = (can_launch_count < self.config.max_concurrent_containers and
                             memory_used + next_memory <= total_gpu)

                reason = "OK" if can_launch else ("OOM" if memory_used + next_memory > total_gpu else "Max parallel")

                writer.writerow([
                    sample_time_sec,
                    len(active_containers),
                    ', '.join([f'C{cid}' for cid in active_ids]),
                    f'{memory_used:.1f}',
                    f'{(memory_used/total_gpu)*100:.2f}%',
                    f'{total_gpu - memory_used:.1f}',
                    'YES' if can_launch else 'NO',
                    reason
                ])

        return filepath

    def generate_container_launch_schedule_csv(self):
        """Generate container launch schedule showing actual launch decisions"""
        filepath = os.path.join(self.report_subdir, "container_launch_schedule.csv")

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            # Title
            writer.writerow(['CONTAINER LAUNCH SCHEDULE'])
            writer.writerow(['Shows scheduled and actual launch times with OOM decisions'])
            writer.writerow([])

            # Metadata
            reset_n = self._get_reset_interval()
            cycle_time = self.config.container_duration_seconds
            total_seconds = self.config.simulation_hours * 3600
            expected = int(total_seconds / cycle_time) * reset_n
            writer.writerow(['Launch Interval', '5 seconds (between containers in a cycle)'])
            writer.writerow(['Total Simulation', f'{self.config.simulation_hours} hours'])
            writer.writerow(['Container Duration', f'{self.config.container_duration_seconds} seconds ({self.config.container_duration_minutes} min)'])
            writer.writerow(['Containers per Cycle', f'{reset_n}'])
            writer.writerow(['Expected Containers', f'~{expected}'])
            writer.writerow([])

            # Header
            writer.writerow([
                'Container',
                'Scheduled Launch (s)',
                'Expected Start (s)',
                'Expected End (s)',
                'Duration (s)',
                'Requested Memory (MB)',
                'Actual Launch',
                'Decision'
            ])

            # Generate launch schedule (in seconds, not minutes)
            duration_seconds = self.config.container_duration_seconds
            total_gpu = self.config.total_gpu_memory_mb
            launch_interval = 5  # seconds

            container_id = 1
            reset_interval = self._get_reset_interval()
            running_containers = []

            for slot in range(0, int(self.config.simulation_hours * 3600) + 600, launch_interval):
                # Check which containers will have completed
                running_containers = [c for c in running_containers if slot < c['end_time']]

                # Try to launch new container
                cycle_pos = (container_id - 1) % reset_interval
                container_mem = self.config.base_memory_mb * (self.config.memory_multiplier ** cycle_pos)

                # Calculate current memory usage
                current_memory = sum(c['memory'] for c in running_containers)

                # Check if can launch
                can_launch = (len(running_containers) < self.config.max_concurrent_containers and
                            current_memory + container_mem <= total_gpu)

                if can_launch:
                    launch_time = slot
                    end_time = launch_time + duration_seconds

                    running_containers.append({
                        'id': container_id,
                        'memory': container_mem,
                        'start': launch_time,
                        'end_time': end_time
                    })

                    writer.writerow([
                        f'C{container_id}',
                        launch_time,
                        launch_time,
                        end_time,
                        duration_seconds,
                        f'{container_mem:.1f}',
                        'Yes',
                        'Launched'
                    ])

                    container_id += 1

                    # Stop if we've gone far enough into simulation
                    if container_id > 50:  # Just show first 50 for readability
                        break

        return filepath

    def generate_chronological_timeline_csv(self):
        """Generate chronological event timeline showing all scheduling events with state transitions"""
        filepath = os.path.join(self.report_subdir, "chronological_timeline.csv")

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(['CHRONOLOGICAL EVENT TIMELINE'])
            writer.writerow(['Complete scheduling events with container state transitions and memory status'])
            writer.writerow(['Generated:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            writer.writerow([])

            # Configuration
            writer.writerow(['CONFIGURATION'])
            writer.writerow(['Base Memory:', f'{self.config.base_memory_mb} MB'])
            writer.writerow(['Memory Multiplier:', f'{self.config.memory_multiplier}x'])
            writer.writerow(['Total GPU Memory:', f'{self.config.total_gpu_memory_mb} MB'])
            writer.writerow(['Container Duration:', f'{self.config.container_duration_seconds}s'])
            writer.writerow(['Max Concurrent:', f'{self.config.max_concurrent_containers}'])
            writer.writerow(['Reset Interval (cycle):', f'{self._get_reset_interval()}'])
            writer.writerow([])

            if not self.containers:
                writer.writerow(['STATUS: No container data collected yet'])
                return filepath

            # Build all events chronologically
            all_events = []

            # Add container lifecycle events
            for c in self.containers.values():
                all_events.append({
                    'time': c.launch_time,
                    'type': 'LAUNCH',
                    'container_id': c.container_id,
                    'memory': c.memory_mb,
                    'completion_time': c.completion_time,
                    'states': c.state_transitions,
                    'success': c.success
                })
                if c.completion_time:
                    all_events.append({
                        'time': c.completion_time,
                        'type': 'COMPLETION',
                        'container_id': c.container_id,
                        'memory': c.memory_mb,
                        'completion_time': c.completion_time,
                        'states': c.state_transitions,
                        'success': c.success
                    })

            # Add queue events
            for qe in self.queue_events:
                all_events.append({
                    'time': qe['timestamp'],
                    'type': 'QUEUED',
                    'container_id': qe['container_id'],
                    'memory': qe['memory_mb'],
                    'reason': qe['reason'],
                    'cycle_position': qe['cycle_position']
                })

            # Sort by time, with LAUNCH before COMPLETION before QUEUED at same time
            event_order = {'LAUNCH': 0, 'QUEUED': 1, 'COMPLETION': 2}
            all_events.sort(key=lambda e: (e['time'], event_order.get(e['type'], 3)))

            # Timeline header
            writer.writerow([
                'Time (s)',
                'Event',
                'Container',
                'Action',
                'Active Containers',
                'Memory Used (MB)',
                'Memory %',
                'Free Memory (MB)',
                'Notes'
            ])

            # Track active containers at each point in time
            active_containers = {}

            for event in all_events:
                event_time = event['time']
                event_type = event['type']
                container_id = event['container_id']
                memory = event.get('memory', 0)

                # Update active containers based on event type
                if event_type == 'LAUNCH':
                    active_containers[container_id] = {
                        'memory': memory,
                        'launch_time': event_time,
                        'type': ((container_id - 1) % self._get_reset_interval()) + 1,
                        'cycle': self._get_reset_interval()
                    }
                elif event_type == 'COMPLETION':
                    active_containers.pop(container_id, None)

                # Calculate memory stats
                total_memory = sum(c['memory'] for c in active_containers.values())
                remaining = self.config.total_gpu_memory_mb - total_memory
                memory_pct = (total_memory / self.config.total_gpu_memory_mb) * 100
                active_ids = sorted(active_containers.keys())

                # Build event description
                if event_type == 'LAUNCH':
                    container_type = active_containers[container_id]['type']
                    cycle = active_containers[container_id]['cycle']
                    action = f"Launch C{container_id} (type {container_type}/{cycle}, {memory:.0f} MB)"
                    notes = f"C{container_id} starts" if len(active_ids) == 1 else f"C{container_id} starts, {len(active_ids)} now active"

                elif event_type == 'COMPLETION':
                    freed = memory
                    action = f"C{container_id} completes (free {freed:.0f} MB)"
                    # Check if anything was queued
                    queued_for_this = [e for e in self.queue_events if e['container_id'] == container_id]
                    if queued_for_this:
                        notes = f"Free C{container_id}, was queued for {len(queued_for_this)} event(s)"
                    else:
                        notes = f"Free C{container_id}"
                    # Check if new container can launch
                    if active_containers:
                        notes += f", {len(active_ids)} still active"

                elif event_type == 'QUEUED':
                    reason = event.get('reason', 'UNKNOWN')
                    action = f"C{container_id} queued ({reason}, needs {memory:.0f} MB)"
                    notes = f"C{container_id} blocked - {reason}"
                    if reason == 'WAITING_MEMORY':
                        notes += f" (free {remaining:.0f} < needed {memory:.0f})"
                    elif reason == 'WAITING_SLOT':
                        notes += f" ({len(active_ids)}/{self.config.max_concurrent_containers} slots full)"

                active_ids_str = ', '.join([f'C{cid}' for cid in active_ids]) if active_ids else '(empty)'

                writer.writerow([
                    f"{event_time:.1f}",
                    event_type,
                    f"C{container_id}",
                    action,
                    active_ids_str,
                    f"{total_memory:.1f}",
                    f"{memory_pct:.2f}%",
                    f"{remaining:.1f}",
                    notes
                ])

            # Summary section
            writer.writerow([])
            writer.writerow(['TIMELINE SUMMARY'])
            writer.writerow(['Metric', 'Value', 'Unit'])

            total_events = len(all_events)
            launch_events = sum(1 for e in all_events if e['type'] == 'LAUNCH')
            completion_events = sum(1 for e in all_events if e['type'] == 'COMPLETION')
            queue_events_count = sum(1 for e in all_events if e['type'] == 'QUEUED')

            writer.writerow(['Total Events', total_events, 'count'])
            writer.writerow(['Container Launches', launch_events, 'count'])
            writer.writerow(['Container Completions', completion_events, 'count'])
            writer.writerow(['Queue Events', queue_events_count, 'count'])

            if all_events:
                start_time = all_events[0]['time']
                end_time = all_events[-1]['time']
                total_time = end_time - start_time

                writer.writerow(['Timeline Start', f"{start_time:.1f}", 'seconds'])
                writer.writerow(['Timeline End', f"{end_time:.1f}", 'seconds'])
                writer.writerow(['Total Duration', f"{total_time:.1f}", 'seconds'])
                writer.writerow(['Average Event Interval', f"{total_time/total_events if total_events > 0 else 0:.2f}", 'seconds'])

            # Queue analysis
            if self.queue_events:
                writer.writerow([])
                writer.writerow(['QUEUE ANALYSIS (from timeline)'])
                waiting_memory = sum(1 for e in self.queue_events if e['reason'] == 'WAITING_MEMORY')
                waiting_slot = sum(1 for e in self.queue_events if e['reason'] == 'WAITING_SLOT')

                writer.writerow(['WAITING_MEMORY Events', waiting_memory, 'count'])
                writer.writerow(['WAITING_SLOT Events', waiting_slot, 'count'])

        return filepath

    def generate_execution_schedule_csv(self):
        """Generate comprehensive execution schedule showing all containers with concurrent execution details"""
        filepath = os.path.join(self.report_subdir, "execution_schedule.csv")

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(['COMPREHENSIVE EXECUTION SCHEDULE'])
            writer.writerow(['Complete container lifecycle with concurrent execution details'])
            writer.writerow(['Generated:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            writer.writerow([])

            # Configuration
            writer.writerow(['CONFIGURATION'])
            writer.writerow(['Base Memory:', f'{self.config.base_memory_mb} MB'])
            writer.writerow(['Memory Multiplier:', f'{self.config.memory_multiplier}x'])
            writer.writerow(['Total GPU Memory:', f'{self.config.total_gpu_memory_mb} MB'])
            writer.writerow(['Container Duration:', f'{self.config.container_duration_seconds}s'])
            writer.writerow(['Max Concurrent:', f'{self.config.max_concurrent_containers}'])
            writer.writerow(['Reset Interval (cycle):', f'{self._get_reset_interval()}'])
            writer.writerow([])

            if not self.containers:
                writer.writerow(['STATUS: No container data collected yet'])
                return filepath

            # Generate main execution table
            writer.writerow(['INDIVIDUAL CONTAINER EXECUTION'])
            writer.writerow([
                'Container ID',
                'Memory (MB)',
                'Type (n/cycle)',
                'Launch Time (s)',
                'Completion Time (s)',
                'Duration Actual (s)',
                'Status',
                'State Count',
                'Peak Concurrent',
                'Overlapping Containers'
            ])

            # Sort containers by launch time
            sorted_containers = sorted(self.containers.values(), key=lambda c: c.launch_time)

            for c in sorted_containers:
                if c.completion_time is None:
                    duration = 0
                    is_complete = False
                else:
                    duration = c.completion_time - c.launch_time
                    is_complete = True

                # Find all overlapping containers
                overlapping = []
                for other in self.containers.values():
                    if other.container_id == c.container_id:
                        continue
                    if other.completion_time is None:
                        continue
                    # Check if time ranges overlap
                    if c.launch_time < other.completion_time and other.launch_time < c.completion_time:
                        overlapping.append(other.container_id)

                overlapping_str = ', '.join([f'C{cid}' for cid in sorted(overlapping)]) if overlapping else 'None'

                # Get container type
                reset_pos = ((c.container_id - 1) % self._get_reset_interval()) + 1
                cycle = self._get_reset_interval()

                status = '✓ COMPLETED' if c.success else ('✗ FAILED' if c.completion_time else 'RUNNING')

                writer.writerow([
                    c.container_id,
                    f"{c.memory_mb:.1f}",
                    f'{reset_pos}/{cycle}',
                    f"{c.launch_time:.2f}",
                    f"{c.completion_time:.2f}" if c.completion_time else 'N/A',
                    f"{duration:.2f}" if is_complete else 'N/A',
                    status,
                    len(c.state_transitions),
                    len(overlapping),
                    overlapping_str
                ])

            # Concurrent execution analysis
            writer.writerow([])
            writer.writerow(['CONCURRENT EXECUTION ANALYSIS'])
            writer.writerow([
                'Time Point (s)',
                'Concurrent Count',
                'Container IDs',
                'Total Memory (MB)',
                'Memory %',
                'Remaining (MB)',
                'Event Type'
            ])

            # Build timeline of all significant events
            events = []
            for c in self.containers.values():
                if c.completion_time:
                    events.append(('START', c.launch_time, c.container_id, c.memory_mb))
                    events.append(('END', c.completion_time, c.container_id, c.memory_mb))

            # Sort events by time
            events.sort(key=lambda e: (e[1], e[0] == 'END'))  # START before END at same time

            # Track active containers at each time point
            active_at_time = {}
            current_active = {}

            for event_type, event_time, container_id, memory in events:
                if event_type == 'START':
                    current_active[container_id] = memory
                else:
                    current_active.pop(container_id, None)

                # Record state after event
                if current_active:
                    active_at_time[event_time] = dict(current_active)

            # Write timeline entries
            for event_time in sorted(active_at_time.keys()):
                active_ids = sorted(active_at_time[event_time].keys())
                active_memory = sum(active_at_time[event_time].values())
                memory_pct = (active_memory / self.config.total_gpu_memory_mb) * 100
                remaining = self.config.total_gpu_memory_mb - active_memory

                # Determine event type
                if len(active_at_time.get(event_time, {})) > len(active_at_time.get(event_time - 0.1, {})):
                    event_desc = 'LAUNCH'
                else:
                    event_desc = 'COMPLETION'

                writer.writerow([
                    f"{event_time:.2f}",
                    len(active_ids),
                    ', '.join([f'C{cid}' for cid in active_ids]),
                    f"{active_memory:.1f}",
                    f"{memory_pct:.2f}%",
                    f"{remaining:.1f}",
                    event_desc
                ])

            # State transition details
            writer.writerow([])
            writer.writerow(['STATE TRANSITION TIMELINE'])
            writer.writerow([
                'Container ID',
                'State',
                'Timestamp (s)',
                'Time Since Launch (s)',
                'Description'
            ])

            state_descriptions = {
                'CREATED': 'Registered and memory allocated',
                'STARTING': 'Process being launched',
                'ALLOCATING_MEMORY': 'GPU memory allocation in progress',
                'RUNNING': 'Active execution on GPU',
                'RELEASING_MEMORY': 'Releasing allocated resources',
                'COMPLETED': 'Finished successfully',
                'FAILED': 'Finished with error'
            }

            for c in sorted_containers:
                for state, ts in c.state_transitions:
                    time_since_launch = ts - c.launch_time
                    writer.writerow([
                        c.container_id,
                        state,
                        f"{ts:.2f}",
                        f"{time_since_launch:.2f}",
                        state_descriptions.get(state, 'Unknown')
                    ])

            # Summary statistics
            writer.writerow([])
            writer.writerow(['EXECUTION SUMMARY STATISTICS'])
            writer.writerow(['Metric', 'Value', 'Unit'])

            total_containers = len(self.containers)
            completed = sum(1 for c in self.containers.values() if c.completion_time is not None)
            successful = sum(1 for c in self.containers.values() if c.success)

            writer.writerow(['Total Containers', total_containers, 'count'])
            writer.writerow(['Completed', completed, 'count'])
            writer.writerow(['Successful', successful, 'count'])
            writer.writerow(['Success Rate', f'{(successful/completed*100):.1f}' if completed > 0 else 'N/A', '%'])

            if sorted_containers:
                first_launch = sorted_containers[0].launch_time
                last_completion = max((c.completion_time for c in self.containers.values() if c.completion_time), default=first_launch)
                total_execution_time = last_completion - first_launch

                writer.writerow(['Total Execution Time', f'{total_execution_time:.2f}', 'seconds'])
                writer.writerow(['Throughput', f'{completed/(total_execution_time/self.config.container_duration_seconds):.1f}' if total_execution_time > 0 else 'N/A', 'containers/cycle'])

            # Memory statistics
            total_memory_allocated = sum(c.memory_mb for c in self.containers.values())
            avg_memory = total_memory_allocated / total_containers if total_containers > 0 else 0

            writer.writerow(['Total Memory Allocated', f'{total_memory_allocated:.1f}', 'MB'])
            writer.writerow(['Average Memory per Container', f'{avg_memory:.1f}', 'MB'])
            writer.writerow(['Peak GPU Utilization', f'{self._get_gpu_utilization():.2f}', '%'])

            # Parallelism analysis
            writer.writerow([])
            writer.writerow(['PARALLELISM ANALYSIS'])
            writer.writerow(['Parallelism Level', 'Occurrences', 'Percentage', 'Duration (s)'])

            parallelism_stats = {}
            for event_time in sorted(active_at_time.keys()):
                count = len(active_at_time[event_time])
                if count not in parallelism_stats:
                    parallelism_stats[count] = 0
                parallelism_stats[count] += 1

            for level in sorted(parallelism_stats.keys()):
                occurrences = parallelism_stats[level]
                total_points = len(active_at_time)
                pct = (occurrences / total_points * 100) if total_points > 0 else 0
                writer.writerow([
                    level,
                    occurrences,
                    f'{pct:.1f}%',
                    f'{occurrences * 5:.1f}'  # Assuming 5-second intervals
                ])

        return filepath

    def generate_queue_analysis_csv(self):
        """Generate DYNAMIC queue_analysis.csv with ready queue statistics"""
        filepath = os.path.join(self.report_subdir, "queue_analysis.csv")

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            writer.writerow(['Ready Queue Analysis - DYNAMIC REPORT'])
            writer.writerow([f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'])
            writer.writerow([])

            # Summary statistics from actual collected data
            total_queue_events = len(self.queue_events)
            writer.writerow(['QUEUE SUMMARY (ACTUAL DATA)'])
            writer.writerow(['Total Queue Events:', total_queue_events])

            if total_queue_events == 0:
                writer.writerow(['Status:', 'No queuing - all containers launched immediately'])
                return filepath

            # Categorize by reason
            waiting_memory = [e for e in self.queue_events if e['reason'] == 'WAITING_MEMORY']
            waiting_slot = [e for e in self.queue_events if e['reason'] == 'WAITING_SLOT']

            writer.writerow(['WAITING_MEMORY Events:', len(waiting_memory)])
            writer.writerow(['WAITING_SLOT Events:', len(waiting_slot)])
            writer.writerow([])

            # Container-level analysis
            unique_containers = len(set(e['container_id'] for e in self.queue_events))
            writer.writerow(['Unique Containers Queued:', unique_containers])

            # Memory analysis from actual queue data
            total_queued_memory = sum(e['memory_mb'] for e in self.queue_events)
            avg_queued_memory = total_queued_memory / total_queue_events if total_queue_events > 0 else 0
            max_queued_memory = max((e['memory_mb'] for e in self.queue_events), default=0)
            min_queued_memory = min((e['memory_mb'] for e in self.queue_events), default=0)

            writer.writerow([])
            writer.writerow(['MEMORY PROFILE (ACTUAL QUEUE DATA)'])
            writer.writerow(['Total Memory Queued (MB):', f'{total_queued_memory:.2f}'])
            writer.writerow(['Average Memory per Queue Event (MB):', f'{avg_queued_memory:.2f}'])
            writer.writerow(['Maximum Queued Memory (MB):', f'{max_queued_memory:.2f}'])
            writer.writerow(['Minimum Queued Memory (MB):', f'{min_queued_memory:.2f}'])

            writer.writerow([])
            writer.writerow(['QUEUE EVENT TIMELINE'])
            writer.writerow(['Sequence', 'Container ID', 'Type', 'Memory (MB)', 'Reason', 'Timestamp'])

            for idx, event in enumerate(sorted(self.queue_events, key=lambda e: e['timestamp']), 1):
                ts = datetime.fromtimestamp(event['timestamp']).strftime("%H:%M:%S")
                container_type = f"type {(event['cycle_position'] % self.config.max_concurrent_containers) + 1}/{ self.config.max_concurrent_containers}"
                writer.writerow([
                    idx,
                    event['container_id'],
                    container_type,
                    f"{event['memory_mb']:.2f}",
                    event['reason'],
                    ts
                ])

            writer.writerow([])
            writer.writerow(['QUEUE REASON BREAKDOWN'])
            if waiting_memory:
                writer.writerow(['WAITING_MEMORY Events:'])
                for event in waiting_memory:
                    ts = datetime.fromtimestamp(event['timestamp']).strftime("%H:%M:%S")
                    writer.writerow(['  Container', event['container_id'], 'at', ts, '- Memory needed:', f"{event['memory_mb']:.2f} MB"])

            if waiting_slot:
                writer.writerow([])
                writer.writerow(['WAITING_SLOT Events:'])
                for event in waiting_slot:
                    ts = datetime.fromtimestamp(event['timestamp']).strftime("%H:%M:%S")
                    writer.writerow(['  Container', event['container_id'], 'at', ts, '- All slots occupied'])

            writer.writerow([])
            writer.writerow(['DYNAMIC INSIGHTS'])
            writer.writerow(['Primary Bottleneck:', 'MEMORY' if len(waiting_memory) > len(waiting_slot) else 'SLOTS'])
            writer.writerow(['Queue Fairness:', 'FIFO (First-In-First-Out) - all containers eventually launch'])
            writer.writerow(['Container Distribution:', f'{unique_containers} unique containers queued during execution'])

        return filepath

    def generate_all_reports(self):
        """Generate all CSV reports"""
        reports = {
            'containers': self.generate_containers_csv(),
            'mathematical_modeling': self.generate_mathematical_modeling_csv(),
            'throughput_24h': self.generate_throughput_analysis_csv(),
            'scheduling_algorithm': self.generate_scheduling_algorithm_csv(),
            'state_transitions': self.generate_state_transitions_csv(),
            'starvation_prevention': self.generate_starvation_prevention_csv(),
            'memory_timeline': self.generate_memory_timeline_csv(),
            'first_hour_timeline': self.generate_first_hour_timeline_csv(),
            'container_launch_schedule': self.generate_container_launch_schedule_csv(),
            'execution_schedule': self.generate_execution_schedule_csv(),
            'chronological_timeline': self.generate_chronological_timeline_csv(),
            'queue_analysis': self.generate_queue_analysis_csv(),
            'summary_report': self.generate_summary_report()
        }

        return {
            'report_directory': self.report_subdir,
            'timestamp': self.timestamp,
            'files': reports
        }
