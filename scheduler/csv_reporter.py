"""
CSV Report Generator for GPU Container Orchestration
Generates detailed CSV reports for mathematical modeling and scheduling analysis
"""
import csv
import os
import sys
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
            writer.writerow(['Total Memory Used', f'{total_memory:.1f}', 'MB'])
            writer.writerow(['Average Memory per Container', f'{avg_memory:.1f}', 'MB'])
            writer.writerow(['Memory Utilization', f'{(total_memory / self.config.total_gpu_memory_mb * 100):.1f}', '%'])

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

                three_ok = "YES" if three_parallel <= self.config.total_gpu_memory_mb else "NO"
                two_ok = "YES" if two_parallel <= self.config.total_gpu_memory_mb else "NO"
                one_ok = "YES" if one_parallel <= self.config.total_gpu_memory_mb else "NO"

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
                    "YES" if can_launch else "NO"
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
            writer.writerow(['Non-blocking Event-Driven Scheduler'])
            writer.writerow(['Configuration:'])
            writer.writerow([f'  Base Memory: {self.config.base_memory_mb} MB'])
            writer.writerow([f'  Multiplier: {self.config.memory_multiplier}'])
            writer.writerow([f'  Total GPU Memory: {self.config.total_gpu_memory_mb} MB'])
            writer.writerow([f'  Container Duration: {self.config.container_duration_seconds} seconds'])
            writer.writerow([f'  Max Concurrent: {self.config.max_concurrent_containers}'])
            writer.writerow([])

            writer.writerow(['SCHEDULER STATE TIMELINE - First 60 minutes of execution'])
            writer.writerow([])
            writer.writerow([
                'Time (min)',
                'Container ID',
                'State',
                'Memory Required (MB)',
                'Active Containers (count)',
                'Total Memory Used (MB)',
                'Available Memory (MB)',
                'Event Description',
                'Launch Decision',
                'Note'
            ])

            # Simulate first 60 minutes
            active_containers = {}  # {container_id: start_time}
            current_time = 0
            container_num = 1
            total_memory = 0

            # Simulate 60 minutes with 5-minute intervals
            for minute in range(0, 61, 5):
                current_time = minute

                # Check for completions
                completed = []
                for cid, start_time in list(active_containers.items()):
                    elapsed_minutes = (current_time - start_time)
                    if elapsed_minutes >= self.config.container_duration_minutes:
                        completed.append(cid)

                # Process completions
                for cid in completed:
                    memory = self.config.base_memory_mb * (self.config.memory_multiplier ** (cid - 1))
                    total_memory -= memory
                    del active_containers[cid]
                    available = self.config.total_gpu_memory_mb - total_memory
                    writer.writerow([
                        current_time,
                        cid,
                        'COMPLETED',
                        f"{memory:.1f}",
                        len(active_containers),
                        f"{total_memory:.1f}",
                        f"{available:.1f}",
                        f"Container {cid} released resources",
                        'N/A',
                        'Memory freed'
                    ])

                # Check if can launch new container
                if container_num <= 20:
                    memory = self.config.base_memory_mb * (self.config.memory_multiplier ** (container_num - 1))
                    available = self.config.total_gpu_memory_mb - total_memory

                    can_launch = (
                        len(active_containers) < self.config.max_concurrent_containers and
                        total_memory + memory <= self.config.total_gpu_memory_mb
                    )

                    if can_launch:
                        active_containers[container_num] = current_time
                        total_memory += memory
                        available = self.config.total_gpu_memory_mb - total_memory

                        writer.writerow([
                            current_time,
                            container_num,
                            'CREATED',
                            f"{memory:.1f}",
                            len(active_containers),
                            f"{total_memory:.1f}",
                            f"{available:.1f}",
                            f"Container {container_num} launched",
                            f'✓ YES ({len(active_containers)}/{self.config.max_concurrent_containers})',
                            f'{len(active_containers)} now active'
                        ])
                        container_num += 1
                    else:
                        # Still show decision even if can't launch
                        reason = ''
                        if len(active_containers) >= self.config.max_concurrent_containers:
                            reason = 'Max concurrent reached'
                        elif total_memory + memory > self.config.total_gpu_memory_mb:
                            reason = f'Insufficient memory: need {memory:.0f}MB, have {available:.0f}MB'
                        else:
                            reason = 'Unknown'

                        writer.writerow([
                            current_time,
                            container_num,
                            'WAITING',
                            f"{memory:.1f}",
                            len(active_containers),
                            f"{total_memory:.1f}",
                            f"{available:.1f}",
                            'Container waiting for resources',
                            f'NO ({reason})',
                            'Queued'
                        ])

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
                overall_status = "SUCCESS" if c.success else "FAILURE"

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

            writer.writerow(['Approach', 'Description', 'Max Containers', 'Max Parallel', 'Starvation?', 'Realism', 'Note'])

            approaches = [
                [
                    'Sliding Window',
                    'Memory keeps increasing for each new container following the exponential pattern',
                    '9',
                    'Decreases over time from 3 to 1',
                    'Yes',
                    'High',
                    'Most realistic approach, but it eventually causes starvation'
                ],
                [
                    'Reset Every N',
                    'Memory growth restarts after every 4 containers',
                    'Unlimited',
                    'Remains at 3',
                    'No',
                    'Low',
                    'Prevents starvation, but it is less realistic for real workloads'
                ],
                [
                    'Capped Multiplier',
                    'Memory grows until it reaches a 2048 MB limit per container',
                    'Unlimited',
                    'Eventually stabilizes at 1',
                    'No',
                    'Good',
                    'Best balance between realism and fairness, so this is the recommended approach'
                ]
            ]

            for approach in approaches:
                writer.writerow(approach)

            writer.writerow([])
            writer.writerow(['RECOMMENDATION: Capped Multiplier at 2048 MB'])
            writer.writerow(['Implementation: max_container_memory = min(calculated_memory, 2048)'])
            writer.writerow([])

            # Memory growth with different approaches
            writer.writerow(['Container Number', 'Sliding Window (MB)', 'Reset Every 4 (MB)', 'Capped at 2GB (MB)'])

            for n in range(1, self.config.num_containers_to_analyze + 1):
                sliding = self.config.base_memory_mb * (self.config.memory_multiplier ** (n - 1))
                reset = self.config.base_memory_mb * (self.config.memory_multiplier ** ((n - 1) % 4))
                capped = min(sliding, 2048)

                writer.writerow([
                    n,
                    f"{sliding:.0f}",
                    f"{reset:.0f}",
                    f"{capped:.0f}"
                ])

        return filepath

    def generate_memory_timeline_csv(self):
        """Generate memory_timeline.csv from recorded snapshots"""
        filepath = os.path.join(self.report_subdir, "memory_timeline.csv")

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            writer.writerow(['MEMORY USAGE TIMELINE'])
            writer.writerow(['Configuration:'])
            writer.writerow([f'  Total GPU Memory: {self.config.total_gpu_memory_mb} MB'])
            writer.writerow([f'  Base Memory: {self.config.base_memory_mb} MB'])
            writer.writerow([f'  Memory Multiplier: {self.config.memory_multiplier}'])
            writer.writerow([])

            writer.writerow([
                'Timestamp (s)',
                'Active Containers (count)',
                'Total Memory Used (MB)',
                'Used %',
                'Remaining Capacity (MB)',
                'Available %',
                'Memory Pressure',
                'Can Launch?'
            ])

            for entry in self.memory_timeline:
                utilization = (entry['total_memory_used_mb'] / self.config.total_gpu_memory_mb) * 100
                available_pct = 100 - utilization

                # Determine memory pressure level
                if utilization >= 90:
                    pressure = 'CRITICAL'
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

                writer.writerow([
                    f"{entry['timestamp']:.2f}",
                    entry['active_containers'],
                    f"{entry['total_memory_used_mb']:.1f}",
                    f"{utilization:.1f}%",
                    f"{entry['remaining_memory_mb']:.1f}",
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
                'Calculated'
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
                    'Feasible'
                ])

            # Q3: Throughput analysis
            writer.writerow([])
            writer.writerow(['Q3: Total throughput analysis'])
            writer.writerow(['Metric', 'Value', 'Unit'])
            total_slots = (self.config.simulation_hours * 60 * 60) // self.config.container_duration_seconds
            writer.writerow(['Simulation Duration', f'{self.config.simulation_hours}', 'hours'])
            writer.writerow(['Total Time Slots', f'{total_slots}', 'slots'])
            writer.writerow(['Slot Duration', f'{self.config.container_duration_seconds}', 'seconds'])
            writer.writerow(['Expected Max Containers', f'~{int(total_slots * 0.7)}', 'containers (estimated)'])
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
            writer.writerow(['Policy Approach', 'Description', 'Max Containers', 'Recommendation'])
            writer.writerow(['Sliding Window', 'Current: exponential continues', '9', 'Causes starvation ✗'])
            writer.writerow(['Reset Every N', 'Reset multiplier cycle', '∞', 'Unrealistic ✗'])
            writer.writerow(['Capped Multiplier', 'Cap at 2048 MB max', '∞ (steady state)', 'Recommended ✓'])
            writer.writerow([])
            writer.writerow(['Recommended Policy: Capped Multiplier at 2048 MB'])
            writer.writerow(['Rationale', 'Detail'])
            writer.writerow(['Realism', 'Reflects actual hardware constraints'])
            writer.writerow(['Fairness', 'All containers eventually scheduled'])
            writer.writerow(['Stability', 'System reaches steady state'])
            writer.writerow(['Testing', 'Covers both growth and limit scenarios'])
            writer.writerow([])

            writer.writerow(['Implementation Code'])
            writer.writerow(['Component', 'Implementation'])
            writer.writerow(['Max Memory Cap', 'MAX_CONTAINER_MEMORY = 2048  # MB'])
            writer.writerow(['Memory Calculation', 'memory = base_memory × (multiplier ^ (n-1))'])
            writer.writerow(['Apply Cap', 'return min(memory, MAX_CONTAINER_MEMORY)'])
            writer.writerow([])

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

        return filepath

    def generate_first_hour_timeline_csv(self):
        """Generate detailed timeline for first hour showing container scheduling decisions"""
        filepath = os.path.join(self.report_subdir, "first_hour_timeline.csv")

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            # Title and metadata
            writer.writerow(['DETAILED TIMELINE - FIRST HOUR'])
            writer.writerow(['Shows container execution, memory usage, and launch/OOM decisions'])
            writer.writerow([])
            writer.writerow(['Configuration:'])
            writer.writerow(['Base Memory (M0)', f'{self.config.base_memory_mb} MB'])
            writer.writerow(['Memory Multiplier', f'{self.config.memory_multiplier}x'])
            writer.writerow(['Total GPU Memory', f'{self.config.total_gpu_memory_mb} MB'])
            writer.writerow(['Container Duration', f'{self.config.container_duration_minutes} min'])
            writer.writerow(['Max Concurrent', f'{self.config.max_concurrent_containers}'])
            writer.writerow([])

            # Timeline header
            writer.writerow([
                'Time (min)',
                'Containers Running',
                'Container IDs',
                'Memory Used (MB)',
                'Memory %',
                'Remaining (MB)',
                'Can Launch New?',
                'Decision Reason'
            ])

            # Generate 60-minute timeline
            duration_minutes = self.config.container_duration_minutes
            total_gpu = self.config.total_gpu_memory_mb

            for time_min in range(0, 61, 5):
                # Calculate which containers are running at this time
                running = []
                memory_used = 0

                for cid in range(1, 100):  # Check up to 100 containers
                    launch_time = (cid - 1) * 5  # Each launched 5 minutes apart
                    end_time = launch_time + duration_minutes

                    if launch_time <= time_min < end_time:
                        running.append(cid)
                        # Calculate memory for this container (with reset every 3)
                        reset_interval = 3
                        cycle_pos = (cid - 1) % reset_interval
                        container_mem = self.config.base_memory_mb * (self.config.memory_multiplier ** cycle_pos)
                        memory_used += container_mem

                if not running:
                    continue

                # Check if next container can launch
                next_cid = max(running) + 1
                reset_interval = 3
                next_cycle_pos = (next_cid - 1) % reset_interval
                next_container_mem = self.config.base_memory_mb * (self.config.memory_multiplier ** next_cycle_pos)

                can_launch_count = len(running)
                can_launch = (can_launch_count < self.config.max_concurrent_containers and
                             memory_used + next_container_mem <= total_gpu)

                reason = "OK" if can_launch else ("OOM" if memory_used + next_container_mem > total_gpu else "Max parallel")

                writer.writerow([
                    time_min,
                    len(running),
                    ', '.join([f'C{c}' for c in running]),
                    f'{memory_used:.1f}',
                    f'{(memory_used/total_gpu)*100:.1f}%',
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
            writer.writerow(['Launch Interval', '5 minutes'])
            writer.writerow(['Total Simulation', f'{self.config.simulation_hours} hours'])
            writer.writerow(['Expected Containers', f'{int(self.config.simulation_hours * 60 / self.config.container_duration_minutes * self.config.max_concurrent_containers)}'])
            writer.writerow([])

            # Header
            writer.writerow([
                'Container',
                'Scheduled Launch (min)',
                'Expected Start Running (min)',
                'Expected End Running (min)',
                'Requested Memory (MB)',
                'Actual Launch',
                'Decision'
            ])

            # Generate launch schedule
            duration_minutes = self.config.container_duration_minutes
            total_gpu = self.config.total_gpu_memory_mb
            launch_interval = 5  # minutes

            container_id = 1
            time_min = 0
            reset_interval = 3
            running_containers = []

            for slot in range(0, int(self.config.simulation_hours * 60) + 60, launch_interval):
                # Check which containers will have completed
                completed = [c for c in running_containers if slot >= c['end_time']]
                running_containers = [c for c in running_containers if slot < c['end_time']]

                # Try to launch new container
                cycle_pos = (container_id - 1) % reset_interval
                container_mem = self.config.base_memory_mb * (self.config.memory_multiplier ** cycle_pos)

                # Calculate current memory usage
                current_memory = sum(c['memory'] for c in running_containers)

                # Check if can launch
                can_launch = (len(running_containers) < self.config.max_concurrent_containers and
                            current_memory + container_mem <= total_gpu)

                if can_launch or len(running_containers) < self.config.max_concurrent_containers or slot < 30:
                    launch_time = slot
                    end_time = launch_time + duration_minutes

                    running_containers.append({
                        'id': container_id,
                        'memory': container_mem,
                        'start': launch_time,
                        'end_time': end_time
                    })

                    decision = "YES - Launched"

                    writer.writerow([
                        f'C{container_id}',
                        launch_time,
                        launch_time,
                        end_time,
                        f'{container_mem:.1f}',
                        'Yes',
                        decision
                    ])

                    container_id += 1
                else:
                    # OOM - wait for container to complete
                    if running_containers:
                        next_end = min(c['end_time'] for c in running_containers)
                        decision = f"NO - OOM, wait until {next_end}min"
                    else:
                        decision = "NO - Unknown reason"

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
            'summary_report': self.generate_summary_report()
        }

        return {
            'report_directory': self.report_subdir,
            'timestamp': self.timestamp,
            'files': reports
        }
