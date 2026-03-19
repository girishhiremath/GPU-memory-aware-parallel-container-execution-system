#!/usr/bin/env python3
"""
GPU Scheduler Simulator - Configurable N-hour simulation
No hardware required - simulates all scheduling decisions and memory management
"""
import json
import csv
import os
import sys
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict
import argparse

@dataclass
class SimulationConfig:
    """Configuration for simulator"""
    simulation_hours: int = 24
    gpu_memory_mb: int = 4096
    base_memory_mb: int = 862
    memory_multiplier: float = 1.5
    reset_interval: int = 3
    max_concurrent: int = 3
    container_duration_seconds: int = 600
    launch_interval_seconds: int = 5

@dataclass
class Container:
    """Simulated container"""
    container_id: int
    memory_mb: float
    launch_time_seconds: int
    complete_time_seconds: int
    state: str = "COMPLETED"
    
    def duration_seconds(self):
        return self.complete_time_seconds - self.launch_time_seconds

class GPUSchedulerSimulator:
    """Simulate GPU scheduler without hardware"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.containers: List[Container] = []
        self.memory_timeline: List[Dict] = []
        self.launch_times: List[int] = []
        self.simulation_duration_seconds = config.simulation_hours * 3600
        self.oom_count = 0
        self.zombie_count = 0
        
    def run_simulation(self):
        """Run the N-hour simulation"""
        print(f"\n{'='*80}")
        print(f"GPU SCHEDULER SIMULATOR - {self.config.simulation_hours}h Simulation")
        print(f"{'='*80}\n")
        
        print(f"Configuration:")
        print(f"  Duration: {self.config.simulation_hours} hours ({self.simulation_duration_seconds}s)")
        print(f"  GPU Memory: {self.config.gpu_memory_mb} MB")
        print(f"  Base Memory: {self.config.base_memory_mb} MB")
        print(f"  Reset Interval: {self.config.reset_interval} containers")
        print(f"  Container Duration: {self.config.container_duration_seconds}s")
        print(f"  Launch Interval: {self.config.launch_interval_seconds}s\n")
        
        current_time = 0
        container_id = 1
        allocated_memory = 0
        running_containers = []
        
        print(f"Simulating scheduling...\n")
        
        while current_time < self.simulation_duration_seconds:
            # Check for completed containers
            newly_completed = []
            for cid, complete_time in running_containers[:]:
                if current_time >= complete_time:
                    container = self.containers[cid - 1]
                    allocated_memory -= container.memory_mb
                    running_containers.remove((cid, complete_time))
                    newly_completed.append(cid)
            
            # Try to launch new container
            if len(running_containers) < self.config.max_concurrent:
                # Calculate memory based on launch order
                position = (container_id - 1) % self.config.reset_interval
                memory = self.config.base_memory_mb * (self.config.memory_multiplier ** position)
                
                # Check OOM
                if allocated_memory + memory <= self.config.gpu_memory_mb:
                    # Launch container
                    complete_time = current_time + self.config.container_duration_seconds
                    
                    if complete_time <= self.simulation_duration_seconds + self.config.container_duration_seconds:
                        container = Container(
                            container_id=container_id,
                            memory_mb=memory,
                            launch_time_seconds=current_time,
                            complete_time_seconds=complete_time
                        )
                        self.containers.append(container)
                        running_containers.append((container_id, complete_time))
                        allocated_memory += memory
                        self.launch_times.append(current_time)
                        container_id += 1
                else:
                    self.oom_count += 1
            
            # Record memory snapshot every 30 seconds
            if current_time % 30 == 0 or current_time in [0]:
                self.memory_timeline.append({
                    'timestamp_s': current_time,
                    'active_containers': len(running_containers),
                    'memory_mb': allocated_memory,
                    'utilization_percent': (allocated_memory / self.config.gpu_memory_mb) * 100,
                    'running_ids': [cid for cid, _ in running_containers]
                })
            
            # Advance time
            current_time += self.config.launch_interval_seconds
        
        print(f" Simulation complete!")
        print(f"   Containers launched: {len(self.containers)}")
        print(f"   OOM events: {self.oom_count}")
        print(f"   Zombie containers: {self.zombie_count}\n")
    
    def generate_json_report(self, output_dir: str = "simulation_reports"):
        """Generate JSON simulation report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate statistics
        containers_completed = len([c for c in self.containers if c.complete_time_seconds <= self.simulation_duration_seconds])
        total_memory_used = sum(c.memory_mb for c in self.containers)
        peak_memory = max((s['memory_mb'] for s in self.memory_timeline), default=0)
        
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "simulator_version": "1.0",
                "simulation_type": "GPU Scheduler (No Hardware Required)"
            },
            "simulation": {
                "duration_hours": self.config.simulation_hours,
                "duration_minutes": self.config.simulation_hours * 60,
                "duration_seconds": self.simulation_duration_seconds,
                "status": "SIMULATED"
            },
            "configuration": asdict(self.config),
            "summary": {
                "total_containers_launched": len(self.containers),
                "containers_completed_within_window": containers_completed,
                "containers_running_at_end": len(self.containers) - containers_completed,
                "success_rate_percent": (containers_completed / max(len(self.containers), 1)) * 100,
                "oom_events": self.oom_count,
                "zombie_events": self.zombie_count
            },
            "memory_analysis": {
                "total_memory_allocated_mb": total_memory_used,
                "peak_memory_mb": peak_memory,
                "peak_utilization_percent": (peak_memory / self.config.gpu_memory_mb) * 100,
                "average_container_memory_mb": total_memory_used / max(len(self.containers), 1)
            },
            "throughput": {
                "containers_per_hour": (len(self.containers) / max(self.config.simulation_hours, 1)),
                "containers_per_minute": (len(self.containers) / max(self.config.simulation_hours * 60, 1)),
                "average_launch_interval_seconds": self.config.launch_interval_seconds
            },
            "timeline_samples": {
                "first_snapshot": self.memory_timeline[0] if self.memory_timeline else None,
                "last_snapshot": self.memory_timeline[-1] if self.memory_timeline else None,
                "mid_snapshot": self.memory_timeline[len(self.memory_timeline)//2] if self.memory_timeline else None
            },
            "containers_sample": [asdict(c) for c in self.containers[:10]]
        }
        
        report_path = os.path.join(output_dir, f"simulation_{self.config.simulation_hours}h_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report_path
    
    def generate_csv_reports(self, output_dir: str = "simulation_reports"):
        """Generate CSV simulation reports"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Memory Timeline CSV
        memory_csv = os.path.join(output_dir, f"memory_timeline_{self.config.simulation_hours}h_{timestamp}.csv")
        with open(memory_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp_s', 'time_hms', 'active_containers', 'memory_mb', 'utilization_percent'])
            writer.writeheader()
            for snapshot in self.memory_timeline:
                hours = snapshot['timestamp_s'] // 3600
                minutes = (snapshot['timestamp_s'] % 3600) // 60
                seconds = snapshot['timestamp_s'] % 60
                writer.writerow({
                    'timestamp_s': snapshot['timestamp_s'],
                    'time_hms': f"{hours:02d}:{minutes:02d}:{seconds:02d}",
                    'active_containers': snapshot['active_containers'],
                    'memory_mb': round(snapshot['memory_mb'], 1),
                    'utilization_percent': round(snapshot['utilization_percent'], 2)
                })
        
        # 2. Container Launch Schedule CSV
        schedule_csv = os.path.join(output_dir, f"container_schedule_{self.config.simulation_hours}h_{timestamp}.csv")
        with open(schedule_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['container_id', 'memory_mb', 'launch_time_s', 'complete_time_s', 'duration_s', 'status'])
            writer.writeheader()
            for container in self.containers:
                status = "COMPLETED" if container.complete_time_seconds <= self.simulation_duration_seconds else "RUNNING"
                writer.writerow({
                    'container_id': f"C{container.container_id}",
                    'memory_mb': round(container.memory_mb, 1),
                    'launch_time_s': container.launch_time_seconds,
                    'complete_time_s': container.complete_time_seconds,
                    'duration_s': container.duration_seconds(),
                    'status': status
                })
        
        # 3. Summary Statistics CSV
        summary_csv = os.path.join(output_dir, f"summary_{self.config.simulation_hours}h_{timestamp}.csv")
        with open(summary_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Simulation Duration (hours)', self.config.simulation_hours])
            writer.writerow(['Total Containers', len(self.containers)])
            writer.writerow(['Containers Completed', len([c for c in self.containers if c.complete_time_seconds <= self.simulation_duration_seconds])])
            writer.writerow(['Peak Memory (MB)', round(max((s['memory_mb'] for s in self.memory_timeline), default=0), 1)])
            writer.writerow(['Peak GPU Utilization (%)', round((max((s['memory_mb'] for s in self.memory_timeline), default=0) / self.config.gpu_memory_mb) * 100, 2)])
            writer.writerow(['OOM Events', self.oom_count])
            writer.writerow(['Containers/Hour', round(len(self.containers) / max(self.config.simulation_hours, 1), 2)])
        
        return [memory_csv, schedule_csv, summary_csv]

def main():
    parser = argparse.ArgumentParser(description="GPU Scheduler Simulator - N-hour simulation")
    parser.add_argument('--hours', type=int, default=24, help='Number of hours to simulate (default: 24)')
    parser.add_argument('--gpu-memory', type=int, default=4096, help='GPU memory in MB (default: 4096)')
    parser.add_argument('--base-memory', type=int, default=862, help='Base container memory in MB (default: 862)')
    parser.add_argument('--multiplier', type=float, default=1.5, help='Memory multiplier (default: 1.5)')
    parser.add_argument('--reset-interval', type=int, default=3, help='Reset interval in containers (default: 3)')
    parser.add_argument('--max-concurrent', type=int, default=3, help='Max concurrent containers (default: 3)')
    parser.add_argument('--container-duration', type=int, default=600, help='Container duration in seconds (default: 600)')
    parser.add_argument('--output-dir', type=str, default='simulation_reports', help='Output directory for reports')
    
    args = parser.parse_args()
    
    config = SimulationConfig(
        simulation_hours=args.hours,
        gpu_memory_mb=args.gpu_memory,
        base_memory_mb=args.base_memory,
        memory_multiplier=args.multiplier,
        reset_interval=args.reset_interval,
        max_concurrent=args.max_concurrent,
        container_duration_seconds=args.container_duration
    )
    
    simulator = GPUSchedulerSimulator(config)
    simulator.run_simulation()
    
    # Generate reports
    print(f"Generating reports in {args.output_dir}/\n")
    json_report = simulator.generate_json_report(args.output_dir)
    csv_reports = simulator.generate_csv_reports(args.output_dir)
    
    print(f"JSON Report: {json_report}")
    for csv_report in csv_reports:
        print(f"CSV Report: {csv_report}")
    
    print(f"\n{'='*80}")
    print(f"Simulation Summary")
    print(f"{'='*80}")
    print(f"Duration: {args.hours} hours")
    print(f"Containers: {len(simulator.containers)}")
    print(f"Completed: {len([c for c in simulator.containers if c.complete_time_seconds <= simulator.simulation_duration_seconds])}")
    print(f"Peak Memory: {round(max((s['memory_mb'] for s in simulator.memory_timeline), default=0), 1)} MB")
    print(f"Peak Utilization: {round((max((s['memory_mb'] for s in simulator.memory_timeline), default=0) / config.gpu_memory_mb) * 100, 2)}%")
    print(f"OOM Events: {simulator.oom_count}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
