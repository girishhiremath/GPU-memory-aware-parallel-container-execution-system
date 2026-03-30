#!/usr/bin/env python3
"""Quick test to verify scheduler and reports work with Part 1 spec parameters"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scheduler.main import Scheduler, SchedulerConfig

def test_quick_5min():
    """Quick test: 5-minute simulation with 10-second containers"""
    print("\n" + "="*70)
    print("TEST 1: QUICK TEST (5 minutes simulation)")
    print("="*70)

    config = SchedulerConfig(
        total_gpu_memory_mb=4096,
        base_memory_mb=862,                    # Part 1 spec
        container_duration_seconds=10,         # 10 seconds (fast)
        step_interval_seconds=1,               # Check every 1 second
        max_concurrent_containers=3,           # Part 1 spec
        memory_multiplier=1.5,
        simulation_duration_hours=0.0833,      # 5 minutes
    )

    print(f"Configuration:")
    print(f"  Base Memory (M₀): {config.base_memory_mb} MB (Part 1 spec)")
    print(f"  Container Duration: {config.container_duration_seconds}s")
    print(f"  Simulation: {config.simulation_duration_hours*60:.1f} minutes")
    print(f"  Max Parallel: {config.max_concurrent_containers}")
    print(f"  Reset Interval: Dynamic (computed at startup)")
    print(f"  Expected Containers: ~30 (5 min ÷ 10s × 3 parallel)")
    print()

    scheduler = Scheduler(config)
    scheduler.start()

    try:
        start = time.time()
        while scheduler.running:
            scheduler.step()
            time.sleep(0.1)
            elapsed = time.time() - start
            if elapsed > config.simulation_duration_hours * 3600 + 10:
                break
    finally:
        scheduler.stop()

    print(f"\nTest 1 completed in {time.time() - start:.1f}s")
    return True


def test_quick_1hour():
    """Quick test: 1-hour simulation with 60-second containers"""
    print("\n" + "="*70)
    print("TEST 2: 1-HOUR SIMULATION (60-second containers)")
    print("="*70)

    config = SchedulerConfig(
        total_gpu_memory_mb=4096,
        base_memory_mb=862,
        container_duration_seconds=60,         # 1 minute
        step_interval_seconds=5,               # Check every 5 seconds
        max_concurrent_containers=3,
        memory_multiplier=1.5,
        simulation_duration_hours=1.0,
    )

    print(f"Configuration:")
    print(f"  Base Memory (M₀): {config.base_memory_mb} MB")
    print(f"  Container Duration: {config.container_duration_seconds}s")
    print(f"  Simulation: {config.simulation_duration_hours} hour")
    print(f"  Max Parallel: {config.max_concurrent_containers}")
    print(f"  Reset Interval: Dynamic (computed at startup)")
    print(f"  Expected Containers: ~180 (60 min ÷ 1 min × 3 parallel)")
    print()

    scheduler = Scheduler(config)
    scheduler.start()

    try:
        start = time.time()
        last_update = start
        while scheduler.running:
            scheduler.step()
            now = time.time()

            # Print progress every 10 seconds
            if now - last_update >= 10:
                elapsed_min = (now - start) / 60
                completed = len(scheduler.state_tracker.completed_containers)
                print(f"  {elapsed_min:.1f}m: {completed} containers completed")
                last_update = now

            time.sleep(0.5)
            if now - start > config.simulation_duration_hours * 3600 + 20:
                break
    finally:
        scheduler.stop()

    print(f"\nTest 2 completed in {time.time() - start:.1f}s")
    return True


def verify_reports():
    """Verify all CSV reports were generated"""
    print("\n" + "="*70)
    print("REPORT VERIFICATION")
    print("="*70)

    import glob
    report_files = glob.glob("reports/report_*/")

    if not report_files:
        print("No report directories found")
        return False

    latest_report = sorted(report_files)[-1]
    csv_files = glob.glob(f"{latest_report}*.csv")

    expected_reports = {
        "summary_report.csv": "High-level statistics",
        "math_modeling.csv": "Part 1 mathematical analysis",
        "throughput_24h.csv": "24-hour throughput metrics",
        "scheduling_algorithm.csv": "State machine details",
        "state_transitions.csv": "All 7-state transitions",
        "starvation_prevention.csv": "Reset policy comparison",
        "containers_data.csv": "Per-container execution data",
        "memory_timeline.csv": "Memory usage over time",
        "first_hour_timeline.csv": "First hour detailed timeline (NEW)",
        "container_launch_schedule.csv": "Container launch schedule (NEW)",
    }

    print(f"Report directory: {latest_report}")
    print(f"CSV files generated: {len(csv_files)}/10")
    print()

    found = set()
    for csv_file in sorted(csv_files):
        filename = os.path.basename(csv_file)
        if filename in expected_reports:
            found.add(filename)
            size = os.path.getsize(csv_file)
            print(f"{filename:<40} ({size:,} bytes) - {expected_reports[filename]}")
        else:
            print(f"  ? {filename:<40} (unknown file)")

    missing = set(expected_reports.keys()) - found
    if missing:
        print()
        for missing_file in sorted(missing):
            print(f"{missing_file:<40} - NOT FOUND")
        return False

    print()
    print(f"All 10 expected reports generated!")
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("GPU SCHEDULER - QUICK TEST SUITE")
    print("Aligned with Part 1 Mathematical Specification")
    print("="*70)

    try:
        # Run Test 1
        if not test_quick_5min():
            print("Test 1 failed")
            sys.exit(1)

        # Verify reports from Test 1
        if not verify_reports():
            print("Report verification failed")
            sys.exit(1)

        print("\n" + "="*70)
        print("QUICK TEST COMPLETE")
        print("="*70)
        print("\nTo run full 24-hour test:")
        print("  python scheduler/main.py")
        print("\nTo view reports:")
        print("  ls reports/report_*/")

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
