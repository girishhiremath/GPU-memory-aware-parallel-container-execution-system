#!/bin/bash
# Run multiple N-hour simulations with different configurations

echo "=================================="
echo "GPU Scheduler - Simulation Suite"
echo "=================================="
echo ""

# Create output directory
mkdir -p simulation_reports

# 1-Hour Quick Test
echo "Running 1-hour simulation..."
python3 simulator.py --hours 1 --output-dir simulation_reports
echo ""

# 6-Hour Test
echo "Running 6-hour simulation..."
python3 simulator.py --hours 6 --container-duration 10 --output-dir simulation_reports
echo ""

# 12-Hour Test
echo "Running 12-hour simulation..."
python3 simulator.py --hours 12 --output-dir simulation_reports
echo ""

# 24-Hour Full Test
echo "Running 24-hour simulation..."
python3 simulator.py --hours 24 --output-dir simulation_reports
echo ""

# Generate summary
echo "All simulations complete!"
ls -lh simulation_reports/
