#!/bin/bash
# Worker entrypoint script
# Meets requirements from 4.2:
# - Accept env vars: CONTAINER_ID, MEMORY_MB, DURATION_SEC
# - Exit with code 0 on success, non-zero on OOM

set -e

# Verify required environment variables
if [ -z "$CONTAINER_ID" ]; then
    echo "ERROR: CONTAINER_ID not set"
    exit 1
fi

if [ -z "$MEMORY_MB" ]; then
    echo "ERROR: MEMORY_MB not set"
    exit 1
fi

# Use DURATION_SEC with fallback to DURATION_SECONDS
DURATION=${DURATION_SEC:-${DURATION_SECONDS:-600}}

echo "=================================="
echo "Starting GPU Memory Worker"
echo "=================================="
echo "CONTAINER_ID: $CONTAINER_ID"
echo "MEMORY_MB: $MEMORY_MB"
echo "DURATION_SEC: $DURATION"
echo "=================================="

# Set Python to unbuffered output
export PYTHONUNBUFFERED=1

# Run worker with exact environment variables as per requirement 4.2
python /app/worker/worker.py
EXIT_CODE=$?

echo "=================================="
echo "Worker completed with exit code: $EXIT_CODE"
echo "=================================="

exit $EXIT_CODE
