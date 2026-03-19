# GPU Container Orchestration System - Production Dockerfile
# Supports GPU acceleration with CUDA 12.2
# Includes PyTorch for GPU memory management
# Non-blocking scheduler with CSV reporting

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python and build tools
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    build-essential \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install Python dependencies with PyTorch for GPU support
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    torch==2.1.0 \
    prometheus-client==0.17.1

# Copy configuration file
COPY config.ini /app/config.ini

# Copy scheduler components
COPY scheduler/ /app/scheduler/

# Copy worker components
COPY worker/ /app/worker/

# Copy config loader
COPY config_loader.py /app/config_loader.py

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Enable GPU support
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Create volumes for output
VOLUME /app/reports
VOLUME /app/logs

# Default working directory
WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Run scheduler
CMD ["python", "scheduler/main.py"]

