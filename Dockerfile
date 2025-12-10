FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
COPY examples/soccer/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install sports package
COPY sports /app/sports
COPY setup.py /app/
RUN pip install --no-cache-dir -e /app

# Copy application code
COPY examples/soccer /app/examples/soccer
WORKDIR /app/examples/soccer

# Download models (if not already present)
RUN chmod +x setup.sh && \
    bash setup.sh || echo "Models will be downloaded on first run"

# Copy handler script to root
COPY examples/soccer/handler.py /app/handler.py
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Expose port (if needed for API)
EXPOSE 8000

# Run handler
CMD ["python", "handler.py"]

