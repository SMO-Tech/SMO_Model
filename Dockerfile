FROM runpod/base:0.4.0-cuda11.8.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements.txt and install Python dependencies
COPY examples/soccer/requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install PyTorch with CUDA 11.8
RUN pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Copy your entire application code
COPY . .

# Set the handler as the entry point
CMD [ "python", "-u", "/app/examples/soccer/handler.py" ]