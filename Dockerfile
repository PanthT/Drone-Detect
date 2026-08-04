FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for audio, video, and deep learning
RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    libasound2-dev \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app/

# Create requirements.txt (if not already present)
# RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
# Note: Adjust based on your actual requirements
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    matplotlib \
    opencv-python \
    pyaudio \
    ultralytics \
    torch \
    torchvision \
    torchaudio \
    PyYAML \
    Pillow \
    requests \
    websockets \
    flask

# Expose the dashboard server port
EXPOSE 9000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9000/status || exit 1

# Run the dashboard server
CMD ["python", "dashboard_server.py"]
