# Docker Deployment Guide for Drone Detection Dashboard

This guide explains how to run the Drone Detection Dashboard using Docker, making it easy to deploy across different systems.

## Prerequisites

- **Docker** (19.03+): [Install Docker](https://docs.docker.com/get-docker/)
- **Docker Compose** (1.25+): [Install Docker Compose](https://docs.docker.com/compose/install/)
- **For hardware access**: Linux (for audio/camera device mounting)

## Quick Start

### 1. Build and Run with Docker Compose (Recommended)

```bash
cd /path/to/Drone-Detect
docker-compose up --build
```

This will:
- Build the Docker image
- Start the dashboard server on port 9000
- Mount audio and camera devices automatically
- Enable auto-restart on failure

### 2. Access the Dashboard

Open your browser and navigate to:
```
http://localhost:9000
```

Or view the HTML directly:
```
http://localhost:9000/dashboard
```

### 3. Stop the Container

```bash
docker-compose down
```

---

## Manual Docker Commands (Without Compose)

### Build the Image

```bash
docker build -t drone-dashboard:latest .
```

### Run the Container

```bash
docker run -d \
  --name drone-dashboard \
  -p 9000:9000 \
  --device /dev/snd \
  --device /dev/video0 \
  -v /dev/snd:/dev/snd \
  -v /dev/bus/usb:/dev/bus/usb \
  --restart unless-stopped \
  drone-dashboard:latest
```

### View Logs

```bash
docker logs -f drone-dashboard
```

### Stop the Container

```bash
docker stop drone-dashboard
docker rm drone-dashboard
```

---

## Configuration

### Environment Variables

Edit `docker-compose.yml` to add environment variables:

```yaml
environment:
  - FLASK_ENV=production
  - LOG_LEVEL=INFO
  - AUDIO_DEVICE=0
  - CAMERA_DEVICE=0
```

### Volume Mounts

To persist detection recordings and data:

```yaml
volumes:
  - ./detections:/app/detections
  - ./recordings:/app/recordings
  - ./logs:/app/logs
```

### Port Configuration

Change the port mapping in `docker-compose.yml`:

```yaml
ports:
  - "8080:9000"  # Access on port 8080 instead
```

---

## GPU Support (NVIDIA Only)

To enable GPU support for YOLO inference:

### 1. Install NVIDIA Docker Runtime

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 2. Update docker-compose.yml

```yaml
services:
  drone-dashboard:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

### 3. Run with GPU

```bash
docker-compose up --build
```

---

## Troubleshooting

### Container won't start

```bash
docker logs drone-dashboard
```

Check for:
- Port 9000 already in use: `sudo lsof -i :9000`
- Missing audio device: `ls -la /dev/snd/`
- Missing camera device: `ls -la /dev/video*`

### No audio input detected

On Linux, ensure your user is in the `audio` group:

```bash
sudo usermod -aG audio $USER
# Log out and back in
```

### Camera not detected

List available camera devices:

```bash
ls -la /dev/video*
```

Update `docker-compose.yml` to mount the correct device:

```yaml
devices:
  - /dev/video2:/dev/video0  # Mount /dev/video2 as /dev/video0 inside container
```

### High CPU usage

Reduce the frame skip or inference image size in `dashboard_server.py`:

```python
detector = JetsonDroneDetector(
    frame_skip=3,  # Increase skip
    imgsz=320      # Reduce size
)
```

---

## Deployment to Cloud

### Docker Hub

Push your image to Docker Hub for easy distribution:

```bash
docker login
docker tag drone-dashboard:latest yourusername/drone-dashboard:latest
docker push yourusername/drone-dashboard:latest
```

### AWS, Google Cloud, Azure

Most cloud platforms support Docker containers. Push to their registries:

```bash
# AWS ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag drone-dashboard:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/drone-dashboard:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/drone-dashboard:latest
```

---

## Production Considerations

### 1. Use a Reverse Proxy (Nginx)

Add to `docker-compose.yml`:

```yaml
nginx:
  image: nginx:latest
  ports:
    - "80:80"
    - "443:443"
  volumes:
    - ./nginx.conf:/etc/nginx/nginx.conf
    - ./ssl:/etc/nginx/ssl
  depends_on:
    - drone-dashboard
```

### 2. Enable HTTPS

Generate SSL certificates:

```bash
certbot certonly --standalone -d yourdomain.com
```

### 3. Resource Limits

```yaml
services:
  drone-dashboard:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### 4. Logging

```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

---

## Development Mode

To run the dashboard with live code reloading:

```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

Create `docker-compose.dev.yml`:

```yaml
version: '3.8'
services:
  drone-dashboard:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - .:/app  # Mount entire project for live reload
      - /app/__pycache__  # Exclude Python cache
    environment:
      - FLASK_ENV=development
      - FLASK_DEBUG=1
```

---

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)

---

For issues or questions, check the main README or open an issue on GitHub.
