# Docker Setup for RunPod Serverless

## 🐳 Building the Docker Image

### Option 1: Using the build script
```bash
cd examples/soccer
./docker-build.sh
```

### Option 2: Manual build
```bash
# From SMO_FINAL root directory
docker build -f examples/soccer/Dockerfile -t soccer-analysis:latest .
```

## 📦 Image Details

- **Base**: `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
- **Python**: 3.10
- **GPU**: CUDA 12.1 support
- **Size**: ~5-6 GB (includes all dependencies)

## 🚀 Deploying to RunPod

### 1. Push to Docker Hub (or your registry)

```bash
# Tag and push
docker tag soccer-analysis:latest YOUR_DOCKERHUB_USER/soccer-analysis:latest
docker push YOUR_DOCKERHUB_USER/soccer-analysis:latest
```

### 2. Create RunPod Serverless Endpoint

1. Go to RunPod → Serverless → Endpoints
2. Create new endpoint
3. Use image: `YOUR_DOCKERHUB_USER/soccer-analysis:latest`
4. Set handler: `/app/handler.py`
5. Configure GPU: RTX 4090 or similar (24GB VRAM recommended)

### 3. Input Format

```json
{
  "input": {
    "video_url": "https://example.com/video.mp4",
    "device": "cuda",
    "output_dir": "/tmp/output"
  }
}
```

Or with local path:
```json
{
  "input": {
    "video_path": "/tmp/video.mp4",
    "device": "cuda"
  }
}
```

### 4. Output Format

```json
{
  "status": "success",
  "video": "/tmp/video.mp4",
  "telemetry_file": "/tmp/output/telemetry.jsonl",
  "metadata_file": "/tmp/output/metadata.json",
  "passes_csv": "/tmp/output/passes_from_telemetry.csv",
  "passes_json": "/tmp/output/passes_from_telemetry.json",
  "total_passes": 22,
  "passes_data": [...]
}
```

## 🧪 Testing Locally

```bash
# Test with GPU
docker run --gpus all \
  -v /path/to/video:/tmp/video.mp4 \
  -v /path/to/output:/tmp/output \
  soccer-analysis:latest \
  python handler.py <<< '{"input": {"video_path": "/tmp/video.mp4", "output_dir": "/tmp/output"}}'
```

## 📋 Requirements

- Docker with GPU support (nvidia-docker2)
- CUDA 12.1 compatible GPU
- 24GB+ VRAM recommended
- Internet connection (for model downloads on first run)

## 🔧 Environment Variables

- `CUDA_VISIBLE_DEVICES`: GPU device selection
- `PYTHONUNBUFFERED`: Real-time output
- `NVIDIA_VISIBLE_DEVICES`: GPU visibility

## 📝 Notes

- Models are downloaded automatically on first run via `setup.sh`
- Analysis outputs are saved to `output_dir` or `video_dir/analysis/`
- All large files (models, videos) are excluded from image via `.dockerignore`

