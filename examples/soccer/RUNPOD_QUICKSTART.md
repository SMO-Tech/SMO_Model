# 🚀 RunPod Serverless Quick Start

## Step 1: Build Docker Image

```bash
cd /workspace/niyas_version2/SMO_FINAL
./examples/soccer/docker-build.sh
```

Or manually:
```bash
docker build -f Dockerfile -t soccer-analysis:latest .
```

## Step 2: Push to Docker Hub

```bash
# Login to Docker Hub
docker login

# Tag your image
docker tag soccer-analysis:latest YOUR_USERNAME/soccer-analysis:latest

# Push
docker push YOUR_USERNAME/soccer-analysis:latest
```

## Step 3: Create RunPod Endpoint

1. Go to [RunPod Serverless](https://www.runpod.io/serverless)
2. Click **"Create Endpoint"**
3. Fill in:
   - **Name**: `soccer-analysis`
   - **Container Image**: `YOUR_USERNAME/soccer-analysis:latest`
   - **Handler**: `/app/handler.py`
   - **GPU Type**: RTX 4090 (24GB) or similar
   - **Container Disk**: 20GB (for models)

## Step 4: Test the Endpoint

### Input JSON:
```json
{
  "input": {
    "video_url": "https://example.com/video.mp4",
    "device": "cuda"
  }
}
```

### Expected Output:
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

## 📊 What It Does

1. **Downloads video** (if URL provided)
2. **Generates telemetry** - Player/ball tracking (20 fps, GPU-accelerated)
3. **Detects passes** - Team A/B classification, success/intercept detection
4. **Returns results** - CSV + JSON with all pass events

## ⚡ Performance

- **Speed**: ~20 frames/second (GPU-accelerated)
- **Accuracy**: Position-based team detection (fast & accurate)
- **Output**: Complete pass analysis with timestamps

## 🔧 Troubleshooting

**GPU not detected?**
- Ensure RunPod endpoint has GPU enabled
- Check `CUDA_VISIBLE_DEVICES` environment variable

**Models not found?**
- Models download automatically on first run
- Takes ~2 minutes to download 3 models (~400MB total)

**Out of memory?**
- Use RTX 4090 (24GB) or larger
- Reduce video resolution if needed

