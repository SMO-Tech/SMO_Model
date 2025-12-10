#!/bin/bash
# Build script for Docker image

IMAGE_NAME="soccer-analysis"
IMAGE_TAG="latest"

echo "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"

# Build from SMO_FINAL root directory to include sports package
cd "$(dirname "$0")/../.."
docker build -f Dockerfile -t ${IMAGE_NAME}:${IMAGE_TAG} .

echo "✅ Build complete!"
echo ""
echo "To test locally:"
echo "  docker run --gpus all -v /path/to/video:/tmp/video.mp4 ${IMAGE_NAME}:${IMAGE_TAG} python handler.py <<< '{\"input\": {\"video_path\": \"/tmp/video.mp4\"}}'"
echo ""
echo "To push to Docker Hub:"
echo "  docker tag ${IMAGE_NAME}:${IMAGE_TAG} YOUR_USERNAME/${IMAGE_NAME}:${IMAGE_TAG}"
echo "  docker push YOUR_USERNAME/${IMAGE_NAME}:${IMAGE_TAG}"

