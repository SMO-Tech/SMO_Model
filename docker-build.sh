#!/bin/bash

# Replace with your Docker Hub username
DOCKERHUB_USERNAME="utsavgohel"
IMAGE_NAME="smo-final"
TAG="latest"

# Build the Docker image
echo "Building Docker image: ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TAG}"
docker build -t ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TAG} .

# Push to Docker Hub
if [ $? -eq 0 ]; then
    echo "Pushing Docker image to Docker Hub..."
    docker push ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TAG}
    echo "Deployment image pushed to Docker Hub!"
    echo "Now go to RunPod console and deploy using: ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TAG}"
else
    echo "Docker build failed. Not pushing to Docker Hub."
fi
