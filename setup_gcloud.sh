#!/bin/bash
# Google Cloud Setup Script with Fixed Permissions

echo "🔧 Setting up Google Cloud CLI with fixed permissions..."

# Add gcloud to PATH
export PATH="$PATH:/Users/essashah/Desktop/google-cloud-sdk/bin"

# Use alternative config location to avoid permission issues
export CLOUDSDK_CONFIG=~/gcloud-config
mkdir -p ~/gcloud-config

echo "✅ Using config location: ~/gcloud-config"
echo ""

# Check authentication
echo "🔐 Checking authentication..."
gcloud auth list

echo ""
echo "📋 Available projects:"
gcloud projects list 2>&1 | head -10

echo ""
echo "⚙️  Current configuration:"
gcloud config list

echo ""
echo "✅ Setup complete!"
echo ""
echo "💡 To use gcloud in future sessions, add these to your ~/.zshrc:"
echo "   export PATH=\"\$PATH:/Users/essashah/Desktop/google-cloud-sdk/bin\""
echo "   export CLOUDSDK_CONFIG=~/gcloud-config"

