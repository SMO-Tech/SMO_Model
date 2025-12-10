#!/bin/bash
# Google Cloud CLI Connection Script

echo "🔧 Setting up Google Cloud CLI..."

# Add gcloud to PATH
export PATH="$PATH:/Users/essashah/Desktop/google-cloud-sdk/bin"

echo "✅ gcloud added to PATH"
echo ""

# Check if already authenticated
if gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "✅ Already authenticated!"
    gcloud auth list
    echo ""
    echo "Current project:"
    gcloud config get-value project
else
    echo "🔐 Starting authentication..."
    echo ""
    echo "This will open your browser to sign in with Google."
    echo "Press Enter to continue..."
    read
    
    # Initialize gcloud (will open browser)
    gcloud init
    
    echo ""
    echo "✅ Setup complete!"
    echo ""
    echo "Your current configuration:"
    gcloud config list
fi

echo ""
echo "📋 Quick commands:"
echo "  - Check auth: gcloud auth list"
echo "  - Set project: gcloud config set project PROJECT_ID"
echo "  - List projects: gcloud projects list"
echo "  - Check quota: gcloud compute project-info describe"

