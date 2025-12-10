# Google Cloud Setup Guide

## Quick Setup Steps

### 1. Add gcloud to PATH (Permanent)

Add this to your `~/.zshrc` file:

```bash
export PATH="$PATH:/Users/essashah/Desktop/google-cloud-sdk/bin"
```

Then reload:

```bash
source ~/.zshrc
```

### 2. Initialize gcloud

```bash
gcloud init
```

This will:

- Open a browser for authentication
- Let you select/create a Google Cloud project
- Set default region/zone

### 3. Authenticate (if init doesn't work)

```bash
gcloud auth login
```

### 4. Set Your Project

```bash
# List available projects
gcloud projects list

# Set your project
gcloud config set project YOUR_PROJECT_ID
```

### 5. Enable Required APIs

For GPU instances, enable:

```bash
gcloud services enable compute.googleapis.com
```

## Using GPU Instances for Soccer Detection

### Option 1: Create a GPU VM Instance

```bash
# Create a VM with GPU (NVIDIA T4 example)
gcloud compute instances create soccer-gpu-vm \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=common-cu121 \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-ssd
```

### Option 2: Connect to Existing Instance

```bash
# SSH into your instance
gcloud compute ssh soccer-gpu-vm --zone=us-central1-a
```

### Option 3: Use Cloud Shell (Easiest for Testing)

```bash
# Open Cloud Shell in browser
gcloud cloud-shell ssh
```

## Upload Your Code to Cloud

### Using gcloud storage:

```bash
# Create a bucket
gsutil mb gs://your-bucket-name

# Upload your code
gsutil -m cp -r /path/to/your/code gs://your-bucket-name/
```

### Using SCP:

```bash
# Copy files to VM
gcloud compute scp --recurse \
  /path/to/local/code \
  soccer-gpu-vm:/home/your-username/ \
  --zone=us-central1-a
```

## Run Your Soccer Detection on GPU

Once connected to GPU instance:

```bash
# Install dependencies
pip install ultralytics supervision opencv-python numpy tqdm transformers

# Run your detection
python main.py \
  --source_video_path input_video/test_video.mp4 \
  --target_video_path output.mp4 \
  --device cuda \
  --mode COMBINED_DETECTION \
  --batch_size 4
```

## Quick Commands Reference

```bash
# Check authentication
gcloud auth list

# Check current project
gcloud config get-value project

# List all instances
gcloud compute instances list

# Start an instance
gcloud compute instances start INSTANCE_NAME --zone=ZONE

# Stop an instance
gcloud compute instances stop INSTANCE_NAME --zone=ZONE

# Delete an instance (careful!)
gcloud compute instances delete INSTANCE_NAME --zone=ZONE
```

## Cost Optimization Tips

1. **Stop instances when not in use** - GPU instances cost money even when idle
2. **Use preemptible instances** - 80% cheaper but can be terminated
3. **Choose right GPU** - T4 is cheaper than A100 for most tasks
4. **Set up billing alerts** - Monitor your spending

## Troubleshooting

### Permission denied errors:

```bash
sudo chmod -R 755 ~/.config/gcloud
```

### GPU not detected:

```bash
# Check NVIDIA driver
nvidia-smi

# Install CUDA if needed
# (usually pre-installed on deeplearning-platform images)
```

### Authentication issues:

```bash
gcloud auth application-default login
```
