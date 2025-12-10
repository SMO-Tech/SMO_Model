#!/bin/bash
# Fix SSH config for GCP GPU instance

echo "🔧 Fixing SSH config for gcp-gpu..."

# Backup current config
cp ~/.ssh/config ~/.ssh/config.backup.$(date +%Y%m%d_%H%M%S)
echo "✅ Backup created"

# Get current instance IP
export PATH="$PATH:/Users/essashah/Desktop/google-cloud-sdk/bin"
export CLOUDSDK_CONFIG=~/gcloud-config

INSTANCE_IP=$(gcloud compute instances list --filter="name:smo-gpu-machine" --format="value(EXTERNAL_IP)")
INSTANCE_ZONE=$(gcloud compute instances list --filter="name:smo-gpu-machine" --format="value(ZONE)")

if [ -z "$INSTANCE_IP" ]; then
    echo "❌ Could not find instance. Make sure it's running."
    exit 1
fi

echo "📍 Found instance: smo-gpu-machine"
echo "   IP: $INSTANCE_IP"
echo "   Zone: $INSTANCE_ZONE"

# Remove old gcp-gpu entries
sed -i.bak '/^Host gcp-gpu$/,/^$/d' ~/.ssh/config 2>/dev/null || sed -i '' '/^Host gcp-gpu$/,/^$/d' ~/.ssh/config

# Add correct entry
cat >> ~/.ssh/config << EOF

Host gcp-gpu
    HostName $INSTANCE_IP
    User essashah10
    IdentityFile ~/.ssh/id_rsa
    IdentitiesOnly yes
    ServerAliveInterval 60
    ServerAliveCountMax 3
EOF

echo "✅ SSH config updated!"
echo ""
echo "🧪 Testing connection..."
ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no gcp-gpu "echo 'Connection successful!'" 2>&1

echo ""
echo "✅ Done! You can now connect with: ssh gcp-gpu"

