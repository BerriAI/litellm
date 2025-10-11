#!/bin/bash

# Quick rebuild script for performance testing
# 
# Usage: ./quick_rebuild.sh [commit_hash]
# 
# Examples:
#   ./quick_rebuild.sh                    # Build with local changes (no git reset)
#   ./quick_rebuild.sh HEAD               # Reset to HEAD and build
#   ./quick_rebuild.sh abc1234            # Reset to specific commit and build
#   ./quick_rebuild.sh origin/main        # Reset to origin/main and build
# 
# Behavior:
# - If no commit_hash is provided, uses local changes (no reset)
# - If 'HEAD' is explicitly provided, resets to HEAD
# - Docker tag is always the shortened commit hash

set -e

COMMIT_HASH=${1:-""}
IMAGE_NAME="alexsanderperf/litellm-perf-test"

echo "🚀 Quick rebuild for performance testing..."

if [ -z "$COMMIT_HASH" ]; then
    echo "📝 Using local changes (no git reset)"
    echo ""
    
    # Get the shortened commit hash for the tag (current HEAD + local changes)
    TAG=$(git rev-parse --short HEAD)
    echo "📋 Current status with local changes:"
    git log --oneline -1
    echo "🏷️  Using commit hash as tag: $TAG"
    echo ""
else
    echo "Target commit: $COMMIT_HASH"
    echo ""
    
    # Git operations
    echo "📦 Stashing any local changes..."
    git stash push -m "Auto-stash before rebuild $(date)"
    
    echo "🔄 Resetting to commit: $COMMIT_HASH"
    git reset --hard $COMMIT_HASH
    
    # Get the shortened commit hash for the tag
    TAG=$(git rev-parse --short HEAD)
    echo "📋 Current status:"
    git log --oneline -1
    echo "🏷️  Using commit hash as tag: $TAG"
    echo ""
fi

echo "Image: $IMAGE_NAME:$TAG"
echo "Platform: linux/amd64"
echo ""

# Build with correct platform (no cache to ensure fresh build)
echo "🏗️ Building image..."
docker build --platform linux/amd64 -f docker/Dockerfile.dev -t $IMAGE_NAME:$TAG .

# Push to Docker Hub
echo "Pushing to Docker Hub..."
docker push $IMAGE_NAME:$TAG

echo ""
echo "✅ Done! Your image is ready:"
echo "   $IMAGE_NAME:$TAG"
echo ""
echo "🌐 Use on Render: $IMAGE_NAME:$TAG"
