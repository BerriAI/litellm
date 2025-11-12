#!/bin/bash
# Setup script for Vertex AI credentials using environment variables
# 
# This script helps configure your environment for Vertex AI Discovery passthrough
# without needing to put credentials in the config.yaml file

echo "🔧 Setting up Vertex AI credentials from environment variables"
echo "================================================================"
echo ""

# Prompt for GCP Project ID
read -p "Enter your GCP Project ID: " PROJECT_ID
export DEFAULT_VERTEXAI_PROJECT="$PROJECT_ID"
echo "✓ Set DEFAULT_VERTEXAI_PROJECT=$PROJECT_ID"

# Prompt for Vertex Location
echo ""
echo "Common locations: us-central1, global, us-east1, europe-west1"
read -p "Enter your Vertex AI location (default: global): " LOCATION
LOCATION=${LOCATION:-global}
export DEFAULT_VERTEXAI_LOCATION="$LOCATION"
echo "✓ Set DEFAULT_VERTEXAI_LOCATION=$LOCATION"

# Prompt for credentials file path
echo ""
read -p "Enter the path to your service account JSON file: " CREDS_PATH

# Expand ~ to home directory if needed
CREDS_PATH="${CREDS_PATH/#\~/$HOME}"

# Check if file exists
if [ -f "$CREDS_PATH" ]; then
    export DEFAULT_GOOGLE_APPLICATION_CREDENTIALS="$CREDS_PATH"
    echo "✓ Set DEFAULT_GOOGLE_APPLICATION_CREDENTIALS=$CREDS_PATH"
    
    # Also set standard GOOGLE_APPLICATION_CREDENTIALS for fallback
    export GOOGLE_APPLICATION_CREDENTIALS="$CREDS_PATH"
    echo "✓ Set GOOGLE_APPLICATION_CREDENTIALS=$CREDS_PATH"
else
    echo "⚠️  Warning: File not found at $CREDS_PATH"
    echo "   Please make sure the path is correct"
fi

echo ""
echo "================================================================"
echo "✅ Environment variables configured!"
echo ""
echo "Current settings:"
echo "  DEFAULT_VERTEXAI_PROJECT=$DEFAULT_VERTEXAI_PROJECT"
echo "  DEFAULT_VERTEXAI_LOCATION=$DEFAULT_VERTEXAI_LOCATION"
echo "  DEFAULT_GOOGLE_APPLICATION_CREDENTIALS=$DEFAULT_GOOGLE_APPLICATION_CREDENTIALS"
echo ""
echo "To make these persistent, add the following to your ~/.zshrc or ~/.bashrc:"
echo ""
echo "  export DEFAULT_VERTEXAI_PROJECT=\"$PROJECT_ID\""
echo "  export DEFAULT_VERTEXAI_LOCATION=\"$LOCATION\""
echo "  export DEFAULT_GOOGLE_APPLICATION_CREDENTIALS=\"$CREDS_PATH\""
echo "  export GOOGLE_APPLICATION_CREDENTIALS=\"$CREDS_PATH\""
echo ""
echo "You can now start LiteLLM proxy with:"
echo "  litellm --config proxy_server_config.yaml"
echo ""

