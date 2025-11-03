# Vertex AI Environment Variables Setup Guide

## Overview

LiteLLM can load Vertex AI credentials from environment variables instead of storing them in config files. This is more secure and easier to manage for local development.

## Environment Variables

LiteLLM looks for these environment variables (in order of precedence):

### 1. **DEFAULT_VERTEXAI_PROJECT** (Required)
Your GCP project ID that has Vertex AI enabled.

```bash
export DEFAULT_VERTEXAI_PROJECT="my-gcp-project-id"
```

### 2. **DEFAULT_VERTEXAI_LOCATION** (Required)
The region/location for Vertex AI services.

```bash
export DEFAULT_VERTEXAI_LOCATION="global"
# or
export DEFAULT_VERTEXAI_LOCATION="us-central1"
```

Common locations:
- `global` - For Discovery Engine and global services
- `us-central1` - US Central region
- `us-east1` - US East region
- `europe-west1` - Europe West region
- `asia-southeast1` - Asia Southeast region

### 3. **DEFAULT_GOOGLE_APPLICATION_CREDENTIALS** (Required)
Path to your service account JSON key file.

```bash
export DEFAULT_GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

### 4. **GOOGLE_APPLICATION_CREDENTIALS** (Fallback)
Standard Google Cloud environment variable (used as fallback).

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

## Quick Setup

### Option 1: Interactive Script

```bash
chmod +x setup_vertex_env.sh
source setup_vertex_env.sh
```

### Option 2: Manual Setup

1. **Set environment variables** (for current session):

```bash
export DEFAULT_VERTEXAI_PROJECT="your-project-id"
export DEFAULT_VERTEXAI_LOCATION="global"
export DEFAULT_GOOGLE_APPLICATION_CREDENTIALS="$HOME/.gcp/service-account.json"
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.gcp/service-account.json"
```

2. **Make them persistent** (add to `~/.zshrc` or `~/.bashrc`):

```bash
echo 'export DEFAULT_VERTEXAI_PROJECT="your-project-id"' >> ~/.zshrc
echo 'export DEFAULT_VERTEXAI_LOCATION="global"' >> ~/.zshrc
echo 'export DEFAULT_GOOGLE_APPLICATION_CREDENTIALS="$HOME/.gcp/service-account.json"' >> ~/.zshrc
echo 'export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.gcp/service-account.json"' >> ~/.zshrc
```

3. **Reload your shell**:

```bash
source ~/.zshrc
```

## Service Account Setup

### 1. Create a Service Account

```bash
gcloud iam service-accounts create litellm-vertex-sa \
    --display-name="LiteLLM Vertex AI Service Account"
```

### 2. Grant Necessary Permissions

For Discovery Engine (vector stores):
```bash
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:litellm-vertex-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/discoveryengine.viewer"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:litellm-vertex-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/discoveryengine.dataStoreEditor"
```

For general Vertex AI:
```bash
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:litellm-vertex-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"
```

### 3. Create and Download Key

```bash
gcloud iam service-accounts keys create ~/service-account-key.json \
    --iam-account=litellm-vertex-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

## Verify Setup

### Check Environment Variables

```bash
python3 << 'EOF'
import os
print("✓ Environment Variables:")
print(f"  DEFAULT_VERTEXAI_PROJECT: {os.getenv('DEFAULT_VERTEXAI_PROJECT')}")
print(f"  DEFAULT_VERTEXAI_LOCATION: {os.getenv('DEFAULT_VERTEXAI_LOCATION')}")
print(f"  DEFAULT_GOOGLE_APPLICATION_CREDENTIALS: {os.getenv('DEFAULT_GOOGLE_APPLICATION_CREDENTIALS')}")
print(f"  GOOGLE_APPLICATION_CREDENTIALS: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")

# Check if credentials file exists
creds_path = os.getenv('DEFAULT_GOOGLE_APPLICATION_CREDENTIALS')
if creds_path and os.path.exists(creds_path):
    print(f"\n✅ Credentials file found at: {creds_path}")
else:
    print(f"\n❌ Credentials file NOT found at: {creds_path}")
EOF
```

### Test Authentication

```bash
python3 << 'EOF'
import os
import json
from google.oauth2 import service_account
from google.auth.transport.requests import Request

creds_path = os.getenv('DEFAULT_GOOGLE_APPLICATION_CREDENTIALS')
project = os.getenv('DEFAULT_VERTEXAI_PROJECT')

try:
    # Load credentials
    credentials = service_account.Credentials.from_service_account_file(
        creds_path,
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )
    
    # Get access token
    credentials.refresh(Request())
    
    print("✅ Authentication successful!")
    print(f"   Project: {project}")
    print(f"   Service Account: {credentials.service_account_email}")
    print(f"   Token expiry: {credentials.expiry}")
    
except Exception as e:
    print(f"❌ Authentication failed: {e}")
EOF
```

## Using with Vector Store Passthrough

Once your environment is set up, the vector store passthrough will work in two ways:

### 1. **With Vector Store Config** (Priority 1)
If you have a vector store configured with its own credentials in `litellm_params`, those will be used first:

```yaml
vector_stores:
  - vector_store_id: test-store-123
    custom_llm_provider: vertex_ai
    litellm_params:
      vertex_project: "specific-project"
      vertex_location: "us-central1"
      vertex_credentials: "{...}"  # Inline credentials
```

### 2. **Environment Variables Fallback** (Priority 2)
If the vector store doesn't have explicit credentials, it falls back to your environment variables:

```yaml
vector_stores:
  - vector_store_id: test-store-123
    custom_llm_provider: vertex_ai
    # No litellm_params - will use DEFAULT_VERTEXAI_PROJECT, DEFAULT_VERTEXAI_LOCATION, etc.
```

### 3. **Model Config Fallback** (Priority 3)
If neither above work, it looks for credentials in your model configuration.

## Troubleshooting

### "No credentials found"

Check that all environment variables are set:
```bash
env | grep -E "(DEFAULT_VERTEXAI|GOOGLE_APPLICATION_CREDENTIALS)"
```

### "Authentication failed"

Verify your service account key is valid:
```bash
cat $DEFAULT_GOOGLE_APPLICATION_CREDENTIALS | python3 -m json.tool
```

### "Permission denied"

Ensure your service account has the necessary roles:
```bash
gcloud projects get-iam-policy YOUR_PROJECT_ID \
    --flatten="bindings[].members" \
    --filter="bindings.members:serviceAccount:litellm-vertex-sa@*"
```

### Different Credentials for Different Projects

If you need to use different credentials for different vector stores, configure them explicitly in the vector store config rather than relying on environment variables.

## Start LiteLLM Proxy

Once your environment is configured:

```bash
# Start the proxy (it will automatically load env vars)
litellm --config proxy_server_config.yaml

# Or with debug logging
export LITELLM_LOG=DEBUG
litellm --config proxy_server_config.yaml
```

You should see logs like:
```
Vertex: Loading vertex credentials from /path/to/service-account.json
Found credentials for vertex_ai_default
```

## Test the Endpoint

```bash
curl -X POST http://0.0.0.0:4000/vertex_ai/discovery/v1/projects/fake-project/locations/global/dataStores/test-store-123/servingConfigs/default_config:search \
  -H 'Authorization: Bearer YOUR_LITELLM_API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{"query": "test query"}'
```

The proxy will use your environment credentials to make the request to Vertex AI!

