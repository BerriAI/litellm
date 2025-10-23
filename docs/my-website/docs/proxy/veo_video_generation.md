import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Veo Video Generation with Google AI Studio

Generate videos using Google's Veo model through LiteLLM's pass-through endpoints.

## Quick Start

LiteLLM allows you to use Google AI Studio's Veo video generation API through pass-through routes with zero configuration.

### 1. Add Google AI Studio API Key to your environment 

```bash
export GEMINI_API_KEY="your_google_ai_studio_api_key"
```

### 2. Start LiteLLM Proxy 

```bash
litellm

# RUNNING on http://0.0.0.0:4000
```

### 3. Generate Video

<Tabs>
<TabItem value="python" label="Python">

```python
import requests
import time
import json

# Configuration
BASE_URL = "http://localhost:4000/gemini/v1beta"
API_KEY = "anything"  # Use "anything" as the key

headers = {
    "x-goog-api-key": API_KEY,
    "Content-Type": "application/json"
}

# Step 1: Initiate video generation
def generate_video(prompt):
    url = f"{BASE_URL}/models/veo-3.0-generate-preview:predictLongRunning"
    payload = {
        "instances": [{
            "prompt": prompt
        }]
    }
    
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    
    data = response.json()
    return data.get("name")  # Operation name

# Step 2: Poll for completion
def wait_for_completion(operation_name):
    operation_url = f"{BASE_URL}/{operation_name}"
    
    while True:
        response = requests.get(operation_url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("done", False):
            # Extract video URI
            video_uri = data["response"]["generateVideoResponse"]["generatedSamples"][0]["video"]["uri"]
            return video_uri
        
        time.sleep(10)  # Wait 10 seconds before next poll

# Step 3: Download video
def download_video(video_uri, filename="generated_video.mp4"):
    # Replace Google URL with LiteLLM proxy URL
    litellm_url = video_uri.replace(
        "https://generativelanguage.googleapis.com/v1beta", 
        BASE_URL
    )
    
    response = requests.get(litellm_url, headers=headers, stream=True)
    response.raise_for_status()
    
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    
    return filename

# Complete workflow
prompt = "A cat playing with a ball of yarn in a sunny garden"

print("Generating video...")
operation_name = generate_video(prompt)

print("Waiting for completion...")
video_uri = wait_for_completion(operation_name)

print("Downloading video...")
filename = download_video(video_uri)

print(f"Video saved as: {filename}")
```

</TabItem>

<TabItem value="curl" label="Curl">

```bash
# Step 1: Initiate video generation
curl -X POST "http://localhost:4000/gemini/v1beta/models/veo-3.0-generate-preview:predictLongRunning" \
  -H "x-goog-api-key: anything" \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [{
      "prompt": "A cat playing with a ball of yarn in a sunny garden"
    }]
  }'

# Response will include operation name:
# {"name": "operations/generate_12345"}

# Step 2: Poll for completion
curl -X GET "http://localhost:4000/gemini/v1beta/operations/generate_12345" \
  -H "x-goog-api-key: anything"

# Step 3: Download video (when done=true)
curl -X GET "http://localhost:4000/gemini/v1beta/files/VIDEO_ID:download?alt=media" \
  -H "x-goog-api-key: anything" \
  --output generated_video.mp4
```

</TabItem>
</Tabs>

## Complete Example

For a full working example with error handling and logging, see our [Veo Video Generation Cookbook](https://github.com/BerriAI/litellm/blob/main/cookbook/veo_video_generation.py).

## How It Works

1. **Video Generation Request**: Send a prompt to Veo's `predictLongRunning` endpoint
2. **Operation Polling**: Monitor the long-running operation until completion
3. **File Download**: Download the generated video through LiteLLM's pass-through with automatic redirect handling

LiteLLM handles:
- ✅ Authentication with Google AI Studio
- ✅ Request routing and proxying
- ✅ Automatic redirect handling for file downloads

## Configuration Options

### Environment Variables

```bash
export GEMINI_API_KEY="your_google_ai_studio_api_key"
```

