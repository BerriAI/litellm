import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# OpenAI Video Generation

LiteLLM supports OpenAI's video generation models including Sora.

## Quick Start

### Required API Keys

```python
import os 
os.environ["OPENAI_API_KEY"] = "your-api-key"
```

### Basic Usage

```python
from litellm import video_generation, video_content
import os

os.environ["OPENAI_API_KEY"] = "your-api-key"

# Generate a video
response = video_generation(
    prompt="A cat playing with a ball of yarn in a sunny garden",
    model="sora-2",
    seconds="8",
    size="720x1280"
)

print(f"Video ID: {response.id}")
print(f"Status: {response.status}")

# Download video content when ready
video_bytes = video_content(
    video_id=response.id,
)

# Save to file
with open("generated_video.mp4", "wb") as f:
    f.write(video_bytes)
```

## **LiteLLM Proxy Usage**

LiteLLM provides OpenAI API compatible video endpoints for complete video generation workflow:

- `/videos/generations` - Generate new videos
- `/videos/remix` - Edit existing videos with reference images  
- `/videos/status` - Check video generation status
- `/videos/retrieval` - Download completed videos

**Setup**

Add this to your litellm proxy config.yaml

```yaml
model_list:
  - model_name: sora-2
    litellm_params:
      model: openai/sora-2
      api_key: os.environ/OPENAI_API_KEY
```

Start litellm

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

Test video generation request

```bash
curl --location 'http://localhost:4000/v1/videos' \
--header 'Content-Type: application/json' \
--header 'x-litellm-api-key: sk-1234' \
--data '{
    "model": "sora-2",
    "prompt": "A beautiful sunset over the ocean"
}'
```

Test video status request

```bash
# Using custom-llm-provider header
curl --location 'http://localhost:4000/v1/videos/video_id' \
--header 'Accept: application/json' \
--header 'x-litellm-api-key: sk-1234' \
--header 'custom-llm-provider: openai'
```

Test video retrieval request

```bash
# Using custom-llm-provider header
curl --location 'http://localhost:4000/v1/videos/video_id/content' \
--header 'Accept: application/json' \
--header 'x-litellm-api-key: sk-1234' \
--header 'custom-llm-provider: openai' \
--output video.mp4

# Or using query parameter
curl --location 'http://localhost:4000/v1/videos/video_id/content?custom_llm_provider=openai' \
--header 'Accept: application/json' \
--header 'x-litellm-api-key: sk-1234' \
--output video.mp4
```

Test video remix request

```bash
# Using custom_llm_provider in request body
curl --location --request POST 'http://localhost:4000/v1/videos/video_id/remix' \
--header 'Accept: application/json' \
--header 'Content-Type: application/json' \
--header 'x-litellm-api-key: sk-1234' \
--data '{
    "prompt": "New remix instructions",
    "custom_llm_provider": "openai"
}'

# Or using custom-llm-provider header
curl --location --request POST 'http://localhost:4000/v1/videos/video_id/remix' \
--header 'Accept: application/json' \
--header 'Content-Type: application/json' \
--header 'x-litellm-api-key: sk-1234' \
--header 'custom-llm-provider: openai' \
--data '{
    "prompt": "New remix instructions"
}'
```

Test OpenAI video generation request

```bash
curl http://localhost:4000/v1/videos \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sora-2",
    "prompt": "A cat playing with a ball of yarn in a sunny garden",
    "seconds": "8",
    "size": "720x1280"
  }'
```


## Supported Models

| Model Name | Description | Max Duration | Supported Sizes |
|------------|-------------|--------------|-----------------|
| sora-2 | OpenAI's latest video generation model | 8 seconds | 720x1280, 1280x720 |

## Video Generation Parameters

- `prompt` (required): Text description of the desired video
- `model` (optional): Model to use, defaults to "sora-2"
- `seconds` (optional): Video duration in seconds (e.g., "8", "16")
- `size` (optional): Video dimensions (e.g., "720x1280", "1280x720")
- `input_reference` (optional): Reference image for video editing
- `user` (optional): User identifier for tracking

## Video Content Retrieval

```python
# Download video content
video_bytes = video_content(
    video_id="video_1234567890"
)

# Save to file
with open("video.mp4", "wb") as f:
    f.write(video_bytes)
```

## Complete Workflow

```python
import litellm
import time

def generate_and_download_video(prompt):
    # Step 1: Generate video
    response = litellm.video_generation(
        prompt=prompt,
        model="sora-2",
        seconds="8",
        size="720x1280"
    )
    
    video_id = response.id
    print(f"Video ID: {video_id}")
    
    # Step 2: Wait for processing (in practice, poll status)
    time.sleep(30)
    
    # Step 3: Download video
    video_bytes = litellm.video_content(
        video_id=video_id
    )
    
    # Step 4: Save to file
    with open(f"video_{video_id}.mp4", "wb") as f:
        f.write(video_bytes)
    
    return f"video_{video_id}.mp4"

# Usage
video_file = generate_and_download_video(
    "A cat playing with a ball of yarn in a sunny garden"
)
```


## Video Editing with Reference Images

```python
# Video editing with reference image
response = litellm.video_generation(
    prompt="Make the cat jump higher",
    input_reference=open("path/to/image.jpg", "rb"),  # Reference image
    model="sora-2",
    seconds="8"
)

print(f"Video ID: {response.id}")
```

## Error Handling

```python
from litellm.exceptions import BadRequestError, AuthenticationError

try:
    response = video_generation(
        prompt="A cat playing with a ball of yarn"
    )
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
except BadRequestError as e:
    print(f"Bad request: {e}")
```
