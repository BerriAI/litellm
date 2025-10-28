# /videos

| Feature | Supported | 
|---------|-----------|
| Cost Tracking | ✅ |
| Logging | ✅ (Full request/response logging) |
Fallbacks | ✅ (Between supported models) |
| Load Balancing | ✅ |
| Guardrails Support | ✅ Content moderation and safety checks |
| Proxy Server Support | ✅ Full proxy integration with virtual keys |
| Spend Management | ✅ Budget tracking and rate limiting |
| Supported Providers | `openai`, `azure` |

:::tip

LiteLLM follows the [OpenAI Video Generation API specification](https://platform.openai.com/docs/guides/video-generation)

:::

## **LiteLLM Python SDK Usage**
### Quick Start 

```python
from litellm import video_generation, video_status, video_retrieval
import os
import time

os.environ["OPENAI_API_KEY"] = "sk-.."

# Generate video
response = video_generation(
    model="openai/sora-2",
    prompt="A cat playing with a ball of yarn in a sunny garden",
    seconds="8",
    size="720x1280"
)

print(f"Video ID: {response.id}")
print(f"Initial Status: {response.status}")

# Check status until video is ready
while True:
    status_response = video_status(
        video_id=response.id,
        model="openai/sora-2"
    )
    
    print(f"Current Status: {status_response.status}")
    
    if status_response.status == "completed":
        break
    elif status_response.status == "failed":
        print("Video generation failed")
        break
    
    time.sleep(10)  # Wait 10 seconds before checking again

# Download video content when ready
video_bytes = video_retrieval(
    video_id=response.id,
    model="openai/sora-2"
)

# Save to file
with open("generated_video.mp4", "wb") as f:
    f.write(video_bytes)
```

### Async Usage 

```python
from litellm import avideo_generation, avideo_status, avideo_retrieval
import os, asyncio

os.environ["OPENAI_API_KEY"] = "sk-.."

async def test_async_video(): 
    response = await avideo_generation(
        model="openai/sora-2",
        prompt="A cat playing with a ball of yarn in a sunny garden",
        seconds="8",
        size="720x1280"
    )
    
    print(f"Video ID: {response.id}")
    print(f"Initial Status: {response.status}")
    
    # Check status until video is ready
    while True:
        status_response = await avideo_status(
            video_id=response.id,
            model="openai/sora-2"
        )
        
        print(f"Current Status: {status_response.status}")
        
        if status_response.status == "completed":
            break
        elif status_response.status == "failed":
            print("Video generation failed")
            break
        
        await asyncio.sleep(10)  # Wait 10 seconds before checking again
    
    # Download video content when ready
    video_bytes = await avideo_retrieval(
        video_id=response.id,
        model="openai/sora-2"
    )
    
    # Save to file
    with open("generated_video.mp4", "wb") as f:
        f.write(video_bytes)

asyncio.run(test_async_video())
```

### Video Status Checking

```python
from litellm import video_status

# Check the status of a video generation
status_response = video_status(
    video_id="video_1234567890",
    model="openai/sora-2"
)

print(f"Video Status: {status_response.status}")
print(f"Created At: {status_response.created_at}")
print(f"Model: {status_response.model}")

# Possible status values:
# - "queued": Video is in the queue
# - "processing": Video is being generated
# - "completed": Video is ready for download
# - "failed": Video generation failed
```

### Video Generation with Reference Image

```python
from litellm import video_generation

# Video generation with reference image
response = video_generation(
    model="openai/sora-2",
    prompt="A cat playing with a ball of yarn in a sunny garden",
    input_reference=open("path/to/image.jpg", "rb"),  # Reference image as file object
    seconds="8",
    size="720x1280"
)

print(f"Video ID: {response.id}")
```

### Video Remix (Video Editing)

```python
from litellm import video_remix

# Video remix with reference image
response = video_remix(
    model="openai/sora-2",
    prompt="Make the cat jump higher",
    input_reference=open("path/to/image.jpg", "rb"),  # Reference image as file object
    seconds="8"
)

print(f"Video ID: {response.id}")
```

### Optional Parameters

```python
response = video_generation(
    model="openai/sora-2",
    prompt="A cat playing with a ball of yarn in a sunny garden",
    seconds="8",                    # Video duration in seconds
    size="720x1280",               # Video dimensions
    input_reference=open("path/to/image.jpg", "rb"),  # Reference image as file object
    user="user_123"                # User identifier for tracking
)
```

### Azure Video Generation

```python
from litellm import video_generation
import os

os.environ["AZURE_OPENAI_API_KEY"] = "your-azure-api-key"
os.environ["AZURE_OPENAI_API_BASE"] = "https://your-resource.openai.azure.com/"
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-15-preview"

response = video_generation(
    model="azure/sora-2",
    prompt="A cat playing with a ball of yarn in a sunny garden",
    seconds="8",
    size="720x1280"
)

print(f"Video ID: {response.id}")
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
  - model_name: azure-sora-2
    litellm_params:
      model: azure/sora-2
      api_key: os.environ/AZURE_OPENAI_API_KEY
      api_base: os.environ/AZURE_OPENAI_API_BASE
      api_version: "2024-02-15-preview"
```

Start litellm

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

Test video generation request

```bash
curl http://0.0.0.0:4000/videos/generations \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sora-2",
    "prompt": "A cat playing with a ball of yarn in a sunny garden",
    "seconds": "8",
    "size": "720x1280"
  }'
```

Test video status request

```bash
curl http://0.0.0.0:4000/videos/status \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "video_1234567890",
    "model": "sora-2"
  }'
```

Test video retrieval request

```bash
curl http://0.0.0.0:4000/videos/retrieval \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "video_1234567890",
    "model": "sora-2"
  }'
```

Test video remix request

```bash
curl http://0.0.0.0:4000/videos/remix \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: multipart/form-data" \
  -F 'model=sora-2' \
  -F 'prompt=Make the cat jump higher' \
  -F 'input_reference=@path/to/image.jpg' \
  -F 'seconds=8'
```

Test Azure video generation request

```bash
curl http://0.0.0.0:4000/videos/generations \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "azure-sora-2",
    "prompt": "A cat playing with a ball of yarn in a sunny garden",
    "seconds": "8",
    "size": "720x1280"
  }'
```

## **Request/Response Format**

:::info

LiteLLM follows the **OpenAI Video Generation API specification**. 

See the [official OpenAI Video Generation documentation](https://platform.openai.com/docs/guides/video-generation) for complete details.

:::

### Example Request

```python
{
    "model": "openai/sora-2",
    "prompt": "A cat playing with a ball of yarn in a sunny garden",
    "seconds": "8",
    "size": "720x1280",
    "user": "user_123"
}
```

### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | The video generation model to use (e.g., `"openai/sora-2"`) |
| `prompt` | string | Yes | Text description of the desired video |
| `seconds` | string | No | Video duration in seconds (e.g., "8", "16") |
| `size` | string | No | Video dimensions (e.g., "720x1280", "1280x720") |
| `input_reference` | file object | No | Reference image for video generation or editing (both generation and remix) |
| `user` | string | No | User identifier for tracking |
| `video_id` | string | Yes (status/retrieval) | Video ID for status checking or retrieval |

#### Video Generation Request Example

**For video generation:**
```json
{
  "model": "sora-2",
  "prompt": "A cat playing with a ball of yarn in a sunny garden",
  "seconds": "8",
  "size": "720x1280"
}
```

**For video generation with reference image:**
```python
{
  "model": "sora-2",
  "prompt": "A cat playing with a ball of yarn in a sunny garden",
  "input_reference": open("path/to/image.jpg", "rb"),  # File object
  "seconds": "8",
  "size": "720x1280"
}
```

**For video status check:**
```json
{
  "video_id": "video_1234567890",
  "model": "sora-2"
}
```

**For video retrieval:**
```json
{
  "video_id": "video_1234567890",
  "model": "sora-2"
}
```

### Response Format

The response follows OpenAI's video generation format with the following structure:

```json
{
  "id": "video_1234567890",
  "object": "video",
  "status": "queued",
  "created_at": 1712697600,
  "model": "sora-2",
  "size": "720x1280",
  "seconds": "8",
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "duration_seconds": 8.0
  }
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier for the video |
| `object` | string | Always `"video"` for video responses |
| `status` | string | Video processing status (`"queued"`, `"processing"`, `"completed"`) |
| `created_at` | integer | Unix timestamp when the video was created |
| `model` | string | The model used for video generation |
| `size` | string | Video dimensions |
| `seconds` | string | Video duration in seconds |
| `usage` | object | Token usage and duration information |


## **Supported Providers**

| Provider    | Link to Usage      |
|-------------|--------------------|
| OpenAI      |   [Usage](providers/openai/videos)  |
| Azure       |   [Usage](providers/azure/videos)   |