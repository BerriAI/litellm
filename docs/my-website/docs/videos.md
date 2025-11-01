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
from litellm import video_generation, video_status, video_content
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
        custom_llm_provider="openai"
    )
    
    print(f"Current Status: {status_response.status}")
    
    if status_response.status == "completed":
        break
    elif status_response.status == "failed":
        print("Video generation failed")
        break
    
    time.sleep(10)  # Wait 10 seconds before checking again

# Download video content when ready
video_bytes = video_content(
    video_id=response.id,
    custom_llm_provider="openai"
)

# Save to file
with open("generated_video.mp4", "wb") as f:
    f.write(video_bytes)
```

### Async Usage 

```python
from litellm import avideo_generation, avideo_status, avideo_content
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
            custom_llm_provider="openai"
        )
        
        print(f"Current Status: {status_response.status}")
        
        if status_response.status == "completed":
            break
        elif status_response.status == "failed":
            print("Video generation failed")
            break
        
        await asyncio.sleep(10)  # Wait 10 seconds before checking again
    
    # Download video content when ready
    video_bytes = await avideo_content(
        video_id=response.id,
        custom_llm_provider="openai"
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
    custom_llm_provider="openai"
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
--header 'custom-llm-provider: azure'

# Or using query parameter
curl --location 'http://localhost:4000/v1/videos/video_id?custom_llm_provider=azure' \
--header 'Accept: application/json' \
--header 'x-litellm-api-key: sk-1234'
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
    "custom_llm_provider": "azure"
}'

# Or using custom-llm-provider header
curl --location --request POST 'http://localhost:4000/v1/videos/video_id/remix' \
--header 'Accept: application/json' \
--header 'Content-Type: application/json' \
--header 'x-litellm-api-key: sk-1234' \
--header 'custom-llm-provider: azure' \
--data '{
    "prompt": "New remix instructions"
}'
```

Test Azure video generation request

```bash
curl http://localhost:4000/v1/videos \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "azure-sora-2",
    "prompt": "A cat playing with a ball of yarn in a sunny garden",
    "seconds": "8",
    "size": "720x1280"
  }'
```

## **Using OpenAI Client with LiteLLM Proxy**

You can use the standard OpenAI Python client to interact with LiteLLM's video endpoints. This provides a familiar interface while leveraging LiteLLM's provider abstraction and proxy features.

### Setup

First, configure your OpenAI client to point to your LiteLLM proxy:

```python
from openai import OpenAI

# Point the OpenAI client to your LiteLLM proxy
client = OpenAI(
    api_key="sk-1234",  # Your LiteLLM proxy API key
    base_url="http://localhost:4000/v1"  # Your LiteLLM proxy URL
)
```

### Video Generation

Generate a new video using the OpenAI client interface:

```python
# Basic video generation
response = client.videos.create(
    model="sora-2",
    prompt="A cat playing with a ball of yarn in a sunny garden",
    seconds=8,
    size="720x1280"
)

print(f"Video ID: {response.id}")
print(f"Status: {response.status}")
```

### Video Generation with Reference Image

Create a video using a reference image:

```python
# Video generation with reference image
response = client.videos.create(
    model="sora-2",
    prompt="Add clouds to the video",
    seconds=4,
    input_reference=open("/path/to/your/image.jpg", "rb")
)

print(f"Video ID: {response.id}")
print(f"Status: {response.status}")
```

### Video Status Checking

Check the status of a video generation:

```python
# Check video status
status_response = client.videos.retrieve(
    video_id="video_6900378779308191a7359266e59b53fc01cd6bbd27a70763"
)

print(f"Status: {status_response.status}")
print(f"Progress: {status_response.progress}%")

# Poll until completion
import time

while status_response.status not in ["completed", "failed"]:
    time.sleep(10)  # Wait 10 seconds
    status_response = client.videos.retrieve(
        video_id="video_6900378779308191a7359266e59b53fc01cd6bbd27a70763"
    )
    print(f"Current status: {status_response.status}")
```

### List Videos

Get a list of your videos:

```python
# List all videos
videos = client.videos.list()

for video in videos.data:
    print(f"Video ID: {video.id}, Status: {video.status}")
```

### Download Video Content

Download the completed video:

```python
# Download video content
response = client.videos.download_content(
    video_id="video_68fa2938848c8190bb718f977503aba6092ab18d68938fed"
)

# Save the video to file
with open("generated_video.mp4", "wb") as f:
    f.write(response.content)

print("Video downloaded successfully!")
```

### Video Remix (Editing)

Edit an existing video with new instructions:

```python
# Remix/edit an existing video
response = client.videos.remix(
    video_id="video_68fa2574bdd88190873a8af06a370ff407094ddbc4bbb91b",
    prompt="Slow the cloud movement",
    seconds=8
)

print(f"Remix Video ID: {response.id}")
print(f"Status: {response.status}")
```

### Complete Workflow Example

Here's a complete example showing the full video generation workflow:

```python
from openai import OpenAI
import time

# Initialize client
client = OpenAI(
    api_key="sk-1234",
    base_url="http://localhost:4000/v1"
)

# 1. Generate video
print("Generating video...")
response = client.videos.create(
    model="sora-2",
    prompt="A serene lake with mountains in the background",
    seconds=8,
    size="1280x720"
)

video_id = response.id
print(f"Video generation started. ID: {video_id}")

# 2. Poll for completion
print("Waiting for video to complete...")
while True:
    status = client.videos.retrieve(video_id=video_id)
    print(f"Status: {status.status}")
    
    if status.status == "completed":
        print("Video generation completed!")
        break
    elif status.status == "failed":
        print("Video generation failed!")
        break
    
    time.sleep(10)

# 3. Download video
if status.status == "completed":
    print("Downloading video...")
    video_content = client.videos.download_content(video_id=video_id)
    
    with open(f"video_{video_id}.mp4", "wb") as f:
        f.write(video_content.content)
    
    print("Video saved successfully!")

# 4. Optional: Remix the video
print("Creating a remix...")
remix_response = client.videos.remix(
    video_id=video_id,
    prompt="Add gentle ripples to the lake surface"
)

print(f"Remix started. ID: {remix_response.id}")
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
    "id": "video_6900378779308191a7359266e59b53fc01cd6bbd27a70763",
    "object": "video",
    "status": "queued",
    "created_at": 1761621895,
    "completed_at": null,
    "expires_at": null,
    "error": null,
    "progress": 0,
    "remixed_from_video_id": null,
    "seconds": "4",
    "size": "720x1280",
    "model": "sora-2",
    "usage": {
        "duration_seconds": 4.0
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