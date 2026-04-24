# PixVerse - Video Generation

LiteLLM supports PixVerse's video generation API, allowing you to generate videos from text prompts, images, and reference videos.

## Quick Start

```python showLineNumbers title="Basic Video Generation"
from litellm import video_generation
import os

os.environ["PIXVERSE_API_KEY"] = "your-api-key"

# Text-to-Video Generation
response = video_generation(
    model="pixverse",
    prompt="A serene sunset over a calm ocean with gentle waves",
    seconds="5",
    size="1280x720",
    custom_llm_provider="pixverse"
)

print(f"Video ID: {response.id}")
print(f"Status: {response.status}")
```

## Authentication

Set your PixVerse API key:

```python showLineNumbers title="Set API Key"
import os

os.environ["PIXVERSE_API_KEY"] = "your-api-key"
```

## Supported Generation Modes

PixVerse supports three video generation modes:

### 1. Text-to-Video

Generate videos from text prompts only:

```python showLineNumbers title="Text-to-Video"
from litellm import video_generation

response = video_generation(
    model="pixverse",
    prompt="A futuristic cityscape at night with neon lights",
    seconds="5",
    size="1920x1080",
    custom_llm_provider="pixverse"
)

print(f"Video ID: {response.id}")
```

### 2. Image-to-Video

Generate videos from an image and text prompt:

```python showLineNumbers title="Image-to-Video"
from litellm import video_generation

response = video_generation(
    model="pixverse",
    prompt="Animate this scene with gentle movement",
    input_reference="https://example.com/image.jpg",
    seconds="5",
    size="1920x1080",
    custom_llm_provider="pixverse"
)

print(f"Video ID: {response.id}")
```

### 3. Video-to-Video (Fusion)

Transform videos using a reference video and text prompt:

```python showLineNumbers title="Video-to-Video (Fusion)"
from litellm import video_generation

response = video_generation(
    model="pixverse",
    prompt="Transform this into a watercolor painting style",
    input_reference="https://example.com/reference.mp4",
    seconds="10",
    size="1920x1080",
    custom_llm_provider="pixverse"
)

print(f"Video ID: {response.id}")
```

## Supported Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model to use (use `"pixverse"`) |
| `prompt` | string | Yes | Text description for the video |
| `input_reference` | string/file | No | URL or file path to reference image or video |
| `seconds` | string | No | Video duration (e.g., "5", "10") |
| `size` | string | No | Video dimensions (e.g., "1280x720", "1920x1080") |
| `custom_llm_provider` | string | Yes | Must be set to `"pixverse"` |

## Complete Workflow

```python showLineNumbers title="Complete Video Generation Workflow"
from litellm import video_generation, video_status, video_content
import os
import time

os.environ["PIXVERSE_API_KEY"] = "your-api-key"

# 1. Generate video
response = video_generation(
    model="pixverse",
    prompt="A serene sunset over a calm ocean with gentle waves",
    seconds="5",
    size="1280x720",
    custom_llm_provider="pixverse"
)

video_id = response.id
print(f"Video generation started: {video_id}")

# 2. Check status until completed
while True:
    status_response = video_status(
        video_id=video_id,
        custom_llm_provider="pixverse"
    )
    print(f"Status: {status_response.status}")

    if status_response.status == "completed":
        print("Video generation completed!")
        break
    elif status_response.status == "failed":
        print("Video generation failed")
        if hasattr(status_response, 'error'):
            print(f"Error: {status_response.error}")
        break

    time.sleep(10)  # Wait 10 seconds before checking again

# 3. Download video content
if status_response.status == "completed":
    video_bytes = video_content(
        video_id=video_id,
        custom_llm_provider="pixverse"
    )

    # 4. Save to file
    with open("generated_video.mp4", "wb") as f:
        f.write(video_bytes)

    print("Video saved successfully!")
```

## Async Usage

```python showLineNumbers title="Async Video Generation"
from litellm import avideo_generation, avideo_status, avideo_content
import os
import asyncio

os.environ["PIXVERSE_API_KEY"] = "your-api-key"

async def generate_video():
    # Generate video
    response = await avideo_generation(
        model="pixverse",
        prompt="A serene lake with mountains in the background",
        seconds="5",
        size="1280x720",
        custom_llm_provider="pixverse"
    )

    video_id = response.id
    print(f"Video generation started: {video_id}")

    # Poll for completion
    while True:
        status_response = await avideo_status(
            video_id=video_id,
            custom_llm_provider="pixverse"
        )
        print(f"Status: {status_response.status}")

        if status_response.status == "completed":
            break
        elif status_response.status == "failed":
            print("Video generation failed")
            return

        await asyncio.sleep(10)

    # Download video
    video_bytes = await avideo_content(
        video_id=video_id,
        custom_llm_provider="pixverse"
    )

    # Save to file
    with open("generated_video.mp4", "wb") as f:
        f.write(video_bytes)

    print("Video saved successfully!")

asyncio.run(generate_video())
```

## LiteLLM Proxy Usage

Add PixVerse to your proxy configuration:

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: pixverse
    litellm_params:
      model: pixverse
      custom_llm_provider: pixverse
      api_key: os.environ/PIXVERSE_API_KEY
```

Start the proxy:

```bash
litellm --config /path/to/config.yaml
```

Generate videos through the proxy:

```bash showLineNumbers title="Text-to-Video Request"
curl --location 'http://localhost:4000/v1/videos' \
--header 'Content-Type: application/json' \
--header 'x-litellm-api-key: sk-1234' \
--data '{
    "model": "pixverse",
    "prompt": "A serene sunset over a calm ocean",
    "seconds": "5",
    "size": "1280x720"
}'
```

```bash showLineNumbers title="Image-to-Video Request"
curl --location 'http://localhost:4000/v1/videos' \
--header 'Content-Type: application/json' \
--header 'x-litellm-api-key: sk-1234' \
--data '{
    "model": "pixverse",
    "prompt": "Animate this scene with gentle movement",
    "input_reference": "https://example.com/image.jpg",
    "seconds": "5",
    "size": "1920x1080"
}'
```

Check video status:

```bash showLineNumbers title="Check Status"
curl --location 'http://localhost:4000/v1/videos/{video_id}' \
--header 'x-litellm-api-key: sk-1234' \
--header 'custom-llm-provider: pixverse'
```

Download video content:

```bash showLineNumbers title="Download Video"
curl --location 'http://localhost:4000/v1/videos/{video_id}/content' \
--header 'x-litellm-api-key: sk-1234' \
--header 'custom-llm-provider: pixverse' \
--output video.mp4
```

## Error Handling

```python showLineNumbers title="Error Handling"
from litellm import video_generation, video_status
import time

try:
    response = video_generation(
        model="pixverse",
        prompt="A scenic mountain view",
        seconds="5",
        custom_llm_provider="pixverse"
    )

    # Poll for completion
    max_attempts = 60  # 10 minutes max
    attempts = 0

    while attempts < max_attempts:
        status_response = video_status(
            video_id=response.id,
            custom_llm_provider="pixverse"
        )

        if status_response.status == "completed":
            print("Video generation completed!")
            break
        elif status_response.status == "failed":
            error = getattr(status_response, 'error', {})
            if isinstance(error, dict):
                print(f"Video generation failed: {error.get('message', 'Unknown error')}")
            else:
                print(f"Video generation failed: {error}")
            break

        time.sleep(10)
        attempts += 1

    if attempts >= max_attempts:
        print("Video generation timed out")

except Exception as e:
    print(f"Error: {str(e)}")
```

## Cost Tracking

LiteLLM automatically tracks PixVerse video generation costs:

```python showLineNumbers title="Cost Tracking"
from litellm import video_generation

response = video_generation(
    model="pixverse",
    prompt="A serene sunset over a calm ocean",
    seconds="5",
    size="1280x720",
    custom_llm_provider="pixverse"
)

# Usage information is included in the response
if hasattr(response, 'usage'):
    print(f"Usage: {response.usage}")
```

## API Reference

PixVerse API documentation: [https://docs.platform.pixverse.ai/](https://docs.platform.pixverse.ai/)

## Supported Features

| Feature | Supported |
|---------|-----------|
| Text-to-Video | ✅ |
| Image-to-Video | ✅ |
| Video-to-Video (Fusion) | ✅ |
| Status Checking | ✅ |
| Content Download | ✅ |
| Task Cancellation | ✅ |
| Video Remix | ❌ (Not supported by PixVerse API) |
| Video List | ❌ (Not supported by PixVerse API) |
| Cost Tracking | ✅ |
| Logging | ✅ |
| Fallbacks | ✅ |
| Load Balancing | ✅ |

## Notes

- The `custom_llm_provider="pixverse"` parameter is required for all PixVerse video operations
- PixVerse automatically detects the generation mode based on the `input_reference` parameter:
  - No `input_reference`: Text-to-Video
  - Image `input_reference`: Image-to-Video
  - Video `input_reference`: Video-to-Video (Fusion)
- Video generation is asynchronous - always check the status before downloading
