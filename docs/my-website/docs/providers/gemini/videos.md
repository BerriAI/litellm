import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Gemini Video Generation (Veo)

LiteLLM supports Google's Veo video generation models through a unified API interface.

| Property | Details |
|-------|-------|
| Description | Google's Veo AI video generation models |
| Provider Route on LiteLLM | `gemini/` |
| Supported Models | `veo-3.0-generate-preview`, `veo-3.1-generate-preview` |
| Cost Tracking | ✅ Duration-based pricing |
| Logging Support | ✅ Full request/response logging |
| Proxy Server Support | ✅ Full proxy integration with virtual keys |
| Spend Management | ✅ Budget tracking and rate limiting |
| Link to Provider Doc | [Google Veo Documentation ↗](https://ai.google.dev/gemini-api/docs/video) |

## Quick Start

### Required API Keys

```python
import os 
os.environ["GEMINI_API_KEY"] = "your-google-api-key"
# OR
os.environ["GOOGLE_API_KEY"] = "your-google-api-key"
```

### Basic Usage

```python
from litellm import video_generation, video_status, video_content
import os
import time

os.environ["GEMINI_API_KEY"] = "your-google-api-key"

# Step 1: Generate video
response = video_generation(
    model="gemini/veo-3.0-generate-preview",
    prompt="A cat playing with a ball of yarn in a sunny garden"
)

print(f"Video ID: {response.id}")
print(f"Initial Status: {response.status}")  # "processing"

# Step 2: Poll for completion
while True:
    status_response = video_status(
        video_id=response.id
    )
    
    print(f"Current Status: {status_response.status}")
    
    if status_response.status == "completed":
        break
    elif status_response.status == "failed":
        print("Video generation failed")
        break
    
    time.sleep(10)  # Wait 10 seconds before checking again

# Step 3: Download video content
video_bytes = video_content(
    video_id=response.id
)

# Save to file
with open("generated_video.mp4", "wb") as f:
    f.write(video_bytes)

print("Video downloaded successfully!")
```

## Supported Models

| Model Name | Description | Max Duration | Status |
|------------|-------------|--------------|--------|
| veo-3.0-generate-preview | Veo 3.0 video generation | 8 seconds | Preview |
| veo-3.1-generate-preview | Veo 3.1 video generation | 8 seconds | Preview |

## Video Generation Parameters

LiteLLM automatically maps OpenAI-style parameters to Veo's format:

| OpenAI Parameter | Veo Parameter | Description | Example |
|------------------|---------------|-------------|---------|
| `prompt` | `prompt` | Text description of the video | "A cat playing" |
| `size` | `aspectRatio` | Video dimensions → aspect ratio | "1280x720" → "16:9" |
| `seconds` | `durationSeconds` | Duration in seconds | "8" → 8 |
| `input_reference` | `image` | Reference image to animate | File object or path |
| `model` | `model` | Model to use | "gemini/veo-3.0-generate-preview" |

### Size to Aspect Ratio Mapping

LiteLLM automatically converts size dimensions to Veo's aspect ratio format:
- `"1280x720"`, `"1920x1080"` → `"16:9"` (landscape)
- `"720x1280"`, `"1080x1920"` → `"9:16"` (portrait)

### Supported Veo Parameters

Based on Veo's API:
- **prompt** (required): Text description with optional audio cues
- **aspectRatio**: `"16:9"` (default) or `"9:16"`
- **resolution**: `"720p"` (default) or `"1080p"` (Veo 3.1 only, 16:9 aspect ratio only)
- **durationSeconds**: Video length (max 8 seconds for most models)
- **image**: Reference image for animation
- **negativePrompt**: What to exclude from the video (Veo 3.1)
- **referenceImages**: Style and content references (Veo 3.1 only)

## Complete Workflow Example

```python
import litellm
import time

def generate_and_download_veo_video(
    prompt: str, 
    output_file: str = "video.mp4",
    size: str = "1280x720",
    seconds: str = "8"
):
    """
    Complete workflow for Veo video generation.
    
    Args:
        prompt: Text description of the video
        output_file: Where to save the video
        size: Video dimensions (e.g., "1280x720" for 16:9)
        seconds: Duration in seconds
        
    Returns:
        bool: True if successful
    """
    print(f"🎬 Generating video: {prompt}")
    
    # Step 1: Initiate generation
    response = litellm.video_generation(
        model="gemini/veo-3.0-generate-preview",
        prompt=prompt,
        size=size,      # Maps to aspectRatio
        seconds=seconds  # Maps to durationSeconds
    )
    
    video_id = response.id
    print(f"✓ Video generation started (ID: {video_id})")
    
    # Step 2: Wait for completion
    max_wait_time = 600  # 10 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        status_response = litellm.video_status(video_id=video_id)
        
        if status_response.status == "completed":
            print("✓ Video generation completed!")
            break
        elif status_response.status == "failed":
            print("✗ Video generation failed")
            return False
        
        print(f"⏳ Status: {status_response.status}")
        time.sleep(10)
    else:
        print("✗ Timeout waiting for video generation")
        return False
    
    # Step 3: Download video
    print("⬇️  Downloading video...")
    video_bytes = litellm.video_content(video_id=video_id)
    
    with open(output_file, "wb") as f:
        f.write(video_bytes)
    
    print(f"✓ Video saved to {output_file}")
    return True

# Use it
generate_and_download_veo_video(
    prompt="A serene lake at sunset with mountains in the background",
    output_file="sunset_lake.mp4"
)
```

## Async Usage

```python
from litellm import avideo_generation, avideo_status, avideo_content
import asyncio

async def async_video_workflow():
    # Generate video
    response = await avideo_generation(
        model="gemini/veo-3.0-generate-preview",
        prompt="A cat playing with a ball of yarn"
    )
    
    # Poll for completion
    while True:
        status = await avideo_status(video_id=response.id)
        if status.status == "completed":
            break
        await asyncio.sleep(10)
    
    # Download content
    video_bytes = await avideo_content(video_id=response.id)
    
    with open("video.mp4", "wb") as f:
        f.write(video_bytes)

# Run it
asyncio.run(async_video_workflow())
```

## LiteLLM Proxy Usage

### Configuration

Add Veo models to your `config.yaml`:

```yaml
model_list:
  - model_name: veo-3
    litellm_params:
      model: gemini/veo-3.0-generate-preview
      api_key: os.environ/GEMINI_API_KEY
```

Start the proxy:

```bash
litellm --config config.yaml
# Server running on http://0.0.0.0:4000
```

### Making Requests

<Tabs>
<TabItem value="curl" label="Curl">

```bash
# Step 1: Generate video
curl --location 'http://0.0.0.0:4000/v1/videos' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer sk-1234' \
--data '{
    "model": "veo-3",
    "prompt": "A cat playing with a ball of yarn in a sunny garden"
}'

# Response: {"id": "gemini::operations/generate_12345::...", "status": "processing", ...}

# Step 2: Check status
curl --location 'http://localhost:4000/v1/videos/{video_id}' \
--header 'x-litellm-api-key: sk-1234'

# Step 3: Download video (when status is "completed")
curl --location 'http://localhost:4000/v1/videos/{video_id}/content' \
--header 'x-litellm-api-key: sk-1234' \
--output video.mp4
```

</TabItem>
<TabItem value="python" label="Python SDK">

```python
import litellm

litellm.api_base = "http://0.0.0.0:4000"
litellm.api_key = "sk-1234"

# Generate video
response = litellm.video_generation(
    model="veo-3",
    prompt="A cat playing with a ball of yarn in a sunny garden"
)

# Check status
import time
while True:
    status = litellm.video_status(video_id=response.id)
    if status.status == "completed":
        break
    time.sleep(10)

# Download video
video_bytes = litellm.video_content(video_id=response.id)
with open("video.mp4", "wb") as f:
    f.write(video_bytes)
```

</TabItem>
</Tabs>

## Cost Tracking

LiteLLM automatically tracks costs for Veo video generation:

```python
response = litellm.video_generation(
    model="gemini/veo-3.0-generate-preview",
    prompt="A beautiful sunset"
)

# Cost is calculated based on video duration
# Veo pricing: ~$0.10 per second (estimated)
# Default video duration: ~5 seconds
# Estimated cost: ~$0.50
```

## Differences from OpenAI Video API

| Feature | OpenAI (Sora) | Gemini (Veo) |
|---------|---------------|--------------|
| Reference Images | ✅ Supported | ❌ Not supported |
| Size Control | ✅ Supported | ❌ Not supported |
| Duration Control | ✅ Supported | ❌ Not supported |
| Video Remix/Edit | ✅ Supported | ❌ Not supported |
| Video List | ✅ Supported | ❌ Not supported |
| Prompt-based Generation | ✅ Supported | ✅ Supported |
| Async Operations | ✅ Supported | ✅ Supported |

## Error Handling

```python
from litellm import video_generation, video_status, video_content
from litellm.exceptions import APIError, Timeout

try:
    response = video_generation(
        model="gemini/veo-3.0-generate-preview",
        prompt="A beautiful landscape"
    )
    
    # Poll with timeout
    max_attempts = 60  # 10 minutes (60 * 10s)
    for attempt in range(max_attempts):
        status = video_status(video_id=response.id)
        
        if status.status == "completed":
            video_bytes = video_content(video_id=response.id)
            with open("video.mp4", "wb") as f:
                f.write(video_bytes)
            break
        elif status.status == "failed":
            raise APIError("Video generation failed")
        
        time.sleep(10)
    else:
        raise Timeout("Video generation timed out")
        
except APIError as e:
    print(f"API Error: {e}")
except Timeout as e:
    print(f"Timeout: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Best Practices

1. **Always poll for completion**: Veo video generation is asynchronous and can take several minutes
2. **Set reasonable timeouts**: Allow at least 5-10 minutes for video generation
3. **Handle failures gracefully**: Check for `failed` status and implement retry logic
4. **Use descriptive prompts**: More detailed prompts generally produce better results
5. **Store video IDs**: Save the operation ID/video ID to resume polling if your application restarts

## Troubleshooting

### Video generation times out

```python
# Increase polling timeout
max_wait_time = 900  # 15 minutes instead of 10
```

### Video not found when downloading

```python
# Make sure video is completed before downloading
status = video_status(video_id=video_id)
if status.status != "completed":
    print("Video not ready yet!")
```

### API key errors

```python
# Verify your API key is set
import os
print(os.environ.get("GEMINI_API_KEY"))

# Or pass it explicitly
response = video_generation(
    model="gemini/veo-3.0-generate-preview",
    prompt="...",
    api_key="your-api-key-here"
)
```

## See Also

- [OpenAI Video Generation](../openai/videos.md)
- [Azure Video Generation](../azure/videos.md)
- [Vertex AI Video Generation](../vertex_ai/videos.md)
- [Video Generation API Reference](/docs/videos)
- [Veo Pass-through Endpoints](/docs/pass_through/google_ai_studio#example-4-video-generation-with-veo)

