# Video Generation

LiteLLM supports OpenAI's video generation API, enabling you to create videos using Sora-2 models through a unified interface.

## Supported Models

- **sora-2**: OpenAI's latest video generation model
- **sora-2-pro**: Enhanced version with higher quality and more resolution options

## Quick Start

### Basic Video Generation

```python
import litellm

# Create a video
response = litellm.acreate_video(
    prompt="A cat playing with yarn in a sunny garden",
    model="sora-2",
    seconds="4",
    size="720x1280"
)

print(f"Video ID: {response.id}")
print(f"Status: {response.status}")
```

### Video with Input Reference

```python
# Create a video with an input reference image
with open("start_frame.jpg", "rb") as f:
    image_data = f.read()

response = litellm.acreate_video(
    prompt="Continue this scene with the character walking away",
    model="sora-2-pro",
    seconds="8",
    size="1280x720",
    input_reference=("start_frame.jpg", image_data, "image/jpeg")
)
```

## API Reference

### Create Video

Generate a new video from a text prompt.

```python
litellm.acreate_video(
    prompt: str,
    model: str = "sora-2",
    seconds: str = "4",
    size: str = "720x1280",
    input_reference: Optional[Tuple[str, bytes, str]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    organization: Optional[str] = None
) -> OpenAIVideoObject
```

**Parameters:**
- `prompt` (str): Text description of the video to generate
- `model` (str): Model to use for generation ("sora-2" or "sora-2-pro")
- `seconds` (str): Duration in seconds ("4", "8", or "12")
- `size` (str): Video resolution ("720x1280", "1280x720", "1024x1792", "1792x1024")
- `input_reference` (Optional): Tuple of (filename, image_bytes, content_type) for image-guided generation
- `api_key` (Optional): OpenAI API key
- `api_base` (Optional): Custom API base URL
- `timeout` (Optional): Request timeout in seconds
- `max_retries` (Optional): Maximum number of retries
- `organization` (Optional): OpenAI organization ID

**Returns:** `OpenAIVideoObject` with video metadata

### Retrieve Video

Get details about a specific video.

```python
litellm.avideo_retrieve(
    video_id: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    organization: Optional[str] = None
) -> OpenAIVideoObject
```

### Download Video Content

Download the actual video file.

```python
litellm.avideo_content(
    video_id: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    organization: Optional[str] = None
) -> HttpxBinaryResponseContent
```

### Delete Video

Remove a video from your account.

```python
litellm.avideo_delete(
    video_id: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    organization: Optional[str] = None
) -> OpenAIVideoObject
```

### List Videos

Get a list of all your videos.

```python
litellm.avideo_list(
    limit: Optional[int] = None,
    after: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    organization: Optional[str] = None
) -> OpenAIVideoListResponse
```

## Proxy Server Usage

### Configuration

Add video settings to your `config.yaml`:

```yaml
# for /videos endpoints
videos_settings:
  - custom_llm_provider: openai
    api_key: os.environ/OPENAI_API_KEY
```

### Endpoints

The proxy server exposes the following endpoints:

- `POST /v1/videos` - Create video
- `GET /v1/videos/{video_id}` - Retrieve video details
- `GET /v1/videos/{video_id}/content` - Download video content
- `DELETE /v1/videos/{video_id}` - Delete video
- `GET /v1/videos` - List videos

### Example cURL Requests

```bash
# Create a video
curl -X POST "http://localhost:4000/v1/videos" \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat playing with yarn",
    "model": "sora-2",
    "seconds": "4",
    "size": "720x1280"
  }'

# Create video with input reference
curl -X POST "http://localhost:4000/v1/videos" \
  -H "Authorization: Bearer sk-1234" \
  -F "prompt=A cat playing with yarn" \
  -F "model=sora-2" \
  -F "seconds=4" \
  -F "size=720x1280" \
  -F "input_reference=@start_frame.jpg;type=image/jpeg"

# Retrieve video details
curl -X GET "http://localhost:4000/v1/videos/video_123" \
  -H "Authorization: Bearer sk-1234"

# Download video content
curl -X GET "http://localhost:4000/v1/videos/video_123/content" \
  -H "Authorization: Bearer sk-1234" \
  --output video.mp4

# Delete video
curl -X DELETE "http://localhost:4000/v1/videos/video_123" \
  -H "Authorization: Bearer sk-1234"

# List videos
curl -X GET "http://localhost:4000/v1/videos" \
  -H "Authorization: Bearer sk-1234"
```

## Cost Management

Video generation costs are automatically calculated based on:

- **Model**: sora-2 ($0.10/second) or sora-2-pro ($0.30/second)
- **Duration**: Actual video length in seconds
- **Resolution**: Higher resolutions may have different pricing

### Cost Calculation

```python
# Cost is automatically calculated and included in usage tracking
response = litellm.acreate_video(
    prompt="A beautiful sunset",
    model="sora-2",
    seconds="8"
)

# Cost information is available in the response
print(f"Video duration: {response.seconds} seconds")
print(f"Model: {response.model}")
print(f"Usage: {response.usage}")
```

## Error Handling

```python
import litellm
from litellm import OpenAIError, RateLimitError

try:
    response = litellm.acreate_video(
        prompt="A cat playing with yarn",
        model="sora-2"
    )
except RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
except OpenAIError as e:
    print(f"OpenAI API error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Best Practices

### Prompt Engineering

- Be specific and descriptive in your prompts
- Include details about camera movement, lighting, and style
- Mention the desired mood or atmosphere
- Specify the type of scene or action clearly

```python
# Good prompt
prompt = "A majestic eagle soaring over a mountain range at sunset, with dramatic lighting and slow camera movement following the bird's flight path"

# Less effective prompt
prompt = "A bird flying"
```

### Input Reference Usage

- Use high-quality reference images
- Ensure the image matches your desired video resolution
- The reference image should represent the starting frame of your video
- Consider the composition and framing of your reference image

### Performance Optimization

- Use appropriate video durations (4-12 seconds)
- Choose the right resolution for your use case
- Consider using sora-2 for faster generation, sora-2-pro for higher quality
- Implement proper error handling and retry logic

## Limitations

- Maximum video duration: 12 seconds
- Supported resolutions depend on the model
- Generation time varies based on complexity and length
- Rate limits apply based on your OpenAI plan
- Videos are automatically deleted after a certain period

## Troubleshooting

### Common Issues

1. **Invalid prompt**: Ensure your prompt is descriptive and appropriate
2. **Unsupported resolution**: Check that your chosen resolution is supported by the model
3. **Rate limiting**: Implement exponential backoff for retry logic
4. **File upload issues**: Ensure proper multipart/form-data encoding for input_reference

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import litellm
litellm.set_verbose = True

response = litellm.acreate_video(
    prompt="A cat playing with yarn",
    model="sora-2"
)
```

## Examples

### Complete Workflow

```python
import litellm
import asyncio

async def create_and_download_video():
    # Create video
    video = await litellm.acreate_video(
        prompt="A serene lake with gentle ripples and morning mist",
        model="sora-2-pro",
        seconds="8",
        size="1280x720"
    )
    
    print(f"Created video: {video.id}")
    print(f"Status: {video.status}")
    
    # Wait for processing
    while video.status != "completed":
        await asyncio.sleep(5)
        video = await litellm.avideo_retrieve(video.id)
        print(f"Status: {video.status}")
    
    # Download video
    content = await litellm.avideo_content(video.id)
    
    with open("generated_video.mp4", "wb") as f:
        f.write(content.content)
    
    print("Video downloaded successfully!")
    
    # Clean up
    await litellm.avideo_delete(video.id)
    print("Video deleted")

# Run the example
asyncio.run(create_and_download_video())
```

### Batch Video Generation

```python
import litellm
import asyncio

async def generate_multiple_videos():
    prompts = [
        "A cat playing with yarn",
        "A dog running in a park",
        "A bird flying over the ocean"
    ]
    
    tasks = []
    for prompt in prompts:
        task = litellm.acreate_video(
            prompt=prompt,
            model="sora-2",
            seconds="4"
        )
        tasks.append(task)
    
    videos = await asyncio.gather(*tasks)
    
    for i, video in enumerate(videos):
        print(f"Video {i+1}: {video.id} - {video.status}")

asyncio.run(generate_multiple_videos())
```
