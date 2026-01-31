# RunwayML - Video Generation

LiteLLM supports RunwayML's Gen-4 video generation API, allowing you to generate videos from text prompts and images.

## Quick Start

```python showLineNumbers title="Basic Video Generation"
from litellm import video_generation
import os

os.environ["RUNWAYML_API_KEY"] = "your-api-key"

# Generate video from text and image
response = video_generation(
    model="runwayml/gen4_turbo",
    prompt="A high quality demo video of litellm ai gateway",
    input_reference="https://media.licdn.com/dms/image/v2/D4D0BAQFqOrIAJEgtLw/company-logo_200_200/company-logo_200_200/0/1714076049190/berri_ai_logo?e=2147483647&v=beta&t=7tG_KRZZ4MPGc7Iin79PcFcrpvf5Hu6rBM4ptHGU1DY",
    seconds=5,
    size="1280x720"
)

print(f"Video ID: {response.id}")
print(f"Status: {response.status}")
```

## Authentication

Set your RunwayML API key:

```python showLineNumbers title="Set API Key"
import os

os.environ["RUNWAYML_API_KEY"] = "your-api-key"
```

## Supported Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model to use (e.g., `runwayml/gen4_turbo`) |
| `prompt` | string | Yes | Text description for the video |
| `input_reference` | string/file | Yes | URL or file path to reference image |
| `seconds` | int | No | Video duration (5 or 10 seconds) |
| `size` | string | No | Video dimensions (`1280x720` or `720x1280`). Can also use `ratio` format (`1280:720`) |

## Complete Workflow

```python showLineNumbers title="Complete Video Generation Workflow"
from litellm import video_generation, video_status, video_content
import os
import time

os.environ["RUNWAYML_API_KEY"] = "your-api-key"

# 1. Generate video
response = video_generation(
    model="runwayml/gen4_turbo",
    prompt="A high quality demo video of litellm ai gateway",
    input_reference="https://media.licdn.com/dms/image/v2/D4D0BAQFqOrIAJEgtLw/company-logo_200_200/company-logo_200_200/0/1714076049190/berri_ai_logo?e=2147483647&v=beta&t=7tG_KRZZ4MPGc7Iin79PcFcrpvf5Hu6rBM4ptHGU1DY",
    seconds=5,
    size="1280x720"
)

video_id = response.id
print(f"Video generation started: {video_id}")

# 2. Check status until completed
while True:
    status_response = video_status(video_id=video_id)
    print(f"Status: {status_response.status}")
    
    if status_response.status == "completed":
        print("Video generation completed!")
        break
    elif status_response.status == "failed":
        print("Video generation failed")
        break
    
    time.sleep(10)  # Wait 10 seconds before checking again

# 3. Download video content
video_bytes = video_content(video_id=video_id)

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

os.environ["RUNWAYML_API_KEY"] = "your-api-key"

async def generate_video():
    # Generate video
    response = await avideo_generation(
        model="runwayml/gen4_turbo",
        prompt="A serene lake with mountains in the background",
        input_reference="https://example.com/lake.jpg",
        seconds=5,
        size="1280x720"
    )
    
    video_id = response.id
    print(f"Video generation started: {video_id}")
    
    # Poll for completion
    while True:
        status_response = await avideo_status(video_id=video_id)
        print(f"Status: {status_response.status}")
        
        if status_response.status == "completed":
            break
        elif status_response.status == "failed":
            print("Video generation failed")
            return
        
        await asyncio.sleep(10)
    
    # Download video
    video_bytes = await avideo_content(video_id=video_id)
    
    # Save to file
    with open("generated_video.mp4", "wb") as f:
        f.write(video_bytes)
    
    print("Video saved successfully!")

asyncio.run(generate_video())
```

## LiteLLM Proxy Usage

Add RunwayML to your proxy configuration:

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: gen4-turbo
    litellm_params:
      model: runwayml/gen4_turbo
      api_key: os.environ/RUNWAYML_API_KEY
```

Start the proxy:

```bash
litellm --config /path/to/config.yaml
```

Generate videos through the proxy:

```bash showLineNumbers title="Proxy Request"
curl --location 'http://localhost:4000/v1/videos' \
--header 'Content-Type: application/json' \
--header 'x-litellm-api-key: sk-1234' \
--data '{
    "model": "runwayml/gen4_turbo",
    "prompt": "A high quality demo video of litellm ai gateway",
    "input_reference": "https://media.licdn.com/dms/image/v2/D4D0BAQFqOrIAJEgtLw/company-logo_200_200/company-logo_200_200/0/1714076049190/berri_ai_logo?e=2147483647&v=beta&t=7tG_KRZZ4MPGc7Iin79PcFcrpvf5Hu6rBM4ptHGU1DY",
    "ratio": "1280:720"
}'
```

Check video status:

```bash showLineNumbers title="Check Status"
curl --location 'http://localhost:4000/v1/videos/{video_id}' \
--header 'x-litellm-api-key: sk-1234'
```

Download video content:

```bash showLineNumbers title="Download Video"
curl --location 'http://localhost:4000/v1/videos/{video_id}/content' \
--header 'x-litellm-api-key: sk-1234' \
--output video.mp4
```

## Supported Models

| Model | Description | Duration | Aspect Ratios |
|-------|-------------|----------|---------------|
| `runwayml/gen4_turbo` | Fast video generation | 5-10s | 1280x720, 720x1280 |

## Error Handling

```python showLineNumbers title="Error Handling"
from litellm import video_generation, video_status
import time

try:
    response = video_generation(
        model="runwayml/gen4_turbo",
        prompt="A scenic mountain view",
        input_reference="https://example.com/mountain.jpg",
        seconds=5
    )
    
    # Poll for completion
    max_attempts = 60  # 10 minutes max
    attempts = 0
    
    while attempts < max_attempts:
        status_response = video_status(video_id=response.id)
        
        if status_response.status == "completed":
            print("Video generation completed!")
            break
        elif status_response.status == "failed":
            error = status_response.error or {}
            print(f"Video generation failed: {error.get('message', 'Unknown error')}")
            break
        
        time.sleep(10)
        attempts += 1
    
    if attempts >= max_attempts:
        print("Video generation timed out")
        
except Exception as e:
    print(f"Error: {str(e)}")
```

## Cost Tracking

LiteLLM automatically tracks RunwayML video generation costs:

```python showLineNumbers title="Cost Tracking"
from litellm import video_generation, completion_cost

response = video_generation(
    model="runwayml/gen4_turbo",
    prompt="A high quality demo video of litellm ai gateway",
    input_reference="https://media.licdn.com/dms/image/v2/D4D0BAQFqOrIAJEgtLw/company-logo_200_200/company-logo_200_200/0/1714076049190/berri_ai_logo?e=2147483647&v=beta&t=7tG_KRZZ4MPGc7Iin79PcFcrpvf5Hu6rBM4ptHGU1DY",
    seconds=5,
    size="1280x720"
)

# Calculate cost
cost = completion_cost(completion_response=response)
print(f"Video generation cost: ${cost}")
```

## API Reference

For complete API details, see the [OpenAI Video Generation API specification](https://platform.openai.com/docs/guides/video-generation) which LiteLLM follows.

## Supported Features

| Feature | Supported |
|---------|-----------|
| Video Generation | ✅ |
| Image-to-Video | ✅ |
| Status Checking | ✅ |
| Content Download | ✅ |
| Cost Tracking | ✅ |
| Logging | ✅ |
| Fallbacks | ✅ |
| Load Balancing | ✅ |

