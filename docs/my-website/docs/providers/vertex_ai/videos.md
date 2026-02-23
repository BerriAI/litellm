import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Vertex AI Video Generation (Veo)

LiteLLM supports Vertex AI's Veo video generation models using the unified OpenAI video API surface.

| Property | Details |
|-------|-------|
| Description | Google Cloud Vertex AI Veo video generation models |
| Provider Route on LiteLLM | `vertex_ai/` |
| Supported Models | `veo-2.0-generate-001`, `veo-3.0-generate-preview`, `veo-3.0-fast-generate-preview`, `veo-3.1-generate-preview`, `veo-3.1-fast-generate-preview` |
| Cost Tracking | ✅ Duration-based pricing |
| Logging Support | ✅ Full request/response logging |
| Proxy Server Support | ✅ Full proxy integration with virtual keys |
| Spend Management | ✅ Budget tracking and rate limiting |
| Link to Provider Doc | [Vertex AI Veo Documentation ↗](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/veo-video-generation) |

## Quick Start

### Required Environment Setup

```python
import json
import os

os.environ["VERTEXAI_PROJECT"] = "your-gcp-project-id"
os.environ["VERTEXAI_LOCATION"] = "us-central1"

# Option 1: Point to a service account file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/service_account.json"

# Option 2: Store the service account JSON directly
with open("/path/to/service_account.json", "r", encoding="utf-8") as f:
    os.environ["VERTEXAI_CREDENTIALS"] = f.read()
```

### Basic Usage

```python
from litellm import video_generation, video_status, video_content
import json
import os
import time

with open("/path/to/service_account.json", "r", encoding="utf-8") as f:
    vertex_credentials = f.read()

response = video_generation(
    model="vertex_ai/veo-3.0-generate-preview",
    prompt="A cat playing with a ball of yarn in a sunny garden",
    vertex_project="your-gcp-project-id",
    vertex_location="us-central1",
    vertex_credentials=vertex_credentials,
    seconds="8",
    size="1280x720",
)

print(f"Video ID: {response.id}")
print(f"Initial Status: {response.status}")

# Poll for completion
while True:
    status = video_status(
        video_id=response.id,
        vertex_project="your-gcp-project-id",
        vertex_location="us-central1",
        vertex_credentials=vertex_credentials,
    )

    print(f"Current Status: {status.status}")

    if status.status == "completed":
        break
    if status.status == "failed":
        raise RuntimeError("Video generation failed")

    time.sleep(10)

# Download the rendered video
video_bytes = video_content(
    video_id=response.id,
    vertex_project="your-gcp-project-id",
    vertex_location="us-central1",
    vertex_credentials=vertex_credentials,
)

with open("generated_video.mp4", "wb") as f:
    f.write(video_bytes)
```

## Supported Models

| Model Name | Description | Max Duration | Status |
|------------|-------------|--------------|--------|
| veo-2.0-generate-001 | Veo 2.0 video generation | 5 seconds | GA |
| veo-3.0-generate-preview | Veo 3.0 high quality | 8 seconds | Preview |
| veo-3.0-fast-generate-preview | Veo 3.0 fast generation | 8 seconds | Preview |
| veo-3.1-generate-preview | Veo 3.1 high quality | 10 seconds | Preview |
| veo-3.1-fast-generate-preview | Veo 3.1 fast | 10 seconds | Preview |

## Video Generation Parameters

LiteLLM converts OpenAI-style parameters to Veo's API shape automatically:

| OpenAI Parameter | Vertex AI Parameter | Description | Example |
|------------------|---------------------|-------------|---------|
| `prompt` | `instances[].prompt` | Text description of the video | "A cat playing" |
| `size` | `parameters.aspectRatio` | Converted to `16:9` or `9:16` | "1280x720" → `16:9` |
| `seconds` | `parameters.durationSeconds` | Clip length in seconds | "8" → `8` |
| `input_reference` | `instances[].image` | Reference image for animation | `open("image.jpg", "rb")` |
| Provider-specific params | `extra_body` | Forwarded to Vertex API | `{"negativePrompt": "blurry"}` |

### Size to Aspect Ratio Mapping

- `1280x720`, `1920x1080` → `16:9`
- `720x1280`, `1080x1920` → `9:16`
- Unknown sizes default to `16:9`

## Async Usage

```python
from litellm import avideo_generation, avideo_status, avideo_content
import asyncio
import json

with open("/path/to/service_account.json", "r", encoding="utf-8") as f:
    vertex_credentials = f.read()


async def workflow():
    response = await avideo_generation(
        model="vertex_ai/veo-3.1-generate-preview",
        prompt="Slow motion water droplets splashing into a pool",
        seconds="10",
        vertex_project="your-gcp-project-id",
        vertex_location="us-central1",
        vertex_credentials=vertex_credentials,
    )

    while True:
        status = await avideo_status(
            video_id=response.id,
            vertex_project="your-gcp-project-id",
            vertex_location="us-central1",
            vertex_credentials=vertex_credentials,
        )

        if status.status == "completed":
            break
        if status.status == "failed":
            raise RuntimeError("Video generation failed")

        await asyncio.sleep(10)

    video_bytes = await avideo_content(
        video_id=response.id,
        vertex_project="your-gcp-project-id",
        vertex_location="us-central1",
        vertex_credentials=vertex_credentials,
    )

    with open("veo_water.mp4", "wb") as f:
        f.write(video_bytes)

asyncio.run(workflow())
```

## LiteLLM Proxy Usage

Add Veo models to your `config.yaml`:

```yaml
model_list:
  - model_name: veo-3
    litellm_params:
      model: vertex_ai/veo-3.0-generate-preview
      vertex_project: os.environ/VERTEXAI_PROJECT
      vertex_location: os.environ/VERTEXAI_LOCATION
      vertex_credentials: os.environ/VERTEXAI_CREDENTIALS
```

Start the proxy and make requests:

<Tabs>
<TabItem value="curl" label="Curl">

```bash
# Step 1: Generate video
curl --location 'http://0.0.0.0:4000/videos' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer sk-1234' \
--data '{
  "model": "veo-3",
  "prompt": "Aerial shot over a futuristic city at sunrise",
  "seconds": "8"
}'

# Step 2: Poll status
curl --location 'http://localhost:4000/v1/videos/{video_id}' \
--header 'x-litellm-api-key: sk-1234'

# Step 3: Download video
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

response = litellm.video_generation(
    model="veo-3",
    prompt="Aerial shot over a futuristic city at sunrise",
)

status = litellm.video_status(video_id=response.id)
while status.status not in ["completed", "failed"]:
    status = litellm.video_status(video_id=response.id)

if status.status == "completed":
    content = litellm.video_content(video_id=response.id)
    with open("veo_city.mp4", "wb") as f:
        f.write(content)
```

</TabItem>
</Tabs>

## Cost Tracking

LiteLLM records the duration returned by Veo so you can apply duration-based pricing.

```python
with open("/path/to/service_account.json", "r", encoding="utf-8") as f:
    vertex_credentials = f.read()

response = video_generation(
    model="vertex_ai/veo-2.0-generate-001",
    prompt="Flowers blooming in fast forward",
    seconds="5",
    vertex_project="your-gcp-project-id",
    vertex_location="us-central1",
    vertex_credentials=vertex_credentials,
)

print(response.usage)  # {"duration_seconds": 5.0}
```

## Troubleshooting

- **`vertex_project is required`**: set `VERTEXAI_PROJECT` env var or pass `vertex_project` in the request.
- **`Permission denied`**: ensure the service account has the `Vertex AI User` role and the correct region enabled.
- **Video stuck in `processing`**: Veo operations are long-running. Continue polling every 10–15 seconds up to ~10 minutes.

## See Also

- [OpenAI Video Generation](../openai/videos.md)
- [Azure Video Generation](../azure/videos.md)
- [Gemini Video Generation](../gemini/videos.md)
- [Video Generation API Reference](/docs/videos)

