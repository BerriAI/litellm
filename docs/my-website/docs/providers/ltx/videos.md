# LTX Video Generation

LiteLLM supports LTX video generation through the `/videos` API.

| Property | Details |
|-------|-------|
| Description | LTX text-to-video and image-to-video generation |
| Provider Route on LiteLLM | `ltx/` |
| Supported Models | `ltx/ltx-2-3-fast`, `ltx/ltx-2-3-pro` |
| Cost Tracking | ✅ Duration-based pricing |
| Logging Support | ✅ |
| Proxy Server Support | ✅ |
| Spend Management | ✅ |
| Link to Provider Doc | [LTX Documentation ↗](https://docs.ltx.video/) |

## Quick Start

### Required API Key

```python
import os

os.environ["LTX_API_KEY"] = "your-ltx-api-key"
```

### Basic Usage

LTX generation is synchronous in this LiteLLM integration. `video_generation()` returns a completed `VideoObject`, and you can immediately call `video_content()` to fetch the generated MP4.

```python
from litellm import video_generation, video_content
import os

os.environ["LTX_API_KEY"] = "your-ltx-api-key"

response = video_generation(
    model="ltx/ltx-2-3-fast",
    prompt="A cinematic drone shot over snowy mountains at sunrise",
    seconds="5",
    size="1920x1080",
)

print(f"Video ID: {response.id}")
print(f"Status: {response.status}")  # completed

video_bytes = video_content(video_id=response.id)

with open("ltx_output.mp4", "wb") as f:
    f.write(video_bytes)
```

## Supported Models

| Model | Description | Input Types | Supported Resolutions |
|-------|-------------|-------------|-----------------------|
| `ltx/ltx-2-3-fast` | Faster LTX video generation | text, image | `1280x720`, `1920x1080`, `2560x1440`, `3840x2160` |
| `ltx/ltx-2-3-pro` | Higher-quality LTX video generation | text, image | `1280x720`, `1920x1080`, `2560x1440`, `3840x2160` |

## Supported Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | LTX model name |
| `prompt` | string | Yes | Text prompt for the generated video |
| `seconds` | string/int | No | Duration in seconds |
| `size` | string | No | Output resolution |
| `input_reference` | string | No | Image reference for image-to-video |
| `extra_body` | dict | No | LTX-specific fields like `fps`, `generate_audio`, `camera_motion` |

## Text-to-Video

```python
from litellm import video_generation, video_content

response = video_generation(
    model="ltx/ltx-2-3-fast",
    prompt="A slow cinematic pan across a neon-lit city at night",
    seconds="5",
    size="1920x1080",
)

video_bytes = video_content(video_id=response.id)

with open("ltx_text_to_video.mp4", "wb") as f:
    f.write(video_bytes)
```

## Image-to-Video

Pass an image URL, a data URI, or another LTX-supported URI as `input_reference`. LiteLLM maps this to LTX's `image_uri` request field.

```python
from litellm import video_generation, video_content

response = video_generation(
    model="ltx/ltx-2-3-pro",
    prompt="Animate the clouds and add a slow camera push-in",
    input_reference="https://example.com/reference-image.jpg",
    seconds="5",
    size="1280x720",
)

video_bytes = video_content(video_id=response.id)

with open("ltx_image_to_video.mp4", "wb") as f:
    f.write(video_bytes)
```

## Passing LTX-Specific Parameters

Use `extra_body` for provider-specific fields that are not part of LiteLLM's generic video API.

```python
from litellm import video_generation

response = video_generation(
    model="ltx/ltx-2-3-pro",
    prompt="A product shot with slow dolly motion and ambient sound",
    seconds="5",
    size="1920x1080",
    extra_body={
        "fps": 30,
        "generate_audio": True,
        "camera_motion": "dolly_in",
    },
)

print(response)
```

## Local File Convenience Script

If you want to test image-to-video with a local file path, use the helper script in the repo:

- `cookbook/ltx_video_generation.py`

It converts local files to data URIs, calls `litellm.video_generation()`, then saves the bytes returned by `litellm.video_content()`.

Example:

```bash
export LTX_API_KEY=your-ltx-api-key

poetry run python cookbook/ltx_video_generation.py \
  --prompt "The camera slowly pushes in while clouds drift overhead" \
  --input-reference ./path/to/reference.jpg \
  --model ltx/ltx-2-3-pro
```

## LiteLLM Proxy Usage

### Configuration

Add LTX to your `config.yaml`:

```yaml
model_list:
  - model_name: ltx-fast
    litellm_params:
      model: ltx/ltx-2-3-fast
      api_key: os.environ/LTX_API_KEY
```

Start the proxy:

```bash
litellm --config config.yaml
```

### Generate Video

```bash
curl --location 'http://localhost:4000/v1/videos' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer sk-1234' \
--data '{
  "model": "ltx-fast",
  "prompt": "A cinematic drone shot over snowy mountains at sunrise",
  "seconds": "5",
  "size": "1920x1080"
}'
```

### Download Video Content

```bash
curl --location 'http://localhost:4000/v1/videos/{video_id}/content' \
--header 'Authorization: Bearer sk-1234' \
--output ltx_output.mp4
```

## Important Behavior Notes

- LTX generation is synchronous in this integration. `video_generation()` returns a completed response rather than a queued job.
- `video_status()`, `video_list()`, `video_delete()`, and `video_remix()` are not supported for LTX on this branch.
- Video content is persisted locally and then served by `video_content()`. In practice, generation and retrieval should happen on the same process / instance for reliable access.

## Troubleshooting

### `video_content()` cannot find the generated file

This usually means you are trying to retrieve the video from a different process or instance than the one that generated it.

### `input_reference` with a local file path does not work directly

Pass a URL or data URI directly, or use `cookbook/ltx_video_generation.py` to convert a local file path before sending the request.

### Unsupported parameter errors

Pass LTX-specific fields through `extra_body`. Generic unsupported fields like `user` are not forwarded to LTX.
