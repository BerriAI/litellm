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
from litellm import video_generation, video_retrieval
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
video_bytes = video_retrieval(
    video_id=response.id,
    model="sora-2"
)

# Save to file
with open("generated_video.mp4", "wb") as f:
    f.write(video_bytes)
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
video_bytes = video_retrieval(
    video_id="video_1234567890",
    model="sora-2"
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
    video_bytes = litellm.video_retrieval(
        video_id=video_id,
        model="sora-2"
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
    input_reference="path/to/image.jpg",  # Reference image
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
        prompt="A cat playing with a ball of yarn",
        model="sora-2"
    )
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
except BadRequestError as e:
    print(f"Bad request: {e}")
```
