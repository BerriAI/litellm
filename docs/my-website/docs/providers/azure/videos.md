import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Azure Video Generation

LiteLLM supports Azure OpenAI's video generation models including Sora with full end-to-end integration.

| Property | Details |
|-------|-------|
| Description | Azure OpenAI's video generation models including Sora-2 |
| Provider Route on LiteLLM | `azure/` |
| Supported Models | `sora-2` |
| Cost Tracking | ✅ Duration-based pricing ($0.10/second) |
| Logging Support | ✅ Full request/response logging |
| Guardrails Support | ✅ Content moderation and safety checks |
| Proxy Server Support | ✅ Full proxy integration with virtual keys |
| Spend Management | ✅ Budget tracking and rate limiting |
| Link to Provider Doc | [Azure OpenAI Video Generation ↗](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/concepts/video-generation) |

## Quick Start

### Required API Keys

```python
import os 
os.environ["AZURE_OPENAI_API_KEY"] = "your-azure-api-key"
os.environ["AZURE_OPENAI_API_BASE"] = "https://your-resource.openai.azure.com/"
```

### Basic Usage

```python
from litellm import video_generation, video_status, video_content
import os
import time

os.environ["AZURE_OPENAI_API_KEY"] = "your-azure-api-key"
os.environ["AZURE_OPENAI_API_BASE"] = "https://your-resource.openai.azure.com/"

# Generate video
response = video_generation(
    model="azure/sora-2",
    prompt="A cat playing with a ball of yarn in a sunny garden",
    seconds="8",
    size="720x1280"
)

print(f"Video ID: {response.id}")
print(f"Initial Status: {response.status}")

# Check status until video is ready
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

# Download video content when ready
video_bytes = video_content(
    video_id=response.id
)

# Save to file
with open("generated_video.mp4", "wb") as f:
    f.write(video_bytes)
```

## Usage - LiteLLM Proxy Server

Here's how to call Azure video generation models with the LiteLLM Proxy Server

### 1. Save key in your environment

```bash
export AZURE_OPENAI_API_KEY="your-azure-api-key"
export AZURE_OPENAI_API_BASE="https://your-resource.openai.azure.com/"
```

### 2. Start the proxy 

<Tabs>
<TabItem value="config" label="config.yaml">

```yaml
model_list:
  - model_name: azure-sora-2
    litellm_params:
      model: azure/sora-2
      api_key: os.environ/AZURE_OPENAI_API_KEY
      api_base: os.environ/AZURE_OPENAI_API_BASE
```

</TabItem>
<TabItem value="cli" label="CLI">

```bash
$ litellm --model azure/sora-2

# Server running on http://0.0.0.0:4000
```

</TabItem>

</Tabs>

### 3. Test it

<Tabs>
<TabItem value="Curl" label="Curl Request">

```shell
curl --location 'http://0.0.0.0:4000/videos/generations' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer sk-1234' \
--data '{
    "model": "azure-sora-2",
    "prompt": "A cat playing with a ball of yarn in a sunny garden",
    "seconds": "8",
    "size": "720x1280"
}'
```

</TabItem>
<TabItem value="openai" label="OpenAI v1.0.0+">

```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

# request sent to model set on litellm proxy, `litellm --model`
response = client.videos.create(
    model="azure-sora-2",
    prompt="A cat playing with a ball of yarn in a sunny garden",
    seconds=8,
    size="720x1280"
)

print(response)
```

</TabItem>
</Tabs>

## Supported Models

| Model Name | 
|------------|
| sora-2 | 
|sora-2-pro |
|sora-2-pro-high-res|


## Logging & Observability

### Request/Response Logging

All video generation requests are automatically logged with:

- **Request details**: prompt, model, duration, size
- **Response details**: video ID, status, creation time
- **Cost tracking**: duration-based pricing calculation
- **Performance metrics**: request latency, processing time

### Logging Providers

Video generation works with all LiteLLM logging providers:

- **Datadog**: Real-time monitoring and alerting
- **Helicone**: Request tracing and debugging
- **LangSmith**: LangChain integration and tracing
- **Custom webhooks**: Send logs to your own endpoints

**Example: Enable Datadog logging**

```yaml
general_settings:
  alerting: ["datadog"]
  datadog_api_key: os.environ/DATADOG_API_KEY
```


## Video Generation Parameters

- `prompt` (required): Text description of the desired video
- `model` (optional): Model to use, defaults to "azure/sora-2"
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
        model="azure/sora-2",
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

## Video Remix (Video Editing)

```python
# Video editing with reference image
response = litellm.video_remix(
    video_id="video_456",
    prompt="Make the cat jump higher",
    input_reference=open("path/to/image.jpg", "rb"),  # Reference image as file object
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
        model="azure/sora-2"
    )
except AuthenticationError as e:
    print(f"Authentication failed: {e}")
except BadRequestError as e:
    print(f"Bad request: {e}")
```
