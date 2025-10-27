import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# /images/edits

LiteLLM provides image editing functionality that maps to OpenAI's `/images/edits` API endpoint. Now supports both single and multiple image editing.

| Feature | Supported | Notes |
|---------|-----------|--------|
| Cost Tracking | ✅ | Works with all supported models |
| Logging | ✅ | Works across all integrations |
| End-user Tracking | ✅ | |
| Fallbacks | ✅ | Works between supported models |
| Loadbalancing | ✅ | Works between supported models |
| Supported operations | Create image edits | Single and multiple images supported |
| Supported LiteLLM SDK Versions | 1.63.8+ | |
| Supported LiteLLM Proxy Versions | 1.71.1+ | |
| Supported LLM providers | **OpenAI** | Currently only `openai` is supported |

 #### ⚡️See all supported models and providers at [models.litellm.ai](https://models.litellm.ai/)


## Usage

### LiteLLM Python SDK

<Tabs>
<TabItem value="openai" label="OpenAI">

#### Basic Image Edit
```python showLineNumbers title="OpenAI Image Edit"
import litellm

# Edit an image with a prompt
response = litellm.image_edit(
    model="gpt-image-1",
    image=open("original_image.png", "rb"),
    prompt="Add a red hat to the person in the image",
    n=1,
    size="1024x1024"
)

print(response)
```

#### Multiple Images Edit
```python showLineNumbers title="OpenAI Multiple Images Edit"
import litellm

# Edit multiple images with a prompt
response = litellm.image_edit(
    model="gpt-image-1",
    image=[
        open("image1.png", "rb"),
        open("image2.png", "rb"),
        open("image3.png", "rb")
    ],
    prompt="Apply vintage filter to all images",
    n=1,
    size="1024x1024"
)

print(response)
```

#### Image Edit with Mask
```python showLineNumbers title="OpenAI Image Edit with Mask"
import litellm

# Edit an image with a mask to specify the area to edit
response = litellm.image_edit(
    model="gpt-image-1",
    image=open("original_image.png", "rb"),
    mask=open("mask_image.png", "rb"),  # Transparent areas will be edited
    prompt="Replace the background with a beach scene",
    n=2,
    size="512x512",
    response_format="url"
)

print(response)
```

#### Async Image Edit
```python showLineNumbers title="Async OpenAI Image Edit"
import litellm
import asyncio

async def edit_image():
    response = await litellm.aimage_edit(
        model="gpt-image-1",
        image=open("original_image.png", "rb"),
        prompt="Make the image look like a painting",
        n=1,
        size="1024x1024",
        response_format="b64_json"
    )
    return response

# Run the async function
response = asyncio.run(edit_image())
print(response)
```

#### Async Multiple Images Edit
```python showLineNumbers title="Async OpenAI Multiple Images Edit"
import litellm
import asyncio

async def edit_multiple_images():
    response = await litellm.aimage_edit(
        model="gpt-image-1",
        image=[
            open("portrait1.png", "rb"),
            open("portrait2.png", "rb")
        ],
        prompt="Add professional lighting to the portraits",
        n=1,
        size="1024x1024",
        response_format="url"
    )
    return response

# Run the async function
response = asyncio.run(edit_multiple_images())
print(response)
```

#### Image Edit with Custom Parameters
```python showLineNumbers title="OpenAI Image Edit with Custom Parameters"
import litellm

# Edit image with additional parameters
response = litellm.image_edit(
    model="gpt-image-1",
    image=open("portrait.png", "rb"),
    prompt="Add sunglasses and a smile",
    n=3,
    size="1024x1024",
    response_format="url",
    user="user-123",
    timeout=60,
    extra_headers={"Custom-Header": "value"}
)

print(f"Generated {len(response.data)} image variations")
for i, image_data in enumerate(response.data):
    print(f"Image {i+1}: {image_data.url}")
```

</TabItem>
</Tabs>

### LiteLLM Proxy with OpenAI SDK


<Tabs>
<TabItem value="openai" label="OpenAI">

First, add this to your litellm proxy config.yaml:
```yaml showLineNumbers title="OpenAI Proxy Configuration"
model_list:
  - model_name: gpt-image-1
    litellm_params:
      model: gpt-image-1
      api_key: os.environ/OPENAI_API_KEY
```

Start the LiteLLM proxy server:

```bash showLineNumbers title="Start LiteLLM Proxy Server"
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

#### Basic Image Edit via Proxy
```python showLineNumbers title="OpenAI Proxy Image Edit"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-api-key"             # Your proxy API key
)

# Edit an image
response = client.images.edit(
    model="gpt-image-1",
    image=open("original_image.png", "rb"),
    prompt="Add a red hat to the person in the image",
    n=1,
    size="1024x1024"
)

print(response)
```

#### cURL Example
```bash showLineNumbers title="cURL Image Edit Request"
curl -X POST "http://localhost:4000/v1/images/edits" \
  -H "Authorization: Bearer your-api-key" \
  -F "model=gpt-image-1" \
  -F "image=@original_image.png" \
  -F "mask=@mask_image.png" \
  -F "prompt=Add a beautiful sunset in the background" \
  -F "n=1" \
  -F "size=1024x1024" \
  -F "response_format=url"
```

#### cURL Multiple Images Example
```bash showLineNumbers title="cURL Multiple Images Edit Request"
curl -X POST "http://localhost:4000/v1/images/edits" \
  -H "Authorization: Bearer your-api-key" \
  -F "model=gpt-image-1" \
  -F "image=@image1.png" \
  -F "image=@image2.png" \
  -F "image=@image3.png" \
  -F "prompt=Apply artistic filter to all images" \
  -F "n=1" \
  -F "size=1024x1024" \
  -F "response_format=url"
```

</TabItem>
</Tabs>

## Supported Image Edit Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `image` | `FileTypes` | The image to edit. Must be a valid PNG file, less than 4MB, and square. | ✅ |
| `prompt` | `str` | A text description of the desired image edit. | ✅ |
| `model` | `str` | The model to use for image editing | Optional (defaults to `dall-e-2`) |
| `mask` | `str` | An additional image whose fully transparent areas indicate where the original image should be edited. Must be a valid PNG file, less than 4MB, and have the same dimensions as `image`. | Optional |
| `n` | `int` | The number of images to generate. Must be between 1 and 10. | Optional (defaults to 1) |
| `size` | `str` | The size of the generated images. Must be one of `256x256`, `512x512`, or `1024x1024`. | Optional (defaults to `1024x1024`) |
| `response_format` | `str` | The format in which the generated images are returned. Must be one of `url` or `b64_json`. | Optional (defaults to `url`) |
| `user` | `str` | A unique identifier representing your end-user. | Optional |


## Response Format

The response follows the OpenAI Images API format:

```python showLineNumbers title="Image Edit Response Structure"
{
    "created": 1677649800,
    "data": [
        {
            "url": "https://example.com/edited_image_1.png"
        },
        {
            "url": "https://example.com/edited_image_2.png"
        }
    ]
}
```

For `b64_json` format:
```python showLineNumbers title="Base64 Response Structure"
{
    "created": 1677649800,
    "data": [
        {
            "b64_json": "iVBORw0KGgoAAAANSUhEUgAA..."
        }
    ]
}
```
