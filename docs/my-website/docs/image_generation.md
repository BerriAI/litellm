# Image Generation

## Quick Start

```python
from litellm import image_generation
import os 

# set api keys 
os.environ["OPENAI_API_KEY"] = ""

response = image_generation(prompt="A cute baby sea otter", model="dall-e-3")

print(f"response: {response}")
```

### Input Params for `litellm.image_generation()`
### Required Fields

- `prompt`: *string* - A text description of the desired image(s).  

### Optional LiteLLM Fields

    model: Optional[str] = None,
    n: Optional[int] = None,
    quality: Optional[str] = None,
    response_format: Optional[str] = None,
    size: Optional[str] = None,
    style: Optional[str] = None,
    user: Optional[str] = None,
    timeout=600,  # default to 10 minutes
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    litellm_logging_obj=None,
    custom_llm_provider=None,

- `model`: *string (optional)* The model to use for image generation. Defaults to openai/dall-e-2

- `n`: *int (optional)* The number of images to generate. Must be between 1 and 10. For dall-e-3, only n=1 is supported.

- `quality`: *string (optional)* The quality of the image that will be generated. hd creates images with finer details and greater consistency across the image. This param is only supported for dall-e-3.

- `response_format`: *string (optional)* The format in which the generated images are returned. Must be one of url or b64_json.

- `size`: *string (optional)* The size of the generated images. Must be one of 256x256, 512x512, or 1024x1024 for dall-e-2. Must be one of 1024x1024, 1792x1024, or 1024x1792 for dall-e-3 models.

- `timeout`: *integer* - The maximum time, in seconds, to wait for the API to respond. Defaults to 600 seconds (10 minutes).

- `user`: *string (optional)* A unique identifier representing your end-user, 

- `api_base`: *string (optional)* - The api endpoint you want to call the model with

- `api_version`: *string (optional)* - (Azure-specific) the api version for the call

- `api_key`: *string (optional)* - The API key to authenticate and authorize requests. If not provided, the default API key is used.

- `api_type`: *string (optional)* - The type of API to use.

### Output from `litellm.embedding()`

```json

{
    "created": 1703658209,
    "data": [{
        'b64_json': None, 
        'revised_prompt': 'Adorable baby sea otter with a coat of thick brown fur, playfully swimming in blue ocean waters. Its curious, bright eyes gleam as it is surfaced above water, tiny paws held close to its chest, as it playfully spins in the gentle waves under the soft rays of a setting sun.', 
        'url': 'https://oaidalleapiprodscus.blob.core.windows.net/private/org-ikDc4ex8NB5ZzfTf8m5WYVB7/user-JpwZsbIXubBZvan3Y3GchiiB/img-dpa3g5LmkTrotY6M93dMYrdE.png?st=2023-12-27T05%3A23%3A29Z&se=2023-12-27T07%3A23%3A29Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-12-26T13%3A22%3A56Z&ske=2023-12-27T13%3A22%3A56Z&sks=b&skv=2021-08-06&sig=hUuQjYLS%2BvtsDdffEAp2gwewjC8b3ilggvkd9hgY6Uw%3D'
    }],
    "usage": {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
}
```

## OpenAI Image Generation Models

### Usage
```python
from litellm import image_generation
import os
os.environ['OPENAI_API_KEY'] = ""
response = image_generation(model='dall-e-2', prompt="cute baby otter")
```

| Model Name           | Function Call                               | Required OS Variables                |
|----------------------|---------------------------------------------|--------------------------------------|
| dall-e-2 | `image_generation(model='dall-e-2', prompt="cute baby otter")` | `os.environ['OPENAI_API_KEY']`       |
| dall-e-3 | `image_generation(model='dall-e-2', prompt="cute baby otter")` | `os.environ['OPENAI_API_KEY']`       |

## Azure OpenAI Image Generation Models

### API keys
This can be set as env variables or passed as **params to litellm.image_generation()**
```python
import os
os.environ['AZURE_API_KEY'] = 
os.environ['AZURE_API_BASE'] = 
os.environ['AZURE_API_VERSION'] = 
```

### Usage
```python
from litellm import embedding
response = embedding(
    model="azure/<your deployment name>",
    prompt="cute baby otter",
    api_key=api_key,
    api_base=api_base,
    api_version=api_version,
)
print(response)
```

| Model Name           | Function Call                               |
|----------------------|---------------------------------------------|
| dall-e-2 | `image_generation(model="azure/<your deployment name>", prompt="cute baby otter")` |
| dall-e-3 | `image_generation(model="azure/<your deployment name>", prompt="cute baby otter")` |


## OpenAI Compatible Image Generation Models
Use this for calling `/image_generation` endpoints on OpenAI Compatible Servers, example https://github.com/xorbitsai/inference

**Note add `openai/` prefix to model so litellm knows to route to OpenAI**

### Usage
```python
from litellm import image_generation
response = image_generation(
  model = "openai/<your-llm-name>",     # add `openai/` prefix to model so litellm knows to route to OpenAI
  api_base="http://0.0.0.0:8000/"       # set API Base of your Custom OpenAI Endpoint
  prompt="cute baby otter"
)
```