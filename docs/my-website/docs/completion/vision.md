import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Using Vision Models

## Quick Start
Example passing images to a model 


<Tabs>

<TabItem label="LiteLLMPython SDK" value="Python">

```python
import os 
from litellm import completion

os.environ["OPENAI_API_KEY"] = "your-api-key"

# openai call
response = completion(
    model = "gpt-4-vision-preview", 
    messages=[
        {
            "role": "user",
            "content": [
                            {
                                "type": "text",
                                "text": "What’s in this image?"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                                }
                            }
                        ]
        }
    ],
)

```

</TabItem>
<TabItem label="LiteLLM Proxy Server" value="proxy">

1. Define vision models on config.yaml

```yaml
model_list:
  - model_name: gpt-4-vision-preview # OpenAI gpt-4-vision-preview
    litellm_params:
      model: openai/gpt-4-vision-preview
      api_key: os.environ/OPENAI_API_KEY
  - model_name: llava-hf          # Custom OpenAI compatible model
    litellm_params:
      model: openai/llava-hf/llava-v1.6-vicuna-7b-hf
      api_base: http://localhost:8000
      api_key: fake-key
    model_info:
      supports_vision: True        # set supports_vision to True so /model/info returns this attribute as True

```

2. Run proxy server

```bash
litellm --config config.yaml
```

3. Test it using the OpenAI Python SDK


```python
import os 
from openai import OpenAI

client = OpenAI(
    api_key="sk-1234", # your litellm proxy api key
)

response = client.chat.completions.create(
    model = "gpt-4-vision-preview",  # use model="llava-hf" to test your custom OpenAI endpoint
    messages=[
        {
            "role": "user",
            "content": [
                            {
                                "type": "text",
                                "text": "What’s in this image?"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                                }
                            }
                        ]
        }
    ],
)

```




</TabItem>
</Tabs>



## Checking if a model supports `vision`

<Tabs>
<TabItem label="LiteLLM Python SDK" value="Python">

Use `litellm.supports_vision(model="")` -> returns `True` if model supports `vision` and `False` if not

```python
assert litellm.supports_vision(model="gpt-4-vision-preview") == True
assert litellm.supports_vision(model="gemini-1.0-pro-vision") == True
assert litellm.supports_vision(model="gpt-3.5-turbo") == False
```
</TabItem>

<TabItem label="LiteLLM Proxy Server" value="proxy">


1. Define vision models on config.yaml

```yaml
model_list:
  - model_name: gpt-4-vision-preview # OpenAI gpt-4-vision-preview
    litellm_params:
      model: openai/gpt-4-vision-preview
      api_key: os.environ/OPENAI_API_KEY
  - model_name: llava-hf          # Custom OpenAI compatible model
    litellm_params:
      model: openai/llava-hf/llava-v1.6-vicuna-7b-hf
      api_base: http://localhost:8000
      api_key: fake-key
    model_info:
      supports_vision: True        # set supports_vision to True so /model/info returns this attribute as True
```

2. Run proxy server

```bash
litellm --config config.yaml
```

3. Call `/model_group/info` to check if your model supports `vision`

```shell
curl -X 'GET' \
  'http://localhost:4000/model_group/info' \
  -H 'accept: application/json' \
  -H 'x-api-key: sk-1234'
```

Expected Response 

```json
{
  "data": [
    {
      "model_group": "gpt-4-vision-preview",
      "providers": ["openai"],
      "max_input_tokens": 128000,
      "max_output_tokens": 4096,
      "mode": "chat",
      "supports_vision": true, # 👈 supports_vision is true
      "supports_function_calling": false
    },
    {
      "model_group": "llava-hf",
      "providers": ["openai"],
      "max_input_tokens": null,
      "max_output_tokens": null,
      "mode": null,
      "supports_vision": true, # 👈 supports_vision is true
      "supports_function_calling": false
    }
  ]
}
```

</TabItem>
</Tabs>

## Image Analysis Using REST API

This section showcases how to use REST API for image analysis tasks, such as image description or extracting structured data.

Example: Describing an Image

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_KEY" \
  -d '{
    "model": "azure_ai/phi35-vision-instruct",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Describe the given image."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://www.imperial.ac.uk/media/migration/administration-and-support-services/03--tojpeg_1487001148545_x2.jpg"
            }
          }
        ]
      }
    ]
  }'
  ```
## Input Parameters

- **model**: `string`  
  The model to use for image analysis.  
  **Example**: `azure_ai/phi35-vision-instruct`.

- **messages**: `array`  
  An array of message objects that form the conversation between the user and the assistant.

  - **role**: `string`  
    The role of the message sender.  
    Can be `"system"`, `"user"`, or `"assistant"`.

  - **content**: `array`  
    An array of content items. Each content item can be either a text instruction or an image URL.
  
    - **type**: `string`  
      Specifies the type of content; either `"text"` or `"image_url"`.

      - If type is `"text"`, the content item must include:
        - **text**: `string`  
          The text instruction or message content (e.g., "Describe the image", "Extract information from this receipt").

      - If type is `"image_url"`, the content item must include:
        - **image_url**: `object`  
          An object containing:
          - **url**: `string`  
            The URL of the image to be analyzed.

## Example: Structured Data Extraction

This example demonstrates how to use the API to extract structured data, such as fields from a receipt.

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_KEY" \
  -d '{
    "model": "azure_ai/phi35-vision-instruct",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant. Your task is to extract data from payment receipts in JSON format following the schema: {\"date\": \"receipt date\", \"transaction\": \"transaction number\", \"location\": \"transaction location\", \"amount\": \"transaction amount\"}."
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Extract and provide the information from this receipt in JSON format."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://www.imperial.ac.uk/media/migration/administration-and-support-services/03--tojpeg_1487001148545_x2.jpg"
            }
          }
        ]
      }
    ]
  }'
```

## Required Fields

When making a request to the API, the following fields are required:

- **model**: _string_  
    The identifier of the model used for image analysis (e.g., `azure_ai/phi35-vision-instruct`).
    
- **messages**: _array_  
    The conversation messages, including the user's instructions and the image URL.
    
## Example Model for Image Analysis

In this example, the **`Phi3.5 Vision Instruct`** model was used to demonstrate image analysis tasks. This model supports structured JSON output by default for tasks such as image description and data extraction.

However, other vision models can also be used for image analysis tasks, with the possibility of requiring additional configuration for structured outputs.

### Supported Output Formats

- **JSON**:  
    Extracted structured data in JSON format (default support in `Phi3.5 Vision Instruct`).
    
- **Text Description**:  
    A descriptive text of the image content.
