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
                                "url": "https://awsmp-logos.s3.amazonaws.com/seller-xw5kijmvmzasy/c233c9ade2ccb5491072ae232c814942.png"
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
                                "url": "https://awsmp-logos.s3.amazonaws.com/seller-xw5kijmvmzasy/c233c9ade2ccb5491072ae232c814942.png"
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
assert litellm.supports_vision(model="openai/gpt-4-vision-preview") == True
assert litellm.supports_vision(model="vertex_ai/gemini-1.0-pro-vision") == True
assert litellm.supports_vision(model="openai/gpt-3.5-turbo") == False
assert litellm.supports_vision(model="xai/grok-2-vision-latest") == True
assert litellm.supports_vision(model="xai/grok-2-latest") == False
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


## Explicitly specify image type 

If you have images without a mime-type, or if litellm is incorrectly inferring the mime type of your image (e.g. calling `gs://` url's with vertex ai), you can set this explicitly via the `format` param. 

```python
"image_url": {
  "url": "gs://my-gs-image",
  "format": "image/jpeg"
}
```

LiteLLM will use this for any API endpoint, which supports specifying mime-type (e.g. anthropic/bedrock/vertex ai). 

For others (e.g. openai), it will be ignored. 

<Tabs>
<TabItem label="SDK" value="sdk">

```python
import os 
from litellm import completion

os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

# openai call
response = completion(
    model = "claude-3-7-sonnet-latest", 
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
                                  "url": "https://awsmp-logos.s3.amazonaws.com/seller-xw5kijmvmzasy/c233c9ade2ccb5491072ae232c814942.png",
                                  "format": "image/jpeg"
                                }
                            }
                        ]
        }
    ],
)

```

</TabItem>
<TabItem label="PROXY" value="proxy">

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
                                "url": "https://awsmp-logos.s3.amazonaws.com/seller-xw5kijmvmzasy/c233c9ade2ccb5491072ae232c814942.png",
                                "format": "image/jpeg"
                                }
                            }
                        ]
        }
    ],
)

```




</TabItem>
</Tabs>



## Spec 

```
"image_url": str

OR 

"image_url": {
  "url": "url OR base64 encoded str",
  "detail": "openai-only param", 
  "format": "specify mime-type of image"
}
```