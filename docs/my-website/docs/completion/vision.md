# Using Vision Models

## Quick Start
Example passing images to a model 

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

## Checking if a model supports `vision`

Use `litellm.supports_vision(model="")` -> returns `True` if model supports `vision` and `False` if not

```python
assert litellm.supports_vision(model="gpt-4-vision-preview") == True
assert litellm.supports_vision(model="gemini-1.0-pro-visionn") == True
assert litellm.supports_vision(model="gpt-3.5-turbo") == False
```

