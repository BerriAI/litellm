# Moonshot AI

## Overview

| Property | Details |
|-------|-------|
| Description | Moonshot AI provides large language models including the moonshot-v1 series and kimi models. |
| Provider Route on LiteLLM | `moonshot/` |
| Link to Provider Doc | [Moonshot AI ↗](https://platform.moonshot.ai/) |
| Base URL | `https://api.moonshot.cn/` |
| Supported Operations | [`/chat/completions`](#sample-usage) |

<br />
<br />

https://platform.moonshot.ai/

**We support ALL Moonshot AI models, just set `moonshot/` as a prefix when sending completion requests**

## API Key
```python
# env variable
os.environ['MOONSHOT_API_KEY']
```

## Sample Usage
```python
from litellm import completion
import os

os.environ['MOONSHOT_API_KEY'] = ""
response = completion(
    model="moonshot/moonshot-v1-8k", 
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
)
print(response)
```

## Sample Usage - Streaming
```python
from litellm import completion
import os

os.environ['MOONSHOT_API_KEY'] = ""
response = completion(
    model="moonshot/moonshot-v1-8k", 
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
    stream=True
)

for chunk in response:
    print(chunk)
```

## Moonshot AI Limitations & LiteLLM Handling

LiteLLM automatically handles the following [Moonshot AI limitations](https://platform.moonshot.ai/docs/guide/migrating-from-openai-to-kimi#about-api-compatibility) to provide seamless OpenAI compatibility:

### Temperature Range Limitation
**Limitation**: Moonshot AI only supports temperature range [0, 1] (vs OpenAI's [0, 2])  
**LiteLLM Handling**: Automatically clamps any temperature > 1 to 1

### Temperature + Multiple Outputs Limitation  
**Limitation**: If temperature < 0.3 and n > 1, Moonshot AI raises an exception  
**LiteLLM Handling**: Automatically sets temperature to 0.3 when this condition is detected

### Tool Choice "Required" Not Supported
**Limitation**: Moonshot AI doesn't support `tool_choice="required"`  
**LiteLLM Handling**: Converts this by:
- Adding message: "Please select a tool to handle the current issue."
- Removing the `tool_choice` parameter from the request
