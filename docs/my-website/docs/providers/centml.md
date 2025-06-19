import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# CentML

:::info
**We support ALL CentML models, just set `centml/` as a prefix when sending completion requests**
:::

| Property | Details |
|-------|-------|
| Description | Enterprise-grade inference platform optimized for deploying and scaling large language models efficiently. |
| Provider Route on LiteLLM | `centml/` |
| Provider Doc | [CentML ↗](https://docs.centml.ai/) |
| Supported OpenAI Endpoints | `/chat/completions`, `/completions` |

## Overview

CentML provides an enterprise-grade inference platform optimized for deploying and scaling large language models efficiently. This guide explains how to integrate LiteLLM with CentML.

## API Key
```python
# env variable
os.environ['CENTML_API_KEY']
```

## Sample Usage - Chat Completions
```python
from litellm import completion
import os

os.environ['CENTML_API_KEY'] = ""
response = completion(
    model="centml/meta-llama/Llama-3.3-70B-Instruct", 
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

os.environ['CENTML_API_KEY'] = ""
response = completion(
    model="centml/meta-llama/Llama-3.3-70B-Instruct", 
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
    stream=True
)

for chunk in response:
    print(chunk)
```

## Sample Usage - Text Completions
```python
from litellm import text_completion
import os

os.environ['CENTML_API_KEY'] = ""
response = text_completion(
    model="centml/meta-llama/Llama-3.3-70B-Instruct",
    prompt="The weather today is",
    max_tokens=50
)
print(response)
```

## Function Calling

CentML supports function calling on select models. Here's an example:

```python
from litellm import completion
import os

os.environ['CENTML_API_KEY'] = ""

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country e.g. Paris, France"
                }
            },
            "required": ["location"],
            "additionalProperties": False
        },
        "strict": True
    }
}]

response = completion(
    model="centml/meta-llama/Llama-3.3-70B-Instruct",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. You have access to ONLY get_weather function that provides temperature information for locations."
        },
        {
            "role": "user", 
            "content": "What is the weather like in Paris today?"
        }
    ],
    max_tokens=4096,
    tools=tools,
    tool_choice="auto"
)
print(response)
```

## JSON Schema Response Format

CentML supports structured JSON output on compatible models:

```python
from litellm import completion
import os

os.environ['CENTML_API_KEY'] = ""

schema_json = {
    "title": "WirelessAccessPoint",
    "type": "object",
    "properties": {
        "ssid": {
            "title": "SSID",
            "type": "string"
        },
        "securityProtocol": {
            "title": "SecurityProtocol", 
            "type": "string"
        },
        "bandwidth": {
            "title": "Bandwidth",
            "type": "string"
        }
    },
    "required": ["ssid", "securityProtocol", "bandwidth"]
}

response = completion(
    model="centml/deepseek-ai/DeepSeek-R1",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that answers in JSON."
        },
        {
            "role": "user",
            "content": "The access point's SSID should be 'OfficeNetSecure', it uses WPA2-Enterprise as its security protocol, and it's capable of a bandwidth of up to 1300 Mbps on the 5 GHz band."
        }
    ],
    max_tokens=5000,
    temperature=0,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "schema",
            "schema": schema_json
        }
    }
)
print(response)
```

## Usage with LiteLLM Proxy 

### 1. Set CentML Models on config.yaml

```yaml
model_list:
  - model_name: centml-llama-3-70b
    litellm_params:
      model: centml/meta-llama/Llama-3.3-70B-Instruct
      api_key: "os.environ/CENTML_API_KEY"
  - model_name: centml-deepseek-r1
    litellm_params:
      model: centml/deepseek-ai/DeepSeek-R1
      api_key: "os.environ/CENTML_API_KEY"
```

### 2. Start Proxy 

```
litellm --config config.yaml
```

### 3. Test it

<Tabs>
<TabItem value="Curl" label="Curl Request">

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "centml-llama-3-70b",
      "messages": [
        {
          "role": "user",
          "content": "what llm are you"
        }
      ]
    }
'
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
response = client.chat.completions.create(
    model="centml-llama-3-70b",
    messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ],
)

print(response)
```

</TabItem>
</Tabs>

## Model Capabilities

CentML models have varying support for advanced features. Here's the compatibility matrix:

### JSON ✅ / Function Calling ✅
These models support both JSON schema response format and function calling:
- `centml/meta-llama/Llama-3.3-70B-Instruct`
- `centml/meta-llama/Llama-4-Scout-17B-16E-Instruct`
- `centml/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`
- `centml/meta-llama/Llama-3.2-3B-Instruct`
- `centml/meta-llama/Llama-3.2-11B-Vision-Instruct`

### JSON ✅ / Function Calling ❌
These models support JSON schema response format but not function calling:
- `centml/deepseek-ai/DeepSeek-R1`
- `centml/deepseek-ai/DeepSeek-V3-0324`
- `centml/microsoft/Phi-4-mini-instruct`
- `centml/Qwen/Qwen2.5-VL-7B-Instruct`

### JSON ❌ / Function Calling ❌
These models support neither advanced feature:
- `centml/Qwen/QwQ-32B`
- `centml/meta-llama/Llama-Guard-4-12B`

## Supported Models - ALL CentML Models Supported!

:::info
We support ALL CentML models, just set `centml/` as a prefix when sending completion requests
:::

| Model Name | Function Call |
|------------|---------------|
| Llama-3.3-70B-Instruct | `completion(model="centml/meta-llama/Llama-3.3-70B-Instruct", messages)` |
| Llama-4-Scout-17B-16E-Instruct | `completion(model="centml/meta-llama/Llama-4-Scout-17B-16E-Instruct", messages)` |
| DeepSeek-R1 | `completion(model="centml/deepseek-ai/DeepSeek-R1", messages)` |
| DeepSeek-V3-0324 | `completion(model="centml/deepseek-ai/DeepSeek-V3-0324", messages)` |
| QwQ-32B | `completion(model="centml/Qwen/QwQ-32B", messages)` |
| Phi-4-mini-instruct | `completion(model="centml/microsoft/Phi-4-mini-instruct", messages)` |

## Error Handling

CentML will raise `UnsupportedParamsError` when you try to use unsupported parameters on models that don't support them:

```python
from litellm import completion
from litellm.exceptions import UnsupportedParamsError

try:
    # This will raise an error because QwQ-32B doesn't support function calling
    response = completion(
        model="centml/Qwen/QwQ-32B",
        messages=[{"role": "user", "content": "Hello"}],
        tools=[{"type": "function", "function": {"name": "test"}}]
    )
except UnsupportedParamsError as e:
    print(f"Error: {e}")
```

You can bypass parameter validation by setting `drop_params=True`:

```python
import litellm

# This will work - unsupported params will be dropped
litellm.drop_params = True
response = completion(
    model="centml/Qwen/QwQ-32B",
    messages=[{"role": "user", "content": "Hello"}],
    tools=[{"type": "function", "function": {"name": "test"}}]
)
``` 