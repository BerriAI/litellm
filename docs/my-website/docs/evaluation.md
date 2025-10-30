import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Evaluation - Bedrock Models

This guide shows you how to make completion calls to AWS Bedrock models using LiteLLM.

## Quick Start

### Installation

Install LiteLLM with Bedrock support:

```bash
pip install litellm boto3>=1.28.57
```

### Authentication

Set your AWS credentials as environment variables:

```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_REGION_NAME="us-east-1"
```

## Basic Completion Call

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion

# Basic completion call
response = completion(
    model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    messages=[{"role": "user", "content": "What is machine learning?"}]
)

print(response.choices[0].message.content)
```

</TabItem>
<TabItem value="proxy" label="Proxy">

**1. Setup config.yaml**

```yaml
model_list:
  - model_name: claude-sonnet
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: os.environ/AWS_REGION_NAME
```

**2. Start proxy**

```bash
litellm --config /path/to/config.yaml
```

**3. Test it**

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer sk-1234' \
--data '{
    "model": "claude-sonnet",
    "messages": [
        {
            "role": "user",
            "content": "What is machine learning?"
        }
    ]
}'
```

</TabItem>
</Tabs>

## Streaming Responses

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion

response = completion(
    model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    messages=[{"role": "user", "content": "Explain neural networks"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

</TabItem>
<TabItem value="proxy" label="Proxy">

```python
import openai

client = openai.OpenAI(
    api_key="sk-1234",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="claude-sonnet",
    messages=[{"role": "user", "content": "Explain neural networks"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

</TabItem>
</Tabs>

## Setting Parameters

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion

response = completion(
    model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    messages=[{"role": "user", "content": "Write a haiku about AI"}],
    temperature=0.7,
    max_tokens=100,
    top_p=0.9
)

print(response.choices[0].message.content)
```

</TabItem>
<TabItem value="proxy" label="Proxy">

```python
import openai

client = openai.OpenAI(
    api_key="sk-1234",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="claude-sonnet",
    messages=[{"role": "user", "content": "Write a haiku about AI"}],
    temperature=0.7,
    max_tokens=100,
    top_p=0.9
)

print(response.choices[0].message.content)
```

</TabItem>
</Tabs>

## Function Calling

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    }
]

response = completion(
    model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    tool_choice="auto"
)

print(response.choices[0].message.tool_calls)
```

</TabItem>
<TabItem value="proxy" label="Proxy">

```python
import openai

client = openai.OpenAI(
    api_key="sk-1234",
    base_url="http://0.0.0.0:4000"
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="claude-sonnet",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    tool_choice="auto"
)

print(response.choices[0].message.tool_calls)
```

</TabItem>
</Tabs>

## Structured Output / JSON Mode

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
from pydantic import BaseModel

class WeatherInfo(BaseModel):
    location: str
    temperature: float
    condition: str

response = completion(
    model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    messages=[{"role": "user", "content": "What's the weather in Tokyo? Temperature is 22°C and sunny."}],
    response_format=WeatherInfo
)

print(response.choices[0].message.content)
```

</TabItem>
<TabItem value="proxy" label="Proxy">

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "claude-sonnet",
    "messages": [
      {
        "role": "user",
        "content": "What is the weather in Tokyo? Temperature is 22°C and sunny."
      }
    ],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "weather_info",
        "schema": {
          "type": "object",
          "properties": {
            "location": {"type": "string"},
            "temperature": {"type": "number"},
            "condition": {"type": "string"}
          },
          "required": ["location", "temperature", "condition"]
        }
      }
    }
  }'
```

</TabItem>
</Tabs>

## Vision / Image Input

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import base64

# Encode image to base64
with open("image.jpg", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

response = completion(
    model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
```

</TabItem>
<TabItem value="proxy" label="Proxy">

```python
import openai
import base64

client = openai.OpenAI(
    api_key="sk-1234",
    base_url="http://0.0.0.0:4000"
)

with open("image.jpg", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

response = client.chat.completions.create(
    model="claude-sonnet",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
```

</TabItem>
</Tabs>

## Supported Bedrock Models

| Model | Model Name |
|-------|------------|
| Claude Sonnet 4.5 | `bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0` |
| Claude 3.5 Sonnet | `bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0` |
| Claude 3 Sonnet | `bedrock/anthropic.claude-3-sonnet-20240229-v1:0` |
| Claude 3 Haiku | `bedrock/anthropic.claude-3-haiku-20240307-v1:0` |
| Claude 3 Opus | `bedrock/anthropic.claude-3-opus-20240229-v1:0` |
| Llama 3.1 405B | `bedrock/meta.llama3-1-405b-instruct-v1:0` |
| Llama 3.1 70B | `bedrock/meta.llama3-1-70b-instruct-v1:0` |
| Llama 3.1 8B | `bedrock/meta.llama3-1-8b-instruct-v1:0` |
| Mistral 7B | `bedrock/mistral.mistral-7b-instruct-v0:2` |
| Amazon Nova Pro | `bedrock/us.amazon.nova-pro-v1:0` |
| Amazon Nova Lite | `bedrock/us.amazon.nova-lite-v1:0` |

For a complete list of supported models, see the [Bedrock provider documentation](./providers/bedrock.md).

## Next Steps

- [Advanced Bedrock Features](./providers/bedrock.md) - Cross-region inference, guardrails, reasoning content
- [Bedrock Embedding](./providers/bedrock_embedding.md) - Text embeddings
- [Bedrock Batches](./providers/bedrock_batches.md) - Batch processing
- [Proxy Configuration](./proxy/quick_start.md) - Running LiteLLM proxy server
