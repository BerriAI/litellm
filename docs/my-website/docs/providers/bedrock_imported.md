import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Bedrock Imported Models

Bedrock Imported Models (Deepseek, Deepseek R1, Qwen, OpenAI-compatible models)

### Deepseek R1

This is a separate route, as the chat template is different.

| Property | Details |
|----------|---------|
| Provider Route | `bedrock/deepseek_r1/{model_arn}` |
| Provider Documentation | [Bedrock Imported Models](https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-import-model.html), [Deepseek Bedrock Imported Model](https://aws.amazon.com/blogs/machine-learning/deploy-deepseek-r1-distilled-llama-models-with-amazon-bedrock-custom-model-import/) |

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import os

response = completion(
    model="bedrock/deepseek_r1/arn:aws:bedrock:us-east-1:086734376398:imported-model/r4c4kewx2s0n",  # bedrock/deepseek_r1/{your-model-arn}
    messages=[{"role": "user", "content": "Tell me a joke"}],
)
```

</TabItem>

<TabItem value="proxy" label="Proxy">


**1. Add to config**

```yaml
model_list:
    - model_name: DeepSeek-R1-Distill-Llama-70B
      litellm_params:
        model: bedrock/deepseek_r1/arn:aws:bedrock:us-east-1:086734376398:imported-model/r4c4kewx2s0n

```

**2. Start proxy**

```bash
litellm --config /path/to/config.yaml

# RUNNING at http://0.0.0.0:4000
```

**3. Test it!**

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
      --header 'Authorization: Bearer sk-1234' \
      --header 'Content-Type: application/json' \
      --data '{
            "model": "DeepSeek-R1-Distill-Llama-70B", # 👈 the 'model_name' in config
            "messages": [
                {
                "role": "user",
                "content": "what llm are you"
                }
            ],
        }'
```

</TabItem>
</Tabs>


### Deepseek (not R1)

| Property | Details |
|----------|---------|
| Provider Route | `bedrock/llama/{model_arn}` |
| Provider Documentation | [Bedrock Imported Models](https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-import-model.html), [Deepseek Bedrock Imported Model](https://aws.amazon.com/blogs/machine-learning/deploy-deepseek-r1-distilled-llama-models-with-amazon-bedrock-custom-model-import/) |



Use this route to call Bedrock Imported Models that follow the `llama` Invoke Request / Response spec


<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import os

response = completion(
    model="bedrock/llama/arn:aws:bedrock:us-east-1:086734376398:imported-model/r4c4kewx2s0n",  # bedrock/llama/{your-model-arn}
    messages=[{"role": "user", "content": "Tell me a joke"}],
)
```

</TabItem>

<TabItem value="proxy" label="Proxy">


**1. Add to config**

```yaml
model_list:
    - model_name: DeepSeek-R1-Distill-Llama-70B
      litellm_params:
        model: bedrock/llama/arn:aws:bedrock:us-east-1:086734376398:imported-model/r4c4kewx2s0n

```

**2. Start proxy**

```bash
litellm --config /path/to/config.yaml

# RUNNING at http://0.0.0.0:4000
```

**3. Test it!**

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
      --header 'Authorization: Bearer sk-1234' \
      --header 'Content-Type: application/json' \
      --data '{
            "model": "DeepSeek-R1-Distill-Llama-70B", # 👈 the 'model_name' in config
            "messages": [
                {
                "role": "user",
                "content": "what llm are you"
                }
            ],
        }'
```

</TabItem>
</Tabs>

### Qwen3 Imported Models

| Property | Details |
|----------|---------|
| Provider Route | `bedrock/qwen3/{model_arn}` |
| Provider Documentation | [Bedrock Imported Models](https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-import-model.html), [Qwen3 Models](https://aws.amazon.com/about-aws/whats-new/2025/09/qwen3-models-fully-managed-amazon-bedrock/) |

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import os

response = completion(
    model="bedrock/qwen3/arn:aws:bedrock:us-east-1:086734376398:imported-model/your-qwen3-model",  # bedrock/qwen3/{your-model-arn}
    messages=[{"role": "user", "content": "Tell me a joke"}],
    max_tokens=100,
    temperature=0.7
)
```

</TabItem>

<TabItem value="proxy" label="Proxy">

**1. Add to config**

```yaml
model_list:
    - model_name: Qwen3-32B
      litellm_params:
        model: bedrock/qwen3/arn:aws:bedrock:us-east-1:086734376398:imported-model/your-qwen3-model

```

**2. Start proxy**

```bash
litellm --config /path/to/config.yaml

# RUNNING at http://0.0.0.0:4000
```

**3. Test it!**

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
      --header 'Authorization: Bearer sk-1234' \
      --header 'Content-Type: application/json' \
      --data '{
            "model": "Qwen3-32B", # 👈 the 'model_name' in config
            "messages": [
                {
                "role": "user",
                "content": "what llm are you"
                }
            ],
        }'
```

</TabItem>
</Tabs>

### Qwen2 Imported Models

| Property | Details |
|----------|---------|
| Provider Route | `bedrock/qwen2/{model_arn}` |
| Provider Documentation | [Bedrock Imported Models](https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-import-model.html) |
| Note | Qwen2 and Qwen3 architectures are mostly similar. The main difference is in the response format: Qwen2 uses "text" field while Qwen3 uses "generation" field. |

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import os

response = completion(
    model="bedrock/qwen2/arn:aws:bedrock:us-east-1:086734376398:imported-model/your-qwen2-model",  # bedrock/qwen2/{your-model-arn}
    messages=[{"role": "user", "content": "Tell me a joke"}],
    max_tokens=100,
    temperature=0.7
)
```

</TabItem>

<TabItem value="proxy" label="Proxy">

**1. Add to config**

```yaml
model_list:
    - model_name: Qwen2-72B
      litellm_params:
        model: bedrock/qwen2/arn:aws:bedrock:us-east-1:086734376398:imported-model/your-qwen2-model

```

**2. Start proxy**

```bash
litellm --config /path/to/config.yaml

# RUNNING at http://0.0.0.0:4000
```

**3. Test it!**

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
      --header 'Authorization: Bearer sk-1234' \
      --header 'Content-Type: application/json' \
      --data '{
            "model": "Qwen2-72B", # 👈 the 'model_name' in config
            "messages": [
                {
                "role": "user",
                "content": "what llm are you"
                }
            ],
        }'
```

</TabItem>
</Tabs>

### OpenAI-Compatible Imported Models (Qwen 2.5 VL, etc.)

Use this route for Bedrock imported models that follow the **OpenAI Chat Completions API spec**. This includes models like Qwen 2.5 VL that accept OpenAI-formatted messages with support for vision (images), tool calling, and other OpenAI features.

| Property | Details |
|----------|---------|
| Provider Route | `bedrock/openai/{model_arn}` |
| Provider Documentation | [Bedrock Imported Models](https://docs.aws.amazon.com/bedrock/latest/userguide/model-customization-import-model.html) |
| Supported Features | Vision (images), tool calling, streaming, system messages |

#### LiteLLMSDK Usage

**Basic Usage**

```python
from litellm import completion

response = completion(
    model="bedrock/openai/arn:aws:bedrock:us-east-1:046319184608:imported-model/0m2lasirsp6z",  # bedrock/openai/{your-model-arn}
    messages=[{"role": "user", "content": "Tell me a joke"}],
    max_tokens=300,
    temperature=0.5
)
```

**With Vision (Images)**

```python
import base64
from litellm import completion

# Load and encode image
with open("image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

response = completion(
    model="bedrock/openai/arn:aws:bedrock:us-east-1:046319184608:imported-model/0m2lasirsp6z",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that can analyze images."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                }
            ]
        }
    ],
    max_tokens=300,
    temperature=0.5
)
```

**Comparing Multiple Images**

```python
import base64
from litellm import completion

# Load images
with open("image1.jpg", "rb") as f:
    image1_base64 = base64.b64encode(f.read()).decode("utf-8")
with open("image2.jpg", "rb") as f:
    image2_base64 = base64.b64encode(f.read()).decode("utf-8")

response = completion(
    model="bedrock/openai/arn:aws:bedrock:us-east-1:046319184608:imported-model/0m2lasirsp6z",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that can analyze images."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Spot the difference between these two images?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image1_base64}"}
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image2_base64}"}
                }
            ]
        }
    ],
    max_tokens=300,
    temperature=0.5
)
```

#### LiteLLM Proxy Usage (AI Gateway)

**1. Add to config**

```yaml
model_list:
    - model_name: qwen-25vl-72b
      litellm_params:
        model: bedrock/openai/arn:aws:bedrock:us-east-1:046319184608:imported-model/0m2lasirsp6z
```

**2. Start proxy**

```bash
litellm --config /path/to/config.yaml

# RUNNING at http://0.0.0.0:4000
```

**3. Test it!**

Basic text request:

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
      --header 'Authorization: Bearer sk-1234' \
      --header 'Content-Type: application/json' \
      --data '{
            "model": "qwen-25vl-72b",
            "messages": [
                {
                    "role": "user",
                    "content": "what llm are you"
                }
            ],
            "max_tokens": 300
        }'
```

With vision (image):

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
      --header 'Authorization: Bearer sk-1234' \
      --header 'Content-Type: application/json' \
      --data '{
            "model": "qwen-25vl-72b",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that can analyze images."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZ..."}
                        }
                    ]
                }
            ],
            "max_tokens": 300,
            "temperature": 0.5
        }'
```

### Moonshot Kimi K2 Thinking

Moonshot AI's Kimi K2 Thinking model is now available on Amazon Bedrock. This model features advanced reasoning capabilities with automatic reasoning content extraction.

| Property | Details |
|----------|---------|
| Provider Route | `bedrock/moonshot.kimi-k2-thinking`, `bedrock/invoke/moonshot.kimi-k2-thinking` |
| Provider Documentation | [AWS Bedrock Moonshot Announcement ↗](https://aws.amazon.com/about-aws/whats-new/2025/12/amazon-bedrock-fully-managed-open-weight-models/) |
| Supported Parameters | `temperature`, `max_tokens`, `top_p`, `stream`, `tools`, `tool_choice` |
| Special Features | Reasoning content extraction, Tool calling |

#### Supported Features

- **Reasoning Content Extraction**: Automatically extracts `<reasoning>` tags and returns them as `reasoning_content` (similar to OpenAI's o1 models)
- **Tool Calling**: Full support for function/tool calling with tool responses
- **Streaming**: Both streaming and non-streaming responses
- **System Messages**: System message support

#### Basic Usage

<Tabs>
<TabItem value="sdk" label="SDK">

```python title="Moonshot Kimi K2 SDK Usage" showLineNumbers
from litellm import completion
import os

os.environ["AWS_ACCESS_KEY_ID"] = "your-aws-access-key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "your-aws-secret-key"
os.environ["AWS_REGION_NAME"] = "us-west-2"  # or your preferred region

# Basic completion
response = completion(
    model="bedrock/moonshot.kimi-k2-thinking",  # or bedrock/invoke/moonshot.kimi-k2-thinking
    messages=[
        {"role": "user", "content": "What is 2+2? Think step by step."}
    ],
    temperature=0.7,
    max_tokens=200
)

print(response.choices[0].message.content)

# Access reasoning content if present
if response.choices[0].message.reasoning_content:
    print("Reasoning:", response.choices[0].message.reasoning_content)
```

</TabItem>
<TabItem value="proxy" label="Proxy">

**1. Add to config**

```yaml title="config.yaml" showLineNumbers
model_list:
  - model_name: kimi-k2
    litellm_params:
      model: bedrock/moonshot.kimi-k2-thinking
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: us-west-2
```

**2. Start proxy**

```bash title="Start LiteLLM Proxy" showLineNumbers
litellm --config /path/to/config.yaml

# RUNNING at http://0.0.0.0:4000
```

**3. Test it!**

```bash title="Test Kimi K2 via Proxy" showLineNumbers
curl --location 'http://0.0.0.0:4000/chat/completions' \
  --header 'Authorization: Bearer sk-1234' \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "kimi-k2",
    "messages": [
      {
        "role": "user",
        "content": "What is 2+2? Think step by step."
      }
    ],
    "temperature": 0.7,
    "max_tokens": 200
  }'
```

</TabItem>
</Tabs>

#### Tool Calling Example

```python title="Kimi K2 with Tool Calling" showLineNumbers
from litellm import completion
import os

os.environ["AWS_ACCESS_KEY_ID"] = "your-aws-access-key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "your-aws-secret-key"
os.environ["AWS_REGION_NAME"] = "us-west-2"

# Tool calling example
response = completion(
    model="bedrock/moonshot.kimi-k2-thinking",
    messages=[
        {"role": "user", "content": "What's the weather in Tokyo?"}
    ],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
)

if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    print(f"Tool called: {tool_call.function.name}")
    print(f"Arguments: {tool_call.function.arguments}")
```

#### Streaming Example

```python title="Kimi K2 Streaming" showLineNumbers
from litellm import completion
import os

os.environ["AWS_ACCESS_KEY_ID"] = "your-aws-access-key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "your-aws-secret-key"
os.environ["AWS_REGION_NAME"] = "us-west-2"

response = completion(
    model="bedrock/moonshot.kimi-k2-thinking",
    messages=[
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    stream=True,
    temperature=0.7
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
    
    # Check for reasoning content in streaming
    if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
        print(f"\n[Reasoning: {chunk.choices[0].delta.reasoning_content}]")
```

#### Supported Parameters

| Parameter | Type | Description | Supported |
|-----------|------|-------------|-----------|
| `temperature` | float (0-1) | Controls randomness in output | ✅ |
| `max_tokens` | integer | Maximum tokens to generate | ✅ |
| `top_p` | float | Nucleus sampling parameter | ✅ |
| `stream` | boolean | Enable streaming responses | ✅ |
| `tools` | array | Tool/function definitions | ✅ |
| `tool_choice` | string/object | Tool choice specification | ✅ |
| `stop` | array | Stop sequences | ❌ (Not supported on Bedrock) |