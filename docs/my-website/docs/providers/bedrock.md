import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# AWS Bedrock
ALL Bedrock models (Anthropic, Meta, Deepseek, Mistral, Amazon, etc.) are Supported

| Property | Details |
|-------|-------|
| Description | Amazon Bedrock is a fully managed service that offers a choice of high-performing foundation models (FMs). |
| Provider Route on LiteLLM | `bedrock/`, [`bedrock/converse/`](#set-converse--invoke-route), [`bedrock/invoke/`](#set-invoke-route), [`bedrock/converse_like/`](#calling-via-internal-proxy), [`bedrock/llama/`](#deepseek-not-r1), [`bedrock/deepseek_r1/`](#deepseek-r1) |
| Provider Doc | [Amazon Bedrock ↗](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html) |
| Supported OpenAI Endpoints | `/chat/completions`, `/completions`, `/embeddings`, `/images/generations` |
| Rerank Endpoint | `/rerank` |
| Pass-through Endpoint | [Supported](../pass_through/bedrock.md) |


LiteLLM requires `boto3` to be installed on your system for Bedrock requests
```shell
pip install boto3>=1.28.57
```

:::info

For **Amazon Nova Models**: Bump to v1.53.5+

:::

## Authentication

:::info

LiteLLM uses boto3 to handle authentication. All these options are supported - https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#credentials.

:::
 
LiteLLM supports API key authentication in addition to traditional boto3 authentication methods. For additional API key details, refer to [docs](https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html).

Option 1: use the AWS_BEARER_TOKEN_BEDROCK environment variable 

```bash
export AWS_BEARER_TOKEN_BEDROCK="your-api-key"
```

Option 2: use the api_key parameter to pass in API key for completion, embedding, image_generation API calls.

```python
response = completion(
  model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
  messages=[{ "content": "Hello, how are you?","role": "user"}],
  api_key="your-api-key"
)
```


## Usage

<a target="_blank" href="https://colab.research.google.com/github/BerriAI/litellm/blob/main/cookbook/LiteLLM_Bedrock.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


```python
import os
from litellm import completion

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

response = completion(
  model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
  messages=[{ "content": "Hello, how are you?","role": "user"}]
)
```

## LiteLLM Proxy Usage 

Here's how to call Bedrock with the LiteLLM Proxy Server

### 1. Setup config.yaml

```yaml
model_list:
  - model_name: bedrock-claude-3-5-sonnet
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: os.environ/AWS_REGION_NAME
```

All possible auth params: 

```
aws_access_key_id: Optional[str],
aws_secret_access_key: Optional[str],
aws_session_token: Optional[str],
aws_region_name: Optional[str],
aws_session_name: Optional[str],
aws_profile_name: Optional[str],
aws_role_name: Optional[str],
aws_web_identity_token: Optional[str],
aws_bedrock_runtime_endpoint: Optional[str],
```

### 2. Start the proxy 

```bash
litellm --config /path/to/config.yaml
```
### 3. Test it


<Tabs>
<TabItem value="Curl" label="Curl Request">

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "bedrock-claude-v1",
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
response = client.chat.completions.create(model="bedrock-claude-v1", messages = [
    {
        "role": "user",
        "content": "this is a test request, write a short poem"
    }
])

print(response)

```
</TabItem>
<TabItem value="langchain" label="Langchain">

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage

chat = ChatOpenAI(
    openai_api_base="http://0.0.0.0:4000", # set openai_api_base to the LiteLLM Proxy
    model = "bedrock-claude-v1",
    temperature=0.1
)

messages = [
    SystemMessage(
        content="You are a helpful assistant that im using to make a test request to."
    ),
    HumanMessage(
        content="test from litellm. tell me why it's amazing in 1 sentence"
    ),
]
response = chat(messages)

print(response)
```
</TabItem>
</Tabs>

## Set temperature, top p, etc.

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import os
from litellm import completion

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

response = completion(
  model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
  messages=[{ "content": "Hello, how are you?","role": "user"}],
  temperature=0.7,
  top_p=1
)
```
</TabItem>
<TabItem value="proxy" label="PROXY">

**Set on yaml**

```yaml
model_list:
  - model_name: bedrock-claude-v1
    litellm_params:
      model: bedrock/anthropic.claude-instant-v1
      temperature: <your-temp>
      top_p: <your-top-p>
```

**Set on request**

```python

import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(model="bedrock-claude-v1", messages = [
    {
        "role": "user",
        "content": "this is a test request, write a short poem"
    }
],
temperature=0.7,
top_p=1
)

print(response)

```

</TabItem>
</Tabs>

## Pass provider-specific params 

If you pass a non-openai param to litellm, we'll assume it's provider-specific and send it as a kwarg in the request body. [See more](../completion/input.md#provider-specific-params)

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import os
from litellm import completion

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

response = completion(
  model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
  messages=[{ "content": "Hello, how are you?","role": "user"}],
  top_k=1 # 👈 PROVIDER-SPECIFIC PARAM
)
```
</TabItem>
<TabItem value="proxy" label="PROXY">

**Set on yaml**

```yaml
model_list:
  - model_name: bedrock-claude-v1
    litellm_params:
      model: bedrock/anthropic.claude-instant-v1
      top_k: 1 # 👈 PROVIDER-SPECIFIC PARAM
```

**Set on request**

```python

import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(model="bedrock-claude-v1", messages = [
    {
        "role": "user",
        "content": "this is a test request, write a short poem"
    }
],
temperature=0.7,
extra_body={
    top_k=1 # 👈 PROVIDER-SPECIFIC PARAM
}
)

print(response)

```

</TabItem>
</Tabs>

## Usage - Request Metadata

Attach metadata to Bedrock requests for logging and cost attribution.

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import os
from litellm import completion

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

response = completion(
    model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    requestMetadata={
        "cost_center": "engineering",
        "user_id": "user123"
    }
)
```
</TabItem>
<TabItem value="proxy" label="PROXY">

**Set on yaml**

```yaml
model_list:
  - model_name: bedrock-claude-v1
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0
      requestMetadata:
        cost_center: "engineering"
```

**Set on request**

```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="bedrock-claude-v1",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={
        "requestMetadata": {"cost_center": "engineering"}
    }
)
```

</TabItem>
</Tabs>

## Usage - Function Calling / Tool calling

LiteLLM supports tool calling via Bedrock's Converse and Invoke API's.

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion

# set env
os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]
messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]

response = completion(
    model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)
# Add any assertions, here to check response args
print(response)
assert isinstance(response.choices[0].message.tool_calls[0].function.name, str)
assert isinstance(
    response.choices[0].message.tool_calls[0].function.arguments, str
)
```
</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
model_list:
  - model_name: bedrock-claude-3-7
    litellm_params:
      model: bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0 # for bedrock invoke, specify `bedrock/invoke/<model>`
```

2. Start proxy 

```bash
litellm --config /path/to/config.yaml
```

3. Test it! 

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $LITELLM_API_KEY" \
-d '{
  "model": "bedrock-claude-3-7",
  "messages": [
    {
      "role": "user",
      "content": "What'\''s the weather like in Boston today?"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA"
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
  ],
  "tool_choice": "auto"
}'

```


</TabItem>
</Tabs>


## Usage - Vision 

```python
from litellm import completion

# set env
os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""


def encode_image(image_path):
    import base64

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


image_path = "../proxy/cached_logo.jpg"
# Getting the base64 string
base64_image = encode_image(image_path)
resp = litellm.completion(
    model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Whats in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + base64_image
                    },
                },
            ],
        }
    ],
)
print(f"\nResponse: {resp}")
```


## Usage - 'thinking' / 'reasoning content'

This is currently only supported for Anthropic's Claude 3.7 Sonnet + Deepseek R1 + GPT-OSS models.

Works on v1.61.20+.

Returns 2 new fields in `message` and `delta` object:
- `reasoning_content` - string - The reasoning content of the response
- `thinking_blocks` - list of objects (Anthropic only) - The thinking blocks of the response

Each object has the following fields:
- `type` - Literal["thinking"] - The type of thinking block
- `thinking` - string - The thinking of the response. Also returned in `reasoning_content`
- `signature` - string - A base64 encoded string, returned by Anthropic.

The `signature` is required by Anthropic on subsequent calls, if 'thinking' content is passed in (only required to use `thinking` with tool calling). [Learn more](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#understanding-thinking-blocks)

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion

# set env
os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""


resp = completion(
    model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    reasoning_effort="low",
)

print(resp)
```
</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
model_list:
  - model_name: bedrock-claude-3-7
    litellm_params:
      model: bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0
      reasoning_effort: "low" # 👈 EITHER HERE OR ON REQUEST
```

2. Start proxy 

```bash
litellm --config /path/to/config.yaml
```

3. Test it! 

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR-LITELLM-KEY>" \
  -d '{
    "model": "bedrock-claude-3-7",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "reasoning_effort": "low" # 👈 EITHER HERE OR ON CONFIG.YAML
  }'
```

</TabItem>
</Tabs>


**Expected Response**

Same as [Anthropic API response](../providers/anthropic#usage---thinking--reasoning_content).

```python
{
    "id": "chatcmpl-c661dfd7-7530-49c9-b0cc-d5018ba4727d",
    "created": 1740640366,
    "model": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "object": "chat.completion",
    "system_fingerprint": null,
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "The capital of France is Paris. It's not only the capital city but also the largest city in France, serving as the country's major cultural, economic, and political center.",
                "role": "assistant",
                "tool_calls": null,
                "function_call": null,
                "reasoning_content": "The capital of France is Paris. This is a straightforward factual question.",
                "thinking_blocks": [
                    {
                        "type": "thinking",
                        "thinking": "The capital of France is Paris. This is a straightforward factual question.",
                        "signature": "EqoBCkgIARABGAIiQL2UoU0b1OHYi+yCHpBY7U6FQW8/FcoLewocJQPa2HnmLM+NECy50y44F/kD4SULFXi57buI9fAvyBwtyjlOiO0SDE3+r3spdg6PLOo9PBoMma2ku5OTAoR46j9VIjDRlvNmBvff7YW4WI9oU8XagaOBSxLPxElrhyuxppEn7m6bfT40dqBSTDrfiw4FYB4qEPETTI6TA6wtjGAAqmFqKTo="
                    }
                ]
            }
        }
    ],
    "usage": {
        "completion_tokens": 64,
        "prompt_tokens": 42,
        "total_tokens": 106,
        "completion_tokens_details": null,
        "prompt_tokens_details": null
    }
}
```

### Pass `thinking` to Anthropic models

Same as [Anthropic API response](../providers/anthropic#usage---thinking--reasoning_content).


## Usage - Anthropic Beta Features

LiteLLM supports Anthropic's beta features on AWS Bedrock through the `anthropic-beta` header. This enables access to experimental features like:

- **1M Context Window** - Up to 1 million tokens of context (Claude Sonnet 4)
- **Computer Use Tools** - AI that can interact with computer interfaces
- **Token-Efficient Tools** - More efficient tool usage patterns  
- **Extended Output** - Up to 128K output tokens
- **Enhanced Thinking** - Advanced reasoning capabilities

### Supported Beta Features

| Beta Feature | Header Value | Compatible Models | Description |
|--------------|-------------|------------------|-------------|
| 1M Context Window | `context-1m-2025-08-07` | Claude Sonnet 4 | Enable 1 million token context window |
| Computer Use (Latest) | `computer-use-2025-01-24` | Claude 3.7 Sonnet | Latest computer use tools |
| Computer Use (Legacy) | `computer-use-2024-10-22` | Claude 3.5 Sonnet v2 | Computer use tools for Claude 3.5 |
| Token-Efficient Tools | `token-efficient-tools-2025-02-19` | Claude 3.7 Sonnet | More efficient tool usage |
| Interleaved Thinking | `interleaved-thinking-2025-05-14` | Claude 4 models | Enhanced thinking capabilities |
| Extended Output | `output-128k-2025-02-19` | Claude 3.7 Sonnet | Up to 128K output tokens |
| Developer Thinking | `dev-full-thinking-2025-05-14` | Claude 4 models | Raw thinking mode for developers |

<Tabs>
<TabItem value="sdk" label="SDK">

**Single Beta Feature**

```python
from litellm import completion
import os

# set env
os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

# Use 1M context window with Claude Sonnet 4
response = completion(
    model="bedrock/anthropic.claude-sonnet-4-20250115-v1:0",
    messages=[{"role": "user", "content": "Hello! Testing 1M context window."}],
    max_tokens=100,
    extra_headers={
        "anthropic-beta": "context-1m-2025-08-07"  # 👈 Enable 1M context
    }
)
```

**Multiple Beta Features**

```python
from litellm import completion

# Combine multiple beta features (comma-separated)
response = completion(
    model="bedrock/converse/anthropic.claude-3-5-sonnet-20241022-v2:0",
    messages=[{"role": "user", "content": "Testing multiple beta features"}],
    max_tokens=100,
    extra_headers={
        "anthropic-beta": "computer-use-2024-10-22,context-1m-2025-08-07"
    }
)
```

**Computer Use Tools with Beta Features**

```python
from litellm import completion

# Computer use tools automatically add computer-use-2024-10-22
# You can add additional beta features
response = completion(
    model="bedrock/converse/anthropic.claude-3-5-sonnet-20241022-v2:0",
    messages=[{"role": "user", "content": "Take a screenshot"}],
    tools=[{
        "type": "computer_20241022",
        "name": "computer",
        "display_width_px": 1920,
        "display_height_px": 1080
    }],
    extra_headers={
        "anthropic-beta": "context-1m-2025-08-07"  # Additional beta feature
    }
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

**Set on YAML Config**

```yaml
model_list:
  - model_name: claude-sonnet-4-1m
    litellm_params:
      model: bedrock/anthropic.claude-sonnet-4-20250115-v1:0
      extra_headers:
        anthropic-beta: "context-1m-2025-08-07"  # 👈 Enable 1M context

  - model_name: claude-computer-use
    litellm_params:
      model: bedrock/converse/anthropic.claude-3-5-sonnet-20241022-v2:0
      extra_headers:
        anthropic-beta: "computer-use-2024-10-22,context-1m-2025-08-07"

general_settings:
  forward_client_headers_to_llm_api: true  # 👈 Required for client-side header forwarding
```

**Set on Request**

```python
import openai

client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="claude-sonnet-4-1m",
    messages=[{
        "role": "user", 
        "content": "Testing 1M context window"
    }],
    extra_headers={
        "anthropic-beta": "context-1m-2025-08-07"
    }
)
```

:::info
**For client-side header forwarding**: When using the proxy and sending `anthropic-beta` headers from the client (like the OpenAI SDK), you need to enable `forward_client_headers_to_llm_api: true` in your proxy's `general_settings`. This tells the proxy to extract headers from HTTP requests and forward them to the underlying LLM provider.
:::

</TabItem>
</Tabs>

:::info

Beta features may require special access or permissions in your AWS account. Some features are only available in specific AWS regions. Check the [AWS Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages-request-response.html) for availability and access requirements.

:::


## Usage - Structured Output / JSON mode 

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import os 
from pydantic import BaseModel

# set env
os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

class CalendarEvent(BaseModel):
  name: str
  date: str
  participants: list[str]

class EventsList(BaseModel):
    events: list[CalendarEvent]

response = completion(
  model="bedrock/anthropic.claude-3-7-sonnet-20250219-v1:0", # specify invoke via `bedrock/invoke/anthropic.claude-3-7-sonnet-20250219-v1:0`
  response_format=EventsList,
  messages=[
    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
    {"role": "user", "content": "Who won the world series in 2020?"}
  ],
)
print(response.choices[0].message.content)
```
</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
model_list:
  - model_name: bedrock-claude-3-7
    litellm_params:
      model: bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0 # specify invoke via `bedrock/invoke/<model_name>` 
      aws_access_key_id: os.environ/CUSTOM_AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/CUSTOM_AWS_SECRET_ACCESS_KEY
      aws_region_name: os.environ/CUSTOM_AWS_REGION_NAME
```

2. Start proxy 

```bash
litellm --config /path/to/config.yaml
```

3. Test it!

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -d '{
    "model": "bedrock-claude-3-7",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant designed to output JSON."
      },
      {
        "role": "user",
        "content": "Who won the worlde series in 2020?"
      }
    ],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "math_reasoning",
        "description": "reason about maths",
        "schema": {
          "type": "object",
          "properties": {
            "steps": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "explanation": { "type": "string" },
                  "output": { "type": "string" }
                },
                "required": ["explanation", "output"],
                "additionalProperties": false
              }
            },
            "final_answer": { "type": "string" }
          },
          "required": ["steps", "final_answer"],
          "additionalProperties": false
        },
        "strict": true
      }
    }
  }'
```
</TabItem>
</Tabs>

## Usage - Latency Optimized Inference

Valid from v1.65.1+

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion

response = completion(
    model="bedrock/anthropic.claude-3-7-sonnet-20250219-v1:0",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    performanceConfig={"latency": "optimized"},
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
model_list:
  - model_name: bedrock-claude-3-7
    litellm_params:
      model: bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0
      performanceConfig: {"latency": "optimized"} # 👈 EITHER HERE OR ON REQUEST
```

2. Start proxy 

```bash
litellm --config /path/to/config.yaml
```

3. Test it!

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -d '{
    "model": "bedrock-claude-3-7",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "performanceConfig": {"latency": "optimized"} # 👈 EITHER HERE OR ON CONFIG.YAML
  }'
```

</TabItem>
</Tabs>

## Usage - Bedrock Guardrails

Example of using [Bedrock Guardrails with LiteLLM](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-use-converse-api.html)

### Selective Content Moderation with `guarded_text`

LiteLLM supports selective content moderation using the `guarded_text` content type. This allows you to wrap only specific content that should be moderated by Bedrock Guardrails, rather than evaluating the entire conversation.

**How it works:**
- Content with `type: "guarded_text"` gets automatically wrapped in `guardrailConverseContent` blocks
- Only the wrapped content is evaluated by Bedrock Guardrails
- Regular content with `type: "text"` bypasses guardrail evaluation

:::note
If `guarded_text` is not used, the entire conversation history will be sent to the guardrail for evaluation, which can increase latency and costs.
:::

<Tabs>
<TabItem value="sdk" label="LiteLLM SDK">

```python
from litellm import completion

# set env
os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

response = completion(
    model="anthropic.claude-v2",
    messages=[
        {
            "content": "where do i buy coffee from? ",
            "role": "user",
        }
    ],
    max_tokens=10,
    guardrailConfig={
        "guardrailIdentifier": "ff6ujrregl1q", # The identifier (ID) for the guardrail.
        "guardrailVersion": "DRAFT",           # The version of the guardrail.
        "trace": "disabled",                   # The trace behavior for the guardrail. Can either be "disabled" or "enabled"
    },
)

# Selective guardrail usage with guarded_text - only specific content is evaluated
response_guard = completion(
    model="anthropic.claude-v2",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is the main topic of this legal document?"},
                {"type": "guarded_text", "text": "This      document contains sensitive legal information that should be moderated by guardrails."}
            ]
        }
    ],
    guardrailConfig={
        "guardrailIdentifier": "gr-abc123",
        "guardrailVersion": "DRAFT"
    }
)
```
</TabItem>
<TabItem value="proxy" label="Proxy on request">

```python

import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(model="anthropic.claude-v2", messages = [
    {
        "role": "user",
        "content": "this is a test request, write a short poem"
    }
],
temperature=0.7,
extra_body={
    "guardrailConfig": {
        "guardrailIdentifier": "ff6ujrregl1q", # The identifier (ID) for the guardrail.
        "guardrailVersion": "DRAFT",           # The version of the guardrail.
        "trace": "disabled",                   # The trace behavior for the guardrail. Can either be "disabled" or "enabled"
    },
}
)

print(response)
```
</TabItem>
<TabItem value="proxy-config" label="Proxy on config.yaml">

1. Update config.yaml 

```yaml
model_list:
  - model_name: bedrock-claude-v1
    litellm_params:
      model: bedrock/anthropic.claude-instant-v1
      aws_access_key_id: os.environ/CUSTOM_AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/CUSTOM_AWS_SECRET_ACCESS_KEY
      aws_region_name: os.environ/CUSTOM_AWS_REGION_NAME
      guardrailConfig: {
        "guardrailIdentifier": "ff6ujrregl1q", # The identifier (ID) for the guardrail.
        "guardrailVersion": "DRAFT",           # The version of the guardrail.
        "trace": "disabled",                   # The trace behavior for the guardrail. Can either be "disabled" or "enabled"
    }

```

2. Start proxy 

```bash
litellm --config /path/to/config.yaml
```

3. Test it! 

```python

import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

# request sent to model set on litellm proxy, `litellm --model`
response = client.chat.completions.create(model="bedrock-claude-v1", messages = [
    {
        "role": "user",
        "content": "this is a test request, write a short poem"
    }
],
temperature=0.7
)

# For adding selective guardrail usage with guarded_text
response_guard = client.chat.completions.create(model="bedrock-claude-v1", messages = [
   {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is the main topic of this legal document?"},
                {"type": "guarded_text", "text": "This document contains sensitive legal information that should be moderated by guardrails."}
            ]
  }
],
temperature=0.7
) 

print(response_guard)
```
</TabItem>
</Tabs>

## Usage - "Assistant Pre-fill"

If you're using Anthropic's Claude with Bedrock, you can "put words in Claude's mouth" by including an `assistant` role message as the last item in the `messages` array.

> [!IMPORTANT]
> The returned completion will _**not**_ include your "pre-fill" text, since it is part of the prompt itself. Make sure to prefix Claude's completion with your pre-fill.

```python
import os
from litellm import completion

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

messages = [
    {"role": "user", "content": "How do you say 'Hello' in German? Return your answer as a JSON object, like this:\n\n{ \"Hello\": \"Hallo\" }"},
    {"role": "assistant", "content": "{"},
]
response = completion(model="bedrock/anthropic.claude-v2", messages=messages)
```

### Example prompt sent to Claude

```

Human: How do you say 'Hello' in German? Return your answer as a JSON object, like this:

{ "Hello": "Hallo" }

Assistant: {
```

## Usage - "System" messages
If you're using Anthropic's Claude 2.1 with Bedrock, `system` role messages are properly formatted for you.

```python
import os
from litellm import completion

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

messages = [
    {"role": "system", "content": "You are a snarky assistant."},
    {"role": "user", "content": "How do I boil water?"},
]
response = completion(model="bedrock/anthropic.claude-v2:1", messages=messages)
```

### Example prompt sent to Claude

```
You are a snarky assistant.

Human: How do I boil water?

Assistant:
```



## Usage - Streaming
```python
import os
from litellm import completion

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

response = completion(
  model="bedrock/anthropic.claude-instant-v1",
  messages=[{ "content": "Hello, how are you?","role": "user"}],
  stream=True
)
for chunk in response:
  print(chunk)
```

#### Example Streaming Output Chunk
```json
{
  "choices": [
    {
      "finish_reason": null,
      "index": 0,
      "delta": {
        "content": "ase can appeal the case to a higher federal court. If a higher federal court rules in a way that conflicts with a ruling from a lower federal court or conflicts with a ruling from a higher state court, the parties involved in the case can appeal the case to the Supreme Court. In order to appeal a case to the Sup"
      }
    }
  ],
  "created": null,
  "model": "anthropic.claude-instant-v1",
  "usage": {
    "prompt_tokens": null,
    "completion_tokens": null,
    "total_tokens": null
  }
}
```

## Cross-region inferencing 

LiteLLM supports Bedrock [cross-region inferencing](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) across all [supported bedrock models](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference-support.html).

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion 
import os 


os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""


litellm.set_verbose = True #  👈 SEE RAW REQUEST 

response = completion(
    model="bedrock/us.anthropic.claude-3-haiku-20240307-v1:0",
    messages=messages,
    max_tokens=10,
    temperature=0.1,
)

print("Final Response: {}".format(response))
```

</TabItem>
<TabItem value="proxy" label="PROXY">

#### 1. Setup config.yaml

```yaml
model_list:
  - model_name: bedrock-claude-haiku
    litellm_params:
      model: bedrock/us.anthropic.claude-3-haiku-20240307-v1:0
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: os.environ/AWS_REGION_NAME
```


#### 2. Start the proxy 

```bash
litellm --config /path/to/config.yaml
```

#### 3. Test it


<Tabs>
<TabItem value="Curl" label="Curl Request">

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "bedrock-claude-haiku",
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
response = client.chat.completions.create(model="bedrock-claude-haiku", messages = [
    {
        "role": "user",
        "content": "this is a test request, write a short poem"
    }
])

print(response)

```
</TabItem>
<TabItem value="langchain" label="Langchain">

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage

chat = ChatOpenAI(
    openai_api_base="http://0.0.0.0:4000", # set openai_api_base to the LiteLLM Proxy
    model = "bedrock-claude-haiku",
    temperature=0.1
)

messages = [
    SystemMessage(
        content="You are a helpful assistant that im using to make a test request to."
    ),
    HumanMessage(
        content="test from litellm. tell me why it's amazing in 1 sentence"
    ),
]
response = chat(messages)

print(response)
```

</TabItem>
</Tabs>
</TabItem>
</Tabs>


## Set 'converse' / 'invoke' route 

:::info

Supported from LiteLLM Version `v1.53.5`

:::

LiteLLM defaults to the `invoke` route. LiteLLM uses the `converse` route for Bedrock models that support it.

To explicitly set the route, do `bedrock/converse/<model>` or `bedrock/invoke/<model>`.


E.g. 

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion

completion(model="bedrock/converse/us.amazon.nova-pro-v1:0")
```

</TabItem>
<TabItem value="proxy" label="PROXY">

```yaml
model_list:
  - model_name: bedrock-model
    litellm_params:
      model: bedrock/converse/us.amazon.nova-pro-v1:0
```

</TabItem>
</Tabs>

## Alternate user/assistant messages

Use `user_continue_message` to add a default user message, for cases (e.g. Autogen) where the client might not follow alternating user/assistant messages starting and ending with a user message. 


```yaml
model_list:
  - model_name: "bedrock-claude"
    litellm_params:
      model: "bedrock/anthropic.claude-instant-v1"
      user_continue_message: {"role": "user", "content": "Please continue"}
```

OR 

just set `litellm.modify_params=True` and LiteLLM will automatically handle this with a default user_continue_message.

```yaml
model_list:
  - model_name: "bedrock-claude"
    litellm_params:
      model: "bedrock/anthropic.claude-instant-v1"

litellm_settings:
   modify_params: true
```

Test it! 

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "bedrock-claude",
    "messages": [{"role": "assistant", "content": "Hey, how's it going?"}]
}'
```

## Usage - PDF / Document Understanding

LiteLLM supports Document Understanding for Bedrock models - [AWS Bedrock Docs](https://docs.aws.amazon.com/nova/latest/userguide/modalities-document.html).

:::info

LiteLLM supports ALL Bedrock document types - 

E.g.: "pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md"

You can also pass these as either `image_url` or `base64`

:::

### url 

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm.utils import supports_pdf_input, completion

# set aws credentials
os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""


# pdf url
image_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"

# Download the file
response = requests.get(url)
file_data = response.content

encoded_file = base64.b64encode(file_data).decode("utf-8")

# model
model = "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"

image_content = [
    {"type": "text", "text": "What's this file about?"},
    {
        "type": "file",
        "file": {
            "file_data": f"data:application/pdf;base64,{encoded_file}", # 👈 PDF
        }
    },
]


if not supports_pdf_input(model, None):
    print("Model does not support image input")

response = completion(
    model=model,
    messages=[{"role": "user", "content": image_content}],
)
assert response is not None
```
</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
model_list:
  - model_name: bedrock-model
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: os.environ/AWS_REGION_NAME
```

2. Start the proxy 

```bash
litellm --config /path/to/config.yaml
```

3. Test it! 

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "bedrock-model",
    "messages": [
        {"role": "user", "content": {"type": "text", "text": "What's this file about?"}},
        {
            "type": "file",
            "file": {
                "file_data": f"data:application/pdf;base64,{encoded_file}", # 👈 PDF
            }
        }
    ]
}'
```
</TabItem>
</Tabs>

### base64

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm.utils import supports_pdf_input, completion

# set aws credentials
os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""


# pdf url
image_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
response = requests.get(url)
file_data = response.content

encoded_file = base64.b64encode(file_data).decode("utf-8")
base64_url = f"data:application/pdf;base64,{encoded_file}"

# model
model = "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"

image_content = [
    {"type": "text", "text": "What's this file about?"},
    {
        "type": "image_url",
        "image_url": base64_url, # OR {"url": base64_url}
    },
]


if not supports_pdf_input(model, None):
    print("Model does not support image input")

response = completion(
    model=model,
    messages=[{"role": "user", "content": image_content}],
)
assert response is not None
```
</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
model_list:
  - model_name: bedrock-model
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: os.environ/AWS_REGION_NAME
```

2. Start the proxy 

```bash
litellm --config /path/to/config.yaml
```

3. Test it! 

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "bedrock-model",
    "messages": [
        {"role": "user", "content": {"type": "text", "text": "What's this file about?"}},
        {
            "type": "image_url",
            "image_url": "data:application/pdf;base64,{b64_encoded_file}",
        }
    ]
}'
```
</TabItem>
</Tabs>


## Bedrock Imported Models (Deepseek, Deepseek R1)

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



### OpenAI GPT OSS

| Property | Details |
|----------|---------|
| Provider Route | `bedrock/converse/openai.gpt-oss-20b-1:0`, `bedrock/converse/openai.gpt-oss-120b-1:0` |
| Provider Documentation | [Amazon Bedrock ↗](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html) |

<Tabs>
<TabItem value="sdk" label="SDK">

```python title="GPT OSS SDK Usage" showLineNumbers
from litellm import completion
import os

# Set AWS credentials
os.environ["AWS_ACCESS_KEY_ID"] = "your-aws-access-key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "your-aws-secret-key"
os.environ["AWS_REGION_NAME"] = "us-east-1"

# GPT OSS 20B model
response = completion(
    model="bedrock/converse/openai.gpt-oss-20b-1:0",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
)
print(response.choices[0].message.content)

# GPT OSS 120B model  
response = completion(
    model="bedrock/converse/openai.gpt-oss-120b-1:0",
    messages=[{"role": "user", "content": "Explain machine learning in simple terms"}],
)
print(response.choices[0].message.content)
```

</TabItem>

<TabItem value="proxy" label="Proxy">

**1. Add to config**

```yaml title="config.yaml" showLineNumbers
model_list:
  - model_name: gpt-oss-20b
    litellm_params:
      model: bedrock/converse/openai.gpt-oss-20b-1:0
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: os.environ/AWS_REGION_NAME
      
  - model_name: gpt-oss-120b
    litellm_params:
      model: bedrock/converse/openai.gpt-oss-120b-1:0
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: os.environ/AWS_REGION_NAME
```

**2. Start proxy**

```bash title="Start LiteLLM Proxy" showLineNumbers
litellm --config /path/to/config.yaml

# RUNNING at http://0.0.0.0:4000
```

**3. Test it!**

```bash title="Test GPT OSS via Proxy" showLineNumbers
curl --location 'http://0.0.0.0:4000/chat/completions' \
  --header 'Authorization: Bearer sk-1234' \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "gpt-oss-20b",
    "messages": [
      {
        "role": "user", 
        "content": "What are the key benefits of open source AI?"
      }
    ]
  }'
```

</TabItem>
</Tabs>

## Provisioned throughput models
To use provisioned throughput Bedrock models pass 
- `model=bedrock/<base-model>`, example `model=bedrock/anthropic.claude-v2`. Set `model` to any of the [Supported AWS models](#supported-aws-bedrock-models)
- `model_id=provisioned-model-arn` 

Completion
```python
import litellm
response = litellm.completion(
    model="bedrock/anthropic.claude-instant-v1",
    model_id="provisioned-model-arn",
    messages=[{"content": "Hello, how are you?", "role": "user"}]
)
```

Embedding
```python
import litellm
response = litellm.embedding(
    model="bedrock/amazon.titan-embed-text-v1",
    model_id="provisioned-model-arn",
    input=["hi"],
)
```


## Supported AWS Bedrock Models

LiteLLM supports ALL Bedrock models. 

Here's an example of using a bedrock model with LiteLLM. For a complete list, refer to the [model cost map](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json)

| Model Name                 | Command                                                          |
|----------------------------|------------------------------------------------------------------|
| GPT-OSS 20B | `completion(model='bedrock/converse/openai.gpt-oss-20b-1:0', messages=messages)` | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']` |
| GPT-OSS 120B | `completion(model='bedrock/converse/openai.gpt-oss-120b-1:0', messages=messages)` | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']` |
| Deepseek R1    | `completion(model='bedrock/us.deepseek.r1-v1:0', messages=messages)`   | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`           |
| Anthropic Claude-V3.5 Sonnet    | `completion(model='bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0', messages=messages)`   | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`           |
| Anthropic Claude-V3  sonnet    | `completion(model='bedrock/anthropic.claude-3-sonnet-20240229-v1:0', messages=messages)`   | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`           |
| Anthropic Claude-V3 Haiku     | `completion(model='bedrock/anthropic.claude-3-haiku-20240307-v1:0', messages=messages)`   | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`           |
| Anthropic Claude-V3 Opus     | `completion(model='bedrock/anthropic.claude-3-opus-20240229-v1:0', messages=messages)`   | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`           |
| Anthropic Claude-V2.1      | `completion(model='bedrock/anthropic.claude-v2:1', messages=messages)`   | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`           |
| Anthropic Claude-V2        | `completion(model='bedrock/anthropic.claude-v2', messages=messages)`   | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`           |
| Anthropic Claude-Instant V1 | `completion(model='bedrock/anthropic.claude-instant-v1', messages=messages)` | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`           |
| Meta llama3-1-405b        | `completion(model='bedrock/meta.llama3-1-405b-instruct-v1:0', messages=messages)`   | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`           |
| Meta llama3-1-70b        | `completion(model='bedrock/meta.llama3-1-70b-instruct-v1:0', messages=messages)`   | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`           |
| Meta llama3-1-8b        | `completion(model='bedrock/meta.llama3-1-8b-instruct-v1:0', messages=messages)`   | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`           |
| Meta llama3-70b        | `completion(model='bedrock/meta.llama3-70b-instruct-v1:0', messages=messages)`   | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`           |
| Meta llama3-8b | `completion(model='bedrock/meta.llama3-8b-instruct-v1:0', messages=messages)` | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`           |
| Amazon Titan Lite          | `completion(model='bedrock/amazon.titan-text-lite-v1', messages=messages)` | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']` |
| Amazon Titan Express       | `completion(model='bedrock/amazon.titan-text-express-v1', messages=messages)`   | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']` |
| Cohere Command             | `completion(model='bedrock/cohere.command-text-v14', messages=messages)`   | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']` |
| AI21 J2-Mid                | `completion(model='bedrock/ai21.j2-mid-v1', messages=messages)`   | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']` |
| AI21 J2-Ultra              | `completion(model='bedrock/ai21.j2-ultra-v1', messages=messages)`   | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']` |
| AI21 Jamba-Instruct              | `completion(model='bedrock/ai21.jamba-instruct-v1:0', messages=messages)`   | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']` |
| Meta Llama 2 Chat 13b      | `completion(model='bedrock/meta.llama2-13b-chat-v1', messages=messages)`   | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']` |
| Meta Llama 2 Chat 70b      | `completion(model='bedrock/meta.llama2-70b-chat-v1', messages=messages)`   | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']` |
| Mistral 7B Instruct        | `completion(model='bedrock/mistral.mistral-7b-instruct-v0:2', messages=messages)`   | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']` |
| Mixtral 8x7B Instruct      | `completion(model='bedrock/mistral.mixtral-8x7b-instruct-v0:1', messages=messages)`   | `os.environ['AWS_ACCESS_KEY_ID']`, `os.environ['AWS_SECRET_ACCESS_KEY']`, `os.environ['AWS_REGION_NAME']` |


## Bedrock Embedding

### API keys
This can be set as env variables or passed as **params to litellm.embedding()**
```python
import os
os.environ["AWS_ACCESS_KEY_ID"] = ""        # Access key
os.environ["AWS_SECRET_ACCESS_KEY"] = ""    # Secret access key
os.environ["AWS_REGION_NAME"] = ""           # us-east-1, us-east-2, us-west-1, us-west-2
```

### Usage
```python
from litellm import embedding
response = embedding(
    model="bedrock/amazon.titan-embed-text-v1",
    input=["good morning from litellm"],
)
print(response)
```

#### Titan V2 - encoding_format support
```python
from litellm import embedding
# Float format (default)
response = embedding(
    model="bedrock/amazon.titan-embed-text-v2:0",
    input=["good morning from litellm"],
    encoding_format="float"  # Returns float array
)

# Binary format
response = embedding(
    model="bedrock/amazon.titan-embed-text-v2:0",
    input=["good morning from litellm"],
    encoding_format="base64"  # Returns base64 encoded binary
)
```

## Supported AWS Bedrock Embedding Models

| Model Name           | Usage                               | Supported Additional OpenAI params |
|----------------------|---------------------------------------------|-----|
| Titan Embeddings V2 | `embedding(model="bedrock/amazon.titan-embed-text-v2:0", input=input)` | `dimensions`, `encoding_format` |
| Titan Embeddings - V1 | `embedding(model="bedrock/amazon.titan-embed-text-v1", input=input)` | [here](https://github.com/BerriAI/litellm/blob/f5905e100068e7a4d61441d7453d7cf5609c2121/litellm/llms/bedrock/embed/amazon_titan_g1_transformation.py#L53)
| Titan Multimodal Embeddings | `embedding(model="bedrock/amazon.titan-embed-image-v1", input=input)` | [here](https://github.com/BerriAI/litellm/blob/f5905e100068e7a4d61441d7453d7cf5609c2121/litellm/llms/bedrock/embed/amazon_titan_multimodal_transformation.py#L28) |
| Cohere Embeddings - English | `embedding(model="bedrock/cohere.embed-english-v3", input=input)` | [here](https://github.com/BerriAI/litellm/blob/f5905e100068e7a4d61441d7453d7cf5609c2121/litellm/llms/bedrock/embed/cohere_transformation.py#L18)
| Cohere Embeddings - Multilingual | `embedding(model="bedrock/cohere.embed-multilingual-v3", input=input)` | [here](https://github.com/BerriAI/litellm/blob/f5905e100068e7a4d61441d7453d7cf5609c2121/litellm/llms/bedrock/embed/cohere_transformation.py#L18)

### Advanced - [Drop Unsupported Params](https://docs.litellm.ai/docs/completion/drop_params#openai-proxy-usage)

### Advanced - [Pass model/provider-specific Params](https://docs.litellm.ai/docs/completion/provider_specific_params#proxy-usage)

## Image Generation
Use this for stable diffusion, and amazon nova canvas on bedrock


### Usage

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import os
from litellm import image_generation

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

response = image_generation(
            prompt="A cute baby sea otter",
            model="bedrock/stability.stable-diffusion-xl-v0",
        )
print(f"response: {response}")
```

**Set optional params**
```python
import os
from litellm import image_generation

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

response = image_generation(
            prompt="A cute baby sea otter",
            model="bedrock/stability.stable-diffusion-xl-v0",
            ### OPENAI-COMPATIBLE ###
            size="128x512", # width=128, height=512
            ### PROVIDER-SPECIFIC ### see `AmazonStabilityConfig` in bedrock.py for all params
            seed=30
        )
print(f"response: {response}")
```
</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
model_list:
  - model_name: amazon.nova-canvas-v1:0
    litellm_params:
      model: bedrock/amazon.nova-canvas-v1:0
      aws_region_name: "us-east-1"
      aws_secret_access_key: my-key # OPTIONAL - all boto3 auth params supported
      aws_secret_access_id: my-id # OPTIONAL - all boto3 auth params supported
```

2. Start proxy 

```bash
litellm --config /path/to/config.yaml
```

3. Test it! 

```bash
curl -L -X POST 'http://0.0.0.0:4000/v1/images/generations' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer $LITELLM_VIRTUAL_KEY' \
-d '{
    "model": "amazon.nova-canvas-v1:0",
    "prompt": "A cute baby sea otter"
}'
```

</TabItem>
</Tabs>

### Using Inference Profiles with Image Generation

For AWS Bedrock Application Inference Profiles with image generation, use the `model_id` parameter to specify the inference profile ARN:

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import image_generation

response = image_generation(
    model="bedrock/amazon.nova-canvas-v1:0",
    model_id="arn:aws:bedrock:eu-west-1:000000000000:application-inference-profile/a0a0a0a0a0a0",
    prompt="A cute baby sea otter"
)
print(f"response: {response}")
```

</TabItem>
<TabItem value="proxy" label="PROXY">

```yaml
model_list:
  - model_name: nova-canvas-inference-profile
    litellm_params:
      model: bedrock/amazon.nova-canvas-v1:0
      model_id: arn:aws:bedrock:eu-west-1:000000000000:application-inference-profile/a0a0a0a0a0a0
      aws_region_name: "eu-west-1"
```

</TabItem>
</Tabs>

## Supported AWS Bedrock Image Generation Models

| Model Name           | Function Call                               |
|----------------------|---------------------------------------------|
| Stable Diffusion 3 - v0 | `embedding(model="bedrock/stability.stability.sd3-large-v1:0", prompt=prompt)` |
| Stable Diffusion - v0 | `embedding(model="bedrock/stability.stable-diffusion-xl-v0", prompt=prompt)` |
| Stable Diffusion - v0 | `embedding(model="bedrock/stability.stable-diffusion-xl-v1", prompt=prompt)` |


## Rerank API 

Use Bedrock's Rerank API in the Cohere `/rerank` format. 

Supported Cohere Rerank Params
- `model` - the foundation model ARN
- `query` - the query to rerank against
- `documents` - the list of documents to rerank
- `top_n` - the number of results to return

<Tabs>
<TabItem label="SDK" value="sdk">

```python
from litellm import rerank
import os 

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

response = rerank(
    model="bedrock/arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0", # provide the model ARN - get this here https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock/client/list_foundation_models.html
    query="hello",
    documents=["hello", "world"],
    top_n=2,
)

print(response)
```

</TabItem>
<TabItem label="PROXY" value="proxy">

1. Setup config.yaml

```yaml
model_list:
    - model_name: bedrock-rerank
      litellm_params:
        model: bedrock/arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0
        aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
        aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
        aws_region_name: os.environ/AWS_REGION_NAME
```

2. Start proxy server

```bash
litellm --config config.yaml

# RUNNING on http://0.0.0.0:4000
```

3. Test it! 

```bash
curl http://0.0.0.0:4000/rerank \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bedrock-rerank",
    "query": "What is the capital of the United States?",
    "documents": [
        "Carson City is the capital city of the American state of Nevada.",
        "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
        "Washington, D.C. is the capital of the United States.",
        "Capital punishment has existed in the United States since before it was a country."
    ],
    "top_n": 3


  }'
```

</TabItem>
</Tabs>


## Bedrock Application Inference Profile 

Use Bedrock Application Inference Profile to track costs for projects on AWS. 

You can either pass it in the model name - `model="bedrock/arn:...` or as a separate `model_id="arn:..` param.

### Set via `model_id` 

<Tabs>
<TabItem label="SDK" value="sdk">

```python
from litellm import completion
import os 

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

response = completion(
    model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    model_id="arn:aws:bedrock:eu-central-1:000000000000:application-inference-profile/a0a0a0a0a0a0",
)

print(response)
```

</TabItem>
<TabItem label="PROXY" value="proxy">

1. Setup config.yaml 

```yaml
model_list:
  - model_name: anthropic-claude-3-5-sonnet
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0
      # You have to set the ARN application inference profile in the model_id parameter
      model_id: arn:aws:bedrock:eu-central-1:000000000000:application-inference-profile/a0a0a0a0a0a0
```

2. Start proxy

```bash
litellm --config /path/to/config.yaml
```

3. Test it! 

```bash
curl -L -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer $LITELLM_API_KEY' \
-d '{
  "model": "anthropic-claude-3-5-sonnet",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "List 5 important events in the XIX century"
        }
      ]
    }
  ]
}'
```

</TabItem>
</Tabs>

## Boto3 - Authentication

### Passing credentials as parameters - Completion()
Pass AWS credentials as parameters to litellm.completion
```python
import os
from litellm import completion

response = completion(
            model="bedrock/anthropic.claude-instant-v1",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
            aws_access_key_id="",
            aws_secret_access_key="",
            aws_region_name="",
)
```

### Passing extra headers + Custom API Endpoints

This can be used to override existing headers (e.g. `Authorization`) when calling custom api endpoints

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import os
import litellm
from litellm import completion

litellm.set_verbose = True # 👈 SEE RAW REQUEST

response = completion(
            model="bedrock/anthropic.claude-instant-v1",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
            aws_access_key_id="",
            aws_secret_access_key="",
            aws_region_name="",
            aws_bedrock_runtime_endpoint="https://my-fake-endpoint.com",
            extra_headers={"key": "value"}
)
```
</TabItem>

<TabItem value="proxy" label="PROXY">

1. Setup config.yaml 

```yaml
model_list:
    - model_name: bedrock-model
      litellm_params:
        model: bedrock/anthropic.claude-instant-v1
        aws_access_key_id: "",
        aws_secret_access_key: "",
        aws_region_name: "",
        aws_bedrock_runtime_endpoint: "https://my-fake-endpoint.com",
        extra_headers: {"key": "value"}
```

2. Start proxy 

```bash
litellm --config /path/to/config.yaml --detailed_debug
```

3. Test it! 

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "bedrock-model",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful math tutor. Guide the user through the solution step by step."
      },
      {
        "role": "user",
        "content": "how can I solve 8x + 7 = -23"
      }
    ]
}'
```

</TabItem>

</Tabs>

### SSO Login (AWS Profile)
- Set `AWS_PROFILE` environment variable
- Make bedrock completion call

```python
import os
from litellm import completion

response = completion(
            model="bedrock/anthropic.claude-instant-v1",
            messages=[{ "content": "Hello, how are you?","role": "user"}]
)
```

or pass `aws_profile_name`:

```python
import os
from litellm import completion

response = completion(
            model="bedrock/anthropic.claude-instant-v1",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
            aws_profile_name="dev-profile",
)
```

### STS (Role-based Auth)

- Set `aws_role_name` and `aws_session_name`


| LiteLLM Parameter | Boto3 Parameter | Description | Boto3 Documentation |
|------------------|-----------------|-------------|-------------------|
| `aws_access_key_id` | `aws_access_key_id` | AWS access key associated with an IAM user or role | [Credentials](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html) |
| `aws_secret_access_key` | `aws_secret_access_key` | AWS secret key associated with the access key | [Credentials](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html) |
| `aws_role_name` | `RoleArn` | The Amazon Resource Name (ARN) of the role to assume | [AssumeRole API](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sts.html#STS.Client.assume_role) |
| `aws_session_name` | `RoleSessionName` | An identifier for the assumed role session | [AssumeRole API](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sts.html#STS.Client.assume_role) |



Make the bedrock completion call

---

### Required AWS IAM Policy for AssumeRole

To use `aws_role_name` (STS AssumeRole) with LiteLLM, your IAM user or role **must** have permission to call `sts:AssumeRole` on the target role. If you see an error like:

```
An error occurred (AccessDenied) when calling the AssumeRole operation: User: arn:aws:sts::...:assumed-role/litellm-ecs-task-role/... is not authorized to perform: sts:AssumeRole on resource: arn:aws:iam::...:role/Enterprise/BedrockCrossAccountConsumer
```

This means the IAM identity running LiteLLM does **not** have permission to assume the target role. You must update your IAM policy to allow this action.

#### Example IAM Policy

Replace `<TARGET_ROLE_ARN>` with the ARN of the role you want to assume (e.g., `arn:aws:iam::123456789012:role/Enterprise/BedrockCrossAccountConsumer`).

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "sts:AssumeRole",
      "Resource": "<TARGET_ROLE_ARN>"
    }
  ]
}
```

**Note:** The target role itself must also trust the calling IAM identity (via its trust policy) for AssumeRole to succeed. See [AWS AssumeRole docs](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_use_switch-role-api.html) for more details.

---

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion

response = completion(
            model="bedrock/anthropic.claude-instant-v1",
            messages=messages,
            max_tokens=10,
            temperature=0.1,
            aws_role_name=aws_role_name,
            aws_session_name="my-test-session",
        )
```

If you also need to dynamically set the aws user accessing the role, add the additional args in the completion()/embedding() function

```python
from litellm import completion

response = completion(
            model="bedrock/anthropic.claude-instant-v1",
            messages=messages,
            max_tokens=10,
            temperature=0.1,
            aws_region_name=aws_region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_role_name=aws_role_name,
            aws_session_name="my-test-session",
        )
```
</TabItem>

<TabItem value="proxy" label="PROXY">

```yaml
model_list:
  - model_name: bedrock/*
    litellm_params:
      model: bedrock/*
      aws_role_name: arn:aws:iam::888602223428:role/iam_local_role # AWS RoleArn
      aws_session_name: "bedrock-session" # AWS RoleSessionName
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID # [OPTIONAL - not required if using role]
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY # [OPTIONAL - not required if using role]
```


</TabItem>

</Tabs>

Text to Image : 
```bash
curl -L -X POST 'http://0.0.0.0:4000/v1/images/generations' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer $LITELLM_VIRTUAL_KEY' \
-d '{
    "model": "amazon.nova-canvas-v1:0",
    "prompt": "A cute baby sea otter"
}'
```

Color Guided Generation:
```bash
curl -L -X POST 'http://0.0.0.0:4000/v1/images/generations' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer $LITELLM_VIRTUAL_KEY' \
-d '{
    "model": "amazon.nova-canvas-v1:0",
    "prompt": "A cute baby sea otter",
    "taskType": "COLOR_GUIDED_GENERATION",
    "colorGuidedGenerationParams":{"colors":["#FFFFFF"]}
}'
```

| Model Name              | Function Call                               |
|-------------------------|---------------------------------------------|
| Stable Diffusion 3 - v0 | `image_generation(model="bedrock/stability.stability.sd3-large-v1:0", prompt=prompt)` |
| Stable Diffusion - v0   | `image_generation(model="bedrock/stability.stable-diffusion-xl-v0", prompt=prompt)` |
| Stable Diffusion - v1   | `image_generation(model="bedrock/stability.stable-diffusion-xl-v1", prompt=prompt)` |
| Amazon Nova Canvas - v0 | `image_generation(model="bedrock/amazon.nova-canvas-v1:0", prompt=prompt)` |
  
  
### Passing an external BedrockRuntime.Client as a parameter - Completion()
  
This is a deprecated flow. Boto3 is not async. And boto3.client does not let us make the http call through httpx. Pass in your aws params through the method above 👆. [See Auth Code](https://github.com/BerriAI/litellm/blob/55a20c7cce99a93d36a82bf3ae90ba3baf9a7f89/litellm/llms/bedrock_httpx.py#L284) [Add new auth flow](https://github.com/BerriAI/litellm/issues)

:::warning





Experimental - 2024-Jun-23:
    `aws_access_key_id`, `aws_secret_access_key`, and `aws_session_token` will be extracted from boto3.client and be passed into the httpx client 

:::

Pass an external BedrockRuntime.Client object as a parameter to litellm.completion. Useful when using an AWS credentials profile, SSO session, assumed role session, or if environment variables are not available for auth.

Create a client from session credentials:
```python
import boto3
from litellm import completion

bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id="",
            aws_secret_access_key="",
            aws_session_token="",
)

response = completion(
            model="bedrock/anthropic.claude-instant-v1",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
            aws_bedrock_client=bedrock,
)
```

Create a client from AWS profile in `~/.aws/config`:
```python
import boto3
from litellm import completion

dev_session = boto3.Session(profile_name="dev-profile")
bedrock = dev_session.client(
            service_name="bedrock-runtime",
            region_name="us-east-1",
)

response = completion(
            model="bedrock/anthropic.claude-instant-v1",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
            aws_bedrock_client=bedrock,
)
```
## Calling via Internal Proxy (not bedrock url compatible)

Use the `bedrock/converse_like/model` endpoint to call bedrock converse model via your internal proxy.

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion

response = completion(
    model="bedrock/converse_like/some-model",
    messages=[{"role": "user", "content": "What's AWS?"}],
    api_key="sk-1234",
    api_base="https://some-api-url/models",
    extra_headers={"test": "hello world"},
)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

1. Setup config.yaml

```yaml
model_list:
    - model_name: anthropic-claude
      litellm_params:
        model: bedrock/converse_like/some-model
        api_base: https://some-api-url/models
```

2. Start proxy server

```bash
litellm --config config.yaml

# RUNNING on http://0.0.0.0:4000
```

3. Test it! 

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "model": "anthropic-claude",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful math tutor. Guide the user through the solution step by step."
      },
      { "content": "Hello, how are you?", "role": "user" }
    ]
}'
```

</TabItem>
</Tabs>

**Expected Output URL**

```bash
https://some-api-url/models
```
