---
sidebar_label: Snowflake Cortex
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Snowflake Cortex

LiteLLM supports all models on the Snowflake Cortex REST API, including models from Anthropic (Claude), OpenAI (GPT), Meta (Llama), Mistral, DeepSeek, and Snowflake.

| | |
|---|---|
| Description | Snowflake Cortex REST API provides access to leading frontier LLMs through OpenAI-compatible and Anthropic-compatible endpoints. All inference runs within Snowflake's security perimeter. |
| Provider Route on LiteLLM | `snowflake/` |
| Provider Docs | [Cortex REST API ↗](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-rest-api) |
| API Endpoints | Chat Completions: `https://{account}.snowflakecomputing.com/api/v2/cortex/v1/chat/completions` <br/> Messages: `https://{account}.snowflakecomputing.com/api/v2/cortex/v1/messages` <br/> Legacy: `https://{account}.snowflakecomputing.com/api/v2/cortex/inference:complete` |
| Supported OpenAI Endpoints | `/chat/completions`, `/completions`, `/embeddings` |


Tip : We support ALL Snowflake Cortex models. Use `model=snowflake/<model-name>` as a prefix when sending LiteLLM requests.


## Authentication

Snowflake Cortex REST API supports three authentication methods.

### Programmatic Access Token (PAT) — Recommended

The simplest approach. Generate a PAT in Snowsight under **User Menu → My Profile → Programmatic Access Tokens**.

```python
import os
from litellm import completion

os.environ["SNOWFLAKE_API_KEY"] = "pat/<your-programmatic-access-token>"
os.environ["SNOWFLAKE_API_BASE"] = "https://<account>.snowflakecomputing.com/api/v2/cortex/v1"

response = completion(
    model="snowflake/claude-sonnet-4-5",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### JWT (Key-Pair Authentication)

Generate a JWT from a Snowflake key pair. See [Key-pair authentication](https://docs.snowflake.com/en/user-guide/key-pair-auth).

```python
import os
from litellm import completion

os.environ["SNOWFLAKE_JWT"] = "<your-jwt-token>"
os.environ["SNOWFLAKE_ACCOUNT_ID"] = "<orgname>-<account_name>"

response = completion(
    model="snowflake/claude-sonnet-4-5",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### Pass credentials as parameters

```python
from litellm import completion

# Using PAT
response = completion(
    model="snowflake/claude-sonnet-4-5",
    messages=[{"role": "user", "content": "Hello!"}],
    api_key="pat/<your-pat-token>",
    api_base="https://<account>.snowflakecomputing.com/api/v2/cortex/v1",
)

# Using JWT
response = completion(
    model="snowflake/claude-sonnet-4-5",
    messages=[{"role": "user", "content": "Hello!"}],
    api_key="<your-jwt-token>",
    account_id="<orgname>-<account_name>",
)
```

For all authentication options, see [Authenticating to Cortex REST API](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-rest-api#authenticating-cortex-rest-api-requests).

## Usage

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import os

os.environ["SNOWFLAKE_API_KEY"] = "pat/<your-pat>"
os.environ["SNOWFLAKE_API_BASE"] = "https://<account>.snowflakecomputing.com/api/v2/cortex/v1"

response = completion(
    model="snowflake/claude-sonnet-4-5",
    messages=[{"role": "user", "content": "What is Snowflake Cortex?"}],
)
print(response.choices[0].message.content)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

**1. Config**

```yaml
model_list:
  - model_name: claude-sonnet
    litellm_params:
      model: snowflake/claude-sonnet-4-5
      api_key: pat/<your-pat>
      api_base: https://<account>.snowflakecomputing.com/api/v2/cortex/v1
  - model_name: llama4-maverick
    litellm_params:
      model: snowflake/llama4-maverick
      api_key: pat/<your-pat>
      api_base: https://<account>.snowflakecomputing.com/api/v2/cortex/v1
```

**2. Start proxy**

```bash
litellm --config /path/to/config.yaml
```

**3. Test**

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
    "model": "claude-sonnet",
    "messages": [
        {"role": "user", "content": "What is Snowflake Cortex?"}
    ]
}'
```

</TabItem>
</Tabs>

## Supported OpenAI Parameters

```
temperature, max_tokens, top_p, stream, response_format,
tools, tool_choice
```

## Streaming

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import os

os.environ["SNOWFLAKE_API_KEY"] = "pat/<your-pat>"
os.environ["SNOWFLAKE_API_BASE"] = "https://<account>.snowflakecomputing.com/api/v2/cortex/v1"

response = completion(
    model="snowflake/claude-sonnet-4-5",
    messages=[{"role": "user", "content": "Write a haiku about data."}],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

</TabItem>
<TabItem value="proxy" label="PROXY">

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
    "model": "claude-sonnet",
    "messages": [{"role": "user", "content": "Write a haiku about data."}],
    "stream": true
}'
```

</TabItem>
</Tabs>

## Tool / Function Calling

Supported on Claude and select models. LiteLLM automatically transforms OpenAI tool format to Snowflake's `tool_spec` format.

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import os, json

os.environ["SNOWFLAKE_API_KEY"] = "pat/<your-pat>"
os.environ["SNOWFLAKE_API_BASE"] = "https://<account>.snowflakecomputing.com/api/v2/cortex/v1"

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"],
            },
        },
    }
]

response = completion(
    model="snowflake/claude-sonnet-4-5",
    messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
    tools=tools,
    tool_choice="auto",
)

print(response.choices[0].message.tool_calls)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

```yaml
model_list:
  - model_name: claude-sonnet
    litellm_params:
      model: snowflake/claude-sonnet-4-5
      api_key: pat/<your-pat>
      api_base: https://<account>.snowflakecomputing.com/api/v2/cortex/v1
```

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
    "model": "claude-sonnet",
    "messages": [{"role": "user", "content": "What is the weather in SF?"}],
    "tools": [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }
    }],
    "tool_choice": "auto"
}'
```

</TabItem>
</Tabs>

## Thinking / Reasoning

Claude 3.7 Sonnet, Claude 4 Opus, and DeepSeek R1 on Cortex support extended thinking. LiteLLM translates `reasoning_effort` to the provider's thinking parameter.

| `reasoning_effort` | `budget_tokens` |
|---|---|
| `"low"` | 1024 |
| `"medium"` | 2048 |
| `"high"` | 4096 |

```python
from litellm import completion

response = completion(
    model="snowflake/claude-3-7-sonnet",
    messages=[{"role": "user", "content": "Solve: what is 127 * 389?"}],
    reasoning_effort="low",
)
print(response.choices[0].message.content)
```

## Prompt Caching

Snowflake Cortex supports prompt caching to reduce costs:

- **OpenAI models**: Implicit caching for prompts ≥ 1,024 tokens (no code changes needed)
- **Claude models**: Explicit caching via `cache_control` breakpoints

Cached input tokens are billed at **10% of the regular input rate** (90% discount) when ≥ 1,024 tokens are cached.

See [Cortex REST API Billing & Cost Analysis](https://www.snowflake.com/en/developers/guides/cortex-rest-api-billing-cost/) for details.

## Embeddings

```python
from litellm import embedding
import os

os.environ["SNOWFLAKE_API_KEY"] = "pat/<your-pat>"
os.environ["SNOWFLAKE_API_BASE"] = "https://<account>.snowflakecomputing.com/api/v2/cortex/v1"

response = embedding(
    model="snowflake/snowflake-arctic-embed-l-v2.0",
    input=["Snowflake Cortex provides LLM inference"],
)
print(response.data[0]["embedding"][:5])
```

## Supported Models

All models are available through the `snowflake/` prefix.

:::tip
For current model availability, rate limits, and pricing, see the official [Cortex REST API docs](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-rest-api) and [Service Consumption Table](https://www.snowflake.com/legal-files/CreditConsumptionTable.pdf).
:::

### Chat Completion Models

| Model | `litellm` model name | Function Calling | Vision | Prompt Caching |
|---|---|---|---|---|
| Claude Sonnet 4.5 | `snowflake/claude-sonnet-4-5` | ✅ | ✅ | ✅ |
| Claude Sonnet 4.6 | `snowflake/claude-sonnet-4-6` | ✅ | ✅ | ✅ |
| Claude 4 Sonnet | `snowflake/claude-4-sonnet` | ✅ | ✅ | ✅ |
| Claude 4 Opus | `snowflake/claude-4-opus` | ✅ | ✅ | ✅ |
| Claude Haiku 4.5 | `snowflake/claude-haiku-4-5` | ✅ | ✅ | ✅ |
| Claude 3.7 Sonnet | `snowflake/claude-3-7-sonnet` | ✅ | ✅ | ✅ |
| Claude 3.5 Sonnet | `snowflake/claude-3-5-sonnet` | ✅ | ✅ | ✅ |
| OpenAI GPT-4.1 | `snowflake/openai-gpt-4.1` | ✅ | ✅ | ✅ |
| OpenAI GPT-5 | `snowflake/openai-gpt-5` | ✅ | ✅ | ✅ |
| OpenAI GPT-5 Mini | `snowflake/openai-gpt-5-mini` | ✅ | | |
| OpenAI GPT-5 Nano | `snowflake/openai-gpt-5-nano` | ✅ | | |
| DeepSeek R1 | `snowflake/deepseek-r1` | | | |
| Mistral Large 2 | `snowflake/mistral-large2` | ✅ | | |
| Llama 3.1 8B | `snowflake/llama3.1-8b` | | | |
| Llama 3.1 70B | `snowflake/llama3.1-70b` | ✅ | | |
| Llama 3.1 405B | `snowflake/llama3.1-405b` | ✅ | | |
| Llama 3.3 70B | `snowflake/llama3.3-70b` | ✅ | | |
| Llama 4 Maverick | `snowflake/llama4-maverick` | ✅ | | |
| Snowflake Llama 3.3 70B | `snowflake/snowflake-llama-3.3-70b` | ✅ | | |

### Embedding Models

| Model | `litellm` model name |
|---|---|
| Snowflake Arctic Embed L v2.0 | `snowflake/snowflake-arctic-embed-l-v2.0` |
| Snowflake Arctic Embed M v2.0 | `snowflake/snowflake-arctic-embed-m-v2.0` |

