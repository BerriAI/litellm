import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Claude Code Native

The `claude_code_native` provider is a specialized variant of Anthropic designed for **Claude Code**, Anthropic's official CLI for Claude. This provider uses Bearer authentication and automatically includes a specific system prompt optimized for CLI workflows.

## Key Differences from Anthropic

| Feature | Anthropics | Claude Code Native |
|----------|------------|-------------------|
| Authentication | `x-api-key` header | `Authorization: Bearer <token>` header |
| Required Headers | `anthropic-version` | `anthropic-version`, `anthropic-beta: oauth-2025-04-20` |
| System Prompt | User-defined | Always prepends: *"You are Claude Code, Anthropic's official CLI for Claude."* |
| API Endpoint | `https://api.anthropic.com/` | `https://api.anthropic.com/` (same endpoint, different auth/headers) |

| Property | Details |
|----------|---------|
| Description | Specialized Anthropic provider for Claude Code CLI with OAuth authentication and system prompt |
| Provider Route on LiteLLM | `claude_code_native/` (add this prefix to model names) |
| Provider Doc | [Claude Code ↗](https://docs.anthropic.com/en/docs/claude-code) |
| API Endpoint | https://api.anthropic.com |
| Supported Endpoints | `/chat/completions`, `/v1/messages` (passthrough) |

## Supported Models

Not all Anthropic models are supported through the `claude_code_native` provider.   The supported models are 

- `claude-sonnet-4-5`
- `claude-opus-4-5`
- `claude-haiku-4-5`

## Supported OpenAI Parameters

The `claude_code_native` provider supports all Anthropic parameters:

```
"stream",
"stop",
"temperature",
"top_p",
"max_tokens",
"max_completion_tokens",
"tools",
"tool_choice",
"extra_headers",
"parallel_tool_calls",
"response_format",
"user",
"reasoning_effort",
```

:::info
**Notes:**
- The provider automatically adds the required system prompt for Claude Code
- All system messages provided by the user are appended after the Claude Code system prompt
- The provider requires an OAuth bearer token (not an API key)
- `max_tokens` defaults to `4096` if not specified
:::

## API Keys and Authentication

The `claude_code_native` provider uses **OAuth Bearer authentication** and requires the `CLAUDE_CODE_API_KEY` environment variable:

```python
import os

os.environ["CLAUDE_CODE_API_KEY"] = "your-oauth-bearer-token"  # Used as Bearer token
# os.environ["CLAUDE_CODE_API_BASE"] = ""  # [OPTIONAL] Custom API base
```

:::note
The provider automatically converts the API key to a Bearer token by setting the `Authorization: Bearer <token>` header instead of `x-api-key: <token>`.
:::

:::info
**Priority for API Key:**
1. Explicit `api_key` parameter in completion calls takes highest priority
2. The `CLAUDE_CODE_API_KEY` environment variable is required if no explicit api_key is provided
:::

## Getting a Long-Lived Token

The `claude_code_native` provider is designed to work seamlessly with **Claude Code**, Anthropic's official CLI. Using Claude Code eliminates the need for separate API keys - the long-lived token approach is both cache and token efficient, with improved implementation that speaks directly to the Claude API.

### Preparing to Configure

1. **Install Claude Code** using your preferred method:

- **macOS or Linux** (curl): `curl -fsSL https://claude.ai/install.sh | bash`
- **macOS** (Homebrew): `brew install --cask claude-code`
- **Windows** (PowerShell): `irm https://claude.ai/install.ps1 | iex`

2. **Login and get a long-lived token** (valid for one year):

```bash
claude setup-token
```

3. **Record the token value** - it will not be shown again.

4. **Set the environment variable**:

```bash
export CLAUDE_CODE_API_KEY="your-long-lived-token-from-claude-setup-token"
```

It can also be entered in the UI in the normal fashion.

:::tip
The long-lived token obtained from `claude setup-token` is the recommended way to authenticate with the `claude_code_native` provider. It's valid for one year and provides a secure, cache-efficient way to interact with Claude models.
:::

### Supported Models

The `claude_code_native` provider supports the following Claude models:

- **Claude Sonnet 4.5** (`claude-sonnet-4-5`) - Latest, recommended for most use cases
- **Claude Opus 4.5** (`claude-opus-4-5`) - Most capable model
- **Claude 4.5 Haiku** (`claude-haiku-4-5`) - Fast responses for quick tasks

:::info
The implementation uses an updated technique that speaks directly to the Claude API, eliminating many downsides of previous implementations with better cache and token efficiency.
:::

## Usage

### Basic Usage

```python
import os
from litellm import completion

# Set your OAuth bearer token
os.environ["CLAUDE_CODE_API_KEY"] = "your-oauth-bearer-token"

messages = [{"role": "user", "content": "Hey! how's it going?"}]
response = completion(model="claude_code_native/claude-opus-4-20250514", messages=messages)
print(response)
```

### Usage - Streaming

```python
import os
from litellm import completion

# Set your OAuth bearer token
os.environ["CLAUDE_CODE_API_KEY"] = "your-oauth-bearer-token"

messages = [{"role": "user", "content": "Hey! how's it going?"}]
response = completion(model="claude_code_native/claude-opus-4-20250514", messages=messages, stream=True)
for chunk in response:
    print(chunk["choices"][0]["delta"]["content"])  # same as openai format
```

### Usage with Custom System Message

When you provide your own system message, it will be appended after the Claude Code system prompt:

```python
import os
from litellm import completion

os.environ["CLAUDE_CODE_API_KEY"] = "your-oauth-bearer-token"

messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Help me debug this code."}
]
response = completion(model="claude_code_native/claude-3-5-sonnet-20240620", messages=messages)
print(response)
```

The actual system messages sent to Anthropic will be:
1. *"You are Claude Code, Anthropic's official CLI for Claude."* (auto-added)
2. *"You are a helpful coding assistant."* (user-provided)

## Usage with LiteLLM Proxy

### 1. Save key in your environment

```bash
export CLAUDE_CODE_API_KEY="your-oauth-bearer-token"
```

### 2. Start the proxy

<Tabs>
<TabItem value="config" label="config.yaml">

```yaml
model_list:
  - model_name: claude-code-sonnet-4 ### RECEIVED MODEL NAME ###
    litellm_params: # all params accepted by litellm.completion()
      model: claude_code_native/claude-sonnet-4-20250514 ### MODEL NAME sent to `litellm.completion()` ###
      api_key: "os.environ/CLAUDE_CODE_API_KEY" # does os.getenv("CLAUDE_CODE_API_KEY")
```

```bash
litellm --config /path/to/config.yaml
```
</TabItem>
<TabItem value="config-all" label="config - default all Claude Code Native Models">

Use this if you want to make requests to any claude_code_native model without defining them individually:

#### Required env variables
```
CLAUDE_CODE_API_KEY=your-oauth-bearer-token
```

```yaml
model_list:
  - model_name: "*" 
    litellm_params:
      model: claude_code_native/*
```

```bash
litellm --config /path/to/config.yaml
```

Example Request for this config.yaml

**Ensure you use `claude_code_native/` prefix to route the request to the Claude Code Native provider**

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "claude_code_native/claude-3-haiku-20240307",
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
<TabItem value="cli" label="cli">

```bash
$ litellm --model claude_code_native/claude-opus-4-20250514

# Server running on http://0.0.0.0:4000
```
</TabItem>
</Tabs>

### 3. Test it

<Tabs>
<TabItem value="Curl" label="Curl Request">

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "claude-code-sonnet-4",
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
response = client.chat.completions.create(model="claude-code-sonnet-4", messages = [
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
    model = "claude-code-sonnet-4",
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

## Supported Models

| Model Name       | Function Call                              | OAuth Required |
|------------------|--------------------------------------------|----------------|
| claude-sonnet-4-5  | `completion('claude_code_native/claude-sonnet-4-5-20250929', messages)` | `os.environ['CLAUDE_CODE_API_KEY']` |
| claude-opus-4  | `completion('claude_code_native/claude-opus-4-20250514', messages)` | `os.environ['CLAUDE_CODE_API_KEY']` |
| claude-sonnet-4  | `completion('claude_code_native/claude-sonnet-4-20250514', messages)` | `os.environ['CLAUDE_CODE_API_KEY']` |
| claude-3.7  | `completion('claude_code_native/claude-3-7-sonnet-20250219', messages)` | `os.environ['CLAUDE_CODE_API_KEY']` |
| claude-3-5-sonnet  | `completion('claude_code_native/claude-3-5-sonnet-20240620', messages)` | `os.environ['CLAUDE_CODE_API_KEY']` |
| claude-3-haiku  | `completion('claude_code_native/claude-3-haiku-20240307', messages)` | `os.environ['CLAUDE_CODE_API_KEY']` |
| claude-3-opus  | `completion('claude_code_native/claude-3-opus-20240229', messages)` | `os.environ['CLAUDE_CODE_API_KEY']` |
| claude-3-sonnet  | `completion('claude_code_native/claude-3-sonnet-20240229', messages)` | `os.environ['CLAUDE_CODE_API_KEY']` |
| claude-2.1  | `completion('claude_code_native/claude-2.1', messages)` | `os.environ['CLAUDE_CODE_API_KEY']` |
| claude-2  | `completion('claude_code_native/claude-2', messages)` | `os.environ['CLAUDE_CODE_API_KEY']` |

## Function/Tool Calling

The `claude_code_native` provider supports all Anthropic tool calling features:

```python
from litellm import completion
import os

# Set OAuth bearer token
os.environ["CLAUDE_CODE_API_KEY"] = "your-oauth-bearer-token"

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
    model="claude_code_native/claude-3-opus-20240229",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)
print(response)
```

## Prompt Caching

The `claude_code_native` provider supports Anthropic's prompt caching feature. See the [Anthropic documentation](./anthropic.md#--prompt-caching) for detailed examples with the `cache_control` parameter.

```python
response = await litellm.acompletion(
    model="claude_code_native/claude-3-5-sonnet-20240620",
    messages=[
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an AI assistant tasked with analyzing legal documents.",
                },
                {
                    "type": "text",
                    "text": "Here is the full text of a complex legal agreement",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
        },
        {
            "role": "user",
            "content": "what are the key terms and conditions in this agreement?",
        },
    ]
)
```

## Structured Outputs

The `claude_code_native` provider supports Anthropic's structured outputs feature. See the [Anthropic documentation](./anthropic.md#-structured-outputs) for details.

```python
from litellm import completion

response = completion(
    model="claude_code_native/claude-sonnet-4-5-20250929",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "capital_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "country": {"type": "string"},
                    "capital": {"type": "string"}
                },
                "required": ["country", "capital"],
                "additionalProperties": False
            }
        }
    }
)

print(response.choices[0].message.content)
```

## Vision

All Anthropic vision capabilities are supported by `claude_code_native`:

```python
from litellm import completion
import os
import base64

os.environ["CLAUDE_CODE_API_KEY"] = "your-oauth-bearer-token"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

image_path = "../proxy/cached_logo.jpg"
base64_image = encode_image(image_path)

resp = litellm.completion(
    model="claude_code_native/claude-3-opus-20240229",
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

## MCP Tool Calling

MCP tool calling is supported with the `claude_code_native` provider:

```python
import os 
from litellm import completion

os.environ["CLAUDE_CODE_API_KEY"] = "your-oauth-bearer-token"

tools=[
    {
        "type": "mcp",
        "server_label": "deepwiki",
        "server_url": "https://mcp.deepwiki.com/mcp",
        "require_approval": "never",
    },
]

response = completion(
    model="claude_code_native/claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Who won the World Cup in 2022?"}],
    tools=tools
)
```

## Hosted Tools

The `claude_code_native` provider supports Anthropic's hosted tools (Computer, Text Editor, Web Search, Memory):

### Computer Use

```python
from litellm import completion
import os

os.environ["CLAUDE_CODE_API_KEY"] = "your-oauth-bearer-token"

tools = [
    {
        "type": "computer_20241022",
        "function": {
            "name": "computer",
            "parameters": {
                "display_height_px": 100,
                "display_width_px": 100,
                "display_number": 1,
            },
        },
    }
]
model = "claude_code_native/claude-3-5-sonnet-20241022"
messages = [{"role": "user", "content": "Save a picture of a cat to my desktop."}]

resp = completion(
    model=model,
    messages=messages,
    tools=tools,
)
print(resp)
```

### Web Search

```python
from litellm import completion
import os

os.environ["CLAUDE_CODE_API_KEY"] = "your-oauth-bearer-token"

model = "claude_code_native/claude-3-5-sonnet-20241022"
messages = [{"role": "user", "content": "What's the weather like today?"}]

resp = completion(
    model=model,
    messages=messages,
    web_search_options={
        "search_context_size": "medium",
        "user_location": {
            "type": "approximate",
            "approximate": {
                "city": "San Francisco",
            },
        }
    }
)
print(resp)
```

### Memory

```python
from litellm import completion
import os

os.environ["CLAUDE_CODE_API_KEY"] = "your-oauth-bearer-token"

tools = [{
    "type": "memory_20250818",
    "name": "memory"
}]

model = "claude_code_native/claude-sonnet-4-5-20250929" 
messages = [{"role": "user", "content": "Please remember that my favorite color is blue."}]

response = completion(
    model=model,
    messages=messages,
    tools=tools,
)
print(response)
```

## Passing Extra Headers

```python
from litellm import completion
import os

os.environ["CLAUDE_CODE_API_KEY"] = "your-oauth-bearer-token"

messages = [{"role": "user", "content": "What is Claude Code Native?"}]
response = completion(
    model="claude_code_native/claude-3-5-sonnet-20240620", 
    messages=messages, 
    extra_headers={"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"}
)
```

:::info
The provider automatically merges any extra headers with its required headers (`anthropic-beta: oauth-2025-04-20` and `anthropic-version: 2023-06-01`). The provider's required headers take precedence.
:::

## Comparison with Anthropic Provider

| Feature | `anthropic/` | `claude_code_native/` |
|---------|--------------|----------------------|
| Authentication | `x-api-key` header | `Authorization: Bearer` header |
| System Prompt | User-provided only | Auto: *"You are Claude Code, Anthropic's official CLI for Claude."* + User-provided |
| Required Headers | `anthropic-version` | `anthropic-version`, `anthropic-beta: oauth-2025-04-20` |
| Use Case | General purpose | Claude Code CLI applications |
| Tool Support | Full | Full (same as Anthropic) |
| Prompt Caching | Supported | Supported |
| Vision | Supported | Supported |

## API Reference

For detailed API information, see:
- [Anthropic API Documentation](https://docs.anthropic.com/en/api/welcome)
- [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
- [LiteLLM Anthropic Provider](./anthropic.md)

:::note
The `claude_code_native` provider builds on top of the Anthropic provider implementation. All Anthropic-specific features and parameters are fully supported.
:::
