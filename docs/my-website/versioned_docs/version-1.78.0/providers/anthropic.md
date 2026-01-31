import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Anthropic
LiteLLM supports all anthropic models.

- `claude-sonnet-4-5-20250929`
- `claude-opus-4-1-20250805`
- `claude-4` (`claude-opus-4-20250514`, `claude-sonnet-4-20250514`)
- `claude-3.7` (`claude-3-7-sonnet-20250219`)
- `claude-3.5` (`claude-3-5-sonnet-20240620`)
- `claude-3` (`claude-3-haiku-20240307`, `claude-3-opus-20240229`, `claude-3-sonnet-20240229`)
- `claude-2`
- `claude-2.1`
- `claude-instant-1.2`


| Property | Details |
|-------|-------|
| Description | Claude is a highly performant, trustworthy, and intelligent AI platform built by Anthropic. Claude excels at tasks involving language, reasoning, analysis, coding, and more. |
| Provider Route on LiteLLM | `anthropic/` (add this prefix to the model name, to route any requests to Anthropic - e.g. `anthropic/claude-3-5-sonnet-20240620`) |
| Provider Doc | [Anthropic ↗](https://docs.anthropic.com/en/docs/build-with-claude/overview) |
| API Endpoint for Provider | https://api.anthropic.com |
| Supported Endpoints | `/chat/completions` |


## Supported OpenAI Parameters

Check this in code, [here](../completion/input.md#translated-openai-params)

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
"user"
```

:::info

Anthropic API fails requests when `max_tokens` are not passed. Due to this litellm passes `max_tokens=4096` when no `max_tokens` are passed.

:::

## API Keys

```python
import os

os.environ["ANTHROPIC_API_KEY"] = "your-api-key"
# os.environ["ANTHROPIC_API_BASE"] = "" # [OPTIONAL] or 'ANTHROPIC_BASE_URL'
# os.environ["LITELLM_ANTHROPIC_DISABLE_URL_SUFFIX"] = "true" # [OPTIONAL] Disable automatic URL suffix appending
```

### Custom API Base

When using a custom API base for Anthropic (e.g., a proxy or custom endpoint), LiteLLM automatically appends the appropriate suffix (`/v1/messages` or `/v1/complete`) to your base URL.

If your custom endpoint already includes the full path or doesn't follow Anthropic's standard URL structure, you can disable this automatic suffix appending:

```python
import os

os.environ["ANTHROPIC_API_BASE"] = "https://my-custom-endpoint.com/custom/path"
os.environ["LITELLM_ANTHROPIC_DISABLE_URL_SUFFIX"] = "true"  # Prevents automatic suffix
```

Without `LITELLM_ANTHROPIC_DISABLE_URL_SUFFIX`:
- Base URL `https://my-proxy.com` → `https://my-proxy.com/v1/messages`
- Base URL `https://my-proxy.com/api` → `https://my-proxy.com/api/v1/messages`

With `LITELLM_ANTHROPIC_DISABLE_URL_SUFFIX=true`:
- Base URL `https://my-proxy.com/custom/path` → `https://my-proxy.com/custom/path` (unchanged)

## Usage

```python
import os
from litellm import completion

# set env - [OPTIONAL] replace with your anthropic key
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

messages = [{"role": "user", "content": "Hey! how's it going?"}]
response = completion(model="claude-opus-4-20250514", messages=messages)
print(response)
```


## Usage - Streaming
Just set `stream=True` when calling completion.

```python
import os
from litellm import completion

# set env
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

messages = [{"role": "user", "content": "Hey! how's it going?"}]
response = completion(model="claude-opus-4-20250514", messages=messages, stream=True)
for chunk in response:
    print(chunk["choices"][0]["delta"]["content"])  # same as openai format
```

## Usage with LiteLLM Proxy 

Here's how to call Anthropic with the LiteLLM Proxy Server

### 1. Save key in your environment

```bash
export ANTHROPIC_API_KEY="your-api-key"
```

### 2. Start the proxy 

<Tabs>
<TabItem value="config" label="config.yaml">

```yaml
model_list:
  - model_name: claude-4 ### RECEIVED MODEL NAME ###
    litellm_params: # all params accepted by litellm.completion() - https://docs.litellm.ai/docs/completion/input
      model: claude-opus-4-20250514 ### MODEL NAME sent to `litellm.completion()` ###
      api_key: "os.environ/ANTHROPIC_API_KEY" # does os.getenv("ANTHROPIC_API_KEY")
```

```bash
litellm --config /path/to/config.yaml
```
</TabItem>
<TabItem value="config-all" label="config - default all Anthropic Model">

Use this if you want to make requests to `claude-3-haiku-20240307`,`claude-3-opus-20240229`,`claude-2.1` without defining them on the config.yaml

#### Required env variables
```
ANTHROPIC_API_KEY=sk-ant****
```

```yaml
model_list:
  - model_name: "*" 
    litellm_params:
      model: "*"
```

```bash
litellm --config /path/to/config.yaml
```

Example Request for this config.yaml

**Ensure you use `anthropic/` prefix to route the request to Anthropic API**

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "anthropic/claude-3-haiku-20240307",
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
$ litellm --model claude-opus-4-20250514

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
      "model": "claude-3",
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
response = client.chat.completions.create(model="claude-3", messages = [
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
    model = "claude-3",
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

`Model Name` 👉 Human-friendly name.  
`Function Call` 👉 How to call the model in LiteLLM.

| Model Name       | Function Call                              |
|------------------|--------------------------------------------|
| claude-sonnet-4-5  | `completion('claude-sonnet-4-5-20250929', messages)` | `os.environ['ANTHROPIC_API_KEY']`       |
| claude-opus-4  | `completion('claude-opus-4-20250514', messages)` | `os.environ['ANTHROPIC_API_KEY']`       |
| claude-sonnet-4  | `completion('claude-sonnet-4-20250514', messages)` | `os.environ['ANTHROPIC_API_KEY']`       |
| claude-3.7  | `completion('claude-3-7-sonnet-20250219', messages)` | `os.environ['ANTHROPIC_API_KEY']`       |
| claude-3-5-sonnet  | `completion('claude-3-5-sonnet-20240620', messages)` | `os.environ['ANTHROPIC_API_KEY']`       |
| claude-3-haiku  | `completion('claude-3-haiku-20240307', messages)` | `os.environ['ANTHROPIC_API_KEY']`       |
| claude-3-opus  | `completion('claude-3-opus-20240229', messages)` | `os.environ['ANTHROPIC_API_KEY']`       |
| claude-3-5-sonnet-20240620  | `completion('claude-3-5-sonnet-20240620', messages)` | `os.environ['ANTHROPIC_API_KEY']`       |
| claude-3-sonnet  | `completion('claude-3-sonnet-20240229', messages)` | `os.environ['ANTHROPIC_API_KEY']`       |
| claude-2.1  | `completion('claude-2.1', messages)` | `os.environ['ANTHROPIC_API_KEY']`       |
| claude-2  | `completion('claude-2', messages)` | `os.environ['ANTHROPIC_API_KEY']`       |
| claude-instant-1.2  | `completion('claude-instant-1.2', messages)` | `os.environ['ANTHROPIC_API_KEY']`       |
| claude-instant-1  | `completion('claude-instant-1', messages)` | `os.environ['ANTHROPIC_API_KEY']`       |

## **Prompt Caching**

Use Anthropic Prompt Caching


[Relevant Anthropic API Docs](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)

:::note

Here's what a sample Raw Request from LiteLLM for Anthropic Context Caching looks like: 

```bash
POST Request Sent from LiteLLM:
curl -X POST \
https://api.anthropic.com/v1/messages \
-H 'accept: application/json' -H 'anthropic-version: 2023-06-01' -H 'content-type: application/json' -H 'x-api-key: sk-...' -H 'anthropic-beta: prompt-caching-2024-07-31' \
-d '{'model': 'claude-3-5-sonnet-20240620', [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What are the key terms and conditions in this agreement?",
          "cache_control": {
            "type": "ephemeral"
          }
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "Certainly! The key terms and conditions are the following: the contract is 1 year long for $10/mo"
        }
      ]
    }
  ],
  "temperature": 0.2,
  "max_tokens": 10
}'
```
::: 

### Caching - Large Context Caching 


This example demonstrates basic Prompt Caching usage, caching the full text of the legal agreement as a prefix while keeping the user instruction uncached.


<Tabs>
<TabItem value="sdk" label="LiteLLM SDK">

```python 
response = await litellm.acompletion(
    model="anthropic/claude-3-5-sonnet-20240620",
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
</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

:::info

LiteLLM Proxy is OpenAI compatible

This is an example using the OpenAI Python SDK sending a request to LiteLLM Proxy

Assuming you have a model=`anthropic/claude-3-5-sonnet-20240620` on the [litellm proxy config.yaml](#usage-with-litellm-proxy)

:::

```python 
import openai
client = openai.AsyncOpenAI(
    api_key="anything",            # litellm proxy api key
    base_url="http://0.0.0.0:4000" # litellm proxy base url
)


response = await client.chat.completions.create(
    model="anthropic/claude-3-5-sonnet-20240620",
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

</TabItem>
</Tabs>

### Caching - Tools definitions

In this example, we demonstrate caching tool definitions.

The cache_control parameter is placed on the final tool

<Tabs>
<TabItem value="sdk" label="LiteLLM SDK">

```python 
import litellm

response = await litellm.acompletion(
    model="anthropic/claude-3-5-sonnet-20240620",
    messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]
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
                "cache_control": {"type": "ephemeral"}
            },
        }
    ]
)
```
</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

:::info

LiteLLM Proxy is OpenAI compatible

This is an example using the OpenAI Python SDK sending a request to LiteLLM Proxy

Assuming you have a model=`anthropic/claude-3-5-sonnet-20240620` on the [litellm proxy config.yaml](#usage-with-litellm-proxy)

:::

```python 
import openai
client = openai.AsyncOpenAI(
    api_key="anything",            # litellm proxy api key
    base_url="http://0.0.0.0:4000" # litellm proxy base url
)

response = await client.chat.completions.create(
    model="anthropic/claude-3-5-sonnet-20240620",
    messages = [{"role": "user", "content": "What's the weather like in Boston today?"}]
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
                "cache_control": {"type": "ephemeral"}
            },
        }
    ]
)
```

</TabItem>
</Tabs>


### Caching - Continuing Multi-Turn Convo

In this example, we demonstrate how to use Prompt Caching in a multi-turn conversation.

The cache_control parameter is placed on the system message to designate it as part of the static prefix.

The conversation history (previous messages) is included in the messages array. The final turn is marked with cache-control, for continuing in followups. The second-to-last user message is marked for caching with the cache_control parameter, so that this checkpoint can read from the previous cache.

<Tabs>
<TabItem value="sdk" label="LiteLLM SDK">

```python 
import litellm

response = await litellm.acompletion(
    model="anthropic/claude-3-5-sonnet-20240620",
    messages=[
        # System Message
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Here is the full text of a complex legal agreement"
                    * 400,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        # marked for caching with the cache_control parameter, so that this checkpoint can read from the previous cache.
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What are the key terms and conditions in this agreement?",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "assistant",
            "content": "Certainly! the key terms and conditions are the following: the contract is 1 year long for $10/mo",
        },
        # The final turn is marked with cache-control, for continuing in followups.
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What are the key terms and conditions in this agreement?",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
    ]
)
```
</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

:::info

LiteLLM Proxy is OpenAI compatible

This is an example using the OpenAI Python SDK sending a request to LiteLLM Proxy

Assuming you have a model=`anthropic/claude-3-5-sonnet-20240620` on the [litellm proxy config.yaml](#usage-with-litellm-proxy)

:::

```python 
import openai
client = openai.AsyncOpenAI(
    api_key="anything",            # litellm proxy api key
    base_url="http://0.0.0.0:4000" # litellm proxy base url
)

response = await client.chat.completions.create(
    model="anthropic/claude-3-5-sonnet-20240620",
    messages=[
        # System Message
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Here is the full text of a complex legal agreement"
                    * 400,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        # marked for caching with the cache_control parameter, so that this checkpoint can read from the previous cache.
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What are the key terms and conditions in this agreement?",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "assistant",
            "content": "Certainly! the key terms and conditions are the following: the contract is 1 year long for $10/mo",
        },
        # The final turn is marked with cache-control, for continuing in followups.
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What are the key terms and conditions in this agreement?",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
    ]
)
```

</TabItem>
</Tabs>

## **Function/Tool Calling**

```python
from litellm import completion

# set env
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

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
    model="anthropic/claude-3-opus-20240229",
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


### Forcing Anthropic Tool Use

If you want Claude to use a specific tool to answer the user’s question

You can do this by specifying the tool in the `tool_choice` field like so:
```python
response = completion(
    model="anthropic/claude-3-opus-20240229",
    messages=messages,
    tools=tools,
    tool_choice={"type": "tool", "name": "get_weather"},
)
```

### Disable Tool Calling

You can disable tool calling by setting the `tool_choice` to `"none"`.

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion

response = completion(
    model="anthropic/claude-3-opus-20240229",
    messages=messages,
    tools=tools,
    tool_choice="none",
)

```
</TabItem>
<TabItem value="proxy" label="Proxy">

1. Setup config.yaml

```yaml
model_list:
  - model_name: anthropic-claude-model
    litellm_params:
        model: anthropic/claude-3-opus-20240229
        api_key: os.environ/ANTHROPIC_API_KEY
```

2. Start proxy

```bash
litellm --config /path/to/config.yaml
```

3. Test it! 

Replace `anything` with your LiteLLM Proxy Virtual Key, if [setup](../proxy/virtual_keys).

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer anything" \
  -d '{
    "model": "anthropic-claude-model",
    "messages": [{"role": "user", "content": "Who won the World Cup in 2022?"}],
    "tools": [{"type": "mcp", "server_label": "deepwiki", "server_url": "https://mcp.deepwiki.com/mcp", "require_approval": "never"}],
    "tool_choice": "none"
  }'
```
</TabItem>
</Tabs>



### MCP Tool Calling 

Here's how to use MCP tool calling with Anthropic:

<Tabs>
<TabItem value="sdk" label="LiteLLM SDK">

LiteLLM supports MCP tool calling with Anthropic in the OpenAI Responses API format.

<Tabs>
<TabItem value="openai_format" label="OpenAI Format">


```python
import os 
from litellm import completion

os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."

tools=[
    {
        "type": "mcp",
        "server_label": "deepwiki",
        "server_url": "https://mcp.deepwiki.com/mcp",
        "require_approval": "never",
    },
]

response = completion(
    model="anthropic/claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Who won the World Cup in 2022?"}],
    tools=tools
)
```

</TabItem>
<TabItem value="anthropic_format" label="Anthropic Format">

```python
import os 
from litellm import completion

os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."

tools = [
    {
        "type": "url",
        "url": "https://mcp.deepwiki.com/mcp",
        "name": "deepwiki-mcp",
    }
]
response = completion(
    model="anthropic/claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Who won the World Cup in 2022?"}],
    tools=tools
)

print(response)
```
</TabItem>

</Tabs>

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

1. Setup config.yaml

```yaml
model_list:
  - model_name: claude-4-sonnet
    litellm_params:
        model: anthropic/claude-sonnet-4-20250514
        api_key: os.environ/ANTHROPIC_API_KEY
```

2. Start proxy

```bash
litellm --config /path/to/config.yaml
```

3. Test it! 

<Tabs>
<TabItem value="openai" label="OpenAI Format">

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -d '{
    "model": "claude-4-sonnet",
    "messages": [{"role": "user", "content": "Who won the World Cup in 2022?"}],
    "tools": [{"type": "mcp", "server_label": "deepwiki", "server_url": "https://mcp.deepwiki.com/mcp", "require_approval": "never"}]
  }'
```

</TabItem>
<TabItem value="anthropic" label="Anthropic Format">

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -d '{
    "model": "claude-4-sonnet",
    "messages": [{"role": "user", "content": "Who won the World Cup in 2022?"}],
    "tools": [
        {
            "type": "url",
            "url": "https://mcp.deepwiki.com/mcp",
            "name": "deepwiki-mcp",
        }
    ]
  }'
```

</TabItem>
</Tabs>
</TabItem>
</Tabs>

### Parallel Function Calling 

Here's how to pass the result of a function call back to an anthropic model: 

```python
from litellm import completion
import os 

os.environ["ANTHROPIC_API_KEY"] = "sk-ant.."


litellm.set_verbose = True

### 1ST FUNCTION CALL ###
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
messages = [
    {
        "role": "user",
        "content": "What's the weather like in Boston today in Fahrenheit?",
    }
]
try:
    # test without max tokens
    response = completion(
        model="anthropic/claude-3-opus-20240229",
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

    messages.append(
        response.choices[0].message.model_dump()
    )  # Add assistant tool invokes
    tool_result = (
        '{"location": "Boston", "temperature": "72", "unit": "fahrenheit"}'
    )
    # Add user submitted tool results in the OpenAI format
    messages.append(
        {
            "tool_call_id": response.choices[0].message.tool_calls[0].id,
            "role": "tool",
            "name": response.choices[0].message.tool_calls[0].function.name,
            "content": tool_result,
        }
    )
    ### 2ND FUNCTION CALL ###
    # In the second response, Claude should deduce answer from tool results
    second_response = completion(
        model="anthropic/claude-3-opus-20240229",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    print(second_response)
except Exception as e:
    print(f"An error occurred - {str(e)}")
```

s/o @[Shekhar Patnaik](https://www.linkedin.com/in/patnaikshekhar) for requesting this!

### Anthropic Hosted Tools (Computer, Text Editor, Web Search)


<Tabs>
<TabItem value="computer" label="Computer">

```python
from litellm import completion

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
model = "claude-3-5-sonnet-20241022"
messages = [{"role": "user", "content": "Save a picture of a cat to my desktop."}]

resp = completion(
    model=model,
    messages=messages,
    tools=tools,
    # headers={"anthropic-beta": "computer-use-2024-10-22"},
)

print(resp)
```

</TabItem>
<TabItem value="text_editor" label="Text Editor">

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion

tools = [{
    "type": "text_editor_20250124",
    "name": "str_replace_editor"
}]
model = "claude-3-5-sonnet-20241022"
messages = [{"role": "user", "content": "There's a syntax error in my primes.py file. Can you help me fix it?"}]

resp = completion(
    model=model,
    messages=messages,
    tools=tools,
)

print(resp)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
- model_name: claude-3-5-sonnet-latest
  litellm_params:
    model: anthropic/claude-3-5-sonnet-latest
    api_key: os.environ/ANTHROPIC_API_KEY
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
    "model": "claude-3-5-sonnet-latest",
    "messages": [{"role": "user", "content": "There's a syntax error in my primes.py file. Can you help me fix it?"}],
    "tools": [{"type": "text_editor_20250124", "name": "str_replace_editor"}]
  }'
```
</TabItem>
</Tabs>

</TabItem>
<TabItem value="web_search" label="Web Search">

:::info
Live from v1.70.1+
:::

LiteLLM maps OpenAI's `search_context_size` param to Anthropic's `max_uses` param.

| OpenAI | Anthropic |
| --- | --- |
| Low | 1 | 
| Medium | 5 | 
| High | 10 | 


<Tabs>
<TabItem value="sdk" label="SDK">


<Tabs>
<TabItem value="openai" label="OpenAI Format">

```python
from litellm import completion

model = "claude-3-5-sonnet-20241022"
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
</TabItem>
<TabItem value="anthropic" label="Anthropic Format">

```python
from litellm import completion

tools = [{
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 5
}]
model = "claude-3-5-sonnet-20241022"
messages = [{"role": "user", "content": "There's a syntax error in my primes.py file. Can you help me fix it?"}]

resp = completion(
    model=model,
    messages=messages,
    tools=tools,
)

print(resp)
```
</TabItem>

</Tabs>
</TabItem>

<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
- model_name: claude-3-5-sonnet-latest
  litellm_params:
    model: anthropic/claude-3-5-sonnet-latest
    api_key: os.environ/ANTHROPIC_API_KEY
```

2. Start proxy

```bash
litellm --config /path/to/config.yaml
```

3. Test it! 

<Tabs>
<TabItem value="openai" label="OpenAI Format">


```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -d '{
    "model": "claude-3-5-sonnet-latest",
    "messages": [{"role": "user", "content": "What's the weather like today?"}],
    "web_search_options": {
        "search_context_size": "medium",
        "user_location": {
            "type": "approximate",
            "approximate": {
                "city": "San Francisco",
            },
        }
    }
  }'
```
</TabItem>
<TabItem value="anthropic" label="Anthropic Format">

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -d '{
    "model": "claude-3-5-sonnet-latest",
    "messages": [{"role": "user", "content": "What's the weather like today?"}],
    "tools": [{
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 5
    }]
  }'
```

</TabItem>
</Tabs>
</TabItem>
</Tabs>

</TabItem>
</Tabs>



## Usage - Vision 

```python
from litellm import completion

# set env
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

def encode_image(image_path):
    import base64

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


image_path = "../proxy/cached_logo.jpg"
# Getting the base64 string
base64_image = encode_image(image_path)
resp = litellm.completion(
    model="anthropic/claude-3-opus-20240229",
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

## Usage - Thinking / `reasoning_content`

LiteLLM translates OpenAI's `reasoning_effort` to Anthropic's `thinking` parameter. [Code](https://github.com/BerriAI/litellm/blob/23051d89dd3611a81617d84277059cd88b2df511/litellm/llms/anthropic/chat/transformation.py#L298)

| reasoning_effort | thinking |
| ---------------- | -------- |
| "low"            | "budget_tokens": 1024 |
| "medium"         | "budget_tokens": 2048 |
| "high"           | "budget_tokens": 4096 |

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion

resp = completion(
    model="anthropic/claude-3-7-sonnet-20250219",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    reasoning_effort="low",
)

```

</TabItem>

<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
- model_name: claude-3-7-sonnet-20250219
  litellm_params:
    model: anthropic/claude-3-7-sonnet-20250219
    api_key: os.environ/ANTHROPIC_API_KEY
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
    "model": "claude-3-7-sonnet-20250219",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "reasoning_effort": "low"
  }'
```

</TabItem>
</Tabs>


**Expected Response**

```python
ModelResponse(
    id='chatcmpl-c542d76d-f675-4e87-8e5f-05855f5d0f5e',
    created=1740470510,
    model='claude-3-7-sonnet-20250219',
    object='chat.completion',
    system_fingerprint=None,
    choices=[
        Choices(
            finish_reason='stop',
            index=0,
            message=Message(
                content="The capital of France is Paris.",
                role='assistant',
                tool_calls=None,
                function_call=None,
                provider_specific_fields={
                    'citations': None,
                    'thinking_blocks': [
                        {
                            'type': 'thinking',
                            'thinking': 'The capital of France is Paris. This is a very straightforward factual question.',
                            'signature': 'EuYBCkQYAiJAy6...'
                        }
                    ]
                }
            ),
            thinking_blocks=[
                {
                    'type': 'thinking',
                    'thinking': 'The capital of France is Paris. This is a very straightforward factual question.',
                    'signature': 'EuYBCkQYAiJAy6AGB...'
                }
            ],
            reasoning_content='The capital of France is Paris. This is a very straightforward factual question.'
        )
    ],
    usage=Usage(
        completion_tokens=68,
        prompt_tokens=42,
        total_tokens=110,
        completion_tokens_details=None,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=None,
            cached_tokens=0,
            text_tokens=None,
            image_tokens=None
        ),
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0
    )
)
```

### Pass `thinking` to Anthropic models

You can also pass the `thinking` parameter to Anthropic models.


You can also pass the `thinking` parameter to Anthropic models.

<Tabs>
<TabItem value="sdk" label="SDK">

```python
response = litellm.completion(
  model="anthropic/claude-3-7-sonnet-20250219",
  messages=[{"role": "user", "content": "What is the capital of France?"}],
  thinking={"type": "enabled", "budget_tokens": 1024},
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -d '{
    "model": "anthropic/claude-3-7-sonnet-20250219",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "thinking": {"type": "enabled", "budget_tokens": 1024}
  }'
```

</TabItem>
</Tabs>




## **Passing Extra Headers to Anthropic API**

Pass `extra_headers: dict` to `litellm.completion`

```python
from litellm import completion
messages = [{"role": "user", "content": "What is Anthropic?"}]
response = completion(
    model="claude-3-5-sonnet-20240620", 
    messages=messages, 
    extra_headers={"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"}
)
```

## Usage - "Assistant Pre-fill"

You can "put words in Claude's mouth" by including an `assistant` role message as the last item in the `messages` array.

> [!IMPORTANT]
> The returned completion will _not_ include your "pre-fill" text, since it is part of the prompt itself. Make sure to prefix Claude's completion with your pre-fill.

```python
import os
from litellm import completion

# set env - [OPTIONAL] replace with your anthropic key
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

messages = [
    {"role": "user", "content": "How do you say 'Hello' in German? Return your answer as a JSON object, like this:\n\n{ \"Hello\": \"Hallo\" }"},
    {"role": "assistant", "content": "{"},
]
response = completion(model="claude-2.1", messages=messages)
print(response)
```

#### Example prompt sent to Claude

```

Human: How do you say 'Hello' in German? Return your answer as a JSON object, like this:

{ "Hello": "Hallo" }

Assistant: {
```

## Usage - "System" messages
If you're using Anthropic's Claude 2.1, `system` role messages are properly formatted for you.

```python
import os
from litellm import completion

# set env - [OPTIONAL] replace with your anthropic key
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

messages = [
    {"role": "system", "content": "You are a snarky assistant."},
    {"role": "user", "content": "How do I boil water?"},
]
response = completion(model="claude-2.1", messages=messages)
```

#### Example prompt sent to Claude

```
You are a snarky assistant.

Human: How do I boil water?

Assistant:
```


## Usage - PDF 

Pass base64 encoded PDF files to Anthropic models using the `image_url` field.

<Tabs>
<TabItem value="sdk" label="SDK">

### **using base64**
```python
from litellm import completion, supports_pdf_input
import base64
import requests

# URL of the file
url = "https://storage.googleapis.com/cloud-samples-data/generative-ai/pdf/2403.05530.pdf"

# Download the file
response = requests.get(url)
file_data = response.content

encoded_file = base64.b64encode(file_data).decode("utf-8")

## check if model supports pdf input - (2024/11/11) only claude-3-5-haiku-20241022 supports it
supports_pdf_input("anthropic/claude-3-5-haiku-20241022") # True

response = completion(
    model="anthropic/claude-3-5-haiku-20241022",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "You are a very professional document summarization specialist. Please summarize the given document."},
                {
                    "type": "file",
                    "file": {
                       "file_data": f"data:application/pdf;base64,{encoded_file}", # 👈 PDF
                    }
                },
            ],
        }
    ],
    max_tokens=300,
)

print(response.choices[0])
```
</TabItem>
<TabItem value="proxy" label="PROXY">

1. Add model to config 

```yaml
- model_name: claude-3-5-haiku-20241022
  litellm_params:
    model: anthropic/claude-3-5-haiku-20241022
    api_key: os.environ/ANTHROPIC_API_KEY
```

2. Start Proxy

```
litellm --config /path/to/config.yaml
```

3. Test it! 

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR-LITELLM-KEY>" \
  -d '{
    "model": "claude-3-5-haiku-20241022",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "You are a very professional document summarization specialist. Please summarize the given document"
          },
          {
                "type": "file",
                "file": {
                    "file_data": f"data:application/pdf;base64,{encoded_file}", # 👈 PDF
                }
            }
          }
        ]
      }
    ],
    "max_tokens": 300
  }'

```
</TabItem>
</Tabs>

## [BETA] Citations API 

Pass `citations: {"enabled": true}` to Anthropic, to get citations on your document responses. 

Note: This interface is in BETA. If you have feedback on how citations should be returned, please [tell us here](https://github.com/BerriAI/litellm/issues/7970#issuecomment-2644437943)

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion

resp = completion(
    model="claude-3-5-sonnet-20241022",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "text",
                        "media_type": "text/plain",
                        "data": "The grass is green. The sky is blue.",
                    },
                    "title": "My Document",
                    "context": "This is a trustworthy document.",
                    "citations": {"enabled": True},
                },
                {
                    "type": "text",
                    "text": "What color is the grass and sky?",
                },
            ],
        }
    ],
)

citations = resp.choices[0].message.provider_specific_fields["citations"]

assert citations is not None
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
model_list:
    - model_name: anthropic-claude
      litellm_params:
        model: anthropic/claude-3-5-sonnet-20241022
        api_key: os.environ/ANTHROPIC_API_KEY
```

2. Start proxy 

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

3. Test it! 

```bash
curl -L -X POST 'http://0.0.0.0:4000/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
  "model": "anthropic-claude",
  "messages": [
    {
        "role": "user",
        "content": [
            {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": "The grass is green. The sky is blue.",
                },
                "title": "My Document",
                "context": "This is a trustworthy document.",
                "citations": {"enabled": True},
            },
            {
                "type": "text",
                "text": "What color is the grass and sky?",
            },
        ],
    }
  ]
}'
```

</TabItem>
</Tabs>

## Usage - passing 'user_id' to Anthropic

LiteLLM translates the OpenAI `user` param to Anthropic's `metadata[user_id]` param.

<Tabs>
<TabItem value="sdk" label="SDK">

```python
response = completion(
    model="claude-3-5-sonnet-20240620",
    messages=messages,
    user="user_123",
)
```
</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml
model_list:
    - model_name: claude-3-5-sonnet-20240620
      litellm_params:
        model: anthropic/claude-3-5-sonnet-20240620
        api_key: os.environ/ANTHROPIC_API_KEY
```

2. Start Proxy

```
litellm --config /path/to/config.yaml
```

3. Test it! 

```bash
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR-LITELLM-KEY>" \
  -d '{
    "model": "claude-3-5-sonnet-20240620",
    "messages": [{"role": "user", "content": "What is Anthropic?"}],
    "user": "user_123"
  }'
```

</TabItem>
</Tabs>

