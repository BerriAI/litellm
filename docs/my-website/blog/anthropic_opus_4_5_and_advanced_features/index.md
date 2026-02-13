---
slug: anthropic_advanced_features
title: "Day 0 Support: Claude 4.5 Opus (+Advanced Features)"
date: 2025-11-25T10:00:00
authors:
  - name: Sameer Kankute
    title: SWE @ LiteLLM (LLM Translation)
    url: https://www.linkedin.com/in/sameer-kankute/
    image_url: https://pbs.twimg.com/profile_images/2001352686994907136/ONgNuSk5_400x400.jpg
  - name: Krrish Dholakia
    title: "CEO, LiteLLM"
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://pbs.twimg.com/profile_images/1298587542745358340/DZv3Oj-h_400x400.jpg
  - name: Ishaan Jaff
    title: "CTO, LiteLLM"
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://pbs.twimg.com/profile_images/1613813310264340481/lz54oEiB_400x400.jpg
description: "Guide to Claude Opus 4.5 and advanced features in LiteLLM: Tool Search, Programmatic Tool Calling, and Effort Parameter."
tags: [anthropic, claude, tool search, programmatic tool calling, effort, advanced features]
hide_table_of_contents: false
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

This guide covers Anthropic's latest model (Claude Opus 4.5) and its advanced features now available in LiteLLM: Tool Search, Programmatic Tool Calling, Tool Input Examples, and the Effort Parameter.

---

| Feature | Supported Models |
|---------|-----------------|
| Tool Search | Claude Opus 4.5, Sonnet 4.5 |
| Programmatic Tool Calling | Claude Opus 4.5, Sonnet 4.5 |
| Input Examples | Claude Opus 4.5, Sonnet 4.5 |
| Effort Parameter | Claude Opus 4.5 only |

Supported Providers: [Anthropic](../../docs/providers/anthropic), [Bedrock](../../docs/providers/bedrock), [Vertex AI](../../docs/providers/vertex_partner#vertex-ai---anthropic-claude), [Azure AI](../../docs/providers/azure_ai).

## Usage

<Tabs>
<TabItem value="sdk" label="LiteLLM Python SDK">


```python
import os
from litellm import completion

# set env - [OPTIONAL] replace with your anthropic key
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

messages = [{"role": "user", "content": "Hey! how's it going?"}]

## OPENAI /chat/completions API format
response = completion(model="claude-opus-4-5-20251101", messages=messages)
print(response)

```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

**1. Setup config.yaml**

```yaml
model_list:
  - model_name: claude-4 ### RECEIVED MODEL NAME ###
    litellm_params: # all params accepted by litellm.completion() - https://docs.litellm.ai/docs/completion/input
      model: claude-opus-4-5-20251101 ### MODEL NAME sent to `litellm.completion()` ###
      api_key: "os.environ/ANTHROPIC_API_KEY" # does os.getenv("ANTHROPIC_API_KEY")
```

**2. Start the proxy**

```bash
litellm --config /path/to/config.yaml
```

**3. Test it!**

<Tabs>
<TabItem value="curl" label="OpenAI Chat Completions">
```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data ' {
      "model": "claude-4",
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
<TabItem value="anthropic" label="Anthropic /v1/messages API">
```bash
curl --location 'http://0.0.0.0:4000/v1/messages' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data ' {
      "model": "claude-4",
      "max_tokens": 1024,
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
</Tabs>
</TabItem>
</Tabs>

## Usage - Bedrock

:::info

LiteLLM uses the boto3 library to authenticate with Bedrock.

For more ways to authenticate with Bedrock, see the [Bedrock documentation](../../docs/providers/bedrock#authentication).

:::

<Tabs>
<TabItem value="sdk" label="LiteLLM Python SDK">


```python
import os
from litellm import completion

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_REGION_NAME"] = ""

## OPENAI /chat/completions API format
response = completion(
  model="bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0",
  messages=[{ "content": "Hello, how are you?","role": "user"}]
)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

**1. Setup config.yaml**

```yaml
model_list:
  - model_name: claude-4 ### RECEIVED MODEL NAME ###
    litellm_params: # all params accepted by litellm.completion() - https://docs.litellm.ai/docs/completion/input
      model: bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0 ### MODEL NAME sent to `litellm.completion()` ###
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: os.environ/AWS_REGION_NAME
```

**2. Start the proxy**

```bash
litellm --config /path/to/config.yaml
```

**3. Test it!**

<Tabs>
<TabItem value="curl" label="OpenAI Chat Completions">
```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data ' {
      "model": "claude-4",
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
<TabItem value="anthropic" label="Anthropic /v1/messages API">
```bash
curl --location 'http://0.0.0.0:4000/v1/messages' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data ' {
      "model": "claude-4",
      "max_tokens": 1024,
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
<TabItem value="invoke" label="Bedrock /invoke API">
```bash
curl --location 'http://0.0.0.0:4000/bedrock/model/claude-4/invoke' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data ' {
      "max_tokens": 1024,
      "messages": [{"role": "user", "content": "Hello, how are you?"}]
    }'
```
</TabItem>
<TabItem value="converse" label="Bedrock /converse API">
```bash
curl --location 'http://0.0.0.0:4000/bedrock/model/claude-4/converse' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data ' {
      "messages": [{"role": "user", "content": "Hello, how are you?"}]
    }'
```
</TabItem>
</Tabs>
</TabItem>
</Tabs>


## Usage - Vertex AI


<Tabs>
<TabItem value="sdk" label="LiteLLM Python SDK">

```python
from litellm import completion
import json 

## GET CREDENTIALS 
## RUN ## 
# !gcloud auth application-default login - run this to add vertex credentials to your env
## OR ## 
file_path = 'path/to/vertex_ai_service_account.json'

# Load the JSON file
with open(file_path, 'r') as file:
    vertex_credentials = json.load(file)

# Convert to JSON string
vertex_credentials_json = json.dumps(vertex_credentials)

## COMPLETION CALL 
response = completion(
  model="vertex_ai/claude-opus-4-5@20251101",
  messages=[{ "content": "Hello, how are you?","role": "user"}],
  vertex_credentials=vertex_credentials_json,
  vertex_project="your-project-id",
  vertex_location="us-east5"
)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

**1. Setup config.yaml**

```yaml
model_list:
  - model_name: claude-4 ### RECEIVED MODEL NAME ###
    litellm_params:
        model: vertex_ai/claude-opus-4-5@20251101
        vertex_credentials: "/path/to/service_account.json"
        vertex_project: "your-project-id"
        vertex_location: "us-east5"
```

**2. Start the proxy**

```bash
litellm --config /path/to/config.yaml
```

**3. Test it!**

<Tabs>
<TabItem value="curl" label="OpenAI Chat Completions">
```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data ' {
      "model": "claude-4",
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
<TabItem value="anthropic" label="Anthropic /v1/messages API">
```bash
curl --location 'http://0.0.0.0:4000/v1/messages' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data ' {
      "model": "claude-4",
      "max_tokens": 1024,
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
</Tabs>
</TabItem>
</Tabs>

## Usage - Azure Anthropic (Azure Foundry Claude)

LiteLLM funnels Azure Claude deployments through the `azure_ai/` provider so Claude Opus models on Azure Foundry keep working with Tool Search, Effort, streaming, and the rest of the advanced feature set. Point `AZURE_AI_API_BASE` to `https://<resource>.services.ai.azure.com/anthropic` (LiteLLM appends `/v1/messages` automatically) and authenticate with `AZURE_AI_API_KEY` or an Azure AD token.

<Tabs>
<TabItem value="sdk" label="LiteLLM Python SDK">

```python
import os
from litellm import completion

# Configure Azure credentials
os.environ["AZURE_AI_API_KEY"] = "your-azure-ai-api-key"
os.environ["AZURE_AI_API_BASE"] = "https://my-resource.services.ai.azure.com/anthropic"

response = completion(
    model="azure_ai/claude-opus-4-1",
    messages=[{"role": "user", "content": "Explain how Azure Anthropic hosts Claude Opus differently from the public Anthropic API."}],
    max_tokens=1200,
    temperature=0.7,
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

**1. Set environment variables**

```bash
export AZURE_AI_API_KEY="your-azure-ai-api-key"
export AZURE_AI_API_BASE="https://my-resource.services.ai.azure.com/anthropic"
```

**2. Configure the proxy**

```yaml
model_list:
  - model_name: claude-4-azure
    litellm_params:
      model: azure_ai/claude-opus-4-1
      api_key: os.environ/AZURE_AI_API_KEY
      api_base: os.environ/AZURE_AI_API_BASE
```

**3. Start LiteLLM**

```bash
litellm --config /path/to/config.yaml
```

**4. Test the Azure Claude route**

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Bearer $LITELLM_KEY' \
  --data '{
    "model": "claude-4-azure",
    "messages": [
      {
        "role": "user",
        "content": "How do I use Claude Opus 4 via Azure Anthropic in LiteLLM?"
      }
    ],
    "max_tokens": 1024
  }'
```

</TabItem>
</Tabs>


## Tool Search {#tool-search}

This lets Claude work with thousands of tools, by dynamically loading tools on-demand, instead of loading all tools into the context window upfront.

### Usage Example

<Tabs>
<TabItem value="sdk" label="LiteLLM Python SDK">

```python
import litellm
import os

# Configure your API key
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

# Define your tools with defer_loading
tools = [
    # Tool search tool (regex variant)
    {
        "type": "tool_search_tool_regex_20251119",
        "name": "tool_search_tool_regex"
    },
    # Deferred tools - loaded on-demand
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location. Returns temperature and conditions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        },
        "defer_loading": True  # Load on-demand
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search through files in the workspace using keywords",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "file_types": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["query"]
            }
        },
        "defer_loading": True
    },
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Execute SQL queries against the database",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string"}
                },
                "required": ["sql"]
            }
        },
        "defer_loading": True
    }
]

# Make a request - Claude will search for and use relevant tools
response = litellm.completion(
    model="anthropic/claude-opus-4-5-20251101",
    messages=[{
        "role": "user",
        "content": "What's the weather like in San Francisco?"
    }],
    tools=tools
)

print("Claude's response:", response.choices[0].message.content)
print("Tool calls:", response.choices[0].message.tool_calls)

# Check tool search usage
if hasattr(response.usage, 'server_tool_use'):
    print(f"Tool searches performed: {response.usage.server_tool_use.tool_search_requests}")
```
</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

1. Setup config.yaml

```yaml
model_list:
  - model_name: claude-4
    litellm_params:
      model: anthropic/claude-opus-4-5-20251101
      api_key: os.environ/ANTHROPIC_API_KEY
```

2. Start the proxy

```bash
litellm --config /path/to/config.yaml
```

3. Test it!


```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data ' {
      "model": "claude-4",
      "messages": [{
        "role": "user",
        "content": "What's the weather like in San Francisco?"
       }],
       "tools": [
        # Tool search tool (regex variant)
        {
            "type": "tool_search_tool_regex_20251119",
            "name": "tool_search_tool_regex"
        },
        # Deferred tools - loaded on-demand
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location. Returns temperature and conditions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit"
                        }
                    },
                    "required": ["location"]
                }
            },
            "defer_loading": True  # Load on-demand
        },
        {
            "type": "function",
            "function": {
                "name": "search_files",
                "description": "Search through files in the workspace using keywords",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "file_types": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["query"]
                }
            },
            "defer_loading": True
        },
        {
            "type": "function",
            "function": {
                "name": "query_database",
                "description": "Execute SQL queries against the database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string"}
                    },
                    "required": ["sql"]
                }
            },
            "defer_loading": True
        }
    ]
}
'
```
</TabItem>
</Tabs>

### BM25 Variant (Natural Language Search)

For natural language queries instead of regex patterns:

```python
tools = [
    {
        "type": "tool_search_tool_bm25_20251119",  # Natural language variant
        "name": "tool_search_tool_bm25"
    },
    # ... your deferred tools
]
```

---

## Programmatic Tool Calling {#programmatic-tool-calling}

Programmatic tool calling allows Claude to write code that calls your tools programmatically. [Learn more](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling)

<Tabs>
<TabItem value="sdk" label="LiteLLM Python SDK">

```python
import litellm
import json

# Define tools that can be called programmatically
tools = [
    # Code execution tool (required for programmatic calling)
    {
        "type": "code_execution_20250825",
        "name": "code_execution"
    },
    # Tool that can be called from code
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Execute a SQL query against the sales database. Returns a list of rows as JSON objects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "SQL query to execute"
                    }
                },
                "required": ["sql"]
            }
        },
        "allowed_callers": ["code_execution_20250825"]  # Enable programmatic calling
    }
]

# First request
response = litellm.completion(
    model="anthropic/claude-sonnet-4-5-20250929",
    messages=[{
        "role": "user",
        "content": "Query sales data for West, East, and Central regions, then tell me which had the highest revenue"
    }],
    tools=tools
)

print("Claude's response:", response.choices[0].message)

# Handle tool calls
messages = [
    {"role": "user", "content": "Query sales data for West, East, and Central regions, then tell me which had the highest revenue"},
    {"role": "assistant", "content": response.choices[0].message.content, "tool_calls": response.choices[0].message.tool_calls}
]

# Process each tool call
for tool_call in response.choices[0].message.tool_calls:
    # Check if it's a programmatic call
    if hasattr(tool_call, 'caller') and tool_call.caller:
        print(f"Programmatic call to {tool_call.function.name}")
        print(f"Called from: {tool_call.caller}")
    
    # Simulate tool execution
    if tool_call.function.name == "query_database":
        args = json.loads(tool_call.function.arguments)
        # Simulate database query
        result = json.dumps([
            {"region": "West", "revenue": 150000},
            {"region": "East", "revenue": 180000},
            {"region": "Central", "revenue": 120000}
        ])
        
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": result
            }]
        })

# Get final response
final_response = litellm.completion(
    model="anthropic/claude-sonnet-4-5-20250929",
    messages=messages,
    tools=tools
)

print("\nFinal answer:", final_response.choices[0].message.content)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

1. Setup config.yaml

```yaml
model_list:
  - model_name: claude-4
    litellm_params:
      model: anthropic/claude-opus-4-5-20251101
      api_key: os.environ/ANTHROPIC_API_KEY
```

2. Start the proxy

```bash
litellm --config /path/to/config.yaml
```

3. Test it!


```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data ' {
      "model": "claude-4",
      "messages": [{
        "role": "user",
        "content": "Query sales data for West, East, and Central regions, then tell me which had the highest revenue"
      }],
      "tools": [
        # Code execution tool (required for programmatic calling)
        {
            "type": "code_execution_20250825",
            "name": "code_execution"
        },
        # Tool that can be called from code
        {
            "type": "function",
            "function": {
                "name": "query_database",
                "description": "Execute a SQL query against the sales database. Returns a list of rows as JSON objects.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "SQL query to execute"
                        }
                    },
                    "required": ["sql"]
                }
            },
            "allowed_callers": ["code_execution_20250825"]  # Enable programmatic calling
        }
    ]
}
'
```
</TabItem>
</Tabs>

---

## Tool Input Examples {#tool-input-examples}

You can now provide Claude with examples of how to use your tools. [Learn more](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-input-examples)


<Tabs>
<TabItem value="sdk" label="LiteLLM Python SDK">

```python
import litellm

tools = [
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Create a new calendar event with attendees and reminders",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "start_time": {
                        "type": "string",
                        "description": "ISO 8601 format: YYYY-MM-DDTHH:MM:SS"
                    },
                    "duration_minutes": {"type": "integer"},
                    "attendees": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "email": {"type": "string"},
                                "optional": {"type": "boolean"}
                            }
                        }
                    },
                    "reminders": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "minutes_before": {"type": "integer"},
                                "method": {"type": "string", "enum": ["email", "popup"]}
                            }
                        }
                    }
                },
                "required": ["title", "start_time", "duration_minutes"]
            }
        },
        # Provide concrete examples
        "input_examples": [
            {
                "title": "Team Standup",
                "start_time": "2025-01-15T09:00:00",
                "duration_minutes": 30,
                "attendees": [
                    {"email": "alice@company.com", "optional": False},
                    {"email": "bob@company.com", "optional": False}
                ],
                "reminders": [
                    {"minutes_before": 15, "method": "popup"}
                ]
            },
            {
                "title": "Lunch Break",
                "start_time": "2025-01-15T12:00:00",
                "duration_minutes": 60
                # Demonstrates optional fields can be omitted
            }
        ]
    }
]

response = litellm.completion(
    model="anthropic/claude-sonnet-4-5-20250929",
    messages=[{
        "role": "user",
        "content": "Schedule a team meeting for tomorrow at 2pm for 45 minutes with john@company.com and sarah@company.com"
    }],
    tools=tools
)

print("Tool call:", response.choices[0].message.tool_calls[0].function.arguments)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

1. Setup config.yaml

```yaml
model_list:
  - model_name: claude-4
    litellm_params:
      model: anthropic/claude-opus-4-5-20251101
      api_key: os.environ/ANTHROPIC_API_KEY
```

2. Start the proxy

```bash
litellm --config /path/to/config.yaml
```

3. Test it!


```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data ' {
      "model": "claude-4",
      "messages": [{
        "role": "user",
        "content": "Schedule a team meeting for tomorrow at 2pm for 45 minutes with john@company.com and sarah@company.com"
      }],
      "tools": [
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Create a new calendar event with attendees and reminders",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "start_time": {
                        "type": "string",
                        "description": "ISO 8601 format: YYYY-MM-DDTHH:MM:SS"
                    },
                    "duration_minutes": {"type": "integer"},
                    "attendees": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "email": {"type": "string"},
                                "optional": {"type": "boolean"}
                            }
                        }
                    },
                    "reminders": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "minutes_before": {"type": "integer"},
                                "method": {"type": "string", "enum": ["email", "popup"]}
                            }
                        }
                    }
                },
                "required": ["title", "start_time", "duration_minutes"]
            }
        },
        # Provide concrete examples
        "input_examples": [
            {
                "title": "Team Standup",
                "start_time": "2025-01-15T09:00:00",
                "duration_minutes": 30,
                "attendees": [
                    {"email": "alice@company.com", "optional": False},
                    {"email": "bob@company.com", "optional": False}
                ],
                "reminders": [
                    {"minutes_before": 15, "method": "popup"}
                ]
            },
            {
                "title": "Lunch Break",
                "start_time": "2025-01-15T12:00:00",
                "duration_minutes": 60
                # Demonstrates optional fields can be omitted
            }
        ]
    }
]
}
'
```
</TabItem>
</Tabs>

---

## Effort Parameter: Control Token Usage {#effort-parameter}

Control how much effort Claude puts into its response using the `reasoning_effort` parameter. This allows you to trade off between response thoroughness and token efficiency.

:::info
LiteLLM automatically maps `reasoning_effort` to Anthropic's `output_config` format and adds the required `effort-2025-11-24` beta header for Claude Opus 4.5.
:::

Potential values for `reasoning_effort` parameter: `"high"`, `"medium"`, `"low"`.

### Usage Example

<Tabs>
<TabItem value="sdk" label="LiteLLM Python SDK">

```python
import litellm

message = "Analyze the trade-offs between microservices and monolithic architectures"

# High effort (default) - Maximum capability
response_high = litellm.completion(
    model="anthropic/claude-opus-4-5-20251101",
    messages=[{"role": "user", "content": message}],
    reasoning_effort="high"
)

print("High effort response:")
print(response_high.choices[0].message.content)
print(f"Tokens used: {response_high.usage.completion_tokens}\n")

# Medium effort - Balanced approach
response_medium = litellm.completion(
    model="anthropic/claude-opus-4-5-20251101",
    messages=[{"role": "user", "content": message}],
    reasoning_effort="medium"
)

print("Medium effort response:")
print(response_medium.choices[0].message.content)
print(f"Tokens used: {response_medium.usage.completion_tokens}\n")

# Low effort - Maximum efficiency
response_low = litellm.completion(
    model="anthropic/claude-opus-4-5-20251101",
    messages=[{"role": "user", "content": message}],
    reasoning_effort="low"
)

print("Low effort response:")
print(response_low.choices[0].message.content)
print(f"Tokens used: {response_low.usage.completion_tokens}\n")

# Compare token usage
print("Token Comparison:")
print(f"High:   {response_high.usage.completion_tokens} tokens")
print(f"Medium: {response_medium.usage.completion_tokens} tokens")
print(f"Low:    {response_low.usage.completion_tokens} tokens")
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

1. Setup config.yaml

```yaml
model_list:
  - model_name: claude-4
    litellm_params:
      model: anthropic/claude-opus-4-5-20251101
      api_key: os.environ/ANTHROPIC_API_KEY
```

2. Start the proxy

```bash
litellm --config /path/to/config.yaml
```

3. Test it!

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer $LITELLM_KEY' \
--data ' {
      "model": "claude-4",
      "messages": [{
        "role": "user",
        "content": "Analyze the trade-offs between microservices and monolithic architectures"
      }],
      "reasoning_effort": "high"
    }
'
```
</TabItem>
</Tabs>
