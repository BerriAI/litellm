import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# /mcp Model Context Protocol [BETA]

## Overview

LiteLLM's MCP implementation allows you to define tools that can be called by any MCP compatible client. Define your `mcp_tools` with LiteLLM and all your clients can `list` and `call` available tools. 

## How it works 

LiteLLM exposes the following MCP endpoints:

- `/mcp/list_tools` - List all available tools
- `/mcp/call_tool` - Call a specific tool with the provided arguments

When MCP clients connect to LiteLLM they can follow this workflow:

1. Connect to the LiteLLM MCP server
2. List all available tools on LiteLLM
3. Client makes LLM API request with tool call(s)
4. LLM API returns which tools to call and with what arguments
5. MCP client makes tool calls to LiteLLM
6. LiteLLM makes the tool calls to the appropriate handlers
7. LiteLLM returns the tool call results to the MCP client

## Quick Start

### 1. Define your tools on mcp_tools

LiteLLM allows you to define your tools on the `mcp_tools` section in your config.yaml file. All tools listed here will be available to MCP clients (when they connect to LiteLLM and call `list_tools`).

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: sk-xxxxxxx



mcp_tools:
  - name: "get_current_time"
    description: "Get the current time"
    input_schema: {
      "type": "object",
      "properties": {
        "format": {
          "type": "string",
          "description": "The format of the time to return",
          "enum": ["short"]
        }
      }
    }
    handler: "mcp_tools.get_current_time"
```

### 2. Define a handler for your tool

Create a new file called `mcp_tools.py` and add this code. The key method here is `get_current_time` which gets executed when the `get_current_time` tool is called.

```python
# mcp_tools.py

from datetime import datetime

def get_current_time(format: str = "short"):
    """
    Simple handler for the 'get_current_time' tool.
    
    Args:
        format (str): The format of the time to return ('short').
    
    Returns:
        str: The current time formatted as 'HH:MM'.
    """
    # Get the current time
    current_time = datetime.now()
    
    # Format the time as 'HH:MM'
    return current_time.strftime('%H:%M')
```

### 3. Start LiteLLM Gateway

<Tabs>
<TabItem value="docker" label="Docker Run">

Mount your `mcp_tools.py` on the LiteLLM Docker container.

```shell
docker run -d \
  -p 4000:4000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  --name my-app \
  -v $(pwd)/my_config.yaml:/app/config.yaml \
  -v $(pwd)/mcp_tools.py:/app/mcp_tools.py \
  my-app:latest \
  --config /app/config.yaml \
  --port 4000 \
  --detailed_debug \
```

</TabItem>

<TabItem value="py" label="litellm pip">

```shell
litellm --config config.yaml --detailed_debug
```

</TabItem>
</Tabs>


### 3. Make an LLM API request 



```python
import asyncio
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession
from mcp.client.sse import sse_client


async def main():
    # Initialize the model with your API key
    model = ChatOpenAI(model="gpt-4o")
    
    # Connect to the MCP server
    async with sse_client(url="http://localhost:4000/mcp/") as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            print("Initializing session...")
            await session.initialize()
            print("Session initialized")

            # Load available tools from MCP
            print("Loading tools...")
            tools = await load_mcp_tools(session)
            print(f"Loaded {len(tools)} tools")

            # Create a ReAct agent with the model and tools
            agent = create_react_agent(model, tools)
            
            # Run the agent with a user query
            user_query = "What's the weather in Tokyo?"
            print(f"Asking: {user_query}")
            agent_response = await agent.ainvoke({"messages": user_query})
            print("Agent response:")
            print(agent_response)


if __name__ == "__main__":
    asyncio.run(main())

```


## Specification for `mcp_tools`

The `mcp_tools` section in your LiteLLM config defines tools that can be called by MCP-compatible clients.

### Tool Definition Format

```yaml
mcp_tools:
  - name: string                # Required: Name of the tool
    description: string         # Required: Description of what the tool does
    input_schema: object        # Required: JSON Schema defining the tool's input parameters
    handler: string             # Required: Path to the function that implements the tool
```

### Field Details

- `name`: A unique identifier for the tool
- `description`: A clear description of what the tool does, used by LLMs to determine when to call it
- `input_schema`: JSON Schema object defining the expected input parameters
- `handler`: String path to the Python function that implements the tool (e.g., "module.submodule.function_name")

### Example Tool Definition

```yaml
mcp_tools:
  - name: "get_current_time"
    description: "Get the current time in a specified format"
    input_schema: {
      "type": "object",
      "properties": {
        "format": {
          "type": "string",
          "description": "The format of the time to return",
          "enum": ["short", "long", "iso"]
        },
        "timezone": {
          "type": "string",
          "description": "The timezone to use (e.g., 'UTC', 'America/New_York')",
          "default": "UTC"
        }
      },
      "required": ["format"]
    }
    handler: "mcp_tools.get_current_time"
```
