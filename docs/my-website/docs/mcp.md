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

### 2. Start LiteLLM Proxy Server

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



