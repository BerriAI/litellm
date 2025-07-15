import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import Image from '@theme/IdealImage';

# /mcp [BETA] - Model Context Protocol

LiteLLM Proxy provides an MCP Gateway that allows you to use a fixed endpoint for all MCP tools and control MCP access by Key, Team. 

<Image 
  img={require('../img/mcp_2.png')}
  style={{width: '100%', display: 'block', margin: '2rem auto'}}
/>
<p style={{textAlign: 'left', color: '#666'}}>
  LiteLLM MCP Architecture: Use MCP tools with all LiteLLM supported models
</p>

## Overview
| Feature | Description |
|---------|-------------|
| MCP Operations | • List Tools<br/>• Call Tools |
| Supported MCP Transports | • Streamable HTTP<br/>• SSE<br/>• Standard Input/Output (stdio) |
| MCP Tool Cost Tracking | ✅ Supported |
| Grouping MCPs (Access Groups) | ✅ Supported |
| LiteLLM Permission Management | ✨ Enterprise Only<br/>• By Key<br/>• By Team<br/>• By Organization |

## Adding your MCP

<Tabs>
<TabItem value="ui" label="LiteLLM UI">

On the LiteLLM UI, Navigate to "MCP Servers" and click "Add New MCP Server".

On this form, you should enter your MCP Server URL and the transport you want to use.

LiteLLM supports the following MCP transports:
- Streamable HTTP
- SSE (Server-Sent Events)
- Standard Input/Output (stdio)

<Image 
  img={require('../img/add_mcp.png')}
  style={{width: '80%', display: 'block', margin: '0'}}
/>

#### Adding a stdio MCP Server

For stdio MCP servers, select "Standard Input/Output (stdio)" as the transport type and provide the stdio configuration in JSON format:

<Image 
  img={require('../img/add_stdio_mcp.png')}
  style={{width: '80%', display: 'block', margin: '0'}}
/>

</TabItem>

<TabItem value="config" label="config.yaml">

Add your MCP servers directly in your `config.yaml` file:

```yaml title="config.yaml" showLineNumbers
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: sk-xxxxxxx

mcp_servers:
  # HTTP Streamable Server
  deepwiki_mcp:
    url: "https://mcp.deepwiki.com/mcp"
  # SSE Server
  zapier_mcp:
    url: "https://actions.zapier.com/mcp/sk-akxxxxx/sse"
  
  # Standard Input/Output (stdio) Server - CircleCI Example
  circleci_mcp:
    transport: "stdio"
    command: "npx"
    args: ["-y", "@circleci/mcp-server-circleci"]
    env:
      CIRCLECI_TOKEN: "your-circleci-token"
      CIRCLECI_BASE_URL: "https://circleci.com"
  
  # Full configuration with all optional fields
  my_http_server:
    url: "https://my-mcp-server.com/mcp"
    transport: "http"
    description: "My custom MCP server"
    auth_type: "api_key"
    spec_version: "2025-03-26"
```

**Configuration Options:**
- **Server Name**: Use any descriptive name for your MCP server (e.g., `zapier_mcp`, `deepwiki_mcp`, `circleci_mcp`)
- **URL**: The endpoint URL for your MCP server (required for HTTP/SSE transports)
- **Transport**: Optional transport type (defaults to `sse`)
  - `sse` - SSE (Server-Sent Events) transport
  - `http` - Streamable HTTP transport
  - `stdio` - Standard Input/Output transport
- **Command**: The command to execute for stdio transport (required for stdio)
- **Args**: Array of arguments to pass to the command (optional for stdio)
- **Env**: Environment variables to set for the stdio process (optional for stdio)
- **Description**: Optional description for the server
- **Auth Type**: Optional authentication type
- **Spec Version**: Optional MCP specification version (defaults to `2025-03-26`)

</TabItem>
</Tabs>


## Using your MCP

<Tabs>
<TabItem value="openai" label="OpenAI API">

### Quick Start
#### Connect via OpenAI Responses API

Use the OpenAI Responses API to connect to your LiteLLM MCP server:

```bash title="cURL Example" showLineNumbers
curl --location 'https://api.openai.com/v1/responses' \
--header 'Content-Type: application/json' \
--header "Authorization: Bearer $OPENAI_API_KEY" \
--data '{
    "model": "gpt-4o",
    "tools": [
        {
            "type": "mcp",
            "server_label": "litellm",
            "server_url": "litellm_proxy",
            "require_approval": "never",
            "headers": {
                "x-litellm-api-key": "Bearer YOUR_LITELLM_API_KEY"
            }
        }
    ],
    "input": "Run available tools",
    "tool_choice": "required"
}'
```

</TabItem>

<TabItem value="litellm" label="LiteLLM Proxy">

#### Connect via LiteLLM Proxy Responses API

Use this when calling LiteLLM Proxy for LLM API requests to `/v1/responses` endpoint.

```bash title="cURL Example" showLineNumbers
curl --location '<your-litellm-proxy-base-url>/v1/responses' \
--header 'Content-Type: application/json' \
--header "Authorization: Bearer $LITELLM_API_KEY" \
--data '{
    "model": "gpt-4o",
    "tools": [
        {
            "type": "mcp",
            "server_label": "litellm",
            "server_url": "litellm_proxy",
            "require_approval": "never",
            "headers": {
                "x-litellm-api-key": "Bearer YOUR_LITELLM_API_KEY"
            }
        }
    ],
    "input": "Run available tools",
    "tool_choice": "required"
}'
```

</TabItem>

<TabItem value="cursor" label="Cursor IDE">

#### Connect via Cursor IDE

Use tools directly from Cursor IDE with LiteLLM MCP:

**Setup Instructions:**

1. **Open Cursor Settings**: Use `⇧+⌘+J` (Mac) or `Ctrl+Shift+J` (Windows/Linux)
2. **Navigate to MCP Tools**: Go to the "MCP Tools" tab and click "New MCP Server"
3. **Add Configuration**: Copy and paste the JSON configuration below, then save with `Cmd+S` or `Ctrl+S`

```json title="Basic Cursor MCP Configuration" showLineNumbers
{
  "mcpServers": {
    "LiteLLM": {
      "url": "litellm_proxy",
      "headers": {
        "x-litellm-api-key": "Bearer $LITELLM_API_KEY"
      }
    }
  }
}
```

</TabItem>
</Tabs>

#### How it works when server_url="litellm_proxy"

When server_url="litellm_proxy", LiteLLM bridges non-MCP providers to your MCP tools.

- Tool Discovery: LiteLLM fetches MCP tools and converts them to OpenAI-compatible definitions
- LLM Call: Tools are sent to the LLM with your input; LLM selects which tools to call
- Tool Execution: LiteLLM automatically parses arguments, routes calls to MCP servers, executes tools, and retrieves results
- Response Integration: Tool results are sent back to LLM for final response generation
- Output: Complete response combining LLM reasoning with tool execution results

This enables MCP tool usage with any LiteLLM-supported provider, regardless of native MCP support.

#### Auto-execution for require_approval: "never"

Setting require_approval: "never" triggers automatic tool execution, returning the final response in a single API call without additional user interaction.



### Specific MCP Servers

You can choose to access specific MCP servers and only list their tools using the `x-mcp-servers` header. This header allows you to:
- Limit tool access to one or more specific MCP servers
- Control which tools are available in different environments or use cases

The header accepts a comma-separated list of server names: `"Zapier_Gmail,Server2,Server3"`

Notes:
- Server names with spaces should be replaced with underscores
- If the header is not provided, tools from all available MCP servers will be accessible

<Tabs>
<TabItem value="openai" label="OpenAI API">

```bash title="cURL Example with Server Segregation" showLineNumbers
curl --location 'https://api.openai.com/v1/responses' \
--header 'Content-Type: application/json' \
--header "Authorization: Bearer $OPENAI_API_KEY" \
--data '{
    "model": "gpt-4o",
    "tools": [
        {
            "type": "mcp",
            "server_label": "litellm",
            "server_url": "litellm_proxy",
            "require_approval": "never",
            "headers": {
                "x-litellm-api-key": "Bearer YOUR_LITELLM_API_KEY",
                "x-mcp-servers": "Zapier_Gmail"
            }
        }
    ],
    "input": "Run available tools",
    "tool_choice": "required"
}'
```

In this example, the request will only have access to tools from the "Zapier_Gmail" MCP server.

</TabItem>

<TabItem value="litellm" label="LiteLLM Proxy">

```bash title="cURL Example with Server Segregation" showLineNumbers
curl --location '<your-litellm-proxy-base-url>/v1/responses' \
--header 'Content-Type: application/json' \
--header "Authorization: Bearer $LITELLM_API_KEY" \
--data '{
    "model": "gpt-4o",
    "tools": [
        {
            "type": "mcp",
            "server_label": "litellm",
            "server_url": "litellm_proxy",
            "require_approval": "never",
            "headers": {
                "x-litellm-api-key": "Bearer YOUR_LITELLM_API_KEY",
                "x-mcp-servers": "Zapier_Gmail,Server2"
            }
        }
    ],
    "input": "Run available tools",
    "tool_choice": "required"
}'
```

This configuration restricts the request to only use tools from the specified MCP servers.

</TabItem>

<TabItem value="cursor" label="Cursor IDE">

```json title="Cursor MCP Configuration with Server Segregation" showLineNumbers
{
  "mcpServers": {
    "LiteLLM": {
      "url": "litellm_proxy",
      "headers": {
        "x-litellm-api-key": "Bearer $LITELLM_API_KEY",
        "x-mcp-servers": "Zapier_Gmail,Server2"
      }
    }
  }
}
```

This configuration in Cursor IDE settings will limit tool access to only the specified MCP server.

</TabItem>
</Tabs>

### Grouping MCPs (Access Groups)

MCP Access Groups allow you to group multiple MCP servers together for easier management.

#### 1. Create an Access Group

To create an access group:
- Go to MCP Servers in the LiteLLM UI
- Click "Add a New MCP Server" 
- Under "MCP Access Groups", create a new group (e.g., "dev_group") by typing it
- Add the same group name to other servers to group them together

<Image 
  img={require('../img/mcp_create_access_group.png')}
  style={{width: '80%', display: 'block', margin: '0'}}
/>

#### 2. Use Access Group in Cursor

Include the access group name in the `x-mcp-servers` header:

```json title="Cursor Configuration with Access Groups" showLineNumbers
{
  "mcpServers": {
    "LiteLLM": {
      "url": "litellm_proxy",
      "headers": {
        "x-litellm-api-key": "Bearer $LITELLM_API_KEY",
        "x-mcp-servers": "dev_group"
      }
    }
  }
}
```

This gives you access to all servers in the "dev_group" access group.

#### Advanced: Connecting Access Groups to API Keys

When creating API keys, you can assign them to specific access groups for permission management:

- Go to "Keys" in the LiteLLM UI and click "Create Key"
- Select the desired MCP access groups from the dropdown
- The key will have access to all MCP servers in those groups
- This is reflected in the Test Key page

<Image 
  img={require('../img/mcp_key_access_group.png')}
  style={{width: '80%', display: 'block', margin: '0'}}
/>


## Using your MCP with client side credentials

Use this if you want to pass a client side authentication token to LiteLLM to then pass to your MCP to auth to your MCP.

You can specify your MCP auth token using the header `x-mcp-auth`. LiteLLM will forward this token to your MCP server for authentication.

<Tabs>
<TabItem value="openai" label="OpenAI API">

#### Connect via OpenAI Responses API with MCP Auth

Use the OpenAI Responses API and include the `x-mcp-auth` header for your MCP server authentication:

```bash title="cURL Example with MCP Auth" showLineNumbers
curl --location 'https://api.openai.com/v1/responses' \
--header 'Content-Type: application/json' \
--header "Authorization: Bearer $OPENAI_API_KEY" \
--data '{
    "model": "gpt-4o",
    "tools": [
        {
            "type": "mcp",
            "server_label": "litellm",
            "server_url": "litellm_proxy",
            "require_approval": "never",
            "headers": {
                "x-litellm-api-key": "Bearer YOUR_LITELLM_API_KEY",
                "x-mcp-auth": YOUR_MCP_AUTH_TOKEN
            }
        }
    ],
    "input": "Run available tools",
    "tool_choice": "required"
}'
```

</TabItem>

<TabItem value="litellm" label="LiteLLM Proxy">

#### Connect via LiteLLM Proxy Responses API with MCP Auth

Use this when calling LiteLLM Proxy for LLM API requests to `/v1/responses` endpoint with MCP authentication:

```bash title="cURL Example with MCP Auth" showLineNumbers
curl --location '<your-litellm-proxy-base-url>/v1/responses' \
--header 'Content-Type: application/json' \
--header "Authorization: Bearer $LITELLM_API_KEY" \
--data '{
    "model": "gpt-4o",
    "tools": [
        {
            "type": "mcp",
            "server_label": "litellm",
            "server_url": "litellm_proxy",
            "require_approval": "never",
            "headers": {
                "x-litellm-api-key": "Bearer YOUR_LITELLM_API_KEY",
                "x-mcp-auth": "YOUR_MCP_AUTH_TOKEN"
            }
        }
    ],
    "input": "Run available tools",
    "tool_choice": "required"
}'
```

</TabItem>

<TabItem value="cursor" label="Cursor IDE">

#### Connect via Cursor IDE with MCP Auth

Use tools directly from Cursor IDE with LiteLLM MCP and include your MCP authentication token:

**Setup Instructions:**

1. **Open Cursor Settings**: Use `⇧+⌘+J` (Mac) or `Ctrl+Shift+J` (Windows/Linux)
2. **Navigate to MCP Tools**: Go to the "MCP Tools" tab and click "New MCP Server"
3. **Add Configuration**: Copy and paste the JSON configuration below, then save with `Cmd+S` or `Ctrl+S`

```json title="Cursor MCP Configuration with Auth" showLineNumbers
{
  "mcpServers": {
    "LiteLLM": {
      "url": "litellm_proxy",
      "headers": {
        "x-litellm-api-key": "Bearer $LITELLM_API_KEY",
        "x-mcp-auth": "$MCP_AUTH_TOKEN"
      }
    }
  }
}
```

</TabItem>

<TabItem value="http" label="Streamable HTTP">

#### Connect via Streamable HTTP Transport with MCP Auth

Connect to LiteLLM MCP using HTTP transport with MCP authentication:

**Server URL:**
```text showLineNumbers
litellm_proxy
```

**Headers:**
```text showLineNumbers
x-litellm-api-key: Bearer YOUR_LITELLM_API_KEY
x-mcp-auth: Bearer YOUR_MCP_AUTH_TOKEN
```

This URL can be used with any MCP client that supports HTTP transport. The `x-mcp-auth` header will be forwarded to your MCP server for authentication.

</TabItem>

<TabItem value="fastmcp" label="Python FastMCP">

#### Connect via Python FastMCP Client with MCP Auth

Use the Python FastMCP client to connect to your LiteLLM MCP server with MCP authentication:

```python title="Python FastMCP Example with MCP Auth" showLineNumbers
import asyncio
import json

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

# Create the transport with your LiteLLM MCP server URL and auth headers
server_url = "litellm_proxy"
transport = StreamableHttpTransport(
    server_url,
    headers={
        "x-litellm-api-key": "Bearer YOUR_LITELLM_API_KEY",
        "x-mcp-auth": "Bearer YOUR_MCP_AUTH_TOKEN"
    }
)

# Initialize the client with the transport
client = Client(transport=transport)


async def main():
    # Connection is established here
    print("Connecting to LiteLLM MCP server with authentication...")
    async with client:
        print(f"Client connected: {client.is_connected()}")

        # Make MCP calls within the context
        print("Fetching available tools...")
        tools = await client.list_tools()

        print(f"Available tools: {json.dumps([t.name for t in tools], indent=2)}")
        
        # Example: Call a tool (replace 'tool_name' with an actual tool name)
        if tools:
            tool_name = tools[0].name
            print(f"Calling tool: {tool_name}")
            
            # Call the tool with appropriate arguments
            result = await client.call_tool(tool_name, arguments={})
            print(f"Tool result: {result}")


# Run the example
if __name__ == "__main__":
    asyncio.run(main())
```

</TabItem>
</Tabs>


### Customize the MCP Auth Header Name

By default, LiteLLM uses `x-mcp-auth` to pass your credentials to MCP servers. You can change this header name in one of the following ways:
1. Set the `LITELLM_MCP_CLIENT_SIDE_AUTH_HEADER_NAME` environment variable

```bash title="Environment Variable" showLineNumbers
export LITELLM_MCP_CLIENT_SIDE_AUTH_HEADER_NAME="authorization"
```


2. Set the `mcp_client_side_auth_header_name` in the general settings on the config.yaml file

```yaml title="config.yaml" showLineNumbers
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: sk-xxxxxxx

general_settings:
  mcp_client_side_auth_header_name: "authorization"
```

#### Using the authorization header

In this example the `authorization` header will be passed to the MCP server for authentication.

```bash title="cURL with authorization header" showLineNumbers
curl --location '<your-litellm-proxy-base-url>/v1/responses' \
--header 'Content-Type: application/json' \
--header "Authorization: Bearer $LITELLM_API_KEY" \
--data '{
    "model": "gpt-4o",
    "tools": [
        {
            "type": "mcp",
            "server_label": "litellm",
            "server_url": "litellm_proxy",
            "require_approval": "never",
            "headers": {
                "x-litellm-api-key": "Bearer YOUR_LITELLM_API_KEY",
                "authorization": "Bearer sk-zapier-token-123"
            }
        }
    ],
    "input": "Run available tools",
    "tool_choice": "required"
}'
```



## MCP Cost Tracking

LiteLLM provides cost tracking for MCP tool calls, allowing you to monitor and control expenses associated with MCP operations. You can configure costs at two levels:

- **Default cost per tool**: Set a uniform cost for all tools from a specific MCP server
- **Tool-specific costs**: Define individual costs for specific tools (e.g., `search_tool` costs $10, while `get_weather` costs $5)

### Configure cost tracking

LiteLLM offers two approaches to track MCP tool costs, each designed for different use cases:

| Method | Best For | Capabilities |
|--------|----------|-------------|
| **UI/Config-based Cost Tracking** | Simple, static cost tracking scenarios | • Set default costs for all server tools<br/>• Configure individual tool costs<br/>• Automatic cost tracking based on configuration |
| **Custom Post-MCP Hook** | Dynamic, complex cost tracking requirements | • Custom cost calculation logic<br/>• Real-time cost adjustments<br/>• Response modification capabilities |

### Configuration on UI/config.yaml

<Tabs>
<TabItem value="ui" label="LiteLLM UI">

On the UI when adding a new MCP server, you can navigate to the "Cost Configuration" tab to configure the cost for the MCP server.

<Image 
  img={require('../img/mcp_cost.png')}
  style={{width: '80%', display: 'block', margin: '0'}}
/>

</TabItem>

<TabItem value="config" label="config.yaml">

Configure fixed costs for MCP servers directly in your config.yaml:

```yaml title="config.yaml" showLineNumbers
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: sk-xxxxxxx

mcp_servers:
  zapier_server:
    url: "https://actions.zapier.com/mcp/sk-xxxxx/sse"
    mcp_info:
      mcp_server_cost_info:
        # Default cost for all tools in this server
        default_cost_per_query: 0.01
        # Custom cost for specific tools
        tool_name_to_cost_per_query:
          send_email: 0.05
          create_document: 0.03
          
  expensive_api_server:
    url: "https://api.expensive-service.com/mcp"
    mcp_info:
      mcp_server_cost_info:
        default_cost_per_query: 1.50
```

</TabItem>
</Tabs>

### Custom Post-MCP Hook

Use this when you need dynamic cost calculation or want to modify the MCP response before it's returned to the user.

#### 1. Create a custom MCP hook file

```python title="custom_mcp_hook.py" showLineNumbers
from typing import Optional
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.mcp import MCPPostCallResponseObject


class CustomMCPCostTracker(CustomLogger):
    """
    Custom handler for MCP cost tracking and response modification
    """
    
    async def async_post_mcp_tool_call_hook(
        self, 
        kwargs, 
        response_obj: MCPPostCallResponseObject, 
        start_time, 
        end_time
    ) -> Optional[MCPPostCallResponseObject]:
        """
        Called after each MCP tool call. 
        Modify costs and response before returning to user.
        """
        
        # Extract tool information from kwargs
        tool_name = kwargs.get("name", "")
        server_name = kwargs.get("server_name", "")
        
        # Calculate custom cost based on your logic
        custom_cost = 42.00
        
        # Set the response cost
        response_obj.hidden_params.response_cost = custom_cost
        
  
      
        return response_obj
    

# Create instance for LiteLLM to use
custom_mcp_cost_tracker = CustomMCPCostTracker()
```

#### 2. Configure in config.yaml

```yaml title="config.yaml" showLineNumbers
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: sk-xxxxxxx

# Add your custom MCP hook
callbacks:
  - custom_mcp_hook.custom_mcp_cost_tracker

mcp_servers:
  zapier_server:
    url: "https://actions.zapier.com/mcp/sk-xxxxx/sse"
```

#### 3. Start the proxy

```shell
$ litellm --config /path/to/config.yaml 
```

When MCP tools are called, your custom hook will:
1. Calculate costs based on your custom logic
2. Modify the response if needed
3. Track costs in LiteLLM's logging system

## ✨ MCP Permission Management

LiteLLM supports managing permissions for MCP Servers by Keys, Teams, Organizations (entities) on LiteLLM. When a MCP client attempts to list tools, LiteLLM will only return the tools the entity has permissions to access.

When Creating a Key, Team, or Organization, you can select the allowed MCP Servers that the entity has access to.

<Image 
  img={require('../img/mcp_key.png')}
  style={{width: '80%', display: 'block', margin: '0'}}
/>

## LiteLLM Proxy - Walk through MCP Gateway
LiteLLM exposes an MCP Gateway for admins to add all their MCP servers to LiteLLM. The key benefits of using LiteLLM Proxy with MCP are:

1. Use a fixed endpoint for all MCP tools
2. MCP Permission management by Key, Team, or User

This video demonstrates how you can onboard an MCP server to LiteLLM Proxy, use it and set access controls.

<iframe width="840" height="500" src="https://www.loom.com/embed/f7aa8d217879430987f3e64291757bfc" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

## LiteLLM Python SDK MCP Bridge

LiteLLM Python SDK acts as a MCP bridge to utilize MCP tools with all LiteLLM supported models. LiteLLM offers the following features for using MCP

- **List** Available MCP Tools: OpenAI clients can view all available MCP tools
  - `litellm.experimental_mcp_client.load_mcp_tools` to list all available MCP tools
- **Call** MCP Tools: OpenAI clients can call MCP tools
  - `litellm.experimental_mcp_client.call_openai_tool` to call an OpenAI tool on an MCP server


### 1. List Available MCP Tools

In this example we'll use `litellm.experimental_mcp_client.load_mcp_tools` to list all available MCP tools on any MCP server. This method can be used in two ways:

- `format="mcp"` - (default) Return MCP tools 
  - Returns: `mcp.types.Tool`
- `format="openai"` - Return MCP tools converted to OpenAI API compatible tools. Allows using with OpenAI endpoints.
  - Returns: `openai.types.chat.ChatCompletionToolParam`

<Tabs>
<TabItem value="sdk" label="LiteLLM Python SDK">

```python title="MCP Client List Tools" showLineNumbers
# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
import litellm
from litellm import experimental_mcp_client


server_params = StdioServerParameters(
    command="python3",
    # Make sure to update to the full absolute path to your mcp_server.py file
    args=["./mcp_server.py"],
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        # Initialize the connection
        await session.initialize()

        # Get tools
        tools = await experimental_mcp_client.load_mcp_tools(session=session, format="openai")
        print("MCP TOOLS: ", tools)

        messages = [{"role": "user", "content": "what's (3 + 5)"}]
        llm_response = await litellm.acompletion(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            messages=messages,
            tools=tools,
        )
        print("LLM RESPONSE: ", json.dumps(llm_response, indent=4, default=str))
```

</TabItem>

<TabItem value="openai" label="OpenAI SDK + LiteLLM Proxy">

In this example we'll walk through how you can use the OpenAI SDK pointed to the LiteLLM proxy to call MCP tools. The key difference here is we use the OpenAI SDK to make the LLM API request

```python title="MCP Client List Tools" showLineNumbers
# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
from openai import OpenAI
from litellm import experimental_mcp_client

server_params = StdioServerParameters(
    command="python3",
    # Make sure to update to the full absolute path to your mcp_server.py file
    args=["./mcp_server.py"],
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        # Initialize the connection
        await session.initialize()

        # Get tools using litellm mcp client
        tools = await experimental_mcp_client.load_mcp_tools(session=session, format="openai")
        print("MCP TOOLS: ", tools)

        # Use OpenAI SDK pointed to LiteLLM proxy
        client = OpenAI(
            api_key="your-api-key",  # Your LiteLLM proxy API key
            base_url="http://localhost:4000"  # Your LiteLLM proxy URL
        )

        messages = [{"role": "user", "content": "what's (3 + 5)"}]
        llm_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools
        )
        print("LLM RESPONSE: ", llm_response)
```
</TabItem>
</Tabs>


### 2. List and Call MCP Tools

In this example we'll use 
- `litellm.experimental_mcp_client.load_mcp_tools` to list all available MCP tools on any MCP server
- `litellm.experimental_mcp_client.call_openai_tool` to call an OpenAI tool on an MCP server

The first llm response returns a list of OpenAI tools. We take the first tool call from the LLM response and pass it to `litellm.experimental_mcp_client.call_openai_tool` to call the tool on the MCP server.

#### How `litellm.experimental_mcp_client.call_openai_tool` works

- Accepts an OpenAI Tool Call from the LLM response
- Converts the OpenAI Tool Call to an MCP Tool
- Calls the MCP Tool on the MCP server
- Returns the result of the MCP Tool call

<Tabs>
<TabItem value="sdk" label="LiteLLM Python SDK">

```python title="MCP Client List and Call Tools" showLineNumbers
# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
import litellm
from litellm import experimental_mcp_client


server_params = StdioServerParameters(
    command="python3",
    # Make sure to update to the full absolute path to your mcp_server.py file
    args=["./mcp_server.py"],
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        # Initialize the connection
        await session.initialize()

        # Get tools
        tools = await experimental_mcp_client.load_mcp_tools(session=session, format="openai")
        print("MCP TOOLS: ", tools)

        messages = [{"role": "user", "content": "what's (3 + 5)"}]
        llm_response = await litellm.acompletion(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            messages=messages,
            tools=tools,
        )
        print("LLM RESPONSE: ", json.dumps(llm_response, indent=4, default=str))

        openai_tool = llm_response["choices"][0]["message"]["tool_calls"][0]
        # Call the tool using MCP client
        call_result = await experimental_mcp_client.call_openai_tool(
            session=session,
            openai_tool=openai_tool,
        )
        print("MCP TOOL CALL RESULT: ", call_result)

        # send the tool result to the LLM
        messages.append(llm_response["choices"][0]["message"])
        messages.append(
            {
                "role": "tool",
                "content": str(call_result.content[0].text),
                "tool_call_id": openai_tool["id"],
            }
        )
        print("final messages with tool result: ", messages)
        llm_response = await litellm.acompletion(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            messages=messages,
            tools=tools,
        )
        print(
            "FINAL LLM RESPONSE: ", json.dumps(llm_response, indent=4, default=str)
        )
```

</TabItem>
<TabItem value="proxy" label="OpenAI SDK + LiteLLM Proxy">

In this example we'll walk through how you can use the OpenAI SDK pointed to the LiteLLM proxy to call MCP tools. The key difference here is we use the OpenAI SDK to make the LLM API request

```python title="MCP Client with OpenAI SDK" showLineNumbers
# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
from openai import OpenAI
from litellm import experimental_mcp_client

server_params = StdioServerParameters(
    command="python3",
    # Make sure to update to the full absolute path to your mcp_server.py file
    args=["./mcp_server.py"],
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        # Initialize the connection
        await session.initialize()

        # Get tools using litellm mcp client
        tools = await experimental_mcp_client.load_mcp_tools(session=session, format="openai")
        print("MCP TOOLS: ", tools)

        # Use OpenAI SDK pointed to LiteLLM proxy
        client = OpenAI(
            api_key="your-api-key",  # Your LiteLLM proxy API key
            base_url="http://localhost:8000"  # Your LiteLLM proxy URL
        )

        messages = [{"role": "user", "content": "what's (3 + 5)"}]
        llm_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools
        )
        print("LLM RESPONSE: ", llm_response)

        # Get the first tool call
        tool_call = llm_response.choices[0].message.tool_calls[0]
        
        # Call the tool using MCP client
        call_result = await experimental_mcp_client.call_openai_tool(
            session=session,
            openai_tool=tool_call.model_dump(),
        )
        print("MCP TOOL CALL RESULT: ", call_result)

        # Send the tool result back to the LLM
        messages.append(llm_response.choices[0].message.model_dump())
        messages.append({
            "role": "tool",
            "content": str(call_result.content[0].text),
            "tool_call_id": tool_call.id,
        })

        final_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools
        )
        print("FINAL RESPONSE: ", final_response)
```

</TabItem>
</Tabs>