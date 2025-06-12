# MCP Server Task Group Initialization Fix

## Issue Description

You encountered the following error when using LiteLLM's MCP server:

```
17:53:27 - LiteLLM:ERROR: server.py:271 - Error handling MCP request: Task group is not initialized. Make sure to use run().
Traceback (most recent call last):
  File "/Users/ishaanjaffer/Github/litellm/litellm/proxy/_experimental/mcp_server/server.py", line 269, in handle_streamable_http_mcp
    await session_manager.handle_request(scope, receive, send)
  File "/Users/ishaanjaffer/Github/litellm/litellm/proxy/venv/lib/python3.10/site-packages/mcp/server/streamable_http_manager.py", line 137, in handle_request
    raise RuntimeError("Task group is not initialized. Make sure to use run().")
RuntimeError: Task group is not initialized. Make sure to use run().
```

## Root Cause

This error occurs because the MCP Python SDK's `StreamableHTTPSessionManager` requires proper task group initialization via the `run()` method before it can handle HTTP requests. However, the current LiteLLM implementation doesn't properly initialize the MCP server's task groups in a way that's compatible with FastAPI's request handling model.

This is a known issue in the MCP Python SDK (see [GitHub Issue #838](https://github.com/modelcontextprotocol/python-sdk/issues/838) and [GitHub Issue #713](https://github.com/modelcontextprotocol/python-sdk/issues/713)).

## Solution Applied

I've implemented a fix that addresses this issue by:

1. **Disabling the problematic streamable HTTP endpoint**: The `POST /mcp/` endpoint now returns a 501 status code with a clear explanation of the limitation.

2. **Providing alternative REST API endpoints**: Users can still access MCP functionality through:
   - `GET /mcp/tools/list` - List available tools
   - `POST /mcp/tools/call` - Call a specific tool

3. **Maintaining SSE support**: The Server-Sent Events (SSE) transport continues to work properly at `GET /mcp/` and `POST /mcp/sse/messages`.

## How to Use MCP with LiteLLM

### Option 1: Use REST API Endpoints (Recommended)

#### List Available Tools
```bash
curl -X GET "http://localhost:4000/mcp/tools/list" \
  -H "Authorization: Bearer your_api_key_here"
```

#### Call a Tool
```bash
curl -X POST "http://localhost:4000/mcp/tools/call" \
  -H "Authorization: Bearer your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "tool_name",
    "arguments": {
      "param1": "value1",
      "param2": "value2"
    }
  }'
```

### Option 2: Use SSE Transport

For Server-Sent Events transport (which works properly):

```python
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

async def main():
    async with sse_client("http://localhost:4000/mcp/") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List tools
            tools = await session.list_tools()
            print(f"Available tools: {tools.tools}")
            
            # Call a tool
            result = await session.call_tool("tool_name", {"param": "value"})
            print(f"Result: {result}")

asyncio.run(main())
```

## Alternative Workaround for Streamable HTTP

If you specifically need streamable HTTP transport, you can work around this by using a separate MCP server instance. Here's how:

### 1. Create a standalone MCP server

```python
# standalone_mcp_server.py
from mcp.server.fastmcp import FastMCP
import uvicorn

# Create FastMCP server
mcp = FastMCP(name="MyMCPServer", stateless_http=True)

@mcp.tool(description="Example tool")
def example_tool(message: str) -> str:
    return f"Processed: {message}"

if __name__ == "__main__":
    # Run with streamable HTTP transport
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8001)
```

### 2. Configure LiteLLM to use the external MCP server

Add this to your LiteLLM configuration:

```yaml
mcp_servers:
  my_mcp_server:
    url: "http://localhost:8001/mcp"
    transport: "http"
    description: "My custom MCP server"
```

### 3. Start both servers

```bash
# Terminal 1: Start the standalone MCP server
python standalone_mcp_server.py

# Terminal 2: Start LiteLLM proxy
litellm --config /path/to/config.yaml
```

## Configuration Notes

- The SSE transport (`GET /mcp/` and `POST /mcp/sse/messages`) continues to work normally
- The REST API endpoints (`/mcp/tools/list` and `/mcp/tools/call`) provide full MCP functionality
- MCP servers configured in your LiteLLM config will still work through the client manager

## Testing the Fix

You can test that the fix works by:

1. Starting your LiteLLM proxy
2. Making a request to `POST /mcp/` - you should now get a 501 response instead of the task group error
3. Using the REST API endpoints for MCP functionality
4. Using SSE transport for real-time MCP communication

## Long-term Solution

This issue should be resolved in future versions of the MCP Python SDK. Once the SDK properly supports task group initialization in web frameworks like FastAPI, the streamable HTTP transport can be re-enabled.

For now, the REST API endpoints and SSE transport provide full MCP functionality without the task group initialization issues.