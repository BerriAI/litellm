# LiteLLM Responses API with MCP Integration

The `responses_api_with_mcp` function provides seamless integration between LiteLLM's Responses API and Model Context Protocol (MCP) servers. This allows you to leverage MCP tools directly in your responses API calls with automatic tool execution.

## Overview

When you pass MCP tools with `server_url="litellm_proxy"`, the function will:
1. Automatically fetch available tools from the LiteLLM MCP server manager
2. Transform MCP tools to OpenAI-compatible format
3. Include them in the responses API call
4. If `require_approval="never"`, automatically execute any tool calls returned by the model
5. Return the final response after tool execution

## Installation & Setup

First, ensure you have MCP servers configured in your LiteLLM proxy:

```yaml
# config.yaml
mcp_servers:
  weather_server:
    url: "https://api.weather.com/mcp"
    transport: "sse"
    auth_type: "api_key"
    spec_version: "2025-03-26"
  
  database_server:
    command: "python"
    args: ["./mcp_database_server.py"]
    transport: "stdio"
```

## Basic Usage

### Async Version

```python
import litellm

# Basic usage with MCP integration
response = await litellm.aresponses_api_with_mcp(
    model="gpt-4o",
    input="What's the weather like in San Francisco?",
    tools=[{
        "type": "mcp",
        "server_label": "weather",
        "server_url": "litellm_proxy",  # Special URL to use LiteLLM MCP manager
        "require_approval": "never"     # Automatically execute tool calls
    }]
)

print(response.output[0].content)
```

### Sync Version

```python
import litellm

# Synchronous version
response = litellm.responses_api_with_mcp(
    model="gpt-4o", 
    input="Query the database for user stats",
    tools=[{
        "type": "mcp",
        "server_label": "database",
        "server_url": "litellm_proxy",
        "require_approval": "never"
    }]
)

print(response.output[0].content)
```

## Tool Configuration Options

```python
tools = [{
    "type": "mcp",
    "server_label": "my_server",       # Label for the MCP server
    "server_url": "litellm_proxy",     # Must be "litellm_proxy" for integration
    "require_approval": "never"        # "never" for auto-execution, "always" for manual
}]
```

### require_approval Options

- **"never"**: Tool calls are automatically executed and the final response is returned
- **"always"**: Tool calls are returned for manual approval (not auto-executed)

## Advanced Examples

### Multiple MCP Servers

```python
response = await litellm.aresponses_api_with_mcp(
    model="gpt-4o",
    input="Check the weather and update my calendar",
    tools=[
        {
            "type": "mcp",
            "server_label": "weather",
            "server_url": "litellm_proxy", 
            "require_approval": "never"
        },
        {
            "type": "mcp",
            "server_label": "calendar",
            "server_url": "litellm_proxy",
            "require_approval": "never"
        }
    ]
)
```

### Mixed Tools (MCP + Regular Function Tools)

```python
response = await litellm.aresponses_api_with_mcp(
    model="gpt-4o",
    input="Analyze this data and get weather info",
    tools=[
        # MCP tool
        {
            "type": "mcp",
            "server_label": "weather",
            "server_url": "litellm_proxy",
            "require_approval": "never"
        },
        # Regular function tool
        {
            "type": "function",
            "function": {
                "name": "analyze_data",
                "description": "Analyze provided data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string"}
                    }
                }
            }
        }
    ]
)
```

### With User Authentication

```python
from litellm.proxy._types import UserAPIKeyAuth

user_auth = UserAPIKeyAuth(api_key="your-api-key")

response = await litellm.aresponses_api_with_mcp(
    model="gpt-4o",
    input="Perform authenticated action",
    tools=[{
        "type": "mcp",
        "server_label": "secure_server", 
        "server_url": "litellm_proxy",
        "require_approval": "never"
    }],
    user_api_key_auth=user_auth
)
```

## Error Handling

The function includes comprehensive error handling:

```python
try:
    response = await litellm.aresponses_api_with_mcp(
        model="gpt-4o",
        input="Test message",
        tools=[{
            "type": "mcp",
            "server_label": "test_server",
            "server_url": "litellm_proxy", 
            "require_approval": "never"
        }]
    )
except Exception as e:
    print(f"Error: {e}")
    # Handle errors gracefully
```

## How It Works

1. **Tool Discovery**: When `server_url="litellm_proxy"` is detected, the function queries the LiteLLM MCP server manager for available tools
2. **Tool Transformation**: MCP tools are transformed to OpenAI-compatible format using `transform_mcp_tool_to_openai_tool`
3. **Initial Response**: The standard responses API is called with the transformed tools
4. **Auto-Execution**: If `require_approval="never"` and tool calls are present, they're automatically executed
5. **Follow-up**: A follow-up responses API call is made with the tool results to get the final response

## Response Structure

The response follows the standard LiteLLM ResponsesAPIResponse format:

```python
{
    "id": "resp_...",
    "object": "response",
    "status": "completed",
    "output": [
        {
            "type": "message",
            "role": "assistant", 
            "content": "Final response after tool execution"
        }
    ],
    "usage": {
        "completion_tokens": 50,
        "prompt_tokens": 25,
        "total_tokens": 75
    },
    "model": "gpt-4o",
    # ... other fields
}
```

## Comparison with Regular responses()

| Feature | `responses()` | `responses_api_with_mcp()` |
|---------|---------------|----------------------------|
| MCP Integration | ❌ | ✅ |
| Auto Tool Execution | ❌ | ✅ (with `require_approval="never"`) |
| Tool Discovery | ❌ | ✅ (from MCP server manager) |
| Regular Function Tools | ✅ | ✅ |
| Performance | Faster | Slightly slower (due to MCP integration) |

## Best Practices

1. **Use require_approval="never" carefully**: Only for trusted MCP servers
2. **Error handling**: Always wrap calls in try-catch blocks
3. **Authentication**: Pass user authentication when using secured MCP servers
4. **Tool naming**: Use clear, descriptive server labels
5. **Performance**: Consider caching MCP tool discoveries for high-volume applications

## Limitations

- Only works with MCP servers configured in the LiteLLM proxy
- `server_url` must be exactly "litellm_proxy"
- Auto-execution only works with `require_approval="never"`
- Tool transformations may not support all MCP tool features

## See Also

- [MCP Documentation](./mcp.md)
- [Responses API Documentation](./response_api.md)
- [LiteLLM Proxy Configuration](./proxy/configs.md)