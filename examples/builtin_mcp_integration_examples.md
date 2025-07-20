# Built-in MCP Server Integration Examples

This document demonstrates how to use the new Built-in MCP Server Integration feature with practical examples.

## Overview

Built-in MCP Server Integration allows you to use MCP tools with simple references instead of complex server configurations. This provides:

- **Enhanced Security**: No credential exposure to clients
- **Simplified Usage**: Just specify `"builtin": "service_name"`
- **Centralized Management**: Server-side token and configuration management

## Quick Start Example

### Before: Complex Configuration
```json
{
  "model": "gpt-4o",
  "input": "Add 15 and 27",
  "tools": [{
    "type": "mcp",
    "server_url": "https://external-mcp-server.com/mcp",
    "headers": {"Authorization": "Bearer your-secret-token"},
    "require_approval": "never"
  }]
}
```

### After: Simple Builtin Reference
```json
{
  "model": "gpt-4o", 
  "input": "Add 15 and 27",
  "tools": [{
    "type": "mcp",
    "builtin": "calculator",
    "require_approval": "never"
  }]
}
```

## Available Built-in Servers

### 1. Calculator (Example/Demo)
A sample MCP server demonstrating basic mathematical operations.

**Usage:**
```json
{
  "tools": [{"type": "mcp", "builtin": "calculator"}]
}
```

**Available Tools:**
- `add` - Add two numbers
- `subtract` - Subtract two numbers  
- `multiply` - Multiply two numbers
- `divide` - Divide two numbers
- `calculate` - Evaluate mathematical expressions

**Example:**
```python
import litellm

response = litellm.aresponses(
    model="gpt-4o",
    input="Calculate 25 * 4 + 10",
    tools=[{
        "type": "mcp",
        "builtin": "calculator",
        "require_approval": "never"
    }]
)
```

### 2. Zapier (Production)
Zapier automation platform integration.

**Setup Required:**
```bash
export ZAPIER_TOKEN="your_zapier_api_token"
```

**Usage:**
```json
{
  "tools": [{"type": "mcp", "builtin": "zapier"}]
}
```

### 3. Jira (Production) 
Atlassian Jira project management integration.

**Setup Required:**
```bash
export JIRA_TOKEN="your_jira_api_token"
```

**Usage:**
```json
{
  "tools": [{"type": "mcp", "builtin": "jira"}]
}
```

### 4. GitHub (Production)
GitHub repository and issue management.

**Setup Required:**
```bash
export GITHUB_TOKEN="your_github_personal_access_token"
```

**Usage:**
```json
{
  "tools": [{"type": "mcp", "builtin": "github"}]
}
```

### 5. Slack (Production)
Slack team communication platform.

**Setup Required:**
```bash
export SLACK_BOT_TOKEN="your_slack_bot_token"
```

**Usage:**
```json
{
  "tools": [{"type": "mcp", "builtin": "slack"}]
}
```

## Complete Example: Using the Calculator

### 1. Test the Sample Server
```bash
# Test the calculator MCP server directly
python examples/sample_calculator_mcp_server.py --test
```

### 2. Use with LiteLLM Responses API
```python
import litellm
import asyncio

async def calculator_example():
    response = await litellm.aresponses(
        model="gpt-4o",
        input="I need to calculate the total cost: 3 items at $15.99 each, plus 8.5% tax",
        tools=[{
            "type": "mcp",
            "builtin": "calculator",
            "require_approval": "never"
        }]
    )
    
    print("Response:", response.output[0].content[0].text)

# Run the example
asyncio.run(calculator_example())
```

### 3. Multiple Tool Usage
```python
# Use both calculator and external tools
response = await litellm.aresponses(
    model="gpt-4o",
    input="Calculate 15% tip on $45.30 and remind me to pay the bill",
    tools=[
        {"type": "mcp", "builtin": "calculator"},
        {"type": "function", "function": {
            "name": "set_reminder",
            "description": "Set a reminder",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string"}
                }
            }
        }}
    ]
)
```

## Adding Custom Built-in Servers

### 1. Create Your MCP Server
Follow the MCP protocol specification to create your server. See `examples/sample_calculator_mcp_server.py` for a complete example.

### 2. Add to Built-in Registry
```python
# In builtin_registry.py _initialize_default_servers method
self.register_builtin(
    BuiltinStdioMCPServerConfig(
        name="my_custom_server",
        command="python",
        args=["/path/to/my_mcp_server.py"],
        description="My custom MCP server"
    )
)
```

### 3. Use Your Custom Server
```json
{
  "tools": [{"type": "mcp", "builtin": "my_custom_server"}]
}
```

## Configuration Management

### Environment Variables
Built-in servers that require authentication use environment variables:

```bash
# Set up all available tokens
export ZAPIER_TOKEN="your_zapier_token"
export JIRA_TOKEN="your_jira_token"
export GITHUB_TOKEN="your_github_token"
export SLACK_BOT_TOKEN="your_slack_token"
```

### Mixed Usage
You can mix built-in and external MCP servers in the same request:

```json
{
  "tools": [
    {"type": "mcp", "builtin": "calculator"},
    {"type": "mcp", "server_url": "https://my-custom-server.com/mcp"},
    {"type": "function", "function": {"name": "regular_function"}}
  ]
}
```

## Benefits Recap

### Security
- ✅ No credential exposure to clients
- ✅ Centralized token management
- ✅ Server-side authentication handling

### Simplicity  
- ✅ Simple `"builtin": "name"` reference
- ✅ No complex server configuration
- ✅ Consistent interface across services

### Management
- ✅ Easy to add new built-in servers
- ✅ Environment-based configuration
- ✅ Backward compatible with existing MCP usage

## Troubleshooting

### Server Not Available
If a built-in server shows as unavailable:

1. **Check Environment Variables**: Ensure required tokens are set
2. **Verify Server Path**: For stdio servers, check the script path exists
3. **Check Logs**: Look for initialization warnings in server logs

### Tool Execution Fails
1. **Check Tool Names**: Verify the tool names match what the server provides
2. **Validate Arguments**: Ensure tool arguments match the expected schema
3. **Server Logs**: Check the MCP server logs for detailed error information

---

This integration makes MCP tools more accessible while maintaining security and flexibility!