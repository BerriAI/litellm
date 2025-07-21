# Built-in MCP Server Integration Examples

This document demonstrates how to use the new Built-in MCP Server Integration feature with practical examples.

## 📋 Quick Reference

- 📖 **[Complete User Guide](builtin_mcp_user_guide.md)** - Full documentation with setup instructions
- ⚡ **[3-Minute Quick Setup](builtin_mcp_quick_setup.md)** - Get started immediately  
- 🔑 **[Client Token Usage](builtin_mcp_client_tokens.md)** - Personal token management
- ⚙️ **[Config Example](../litellm/proxy/example_config_yaml/builtin_mcp_config.yaml)** - Sample configuration

## Overview

Built-in MCP Server Integration allows you to use MCP tools with simple references instead of complex server configurations:

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

## 🚀 Quick Start

### 1. Test the Calculator Server
```bash
# Test the calculator MCP server directly
python examples/sample_calculator_mcp_server.py test
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

## 🛠️ Available Built-in Servers

### Calculator (Example/Demo)
- **Usage**: `{"type": "mcp", "builtin": "calculator"}`
- **Tools**: `add`, `subtract`, `multiply`, `divide`, `calculate`
- **Setup**: No setup required (included sample server)

### Production Services
- **Zapier**: `{"type": "mcp", "builtin": "zapier"}` (requires `ZAPIER_TOKEN`)
- **GitHub**: `{"type": "mcp", "builtin": "github"}` (requires `GITHUB_TOKEN`)
- **Jira**: `{"type": "mcp", "builtin": "jira"}` (requires `JIRA_TOKEN`)

## 🔧 Configuration

### Config-based Setup (Recommended)
```yaml
mcp_builtin_servers:
  calculator:
    transport: "stdio"
    command: "python"
    args: ["examples/sample_calculator_mcp_server.py"]
    enabled: true
    
  zapier:
    url: "https://mcp.zapier.com/api/mcp"
    transport: "sse"
    auth_type: "bearer_token"
    env_key: "ZAPIER_TOKEN"
    enabled: true
```

### Client Token Override
```json
{
  "tools": [{
    "type": "mcp",
    "builtin": "zapier",
    "auth_token": "user_personal_token"
  }]
}
```

## 🎯 Complete Examples

### Multiple Tool Usage
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

### Mixed Built-in and External Servers
```json
{
  "tools": [
    {"type": "mcp", "builtin": "calculator"},
    {"type": "mcp", "server_url": "https://my-custom-server.com/mcp"},
    {"type": "function", "function": {"name": "regular_function"}}
  ]
}
```

## 📚 Documentation Files

- **[builtin_mcp_user_guide.md](builtin_mcp_user_guide.md)** - Complete user guide with detailed setup instructions
- **[builtin_mcp_quick_setup.md](builtin_mcp_quick_setup.md)** - 3-minute setup guide
- **[builtin_mcp_client_tokens.md](builtin_mcp_client_tokens.md)** - Client token usage patterns
- **[sample_calculator_mcp_server.py](sample_calculator_mcp_server.py)** - Complete MCP server implementation

## 🎉 Benefits

- ✅ **Enhanced Security**: No credential exposure to clients
- ✅ **Simplified Usage**: Just specify `"builtin": "service_name"`
- ✅ **Centralized Management**: Server-side configuration
- ✅ **Client Personalization**: Optional client token override
- ✅ **Backward Compatible**: Works with existing MCP usage

---

For detailed setup instructions and advanced configuration, see the [Complete User Guide](builtin_mcp_user_guide.md)!