# Built-in MCP Server Integration Examples

This document demonstrates how to use both Built-in and Remote MCP Server Integration features.

## 📋 Quick Reference

- 📖 **[Complete User Guide](builtin_mcp_user_guide.md)** - Full documentation with setup instructions
- ⚡ **[3-Minute Quick Setup](builtin_mcp_quick_setup.md)** - Get started immediately  
- 🔑 **[Client Token Usage](builtin_mcp_client_tokens.md)** - Personal token management
- ⚙️ **[Config Example](../litellm/proxy/example_config_yaml/builtin_mcp_config.yaml)** - Sample configuration

## Overview

LiteLLM supports two MCP integration approaches:

### 1. Built-in MCP Servers (Simplified)
Pre-configured servers managed by the proxy:
```json
{
  "tools": [{"type": "mcp", "builtin": "calculator"}]
}
```

### 2. Remote MCP Servers (OpenAI-compatible)
Direct server URL with full configuration:
```json
{
  "tools": [{
    "type": "mcp",
    "server_label": "stripe",
    "server_url": "https://mcp.stripe.com",
    "headers": {"Authorization": "Bearer sk-xxx"}
  }]
}
```

### 3. Hybrid Usage
Mix both approaches in a single request:
```json
{
  "tools": [
    {"type": "mcp", "builtin": "calculator"},
    {
      "type": "mcp", 
      "server_url": "https://mcp.stripe.com",
      "headers": {"Authorization": "Bearer sk-xxx"}
    }
  ]
}
```

## 🚀 Quick Start Examples

### Built-in MCP Server
```bash
curl -X POST http://localhost:4000/responses \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "input": [{"type": "message", "role": "user", "content": "Calculate 10 + 5"}],
    "tools": [{"type": "mcp", "builtin": "calculator"}]
  }'
```

### Remote MCP Server  
```bash
curl -X POST http://localhost:4000/responses \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "input": [{"type": "message", "role": "user", "content": "Create a payment link for $20"}],
    "tools": [{
      "type": "mcp",
      "server_label": "stripe",
      "server_url": "https://mcp.stripe.com",
      "headers": {"Authorization": "Bearer sk-live-xxx"},
      "require_approval": "never"
    }]
  }'
```

## 🛠️ Available Built-in Servers

### Calculator (Example/Demo)
- **Usage**: `{"type": "mcp", "builtin": "calculator"}`
- **Tools**: `add`, `subtract`, `multiply`, `divide`, `calculate`
- **Setup**: No setup required (included sample server)

### Production Services (Configurable)
- **Zapier**: `{"type": "mcp", "builtin": "zapier"}` (requires `ZAPIER_TOKEN`)
- **GitHub**: `{"type": "mcp", "builtin": "github"}` (requires `GITHUB_TOKEN`)
- **Jira**: `{"type": "mcp", "builtin": "jira"}` (requires `JIRA_TOKEN`)

## 🌐 Remote MCP Server Features

### Full OpenAI Compatibility
```json
{
  "type": "mcp",
  "server_label": "custom_service",
  "server_url": "https://mcp.service.com/api",
  "headers": {
    "Authorization": "Bearer token",
    "X-API-Key": "key123"
  },
  "allowed_tools": ["specific_tool"],
  "require_approval": "never"
}
```

### Tool Filtering
```json
{
  "type": "mcp",
  "server_url": "https://mcp.github.com",
  "allowed_tools": ["create_issue", "list_repos"],
  "headers": {"Authorization": "Bearer ghp-xxx"}
}
```

### Approval Control
Support for OpenAI-compatible approval workflow:

#### Simple Approval Settings
```json
{
  "type": "mcp",
  "server_url": "https://mcp.stripe.com",
  "require_approval": "never"  // Skip all approvals
}
```

#### Granular Approval Control
```json
{
  "type": "mcp",
  "server_url": "https://mcp.stripe.com",
  "require_approval": {
    "never": {
      "tool_names": ["get_balance", "list_transactions"]
    }
  }
}
```

#### Approval Workflow Example
```bash
# 1. Initial request (triggers approval)
curl -X POST http://localhost:4000/responses \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "input": [{"type": "message", "role": "user", "content": "Create a $100 payment"}],
    "tools": [{
      "type": "mcp",
      "server_url": "https://mcp.stripe.com",
      "headers": {"Authorization": "Bearer sk-xxx"}
    }]
  }'

# Response includes approval request:
# {
#   "output": [{
#     "id": "mcpr_12345...",
#     "type": "mcp_approval_request", 
#     "name": "create_payment",
#     "arguments": "{\"amount\": 100}",
#     "server_label": "stripe"
#   }]
# }

# 2. Approval response
curl -X POST http://localhost:4000/responses \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "previous_response_id": "resp_12345...",
    "tools": [{
      "type": "mcp",
      "server_url": "https://mcp.stripe.com", 
      "headers": {"Authorization": "Bearer sk-xxx"}
    }],
    "input": [{
      "type": "mcp_approval_response",
      "approve": true,
      "approval_request_id": "mcpr_12345..."
    }]
  }'

# Final response includes execution result:
# {
#   "output": [{
#     "id": "mcp_67890...",
#     "type": "mcp_call",
#     "name": "create_payment",
#     "arguments": "{\"amount\": 100}",
#     "output": "Payment created successfully",
#     "server_label": "stripe",
#     "approval_request_id": "mcpr_12345...",
#     "error": null
#   }]
# }
```

## 🎯 Complete Examples

### Hybrid Usage - Built-in + Remote
```python
import httpx
import asyncio

async def hybrid_example():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:4000/responses",
            headers={
                "Authorization": "Bearer sk-1234",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "input": [{"type": "message", "role": "user", "content": "Calculate 15% tip on $45 and create a Stripe payment link"}],
                "tools": [
                    # Built-in server
                    {"type": "mcp", "builtin": "calculator"},
                    # Remote server
                    {
                        "type": "mcp",
                        "server_label": "stripe",
                        "server_url": "https://mcp.stripe.com",
                        "headers": {"Authorization": "Bearer sk-live-xxx"},
                        "allowed_tools": ["create_payment_link"]
                    }
                ]
            }
        )
        return response.json()

result = asyncio.run(hybrid_example())
```

### Multi-Remote Server Usage
```json
{
  "model": "gpt-4o-mini",
  "input": [{"type": "message", "role": "user", "content": "Check GitHub issues and create Stripe invoice"}],
  "tools": [
    {
      "type": "mcp",
      "server_label": "github",
      "server_url": "https://mcp.github.com",
      "headers": {"Authorization": "Bearer ghp-xxx"}
    },
    {
      "type": "mcp", 
      "server_label": "stripe",
      "server_url": "https://mcp.stripe.com",
      "headers": {"Authorization": "Bearer sk-xxx"}
    }
  ]
}
```

## 🔧 Configuration Comparison

| Feature | Built-in MCP | Remote MCP |
|---------|-------------|------------|
| **Setup** | Config file once | Per-request |
| **Security** | Server-managed tokens | Client-provided headers |
| **Discovery** | Simple names | Full URLs |
| **Flexibility** | Pre-configured | Fully customizable |
| **Management** | Centralized | Distributed |

### Built-in Server Config
```yaml
mcp_builtin_servers:
  zapier:
    url: "https://mcp.zapier.com/api/mcp"
    transport: "sse"
    auth_type: "bearer_token"
    env_key: "ZAPIER_TOKEN"
    enabled: true
```

### Remote Server Usage
```json
{
  "type": "mcp",
  "server_url": "https://mcp.zapier.com/api/mcp", 
  "headers": {"Authorization": "Bearer user-token"}
}
```

## 📚 Documentation Files

- **[builtin_mcp_user_guide.md](builtin_mcp_user_guide.md)** - Complete user guide with detailed setup instructions
- **[builtin_mcp_quick_setup.md](builtin_mcp_quick_setup.md)** - 3-minute setup guide
- **[builtin_mcp_client_tokens.md](builtin_mcp_client_tokens.md)** - Client token usage patterns
- **[sample_calculator_mcp_server.py](sample_calculator_mcp_server.py)** - Complete MCP server implementation

## 🎉 Benefits

### Built-in MCP
- ✅ **Enhanced Security**: No credential exposure to clients
- ✅ **Simplified Usage**: Just specify `"builtin": "service_name"`
- ✅ **Centralized Management**: Server-side configuration
- ✅ **Client Personalization**: Optional client token override

### Remote MCP  
- ✅ **Full Compatibility**: OpenAI Remote MCP compatible
- ✅ **Maximum Flexibility**: Custom headers and configuration
- ✅ **Tool Filtering**: Granular tool access control
- ✅ **Dynamic Usage**: No server-side setup required

### Hybrid Approach
- ✅ **Best of Both Worlds**: Use appropriate method per service
- ✅ **Gradual Migration**: Move from remote to built-in as needed
- ✅ **Backward Compatible**: Works with existing MCP usage

---

Choose the approach that best fits your use case, or mix both for maximum flexibility!