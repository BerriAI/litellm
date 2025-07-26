# Built-in MCP Server Integration

LiteLLM's Built-in MCP Server Integration provides a simplified, secure way to use MCP (Model Context Protocol) servers with minimal configuration.

## 🚀 Quick Start

### Step 1: Create Configuration File

Create a `config.yaml` file:

```yaml
model_list:
  - model_name: gpt-4o-mini
    litellm_params:
      model: gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY

# Built-in MCP Servers Configuration
mcp_builtin_servers:
  calculator:
    transport: "stdio"
    command: "python"
    args: ["examples/sample_calculator_mcp_server.py"]
    description: "Basic calculator"
    enabled: true
    
  zapier:
    url: "https://mcp.zapier.com/api/mcp"
    transport: "sse"
    auth_type: "bearer_token"
    env_key: "ZAPIER_TOKEN"
    description: "Zapier automation platform"
    enabled: true

general_settings:
  master_key: sk-1234
```

### Step 2: Set Environment Variables

```bash
export ZAPIER_TOKEN="your_zapier_token"
```

### Step 3: Start LiteLLM Proxy

```bash
litellm --config config.yaml
```

### Step 4: Use Built-in Servers

**Simple Usage:**
```json
{
  "model": "gpt-4o-mini",
  "messages": [{"role": "user", "content": "Calculate 15 * 23"}],
  "tools": [{"type": "mcp", "builtin": "calculator"}]
}
```

**With Client Token:**
```json
{
  "model": "gpt-4o-mini", 
  "messages": [{"role": "user", "content": "Create a Zap"}],
  "tools": [{
    "type": "mcp",
    "builtin": "zapier",
    "auth_token": "client_personal_token"
  }]
}
```

## 📋 Integration Types

### 1. Built-in Servers (Simplified)
Pre-configured servers managed by the proxy:
```json
{"tools": [{"type": "mcp", "builtin": "calculator"}]}
```

### 2. Remote Servers (Direct)
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
Mix both approaches:
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

## ⚙️ Configuration Reference

### Server Types

**HTTP/SSE Transport:**
```yaml
mcp_builtin_servers:
  service_name:
    url: "https://mcp.example.com/api"
    transport: "sse"  # or "http"
    auth_type: "bearer_token"
    env_key: "SERVICE_TOKEN"
    description: "Service description"
    enabled: true
```

**Stdio Transport:**
```yaml
mcp_builtin_servers:
  local_tool:
    transport: "stdio"
    command: "python"
    args: ["path/to/server.py"]
    env:
      DEBUG: "true"
    description: "Local MCP server"
    enabled: true
```

### Common Built-in Servers

| Server | Env Variable | Description |
|--------|-------------|-------------|
| `calculator` | N/A | Basic math operations |
| `zapier` | `ZAPIER_TOKEN` | Zapier automation |
| `jira` | `JIRA_TOKEN` | Atlassian Jira |
| `github` | `GITHUB_TOKEN` | GitHub integration |

## 🔐 Authentication

### Server-Level (Default)
```bash
export ZAPIER_TOKEN="server_token"
```

### Client-Level (Override)
```json
{
  "tools": [{
    "type": "mcp",
    "builtin": "zapier",
    "auth_token": "personal_token"
  }]
}
```

## 🛡️ Approval Workflow

### Simple Approval
```json
{
  "tools": [{
    "type": "mcp",
    "builtin": "zapier",
    "require_approval": "never"  // "always" (default) or "never"
  }]
}
```

### Granular Approval
```json
{
  "tools": [{
    "type": "mcp",
    "builtin": "zapier",
    "require_approval": {
      "never": {
        "tool_names": ["get_zaps", "list_zaps"]
      }
    }
  }]
}
```

## 📚 Examples

### Calculator Usage
```bash
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "What is 25 * 17?"}],
    "tools": [{"type": "mcp", "builtin": "calculator"}]
  }'
```

### Zapier Integration
```bash
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Create a new Zap"}],
    "tools": [{
      "type": "mcp",
      "builtin": "zapier",
      "auth_token": "your_personal_token"
    }]
  }'
```

## 🔧 Advanced Configuration

### Custom Server
```yaml
mcp_builtin_servers:
  my_service:
    url: "https://my-mcp-server.com/api"
    transport: "sse"
    auth_type: "bearer_token"
    env_key: "MY_SERVICE_TOKEN"
    description: "Custom service integration"
    enabled: true
```

### Environment Variables
```bash
# Server tokens
export ZAPIER_TOKEN="zap_token"
export JIRA_TOKEN="jira_token"
export GITHUB_TOKEN="github_token"

# Custom service
export MY_SERVICE_TOKEN="custom_token"
```

## ✅ Benefits

- **🔒 Enhanced Security**: No credential exposure to clients
- **🎯 Simplified Usage**: Just reference builtin name
- **⚡ Quick Setup**: Minimal configuration required
- **🔄 Backward Compatible**: Works with existing MCP usage
- **🛡️ Centralized Auth**: Manage tokens at proxy level
- **📊 Approval Control**: Fine-grained execution control

## 🐛 Troubleshooting

### Common Issues

**Server not available:**
```bash
# Check if environment variable is set
echo $ZAPIER_TOKEN

# Verify server configuration
curl http://localhost:4000/v1/models
```

**Authentication failed:**
```bash
# Test with server token
export ZAPIER_TOKEN="valid_token"

# Or use client token in request
```

**Tool execution failed:**
- Verify approval settings
- Check server connectivity
- Review proxy logs for errors

---

For more examples, see the [config file](../litellm/proxy/example_config_yaml/builtin_mcp_config.yaml).