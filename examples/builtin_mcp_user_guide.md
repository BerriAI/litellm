# Built-in MCP Server Integration User Guide

LiteLLM's Built-in MCP Server Integration allows you to use external services with simple names instead of complex MCP server configurations.

## 🚀 Quick Start

### Step 1: Configuration File Setup

Create a `config.yaml` file and define the MCP servers you want to use:

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
    
  github:
    url: "https://mcp.github.com/mcp"
    transport: "http"
    auth_type: "bearer_token"
    env_key: "GITHUB_TOKEN"
    description: "GitHub repository management"
    enabled: true

general_settings:
  master_key: sk-1234
```

### Step 2: Environment Variables

Set up environment variables for external services:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ZAPIER_TOKEN="your-zapier-token"
export GITHUB_TOKEN="your-github-token"
```

### Step 3: Start the Server

```bash
litellm --config config.yaml
```

### Step 4: Use Built-in MCP Tools

```bash
curl -X POST http://localhost:4000/responses \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "input": [{"type": "message", "role": "user", "content": "Calculate 25 * 4 + 15"}],
    "tools": [{"type": "mcp", "builtin": "calculator"}]
  }'
```

## 🔧 Configuration Details

### Server Configuration Types

#### 1. Stdio Transport (Local Scripts)
```yaml
calculator:
  transport: "stdio"
  command: "python"
  args: ["examples/sample_calculator_mcp_server.py"]
  env:
    CUSTOM_VAR: "value"
  enabled: true
```

#### 2. HTTP/SSE Transport (External Services)
```yaml
zapier:
  url: "https://mcp.zapier.com/api/mcp"
  transport: "sse"  # or "http"
  auth_type: "bearer_token"
  env_key: "ZAPIER_TOKEN"
  enabled: true
```

### Configuration Parameters

| Parameter | Description | Required | Example |
|-----------|-------------|----------|---------|
| `transport` | Communication method | Yes | `"stdio"`, `"http"`, `"sse"` |
| `command` | Command to run (stdio only) | For stdio | `"python"`, `"node"` |
| `args` | Command arguments (stdio only) | For stdio | `["script.py", "--option"]` |
| `url` | Server URL (http/sse only) | For http/sse | `"https://api.service.com/mcp"` |
| `auth_type` | Authentication method | No | `"bearer_token"` |
| `env_key` | Environment variable name | For auth | `"SERVICE_TOKEN"` |
| `enabled` | Enable/disable server | No | `true` (default) |
| `description` | Human-readable description | No | `"Service description"` |

## 🎯 Usage Examples

### Basic Calculator Usage

```python
import httpx
import asyncio

async def calculator_example():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:4000/responses",
            headers={
                "Authorization": "Bearer sk-1234",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "input": [{"type": "message", "role": "user", "content": "What's 150 * 0.15?"}],
                "tools": [{"type": "mcp", "builtin": "calculator"}]
            }
        )
        return response.json()

result = asyncio.run(calculator_example())
print(result)
```

### Multiple Built-in Services

```json
{
  "model": "gpt-4o-mini",
  "input": [{"type": "message", "role": "user", "content": "Calculate tax and create a Zapier task"}],
  "tools": [
    {"type": "mcp", "builtin": "calculator"},
    {"type": "mcp", "builtin": "zapier"}
  ]
}
```

### Client Token Override

Users can provide their own authentication tokens:

```json
{
  "model": "gpt-4o-mini",
  "input": [{"type": "message", "role": "user", "content": "List my repositories"}],
  "tools": [{
    "type": "mcp",
    "builtin": "github",
    "auth_token": "user_personal_github_token"
  }]
}
```

## 🔐 Authentication Methods

### Server-side Environment Variables (Default)
```bash
export ZAPIER_TOKEN="server_default_token"
export GITHUB_TOKEN="server_default_token"
```

### Client-provided Tokens (Override)
```json
{
  "tools": [{
    "type": "mcp",
    "builtin": "zapier",
    "auth_token": "user_personal_token"
  }]
}
```

### Token Priority
1. **Client Token** (`auth_token` field) - Highest priority
2. **Server Environment Variable** (from `env_key`) - Fallback
3. **No Token** - Service unavailable

## 🛠️ Available Built-in Servers

### Calculator (Sample/Demo)
- **Type**: Stdio transport
- **Setup**: No external dependencies
- **Tools**: `add`, `subtract`, `multiply`, `divide`, `calculate`
- **Usage**: `{"type": "mcp", "builtin": "calculator"}`

### Zapier (Production)
- **Type**: SSE transport
- **Setup**: Requires `ZAPIER_TOKEN` environment variable
- **Purpose**: Workflow automation
- **Usage**: `{"type": "mcp", "builtin": "zapier"}`

### GitHub (Production)
- **Type**: HTTP transport
- **Setup**: Requires `GITHUB_TOKEN` environment variable
- **Purpose**: Repository and issue management
- **Usage**: `{"type": "mcp", "builtin": "github"}`

### Jira (Production)
- **Type**: SSE transport
- **Setup**: Requires `JIRA_TOKEN` environment variable
- **Purpose**: Project management
- **Usage**: `{"type": "mcp", "builtin": "jira"}`

## 📊 Advanced Configuration

### Conditional Server Enabling

```yaml
mcp_builtin_servers:
  development_tools:
    transport: "stdio"
    command: "python"
    args: ["dev-tools.py"]
    enabled: ${ENVIRONMENT:development}  # Only in development
    
  production_service:
    url: "https://prod-service.com/mcp"
    transport: "sse"
    auth_type: "bearer_token"
    env_key: "PROD_TOKEN"
    enabled: ${ENVIRONMENT:production}  # Only in production
```

### Custom Environment Variables

```yaml
mcp_builtin_servers:
  custom_service:
    transport: "stdio"
    command: "python"
    args: ["custom-server.py"]
    env:
      SERVICE_MODE: "production"
      LOG_LEVEL: "info"
      API_ENDPOINT: "https://api.example.com"
    enabled: true
```

### Mixed Transport Types

```yaml
mcp_builtin_servers:
  local_calculator:
    transport: "stdio"
    command: "python"
    args: ["examples/sample_calculator_mcp_server.py"]
    
  remote_weather:
    url: "https://weather-mcp.service.com/mcp"
    transport: "sse"
    auth_type: "bearer_token"
    env_key: "WEATHER_API_KEY"
    
  github_integration:
    url: "https://github-mcp.service.com/api"
    transport: "http"
    auth_type: "bearer_token"
    env_key: "GITHUB_TOKEN"
```

## 🚨 Troubleshooting

### Server Not Available

**Symptom**: Built-in server shows as unavailable

**Solutions**:
1. **Check Environment Variables**:
   ```bash
   echo $ZAPIER_TOKEN  # Should not be empty
   ```

2. **Verify Server Configuration**:
   ```yaml
   mcp_builtin_servers:
     zapier:
       enabled: true  # Must be true
       env_key: "ZAPIER_TOKEN"  # Correct env var name
   ```

3. **Check Server Logs**:
   ```bash
   litellm --config config.yaml --debug
   ```

### Tool Execution Fails

**Symptom**: MCP tool calls return errors

**Solutions**:
1. **Verify Tool Names**: Check what tools the server actually provides
2. **Validate Arguments**: Ensure parameters match the tool schema
3. **Check Authentication**: Verify tokens have correct permissions
4. **Review Server Logs**: Look for detailed error messages

### Stdio Server Issues

**Symptom**: Local stdio servers don't start

**Solutions**:
1. **Check File Paths**:
   ```bash
   ls -la examples/sample_calculator_mcp_server.py
   ```

2. **Test Script Directly**:
   ```bash
   python examples/sample_calculator_mcp_server.py test
   ```

3. **Verify Python Environment**:
   ```bash
   which python  # Should point to correct Python
   ```

## 🔄 Migration from Direct MCP Usage

### Before (Direct MCP)
```json
{
  "tools": [{
    "type": "mcp",
    "server_url": "https://external-service.com/mcp",
    "headers": {"Authorization": "Bearer secret-token"},
    "require_approval": "never"
  }]
}
```

### After (Built-in MCP)
```yaml
# In config.yaml
mcp_builtin_servers:
  external_service:
    url: "https://external-service.com/mcp"
    transport: "sse"
    auth_type: "bearer_token"
    env_key: "EXTERNAL_SERVICE_TOKEN"
```

```json
{
  "tools": [{
    "type": "mcp",
    "builtin": "external_service",
    "require_approval": "never"
  }]
}
```

## 🎉 Benefits Summary

### Security
- ✅ No credential exposure to clients
- ✅ Centralized token management
- ✅ Server-side authentication handling

### Simplicity
- ✅ Simple `"builtin": "name"` reference
- ✅ No complex server configuration in requests
- ✅ Consistent interface across services

### Flexibility
- ✅ Config-based server definitions
- ✅ Client token override capability
- ✅ Support for multiple transport types
- ✅ Enable/disable servers per environment

### Management
- ✅ Easy to add new built-in servers
- ✅ Environment-based configuration
- ✅ Backward compatible with existing MCP usage
- ✅ Centralized server management

---

For quick setup, see the [3-Minute Quick Setup Guide](builtin_mcp_quick_setup.md).
For client token usage patterns, see [Client Token Examples](builtin_mcp_client_tokens.md).