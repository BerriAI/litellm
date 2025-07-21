# LiteLLM Built-in MCP - Client Token Usage

## 📖 Overview
Clients can provide their own authentication tokens for personalized MCP service usage.

## 🔑 Basic Usage

### Server Environment Variable Method (Default)
```bash
export ZAPIER_TOKEN="server_token"
```
```json
{"tools": [{"type": "mcp", "builtin": "zapier"}]}
```

### Client Token Method (New)
```json
{
  "tools": [{
    "type": "mcp",
    "builtin": "zapier",
    "auth_token": "user_personal_token"
  }]
}
```

## 🎯 Real-world Examples

### Personal GitHub Repository Management
```json
{
  "model": "gpt-4o-mini",
  "input": [{"type": "message", "role": "user", "content": "List my repositories"}],
  "tools": [{
    "type": "mcp",
    "builtin": "github", 
    "auth_token": "ghp_your_personal_token"
  }]
}
```

### Personal Zapier Workflows
```json
{
  "model": "gpt-4o-mini",
  "input": [{"type": "message", "role": "user", "content": "Send message to Slack"}],
  "tools": [{
    "type": "mcp",
    "builtin": "zapier",
    "auth_token": "user_zapier_key_123"
  }]
}
```

### Multi-service Personalized Usage
```json
{
  "tools": [
    {"type": "mcp", "builtin": "calculator"},
    {"type": "mcp", "builtin": "zapier", "auth_token": "personal_zapier"},
    {"type": "mcp", "builtin": "github", "auth_token": "personal_github"},
    {"type": "mcp", "builtin": "jira", "auth_token": "work_jira_token"}
  ]
}
```

## 💡 Token Priority

1. **Client Token** (`auth_token`) ← Highest priority
2. **Server Environment Variable** (`ZAPIER_TOKEN` etc.) ← Fallback
3. **No Token** ← Service unavailable

## 🛡️ Security Recommendations

- Manage sensitive tokens directly on the client side
- Set only public/test tokens on the server
- Pass personal tokens only in requests

## 🔧 Supported Field Names

- `auth_token` (recommended)
- `authentication_token` (compatibility)

Use client tokens for safer and more personalized MCP services! 🚀