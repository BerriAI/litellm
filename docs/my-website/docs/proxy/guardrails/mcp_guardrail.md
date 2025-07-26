import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# MCP Guardrails

LiteLLM supports guardrails for MCP (Model Context Protocol) tool calls, allowing you to apply security and validation checks to MCP tool executions through pre-call, during-call, and post-call hooks.

## Overview

MCP Guardrails provide:
- **Pre-call validation**: Validate tool arguments before execution
- **During-call monitoring**: Monitor tool execution in real-time
- **Post-call validation**: Validate tool results after execution
- **Custom validation functions**: Define your own validation logic
- **Server and tool-specific filtering**: Apply guardrails to specific MCP servers or tools

## Quick Start

### 1. Define MCP Guardrails in your config.yaml

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

# MCP Servers Configuration
mcp_servers:
  - server_name: github
    url: "https://api.github.com/mcp"
    transport: "http"
    auth_type: "bearer_token"
    authentication_token: os.environ/GITHUB_TOKEN

guardrails:
  - guardrail_name: "github-security"
    guardrail_type: "mcp"
    mcp_server_name: "github"
    block_on_error: true
    timeout: 30.0
    validation_rules:
      max_title_length: 256
      max_body_length: 65536
      forbidden_keywords: ["password", "secret", "api_key"]
    custom_validation_function: "my_validation:validate_github_request"
    event_hooks:
      - "pre_call"
      - "post_call"
```

### 2. Create Custom Validation Functions

```python
# my_validation.py
def validate_github_request(tool_name: str, arguments: dict, validation_rules: dict) -> dict:
    """Custom validation for GitHub MCP tools"""
    
    if tool_name == "create_issue":
        # Check for required fields
        required_fields = ["title", "body"]
        for field in required_fields:
            if field not in arguments:
                raise ValueError(f"Required field '{field}' is missing")
        
        # Validate title length
        max_title_length = validation_rules.get("max_title_length", 256)
        if len(arguments.get("title", "")) > max_title_length:
            raise ValueError(f"Title exceeds maximum length of {max_title_length}")
        
        # Check for forbidden keywords
        forbidden_keywords = validation_rules.get("forbidden_keywords", [])
        title_lower = arguments.get("title", "").lower()
        body_lower = arguments.get("body", "").lower()
        
        for keyword in forbidden_keywords:
            if keyword.lower() in title_lower or keyword.lower() in body_lower:
                raise ValueError(f"Forbidden keyword '{keyword}' found in issue content")
    
    return arguments
```

### 3. Start the Proxy with Guardrails

```bash
litellm --config config.yaml
```

## Configuration Options

### Guardrail Configuration

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `guardrail_name` | string | Unique name for the guardrail | Required |
| `guardrail_type` | string | Must be "mcp" for MCP guardrails | Required |
| `mcp_server_name` | string | MCP server to apply guardrails to | Optional |
| `tool_name` | string | Specific tool to apply guardrails to | Optional |
| `block_on_error` | boolean | Whether to block requests on validation failure | false |
| `timeout` | float | Timeout for validation in seconds | 30.0 |
| `validation_rules` | object | Custom validation rules | {} |
| `custom_validation_function` | string | Path to custom validation function | Optional |
| `event_hooks` | array | Which hooks to enable: pre_call, during_call, post_call | ["pre_call"] |

### Validation Rules

Common validation rules you can configure:

```yaml
validation_rules:
  # Length limits
  max_title_length: 256
  max_body_length: 65536
  max_argument_size: 10240  # 10KB
  
  # Content restrictions
  forbidden_keywords: ["password", "secret", "api_key"]
  allowed_repositories: ["my-org/repo1", "my-org/repo2"]
  allowed_webhook_urls: ["https://my-app.com/webhook"]
  
  # Tool patterns
  allowed_tool_patterns: ["get_*", "list_*", "create_*"]
  forbidden_tool_patterns: ["delete_*", "admin_*"]
  
  # Rate limiting
  max_webhook_count: 10
  max_per_page: 100
```

## Hook Types

### Pre-Call Hooks

Pre-call hooks validate tool arguments before execution:

```python
async def async_pre_call_hook(self, data: dict, user_api_key_dict: UserAPIKeyAuth, ...) -> dict:
    # Validate tool arguments
    tool_name = data.get("name")
    arguments = data.get("arguments", {})
    
    # Apply validation rules
    validated_arguments = self._validate_tool_arguments(tool_name, arguments)
    
    # Return validated data
    data["arguments"] = validated_arguments
    return data
```

### During-Call Hooks

During-call hooks monitor tool execution in real-time:

```python
async def async_moderation_hook(self, data: dict, user_api_key_dict: UserAPIKeyAuth, ...) -> dict:
    # Monitor execution
    # Track performance
    # Apply rate limiting
    # Security monitoring
    return data
```

### Post-Call Hooks

Post-call hooks validate tool results after execution:

```python
async def async_post_call_success_hook(self, data: dict, user_api_key_dict: UserAPIKeyAuth, response: Any) -> Any:
    # Validate result
    validated_result = self._validate_tool_result(tool_name, response)
    
    # Check for sensitive data
    # Apply result filtering
    return validated_result
```

## Examples

### GitHub Issue Creation Guardrail

```yaml
guardrails:
  - guardrail_name: "github-issue-guard"
    guardrail_type: "mcp"
    mcp_server_name: "github"
    tool_name: "create_issue"
    block_on_error: true
    validation_rules:
      max_title_length: 100
      max_body_length: 10000
      forbidden_keywords: ["password", "secret", "token"]
      allowed_repositories: ["my-org/repo1", "my-org/repo2"]
    event_hooks:
      - "pre_call"
      - "post_call"
```

### Zapier Webhook Security

```yaml
guardrails:
  - guardrail_name: "zapier-webhook-guard"
    guardrail_type: "mcp"
    mcp_server_name: "zapier"
    tool_name: "create_webhook"
    block_on_error: true
    validation_rules:
      allowed_webhook_urls: ["https://my-app.com/webhook"]
      forbidden_patterns: ["localhost", "127.0.0.1"]
    custom_validation_function: "my_validation:validate_zapier_webhook"
    event_hooks:
      - "pre_call"
      - "during_call"
      - "post_call"
```

### General MCP Security

```yaml
guardrails:
  - guardrail_name: "general-mcp-guard"
    guardrail_type: "mcp"
    block_on_error: false  # Log warnings but don't block
    validation_rules:
      max_argument_size: 10240
      allowed_tool_patterns: ["get_*", "list_*", "create_*"]
      forbidden_tool_patterns: ["delete_*", "admin_*"]
      forbidden_patterns: ["<script>", "javascript:", "data:"]
    event_hooks:
      - "pre_call"
      - "post_call"
```

## Custom Validation Functions

You can create custom validation functions for specific use cases:

```python
# my_validation.py
import re
import json
from typing import Any, Dict

def validate_github_request(tool_name: str, arguments: Dict[str, Any], validation_rules: Dict[str, Any]) -> Dict[str, Any]:
    """GitHub-specific validation"""
    if tool_name == "create_issue":
        # Your custom validation logic
        pass
    return arguments

def validate_zapier_webhook(tool_name: str, arguments: Dict[str, Any], validation_rules: Dict[str, Any]) -> Dict[str, Any]:
    """Zapier webhook validation"""
    if tool_name == "create_webhook":
        # Your custom validation logic
        pass
    return arguments

def validate_general_mcp(tool_name: str, arguments: Dict[str, Any], validation_rules: Dict[str, Any]) -> Dict[str, Any]:
    """General MCP validation"""
    # Your custom validation logic
    return arguments
```

## Error Handling

### Validation Errors

When validation fails, the guardrail can either:

1. **Block the request** (when `block_on_error: true`):
   ```python
   raise ValueError("MCP Guardrail pre-call validation failed: Forbidden keyword 'password' found")
   ```

2. **Log warnings** (when `block_on_error: false`):
   ```python
   verbose_proxy_logger.warning("MCP Guardrail validation failed: Forbidden keyword 'password' found")
   ```

### Timeout Handling

If validation takes too long:

```python
# Guardrail will timeout after the specified timeout period
if timeout_exceeded:
    raise TimeoutError("MCP Guardrail validation timed out")
```

## Integration with MCP Server Manager

The MCP guardrails are automatically integrated into the MCP server manager:

```python
# In mcp_server_manager.py
async def call_tool(self, name: str, arguments: Dict[str, Any], ...) -> CallToolResult:
    # Apply pre-call guardrails
    guardrail_data = await self._apply_mcp_pre_call_guardrails(
        data={"name": name, "arguments": arguments, "mcp_server_name": server_name},
        user_api_key_auth=user_api_key_auth
    )
    
    # Execute tool call
    result = await self._execute_tool_call(guardrail_data)
    
    # Apply post-call guardrails
    validated_result = await self._apply_mcp_post_call_guardrails(
        data=guardrail_data,
        user_api_key_auth=user_api_key_auth,
        result=result
    )
    
    return validated_result
```

## Testing

You can test your MCP guardrails using the provided test suite:

```bash
# Run MCP guardrail tests
pytest tests/mcp_tests/test_mcp_guardrail.py -v

# Run specific test
pytest tests/mcp_tests/test_mcp_guardrail.py::TestMCPGuardrail::test_pre_call_hook -v
```

## Best Practices

1. **Start with specific guardrails**: Begin with guardrails for specific tools/servers before implementing general ones.

2. **Use custom validation functions**: For complex validation logic, create custom functions rather than relying solely on validation rules.

3. **Monitor and log**: Use `block_on_error: false` initially to monitor validation failures without blocking requests.

4. **Test thoroughly**: Test your guardrails with various inputs to ensure they work as expected.

5. **Document validation rules**: Keep your validation rules well-documented for team members.

6. **Gradual rollout**: Roll out guardrails gradually, starting with non-critical tools.

## Troubleshooting

### Common Issues

1. **Guardrail not applying**: Check that the `mcp_server_name` and `tool_name` match your MCP server configuration.

2. **Custom validation function not loading**: Ensure the function path is correct and the function is importable.

3. **Validation timeouts**: Increase the `timeout` value if your validation functions are slow.

4. **False positives**: Review your validation rules and adjust them based on actual usage patterns.

### Debugging

Enable debug logging to see guardrail activity:

```yaml
logging:
  level: "DEBUG"
  format: "json"
```

Look for log messages like:
- `"MCP Guardrail: Pre-call validation completed for tool create_issue"`
- `"MCP Guardrail: Skipping validation for tool list_issues (not in scope)"`
- `"MCP Guardrail: Pre-call validation failed: Forbidden keyword 'password' found"`

## API Reference

For detailed API documentation, see the [MCP Guardrail API Reference](../api/guardrails/mcp.md). 