# MCP Guardrails

LiteLLM supports guardrails for MCP (Model Context Protocol) tool calls, allowing you to apply security and validation checks to MCP tool executions through pre-call, during-call, and post-call hooks.

## Overview

MCP Guardrails provide:
- **Security policy enforcement**: Block forbidden keywords, unsafe content, and malicious patterns
- **Input validation**: Enforce query length limits, content rules, and data sanitization
- **Custom validation functions**: Implement user-defined security and business logic
- **Server and tool-specific filtering**: Apply guardrails to specific MCP servers or tools
- **Comprehensive monitoring**: Pre-call, during-call, and post-call hooks for complete oversight

## Use Cases

### 1. Security Policies and Prompt Injection Protection

Protect MCP tools from security threats and prompt injection attacks:

```yaml
guardrails:
  # Prompt injection protection using existing services
  - guardrail_name: "prompt-injection-guard"
    litellm_params:
      guardrail: "aporia"  # or "lakera", "bedrock"
      mode: "pre_call"
      api_key: os.environ/APORIA_API_KEY
      api_base: os.environ/APORIA_API_BASE

  # MCP-specific security policies
  - guardrail_name: "mcp-security-guard"
    litellm_params:
      guardrail_type: "mcp"
      mcp_server_name: "github"
      validation_rules:
        forbidden_keywords: [
          "ignore previous instructions",
          "ignore above",
          "forget everything", 
          "new instructions",
          "system prompt",
          "roleplay as",
          "act as",
          "pretend to be"
        ]
        unsafe_patterns: [
          "<script>",
          "javascript:",
          "data:",
          "vbscript:",
          "onload=",
          "onerror="
        ]
        max_query_length: 1000
        forbidden_search_terms: [
          "admin",
          "password",
          "secret",
          "private"
        ]
      custom_validation_function: "security.mcp_validation:validate_security"
      block_on_error: true
```

**Custom Security Validation Function:**
```python
# security/mcp_validation.py
def validate_security(tool_name: str, arguments: dict, validation_rules: dict) -> dict:
    """Comprehensive security validation for MCP tools"""
    
    # Check for prompt injection patterns
    prompt_injection_patterns = [
        "ignore previous instructions",
        "ignore above",
        "forget everything",
        "new instructions", 
        "system prompt",
        "roleplay as",
        "act as",
        "pretend to be"
    ]
    
    # Check for security threats
    for key, value in arguments.items():
        if isinstance(value, str):
            value_lower = value.lower()
            
            # Prompt injection detection
            for pattern in prompt_injection_patterns:
                if pattern in value_lower:
                    raise ValueError(f"Potential prompt injection detected: '{pattern}'")
            
            # Unsafe content detection
            unsafe_patterns = ["<script>", "javascript:", "data:", "vbscript:"]
            for pattern in unsafe_patterns:
                if pattern in value_lower:
                    raise ValueError(f"Unsafe content detected: '{pattern}'")
    
    return arguments
```

### 2. Content Filtering and Data Protection

Protect sensitive data and enforce content policies:

```yaml
guardrails:
  - guardrail_name: "content-protection-guard"
    litellm_params:
      guardrail_type: "mcp"
      mcp_server_name: "deepwiki"
      validation_rules:
        forbidden_keywords: [
          "password",
          "secret",
          "private",
          "confidential",
          "internal"
        ]
        pii_patterns: [
          r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
          r"\b\d{4}-\d{4}-\d{4}-\d{4}\b",  # Credit card
          r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"  # Email
        ]
        max_query_length: 500
        min_query_length: 3
      custom_validation_function: "security.content_validation:validate_content"
      block_on_error: true
```

## Quick Start

### 1. Define MCP Guardrails in your config.yaml

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY

# Define MCP servers
mcp_servers:
  - server_name: "github"
    auth_type: "oauth"
    server_url: "https://api.github.com"
    tools: ["search_repositories", "get_user", "create_issue"]
  
  - server_name: "deepwiki"
    auth_type: "none"
    server_url: "https://deepwiki.example.com"
    tools: ["search_wikipedia", "get_article"]

# Define guardrails
guardrails:
  - guardrail_name: "github-security-guard"
    litellm_params:
      guardrail_type: "mcp"
      mcp_server_name: "github"
      tool_name: "search_repositories"
      validation_rules:
        forbidden_keywords: ["secret", "password", "private"]
        max_query_length: 100
      custom_validation_function: "cookbook.github_validation:validate_github_request"
      block_on_error: true
      timeout: 30.0

  - guardrail_name: "deepwiki-content-guard"
    litellm_params:
      guardrail_type: "mcp"
      mcp_server_name: "deepwiki"
      validation_rules:
        forbidden_search_terms: ["admin", "password", "secret"]
        max_query_length: 200
        min_query_length: 3
      custom_validation_function: "cookbook.deepwiki_validation:validate_deepwiki_request"
      block_on_error: true
```

### 2. Create Custom Validation Functions

```python
# cookbook/github_validation.py
def validate_github_request(tool_name: str, arguments: dict, validation_rules: dict) -> dict:
    """Validate GitHub MCP tool requests"""
    
    # Check for sensitive data in search queries
    if "query" in arguments:
        query = arguments["query"].lower()
        forbidden = validation_rules.get("forbidden_keywords", [])
        
        for keyword in forbidden:
            if keyword.lower() in query:
                raise ValueError(f"Forbidden keyword '{keyword}' found in query")
    
    # Enforce query length limits
    max_length = validation_rules.get("max_query_length", 100)
    if len(arguments.get("query", "")) > max_length:
        raise ValueError(f"Query exceeds maximum length of {max_length} characters")
    
    return arguments

def validate_github_result(tool_name: str, result: dict, validation_rules: dict) -> dict:
    """Validate GitHub MCP tool results"""
    
    # Check result size limits
    if "items" in result and len(result["items"]) > 50:
        result["items"] = result["items"][:50]  # Limit to 50 results
    
    return result
```

### 3. Start the Proxy with Guardrails

```bash
litellm --config config.yaml --detailed_debug
```

## Configuration Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `guardrail_type` | string | Must be "mcp" for MCP guardrails | Required |
| `mcp_server_name` | string | MCP server to apply guardrails to | Optional |
| `tool_name` | string | Specific tool to apply guardrails to | Optional |
| `validation_rules` | dict | Rules for input validation | Optional |
| `custom_validation_function` | string | Path to custom validation function | Optional |
| `block_on_error` | boolean | Whether to block requests on validation failure | Optional |
| `timeout` | float | Timeout for guardrail execution (seconds) | Optional |

### Validation Rules

```yaml
validation_rules:
  # Keyword filtering
  forbidden_keywords: ["secret", "password", "private"]
  forbidden_search_terms: ["admin", "internal"]
  
  # Length limits
  max_query_length: 1000
  min_query_length: 3
  
  # Content patterns
  unsafe_patterns: ["<script>", "javascript:", "data:"]
  
  # Result validation
  max_result_size: 1000
  max_result_count: 50
```

## Custom Validation Functions

### Function Signature
```python
def validate_function(tool_name: str, arguments: dict, validation_rules: dict) -> dict:
    """
    Custom validation function for MCP tools
    
    Args:
        tool_name: Name of the MCP tool being called
        arguments: Tool arguments to validate
        validation_rules: Rules from configuration
    
    Returns:
        dict: Validated/modified arguments
    
    Raises:
        ValueError: If validation fails
    """
    # Your validation logic here
    return arguments
```

### Example: Advanced Security Validation

```python
# security/advanced_validation.py
import re
from typing import Dict, Any

def validate_advanced_security(tool_name: str, arguments: dict, validation_rules: dict) -> dict:
    """Advanced security validation for MCP tools"""
    
    # 1. Prompt injection detection
    prompt_injection_patterns = [
        r"ignore\s+(?:all\s+)?(?:previous\s+)?instructions",
        r"forget\s+(?:everything|all)",
        r"new\s+instructions",
        r"system\s+prompt",
        r"roleplay\s+as",
        r"act\s+as",
        r"pretend\s+to\s+be"
    ]
    
    # 2. PII detection
    pii_patterns = {
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}-\d{4}-\d{4}-\d{4}\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b\d{3}-\d{3}-\d{4}\b"
    }
    
    # 3. SQL injection detection
    sql_patterns = [
        r"(\b(union|select|insert|update|delete|drop|create|alter)\b)",
        r"(--|\b(union|select|insert|update|delete|drop|create|alter)\b)",
        r"(\b(union|select|insert|update|delete|drop|create|alter)\b.*\b(union|select|insert|update|delete|drop|create|alter)\b)"
    ]
    
    for key, value in arguments.items():
        if isinstance(value, str):
            value_lower = value.lower()
            
            # Check for prompt injection
            for pattern in prompt_injection_patterns:
                if re.search(pattern, value_lower, re.IGNORECASE):
                    raise ValueError(f"Potential prompt injection detected: '{pattern}'")
            
            # Check for PII
            for pii_type, pattern in pii_patterns.items():
                if re.search(pattern, value):
                    raise ValueError(f"Potential PII detected: {pii_type}")
            
            # Check for SQL injection
            for pattern in sql_patterns:
                if re.search(pattern, value_lower):
                    raise ValueError(f"Potential SQL injection detected")
    
    return arguments
```

## Hook Types

### Pre-call Hooks
Run before tool execution to validate inputs:

```python
async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
    """Validate MCP tool arguments before execution"""
    if self._should_apply_to_tool(data.get("name"), data.get("mcp_server_name")):
        validated_args = self._validate_tool_arguments(data["name"], data["arguments"])
        data["arguments"] = validated_args
    return None
```

### During-call Hooks
Run during tool execution for real-time monitoring:

```python
async def async_moderation_hook(self, data, user_api_key_dict, call_type):
    """Monitor MCP tool execution in real-time"""
    if self._should_apply_to_tool(data.get("name"), data.get("mcp_server_name")):
        # Real-time monitoring logic
        pass
    return None
```

### Post-call Hooks
Run after tool execution to validate results:

```python
async def async_post_call_success_hook(self, data, user_api_key_dict, response):
    """Validate MCP tool results after execution"""
    if self._should_apply_to_tool(data.get("name"), data.get("mcp_server_name")):
        validated_result = self._validate_tool_result(data["name"], response)
        return validated_result
    return response
```

## Integration with MCP Server Manager

The MCP guardrails are automatically integrated into the MCP server manager:

```python
# In MCP Server Manager
async def call_tool(self, tool_name: str, arguments: dict) -> dict:
    """Call MCP tool with guardrail protection"""
    
    # Apply pre-call guardrails
    guardrail_data = await self._apply_mcp_pre_call_guardrails(
        tool_name, arguments, user_api_key, cache
    )
    
    # Execute tool call
    result = await self._execute_tool_call(tool_name, guardrail_data["arguments"])
    
    # Apply post-call guardrails
    validated_result = await self._apply_mcp_post_call_guardrails(
        tool_name, result, user_api_key
    )
    
    return validated_result
```

## Testing

You can test your MCP guardrails using the provided test suite:

```bash
# Run simple test script
python test_mcp_guardrails_simple.py

# Run comprehensive tests
python -m pytest tests/mcp_tests/test_deepwiki_guardrail.py -v
```

## Best Practices

1. **Start with specific guardrails**: Begin with guardrails for specific tools/servers before implementing general ones.

2. **Use custom validation functions**: Implement domain-specific security logic for your use cases.

3. **Monitor and log**: Use the comprehensive logging to monitor guardrail effectiveness.

4. **Test thoroughly**: Test your guardrails with various inputs to ensure they work as expected.

5. **Gradual rollout**: Roll out guardrails gradually, starting with non-critical tools.

6. **Combine with existing guardrails**: Use MCP guardrails alongside existing security services for comprehensive protection.

## Troubleshooting

### Common Issues

1. **Custom validation function not loading**:
   - Ensure the function path is correct (module:function)
   - Check that the module is importable
   - Verify the function signature matches the expected format

2. **Guardrails not applying**:
   - Check that `mcp_server_name` and `tool_name` match your configuration
   - Verify that the guardrail is properly registered in the configuration

3. **Validation errors**:
   - Review the validation rules and custom functions
   - Check the logs for specific error messages
   - Test with simpler validation rules first

### Debug Mode

Enable detailed logging to troubleshoot issues:

```bash
litellm --config config.yaml --detailed_debug
```

Look for logs starting with:
- `MCP Guardrail validation error`
- `Custom validation function error`
- `MCP Guardrail: Pre-call validation failed`

## API Reference

For detailed API documentation, see the [MCP Guardrail API Reference](../api/guardrails/mcp.md).

## Examples

### GitHub Repository Search Protection

```yaml
guardrails:
  - guardrail_name: "github-search-guard"
    litellm_params:
      guardrail_type: "mcp"
      mcp_server_name: "github"
      tool_name: "search_repositories"
      validation_rules:
        forbidden_keywords: ["password", "secret", "private", "internal"]
        max_query_length: 100
        forbidden_search_terms: ["admin", "root", "system"]
      custom_validation_function: "security.github_validation:validate_search"
      block_on_error: true
```

### Wikipedia Content Filtering

```yaml
guardrails:
  - guardrail_name: "wikipedia-content-guard"
    litellm_params:
      guardrail_type: "mcp"
      mcp_server_name: "deepwiki"
      validation_rules:
        forbidden_search_terms: ["admin", "password", "secret"]
        max_query_length: 200
        min_query_length: 3
        unsafe_patterns: ["<script>", "javascript:", "data:"]
      custom_validation_function: "security.content_validation:validate_wikipedia"
      block_on_error: true
```

### Custom Security Validation

```python
# security/custom_validation.py
def validate_custom_security(tool_name: str, arguments: dict, rules: dict) -> dict:
    """Custom security validation for your specific use case"""
    
    # Your custom security logic here
    # Example: Check for company-specific patterns
    company_patterns = [
        r"internal-",
        r"secret-",
        r"admin-"
    ]
    
    for key, value in arguments.items():
        if isinstance(value, str):
            for pattern in company_patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    raise ValueError(f"Company policy violation: '{pattern}'")
    
    return arguments
``` 