# Forward Client Headers to LLM API

Control which model groups can forward client headers to the underlying LLM provider APIs.

## Overview

By default, LiteLLM does not forward client headers to LLM provider APIs for security reasons. However, you can selectively enable header forwarding for specific model groups using the `forward_client_headers_to_llm_api` setting.

## Configuration

## Enable Globally

```yaml
general_settings:
  forward_client_headers_to_llm_api: true
```

## Enable for a Model Group

Add the `forward_client_headers_to_llm_api` setting under `model_group_settings` in your configuration:

```yaml
model_list:
  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: "your-api-key"
  - model_name: "wildcard-models/*"
    litellm_params:
      model: "openai/*"
      api_key: "your-api-key"

litellm_settings:
  model_group_settings:
    forward_client_headers_to_llm_api:
      - gpt-4o-mini
      - wildcard-models/*
```

## Supported Model Patterns

The configuration supports various model matching patterns:

### 1. Exact Model Names
```yaml
forward_client_headers_to_llm_api:
  - gpt-4o-mini
  - claude-3-sonnet
```

### 2. Wildcard Patterns
```yaml
forward_client_headers_to_llm_api:
  - "openai/*"          # All OpenAI models
  - "anthropic/*"       # All Anthropic models
  - "wildcard-group/*"  # All models in wildcard-group
```

### 3. Team Model Aliases
If your team has model aliases configured, the forwarding will work with both the original model name and the alias.

## Forwarded Headers

When enabled for a model group, LiteLLM forwards the following types of headers:

### Custom Headers (x- prefix)
- Any header starting with `x-` (except `x-stainless-*` which can cause OpenAI SDK issues)
- Examples: `x-custom-header`, `x-request-id`, `x-trace-id`

### Provider-Specific Headers
- **Anthropic**: `anthropic-beta` headers
- **OpenAI**: `openai-organization` (when enabled via `forward_openai_org_id: true`)

### User Information Headers (Optional)
When `add_user_information_to_llm_headers` is enabled, LiteLLM adds:
- `x-litellm-user-id`
- `x-litellm-org-id`
- Other user metadata as `x-litellm-*` headers

## Security Considerations

⚠️ **Important Security Notes:**

1. **Sensitive Data**: Only enable header forwarding for trusted model groups, as headers may contain sensitive information
2. **API Keys**: Never include API keys or secrets in forwarded headers
3. **PII**: Be cautious about forwarding headers that might contain personally identifiable information
4. **Provider Limits**: Some providers have restrictions on custom headers

## Example Use Cases

### 1. Request Tracing
Forward tracing headers to track requests across your system:

```bash
curl -X POST "https://your-proxy.com/v1/chat/completions" \
  -H "Authorization: Bearer your-key" \
  -H "x-trace-id: abc123" \
  -H "x-request-source: mobile-app" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### 2. Custom Metadata
Pass custom metadata to your LLM provider:

```bash
curl -X POST "https://your-proxy.com/v1/chat/completions" \
  -H "Authorization: Bearer your-key" \
  -H "x-customer-id: customer-123" \
  -H "x-environment: production" \
  -d '{
    "model": "gpt-4o-mini", 
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### 3. Anthropic Beta Features
Enable beta features for Anthropic models:

```bash
curl -X POST "https://your-proxy.com/v1/chat/completions" \
  -H "Authorization: Bearer your-key" \
  -H "anthropic-beta: tools-2024-04-04" \
  -d '{
    "model": "claude-3-sonnet",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

## Complete Configuration Example

```yaml
model_list:
  # Fixed model with header forwarding
  - model_name: byok-fixed-gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_base: "https://your-openai-endpoint.com"
      api_key: "your-api-key"
      
  # Wildcard model group with header forwarding
  - model_name: "byok-wildcard/*"
    litellm_params:
      model: "openai/*"
      api_base: "https://your-openai-endpoint.com"
      api_key: "your-api-key"
      
  # Standard model without header forwarding
  - model_name: standard-gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: "your-api-key"

litellm_settings:
  # Enable user info headers globally (optional)
  add_user_information_to_llm_headers: true
  
  model_group_settings:
    forward_client_headers_to_llm_api:
      - byok-fixed-gpt-4o-mini
      - byok-wildcard/*
      # Note: standard-gpt-4 is NOT included, so no headers forwarded

general_settings:
  # Enable OpenAI organization header forwarding (optional)
  forward_openai_org_id: true
```

## Testing Header Forwarding

To test if headers are being forwarded:

1. **Enable Debug Logging**: Set `set_verbose: true` in your config
2. **Check Provider Logs**: Monitor your LLM provider's request logs
3. **Use Webhook Sites**: For testing, you can use webhook.site URLs as api_base to see forwarded headers

## Troubleshooting

### Headers Not Being Forwarded

1. **Check Model Name**: Ensure the model name in your request matches the configuration
2. **Verify Pattern Matching**: Wildcard patterns must match exactly
3. **Review Logs**: Enable verbose logging to see header processing

### Provider Errors

1. **Invalid Headers**: Some providers reject unknown headers
2. **Header Limits**: Providers may have limits on header count/size
3. **Authentication**: Ensure forwarded headers don't conflict with authentication

## Related Features

- [Request Headers](./request_headers.md) - Complete list of supported request headers
- [Response Headers](./response_headers.md) - Headers returned by LiteLLM
- [Team Model Aliases](./team_model_add.md) - Configure model aliases for teams
- [Model Access Control](./model_access.md) - Control which users can access which models

## API Reference

The header forwarding is controlled by the `ModelGroupSettings` configuration:

```python
class ModelGroupSettings(BaseModel):
    forward_client_headers_to_llm_api: Optional[List[str]] = None
```

Where each string in the list can be:
- An exact model name (e.g., `"gpt-4o-mini"`)
- A wildcard pattern (e.g., `"openai/*"`)
- A model group name (e.g., `"my-model-group/*"`)
