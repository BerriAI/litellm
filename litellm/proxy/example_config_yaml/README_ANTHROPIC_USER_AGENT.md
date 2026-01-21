# Custom User-Agent Configuration for Anthropic

This guide explains how to configure custom User-Agent headers for Anthropic (Claude) API requests in LiteLLM Proxy.

## Problem

By default, LiteLLM adds `User-Agent: litellm/{version}` to all API requests. However, some Anthropic credentials are restricted to specific User-Agent values. For example, Claude Code credentials may return an error:

```
This credential is only authorized for use with Claude Code and cannot be used for other API requests.
```

## Solution

LiteLLM now supports customizing the User-Agent header for Anthropic requests in three ways:

### Option 1: Per-Model Configuration (Recommended)

Add `custom_user_agent` to your model's `litellm_params` in the proxy config YAML:

```yaml
model_list:
  - model_name: claude-code
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY
      custom_user_agent: "Claude Code/1.0"
```

### Option 2: Global Environment Variable

Set the `ANTHROPIC_USER_AGENT` environment variable to apply a custom User-Agent to all Anthropic requests:

```bash
export ANTHROPIC_USER_AGENT="Claude Code/1.0"
```

Then start your proxy:

```bash
litellm --config /path/to/config.yaml
```

### Option 3: Via Extra Headers

You can also set the User-Agent through `extra_headers`:

```yaml
model_list:
  - model_name: claude-code
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY
      extra_headers:
        User-Agent: "Claude Code/1.0"
```

## Priority Order

If multiple User-Agent configurations are present, they are applied in this priority order:

1. **`custom_user_agent` parameter** (highest priority)
2. **`ANTHROPIC_USER_AGENT` environment variable**
3. **`User-Agent` in `extra_headers`**
4. **Default `litellm/{version}`** (lowest priority)

## Complete Example

See [`anthropic_custom_user_agent_config.yaml`](./anthropic_custom_user_agent_config.yaml) for a complete working example.

## Testing

To verify your configuration works:

```bash
curl -X POST http://localhost:4000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-key" \
  -d '{
    "model": "claude-code",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Python SDK Usage

This feature also works when using LiteLLM as a Python SDK:

```python
import litellm

# Option 1: Via parameter
response = litellm.completion(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Hello"}],
    custom_user_agent="Claude Code/1.0"
)

# Option 2: Via environment variable
import os
os.environ["ANTHROPIC_USER_AGENT"] = "Claude Code/1.0"
response = litellm.completion(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Hello"}]
)

# Option 3: Via extra_headers
response = litellm.completion(
    model="anthropic/claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Hello"}],
    extra_headers={"User-Agent": "Claude Code/1.0"}
)
```

## Related

- GitHub Issue: [#19017](https://github.com/BerriAI/litellm/issues/19017)
- Anthropic API Documentation: https://docs.anthropic.com/
