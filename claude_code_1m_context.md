# Using Claude Code with 1M Context Window via LiteLLM

This guide explains how to use Claude's 1 million token context window with Claude Code through LiteLLM proxy.

## Overview

Claude Code supports extended context (1M tokens) using the `[1m]` suffix on model names. When you use this suffix, Claude Code automatically handles the necessary configuration to enable the larger context window.

## How It Works

When you specify a model with `[1m]` in Claude Code:

1. **Claude Code strips the `[1m]` suffix** before sending the request to LiteLLM
2. **Claude Code adds the header** `anthropic-beta: context-1m-2025-08-07` automatically
3. **LiteLLM routes the request** to the configured model (without the `[1m]` suffix)
4. **Anthropic API receives** the request with the proper header for 1M context

## Configuration

### LiteLLM Config

Your LiteLLM config should **NOT** include the `[1m]` suffix in model names:

```yaml
model_list:
  # Regular model name - NO [1m] suffix in config
  - model_name: claude-sonnet-4-5-20250929
    litellm_params:
      model: anthropic/claude-sonnet-4-5-20250929
      api_key: os.environ/ANTHROPIC_API_KEY

  # Other models
  - model_name: claude-haiku-4-5-20251001
    litellm_params:
      model: anthropic/claude-haiku-4-5-20251001
      api_key: os.environ/ANTHROPIC_API_KEY

litellm_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

**Important:** Do NOT add `extra_headers` with `anthropic-beta` in the config - Claude Code sends this header automatically when you use `[1m]`.

### Using 1M Context in Claude Code

In Claude Code, simply add `[1m]` to the model name:

```bash
# Start Claude Code with 1M context
claude --model claude-sonnet-4-5-20250929[1m]

# Or change model during session
/model claude-sonnet-4-5-20250929[1m]
```

### With Bedrock

For Bedrock deployments, the configuration is similar:

```yaml
model_list:
  - model_name: us.anthropic.claude-sonnet-4-20250514-v1:0
    litellm_params:
      model: bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: us-east-1
```

In Claude Code:
```bash
/model us.anthropic.claude-sonnet-4-20250514-v1:0[1m]
```

## Verifying 1M Context

To verify that 1M context is working, use the `/context` command in Claude Code:

```bash
/context
```

You should see:
```
Context Usage
⛀ ⛀ ⛀ ⛀ ⛶   claude-sonnet-4-5-20250929[1m] ·
21k/1000k tokens (2%)
```

The **1000k** (1,000,000 tokens) confirms that 1M context is active.

You can also check the LiteLLM proxy logs for the header:

```
'anthropic-beta': 'claude-code-20250219,oauth-2025-04-20,context-1m-2025-08-07,interleaved-thinking-2025-05-14'
```

The presence of `context-1m-2025-08-07` confirms that Claude Code is requesting 1M context.

## Troubleshooting

### Context still showing 200k

**Issue:** Context shows 200k tokens instead of 1000k even when using `[1m]`

**Solutions:**
1. **Verify the `[1m]` suffix is present** in Claude Code model name
2. **Check LiteLLM logs** to confirm the `context-1m-2025-08-07` header is received
3. **Ensure the model supports 1M context** - only certain Claude models support extended context
4. **Restart Claude Code session** after changing the model

### Model not found error

**Issue:** LiteLLM returns "model not found" when using `[1m]`

**Solution:** Make sure your LiteLLM config has the model name **without** `[1m]`:
- ✅ Config: `claude-sonnet-4-5-20250929`
- ❌ Config: `claude-sonnet-4-5-20250929[1m]`

Claude Code strips the `[1m]` before sending to LiteLLM, so the config should not include it.

## Pricing Considerations

Models using 1M context have different pricing:
- Input tokens above 200k are charged at a higher rate
- Check Anthropic's pricing page for current rates
- Use `/context` to monitor token usage

## Supported Models

As of Claude Code v2.1.14, the following models support 1M context with the `[1m]` suffix:

- `claude-sonnet-4-5-20250929[1m]`
- `claude-sonnet-4-20250514[1m]` (Bedrock)
- `anthropic.claude-sonnet-4-20250514-v1:0[1m]` (Bedrock with region prefix)
- `us.anthropic.claude-sonnet-4-20250514-v1:0[1m]` (Bedrock with region prefix)

Check Anthropic's documentation for the latest list of supported models.

## Example: Full Working Setup

### 1. LiteLLM Config (`config.yaml`)

```yaml
model_list:
  - model_name: claude-sonnet-4-5-20250929
    litellm_params:
      model: anthropic/claude-sonnet-4-5-20250929
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: claude-haiku-4-5-20251001
    litellm_params:
      model: anthropic/claude-haiku-4-5-20251001
      api_key: os.environ/ANTHROPIC_API_KEY

litellm_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

### 2. Start LiteLLM Proxy

```bash
export ANTHROPIC_API_KEY=your-anthropic-key
export LITELLM_MASTER_KEY=sk-1234
litellm --config config.yaml --port 4000
```

### 3. Configure Claude Code

```bash
export ANTHROPIC_BASE_URL=http://localhost:4000
export ANTHROPIC_API_KEY=sk-1234  # Must match LITELLM_MASTER_KEY
```

### 4. Start Claude Code with 1M Context

```bash
claude --model claude-sonnet-4-5-20250929[1m]
```

### 5. Verify

```bash
# In Claude Code
/context

# Should show:
# 21k/1000k tokens (2%)
```

## Related Issues

- GitHub Issue #14444: Claude Code with Sonnet 4 1M context Window on LiteLLM
