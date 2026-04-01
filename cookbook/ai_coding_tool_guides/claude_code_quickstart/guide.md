# Claude Code with LiteLLM Quickstart

This guide shows how to call Claude models (and any LiteLLM-supported model) through LiteLLM proxy from Claude Code.

> **Note:** This integration is based on [Anthropic's official LiteLLM configuration documentation](https://docs.anthropic.com/en/docs/claude-code/llm-gateway#litellm-configuration). It allows you to use any LiteLLM supported model through Claude Code with centralized authentication, usage tracking, and cost controls.

## Video Walkthrough

Watch the full tutorial: https://www.loom.com/embed/3c17d683cdb74d36a3698763cc558f56

## Prerequisites

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) installed
- API keys for your chosen providers

## Installation

First, install LiteLLM with proxy support:

```bash
pip install 'litellm[proxy]'
```

## Step 1: Setup config.yaml

Create a secure configuration using environment variables:

```yaml
model_list:
  # Claude models
  - model_name: claude-3-5-sonnet-20241022    
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY
  
  - model_name: claude-3-5-haiku-20241022
    litellm_params:
      model: anthropic/claude-3-5-haiku-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

  
litellm_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

Set your environment variables:

```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export LITELLM_MASTER_KEY="sk-1234567890"  # Generate a secure key
```

## Step 2: Start Proxy

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

## Step 3: Verify Setup

Test that your proxy is working correctly:

```bash
curl -X POST http://0.0.0.0:4000/v1/messages \
-H "Authorization: Bearer $LITELLM_MASTER_KEY" \
-H "Content-Type: application/json" \
-d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1000,
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
}'
```

## Step 4: Configure Claude Code

### Method 1: Unified Endpoint (Recommended)

Configure Claude Code to use LiteLLM's unified endpoint. Either a virtual key or master key can be used here:

```bash
export ANTHROPIC_BASE_URL="http://0.0.0.0:4000"
export ANTHROPIC_AUTH_TOKEN="$LITELLM_MASTER_KEY"
```

> **Tip:** LITELLM_MASTER_KEY gives Claude access to all proxy models, whereas a virtual key would be limited to the models set in the UI.

### Method 2: Provider-specific Pass-through Endpoint

Alternatively, use the Anthropic pass-through endpoint:

```bash
export ANTHROPIC_BASE_URL="http://0.0.0.0:4000/anthropic"
export ANTHROPIC_AUTH_TOKEN="$LITELLM_MASTER_KEY"
```

## Step 5: Use Claude Code

### Choosing Your Model

You have two options for specifying which model Claude Code uses:

#### Option 1: Command Line / Session Model Selection

Specify the model directly when starting Claude Code or during a session:

```bash
# Specify model at startup
claude --model claude-3-5-sonnet-20241022

# Or change model during a session
/model claude-3-5-haiku-20241022
```

This method uses the exact model you specify.

#### Option 2: Environment Variables

Configure default models using environment variables:

```bash
# Tell Claude Code which models to use by default
export ANTHROPIC_DEFAULT_SONNET_MODEL=claude-3-5-sonnet-20241022
export ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-3-5-haiku-20241022
export ANTHROPIC_DEFAULT_OPUS_MODEL=claude-opus-3-5-20240229

claude  # Will use the models specified above
```

**Note:** Claude Code may cache the model from a previous session. If environment variables don't take effect, use Option 1 to explicitly set the model.

**Important:** The `model_name` in your LiteLLM config must match what Claude Code requests (either from env vars or command line).

### Using 1M Context Window

Claude Code supports extended context (1 million tokens) using the `[1m]` suffix with Claude 4+ models:

```bash
# Use Sonnet 4.5 with 1M context (requires quotes for shell)
claude --model 'claude-sonnet-4-5-20250929[1m]'

# Inside a Claude Code session (no quotes needed)
/model claude-sonnet-4-5-20250929[1m]
```

**Important:** When using `--model` with `[1m]` in the shell, you must use quotes to prevent the shell from interpreting the brackets.

Alternatively, set as default with environment variables:

```bash
export ANTHROPIC_DEFAULT_SONNET_MODEL='claude-sonnet-4-5-20250929[1m]'
claude
```

**How it works:**
- Claude Code strips the `[1m]` suffix before sending to LiteLLM
- Claude Code automatically adds the header `anthropic-beta: context-1m-2025-08-07`
- Your LiteLLM config should **NOT** include `[1m]` in model names

**Verify 1M context is active:**
```bash
/context
# Should show: 21k/1000k tokens (2%)
```

**Pricing:** Models using 1M context have different pricing. Input tokens above 200k are charged at a higher rate.

## Troubleshooting

Common issues and solutions:

**Claude Code not connecting:**
- Verify your proxy is running: `curl http://0.0.0.0:4000/health`
- Check that `ANTHROPIC_BASE_URL` is set correctly
- Ensure your `ANTHROPIC_AUTH_TOKEN` matches your LiteLLM master key

**Authentication errors:**
- Verify your environment variables are set: `echo $LITELLM_MASTER_KEY`
- Check that your API keys are valid and have sufficient credits
- Ensure the `ANTHROPIC_AUTH_TOKEN` matches your LiteLLM master key

**Model not found:**
- Check what model Claude Code is requesting in LiteLLM logs
- Ensure your `config.yaml` has a matching `model_name` entry
- If using environment variables, verify they're set: `echo $ANTHROPIC_DEFAULT_SONNET_MODEL`

**1M context not working (showing 200k instead of 1000k):**
- Verify you're using the `[1m]` suffix: `/model your-model-name[1m]`
- Check LiteLLM logs for the header `context-1m-2025-08-07` in the request
- Ensure your model supports 1M context (only certain Claude models do)
- Your LiteLLM config should **NOT** include `[1m]` in the `model_name`

## Using Multiple Models and Providers

You can configure LiteLLM to route to any supported provider. Here's an example with multiple providers:

```yaml
model_list:
  # OpenAI models
  - model_name: codex-mini
    litellm_params:
      model: openai/codex-mini
      api_key: os.environ/OPENAI_API_KEY
      api_base: https://api.openai.com/v1

  - model_name: o3-pro
    litellm_params:
      model: openai/o3-pro
      api_key: os.environ/OPENAI_API_KEY
      api_base: https://api.openai.com/v1

  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
      api_base: https://api.openai.com/v1

  # Anthropic models
  - model_name: claude-3-5-sonnet-20241022
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: claude-3-5-haiku-20241022
    litellm_params:
      model: anthropic/claude-3-5-haiku-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

  # AWS Bedrock
  - model_name: claude-bedrock
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: us-east-1

litellm_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

**Note:** The `model_name` can be anything you choose. Claude Code will request whatever model you specify (via env vars or command line), and LiteLLM will route to the `model` configured in `litellm_params`.

Switch between models seamlessly:

```bash
# Use environment variables to set defaults
export ANTHROPIC_DEFAULT_SONNET_MODEL=claude-3-5-sonnet-20241022
export ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-3-5-haiku-20241022

# Or specify directly
claude --model claude-3-5-sonnet-20241022  # Complex reasoning
claude --model claude-3-5-haiku-20241022    # Fast responses
claude --model claude-bedrock                # Bedrock deployment
```

## Default Models Used by Claude Code

If you **don't** set environment variables, Claude Code uses these default model names:

| Purpose | Default Model Name (v2.1.14) |
|---------|------------------------------|
| Main model | `claude-sonnet-4-5-20250929` |
| Light tasks (subagents, summaries) | `claude-haiku-4-5-20251001` |
| Planning mode | `claude-opus-4-5-20251101` |

Your LiteLLM config should include these model names if you want Claude Code to work without setting environment variables:

```yaml
model_list:
  - model_name: claude-sonnet-4-5-20250929
    litellm_params:
      # Can be any provider - Anthropic, Bedrock, Vertex AI, etc.
      model: anthropic/claude-sonnet-4-5-20250929
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: claude-haiku-4-5-20251001
    litellm_params:
      model: anthropic/claude-haiku-4-5-20251001
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: claude-opus-4-5-20251101
    litellm_params:
      model: anthropic/claude-opus-4-5-20251101
      api_key: os.environ/ANTHROPIC_API_KEY
```

**Warning:** These default model names may change with new Claude Code versions. Check LiteLLM proxy logs for "model not found" errors to identify what Claude Code is requesting.

## Additional Resources

- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code/overview)
- [Anthropic's LiteLLM Configuration Guide](https://docs.anthropic.com/en/docs/claude-code/llm-gateway#litellm-configuration)

