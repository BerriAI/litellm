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

Start Claude Code and it will automatically use your configured models:

```bash
# Claude Code will use the models configured in your LiteLLM proxy
claude

# Or specify a model if you have multiple configured
claude --model claude-3-5-sonnet-20241022
claude --model claude-3-5-haiku-20241022
```

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
- Ensure the model name in Claude Code matches exactly with your `config.yaml`
- Check LiteLLM logs for detailed error messages

## Using Multiple Models and Providers

Expand your configuration to support multiple providers and models:

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

Switch between models seamlessly:

```bash
# Use Claude for complex reasoning
claude --model claude-3-5-sonnet-20241022

# Use Haiku for fast responses
claude --model claude-3-5-haiku-20241022

# Use Bedrock deployment
claude --model claude-bedrock
```

## Additional Resources

- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code/overview)
- [Anthropic's LiteLLM Configuration Guide](https://docs.anthropic.com/en/docs/claude-code/llm-gateway#litellm-configuration)

