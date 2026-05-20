import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Claude Code Quickstart

This tutorial shows how to call Claude models through LiteLLM proxy from Claude Code.

:::info 

This tutorial is based on [Anthropic's official LiteLLM configuration documentation](https://code.claude.com/docs/en/llm-gateway#litellm-configuration). This integration allows you to use any LiteLLM supported model through Claude Code with centralized authentication, usage tracking, and cost controls.

:::

<br />

### Video Walkthrough

<iframe width="840" height="500" src="https://www.loom.com/embed/3c17d683cdb74d36a3698763cc558f56" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

## Prerequisites

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) installed
- API keys for your chosen providers

## Installation

First, install LiteLLM with proxy support:

```bash
uv tool install 'litellm[proxy]'
```

### 1. Setup config.yaml

Create a secure configuration using environment variables:

```yaml
model_list:
  # Configure the models you want to use
  - model_name: claude-sonnet-4-5-20250929
    litellm_params:
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

litellm_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

Set your environment variables:

```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export LITELLM_MASTER_KEY="sk-1234567890"  # Generate a secure key
```

:::tip
Alternatively, you can store `ANTHROPIC_API_KEY` in a `.env` file in your proxy directory. LiteLLM will automatically load it when starting.
:::

### 2. Start proxy

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

### 3. Verify Setup

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

### 4. Configure Claude Code

#### Authentication methods

Claude Code authenticates to LiteLLM using your proxy **master key** or a [**virtual key**](https://docs.litellm.ai/docs/proxy/virtual_keys). You can provide that credential as a **static** value or fetch it **dynamically** with a helper script. See also [Anthropic's LLM gateway — authentication methods](https://code.claude.com/docs/en/llm-gateway#authentication-methods).

##### Static API key

Set a fixed LiteLLM master key or virtual key in the environment:

```bash
export ANTHROPIC_AUTH_TOKEN="$LITELLM_MASTER_KEY"
```

Or in Claude Code settings (`~/.claude/settings.json`):

```json
{
  "env": {
    "ANTHROPIC_AUTH_TOKEN": "sk-litellm-static-key"
  }
}
```

Claude Code sends this value as the `Authorization` header.

:::tip
`LITELLM_MASTER_KEY` gives Claude Code access to all proxy models. A [virtual key](https://docs.litellm.ai/docs/proxy/virtual_keys) limits access to specific models and enables per-user spend tracking.
:::

##### Dynamic API key with helper

For rotating keys or per-user authentication, use Claude Code's `apiKeyHelper` to run a script that prints a LiteLLM key to stdout (for example, fetch from a vault or mint a short-lived token):

1. Create a helper script:

```bash
#!/bin/bash
# ~/bin/get-litellm-key.sh
# Example: return a per-user LiteLLM virtual key
echo "$LITELLM_VIRTUAL_KEY"
```

2. Configure Claude Code settings:

```json
{
  "apiKeyHelper": "~/bin/get-litellm-key.sh"
}
```

3. Optional refresh interval (milliseconds):

```bash
export CLAUDE_CODE_API_KEY_HELPER_TTL_MS=3600000
```

The helper output is sent as `Authorization` and `X-Api-Key`. It has **lower precedence** than `ANTHROPIC_AUTH_TOKEN` or `ANTHROPIC_API_KEY` when those are also set.

#### Endpoint URL

Set `ANTHROPIC_BASE_URL` to point Claude Code at your LiteLLM proxy (in addition to the authentication settings above).

##### Method 1: Unified Endpoint (Recommended)

```bash
export ANTHROPIC_BASE_URL="http://0.0.0.0:4000"
```

Benefits: load balancing, fallbacks, and consistent cost tracking. See [Anthropic unified `/v1/messages`](https://docs.litellm.ai/docs/anthropic_unified).

##### Method 2: Provider-specific Pass-through Endpoint

```bash
export ANTHROPIC_BASE_URL="http://0.0.0.0:4000/anthropic"
```

### 5. Use Claude Code

Start Claude Code with the model you want to use:

```bash
# Specify model at startup
claude --model claude-sonnet-4-5-20250929

# Or specify a different model
claude --model claude-haiku-4-5-20251001
claude --model claude-opus-4-5-20251101

# Or change model during a session
claude
/model claude-sonnet-4-5-20250929
```

Alternatively, set default models with environment variables:

```bash
export ANTHROPIC_DEFAULT_SONNET_MODEL=claude-sonnet-4-5-20250929
export ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-haiku-4-5-20251001
export ANTHROPIC_DEFAULT_OPUS_MODEL=claude-opus-4-5-20251101
claude
```

### Using 1M Context Window

Claude Code supports extended context (1 million tokens) using the `[1m]` suffix:

```bash
# Use Sonnet with 1M context (requires quotes in shell)
claude --model 'claude-sonnet-4-5-20250929[1m]'

# Inside a Claude Code session (no quotes needed)
/model claude-sonnet-4-5-20250929[1m]
```

:::warning
**Important:** When using `--model` with `[1m]` in the shell, you must use quotes to prevent the shell from interpreting the brackets.
:::

**How it works:**
- Claude Code strips the `[1m]` suffix before sending to LiteLLM
- Claude Code automatically adds the header `anthropic-beta: context-1m-2025-08-07`
- Your LiteLLM config should **NOT** include `[1m]` in model names

**Verify 1M context is active:**
```bash
/context
# Should show: 21k/1000k tokens (2%)
```

Example conversation:

## Troubleshooting

Common issues and solutions:

**Claude Code not connecting:**
- Verify your proxy is running: `curl http://0.0.0.0:4000/health`
- Check that `ANTHROPIC_BASE_URL` is set correctly
- Ensure your `ANTHROPIC_AUTH_TOKEN` matches your LiteLLM master key

**Authentication errors:**
- Verify your environment variables are set: `echo $LITELLM_MASTER_KEY`
- Check that your API keys are valid and have sufficient credits
- Ensure the `ANTHROPIC_AUTH_TOKEN` matches your LiteLLM master key or virtual key
- If using `apiKeyHelper`, confirm the script is executable and prints only the key (no extra output). Remember `ANTHROPIC_AUTH_TOKEN` overrides the helper when both are set

**Model not found:**
- Ensure the model name in Claude Code matches exactly with your `config.yaml`
- Use `--model` flag or environment variables to specify the model
- Check LiteLLM logs for detailed error messages

## Using Bedrock/Vertex AI/Azure Foundry Models

Expand your configuration to support multiple providers and models:

<Tabs>
<TabItem value="multi-provider" label="Multi-Provider Setup">

```yaml
model_list:
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
      model: bedrock/anthropic.claude-haiku-4-5-20251001:0
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: us-east-1

  # Azure Foundry
  - model_name: claude-4-azure
    litellm_params:
      model: azure_ai/claude-opus-4-1
      api_key: os.environ/AZURE_AI_API_KEY
      api_base: os.environ/AZURE_AI_API_BASE # https://my-resource.services.ai.azure.com/anthropic

  # Google Vertex AI
  - model_name: anthropic-vertex
    litellm_params:
      model: vertex_ai/claude-haiku-4-5@20251001
      vertex_ai_project: "my-test-project"
      vertex_ai_location: "us-east-1"
      vertex_credentials: os.environ/VERTEX_FILE_PATH_ENV_VAR # os.environ["VERTEX_FILE_PATH_ENV_VAR"] = "/path/to/service_account.json" 




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

# Use Azure Foundry deployment
claude --model claude-4-azure

# Use Vertex AI deployment
claude --model anthropic-vertex
```

</TabItem>
</Tabs>

<Image img={require('../../img/release_notes/claude_code_demo.png')} style={{ width: '500px', height: 'auto' }} />

