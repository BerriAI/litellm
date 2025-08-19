import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Claude Code

This tutorial shows how to use Anthropic Claude models with Claude Code through LiteLLM, and also supports Responses API models like `codex-mini` and `o3-pro`.

:::info 

This tutorial is based on [Anthropic's official LiteLLM configuration documentation](https://docs.anthropic.com/en/docs/claude-code/llm-gateway#litellm-configuration). This integration allows you to use any LiteLLM supported model through Claude Code.

:::

## Prerequisites

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) installed
- API keys for your chosen providers

## Installation

First, install LiteLLM with proxy support:

```bash
pip install 'litellm[proxy]'
```

### 1. Setup config.yaml

Create a secure configuration using environment variables:

<Tabs>
<TabItem value="anthropic" label="Anthropic Claude (Recommended)" default>

```yaml
model_list:
  # Anthropic Claude models
  - model_name: claude-3-5-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY
  
  - model_name: claude-3-5-haiku
    litellm_params:
      model: anthropic/claude-3-5-haiku-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: claude-3-opus
    litellm_params:
      model: anthropic/claude-3-opus-20240229
      api_key: os.environ/ANTHROPIC_API_KEY

litellm_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

Set your environment variables:

```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export LITELLM_MASTER_KEY="sk-1234567890"  # Generate a secure key
```

</TabItem>
<TabItem value="responses-api" label="Responses API Models">

```yaml
model_list:
  # Responses API models
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

litellm_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

Set your environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export LITELLM_MASTER_KEY="sk-1234567890"  # Generate a secure key
```

</TabItem>
</Tabs>

### 2. Start proxy

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

### 3. Verify Setup

Test that your proxy is working correctly:

<Tabs>
<TabItem value="anthropic-test" label="Test Anthropic Claude" default>

```bash
curl -X POST http://0.0.0.0:4000/v1/messages \
-H "Authorization: Bearer $LITELLM_MASTER_KEY" \
-H "Content-Type: application/json" \
-d '{
    "model": "claude-3-5-sonnet",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
}'
```

</TabItem>
<TabItem value="responses-test" label="Test Responses API">

```bash
curl -X POST http://0.0.0.0:4000/v1/messages \
-H "Authorization: Bearer $LITELLM_MASTER_KEY" \
-H "Content-Type: application/json" \
-d '{
    "model": "codex-mini",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
}'
```

</TabItem>
</Tabs>

### 4. Configure Claude Code

Setup Claude Code to use your LiteLLM proxy:

```bash
export ANTHROPIC_BASE_URL="http://0.0.0.0:4000"
export ANTHROPIC_AUTH_TOKEN="$LITELLM_MASTER_KEY"
```

### 5. Use Claude Code

Start Claude Code with any configured model:

<Tabs>
<TabItem value="anthropic-usage" label="Use Anthropic Claude" default>

```bash
# Use Anthropic Claude models
claude --model claude-3-5-sonnet
claude --model claude-3-5-haiku
claude --model claude-3-opus
```

</TabItem>
<TabItem value="responses-usage" label="Use Responses API">

```bash
# Use Responses API models
claude --model codex-mini
claude --model o3-pro

# Or use the latest model alias
claude --model codex-mini-latest
```

</TabItem>
</Tabs>

Example conversation:

## Troubleshooting

Common issues and solutions:

**Claude Code not connecting:**
- Verify your proxy is running: `curl http://0.0.0.0:4000/health`
- Check that `ANTHROPIC_BASE_URL` is set correctly
- Ensure your `ANTHROPIC_AUTH_TOKEN` matches your LiteLLM master key

**Authentication errors:**
- Verify your environment variables are set: `echo $LITELLM_MASTER_KEY`
- Check that your Anthropic API key is valid and has sufficient credits: `echo $ANTHROPIC_API_KEY`
- For Responses API models, check that your OpenAI API key is valid and has sufficient credits

**Model not found:**
- Ensure the model name in Claude Code matches exactly with your `config.yaml`
- Check LiteLLM logs for detailed error messages

## Using Multiple Models

Expand your configuration to support multiple providers and models:

<Tabs>
<TabItem value="all-models" label="Anthropic + Responses API + Other Models" default>

```yaml
model_list:
  # Anthropic Claude models (Recommended)
  - model_name: claude-3-5-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: claude-3-5-haiku
    litellm_params:
      model: anthropic/claude-3-5-haiku-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: claude-3-opus
    litellm_params:
      model: anthropic/claude-3-opus-20240229
      api_key: os.environ/ANTHROPIC_API_KEY

  # Responses API models
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

  # Other standard models
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

litellm_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

Switch between models seamlessly:

```bash
# Use Anthropic Claude models for general tasks
claude --model claude-3-5-sonnet
claude --model claude-3-5-haiku
claude --model claude-3-opus

# Use Responses API models for advanced reasoning
claude --model o3-pro
claude --model codex-mini

# Use other models as needed
claude --model gpt-4o
```

</TabItem>
<TabItem value="anthropic-only" label="Anthropic Claude Only">

```yaml
model_list:
  # Anthropic Claude models
  - model_name: claude-3-5-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: claude-3-5-haiku
    litellm_params:
      model: anthropic/claude-3-5-haiku-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: claude-3-opus
    litellm_params:
      model: anthropic/claude-3-opus-20240229
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: claude-3-7-sonnet
    litellm_params:
      model: anthropic/claude-3-7-sonnet-20250219
      api_key: os.environ/ANTHROPIC_API_KEY

litellm_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

</TabItem>
</Tabs>

<Image img={require('../../img/release_notes/claude_code_demo.png')} style={{ width: '500px', height: 'auto' }} />