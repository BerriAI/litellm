---
sidebar_label: "Use LiteLLM with Claude Code"
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Use LiteLLM with Claude Code

This tutorial shows you how to configure Claude Code to use LiteLLM as an LLM gateway, allowing you to access multiple model providers through Claude Code's interface.

:::info 

This tutorial is based on [Anthropic's official LiteLLM configuration documentation](https://docs.anthropic.com/en/docs/claude-code/llm-gateway#litellm-configuration). This integration allows you to use any LiteLLM supported model through Claude Code.

:::

## Benefits of using Claude Code with LiteLLM Gateway

When you use Claude Code with LiteLLM Gateway you get the following benefits:

**Developer Benefits:**
- Multi-Model Access: Use OpenAI, Google, Azure, Bedrock, and 100+ other providers through Claude Code's interface.
- Cost Management: Track spending and set budgets across all Claude Code usage.
- Fallback & Reliability: Configure automatic fallbacks if one provider is down.

**Proxy Admin Benefits:**
- Centralized Control: Manage access to multiple models through a single LiteLLM Gateway.
- Authentication & Security: Implement unified auth, rate limiting, and access controls.
- Observability: Monitor usage, costs, and performance across all providers.

## Prerequisites

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) installed
- LiteLLM Proxy Server
- API keys for your chosen providers (OpenAI, Anthropic, etc.)

## Quick Start Guide

### Step 1: Install LiteLLM

```bash
pip install 'litellm[proxy]'
```

### Step 2: Create Configuration File

Create a `config.yaml` file:

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

  - model_name: claude-3-5-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: gemini-pro
    litellm_params:
      model: vertex_ai/gemini-1.5-pro
      vertex_project: your-project-id
      vertex_location: us-central1

litellm_settings:
  master_key: sk-1234567890 # Replace with your secure key
```

### Step 3: Start LiteLLM Proxy

```bash
litellm --config config.yaml

# RUNNING on http://0.0.0.0:4000
```

### Step 4: Configure Claude Code

Set up Claude Code to use LiteLLM as the gateway:

```bash
export ANTHROPIC_BASE_URL="http://localhost:4000"
export ANTHROPIC_API_KEY="sk-1234567890" # Your LiteLLM master key
```

### Step 5: Use Claude Code

Start Claude Code with any configured model:

```bash
# Use OpenAI GPT-4o through LiteLLM
claude --model gpt-4o

# Use Anthropic Claude through LiteLLM  
claude --model claude-3-5-sonnet

# Use Google Gemini through LiteLLM
claude --model gemini-pro
```

## Advanced: Multiple Provider Configuration

<Tabs>
<TabItem value="aws-bedrock" label="AWS Bedrock">

```yaml
model_list:
  - model_name: claude-3-bedrock
    litellm_params:
      model: bedrock/anthropic.claude-3-sonnet-20240229-v1:0
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: us-east-1

  - model_name: titan-express
    litellm_params:
      model: bedrock/amazon.titan-text-express-v1
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: us-east-1
```

</TabItem>
<TabItem value="azure-openai" label="Azure OpenAI">

```yaml
model_list:
  - model_name: gpt-4-azure
    litellm_params:
      model: azure/gpt-4
      api_base: https://your-endpoint.openai.azure.com/
      api_key: os.environ/AZURE_API_KEY
      api_version: "2024-02-15-preview"

  - model_name: gpt-35-turbo-azure
    litellm_params:
      model: azure/gpt-35-turbo
      api_base: https://your-endpoint.openai.azure.com/
      api_key: os.environ/AZURE_API_KEY
      api_version: "2024-02-15-preview"
```

</TabItem>
<TabItem value="load-balancing" label="Load Balancing">

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY_1

  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY_2

router_settings:
  routing_strategy: "least-busy" # Load balance between identical models
  num_retries: 3
  timeout: 30
```

</TabItem>
</Tabs>

## Troubleshooting

If you encounter issues:

1. **Claude Code not connecting**: Verify `ANTHROPIC_BASE_URL` points to your LiteLLM proxy and the proxy is running
2. **Authentication errors**: Ensure your LiteLLM master key is correctly set in `ANTHROPIC_API_KEY`
3. **Model not found**: Check that the model name in Claude Code matches exactly with your `config.yaml` model_name

## Credits

This tutorial is based on [Anthropic's official documentation](https://docs.anthropic.com/en/docs/claude-code/llm-gateway#litellm-configuration) for configuring LiteLLM with Claude Code. 