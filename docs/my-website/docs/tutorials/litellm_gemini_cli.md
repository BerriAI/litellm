# Gemini CLI

This tutorial shows you how to integrate the Gemini CLI with LiteLLM Proxy, allowing you to route requests through LiteLLM's unified interface.


:::info 

This integration is supported from LiteLLM v1.73.3-nightly and above.

:::

<br />

<iframe width="840" height="500" src="https://www.loom.com/embed/d5dadd811ae64c70b29a16ecd558d4ba" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

## Benefits of using gemini-cli with LiteLLM

When you use gemini-cli with LiteLLM you get the following benefits:

**Developer Benefits:**
- Universal Model Access: Use any LiteLLM supported model (Anthropic, OpenAI, Vertex AI, Bedrock, etc.) through the gemini-cli interface.
- Higher Rate Limits & Reliability: Load balance across multiple models and providers to avoid hitting individual provider limits, with fallbacks to ensure you get responses even if one provider fails.

**Proxy Admin Benefits:**
- Centralized Management: Control access to all models through a single LiteLLM proxy instance without giving your developers API Keys to each provider.
- Budget Controls: Set spending limits and track costs across all gemini-cli usage.



## Prerequisites

Before you begin, ensure you have:
- Node.js and npm installed on your system
- A running LiteLLM Proxy instance
- A valid LiteLLM Proxy API key
- Git installed for cloning the repository

## Quick Start Guide

### Step 1: Install Gemini CLI

Clone the Gemini CLI repository and navigate to the project directory:

```bash
npm install -g @google/gemini-cli
```

### Step 2: Configure Gemini CLI for LiteLLM Proxy

Configure the Gemini CLI to point to your LiteLLM Proxy instance by setting the required environment variables:

```bash
export GOOGLE_GEMINI_BASE_URL="http://localhost:4000"
export GEMINI_API_KEY=sk-1234567890
```

**Note:** Replace the values with your actual LiteLLM Proxy configuration:
- `BASE_URL`: The URL where your LiteLLM Proxy is running
- `GEMINI_API_KEY`: Your LiteLLM Proxy API key

### Step 3: Build and Start Gemini CLI

Build the project and start the CLI:

```bash
gemini
```

### Step 4: Test the Integration

Once the CLI is running, you can send test requests. These requests will be automatically routed through LiteLLM Proxy to the configured Gemini model.

The CLI will now use LiteLLM Proxy as the backend, giving you access to LiteLLM's features like:
- Request/response logging
- Rate limiting
- Cost tracking
- Model routing and fallbacks


## Advanced

### Use Anthropic, OpenAI, Bedrock, etc. models on gemini-cli

In order to use non-gemini models on gemini-cli, you need to set a `model_group_alias` in the LiteLLM Proxy config. This tells LiteLLM that requests with model = `gemini-2.5-pro` should be routed to your desired model from any provider.

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs>
<TabItem value="anthropic" label="Anthropic">

Route `gemini-2.5-pro` requests to Claude Sonnet:

```yaml showLineNumbers title="proxy_config.yaml"
model_list:
  - model_name: claude-sonnet-4-20250514
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

router_settings:
  model_group_alias: {"gemini-2.5-pro": "claude-sonnet-4-20250514"}
```

</TabItem>
<TabItem value="openai" label="OpenAI">

Route `gemini-2.5-pro` requests to GPT-4o:

```yaml showLineNumbers title="proxy_config.yaml"
model_list:
  - model_name: gpt-4o-model
    litellm_params:
      model: gpt-4o
      api_key: os.environ/OPENAI_API_KEY

router_settings:
  model_group_alias: {"gemini-2.5-pro": "gpt-4o-model"}
```

</TabItem>
<TabItem value="bedrock" label="Bedrock">

Route `gemini-2.5-pro` requests to Claude on Bedrock:

```yaml showLineNumbers title="proxy_config.yaml"
model_list:
  - model_name: bedrock-claude
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: us-east-1

router_settings:
  model_group_alias: {"gemini-2.5-pro": "bedrock-claude"}
```

</TabItem>
<TabItem value="multi-provider" label="Multi-Provider Load Balancing">

All deployments with model_name=`anthropic-claude` will be load balanced. In this example we load balance between Anthropic and Bedrock.

```yaml showLineNumbers title="proxy_config.yaml"
model_list:
  - model_name: anthropic-claude
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY  
  - model_name: anthropic-claude
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: us-east-1

router_settings:
  model_group_alias: {"gemini-2.5-pro": "anthropic-claude"}
```

</TabItem>
</Tabs>

With this configuration, when you use `gemini-2.5-pro` in the CLI, LiteLLM will automatically route your requests to the configured provider(s) with load balancing and fallbacks.







## Troubleshooting

If you encounter issues:

1. **Connection errors**: Verify that your LiteLLM Proxy is running and accessible at the configured `GOOGLE_GEMINI_BASE_URL`
2. **Authentication errors**: Ensure your `GEMINI_API_KEY` is valid and has the necessary permissions
3. **Build failures**: Make sure all dependencies are installed with `npm install`

