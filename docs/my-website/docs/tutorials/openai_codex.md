import Image from '@theme/IdealImage';

# Using LiteLLM with OpenAI Codex

This tutorial walks you through setting up and using [OpenAI Codex](https://github.com/openai/codex) with LiteLLM Proxy. LiteLLM enables you to use various LLM models (including Gemini) through the Codex interface.

<Image img={require('../../img/litellm_codex.gif')} />

## Prerequisites

- LiteLLM Proxy running (see [Docker Quick Start Guide](../proxy/docker_quick_start.md) for setup details)
- Node.js and npm installed

## Step 1: Install OpenAI Codex

Install the OpenAI Codex CLI tool globally using npm:

```bash
npm i -g @openai/codex
```

## Step 2: Configure Codex to use LiteLLM Proxy

Set the required environment variables to point Codex to your LiteLLM Proxy:

```bash
# Point to your LiteLLM Proxy server
export OPENAI_BASE_URL=http://0.0.0.0:4000 

# Use your LiteLLM API key
export OPENAI_API_KEY="sk-1234"
```

## Step 3: LiteLLM Configuration

Ensure your LiteLLM Proxy is properly configured to route to your desired models. Here's the example configuration:

```yaml
model_list:
  - model_name: openai/*
    litellm_params:
      model: openai/*
  - model_name: anthropic/*
    litellm_params:
      model: anthropic/*
  - model_name: gemini/*
    litellm_params:
      model: gemini/*
litellm_settings:
  drop_params: true
```

This configuration enables routing to OpenAI, Anthropic, and Gemini models.

## Step 4: Run Codex with Gemini

With everything configured, you can now run Codex with Gemini:

```bash
codex --model gemini/gemini-2.0-flash --full-auto
```

The `--full-auto` flag allows Codex to automatically generate code without additional prompting.

## Troubleshooting

- If you encounter connection issues, ensure your LiteLLM Proxy is running and accessible at the specified URL
- Verify your LiteLLM API key is valid
- Check that your model routing configuration is correct

## Additional Resources

For more details on starting and configuring LiteLLM, refer to the [Docker Quick Start Guide](../proxy/docker_quick_start.md).
