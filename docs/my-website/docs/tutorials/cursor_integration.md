---
sidebar_label: "Cursor IDE"
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Cursor IDE Integration with LiteLLM

This tutorial shows you how to integrate Cursor IDE with LiteLLM Proxy, allowing you to use any LiteLLM-supported model through Cursor's interface with BYOK (Bring Your Own Key) and custom base URL.

## Benefits of using Cursor with LiteLLM

When you use Cursor IDE with LiteLLM you get the following benefits:

**Developer Benefits:**
- Universal Model Access: Use any LiteLLM supported model (Anthropic, OpenAI, Vertex AI, Bedrock, etc.) through the Cursor IDE interface.
- Higher Rate Limits & Reliability: Load balance across multiple models and providers to avoid hitting individual provider limits, with fallbacks to ensure you get responses even if one provider fails.
- Streaming Support: Full streaming support with proper response transformation for Cursor's expected format.

**Proxy Admin Benefits:**
- Centralized Management: Control access to all models through a single LiteLLM proxy instance without giving your developers API Keys to each provider.
- Budget Controls: Set spending limits and track costs across all Cursor usage.
- Request Logging: Track all requests made through Cursor for debugging and monitoring.

## Prerequisites

Before you begin, ensure you have:
- Cursor IDE installed
- A running LiteLLM Proxy instance with **HTTPS enabled** (HTTP is not supported)
- A valid LiteLLM Proxy API key
- An HTTPS domain for your LiteLLM Proxy (required by Cursor)

## Quick Start Guide

### Step 1: Install LiteLLM

Install LiteLLM with proxy support:

```bash
pip install litellm[proxy]
```

### Step 2: Configure LiteLLM Proxy

Create a `config.yaml` file with your model configurations:

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: gpt-4o
      api_key: os.environ/OPENAI_API_KEY
  
  - model_name: claude-3-5-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

general_settings:
  master_key: sk-1234567890 # Change this to a secure key
```

### Step 3: Start LiteLLM Proxy

Start the proxy server with HTTPS enabled:

```bash
litellm --config config.yaml --port 4000
```

:::warning HTTPS Required

**Important**: Cursor IDE requires HTTPS connections. HTTP (`http://`) will not work. You must:
- Deploy your LiteLLM Proxy with HTTPS enabled
- Use a valid SSL certificate
- Access the proxy via an HTTPS domain (e.g., `https://your-proxy-domain.com`)

For local development, you'll need to set up HTTPS (e.g., using a reverse proxy like nginx with SSL, or deploying to a cloud service with HTTPS).

:::

### Step 4: Configure Cursor IDE

Configure Cursor IDE to use your LiteLLM proxy with the `/cursor/chat/completions` endpoint:

1. Open Cursor IDE
2. Go to **Settings** → **Features** → **AI**
3. Enable **"Use Custom API"** or **"Bring Your Own Key"**
4. Set the following:
   - **Base URL**: `https://your-proxy-domain.com/cursor` (⚠️ **Important**: Must use HTTPS and include `/cursor`)
   - **API Key**: Your LiteLLM Proxy API key (e.g., `sk-1234567890`)

:::warning HTTPS Required

Cursor IDE **requires HTTPS** connections. HTTP (`http://`) will not work. You must:
- Use an HTTPS URL for your base URL (e.g., `https://your-proxy-domain.com/cursor`)
- Ensure your LiteLLM Proxy is accessible via HTTPS
- Have a valid SSL certificate configured

:::

**Example Configuration:**

```
Base URL: https://your-proxy-domain.com/cursor
API Key: sk-1234567890
```

Replace `your-proxy-domain.com` with your actual HTTPS domain where LiteLLM Proxy is running.

:::info Why `/cursor` in the base URL?

Cursor automatically appends `/chat/completions` to the base URL you provide. By setting the base URL to `https://your-proxy-domain.com/cursor`, Cursor will send requests to `/cursor/chat/completions`, which is the special endpoint that handles Cursor's Responses API input format and transforms it to Chat Completions output format.

If you set the base URL to just `https://your-proxy-domain.com`, Cursor would send requests to `/chat/completions`, which won't work correctly with Cursor's request format.


:::

### Step 5: Test the Integration

1. Restart Cursor IDE to apply the settings
2. Open a code file and try using Cursor's AI features (completions, chat, etc.)
3. Your requests will now be routed through LiteLLM Proxy

You can verify it's working by:
- Checking the LiteLLM Proxy logs for incoming requests
- Using Cursor's chat feature and seeing responses stream correctly
- Checking your LiteLLM dashboard for request logs and cost tracking

## How It Works

The `/cursor/chat/completions` endpoint is specifically designed to handle Cursor's unique request format:

1. **Input**: Cursor sends requests in OpenAI Responses API format (with `input` field)
2. **Processing**: LiteLLM processes the request through its internal `/responses` flow
3. **Output**: The response is transformed to OpenAI Chat Completions format (with `choices` field) that Cursor expects

This transformation happens automatically for both streaming and non-streaming responses.

## Advanced Configuration

### Using Different Models

You can configure Cursor to use different models by updating your `config.yaml`:

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: gpt-4o
      api_key: os.environ/OPENAI_API_KEY
  
  - model_name: claude-3-5-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY
  
  - model_name: gemini-pro
    litellm_params:
      model: gemini/gemini-1.5-pro
      api_key: os.environ/GEMINI_API_KEY
```

Then in Cursor, you can specify which model to use in your requests.

### Rate Limiting and Budgets

Set up rate limits and budgets in your `config.yaml`:

```yaml showLineNumbers title="config.yaml"
general_settings:
  master_key: sk-1234567890

litellm_settings:
  # Set max budget per user
  max_budget: 100.0
  
  # Set rate limits
  rate_limit: 100  # requests per minute
```

### Request Logging

All requests from Cursor will be logged by LiteLLM Proxy. You can:
- View logs in the LiteLLM Admin UI
- Export logs to your preferred logging service
- Track costs per user/team

## Troubleshooting

### Cursor shows no output

- **Check base URL**: Ensure it uses HTTPS and includes `/cursor` (e.g., `https://your-proxy-domain.com/cursor`, not `http://` or without `/cursor`)
- **Verify HTTPS**: Cursor requires HTTPS - HTTP connections will not work
- **Check API key**: Verify your LiteLLM Proxy API key is correct
- **Check proxy logs**: Look for errors in the LiteLLM Proxy logs

### Requests failing

- **Verify HTTPS is enabled**: Cursor requires HTTPS connections. Ensure your LiteLLM Proxy is accessible via HTTPS with a valid SSL certificate
- **Verify proxy is running**: Check that LiteLLM Proxy is accessible at your HTTPS base URL
- **Check SSL certificate**: Ensure your SSL certificate is valid and not expired
- **Check model configuration**: Ensure the model you're trying to use is configured in `config.yaml`
- **Check API keys**: Verify provider API keys are set correctly in environment variables

### HTTP not working

If you're trying to use HTTP (`http://`) and it's not working:
- **This is expected**: Cursor IDE requires HTTPS connections
- **Solution**: Deploy your LiteLLM Proxy with HTTPS enabled (use a reverse proxy like nginx, or deploy to a cloud service that provides HTTPS)

### Streaming not working

The `/cursor/chat/completions` endpoint automatically handles streaming. If streaming isn't working:
- Check that your model supports streaming
- Verify the proxy logs for any transformation errors
- Ensure Cursor IDE is up to date

## Related Documentation

- [Cursor Endpoint Documentation](/docs/proxy/cursor) - Detailed endpoint documentation
- [LiteLLM Proxy Setup](/docs/proxy/quick_start) - General proxy setup guide
- [Model Configuration](/docs/proxy/configs) - How to configure models

