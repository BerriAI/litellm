# Use LiteLLM with Gemini CLI

This tutorial shows you how to integrate the Gemini CLI with LiteLLM Proxy, allowing you to route requests through LiteLLM's unified interface.


:::info 

This integration is supported from LiteLLMv1.73.3-nightly and above.

:::

<br />

<iframe width="840" height="500" src="https://www.loom.com/embed/d5dadd811ae64c70b29a16ecd558d4ba" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>


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

## Troubleshooting

If you encounter issues:

1. **Connection errors**: Verify that your LiteLLM Proxy is running and accessible at the configured `GOOGLE_GEMINI_BASE_URL`
2. **Authentication errors**: Ensure your `GEMINI_API_KEY` is valid and has the necessary permissions
3. **Build failures**: Make sure all dependencies are installed with `npm install`

