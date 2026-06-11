---
description: Set up LiteLLM from scratch - installation, configuration, and running the proxy server. Use when users want to get started with LiteLLM, configure providers, or troubleshoot setup issues.
---

# LiteLLM Setup Skill

Guide users through setting up LiteLLM, an AI Gateway that provides a unified interface to 100+ LLM providers.

## When to Use

- User wants to install LiteLLM
- User wants to configure LiteLLM with their API keys
- User wants to set up the LiteLLM proxy server
- User is troubleshooting LiteLLM configuration issues
- User wants to add new providers to their existing config

## Setup Flow

### 1. Check Prerequisites

First, verify the environment is ready:

```bash
# Check Python version (needs 3.10+, <3.14)
python3 --version

# Check if uv is installed (preferred package manager)
uv --version 2>/dev/null || echo "uv not installed - recommend: curl -LsSf https://astral.sh/uv/install.sh | sh"
```

### 2. Installation Options

**Option A: Install as a tool (recommended for running the proxy)**
```bash
uv tool install 'litellm[proxy]'
```

**Option B: Add to a project**
```bash
uv add litellm
# Or with proxy features:
uv add 'litellm[proxy]'
```

**Option C: Development install (for contributors)**
```bash
git clone https://github.com/BerriAI/litellm.git
cd litellm
make install-proxy-dev
```

### 3. Interactive Setup Wizard

The fastest way to configure LiteLLM:

```bash
litellm --setup
```

This interactive wizard will:
1. Let you select providers (OpenAI, Anthropic, Google Gemini, Azure, AWS Bedrock, Ollama)
2. Prompt for API keys and validate them
3. Configure port and master key
4. Generate a `litellm_config.yaml` file

### 4. Manual Configuration

If you prefer manual setup, create a `litellm_config.yaml` file:

```yaml
model_list:
  # OpenAI
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

  # Anthropic
  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-sonnet-4-6
      api_key: os.environ/ANTHROPIC_API_KEY

  # Google Gemini
  - model_name: gemini-flash
    litellm_params:
      model: gemini/gemini-2.0-flash
      api_key: os.environ/GEMINI_API_KEY

  # AWS Bedrock
  - model_name: bedrock-claude
    litellm_params:
      model: bedrock/anthropic.claude-haiku-4-5-20251001-v1:0
      aws_region_name: us-east-1

  # Ollama (local)
  - model_name: llama
    litellm_params:
      model: ollama/llama3.2
      api_base: http://localhost:11434

# General Settings
general_settings:
  master_key: sk-your-master-key  # Used to authenticate requests to the proxy
  # store_model_in_db: true  # Enable for database-backed config

# Optional: MCP Server Configuration
mcp_servers:
  fetch:
    transport: stdio
    command: uvx
    args: ["mcp-server-fetch"]
    description: "Fetch web content"
```

### 5. Environment Variables

Set your API keys as environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Google Gemini
export GEMINI_API_KEY="AIza..."

# Azure OpenAI
export AZURE_AI_API_KEY="your-azure-key"
export AZURE_API_BASE="https://<resource>.openai.azure.com/"

# AWS Bedrock
export AWS_ACCESS_KEY_ID="AKIA..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION_NAME="us-east-1"
```

### 6. Start the Proxy Server

```bash
# Using config file
litellm --config litellm_config.yaml --port 4000

# Quick start with a single model (no config needed)
litellm --model gpt-4o --port 4000
```

### 7. Test the Setup

```bash
# Health check
curl http://localhost:4000/health

# List available models
curl http://localhost:4000/v1/models -H "Authorization: Bearer sk-your-master-key"

# Test a completion
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-master-key" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Common Issues

### "No module named 'litellm'"
Install with: `uv tool install 'litellm[proxy]'` or `pip install 'litellm[proxy]'`

### "Invalid API key" errors
- Verify the environment variable is set: `echo $OPENAI_API_KEY`
- Check the key format matches the provider's format
- Ensure the key has the correct permissions

### Port already in use
```bash
# Find what's using the port
lsof -i :4000
# Use a different port
litellm --config litellm_config.yaml --port 8000
```

### Config file not found
- Use absolute path: `litellm --config /path/to/litellm_config.yaml`
- Or run from the directory containing the config

## Provider-Specific Notes

### Azure OpenAI
Requires deployment name in the model:
```yaml
- model_name: azure-gpt4
  litellm_params:
    model: azure/<deployment-name>
    api_key: os.environ/AZURE_AI_API_KEY
    api_base: os.environ/AZURE_API_BASE
    api_version: "2024-07-01-preview"
```

### AWS Bedrock
Uses IAM credentials - ensure your AWS credentials are configured:
```bash
aws configure
# Or set environment variables directly
```

### Ollama
Must have Ollama running locally:
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
# Pull a model
ollama pull llama3.2
# Ollama runs on http://localhost:11434 by default
```

## Next Steps

- **Dashboard**: Access the UI at `http://localhost:4000/ui` (requires `store_model_in_db: true`)
- **Documentation**: https://docs.litellm.ai
- **API Reference**: The proxy is OpenAI-compatible - use any OpenAI SDK with your proxy URL
