import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Use Claude Code with Non-Anthropic Models

This tutorial shows how to use Claude Code with non-Anthropic models like OpenAI, Gemini, and other LLM providers through LiteLLM proxy.

:::info 

LiteLLM automatically translates between different provider formats, allowing you to use any supported LLM provider with Claude Code while maintaining the Anthropic Messages API format.

:::

## Prerequisites

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) installed
- API keys for your chosen providers (OpenAI, Vertex AI, etc.)

## Installation

First, install LiteLLM with proxy support:

```bash
pip install 'litellm[proxy]'
```

## Configuration

### 1. Setup config.yaml

Create a configuration file with your preferred non-Anthropic models:

<Tabs>
<TabItem value="openai" label="OpenAI">

```yaml
model_list:
  # OpenAI GPT-4o
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
  
  # OpenAI GPT-4o-mini
  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY
```

Set your environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export LITELLM_MASTER_KEY="sk-1234567890"  # Generate a secure key
```

</TabItem>
<TabItem value="gemini" label="Google AI Studio">

```yaml
model_list:
  # Google Gemini
  - model_name: gemini-3.0-flash-exp
    litellm_params:
      model: gemini/gemini-3.0-flash-exp
      api_key: os.environ/GEMINI_API_KEY
```

Set your environment variables:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
export LITELLM_MASTER_KEY="sk-1234567890"  # Generate a secure key
```

</TabItem>
<TabItem value="vertex_ai" label="Vertex AI">

```yaml
model_list:
  # Google Gemini
  - model_name: vertex-gemini-3-flash-preview
    litellm_params:
      model: vertex_ai/gemini-3-flash-preview
      vertex_credentials: os.environ/VERTEX_FILE_PATH_ENV_VAR # os.environ["VERTEX_FILE_PATH_ENV_VAR"] = "/path/to/service_account.json" 
      vertex_project: "my-test-project"
      vertex_location: "us-east-1"

  # Anthropic Claude
  - model_name: anthropic-vertex
    litellm_params:
      model: vertex_ai/claude-3-sonnet@20240229
      vertex_ai_project: "my-test-project"
      vertex_ai_location: "us-east-1"
      vertex_credentials: os.environ/VERTEX_FILE_PATH_ENV_VAR # os.environ["VERTEX_FILE_PATH_ENV_VAR"] = "/path/to/service_account.json" 
```

Set your environment variables:

```bash
export VERTEX_FILE_PATH_ENV_VAR="/path/to/service_account.json"
export LITELLM_MASTER_KEY="sk-1234567890"  
```

</TabItem>
<TabItem value="multi" label="Azure OpenAI">

```yaml
model_list:
  # Azure OpenAI
  - model_name: azure-gpt-4
    litellm_params:
      model: azure/gpt-4
      api_key: os.environ/AZURE_API_KEY
      api_base: os.environ/AZURE_API_BASE
      api_version: "2024-02-01"
```

Set your environment variables:

```bash
export AZURE_API_KEY="your-azure-api-key"
export AZURE_API_BASE="https://your-resource.openai.azure.com"
export LITELLM_MASTER_KEY="sk-1234567890"
```

</TabItem>
</Tabs>

### 2. Start LiteLLM Proxy

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

### 3. Verify Setup

Test that your proxy is working correctly:

<Tabs>
<TabItem value="openai-test" label="OpenAI">

```bash
curl -X POST http://0.0.0.0:4000/v1/messages \
-H "Authorization: Bearer $LITELLM_MASTER_KEY" \
-H "Content-Type: application/json" \
-d '{
    "model": "gpt-4o",
    "max_tokens": 1000,
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
}'
```

</TabItem>
<TabItem value="gemini-test" label="Google AI Studio">

```bash
curl -X POST http://0.0.0.0:4000/v1/messages \
-H "Authorization: Bearer $LITELLM_MASTER_KEY" \
-H "Content-Type: application/json" \
-d '{
    "model": "gemini-3.0-flash-exp",
    "max_tokens": 1000,
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
}'
```

</TabItem>
<TabItem value="vertex-test" label="Vertex AI">

```bash
curl -X POST http://0.0.0.0:4000/v1/messages \
-H "Authorization: Bearer $LITELLM_MASTER_KEY" \
-H "Content-Type: application/json" \
-d '{
    "model": "gemini-3.0-flash-exp",
    "max_tokens": 1000,
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
}'
```

</TabItem>
<TabItem value="azure-test" label="Azure OpenAI">

```bash
curl -X POST http://0.0.0.0:4000/v1/messages \
-H "Authorization: Bearer $LITELLM_MASTER_KEY" \
-H "Content-Type: application/json" \
-d '{
    "model": "azure-gpt-4",
    "max_tokens": 1000,
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
}'
```

</TabItem>
</Tabs>

### 4. Configure Claude Code

Configure Claude Code to use your LiteLLM proxy:

```bash
export ANTHROPIC_BASE_URL="http://0.0.0.0:4000"
export ANTHROPIC_AUTH_TOKEN="$LITELLM_MASTER_KEY"
```

:::tip
The `LITELLM_MASTER_KEY` gives Claude Code access to all proxy models. You can also create virtual keys in the LiteLLM UI to limit access to specific models.
:::

### 5. Use Claude Code with Non-Anthropic Models

Start Claude Code and specify which model to use:

```bash
# Use OpenAI GPT-4o
claude --model gpt-4o

# Use OpenAI GPT-4o-mini for faster responses
claude --model gpt-4o-mini

# Use Google Gemini
claude --model gemini-3.0-flash-exp

# Use Vertex AI Gemini
claude --model vertex-gemini-3-flash-preview

# Use Vertex AI Anthropic Claude
claude --model anthropic-vertex

# Use Azure OpenAI
claude --model azure-gpt-4
```

## How It Works

LiteLLM acts as a unified interface that:

1. **Receives requests** from Claude Code in Anthropic Messages API format
2. **Translates** the request to the target provider's format (OpenAI, Gemini, etc.)
3. **Forwards** the request to the actual provider
4. **Translates** the response back to Anthropic Messages API format
5. **Returns** the response to Claude Code

This allows you to use Claude Code's interface with any LLM provider supported by LiteLLM.

## Advanced Features

### Load Balancing and Fallbacks

Configure multiple deployments with automatic fallback:

```yaml
model_list:
  - model_name: gpt-4o  # virtual model name
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
  
  - model_name: gpt-4o  # same virtual name
    litellm_params:
      model: azure/gpt-4o
      api_key: os.environ/AZURE_API_KEY
      api_base: os.environ/AZURE_API_BASE

router_settings:
  routing_strategy: simple-shuffle  # Load balance between deployments
  num_retries: 2
  timeout: 30
```

### Usage Tracking and Budgets

Track usage and set budgets through the LiteLLM UI:

```yaml
litellm_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: "postgresql://..."  # Enable database for tracking
  
general_settings:
  store_model_in_db: true
```

Start the proxy with the UI:

```bash
litellm --config /path/to/config.yaml --detailed_debug
```

Access the UI at `http://0.0.0.0:4000/ui` to:
- View usage analytics
- Set budget limits per user/key
- Monitor costs across different providers
- Create virtual keys with specific permissions


## Supported Providers

LiteLLM supports 100+ providers. Here are some popular ones for use with Claude Code:

- **OpenAI**: GPT-4o, GPT-4o-mini, o1, o3-mini
- **Google**: Gemini 2.0 Flash, Gemini 1.5 Pro/Flash
- **Azure OpenAI**: All OpenAI models via Azure
- **AWS Bedrock**: Llama, Mistral, and other models
- **Vertex AI**: Gemini, Claude, and other models on Google Cloud
- **Groq**: Fast inference for Llama and Mixtral
- **Together AI**: Llama, Mixtral, and other open source models
- **Deepseek**: Deepseek-chat, Deepseek-coder

[View full list of supported providers →](https://docs.litellm.ai/docs/providers)
