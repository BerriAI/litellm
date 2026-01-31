# Using LiteLLM with OpenClaw

[OpenClaw](https://openclaw.ai) is a personal AI assistant that runs locally and connects to your messaging platforms (Slack, Discord, Telegram, etc.). LiteLLM can be used as the model provider backend, giving you access to 100+ LLMs through a unified API.

## Why use LiteLLM with OpenClaw?

- **Model flexibility**: Switch between OpenAI, Anthropic, Azure, Bedrock, and more without changing your OpenClaw config
- **Cost tracking**: Monitor spend across all your AI interactions
- **Enterprise support**: Use Azure OpenAI, AWS Bedrock, or Vertex AI through LiteLLM proxy
- **Load balancing**: Distribute requests across multiple API keys or providers

## Quick Start

### 1. Start LiteLLM Proxy

```bash
litellm --model gpt-4o
```

Or with a config file for multiple models:

```yaml
# litellm_config.yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: gpt-4o
      api_key: os.environ/OPENAI_API_KEY
  - model_name: claude-sonnet
    litellm_params:
      model: claude-sonnet-4-20250514
      api_key: os.environ/ANTHROPIC_API_KEY
```

```bash
litellm --config litellm_config.yaml
```

### 2. Configure OpenClaw

In your OpenClaw config (`~/.openclaw/config.yaml`):

```yaml
providers:
  litellm:
    baseUrl: http://localhost:4000/v1
    apiKey: sk-your-litellm-key  # Optional if no auth configured

models:
  default: litellm/gpt-4o
  # Or use Claude through LiteLLM:
  # default: litellm/claude-sonnet
```

### 3. Start OpenClaw

```bash
openclaw gateway start
```

OpenClaw will now route all model requests through your LiteLLM proxy.

## Using LiteLLM Cloud (Hosted)

If you're using [LiteLLM Cloud](https://litellm.ai), configure OpenClaw with your cloud endpoint:

```yaml
providers:
  litellm:
    baseUrl: https://api.litellm.ai/v1
    apiKey: sk-your-litellm-cloud-key

models:
  default: litellm/gpt-4o
```

## Advanced Configuration

### Multiple Model Routing

Configure different models for different tasks:

```yaml
providers:
  litellm:
    baseUrl: http://localhost:4000/v1

models:
  default: litellm/gpt-4o
  reasoning: litellm/o1-preview
  fast: litellm/gpt-4o-mini
  vision: litellm/gpt-4o
```

### Using Azure OpenAI through LiteLLM

LiteLLM config:
```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: azure/gpt-4o-deployment
      api_base: https://your-resource.openai.azure.com
      api_key: os.environ/AZURE_API_KEY
      api_version: "2024-02-15-preview"
```

OpenClaw sees this as `litellm/gpt-4o` - no Azure-specific config needed.

### Using AWS Bedrock through LiteLLM

LiteLLM config:
```yaml
model_list:
  - model_name: claude-sonnet
    litellm_params:
      model: bedrock/anthropic.claude-sonnet-4-20250514-v1:0
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: us-east-1
```

OpenClaw config:
```yaml
models:
  default: litellm/claude-sonnet
```

## Tool Calling & Function Support

LiteLLM fully supports tool/function calling, which OpenClaw uses extensively. All tools defined in your OpenClaw skills will work through LiteLLM.

Monitor tool calls in the LiteLLM dashboard:
- See which tools were called per request
- Track success/failure rates
- Debug tool execution issues

## Cost Tracking

LiteLLM tracks costs for all requests. View spend in the LiteLLM UI or via API:

```bash
curl http://localhost:4000/spend/logs
```

## Troubleshooting

### Connection refused
Ensure LiteLLM proxy is running and the `baseUrl` is correct.

### Model not found
Check that the model name in OpenClaw matches a `model_name` in your LiteLLM config.

### Authentication errors
Verify your API keys are set correctly in the LiteLLM config.

## Resources

- [LiteLLM Documentation](https://docs.litellm.ai)
- [OpenClaw Documentation](https://docs.openclaw.ai)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [OpenClaw GitHub](https://github.com/openclaw/openclaw)
