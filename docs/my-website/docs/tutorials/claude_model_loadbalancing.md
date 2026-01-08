import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Load Balancing Claude Models Across AWS Bedrock & Vertex AI

This tutorial shows how to load balance Claude models across multiple cloud providers (AWS Bedrock and Google Vertex AI) using LiteLLM proxy, and connect it to Claude Code using the unified API endpoint.

:::info Why Load Balance Across Providers?

- **Higher availability**: If one provider has an outage, traffic automatically routes to healthy providers
- **Better rate limits**: Spread requests across multiple provider quotas
- **Cost optimization**: Route to different providers based on pricing or availability
- **Regional compliance**: Use specific regions for data residency requirements

:::

## Architecture Overview

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────────────────┐
│ Claude Code │────▶│  LiteLLM Proxy  │────▶│ AWS Bedrock (Claude)    │
│             │     │  (Unified API)  │     │ Vertex AI (Claude)      │
└─────────────┘     └─────────────────┘     │ Anthropic API (Claude)  │
                                            └─────────────────────────┘
```

LiteLLM normalizes all requests to the OpenAI format, handling the different protocols between AWS Bedrock, Vertex AI, and Anthropic's native API automatically.

## Prerequisites

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) installed
- AWS credentials with Bedrock access ([setup guide](https://docs.aws.amazon.com/bedrock/latest/userguide/setting-up.html))
- Google Cloud credentials with Vertex AI access ([setup guide](https://cloud.google.com/vertex-ai/docs/start/cloud-environment))
- Python 3.8+

## Installation

```bash
pip install 'litellm[proxy]'
```

For AWS Bedrock, you also need boto3:

```bash
pip install boto3>=1.28.57
```

## Quick Start

### 1. Set Up Environment Variables

<Tabs>
<TabItem value="aws" label="AWS Bedrock">

```bash
# AWS Credentials
export AWS_ACCESS_KEY_ID="your-aws-access-key"
export AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
export AWS_REGION_NAME="us-east-1"  # or your preferred region
```

</TabItem>
<TabItem value="vertex" label="Google Vertex AI">

```bash
# Option 1: Use gcloud CLI (recommended)
gcloud auth application-default login

# Option 2: Use service account
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"

# Set your project details
export VERTEXAI_PROJECT="your-gcp-project-id"
export VERTEXAI_LOCATION="us-east5"  # or your preferred region
```

</TabItem>
</Tabs>

```bash
# LiteLLM Master Key (generate a secure key)
export LITELLM_MASTER_KEY="sk-1234567890"
```

### 2. Create config.yaml

Create a `config.yaml` file with your Claude model deployments:

```yaml
model_list:
  # AWS Bedrock Claude deployments
  - model_name: claude-sonnet  # 👈 model alias for load balancing
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: us-east-1

  - model_name: claude-sonnet
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: us-west-2

  # Google Vertex AI Claude deployments  
  - model_name: claude-sonnet
    litellm_params:
      model: vertex_ai/claude-3-5-sonnet-v2@20241022
      vertex_ai_project: os.environ/VERTEXAI_PROJECT
      vertex_ai_location: us-east5

  - model_name: claude-sonnet
    litellm_params:
      model: vertex_ai/claude-3-5-sonnet-v2@20241022
      vertex_ai_project: os.environ/VERTEXAI_PROJECT
      vertex_ai_location: europe-west1

router_settings:
  routing_strategy: simple-shuffle  # Randomly distributes requests
  num_retries: 3  # Retry on failures
  
litellm_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

:::tip Model Naming
All deployments with the same `model_name` (e.g., `claude-sonnet`) are grouped together for load balancing. LiteLLM will automatically distribute requests across these deployments.
:::

### 3. Start the Proxy

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

### 4. Verify Setup

Test that your proxy is working correctly:

```bash
curl -X POST http://0.0.0.0:4000/v1/messages \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-sonnet",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Hello! Which provider are you running on?"}]
  }'
```

### 5. Configure Claude Code

Configure Claude Code to use LiteLLM's unified endpoint:

```bash
export ANTHROPIC_BASE_URL="http://0.0.0.0:4000"
export ANTHROPIC_AUTH_TOKEN="$LITELLM_MASTER_KEY"
```

### 6. Use Claude Code

```bash
# Start Claude Code - it will automatically use your load-balanced Claude models
claude

# Or specify the model name from your config
claude --model claude-sonnet
```

## Advanced Configuration

### Weighted Load Balancing

Prioritize certain deployments by setting weights:

```yaml
model_list:
  # Primary: Bedrock US East (higher weight = more traffic)
  - model_name: claude-sonnet
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
      aws_region_name: us-east-1
      weight: 3  # 👈 Gets 3x more traffic

  # Secondary: Vertex AI 
  - model_name: claude-sonnet
    litellm_params:
      model: vertex_ai/claude-3-5-sonnet-v2@20241022
      vertex_ai_location: us-east5
      weight: 1
```

### Priority-Based Routing (Fallback)

Use `order` to set primary and fallback deployments:

```yaml
model_list:
  # Primary deployment (order: 1)
  - model_name: claude-sonnet
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
      aws_region_name: us-east-1
      order: 1  # 👈 Highest priority - always tried first

  # Fallback deployment (order: 2)
  - model_name: claude-sonnet
    litellm_params:
      model: vertex_ai/claude-3-5-sonnet-v2@20241022
      vertex_ai_location: us-east5
      order: 2  # 👈 Used when order=1 is unavailable
```

### Rate Limit Aware Routing

Configure RPM/TPM limits to enable intelligent load balancing:

```yaml
model_list:
  - model_name: claude-sonnet
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
      aws_region_name: us-east-1
      rpm: 1000  # Requests per minute limit
      tpm: 100000  # Tokens per minute limit

  - model_name: claude-sonnet
    litellm_params:
      model: vertex_ai/claude-3-5-sonnet-v2@20241022
      vertex_ai_location: us-east5
      rpm: 500
      tpm: 50000

router_settings:
  routing_strategy: simple-shuffle
  enable_pre_call_check: true  # 👈 Enables rate limit checking
```

### Multi-Model Setup

Configure multiple Claude model tiers:

```yaml
model_list:
  # Claude Sonnet (balanced)
  - model_name: claude-3-5-sonnet-20241022
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
      aws_region_name: us-east-1

  - model_name: claude-3-5-sonnet-20241022
    litellm_params:
      model: vertex_ai/claude-3-5-sonnet-v2@20241022
      vertex_ai_location: us-east5

  # Claude Haiku (fast & cheap)
  - model_name: claude-3-5-haiku-20241022
    litellm_params:
      model: bedrock/anthropic.claude-3-5-haiku-20241022-v1:0
      aws_region_name: us-east-1

  - model_name: claude-3-5-haiku-20241022
    litellm_params:
      model: vertex_ai/claude-3-5-haiku@20241022
      vertex_ai_location: us-east5

  # Claude Opus (most capable)
  - model_name: claude-3-opus-20240229
    litellm_params:
      model: bedrock/anthropic.claude-3-opus-20240229-v1:0
      aws_region_name: us-east-1

  - model_name: claude-3-opus-20240229
    litellm_params:
      model: vertex_ai/claude-3-opus@20240229
      vertex_ai_location: us-east5

litellm_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

Then in Claude Code:

```bash
# Use different models for different tasks
claude --model claude-3-5-sonnet-20241022  # Balanced
claude --model claude-3-5-haiku-20241022   # Fast responses
claude --model claude-3-opus-20240229      # Complex reasoning
```

### Adding Anthropic Direct API as Fallback

Include the native Anthropic API as an additional deployment option:

```yaml
model_list:
  # AWS Bedrock
  - model_name: claude-sonnet
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
      aws_region_name: us-east-1
      order: 1

  # Vertex AI  
  - model_name: claude-sonnet
    litellm_params:
      model: vertex_ai/claude-3-5-sonnet-v2@20241022
      vertex_ai_location: us-east5
      order: 1

  # Anthropic Direct API (fallback)
  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY
      order: 2  # 👈 Used as fallback

litellm_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

## Production Configuration

For production deployments, we recommend using Redis to share state across multiple LiteLLM proxy instances:

```yaml
model_list:
  - model_name: claude-sonnet
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
      aws_region_name: us-east-1
      rpm: 1000
      tpm: 100000

  - model_name: claude-sonnet
    litellm_params:
      model: vertex_ai/claude-3-5-sonnet-v2@20241022
      vertex_ai_location: us-east5
      rpm: 500
      tpm: 50000

router_settings:
  routing_strategy: simple-shuffle
  num_retries: 3
  timeout: 120  # seconds
  # Redis for distributed state
  redis_host: os.environ/REDIS_HOST
  redis_password: os.environ/REDIS_PASSWORD
  redis_port: os.environ/REDIS_PORT

litellm_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

## Monitoring & Observability

### Check Load Balancing is Working

LiteLLM returns the deployment used in response headers. You can verify load balancing by checking the `x-litellm-model-id` header:

```bash
curl -v -X POST http://0.0.0.0:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet",
    "messages": [{"role": "user", "content": "Hi"}]
  }' 2>&1 | grep -i "x-litellm"
```

### View Available Models

```bash
curl http://0.0.0.0:4000/v1/models \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY"
```

### Health Check

```bash
curl http://0.0.0.0:4000/health
```

## Troubleshooting

### Claude Code Not Connecting

1. Verify your proxy is running:
   ```bash
   curl http://0.0.0.0:4000/health
   ```

2. Check environment variables are set:
   ```bash
   echo $ANTHROPIC_BASE_URL
   echo $ANTHROPIC_AUTH_TOKEN
   ```

3. Ensure your `ANTHROPIC_AUTH_TOKEN` matches your LiteLLM master key

### AWS Bedrock Authentication Errors

1. Verify AWS credentials:
   ```bash
   aws sts get-caller-identity
   ```

2. Ensure your IAM user/role has Bedrock access:
   ```bash
   aws bedrock list-foundation-models --region us-east-1
   ```

3. Check that Claude models are enabled in your Bedrock console

### Vertex AI Authentication Errors

1. Verify GCP credentials:
   ```bash
   gcloud auth application-default print-access-token
   ```

2. Ensure the Claude model is enabled in your Vertex AI Model Garden

3. Check project and location settings:
   ```bash
   echo $VERTEXAI_PROJECT
   echo $VERTEXAI_LOCATION
   ```

### Model Not Found

- Ensure the `model_name` in Claude Code matches exactly with your `config.yaml`
- Check LiteLLM logs for detailed error messages:
  ```bash
  litellm --config /path/to/config.yaml --detailed_debug
  ```

### Requests Not Load Balancing

- Verify multiple deployments have the same `model_name`
- Check that all deployments are healthy by viewing the logs
- Ensure `routing_strategy` is set in `router_settings`

## Cost Tracking

LiteLLM automatically tracks costs across all providers. View spending:

```bash
curl http://0.0.0.0:4000/spend/logs \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY"
```

## Related Resources

- [Proxy Load Balancing Documentation](../proxy/load_balancing.md)
- [Router & Load Balancing](../routing.md)
- [AWS Bedrock Provider](../providers/bedrock.md)
- [Vertex AI Claude Models](../providers/vertex_partner.md)
- [Claude Code Basic Setup](./claude_responses_api.md)
