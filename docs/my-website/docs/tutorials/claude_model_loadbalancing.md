import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Load Balancing Claude Models Across AWS Bedrock & Vertex AI

Load balance Claude models across AWS Bedrock and Google Vertex AI using LiteLLM proxy with Claude Code.

## Quick Start

### 1. Install

```bash
pip install 'litellm[proxy]' boto3>=1.28.57
```

### 2. Set Environment Variables

```bash
# AWS
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"

# GCP (or use: gcloud auth application-default login)
export VERTEXAI_PROJECT="your-project"
export VERTEXAI_LOCATION="us-east5"

# LiteLLM
export LITELLM_MASTER_KEY="sk-1234"
```

### 3. Create config.yaml

```yaml
model_list:
  # Bedrock deployments
  - model_name: claude-sonnet
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
      aws_access_key_id: os.environ/AWS_ACCESS_KEY_ID
      aws_secret_access_key: os.environ/AWS_SECRET_ACCESS_KEY
      aws_region_name: us-east-1

  - model_name: claude-sonnet
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
      aws_region_name: us-west-2

  # Vertex AI deployments
  - model_name: claude-sonnet
    litellm_params:
      model: vertex_ai/claude-3-5-sonnet-v2@20241022
      vertex_ai_project: os.environ/VERTEXAI_PROJECT
      vertex_ai_location: us-east5

  - model_name: claude-sonnet
    litellm_params:
      model: vertex_ai/claude-3-5-sonnet-v2@20241022
      vertex_ai_location: europe-west1

router_settings:
  routing_strategy: simple-shuffle
  num_retries: 3

litellm_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

All deployments with the same `model_name` are load balanced automatically.

### 4. Start Proxy

```bash
litellm --config /path/to/config.yaml
```

### 5. Configure Claude Code

```bash
export ANTHROPIC_BASE_URL="http://0.0.0.0:4000"
export ANTHROPIC_AUTH_TOKEN="$LITELLM_MASTER_KEY"
claude
```

## Advanced Options

### Priority-Based Fallback

```yaml
- model_name: claude-sonnet
  litellm_params:
    model: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
    order: 1  # Primary

- model_name: claude-sonnet
  litellm_params:
    model: vertex_ai/claude-3-5-sonnet-v2@20241022
    order: 2  # Fallback
```

### Weighted Routing

```yaml
- model_name: claude-sonnet
  litellm_params:
    model: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
    weight: 3  # Gets 3x traffic

- model_name: claude-sonnet
  litellm_params:
    model: vertex_ai/claude-3-5-sonnet-v2@20241022
    weight: 1
```

## Related Docs

- [Load Balancing](../proxy/load_balancing.md)
- [Routing Strategies](../routing.md)
- [AWS Bedrock](../providers/bedrock.md)
- [Vertex AI Claude](../providers/vertex_partner.md)
- [Claude Code Setup](./claude_responses_api.md)
