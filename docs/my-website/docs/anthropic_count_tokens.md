import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# /v1/messages/count_tokens

## Overview

Anthropic-compatible token counting endpoint. Count tokens for messages before sending them to the model.

| Feature | Supported | Notes |
|---------|-----------|-------|
| Cost Tracking | ❌ | Token counting only, no cost incurred |
| Logging | ✅ | Works across all integrations |
| End-user Tracking | ✅ | |
| Supported Providers | Anthropic, Vertex AI (Claude), Bedrock (Claude), Gemini, Vertex AI | Auto-routes to provider-specific token counting APIs |

## Quick Start

### 1. Start LiteLLM Proxy

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

### 2. Count Tokens

<Tabs>
<TabItem value="curl" label="curl">

```bash
curl -X POST "http://localhost:4000/v1/messages/count_tokens" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```

</TabItem>
<TabItem value="python" label="Python (httpx)">

```python
import httpx

response = httpx.post(
    "http://localhost:4000/v1/messages/count_tokens",
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-1234"
    },
    json={
        "model": "claude-3-5-sonnet-20241022",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ]
    }
)

print(response.json())
# {"input_tokens": 14}
```

</TabItem>
</Tabs>

**Expected Response:**

```json
{
  "input_tokens": 14
}
```

## LiteLLM Proxy Configuration

Add models to your `config.yaml`:

```yaml
model_list:
  - model_name: claude-3-5-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: claude-vertex
    litellm_params:
      model: vertex_ai/claude-3-5-sonnet-v2@20241022
      vertex_project: my-project
      vertex_location: us-east5
      vertex_count_tokens_location: us-east5 # Optional: Override location for token counting (count_tokens not available on global location)

  - model_name: claude-bedrock
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
      aws_region_name: us-west-2
```

## Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | ✅ | The model to use for token counting |
| `messages` | array | ✅ | Array of messages in Anthropic format |

### Messages Format

```json
{
  "messages": [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"}
  ]
}
```

## Response Format

```json
{
  "input_tokens": <number>
}
```

| Field | Type | Description |
|-------|------|-------------|
| `input_tokens` | integer | Number of tokens in the input messages |

## Supported Providers

The `/v1/messages/count_tokens` endpoint automatically routes to the appropriate provider-specific token counting API:

| Provider | Token Counting Method |
|----------|----------------------|
| Anthropic | [Anthropic Token Counting API](https://docs.anthropic.com/en/docs/build-with-claude/token-counting) |
| Vertex AI (Claude) | Vertex AI Partner Models Token Counter |
| Bedrock (Claude) | AWS Bedrock CountTokens API |
| Gemini | Google AI Studio countTokens API |
| Vertex AI (Gemini) | Vertex AI countTokens API |

## Examples

### Count Tokens with System Message

```bash
curl -X POST "http://localhost:4000/v1/messages/count_tokens" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
      {"role": "user", "content": "You are a helpful assistant. Please help me write a haiku about programming."}
    ]
  }'
```

### Count Tokens for Multi-turn Conversation

```bash
curl -X POST "http://localhost:4000/v1/messages/count_tokens" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"},
      {"role": "assistant", "content": "The capital of France is Paris."},
      {"role": "user", "content": "What is its population?"}
    ]
  }'
```

### Using with Vertex AI Claude

```bash
curl -X POST "http://localhost:4000/v1/messages/count_tokens" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "claude-vertex",
    "messages": [
      {"role": "user", "content": "Hello, world!"}
    ]
  }'
```

### Using with Bedrock Claude

```bash
curl -X POST "http://localhost:4000/v1/messages/count_tokens" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "claude-bedrock",
    "messages": [
      {"role": "user", "content": "Hello, world!"}
    ]
  }'
```

## Comparison with Anthropic Passthrough

LiteLLM provides two ways to count tokens:

| Endpoint | Description | Use Case |
|----------|-------------|----------|
| `/v1/messages/count_tokens` | LiteLLM's Anthropic-compatible endpoint | Works with all supported providers (Anthropic, Vertex AI, Bedrock, etc.) |
| `/anthropic/v1/messages/count_tokens` | [Pass-through to Anthropic API](./pass_through/anthropic_completion.md#example-2-token-counting-api) | Direct Anthropic API access with native headers |

### Pass-through Example

For direct Anthropic API access with full native headers:

```bash
curl --request POST \
    --url http://0.0.0.0:4000/anthropic/v1/messages/count_tokens \
    --header "x-api-key: $LITELLM_API_KEY" \
    --header "anthropic-version: 2023-06-01" \
    --header "anthropic-beta: token-counting-2024-11-01" \
    --header "content-type: application/json" \
    --data '{
        "model": "claude-3-5-sonnet-20241022",
        "messages": [
            {"role": "user", "content": "Hello, world"}
        ]
    }'
```
