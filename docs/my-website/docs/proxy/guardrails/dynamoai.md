import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# DynamoAI Guardrails

LiteLLM supports DynamoAI guardrails for content moderation and policy enforcement on LLM inputs and outputs.

## Quick Start

### 1. Define Guardrails on your LiteLLM config.yaml

Define your guardrails under the `guardrails` section:

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: "dynamoai-guard"
    litellm_params:
      guardrail: dynamoai
      mode: "pre_call"
      api_key: os.environ/DYNAMOAI_API_KEY
```

#### Supported values for `mode`

- `pre_call` - Run **before** LLM call, on **input**
- `post_call` - Run **after** LLM call, on **output**
- `during_call` - Run **during** LLM call, on **input**. Same as `pre_call` but runs in parallel as LLM call

### 2. Set Environment Variables

```bash
export DYNAMOAI_API_KEY="your-api-key"
# Optional: Set policy IDs via environment variable (comma-separated)
export DYNAMOAI_POLICY_IDS="policy-id-1,policy-id-2,policy-id-3"
```

### 3. Start LiteLLM Gateway

```shell
litellm --config config.yaml --detailed_debug
```

### 4. Test Request

**[Langchain, OpenAI SDK Usage Examples](../proxy/user_keys#request-format)**

<Tabs>
<TabItem label="Successful Call" value="allowed">

```shell showLineNumbers title="Successful Request"
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "guardrails": ["dynamoai-guard"]
  }'
```

**Response: HTTP 200 Success**

Content passes all policy checks and is allowed through.

</TabItem>

<TabItem label="Blocked Call" value="not-allowed">

```shell showLineNumbers title="Blocked Request"
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "Content that violates policy"}
    ],
    "guardrails": ["dynamoai-guard"]
  }'
```

**Expected Response on Block: HTTP 400 Error**

```json showLineNumbers
{
  "error": {
    "message": "Guardrail failed: 1 violation(s) detected\n\n- POLICY NAME:\n  Action: BLOCK\n  Method: TOXICITY\n  Description: Policy description\n  Policy ID: policy-id-123",
    "type": "None",
    "param": "None",
    "code": "400"
  }
}
```

</TabItem>
</Tabs>

## Advanced Configuration

### Specify Policy IDs

Configure specific DynamoAI policies to apply:

```yaml showLineNumbers title="config.yaml"
guardrails:
  - guardrail_name: "dynamoai-policies"
    litellm_params:
      guardrail: dynamoai
      mode: "pre_call"
      api_key: os.environ/DYNAMOAI_API_KEY
      policy_ids:
        - "policy-id-1"
        - "policy-id-2"
        - "policy-id-3"
```

### Custom API Base

Specify a custom DynamoAI API endpoint:

```yaml showLineNumbers title="config.yaml"
guardrails:
  - guardrail_name: "dynamoai-custom"
    litellm_params:
      guardrail: dynamoai
      mode: "pre_call"
      api_key: os.environ/DYNAMOAI_API_KEY
      api_base: "https://custom.dynamo.ai"
```

### Model ID for Tracking

Add a model ID for tracking and logging purposes:

```yaml showLineNumbers title="config.yaml"
guardrails:
  - guardrail_name: "dynamoai-tracked"
    litellm_params:
      guardrail: dynamoai
      mode: "pre_call"
      api_key: os.environ/DYNAMOAI_API_KEY
      model_id: "gpt-4-production"
```

### Input and Output Guardrails

Configure separate guardrails for input and output:

```yaml showLineNumbers title="config.yaml"
guardrails:
  # Input guardrail
  - guardrail_name: "dynamoai-input"
    litellm_params:
      guardrail: dynamoai
      mode: "pre_call"
      api_key: os.environ/DYNAMOAI_API_KEY

  # Output guardrail
  - guardrail_name: "dynamoai-output"
    litellm_params:
      guardrail: dynamoai
      mode: "post_call"
      api_key: os.environ/DYNAMOAI_API_KEY
```

## Configuration Options

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `api_key` | string | DynamoAI API key (required) | `DYNAMOAI_API_KEY` env var |
| `api_base` | string | DynamoAI API base URL | `https://api.dynamo.ai` |
| `policy_ids` | array | List of DynamoAI policy IDs to apply (optional) | `DYNAMOAI_POLICY_IDS` env var (comma-separated) |
| `model_id` | string | Model ID for tracking/logging | `DYNAMOAI_MODEL_ID` env var |
| `mode` | string | When to run: `pre_call`, `post_call`, or `during_call` | Required |

## Observability

DynamoAI guardrail logs include:

- **guardrail_status**: `success`, `guardrail_intervened`, or `guardrail_failed_to_respond`
- **guardrail_provider**: `dynamoai`
- **guardrail_json_response**: Full API response with policy details
- **duration**: Time taken for guardrail check
- **start_time** and **end_time**: Timestamps

These logs are available through your configured LiteLLM logging callbacks.

## Error Handling

The guardrail handles errors gracefully:

- **API Failures**: Logs error and raises exception with status `guardrail_failed_to_respond`
- **Policy Violations**: Raises `ValueError` with detailed violation information
- **Invalid Configuration**: Raises `ValueError` on initialization if API key is missing

## Current Limitations

- Only the `BLOCK` action is currently supported
- `WARN`, `REDACT`, and `SANITIZE` actions are treated as success (pass through)

## Support

For more information about DynamoAI:
- Website: [https://dynamo.ai](https://dynamo.ai)
- Documentation: Contact DynamoAI for API documentation

