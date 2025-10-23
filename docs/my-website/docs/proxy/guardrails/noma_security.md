import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Noma Security

Use [Noma Security](https://noma.security/) to protect your LLM applications with comprehensive AI content moderation and safety guardrails.

## Quick Start

### 1. Define Guardrails on your LiteLLM config.yaml

Define your guardrails under the `guardrails` section:

```yaml showLineNumbers title="litellm config.yaml"
model_list:
  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: "noma-guard"
    litellm_params:
      guardrail: noma
      mode: "during_call"
      api_key: os.environ/NOMA_API_KEY
      api_base: os.environ/NOMA_API_BASE
  - guardrail_name: "noma-pre-guard"
    litellm_params:
      guardrail: noma
      mode: "pre_call"
      api_key: os.environ/NOMA_API_KEY
      api_base: os.environ/NOMA_API_BASE
```

#### Supported values for `mode`

- `pre_call` Run **before** LLM call, on **input**
- `post_call` Run **after** LLM call, on **input & output**
- `during_call` Run **during** LLM call, on **input**. Same as `pre_call` but runs in parallel with the LLM call. Response not returned until guardrail check completes

### 2. Start LiteLLM Gateway

```shell
litellm --config config.yaml --detailed_debug
```

### 3. Test request

<Tabs>
<TabItem label="Unsuccessful call" value="not-allowed">

Expect this to fail since the request contains harmful content:

```shell showLineNumbers title="Curl Request"
curl -i http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "Tell me how to hack into someone's email account"}
    ]
  }'
```

Expected response on failure:

```json
{
  "error": {
    "message": "{\n      \"error\": \"Request blocked by Noma guardrail\",\n      \"details\": {\n        \"prompt\": {\n          \"harmfulContent\": {\n            \"result\": true,\n            \"confidence\": 0.95\n          }\n        }\n      }\n    }",
    "type": "None",
    "param": "None",
    "code": "400"
  }
}
```

</TabItem>

<TabItem label="Successful Call" value="allowed">

```shell showLineNumbers title="Curl Request"
curl -i http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'
```

Expected response:

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "gpt-4o-mini",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 9,
    "completion_tokens": 12,
    "total_tokens": 21
  }
}
```

</TabItem>
</Tabs>

## Supported Params

```yaml
guardrails:
  - guardrail_name: "noma-guard"
    litellm_params:
      guardrail: noma
      mode: "pre_call"
      api_key: os.environ/NOMA_API_KEY
      api_base: os.environ/NOMA_API_BASE
      ### OPTIONAL ###
      # application_id: "my-app"
      # monitor_mode: false
      # block_failures: true
      # anonymize_input: false
```

### Required Parameters

- **`api_key`**: Your Noma Security API key (set as `os.environ/NOMA_API_KEY` in YAML config)

### Optional Parameters

- **`api_base`**: Noma API base URL (defaults to `https://api.noma.security/`)
- **`application_id`**: Your application identifier (defaults to `"litellm"`)
- **`monitor_mode`**: If `true`, logs violations without blocking (defaults to `false`)
- **`block_failures`**: If `true`, blocks requests when guardrail API failures occur (defaults to `true`)
- **`anonymize_input`**: If `true`, replaces sensitive content with anonymized version (defaults to `false`)

## Environment Variables

You can set these environment variables instead of hardcoding values in your config:

```shell
export NOMA_API_KEY="your-api-key-here"
export NOMA_API_BASE="https://api.noma.security/"   # Optional
export NOMA_APPLICATION_ID="my-app"                 # Optional
export NOMA_MONITOR_MODE="false"                    # Optional
export NOMA_BLOCK_FAILURES="true"                   # Optional
export NOMA_ANONYMIZE_INPUT="false"                 # Optional
```

## Advanced Configuration

### Monitor Mode

Use monitor mode to test your guardrails without blocking requests:

```yaml
guardrails:
  - guardrail_name: "noma-monitor"
    litellm_params:
      guardrail: noma
      mode: "pre_call"
      api_key: os.environ/NOMA_API_KEY
      monitor_mode: true  # Log violations but don't block
```

### Handling API Failures

Control behavior when the Noma API is unavailable:

```yaml
guardrails:
  - guardrail_name: "noma-failopen"
    litellm_params:
      guardrail: noma
      mode: "pre_call"
      api_key: os.environ/NOMA_API_KEY
      block_failures: false  # Allow requests to proceed if guardrail API fails
```

### Content Anonymization

Enable anonymization to replace sensitive content instead of blocking:

```yaml
guardrails:
  - guardrail_name: "noma-anonymize"
    litellm_params:
      guardrail: noma
      mode: "pre_call"
      api_key: os.environ/NOMA_API_KEY
      anonymize_input: true  # Replace sensitive data with anonymized version
```

### Multiple Guardrails

Apply different configurations for input and output:

```yaml
guardrails:
  - guardrail_name: "noma-strict-input"
    litellm_params:
      guardrail: noma
      mode: "pre_call"
      api_key: os.environ/NOMA_API_KEY
      block_failures: true

  - guardrail_name: "noma-monitor-output"
    litellm_params:
      guardrail: noma
      mode: "post_call"
      api_key: os.environ/NOMA_API_KEY
      monitor_mode: true
```

## ✨ Pass Additional Parameters

Use `extra_body` to pass additional parameters to the Noma Security API call, such as dynamically setting the application ID for specific requests.

<Tabs>
<TabItem value="openai" label="OpenAI Python">

```python
import openai
client = openai.OpenAI(
    api_key="your-api-key",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    extra_body={
        "guardrails": {
            "noma-guard": {
                "extra_body": {
                    "application_id": "my-specific-app-id"
                }
            }
        }
    }
)
```
</TabItem>

<TabItem value="curl" label="Curl">

```shell
curl 'http://0.0.0.0:4000/v1/chat/completions' \
    -H 'Content-Type: application/json' \
    -d '{
    "model": "gpt-4o-mini",
    "messages": [
        {
            "role": "user",
            "content": "Hello, how are you?"
        }
    ],
    "guardrails": {
        "noma-guard": {
            "extra_body": {
                "application_id": "my-specific-app-id"
            }
        }
    }
}'
```
</TabItem>
</Tabs>

This allows you to override the default `application_id` parameter for specific requests, which is useful for tracking usage across different applications or components.

## Response Details

When content is blocked, Noma provides detailed information about the violations as JSON inside the `message` field, with the following structure:

```json
{
  "error": "Request blocked by Noma guardrail",
  "details": {
    "prompt": {
      "harmfulContent": {
        "result": true,
        "confidence": 0.95
      },
      "sensitiveData": {
        "email": {
          "result": true,
          "entities": ["user@example.com"]
        }
      },
      "bannedTopics": {
        "violence": {
          "result": true,
          "confidence": 0.88
        }
      }
    }
  }
}
```
