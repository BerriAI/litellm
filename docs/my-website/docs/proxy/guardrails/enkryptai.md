import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# EnkryptAI Guardrails

LiteLLM supports EnkryptAI guardrails for content moderation and safety checks on LLM inputs and outputs.

## Quick Start

### 1. Define Guardrails on your LiteLLM config.yaml

Define your guardrails under the `guardrails` section:

```yaml
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: "enkryptai-guard"
    litellm_params:
      guardrail: enkryptai
      mode: "pre_call"
      api_key: os.environ/ENKRYPTAI_API_KEY
      detectors:
        toxicity:
          enabled: true
        nsfw:
          enabled: true
        pii:
          enabled: true
          entities: ["email", "phone", "secrets"]
        injection_attack:
          enabled: true
```

#### Supported values for `mode`

- `pre_call` - Run **before** LLM call, on **input**
- `post_call` - Run **after** LLM call, on **output**
- `during_call` - Run **during** LLM call, on **input**. Same as `pre_call` but runs in parallel as LLM call

#### Available Detectors

EnkryptAI supports multiple content detection types:

- **toxicity** - Detect toxic language
- **nsfw** - Detect NSFW (Not Safe For Work) content
- **pii** - Detect personally identifiable information
  - Configure entities: `["pii", "email", "phone", "secrets", "ip_address", "url"]`
- **injection_attack** - Detect prompt injection attempts
- **keyword_detector** - Detect custom keywords/phrases
- **policy_violation** - Detect policy violations
- **bias** - Detect biased content
- **sponge_attack** - Detect sponge attacks

### 2. Set Environment Variables

```bash
export ENKRYPTAI_API_KEY="your-api-key"
```

### 3. Start LiteLLM Gateway

```shell
litellm --config config.yaml --detailed_debug
```

### 4. Test Request

**[Langchain, OpenAI SDK Usage Examples](../proxy/user_keys#request-format)**

<Tabs>
<TabItem label="Successful Call" value="allowed">

```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "Hello, how can you help me today?"}
    ],
    "guardrails": ["enkryptai-guard"]
  }'
```

**Response: HTTP 200 Success**

Content passes all detector checks and is allowed through.

</TabItem>

<TabItem label="Unsuccessful Call" value="not-allowed">

Expect this to fail if content violates detector policies:

```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "My email is test@example.com and my SSN is 123-45-6789"}
    ],
    "guardrails": ["enkryptai-guard"]
  }'
```

**Expected Response on Failure: HTTP 400 Error**

```json
{
  "error": {
    "message": {
      "error": "Content blocked by EnkryptAI guardrail",
      "detected": true,
      "violations": ["pii"],
      "response": {
        "summary": {
          "pii": 1
        },
        "details": {
          "pii": {
            "detected": ["email", "ssn"]
          }
        }
      }
    },
    "type": "None",
    "param": "None",
    "code": "400"
  }
}
```

</TabItem>
</Tabs>

## Video Walkthrough

<iframe width="840" height="500" src="https://www.loom.com/embed/ff222211e0864937aee4aeef0f28c3b7" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>

## Advanced Configuration

### Using Custom Policies

You can specify a custom EnkryptAI policy:

```yaml
guardrails:
  - guardrail_name: "enkryptai-custom"
    litellm_params:
      guardrail: enkryptai
      mode: "pre_call"
      api_key: os.environ/ENKRYPTAI_API_KEY
      policy_name: "my-custom-policy"  # Sent via x-enkrypt-policy header
      detectors:
        toxicity:
          enabled: true
```

### Using Deployments

Specify an EnkryptAI deployment:

```yaml
guardrails:
  - guardrail_name: "enkryptai-deployment"
    litellm_params:
      guardrail: enkryptai
      mode: "pre_call"
      api_key: os.environ/ENKRYPTAI_API_KEY
      deployment_name: "production"  # Sent via X-Enkrypt-Deployment header
      detectors:
        toxicity:
          enabled: true
```

### Monitor Mode (Logging Without Blocking)

Set `block_on_violation: false` to log violations without blocking requests:

```yaml
guardrails:
  - guardrail_name: "enkryptai-monitor"
    litellm_params:
      guardrail: enkryptai
      mode: "pre_call"
      api_key: os.environ/ENKRYPTAI_API_KEY
      block_on_violation: false  # Log violations but don't block
      detectors:
        toxicity:
          enabled: true
        nsfw:
          enabled: true
```

In monitor mode, all violations are logged but requests are never blocked.

### Input and Output Guardrails

Configure separate guardrails for input and output:

```yaml
guardrails:
  # Input guardrail
  - guardrail_name: "enkryptai-input"
    litellm_params:
      guardrail: enkryptai
      mode: "pre_call"
      api_key: os.environ/ENKRYPTAI_API_KEY
      detectors:
        pii:
          enabled: true
          entities: ["email", "phone", "ssn"]
        injection_attack:
          enabled: true

  # Output guardrail
  - guardrail_name: "enkryptai-output"
    litellm_params:
      guardrail: enkryptai
      mode: "post_call"
      api_key: os.environ/ENKRYPTAI_API_KEY
      detectors:
        toxicity:
          enabled: true
        nsfw:
          enabled: true
```

## Configuration Options

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `api_key` | string | EnkryptAI API key | `ENKRYPTAI_API_KEY` env var |
| `api_base` | string | EnkryptAI API base URL | `https://api.enkryptai.com` |
| `policy_name` | string | Custom policy name (sent via `x-enkrypt-policy` header) | None |
| `deployment_name` | string | Deployment name (sent via `X-Enkrypt-Deployment` header) | None |
| `detectors` | object | Detector configuration | `{}` |
| `block_on_violation` | boolean | Block requests on violations | `true` |
| `mode` | string | When to run: `pre_call`, `post_call`, or `during_call` | Required |

## Observability

EnkryptAI guardrail logs include:

- **guardrail_status**: `success`, `guardrail_intervened`, or `guardrail_failed_to_respond`
- **guardrail_provider**: `enkryptai`
- **guardrail_json_response**: Full API response with detection details
- **duration**: Time taken for guardrail check
- **start_time** and **end_time**: Timestamps

These logs are available through your configured LiteLLM logging callbacks.

## Error Handling

The guardrail handles errors gracefully:

- **API Failures**: Logs error and raises exception
- **Rate Limits (429)**: Logs error and raises exception
- **Invalid Configuration**: Raises `ValueError` on initialization

Set `block_on_violation: false` to continue processing even when violations are detected (monitor mode).

## Support

For more information about EnkryptAI:
- Documentation: [https://docs.enkryptai.com](https://docs.enkryptai.com)
- Website: [https://enkryptai.com](https://enkryptai.com)

