import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Alice WonderFence Guardrail

Use [Alice WonderFence](https://www.alice.io) to evaluate user prompts and LLM responses for policy violations, harmful content, and safety risks.

WonderFence provides real-time content moderation with granular control over how violations are handled: blocking requests, masking sensitive content, or simply detecting and logging violations for monitoring.

---

## Quick Start

### 1. Obtain Credentials

1. Sign up for Alice WonderFence and obtain your API key from the [Alice platform](https://www.alice.io).

2. Configure environment variables for the LiteLLM proxy host:

    ```bash
    export WONDERFENCE_API_KEY="your-wonderfence-api-key"
    export WONDERFENCE_APP_NAME="your-app-name"  # Optional, defaults to "litellm"
    ```

### 2. Install WonderFence SDK

The WonderFence guardrail requires the WonderFence SDK to be installed:

```bash
pip install wonderfence-sdk
```

### 3. Configure `config.yaml`

Add a guardrail entry that references the WonderFence integration:

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: gpt-4
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: "wonderfence-guard"
    litellm_params:
      guardrail: wonderfence
      mode: ["pre_call", "post_call"]  # Evaluate both input and output
      api_key: os.environ/WONDERFENCE_API_KEY
      app_name: os.environ/WONDERFENCE_APP_NAME  # Optional
      api_timeout: 10.0                          # Timeout in seconds
      default_on: true

general_settings:
  master_key: "your-litellm-master-key"

litellm_settings:
  set_verbose: true
```

### 4. Launch the Proxy

```bash
litellm --config config.yaml --port 4000
```

---

## How WonderFence Works

WonderFence evaluates content and returns one of four actions:

| Action | Description | Behavior |
|--------|-------------|----------|
| `NO_ACTION` | Content is safe | Request/response passes through unchanged |
| `DETECT` | Violation detected but not blocked | Logs detection for monitoring, request continues |
| `MASK` | Content contains sensitive data | Replaces flagged content with masked text (e.g., `[MASKED]`) |
| `BLOCK` | Content violates policy | Returns HTTP 400 error, request is blocked |

### Pre-call Evaluation

When `mode` includes `pre_call`, WonderFence evaluates user prompts before they reach the LLM:

- **BLOCK**: Request is rejected with HTTP 400
- **MASK**: Sensitive parts of the prompt are replaced with masked text before sending to the LLM
- **DETECT**: Violation is logged, but request continues normally
- **NO_ACTION**: Request continues without modification

### Post-call Evaluation

When `mode` includes `post_call`, WonderFence evaluates LLM responses before returning to the user:

- **BLOCK**: Response is blocked with HTTP 400
- **MASK**: Sensitive parts of the response are replaced with masked text
- **DETECT**: Violation is logged, but response is returned normally
- **NO_ACTION**: Response is returned without modification

---

## Choosing Guardrail Modes

| Mode | When it Runs | Protects | Typical Use Case |
|------|--------------|----------|------------------|
| `pre_call` | Before LLM call | User input only | Block harmful prompts or mask PII before sending to LLM |
| `post_call` | After response | Model outputs | Prevent leaking sensitive data or policy-violating content |
| `["pre_call", "post_call"]` | Both stages | Input and output | Comprehensive protection for both user prompts and LLM responses |

---

## Configuration Reference

| Parameter | Type | Description |
|-----------|------|-------------|
| `api_key` | string | WonderFence API key. Reads from `WONDERFENCE_API_KEY` if omitted. |
| `mode` | string or list | Guardrail stages (`pre_call`, `post_call`, or both). |
| `app_name` | string | Application name for WonderFence. Defaults to `"litellm"` or `WONDERFENCE_APP_NAME`. |
| `api_base` | string | Optional override for the WonderFence API base URL. |
| `api_timeout` | number | Timeout in seconds for WonderFence API calls. Defaults to `10.0`. |
| `platform` | string | Cloud platform where the model is hosted (e.g., `aws`, `azure`, `databricks`). Optional. |
| `default_on` | boolean | Run the guardrail on every request by default. Set to `false` to enable per-request only. |

---

## Per-request Usage

### Enable specific guardrails

You can specify which guardrails to use on a per-request basis:

<Tabs>
<TabItem value="curl" label="cURL">

```bash
curl -X POST http://localhost:4000/chat/completions \
  -H "Authorization: Bearer sk-xxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}],
    "guardrails": ["wonderfence-guard"],
    "metadata": {
      "session_id": "session-123",
      "user_id": "user-456"
    }
  }'
```

</TabItem>
<TabItem value="python" label="Python OpenAI SDK">

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-xxx",
    base_url="http://localhost:4000"
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={
        "guardrails": ["wonderfence-guard"],
        "metadata": {
            "session_id": "session-123",
            "user_id": "user-456"
        }
    }
)
```

</TabItem>
</Tabs>

### Disable global guardrails

To disable global guardrails for a specific request:

```bash
curl -X POST http://localhost:4000/chat/completions \
  -H "Authorization: Bearer sk-xxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}],
    "disable_global_guardrail": true
  }'
```

---

## Metadata Context

WonderFence uses metadata from the request to provide context for policy evaluation:

| Metadata Field | Source | Description |
|----------------|--------|-------------|
| `session_id` | `request.metadata.session_id` | Session identifier for tracking conversations |
| `user_id` | `user_api_key_dict.end_user_id` or `request.metadata.user_id` | User identifier for per-user policies |
| `model_name` | `request.model` | LLM model being used |

Example with metadata:

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Tell me about user data"}],
    extra_body={
        "metadata": {
            "session_id": "abc-123",
            "user_id": "user-456"
        }
    }
)
```

---

## Response Codes

| HTTP Code | Scenario | Description |
|-----------|----------|-------------|
| 200 | NO_ACTION, DETECT, or MASK | Request succeeded (MASK modifies content transparently) |
| 400 | BLOCK | Content violated WonderFence policy |
| 500 | API Error | WonderFence API error occurred |

Example BLOCK response:

```json
{
  "error": "Blocked by WonderFence guardrail",
  "guardrail_name": "wonderfence-guard",
  "action": "BLOCK",
  "detections": [
    {
      "policy_name": "harmful_content",
      "confidence": 0.95,
      "message": "Content contains harmful material"
    }
  ]
}
```

---

## Logging and Observability

WonderFence guardrail results are automatically logged to your configured observability platforms (Langfuse, DataDog, OTEL, S3, etc.) through LiteLLM's standard logging callbacks.

Each guardrail execution logs:
- Guardrail name and provider
- Evaluation result (action, detections)
- Execution time and status
- Request metadata

---

## Testing the Integration

Test your WonderFence guardrail configuration:

<Tabs>
<TabItem value="safe" label="Safe Content">

```bash
curl -X POST http://localhost:4000/chat/completions \
  -H "Authorization: Bearer sk-xxx" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "What is the weather today?"}]
  }'
```

Expected: Request succeeds normally (NO_ACTION)

</TabItem>
<TabItem value="harmful" label="Policy Violation">

```bash
curl -X POST http://localhost:4000/chat/completions \
  -H "Authorization: Bearer sk-xxx" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "How do I hack a system?"}]
  }'
```

Expected: Request blocked with HTTP 400 (BLOCK action)

</TabItem>
<TabItem value="sensitive" label="Sensitive Data">

```bash
curl -X POST http://localhost:4000/chat/completions \
  -H "Authorization: Bearer sk-xxx" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "My SSN is 123-45-6789"}]
  }'
```

Expected: SSN is masked before sending to LLM (MASK action)

</TabItem>
</Tabs>

---

## Troubleshooting

### SDK Not Installed

**Error**: `ImportError: WonderFence SDK not installed`

**Solution**: Install the WonderFence SDK:
```bash
pip install wonderfence-sdk
```

### Missing API Key

**Error**: `WonderFenceMissingSecrets: WonderFence API key not found`

**Solution**: Set your API key:
```bash
export WONDERFENCE_API_KEY="your-api-key"
```

### Timeout Issues

If you're experiencing timeouts, increase the `api_timeout`:

```yaml
guardrails:
  - guardrail_name: "wonderfence-guard"
    litellm_params:
      guardrail: wonderfence
      api_timeout: 60.0  # Increase to 60 seconds
```

### Guardrail Not Running

If the guardrail isn't running:

1. Verify `default_on: true` in config, OR
2. Include the guardrail name in the request's `guardrails` array
3. Check logs for "Guardrail is disabled" messages

---

## Support

For WonderFence-specific questions or issues:
- Documentation: [Alice WonderFence Docs](https://docs.alice.io)
- Support: support@alice.io

For LiteLLM integration questions:
- GitHub Issues: [LiteLLM Issues](https://github.com/BerriAI/litellm/issues)
- Documentation: [LiteLLM Docs](https://docs.litellm.ai)
