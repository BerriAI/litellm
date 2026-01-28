import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Gray Swan Cygnal Guardrail

Use [Gray Swan Cygnal](https://docs.grayswan.ai/cygnal/monitor-requests) to continuously monitor conversations for policy violations, indirect prompt injection (IPI), jailbreak attempts, and other safety risks.

Cygnal returns a `violation` score between `0` and `1` (higher means more likely to violate policy), plus metadata such as violated rule indices, mutation detection, and IPI flags. LiteLLM can automatically block or monitor requests based on this signal.

---

## Quick Start

### 1. Obtain Credentials

1. Log in to our Gray Swan platform and generate a Cygnal API key. 

    For existing customers, you should already have access to our [platform](https://platform.grayswan.ai).

    For new users, please register at this [page](https://hubs.ly/Q03-sX1J0) and we are more than happy to give you an onboarding!


2. Configure environment variables for the LiteLLM proxy host:

    ```bash
    export GRAYSWAN_API_KEY="your-grayswan-key"
    export GRAYSWAN_API_BASE="https://api.grayswan.ai"
    ```

### 2. Configure `config.yaml`

Add a guardrail entry that references the Gray Swan integration. Below is our recommmended settings.

```yaml
model_list:                                 # this part is a standard litellm configuration for reference
  - model_name: openai/gpt-4.1-mini
    litellm_params:
      model: openai/gpt-4.1-mini
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: "cygnal-monitor"
    litellm_params:
      guardrail: grayswan
      mode: [pre_call, post_call]            # monitor both input and output
      api_key: os.environ/GRAYSWAN_API_KEY
      api_base: os.environ/GRAYSWAN_API_BASE  # optional
      optional_params:
        on_flagged_action: passthrough         # or "block" or "monitor"
        violation_threshold: 0.5               # score >= threshold is flagged
        reasoning_mode: hybrid                 # off | hybrid | thinking
        policy_id: "your-cygnal-policy-id"     # Optional: Your Cygnal policy ID. Defaults to a content safety policy if empty.
      streaming_end_of_stream_only: true       # For streaming API, only send the assembled message to Cygnal (post_call only). Defaults to false.
      default_on: true
      guardrail_timeout: 30                   # Defaults to 30 seconds. Change accordingly.
      fail_open: true                         # Defaults to true; set to false to propagate guardrail errors.

general_settings:
  master_key: "your-litellm-master-key"

litellm_settings:
  set_verbose: true
```

### 3. Launch the Proxy

```bash
litellm --config config.yaml --port 4000
```

---

## Choosing Guardrail Modes

Gray Swan can run during `pre_call`, `during_call`, and `post_call` stages. Combine modes based on your latency and coverage requirements. 

| Mode         | When it Runs      | Protects              | Typical Use Case |
|--------------|-------------------|-----------------------|------------------|
| `pre_call`   | Before LLM call   | User input only       | Block prompt injection before it reaches the model |
| `during_call`| Parallel to call  | User input only       | Low-latency monitoring without blocking |
| `post_call`  | After response    | Model Outputs         | Scan output for policy violations, leaked secrets, or IPI |


When using `during_call` with `on_flagged_action: block` or `on_flagged_action: passthrough`:

- **The LLM call runs in parallel** with the guardrail check using `asyncio.gather`
- **LLM tokens are still consumed** even if the guardrail detects a violation
- The guardrail exception prevents the response from reaching the user, but **does not cancel the running LLM task**
- This means you pay full LLM costs while returning an error/passthrough message to the user

**Recommendation:** Use `pre_call` and `post_call` instead of `during_call` for `passthrough` (or `block`) `on_flagged_action` (see our recommended configuration above). Reserve `during_call` for `monitor` mode ONLY when you want low-latency logging without impacting the user experience.


---

## Work with Claude Code

Follow the official litellm [guide](https://docs.litellm.ai/docs/tutorials/claude_responses_api) on setting up Claude Code with litellm, with the guardrail part mentioned above added to your litellm configuration. Cygnal natively supports coding agent policies defense. Define your own policy or use the provided coding policies on the platform. The example config we show above is also the recommended setup for Claude Code (with the `policy_id` replaced with an appropriate one).

---

## Per-request overrides via `extra_body`

You can override parts of the Gray Swan guardrail configuration on a per-request basis by passing `litellm_metadata.guardrails[*].grayswan.extra_body`.

`extra_body` is merged into the Cygnal request body and takes precedence over specific fields from `config.yaml`, which are `policy_id`, `violation_threshold`, and `reasoning_mode`.

If you include a `metadata` field inside `extra_body`, it is forwarded to the Cygnal API as-is under the request body's `metadata` field.

Example:

```bash
curl -X POST "http://0.0.0.0:4000/v1/messages?beta=true" \
  -H "Authorization: Bearer token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openrouter/anthropic/claude-sonnet-4.5",
    "messages": [{"role": "user", "content": "hello"}],
    "litellm_metadata": {
      "guardrails": [
        {
          "cygnal-monitor": {
            "extra_body": {
              "policy_id": "specific policy id you want to use",
              "metadata": {
                "user": "health-check"
              }
            }
          }
        }
      ]
    }
  }'
```

OpenAI client:

```python
from openai import OpenAI

client = OpenAI(api_key="anything", base_url="http://0.0.0.0:4000")

resp = client.responses.create(
    model="openrouter/anthropic/claude-sonnet-4.5",
    input="hello",
    extra_body={
        "litellm_metadata": {
            "guardrails": [
                {
                    "cygnal-monitor": {
                        "extra_body": {
                            "policy_id": "69038214e5cdb6befc5e991e",
                            "metadata": {"trace_id": "trace-123"},
                        }
                    }
                }
            ]
        }
    },
)
```

Anthropic client:

```python
from anthropic import Anthropic

client = Anthropic(api_key="anything", base_url="http://0.0.0.0:4000")

resp = client.messages.create(
    model="openrouter/anthropic/claude-sonnet-4.5",
    max_tokens=256,
    messages=[{"role": "user", "content": "hello"}],
    extra_body={
        "litellm_metadata": {
            "guardrails": [
                {
                    "cygnal-monitor": {
                        "extra_body": {
                            "policy_id": "69038214e5cdb6befc5e991e",
                            "metadata": {"trace_id": "trace-123"},
                        }
                    }
                }
            ]
        }
    },
)
```

Notes:

- The guardrail name (for example, `cygnal-monitor`) must match the `guardrail_name` in `config.yaml`.
- Per-request guardrail overrides may require a premium license, depending on your proxy settings.

---

## Configuration Reference

| Parameter                             | Type            | Description |
|---------------------------------------|-----------------|-------------|
| `api_key`                             | string          | Gray Swan Cygnal API key. Reads from `GRAYSWAN_API_KEY` if omitted. |
| `api_base`                            | string          | Override for the Gray Swan API base URL. Defaults to `https://api.grayswan.ai` or `GRAYSWAN_API_BASE`. |
| `mode`                                | string or list  | Guardrail stages (`pre_call`, `during_call`, `post_call`). |
| `optional_params.on_flagged_action`   | string          | `monitor` (log only), `block` (raise `HTTPException`), or `passthrough` (replace response content with violation message, no 400 error). |
| `optional_params.violation_threshold` | number (0-1)    | Scores at or above this value are considered violations. |
| `optional_params.reasoning_mode`      | string          | `off`, `hybrid`, or `thinking`. Enables Cygnal's reasoning capabilities. |
| `optional_params.categories`          | object          | Map of custom category names to descriptions. |
| `optional_params.policy_id`           | string          | Gray Swan policy identifier. |
| `guardrail_timeout`                   | number          | Timeout in seconds for the Cygnal request. Defaults to 30. |
| `fail_open`                           | boolean         | If true, errors contacting Cygnal are logged and the request proceeds; if false, errors propagate. Defaults to treu. |
| `streaming_end_of_stream_only`        | boolean         | For streaming `post_call`, only send the final assembled response to Cygnal. Defaults to false. |
| `default_on`                          | boolean         | Run the guardrail on every request by default. |
