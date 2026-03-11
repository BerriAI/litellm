import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Policy Flow Builder

The Policy Flow Builder lets you define **guardrail pipelines** with conditional, sequential execution. Instead of running guardrails independently, you chain them into multi-step flows where each step has configurable **ON PASS** and **ON FAIL** actions.

## Fallback Guardrails & Retry

A primary use case is **fallback and retry**: run a strict guardrail first, and if it fails, **retry with a fallback guardrail** instead of blocking immediately. Use `on_fail: next` to escalate to the next step.

| Pattern | Use case |
|---------|----------|
| **Strict → Permissive fallback** | Fast, cheap guardrail first; if it flags content, retry with a more accurate (or different provider) guardrail before deciding to block |
| **Provider fallback** | Primary guardrail (e.g., Lakera) fails or times out → fall back to a secondary provider (e.g., custom model) |
| **Tiered validation** | Lightweight check first; only run expensive checks when the first step fails |

This reduces false positives (strict-only can over-block) and improves resilience (provider outages don't block all traffic).

## Why use the Flow Builder?

- **Fallback & retry** — Retry with a different guardrail when the first fails instead of blocking immediately
- **Conditional escalation** — Strict fails → route to permissive; only block if both fail
- **Sequential execution** — Run guardrails in order; later steps can use modified data from earlier steps
- **Flexible actions** — Choose Next Step, Block, Allow, or Custom Response per pass/fail outcome
- **Test before deploy** — Run the pipeline against sample messages before saving

## Quick Start

<Tabs>
<TabItem value="ui" label="UI (LiteLLM Dashboard)">

**Step 1: Open the Flow Builder**

1. Go to **Policies** and click **+ Create New Policy**
2. Select **Flow Builder** (instead of Simple)
3. Click **Continue to Builder**

Or, when editing an existing policy with a pipeline, click **Edit** — the Flow Builder opens directly.

**Step 2: Build your pipeline**

1. Add steps by clicking the **+** between steps
2. For each step, select a guardrail and set **ON PASS** and **ON FAIL** actions
3. Use **Test** to run the pipeline against sample messages
4. Click **Save** when done

</TabItem>
<TabItem value="config" label="config.yaml">

```yaml showLineNumbers title="config.yaml"
guardrails:
  - guardrail_name: strict-filter
    litellm_params:
      guardrail: my_guardrails.StrictFilter
      mode: pre_call
  - guardrail_name: permissive-filter
    litellm_params:
      guardrail: my_guardrails.PermissiveFilter
      mode: pre_call

policies:
  content-safety:
    description: "Strict filter with permissive fallback"
    guardrails:
      add: [strict-filter, permissive-filter]
    pipeline:
      mode: pre_call
      steps:
        - guardrail: strict-filter
          on_fail: next    # escalate to permissive
          on_pass: allow   # clean content proceeds
        - guardrail: permissive-filter
          on_fail: block   # hard block
          on_pass: allow
```

</TabItem>
</Tabs>

## Step Actions

Each pipeline step has two action dropdowns:

| Action | Description |
|--------|-------------|
| **Next Step** | Continue to the next step in the pipeline |
| **Allow** | Stop the pipeline and allow the request |
| **Block** | Stop the pipeline and block the request |
| **Custom Response** | Return a custom message instead of the default block/allow response |

### ON PASS vs ON FAIL

- **ON PASS** — Action when the guardrail accepts the content
- **ON FAIL** — Action when the guardrail rejects the content

**Fallback example:** A strict PII filter with `on_fail: next` escalates to a permissive filter; if the permissive filter passes, the request is allowed. The pipeline effectively **retries** with the fallback guardrail when the first one fails.

## Pipeline Mode

| Mode | When it runs |
|------|--------------|
| `pre_call` | Before the request is sent to the LLM (input validation) |
| `post_call` | After the LLM responds (output validation) |

## Example: Fallback & Retry

```yaml showLineNumbers title="config.yaml"
guardrails:
  - guardrail_name: lakera-prompt-injection
    litellm_params:
      guardrail: lakera
      mode: pre_call
      api_key: os.environ/LAKERA_API_KEY
  - guardrail_name: custom-pii-check
    litellm_params:
      guardrail: presidio
      mode: pre_call

policies:
  # Fallback & retry: Lakera first, fall back to custom check when it fails
  prompt-safety-with-fallback:
    guardrails:
      add: [lakera-prompt-injection, custom-pii-check]
    pipeline:
      mode: pre_call
      steps:
        - guardrail: lakera-prompt-injection
          on_fail: next   # retry with fallback guardrail instead of blocking
          on_pass: allow
        - guardrail: custom-pii-check
          on_fail: block  # both failed → block
          on_pass: allow
```

**Flow:** Request → Lakera (fail) → **retry** with custom-pii-check → allow if it passes, block if it fails.

Set `pass_data: true` on a step to forward modified request data (e.g., PII-masked content) to the next step. Useful when an earlier guardrail transforms the input and you want later steps to operate on the transformed data.

```yaml
steps:
  - guardrail: pii_masking
    on_pass: next
    on_fail: block
    pass_data: true   # forward masked content to next step
  - guardrail: prompt_injection
    on_pass: allow
    on_fail: block
```

## Test Pipeline (API)

Run a pipeline against sample messages without saving:

```bash
curl -X POST "http://localhost:4000/policies/test-pipeline" \
  -H "Authorization: Bearer <your_api_key>" \
  -H "Content-Type: application/json" \
  -d '{
    "pipeline": {
      "mode": "pre_call",
      "steps": [
        {"guardrail": "strict-filter", "on_pass": "next", "on_fail": "block"},
        {"guardrail": "permissive-filter", "on_pass": "allow", "on_fail": "block"}
      ]
    },
    "test_messages": [
      {"role": "user", "content": "Sample message to test"}
    ]
  }'
```

Response includes step-by-step results: which guardrails passed/failed, actions taken, and timing.

## Config Reference

### `pipeline` (optional)

When present on a policy, guardrails run in pipeline order instead of independently.

```yaml
pipeline:
  mode: pre_call | post_call
  steps:
    - guardrail: <guardrail_name>
      on_pass: next | allow | block | modify_response
      on_fail: next | allow | block | modify_response
      pass_data: false | true
      modify_response_message: <string>   # for modify_response action
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `string` | `pre_call` | When the pipeline runs: `pre_call` or `post_call` |
| `steps` | `list` | — | Ordered list of pipeline steps (at least 1) |
| `guardrail` | `string` | — | **Required.** Name of the guardrail to run |
| `on_pass` | `string` | `allow` | Action when guardrail passes |
| `on_fail` | `string` | `block` | Action when guardrail fails |
| `pass_data` | `bool` | `false` | Forward modified data to next step |
| `modify_response_message` | `string` | `null` | Custom message for `modify_response` action |

### Relationship to `guardrails.add`

`guardrails.add` lists which guardrails the policy uses. When a `pipeline` is present, those guardrails are executed in the order defined by `pipeline.steps`. If there is no `pipeline`, guardrails in `guardrails.add` run independently (legacy behavior).
