import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Guardrail Pipeline Flow Builder

The **Flow Builder** lets you design guardrail policies with **conditional, sequential execution**. Instead of running guardrails independently, you chain them into a pipeline where each step has configurable **ON PASS** and **ON FAIL** actions. This enables multi-tier fallbacks, retries, and escalation paths.

## When to use the Flow Builder

| Use Case | Simple Policy | Pipeline (Flow Builder) |
|----------|---------------|-------------------------|
| Run multiple guardrails together | ✅ | ✅ |
| All guardrails run independently | ✅ | ❌ |
| Conditional execution (if A fails → try B) | ❌ | ✅ |
| Fallback to different guardrail on failure | ❌ | ✅ |
| Retry same guardrail before blocking | ❌ | ✅ |
| Pass modified data (e.g., PII-masked) to next step | ❌ | ✅ |

**Use the Flow Builder when** you need:
- **Fallbacks** — Try a fast/simple guardrail first; if it fails, escalate to a stricter one
- **Retries** — Run the same guardrail multiple times before blocking (e.g., for flaky APIs)
- **Escalation** — Route to different guardrails based on pass/fail outcomes

## Quick Start

<Tabs>
<TabItem value="ui" label="UI (LiteLLM Dashboard)">

1. Go to **Policies** → **+ Create New Policy**
2. Choose **Flow Builder** (instead of Simple)
3. Click **Continue to Builder** to open the full-screen Flow Builder
4. Add steps, select guardrails, and configure ON PASS / ON FAIL actions
5. Use **Test** to run a sample message through the pipeline before saving
6. Save the policy

</TabItem>
<TabItem value="config" label="config.yaml">

```yaml showLineNumbers title="config.yaml"
guardrails:
  - guardrail_name: strict-filter
    litellm_params:
      guardrail: lakera
      mode: pre_call
      api_key: os.environ/LAKERA_API_KEY
  - guardrail_name: permissive-filter
    litellm_params:
      guardrail: presidio
      mode: pre_call

policies:
  content-safety:
    guardrails:
      add: [strict-filter, permissive-filter]
    pipeline:
      mode: pre_call
      steps:
        - guardrail: strict-filter
          on_fail: next
          on_pass: allow
        - guardrail: permissive-filter
          on_fail: block
          on_pass: allow

policy_attachments:
  - policy: content-safety
    scope: "*"
```

</TabItem>
</Tabs>

## Step Actions

Each pipeline step has two actions:

| Action | When | Description |
|--------|------|-------------|
| **Next Step** | ON PASS or ON FAIL | Continue to the next step in the pipeline |
| **Allow** | ON PASS or ON FAIL | Stop the pipeline and allow the request |
| **Block** | ON PASS or ON FAIL | Stop the pipeline and block the request |
| **Custom Response** | ON PASS or ON FAIL | Stop and return a custom message instead of the default block/allow |

### Common patterns

**Fallback chain** — Try strict first, escalate to permissive on failure:

```yaml
steps:
  - guardrail: strict-filter
    on_fail: next    # strict failed → try next
    on_pass: allow
  - guardrail: permissive-filter
    on_fail: block   # permissive failed → block
    on_pass: allow
```

**Retry same guardrail** — Run the same guardrail twice before blocking:

```yaml
steps:
  - guardrail: lakera-pii
    on_fail: next
    on_pass: allow
  - guardrail: lakera-pii
    on_fail: block
    on_pass: allow
```

**Pass modified data** — Forward PII-masked content to the next step:

```yaml
steps:
  - guardrail: presidio-pii
    on_fail: block
    on_pass: next
    pass_data: true   # PII-masked request/response sent to next step
  - guardrail: prompt-injection
    on_fail: block
    on_pass: allow
```

## Pipeline Fields

### `pipeline`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `mode` | `pre_call` \| `post_call` | Yes | When the pipeline runs (before or after the LLM call) |
| `steps` | `list[PipelineStep]` | Yes | Ordered list of steps (at least 1) |

### `PipelineStep`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `guardrail` | `string` | — | Name of the guardrail to run |
| `on_fail` | `next` \| `block` \| `allow` \| `modify_response` | `block` | Action when guardrail rejects |
| `on_pass` | `next` \| `block` \| `allow` \| `modify_response` | `allow` | Action when guardrail passes |
| `pass_data` | `bool` | `false` | Forward modified request/response to next step |
| `modify_response_message` | `string` | Optional | Custom message for `modify_response` action |

## Example: Multi-tier content safety

```yaml showLineNumbers title="config.yaml"
guardrails:
  - guardrail_name: "pii-masking"
    litellm_params:
      guardrail: presidio
      mode: pre_call
  - guardrail_name: "prompt-injection"
    litellm_params:
      guardrail: lakera
      mode: pre_call
      api_key: os.environ/LAKERA_API_KEY

policies:
  content-safety-pipeline:
    description: "PII mask → prompt injection check → allow or block"
    guardrails:
      add: [pii-masking, prompt-injection]
    pipeline:
      mode: pre_call
      steps:
        - guardrail: pii-masking
          on_fail: block
          on_pass: next
          pass_data: true
        - guardrail: prompt-injection
          on_fail: block
          on_pass: allow

policy_attachments:
  - policy: content-safety-pipeline
    scope: "*"
```

**Flow:** 1) Mask PII → 2) Check masked content for prompt injection → 3) Allow or block.

## Example: Retry with same guardrail

Useful when a guardrail API is flaky or rate-limited:

```yaml showLineNumbers title="config.yaml"
policies:
  retry-on-failure:
    guardrails:
      add: [pii_masking]
    pipeline:
      mode: pre_call
      steps:
        - guardrail: pii_masking
          on_fail: next
          on_pass: allow
        - guardrail: pii_masking
          on_fail: block
          on_pass: allow
```

**Flow:** Run `pii_masking` twice. Block only if it fails both times.

## Example: Custom response on failure

Return a branded message instead of the default block:

```yaml
steps:
  - guardrail: strict-filter
    on_fail: modify_response
    modify_response_message: "Your request was blocked. Please remove sensitive content and try again."
    on_pass: allow
```

## Pipeline vs Simple Policy

**Simple policy** — All guardrails run independently. If any fail, the request is blocked (or handled per guardrail config).

**Pipeline policy** — Guardrails run in order. Each step has conditional actions. You control the flow (fallback, retry, escalate).

```mermaid
flowchart TD
    subgraph Simple["Simple Policy"]
        S1[Guardrail A] --> R[Result: block if any fail]
        S2[Guardrail B] --> R
    end

    subgraph Pipeline["Pipeline Policy"]
        P1[Step 1: Guardrail A] -->|on_fail: next| P2[Step 2: Guardrail B]
        P1 -->|on_pass: allow| Allow
        P2 -->|on_fail: block| Block
        P2 -->|on_pass: allow| Allow
    end

## Testing the pipeline

### In the UI

The Flow Builder includes a **Test** panel. Enter a sample message and click **Run** to see which steps pass or fail and what action is taken.

### Via API

Use the [Test Playground](/docs/proxy/guardrails/test_playground) or send a request with the policy attached:

```bash
curl -X POST http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-..." \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Test message"}],
    "guardrails": ["content-safety-pipeline"]
  }'
```

## Response headers

When a pipeline runs, response headers include:

| Header | Description |
|--------|-------------|
| `x-litellm-applied-policies` | Policies that matched |
| `x-litellm-applied-guardrails` | Guardrails that ran |
| `x-litellm-policy-sources` | Why each policy matched |

## Related

- [Guardrail Policies](/docs/proxy/guardrails/guardrail_policies) — Policies overview, attachments, inheritance
- [Policy Templates](/docs/proxy/guardrails/policy_templates) — Pre-configured policy templates
- [Guardrails Quick Start](/docs/proxy/guardrails/quick_start) — Defining guardrails
