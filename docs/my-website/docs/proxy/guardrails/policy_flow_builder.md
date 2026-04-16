# Policy Flow Builder

The Policy Flow Builder lets you design guardrail pipelines with **conditional execution**. Instead of running guardrails independently, you chain them into ordered steps and control what happens when each guardrail passes or fails.

Two powerful patterns it enables: **guardrail fallbacks** (try a different guardrail when one fails) and **retrying the same guardrail** (run the same guardrail again if it fails, e.g. to handle transient errors).

## When to use the Flow Builder

| Approach | Use case |
|----------|----------|
| **Simple policy** (`guardrails.add`) | All guardrails run in parallel; any failure blocks the request. |
| **Flow Builder** (pipeline) | Guardrails run in sequence; you choose actions per step (next, block, allow, custom response). |

Use the Flow Builder when you need:

- **Guardrail fallbacks** — use `on_fail: next` to try a different guardrail when one fails (e.g., fast filter → stricter filter)
- **Retrying the same guardrail** — add the same guardrail as multiple steps; if it fails, `on_fail: next` moves to the next step, which can be the same guardrail again (useful for transient API errors or rate limits)
- **Conditional routing** — e.g., if a fast guardrail fails, run a more advanced one instead of blocking immediately
- **Custom responses** — return a specific message when a guardrail fails instead of a generic block
- **Data chaining** — pass modified data (e.g., PII-masked content) from one step to the next
- **Fine-grained control** — different actions on pass vs. fail per step

## Concepts

### Pipeline

A pipeline has:

- **Mode**: `pre_call` (before the LLM) or `post_call` (after the LLM)
- **Steps**: Ordered list of guardrail steps

### Step actions

Each step defines what happens when the guardrail **passes** and when it **fails**:

| Action | Description |
|--------|-------------|
| **Next Step** | Continue to the next guardrail in the pipeline |
| **Allow** | Stop the pipeline and allow the request to proceed |
| **Block** | Stop the pipeline and block the request |
| **Custom Response** | Return a custom message instead of the default block |

### Step options

| Field | Type | Description |
|-------|------|--------------|
| `guardrail` | `string` | Name of the guardrail to run |
| `on_pass` | `string` | Action when guardrail passes: `next`, `allow`, `block`, `modify_response` |
| `on_fail` | `string` | Action when guardrail fails: `next`, `allow`, `block`, `modify_response` |
| `pass_data` | `boolean` | Forward modified request data (e.g., PII-masked) to the next step |
| `modify_response_message` | `string` | Custom message when using `modify_response` action |

## Using the Flow Builder (UI)

1. Go to **Policies** in the LiteLLM Admin UI
2. Click **+ Create New Policy** or **Edit** on an existing policy
3. Select **Flow Builder** (instead of the simple form)
4. Design your flow:
   - **Trigger** — Incoming LLM request (runs when the policy matches)
   - **Steps** — Add guardrails, set ON PASS and ON FAIL actions per step
   - **End** — Request proceeds to the LLM
5. Use the **+** between steps to insert new steps
6. Use the **Test** panel to run sample messages through the pipeline before saving
7. Click **Save** to create or update the policy

## Config (YAML)

Define a pipeline in your policy config:

```yaml showLineNumbers title="config.yaml"
guardrails:
  - guardrail_name: pii_masking
    litellm_params:
      guardrail: presidio
      mode: pre_call

  - guardrail_name: prompt_injection
    litellm_params:
      guardrail: lakera
      mode: pre_call

policies:
  my-pipeline-policy:
    description: "PII mask first, then check for prompt injection"
    guardrails:
      add:
        - pii_masking
        - prompt_injection
    pipeline:
      mode: pre_call
      steps:
        - guardrail: pii_masking
          on_pass: next
          on_fail: block
          pass_data: true
        - guardrail: prompt_injection
          on_pass: allow
          on_fail: block

policy_attachments:
  - policy: my-pipeline-policy
    scope: "*"
```

## Fallbacks and retries

### Guardrail fallbacks

Use `on_fail: next` to fall back to another guardrail when one fails. Run a lightweight guardrail first; if it fails, escalate to a stricter or different provider:

```yaml
policies:
  fallback-policy:
    guardrails:
      add:
        - fast_content_filter
        - strict_content_filter
    pipeline:
      mode: pre_call
      steps:
        - guardrail: fast_content_filter
          on_pass: allow
          on_fail: next
        - guardrail: strict_content_filter
          on_pass: allow
          on_fail: block
```

If `fast_content_filter` passes → allow. If it fails → run `strict_content_filter`; pass → allow, fail → block.

### Retrying the same guardrail

Add the same guardrail as multiple steps to retry on failure. Useful for transient errors (API timeouts, rate limits):

```yaml
policies:
  retry-policy:
    guardrails:
      add:
        - lakera_prompt_injection
    pipeline:
      mode: pre_call
      steps:
        - guardrail: lakera_prompt_injection
          on_pass: allow
          on_fail: next
        - guardrail: lakera_prompt_injection
          on_pass: allow
          on_fail: block
```

First attempt passes → allow. First attempt fails → retry the same guardrail; second pass → allow, second fail → block.

## Example: Custom response on fail

Return a branded message instead of a generic block:

```yaml
policies:
  branded-block-policy:
    guardrails:
      add:
        - pii_detector
    pipeline:
      mode: pre_call
      steps:
        - guardrail: pii_detector
          on_pass: allow
          on_fail: modify_response
          modify_response_message: "Your message contains sensitive information. Please remove PII and try again."
```

## Test a pipeline (API)

Test a pipeline with sample messages before attaching it:

```bash
curl -X POST "http://localhost:4000/policies/test-pipeline" \
  -H "Authorization: Bearer <your_api_key>" \
  -H "Content-Type: application/json" \
  -d '{
    "pipeline": {
      "mode": "pre_call",
      "steps": [
        {
          "guardrail": "pii_masking",
          "on_pass": "next",
          "on_fail": "block",
          "pass_data": true
        },
        {
          "guardrail": "prompt_injection",
          "on_pass": "allow",
          "on_fail": "block"
        }
      ]
    },
    "test_messages": [
      {"role": "user", "content": "What is 2+2?"},
      {"role": "user", "content": "My SSN is 123-45-6789"}
    ]
  }'
```

Response includes per-step outcomes (pass/fail/error), actions taken, and timing.

## Pipeline vs simple policy

When a policy has a `pipeline`, the pipeline defines execution order and actions. The `guardrails.add` list must include all guardrails used in the pipeline steps.

| Policy type | Execution |
|-------------|-----------|
| Simple (`guardrails.add` only) | All guardrails run; any failure blocks |
| Pipeline (`pipeline` present) | Steps run in order; actions control flow |

## Related docs

- [Guardrail Policies](./guardrail_policies) — Policy basics, attachments, inheritance
- [Policy Templates](./policy_templates) — Pre-built policy templates
