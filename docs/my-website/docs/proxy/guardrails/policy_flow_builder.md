# Policy Flow Builder

The Policy Flow Builder lets you design guardrail pipelines with **conditional execution**. Instead of running guardrails independently, you chain them into ordered steps and control what happens when each guardrail **passes**, **fails a policy check** (content intervention), or hits a **technical error** (e.g. timeout, unreachable provider, missing guardrail).

Two powerful patterns it enables: **guardrail fallbacks** (try a different guardrail when one fails) and **retrying the same guardrail** (run the same guardrail again if it fails, e.g. to handle transient errors). With **`on_error`**, you can treat **technical** failures differently from **policy** failures—for example, fall back to another provider when the primary API errors, while still blocking on flagged content.

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
- **Technical-error routing** — set `on_error` separately from `on_fail` so outages or timeouts can **allow**, **block**, **go to the next step**, or return a **custom response** without conflating them with content violations

## Concepts

### Pipeline

A pipeline has:

- **Mode**: `pre_call` (before the LLM) or `post_call` (after the LLM)
- **Steps**: Ordered list of guardrail steps

### Outcomes: pass, fail, and error

Each step run produces one of three outcomes:

| Outcome | Meaning | Typical cause |
|--------|---------|----------------|
| **pass** | Guardrail completed without blocking | Content allowed, or data was modified and returned |
| **fail** | Policy intervention | Guardrail raised an intervention (e.g. flagged content, blocked request) |
| **error** | Technical failure | Timeouts, network errors, guardrail not registered, or other non-intervention exceptions |

`on_pass` and `on_fail` apply to **pass** and **fail** respectively. **`on_error`** applies only to **error**. If `on_error` is omitted, the pipeline uses **`on_fail`** for error outcomes (backward compatible).

### Step actions

For each step you choose an action for **pass**, **fail**, and optionally **error**. Allowed values are: `next`, `allow`, `block`, `modify_response`.

| Action | Description |
|--------|-------------|
| **Next Step** (`next`) | Continue to the next guardrail in the pipeline |
| **Allow** (`allow`) | Stop the pipeline and allow the request to proceed |
| **Block** (`block`) | Stop the pipeline and block the request |
| **Custom Response** (`modify_response`) | Return a custom message instead of the default block |

### Step options

| Field | Type | Description |
|-------|------|--------------|
| `guardrail` | `string` | Name of the guardrail to run |
| `on_pass` | `string` | Action when outcome is **pass**: `next`, `allow`, `block`, `modify_response` |
| `on_fail` | `string` | Action when outcome is **fail** (policy intervention): `next`, `allow`, `block`, `modify_response` |
| `on_error` | `string` (optional) | Action when outcome is **error** (technical). If omitted, **error** uses `on_fail`. |
| `pass_data` | `boolean` | Forward modified request data (e.g., PII-masked) to the next step |
| `modify_response_message` | `string` | Custom message when using `modify_response` action |

## Using the Flow Builder (UI)

1. Go to **Policies** in the LiteLLM Admin UI
2. Click **+ Create New Policy** or **Edit** on an existing policy
3. Select **Flow Builder** (instead of the simple form)
4. Design your flow:
   - **Trigger** — Incoming LLM request (runs when the policy matches)
   - **Steps** — Add guardrails; set **ON PASS**, **ON FAIL**, and **ON API FAILURE** / **ON ERROR** per step (when **ON API FAILURE** is unset, technical errors follow **ON FAIL**)
   - **End** — Request proceeds to the LLM when the pipeline allows it
5. Use **+** between steps to insert another guardrail step (for fallbacks, retries, or stricter second checks)
6. Use **Test Pipeline** to run sample messages before saving
7. Click **Save Policy** (or **Save**) to create or update the policy

### Configure guardrail fallbacks in the UI (walkthrough)

1. Click **Policies**

![Policies tab in the Admin UI](https://colony-recorder.s3.amazonaws.com/files/2026-04-15/1333f4ae-d7df-4645-bd33-fee11c80cb96/ascreenshot_ce21e8bd79324c4685ad6c191e39d89e_text_export.jpeg)

2. Click **+ Add New Policy**

![Add new policy](https://colony-recorder.s3.amazonaws.com/files/2026-04-15/353c08ab-cdb5-490f-b54f-734f77c87c45/ascreenshot_223033a61071485187e87cbb8c41081e_text_export.jpeg)

3. Click **Flow Builder**

![Choose Flow Builder](https://colony-recorder.s3.amazonaws.com/files/2026-04-15/70e99d1b-fd76-4143-93f4-296b8b4c3904/ascreenshot_ef49b2e2c5dc40e39cf8da7a37f346ac_text_export.jpeg)

4. Click **Continue to Builder**

![Continue to Builder](https://colony-recorder.s3.amazonaws.com/files/2026-04-15/3de1beaf-9c52-4f03-9100-ce4d47e41967/ascreenshot_a1d64e7e58c54b6cb8a311173ffe435a_text_export.jpeg)

5. Click the **guardrail search** field on the first step

![Select first guardrail — search field](https://colony-recorder.s3.amazonaws.com/files/2026-04-15/640f699b-bdde-4e6d-a226-1fede9477b22/ascreenshot_27f14445b78b4e61872f3f95c1c9bacd_text_export.jpeg)

6. Choose **Test Moderation** (or your primary guardrail)

![Pick Test Moderation](https://colony-recorder.s3.amazonaws.com/files/2026-04-15/d46f7ab6-4231-44fb-b377-59f817cdfbe5/ascreenshot_e3a9f8e25ffe46ad82a73641b81d157c_text_export.jpeg)

7. For one branch (e.g. **ON API FAILURE**), set the action to **Next Step** so the pipeline can fall through to the next guardrail when the API errors

![Set action to Next Step](https://colony-recorder.s3.amazonaws.com/files/2026-04-15/3a7ddc2a-4317-417b-9341-ff6b0913e64b/ascreenshot_8878486dc12b4dddafe0c8ba4382a0fb_text_export.jpeg)

8. For **ON PASS**, set **Allow** (or **Next Step** if you need more steps before allowing)

![Set ON PASS to Allow](https://colony-recorder.s3.amazonaws.com/files/2026-04-15/0e31cde8-3075-4e17-b771-b2b1696db98f/ascreenshot_b4b1d232459e4941904c9fbcf90c70ca_text_export.jpeg)

9. Open the next outcome’s search/dropdown (e.g. **ON FAIL**)

![Configure another branch — search field](https://colony-recorder.s3.amazonaws.com/files/2026-04-15/715fc3ad-f245-4ee8-bb36-cc13400d635d/ascreenshot_395fece82c124d4d826fb5d84c9c0529_text_export.jpeg)

10. Set that branch to **Next Step** if failed checks should continue to your backup guardrail

![ON FAIL or branch — Next Step](https://colony-recorder.s3.amazonaws.com/files/2026-04-15/83156e9b-fc3f-4cc2-a6cb-2a13a5e77b06/ascreenshot_c61429bf7b354063afc57c40a6b45c7a_text_export.jpeg)

11. Click **+** between steps to add a second guardrail

![Add step — plus control](https://colony-recorder.s3.amazonaws.com/files/2026-04-15/e76cff13-af73-4775-90f6-4d29cb97d401/ascreenshot_52c478e7afd5410f9f63b616c753c851_text_export.jpeg)

12. Open the guardrail search field on the new step

![Second step — guardrail search](https://colony-recorder.s3.amazonaws.com/files/2026-04-15/5c1c4eea-d7da-41e5-bebd-945e97562aa5/ascreenshot_cef70e9146b148b1936e721638de0783_text_export.jpeg)

13. Select **Insults & Personal Attacks** (or your fallback / stricter guardrail)

![Pick Insults and Personal Attacks](https://colony-recorder.s3.amazonaws.com/files/2026-04-15/e796c733-351f-494f-9261-795c27f2b519/ascreenshot_f0f778d50c2146e48829ffb203c7de92_text_export.jpeg)

14. Set **Next Step** or **Block** on the branches as needed for this step

![Second step branch — Next Step](https://colony-recorder.s3.amazonaws.com/files/2026-04-15/c5fad953-4f4b-47ec-ab6d-81d21b2fb7b8/ascreenshot_b515fadec0534c6a9b9d66091398d82d_text_export.jpeg)

15. Set **ON PASS** to **Allow** when this guardrail should complete the pipeline successfully

![Second step — Allow on pass](https://colony-recorder.s3.amazonaws.com/files/2026-04-15/8210f32a-8704-41b1-97cc-7d183682a2a4/ascreenshot_23361af2b7da482a8d89025ab285a72e_text_export.jpeg)

16. Open the branch where you want a **Custom Response** (e.g. **ON FAIL** on the last step)

![Custom response — open branch selector](https://colony-recorder.s3.amazonaws.com/files/2026-04-15/98ab3a2c-f22f-4478-a146-d5d26cae9b10/ascreenshot_6a3b673654e64ce29c8c93fbf30c52ed_text_export.jpeg)

17. Choose **Custom Response**

![Select Custom Response](https://colony-recorder.s3.amazonaws.com/files/2026-04-15/a9e69e82-d517-4426-95da-034643a2388b/ascreenshot_f8ef581fbfb440cdbf145a2e9368c8e8_text_export.jpeg)

18. Click **Enter custom response...** and type your message

![Custom response text field](https://colony-recorder.s3.amazonaws.com/files/2026-04-15/ef0f90ba-d0bc-4220-874f-4998b2dcc5f6/ascreenshot_f3e825b57fa0478a92f56840af266e03_text_export.jpeg)

19. Confirm or edit the message in **Enter custom response...** as needed

![Custom response — confirm message](https://colony-recorder.s3.amazonaws.com/files/2026-04-15/f9a4711d-655c-4f15-b0ea-6b7d33fe6e60/ascreenshot_5df4b465bc484d8f86a4af5a45e9ab42_text_export.jpeg)

20. Open **Test Pipeline**

![Test Pipeline panel](https://colony-recorder.s3.amazonaws.com/files/2026-04-15/3f9ac555-66fe-43e0-a8d8-2288a5966c73/ascreenshot_b2319dae363346ebb4da5d09180b56e8_text_export.jpeg)

21. Click **Run Test**

![Run Test](https://colony-recorder.s3.amazonaws.com/files/2026-04-15/8e21e973-8193-404b-9d97-fd85be5f90b6/ascreenshot_619ca71e3be244449ca2ab01dde3cc45_text_export.jpeg)

22. Expand **Step 1** (or the first guardrail row) in the results to see **ERROR** / **Next Step** vs **PASS** / **Allow**

![Expand first step in test results](https://colony-recorder.s3.amazonaws.com/files/2026-04-15/b8010e20-dd9a-4e59-b0ca-1f2ba4c7b6ac/ascreenshot_da99f5761bbf44a08af4f1e1175a95fc_text_export.jpeg)

23. Expand **Step 2** (e.g. **Insults & Personal Attacks**) to confirm **PASS** and **Allow** after the fallback

![Expand Step 2 — second guardrail outcome](https://colony-recorder.s3.amazonaws.com/files/2026-04-15/cac5273c-dd4f-48a0-af58-12c428d0f0d0/ascreenshot_f74da58e280a47319a7d2fa41519f4fb_text_export.jpeg)

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

## Technical errors vs policy failures (`on_error`)

Use **`on_error`** when you want different behavior for **API/infra problems** than for **content policy** violations.

- **`on_fail`** — Runs when the guardrail **intervenes** (e.g. toxic content, PII detected).
- **`on_error`** — Runs when the step ends in **error** (timeout, connection failure, guardrail not loaded, etc.). If you omit `on_error`, **error** outcomes use **`on_fail`**.

Example: block on bad content, but if the primary scanner is down, fall back to a second guardrail instead of blocking every request:

```yaml
policies:
  error-fallback-policy:
    guardrails:
      add:
        - primary_scanner
        - backup_scanner
    pipeline:
      mode: pre_call
      steps:
        - guardrail: primary_scanner
          on_pass: allow
          on_fail: block
          on_error: next
        - guardrail: backup_scanner
          on_pass: allow
          on_fail: block
          on_error: allow
```

If `primary_scanner` errors → run `backup_scanner`. If `backup_scanner` errors → allow the request (set `on_error` to `block` if you prefer fail-closed).

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
