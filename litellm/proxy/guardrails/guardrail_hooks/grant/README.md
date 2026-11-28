# Grant guardrail for LiteLLM · upstream PR package

**Target repo:** `BerriAI/litellm`
**Proposed path:** `litellm/proxy/guardrails/guardrail_hooks/grant/grant_guardrail.py`
**Tests:** `litellm/proxy/guardrails/guardrail_hooks/grant/test_grant_guardrail.py`

Grant pauses a tool call returned by any LLM until a human approves it. No wrapper, no new dependency, no custody of credentials or arguments.

## Behavior

| Grant response | Guardrail behavior |
| --- | --- |
| `status: approved` (rule / Always allow) | Response passes through untouched after one `POST /v1/approvals`; target <150ms. |
| `status: pending`, `approval_mode: checkpoint` | Raises `GrantApprovalRequired` with `request_id`, `approval_url`, `receipt_url`, `expires_at`. The caller shows the Grant popup immediately. |
| `status: pending`, `approval_mode: blocking` | Polls `GET /v1/approvals/{request_id}` until decided, approved, rejected, or expired. |
| Grant unreachable / 5xx | Fails closed unless `fail_open: true`. |

### Tool-call formats gated by the same checkpoint

The guardrail normalizes and gates every protected call from:

- OpenAI `choices[].message.tool_calls` (including streamed deltas)
- Legacy OpenAI `choices[].message.function_call`
- Anthropic `/v1/messages` `content[].type == "tool_use"` blocks (including raw SSE)

### Streaming policy

`stream: true` is rejected before the model call by `async_pre_call_hook` unless `allow_streaming: true` is explicitly configured:

```text
GrantGuardrail does not permit unbuffered stream=True; set stream=False or disable streaming for protected tool calls
```

When `allow_streaming: true` is set, `async_post_call_streaming_iterator_hook` buffers the entire stream, reconstructs tool calls, runs the identical Grant gate, and only then yields chunks. No chunk is released before the decision. The guardrail automatically registers both `pre_call` and `post_call` hooks so the rejection and the buffered gate both run.

### Zero-custody scope payload

Grant receives only:

- `tool_name`
- `tool_call_id`
- `argument_key_count`
- `arguments_hmac_sha256`

Raw argument keys, argument values, credentials, payment details, and PII are never transmitted. `arguments_hmac_sha256` is an HMAC-SHA256 digest of canonicalized arguments keyed by, in precedence order:

1. `GRANT_LOCAL_HMAC_SECRET`
2. `GRANT_API_KEY`
3. An ephemeral per-process secret (stable for retries in that process only)

Set `GRANT_LOCAL_HMAC_SECRET` explicitly if Grant must correlate identical retries across restarts without using the API key as the HMAC key.

### Base URL resolution

`grant_base_url` is resolved strictly in this order:

1. Explicit `grant_base_url` config
2. `GRANT_BASE_URL`
3. `GRANT_PUBLIC_BASE_URL`
4. `https://grant-approve.vercel.app`

`.vercel/project.json` is never read. A file in the proxy working directory cannot redirect approval traffic or leak the configured API key to an unintended host.

## Copy-paste `config.yaml`

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: grant-approval
    litellm_params:
      guardrail: litellm.proxy.guardrails.guardrail_hooks.grant.grant_guardrail.GrantGuardrail
      mode: post_call
      protected_tools:
        - send_email
        - issue_refund
        - create_charge
        - transfer_funds
        - publish_content
      grant_base_url: https://grant-approve.vercel.app
      api_key: os.environ/GRANT_API_KEY
      workspace: production
      approval_mode: checkpoint   # or: blocking
      timeout: 2.0
      poll_interval: 1.0
      poll_timeout: 300.0
      fail_open: false
      allow_streaming: false      # default; true enables buffered, gated streaming
```

Start with:

```bash
GRANT_LOCAL_HMAC_SECRET="$(openssl rand -hex 32)" litellm --config config.yaml
```

Request-level opt-in (when `default_on` is false):

```json
{
  "model": "gpt-4o",
  "messages": [{"role": "user", "content": "Draft and send a refund note"}],
  "guardrails": ["grant-approval"]
}
```

`approval_mode: blocking` waits for the human; `checkpoint` returns the approval URL so the product can render the popup and retry the call after approval.

## Running the tests

```bash
python -m pytest integrations/litellm/test_grant_guardrail.py -q
```

The suite covers URL precedence and ignored `.vercel/project.json`, no tool calls, unprotected tools, the approved fast path under 150ms with no raw argument leakage, checkpoint approval, blocking approve/reject, wildcard and fail-closed/fail-open, invalid configuration, streaming rejection, buffered streaming enforcement, all three tool-call formats, and HMAC key precedence/determinism. `httpx` is already bundled with LiteLLM, so the test suite adds no dependency.

## PR checklist for BerriAI/litellm

1. Fork and branch from `main`: `git checkout -b feat/grant-approval-guardrail`.
2. Copy `grant_guardrail.py` and `test_grant_guardrail.py` into the proposed directory above; add an empty `__init__.py`.
3. Register the guardrail in `litellm/proxy/guardrails/guardrail_hooks/__init__.py` and, if the repo uses a registry map, add `grant_guardrail` there. Do not add a dependency to `pyproject.toml` or `requirements.txt`.
4. Add a docs page under `docs/my-website/docs/proxy/guardrails/` using the `config.yaml` snippet above, and link it from the guardrails index.
5. Run `python -m pytest litellm/proxy/guardrails/guardrail_hooks/grant/test_grant_guardrail.py -q` and the repo's lint target.
6. Open the PR at **https://github.com/BerriAI/litellm/pulls** titled *Add Grant zero-custody human approval post-call guardrail*. State clearly that the network target is `https://grant-approve.vercel.app`, that the gate is optional/default-off, that streaming is rejected by default, and that no credentials, raw arguments, or raw argument keys are transmitted.
7. Do not post fabricated benchmarks, paid reviewers, or fake upvotes. The included <150ms assertion is a local mock-transport test, not a network benchmark.
