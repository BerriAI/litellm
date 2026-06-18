# External Batch E2E Suite

End-to-end tests for the **/files** + **/batches** lifecycle against a **live,
published LiteLLM proxy image**, driven by the real OpenAI SDK with an external
user's API key. Nothing is mocked.

> **This suite is NOT part of CI/CD.** No GitHub workflow references this folder,
> and every test is gated behind the `batch_e2e` marker. It is meant to be run
> by an external test suite after an image is published.

## What it verifies

For every configured **case** (a *routing strategy* × *provider*) it walks the
full lifecycle and asserts each step:

```
create file (purpose="batch")
  -> create batch
  -> retrieve batch
  -> list batches
  -> (optional) await completion + fetch output
  -> cancel batch
  -> delete file
```

### Three routing scenarios

These map 1:1 to the proxy's file-create routing priority in
`litellm/proxy/openai_files_endpoints/files_endpoints.py`:

| Strategy | How it routes | Notes |
|---|---|---|
| `model_param` | `model` in the request body | Proxy encodes the model into the file/batch id; downstream calls need no extra hint. No DB. |
| `target_model_names` | Managed files, load-balanced across the named models | **Requires a DB-backed deployment.** |
| `custom_llm_provider` | `custom-llm-provider` header, repeated on every call | No DB. |

## Hard-fail contract

This suite exists to catch client-level regressions, so it does **not** skip:

- A supported op **must succeed**. Any failure fails the test — including a
  missing DB for the `target_model_names` scenario.
- Misconfiguration (missing/!malformed env) **fails the session**, it is not skipped.
- An op declared unsupported for a provider (`expected_unsupported`) **must raise
  the declared error**. Success there, or a different error, is a failure. This
  is how a documented contract gap is told apart from a regression.

## Running

```bash
export LITELLM_E2E_BASE_URL="https://my-published-proxy.example.com"
export LITELLM_E2E_API_KEY="sk-..."         # the user-specific key
export LITELLM_E2E_CONFIG="@./e2e_config.json"   # inline JSON also accepted

pytest -m batch_e2e tests/batch_e2e_external
```

## Configuration (`LITELLM_E2E_CONFIG`)

Either inline JSON or `@/path/to/file.json`.

```jsonc
{
  "endpoint": "/v1/chat/completions",   // default
  "await_completion": false,            // poll to terminal + fetch output (slow)
  "completion_timeout_s": 900,
  "cases": [
    {
      "strategy": "custom_llm_provider",
      "provider": "openai",
      "request_model": "gpt-4o"          // model written into the .jsonl body
    },
    {
      "strategy": "model_param",
      "provider": "azure",
      "model": "azure-gpt-4o",           // a model name in the deployment's model_list
      "request_model": "gpt-4o"
    },
    {
      "strategy": "target_model_names",
      "provider": "vertex_ai",
      "target_model_names": "gemini-batch-group",
      "request_model": "gemini-1.5-flash-001",
      "expected_unsupported": {          // declared, sanctioned contract gaps
        "cancel": { "status": 400, "match": "not supported" }
      }
    }
  ]
}
```

### `cases[]` fields

| Field | Required | Meaning |
|---|---|---|
| `strategy` | yes | `model_param` \| `target_model_names` \| `custom_llm_provider` |
| `provider` | yes | Provider label (used in the test id and for `custom_llm_provider`). |
| `request_model` | yes | Model string written into the generated `.jsonl` request body. |
| `model` | for `model_param` | Model name present in the deployment's `model_list`. |
| `target_model_names` | for `target_model_names` | Managed-files model group name. |
| `expected_unsupported` | no | Map of `op -> { status?, match? }`. Ops: `file`, `create`, `retrieve`, `list`, `cancel`, `delete`. |

Case ids are `"{strategy}-{provider}"` and must be unique.
