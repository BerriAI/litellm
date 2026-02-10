---
slug: model-cost-map-incident
title: "Incident Report: Broken Model Cost Map on main"
date: 2026-02-10T10:00:00
authors:
  - name: Ishaan Jaffer
    title: "CTO, LiteLLM"
    url: https://www.linkedin.com/in/ishaanjaffer/
    image_url: https://pbs.twimg.com/profile_images/1298587542745358340/DZv3Oj-h_400x400.jpg
tags: [incident-report, stability, model-cost-map]
---

# Incident Report: Broken Model Cost Map on `main`

## What happened?

2 weeks ago a contributor PR with changes to the model cost map had a poorly formatted JSON entry. When this was merged into `main` ([commit `562f0a0`](https://github.com/BerriAI/litellm/commit/562f0a028251750e3d75386bee0e630d9796d0df)) it led to the following error message reported from users.

Users found that their code started erroring out with the following message:

```
{"type":"error","error":"This model isn't mapped yet. model=gpt-5.2, custom_llm_provider=azure. Add it here - https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json."}
```

The bad commit added an extra `{` bracket at line 24258 of `model_prices_and_context_window.json`, making the entire file invalid JSON:

```
json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes: line 24258 column 5
```

---

## What caused this type of exception?

### How the model cost map loading works

1. At **import time**, `litellm` fetches the model cost map from GitHub `main`:
   `https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json`
2. If the fetch fails (network error, invalid JSON, etc.), it **silently** falls back to a local backup file (`model_prices_and_context_window_backup.json`) bundled with the installed package
3. There is **no log or warning** when the fallback occurs -- users have no way to know they're running on stale data

### What happened in this incident

1. The bad commit broke the JSON on `main`
2. Every litellm installation (not just a specific version) fetches from `main` at import time
3. `response.json()` threw `JSONDecodeError`, caught by `except Exception`
4. Silently fell back to the local backup, which is pinned to the installed package version
5. A customer on v1.80.5 had a backup missing **661+ newer models** including `azure/gpt-5.2`
6. Any call to `get_model_info("azure/gpt-5.2")` raised `"This model isn't mapped yet"`
7. This affected **all litellm users** (not just those on a specific version), since every installation fetches the remote JSON from `main` at import time

### Impact matrix

| Call Path | Worked? | Error |
|---|---|---|
| `litellm.completion(model="azure/gpt-5.2")` | Yes -- model map error caught silently in debug logs, request still sent to Azure | |
| `litellm.completion(model="azure/gpt-5.2", stream=True)` | Yes -- same behavior, streams fine | |
| `litellm.get_model_info("azure/gpt-5.2")` | No | `This model isn't mapped yet. model=gpt-5.2, custom_llm_provider=azure.` |
| Proxy routing (request forwarding) | Yes -- routes based on config `model_list`, not cost map | |
| Proxy cost tracking / spend logging | No | `get_model_info()` fails in cost calculation callbacks, error surfaces in logging |
| Proxy `/model/info` endpoint | Partial | Returns default values (0 cost, null limits) for unmapped models |

### Key finding

`litellm.completion()` **never blocks** on a missing model in the cost map. It catches the `get_model_info()` error and proceeds to the API. The customer error surfaced from the **cost tracking / logging callbacks** path, where `get_model_info()` is called for spend calculation.

---

## Root cause analysis

### Why did this happen?

- No CI tests validating the JSON structure of `model_prices_and_context_window.json` before merge
- The fallback in `get_model_cost_map()` is **completely silent** -- no log, no metric, no warning
- The backup file can be arbitrarily stale depending on the installed package version

### Why fetch from GitHub?

To get live day-0 model updates (new model pricing, context windows) without requiring a package upgrade. This is valuable but creates a hard dependency on the correctness of a file on `main`.

---

## Shippable improvements

| # | Improvement | Status | Details |
|---|---|---|---|
| 1 | **CI validation for model cost map JSON** | Shipped | [PR #20605](https://github.com/BerriAI/litellm/pull/20605) -- Validates JSON schema + structure on every PR that touches `model_prices_and_context_window.json` |
| 2 | **Warning logging on fallback** | Shipped | `get_model_cost_map()` now logs a `WARNING` when the remote fetch fails and falls back to the local backup, instead of silently swallowing the error. See `litellm/litellm_core_utils/get_model_cost_map.py` |
| 3 | **Fetched JSON integrity validation** | Shipped | New `validate_model_cost_map()` helper in `litellm/litellm_core_utils/get_model_cost_map.py` checks: (a) fetched map is a dict, (b) has minimum model count, (c) hasn't shrunk >50% vs backup. If any check fails, falls back to backup with a warning |
| 4 | **CI/CD test for bad cost map resilience** | Shipped | `tests/llm_translation/test_model_cost_map_resilience.py` -- 13 tests covering: empty map, invalid JSON, network errors, shrinkage detection, `get_model_info()` error messages, and `litellm.completion()` resilience |
| 5 | Keep backup file in sync on every release | Planned | Update `model_prices_and_context_window_backup.json` as part of the release process so fallback data is never more than 1 release behind |
| 6 | `LITELLM_LOCAL_MODEL_COST_MAP=True` as default for production | Planned | Eliminates runtime GitHub dependency. Users who want live updates can opt in |
| 7 | Health check endpoint for external deps | Planned | Proxy endpoint (e.g., `/health/dependencies`) reporting status of all external fetches |

---

## Other upstream dependencies in the codebase

The model cost map is not the only external dependency that could impact LLM calls. Here is a full audit:

### Critical (can block LLM calls or auth)

| Dependency | File | URL | When Fetched | Fallback | Silent? |
|---|---|---|---|---|---|
| **Model cost map** | `litellm/__init__.py` | `raw.githubusercontent.com/.../model_prices_and_context_window.json` | Import time | Local backup file | Yes (now logs warning) |
| **Model cost map reload** | `litellm/proxy/proxy_server.py` | Same as above | Runtime (scheduled / manual) | Keeps existing map | No (logged) |
| **JWT public keys** | `litellm/proxy/auth/handle_jwt.py` | Configurable via `JWT_PUBLIC_KEY_URL` | Runtime (on-demand, cached with TTL) | **None -- raises exception** | No (exception) |
| **OIDC UserInfo** | `litellm/proxy/auth/handle_jwt.py` | Configurable via `oidc_userinfo_endpoint` | Runtime (on-demand, cached 300s) | **None -- raises exception** | No (exception) |

### Medium impact

| Dependency | File | URL | When Fetched | Fallback | Silent? |
|---|---|---|---|---|---|
| **HuggingFace provider mapping** | `litellm/llms/huggingface/common_utils.py` | `huggingface.co/api/models/{model}` | Runtime (on-demand, LRU cached) | Raises `HuggingFaceError` | No |

### Low impact (non-blocking)

| Dependency | File | URL | When Fetched | Fallback | Silent? |
|---|---|---|---|---|---|
| **Ollama model tags** | `litellm/llms/ollama/common_utils.py` | `{ollama_base}/api/tags` (localhost) | Runtime (on-demand) | Static model list | Warning logged |
| **Together AI model info** | `litellm/litellm_core_utils/prompt_templates/factory.py` | `api.together.xyz/models/info` | Runtime (on-demand) | Returns `None` | Yes (silent) |
| **AssemblyAI transcript polling** | `litellm/proxy/pass_through_endpoints/...` | `api.assemblyai.com/v2/transcript/{id}` | Runtime (on-demand) | Returns `None` | Logged |

---

## How to prevent this type of error again

### Shipped fixes

1. **JSON validation CI check** ([PR #20605](https://github.com/BerriAI/litellm/pull/20605)) -- Runs `json.loads()` and schema validation on `model_prices_and_context_window.json` on every PR. This would have caught the bad commit before merge.

2. **Warning logging on fallback** -- When `get_model_cost_map()` falls back to the backup, it now logs a `WARNING`:
   ```
   LiteLLM: Failed to fetch remote model cost map from <url>: <error>. Falling back to local backup.
   ```

3. **Fetched JSON integrity validation** -- New `validate_model_cost_map()` helper validates the fetched map before using it. Catches: non-dict responses, empty maps, and maps that have shrunk significantly compared to the backup.

4. **CI/CD resilience tests** -- 13 tests in `tests/llm_translation/test_model_cost_map_resilience.py` that simulate bad upstream, bad backup, and verify `litellm.completion()` and `litellm.get_model_info()` behavior.

### Planned improvements

5. **Keep the backup file in sync** -- Update `model_prices_and_context_window_backup.json` more frequently (e.g., on every release) so the fallback has recent models.

6. **Consider `LITELLM_LOCAL_MODEL_COST_MAP=True` as default for production** -- This eliminates the runtime dependency on GitHub entirely. Users who want live updates can opt in.

7. **Audit all upstream dependencies** -- Apply the same resilience patterns (fallback, logging, validation) to the other external dependencies listed above, especially:
   - JWT public key fetch (no fallback today -- should add retry + caching)
   - OIDC UserInfo fetch (no fallback today -- should add graceful degradation)

8. **Add health check for external dependencies** -- A proxy endpoint (e.g., `/health/dependencies`) that reports the status of all external fetches: whether the model cost map was loaded from remote or backup, whether JWT keys are fresh, etc.
