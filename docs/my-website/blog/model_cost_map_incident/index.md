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

A contributor PR with changes to the model cost map had a poorly formatted JSON entry (extra `{` bracket). When this was merged into `main` ([commit `562f0a0`](https://github.com/BerriAI/litellm/commit/562f0a028251750e3d75386bee0e630d9796d0df)) the remote `model_prices_and_context_window.json` became invalid JSON. Since `litellm` fetches this file from GitHub `main` at import time, every installation silently fell back to its local backup copy. Customers on older versions had backups missing newer models.

**Impact:**

- **SDK calls (`litellm.completion`)** -- Worked. The SDK catches model map errors internally and proceeds to the API.
- **AI Gateway calls (proxy routing)** -- Worked. The proxy routes based on its config `model_list`, not the cost map.
- **Cost tracking, `get_model_info` on SDK and proxy** -- Impacted for users relying on the model cost map. `get_model_info()` raised `"This model isn't mapped yet"` for models missing from the stale backup. The incident lasted ~20 minutes until we fixed it.

---

## What caused this?

1. At import time, `litellm` fetches `model_prices_and_context_window.json` from GitHub `main`
2. If the fetch fails (network error, invalid JSON, etc.), it **silently** falls back to a local backup bundled with the installed package
3. The bad commit broke the JSON on `main` -- every litellm installation hit the fallback
4. Customers on older versions (e.g. v1.80.5) had backups missing 661+ newer models including `azure/gpt-5.2`
5. Any call to `get_model_info("azure/gpt-5.2")` then raised `"This model isn't mapped yet"`

**Root cause:** No CI validation on the JSON file, and the fallback was completely silent -- no log, no warning.

---

## Shippable improvements

| # | Improvement | Status | Details |
|---|---|---|---|
| 1 | **CI validation for model cost map JSON** | Shipped | [PR #20605](https://github.com/BerriAI/litellm/pull/20605) -- Validates JSON schema + structure on every PR |
| 2 | **Warning logging on fallback** | Shipped | `get_model_cost_map()` now logs a `WARNING` when falling back to backup instead of silently swallowing the error |
| 3 | **Fetched JSON integrity validation** | Shipped | `GetModelCostMap.validate_model_cost_map()` checks fetched map is a dict, has minimum model count, and hasn't shrunk >50% vs backup |
| 4 | **CI/CD resilience tests** | Shipped | `tests/llm_translation/test_model_cost_map_resilience.py` -- 13 tests for empty map, invalid JSON, network errors, shrinkage, and `litellm.completion()` resilience |
| 5 | Keep backup file in sync on every release | Planned | Update backup as part of release so fallback data is never more than 1 release behind |
| 6 | `LITELLM_LOCAL_MODEL_COST_MAP=True` as default for production | Planned | Eliminates runtime GitHub dependency entirely |
| 7 | Health check endpoint for external deps | Planned | Proxy endpoint reporting status of all external fetches |

---

## Other upstream dependencies in the codebase

| Dependency | Impact | Fallback | Silent? |
|---|---|---|---|
| **Model cost map** (GitHub `main`) | Critical -- cost tracking breaks | Local backup file | Now logs warning |
| **JWT public keys** (`JWT_PUBLIC_KEY_URL`) | Critical -- auth breaks | None (raises exception) | No |
| **OIDC UserInfo** (`oidc_userinfo_endpoint`) | Critical -- auth breaks | None (raises exception) | No |
| **HuggingFace provider mapping** (`huggingface.co/api`) | Medium -- HF calls fail | Raises `HuggingFaceError` | No |
| **Ollama model tags** (localhost) | Low | Static model list | Warning logged |
| **Together AI model info** (`api.together.xyz`) | Low | Returns `None` | Silent |
