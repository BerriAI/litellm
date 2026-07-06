# Compresr guardrail for LiteLLM — what we built, how to try it, how we maintain it

**Branch:** `litellm_compresr_cleanup` on [`Ousso11/litellm`](https://github.com/Ousso11/litellm/tree/litellm_compresr_cleanup) (commit `6caaacc55d`)
**Base:** `litellm_internal_staging` (current as of 2026-07-05, includes everything upstream `main` has)
**Status:** implemented, all repo lint/type gates green, 45/45 unit tests passing, verified live against the real Compresr API

---

## 1. What this is

A first-class LiteLLM **guardrail** that compresses bulky message content (tool outputs, RAG chunks, search results) through the Compresr API before the request reaches the LLM — and, unlike every other compression integration we know of, makes it **recoverable**: the model can pull the original content back on demand via an injected tool.

It's the "best of both" between our existing SDK-side guardrail (`compresr.integrations.litellm`, shipped in the `compresr` pip package) and the Headroom guardrail that LiteLLM merged upstream in June 2026:

| Aspect | Taken from | Detail |
|---|---|---|
| Hook surface | Headroom | `apply_guardrail` + `structured_messages` — fires on `/chat/completions`, `/v1/messages` (Anthropic), **and** `/v1/responses`. Our old SDK guardrail used `async_pre_call_hook`, which misses the Responses API. |
| Transport | Headroom | Plain httpx via LiteLLM's `get_async_httpx_client`. **Zero pip dependencies** — no `pip install compresr` needed. |
| Query-aware compression | Ours | Each message is compressed against the *intent* that produced it: for a tool output, the originating tool call's `name + arguments` (resolved via `tool_call_id`); otherwise the last user message. Headroom compresses the whole message list blind. |
| Target selection | Ours | Tool/function outputs by default; system prompt, prior user history, and the last user message are opt-in; messages under 500 chars skipped; multimodal messages have text parts compressed and images/audio passed through untouched. |
| Recovery (new) | Neither — new design | Headroom stores originals **server-side** and retrieves over HTTP. We don't need a server: the guardrail already holds the original text (we sent it to the API ourselves), so originals are cached in-process and served back through LiteLLM's agentic loop. No backend changes required. |
| Bypass header | Headroom | `x-compresr-bypass: true` skips compression per request. |
| Failure policy | Both | `unreachable_fallback: fail_closed` (default, request 502s if Compresr is down) or `fail_open` (logs critical, forwards uncompressed). |

### How recovery works (the cool part)

1. During `apply_guardrail`, each compressed message gets a marker appended:

   ```
   [compresr hash=810a5b8d17dd56ddf95396cb: parts of this content were compressed away.
    If you need the full original, call the compresr_retrieve tool with this hash.]
   ```

   The original text is stored in-process under `sha256(original)[:24]`, scoped to this request's `litellm_call_id`, with a 15-minute TTL and a hard cap of 256 tracked calls.

2. A `compresr_retrieve` tool is merged into the request's tool list.

3. If the model calls it, LiteLLM's agentic-loop hooks (`async_should_run_agentic_loop` / `async_build_agentic_loop_plan`) answer the tool call with the stored original and re-run the request. One extra LLM round-trip, and only when the model actually asks. Follow-up message shapes are handled for all three API styles (chat tool-role, Anthropic `tool_use`/`tool_result`, Responses `function_call`/`function_call_output`).

4. Security: a hash is only honored for the request that issued it. A hash pasted in from another conversation (or forged in a prompt) returns a not-found message, never content. This matters because the store is shared proxy memory.

Set `enable_retrieval: false` in `optional_params` to get plain lossy compression with no marker and no tool.

## 2. Exact changes

Seven files, +2,375 / −17:

| File | Lines | What it does |
|---|---|---|
| `litellm/proxy/guardrails/guardrail_hooks/compresr/compresr.py` | 837 | The guardrail: target selection, query derivation, dependency-free API client (`POST /api/compress/question-specific/` and `/batch`, `X-API-Key` auth), recovery store, agentic-loop hooks, stats logging via `add_standard_logging_guardrail_information_to_request_data`. |
| `litellm/proxy/guardrails/guardrail_hooks/compresr/__init__.py` | 69 | `initialize_guardrail` + the two registries LiteLLM's discovery walk looks for. No central registration needed — the proxy discovers guardrail packages by walking `guardrail_hooks/`. |
| `litellm/proxy/guardrails/guardrail_hooks/compresr/README.md` | 79 | Quickstart + config reference (Headroom shipped no docs at all). |
| `litellm/types/proxy/guardrails/guardrail_hooks/compresr.py` | 78 | Pydantic config model (drives YAML validation and the admin-UI form). UI name: "Compresr (context compression)". |
| `litellm/types/guardrails.py` | +39/−17 | `COMPRESR = "compresr"` enum entry, config-model import + registration in `LitellmParams`, `'compresr'` added to the `unreachable_fallback` docstring. (Most of the churn is ruff-strict import re-sorting.) |
| `tests/test_litellm/proxy/guardrails/guardrail_hooks/test_compresr.py` | 1036 | 45 tests mirroring the structure of `test_headroom.py`. |

Defaults, all overridable in YAML: model `latte_v2`, `target_compression_ratio 0.5`, `coarse true`, `min_chars_to_compress 500`, `enable_retrieval true`, API base `https://api.compresr.ai` (env `COMPRESR_API_BASE` for on-prem), key from `COMPRESR_API_KEY`.

## 3. Try it

### Setup (5 minutes)

```bash
git clone --filter=blob:none --branch compresr-integration git@github.com:charafkamel/litellm.git
cd litellm
python3 -m venv .venv
.venv/bin/pip install -e ".[proxy]"
export COMPRESR_API_KEY=cmp_...        # ask Kamel for a key
```

### Example A — see the compression with your own eyes (no LLM key needed)

Uses a tiny fake upstream that records exactly what the proxy forwards, so you can diff original vs compressed. Grab `echo_upstream.py` and `websearch_request.json` from `Compresr-SDK-Private/python/tutorial/litellm/demo/` (or ask me for them). Note: `Compresr-SDK-Private/` is a private repository — these demo files are not required to run the guardrail itself.

`config.yaml`:

```yaml
model_list:
  - model_name: demo-model
    litellm_params:
      model: openai/demo-model
      api_base: http://127.0.0.1:8199/v1
      api_key: fake-upstream-key

guardrails:
  - guardrail_name: compresr
    litellm_params:
      guardrail: compresr
      mode: pre_call
      default_on: true

general_settings:
  master_key: sk-demo-1234
```

Run:

```bash
python3 echo_upstream.py &                                   # fake upstream on :8199
.venv/bin/litellm --config config.yaml --port 4100 &         # proxy with the guardrail

curl -s http://127.0.0.1:4100/v1/chat/completions \
  -H "Authorization: Bearer sk-demo-1234" \
  -H "Content-Type: application/json" \
  -d @websearch_request.json | head -c 300

python3 -c "
import json
orig = json.load(open('websearch_request.json'))
fwd  = json.load(open('forwarded_request.json'))   # written by echo_upstream
o = next(m for m in orig['messages'] if m['role']=='tool')['content']
f = next(m for m in fwd['messages']  if m['role']=='tool')['content']
print(f'tool output: {len(o)} chars -> {len(f)} chars')
print('retrieve tool injected:', any(t['function']['name']=='compresr_retrieve' for t in fwd.get('tools',[])))
"
```

What we measured on this exact request: `3396 chars -> 2077 chars` (39% smaller), recovery marker present, `compresr_retrieve` injected, and the user question + system prompt byte-identical to the original.

### Example B — bypass header

```bash
curl -s http://127.0.0.1:4100/v1/chat/completions \
  -H "Authorization: Bearer sk-demo-1234" \
  -H "Content-Type: application/json" \
  -H "x-compresr-bypass: true" \
  -d @websearch_request.json > /dev/null
```

`forwarded_request.json` now shows the tool output untouched and no injected tool. Verified live.

### Example C — recovery loop with a real model

Point the model at a real provider and ask for a detail that compression is likely to drop:

```yaml
model_list:
  - model_name: demo-model
    litellm_params:
      model: openai/gpt-5-mini
      api_key: os.environ/OPENAI_API_KEY
```

Then send a conversation whose tool output buries a precise figure, and ask for that figure verbatim. If the compressed version kept it, you get a direct answer (cheap path). If it didn't, the model calls `compresr_retrieve`, LiteLLM runs the agentic round-trip transparently, and the final response comes back grounded in the full original — your client sees one normal response either way. Watch the proxy logs with `--detailed_debug` to see the `Compresr retrieve: hash=... -> N chars` line when the loop fires. Note the loop triggers only when the model decides it needs more; adding "If content was compressed away and you're missing a detail, retrieve it before answering" to the system prompt makes it eager.

### Example D — tuning

```yaml
guardrails:
  - guardrail_name: compresr
    litellm_params:
      guardrail: compresr
      mode: pre_call
      default_on: true
      model: latte_v2
      unreachable_fallback: fail_open     # never block traffic on a Compresr outage
      optional_params:
        target_compression_ratio: 4       # >1 means Nx smaller; 0-1 means fraction removed
        coarse: false                     # token-level instead of paragraph-level
        compress_system: true             # also compress long system prompts
        enable_retrieval: false           # plain lossy mode
```

### Run the tests and gates

```bash
.venv/bin/python -m pytest tests/test_litellm/proxy/guardrails/guardrail_hooks/test_compresr.py -q   # 37 passed
.venv/bin/pip install ruff basedpyright
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python scripts/ruff_strict_gate.py    --base origin/litellm_internal_staging
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python scripts/type_discipline_gate.py --base origin/litellm_internal_staging
PATH="$PWD/.venv/bin:$PATH" bash -c '(basedpyright --outputjson || true) | python scripts/type_check_gate.py --base origin/litellm_internal_staging'
```

All three pass on this branch. The wider `tests/test_litellm/proxy/guardrails/` folder has 58 pre-existing failures (prisma/DB fixtures missing in a bare env); we verified they reproduce identically on the clean base, so they are not ours.

## 4. Known limitations (honest list)

### ⚠️ Multi-worker deployments

**Originals are stored in process memory.** When you run LiteLLM with multiple workers (`gunicorn`/`uvicorn --workers N` where N > 1), pre-call and post-call (retrieval) hooks can land on **different worker processes**. The worker that compressed the message never shares its in-process store with the worker that later handles the `compresr_retrieve` tool call. The model silently receives a not-found response instead of the original content.

**Mitigation options:**
- Run with `--workers 1` (default for the LiteLLM proxy — recommended unless you need concurrency at the proxy layer).
- Set `enable_retrieval: false` in `optional_params` to use plain lossy compression with no recovery store, no tool injection, and no cross-worker problem.
- Future: Redis-backed originals store (tracked in maintenance plan below).

This constraint is documented in the module docstring of `compresr.py` near the `_originals_by_call_id` store.

- **The recovery store is per-process.** Single proxy instance: fine (retrieval happens seconds after compression, within the same conversation turn). Multi-replica without sticky routing: a retrieve can land on a pod that never saw the compress, and the model gets a polite not-found. Same limitation Headroom's own hash-validity store has. Redis-backed storage is the known fix if it ever matters.
- **`source` analytics tag is `gateway:unknown`.** The platform API validates `source` against an enum and has no `gateway:litellm` member yet — the live test caught this (422). Backend should add one; one-line change here afterwards.
- **Streaming responses:** compression happens pre-call so streaming requests are compressed fine, but the recovery agentic loop on a *streamed* tool-call response follows whatever LiteLLM's loop supports — we have unit coverage, not live-stream coverage.
- **Marker collision:** our marker regex is `compresr hash=<24hex>`, deliberately distinct, but Headroom's looser `hash=<24hex>` regex would also match inside our marker. Only matters if someone runs both guardrails on the same request, which nobody should.

## 5. Maintenance plan

**Ownership.** This lives on our fork until we decide to PR it upstream. Treat `compresr-integration` as the integration trunk; branch off it for changes, land via PR into it (base PRs on `litellm_internal_staging` only if/when we go upstream — that repo's convention is never to base on `main`).

**Weekly (or before any demo): rebase + gates.**

```bash
git fetch origin litellm_internal_staging
git rebase origin/litellm_internal_staging compresr-integration
# then: pytest test_compresr.py + the three gate commands above
```

The three things upstream churn can break, in likelihood order: (1) the agentic-loop hook signatures in `litellm/integrations/custom_logger.py` — new, still evolving (Headroom needed 4 PRs in one week); (2) the `guardrail_translation` handlers that feed `structured_messages`; (3) `LitellmParams`/config-model plumbing in `types/guardrails.py`. Cheap tripwire: `git log --oneline <old>..origin/litellm_internal_staging -- litellm/integrations/custom_logger.py litellm/llms/*/chat/guardrail_translation/ litellm/proxy/guardrails/guardrail_hooks/headroom/` after each fetch — if Headroom's guardrail changed, we probably want the same change.

**Sync with our backend.** The guardrail hardcodes the two endpoint paths and the response envelope (`data.compressed_context`, `data.results[]`). Any platform API change to `/api/compress/question-specific/*` must update `_call_compress` and the test fixtures in the same PR. Pending backend asks: add `gateway:litellm` to the `source` enum; optionally a keyed-placeholder mode so markers can sit inline where content was dropped instead of appended.

**Testing debt, in priority order:** (1) an `initialize_guardrail` test that feeds a real `LitellmParams` and asserts every optional_param lands on the constructor — that mapping is currently untested here *and* in our SDK package, and it's where config silently rots; (2) fail-open variants for the batch path; (3) a live-stream recovery test. Target: parity with Headroom's 43 tests before any upstream PR.

**Relationship to the SDK package.** `compresr.integrations.litellm` (pip) still exists and works on older LiteLLM via the shim/CLI. Decision made: new features (recovery, Responses API coverage) go **here first**; the SDK package is maintenance-only until we decide whether to port this back or deprecate it in favor of the upstream path. Don't implement features twice.

**Path to upstream.** When we're ready: rebase, rename base to a fresh branch off `litellm_internal_staging`, bring the test count to Headroom parity, write the PR body per `.github/pull_request_template.md` with a live curl proof (their maintainers explicitly want real-traffic proof, not pytest output — we already have the workflow for this), and include the README as the docs story Headroom never shipped. The precedent is exactly PR #31407 + #31681 (Headroom, merged within days, maintainer-authored follow-ups) — our footprint is the same shape and smaller blast radius.
