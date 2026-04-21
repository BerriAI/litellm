# Batch 01 — Claude / Anthropic compatibility

**Commits:** 6 (reclassified — `3be74052a5` moved to batch 02 during audit; it's a GCS-logger fix not a Claude-compat fix)
**Scope:** Drop unsupported `prompt-caching-scope-*` headers for Claude Code compatibility; map reasoning fields across providers; fix content-block handling for anthropic-adapter responses.

## Upstream landscape (v1.81.3 → v1.83.3)

| File | # upstream commits | Notes |
|---|---|---|
| `litellm/llms/anthropic/experimental_pass_through/messages/transformation.py` | 19 | Moderate churn |
| `litellm/types/llms/anthropic.py` | 12 | Moderate |
| `litellm/llms/vertex_ai/vertex_ai_partner_models/anthropic/transformation.py` | 5 | Low |
| `litellm/proxy/litellm_pre_call_utils.py` | 51 | High |
| `litellm/litellm_core_utils/streaming_handler.py` | 34 | High |
| `litellm/llms/openai/chat/gpt_transformation.py` | 9 | Low |
| `litellm/llms/openai/openai.py` | 10 | Low |
| `litellm/llms/anthropic/experimental_pass_through/adapters/transformation.py` | 23 | Moderate |

### Upstream equivalents found (drop-gate checks)

- **`8363a26d2e`** (PR #20058) — "Fix: remove unsupported prompt-caching-scope-2026-01-05 header for vertex ai". Fixes file `litellm/llms/vertex_ai/vertex_ai_partner_models/anthropic/experimental_pass_through/transformation.py` + `types/llms/anthropic.py` + test.
- **`e48b7ae8f9`** — "fix(streaming): map reasoning to reasoning_content in Delta for gpt-oss providers". Fixes `litellm/types/utils.py` only (Delta type-level).
- **Content-block strip (`2c738cc939`)** — REVERTED twice upstream (`7542845e8d`, `c1b860b3c1`). **Upstream currently has NO content-block strip.**

### Drop-gate verdicts

| Problem | Upstream fix exists? | Scope overlap with ours? | Conclusion |
|---|---|---|---|
| Prompt-caching-scope header | Yes (8363a26d2e) | **Only** vertex_ai experimental-pass-through path. Does NOT cover our direct-anthropic v1/messages path or proxy pre-call path. | **Cannot DROP** |
| Reasoning field mapping | Yes (e48b7ae8f9) | Type-level only (`types/utils.py`). Does NOT cover our OpenAI iterator / streaming-handler path. Likely complementary, not replacement. | **Cannot DROP** (may be partially redundant; keep for defense-in-depth) |
| Content block / reasoning → thinking block | No (reverted upstream) | n/a | **Cannot DROP** |

**No DROPs in batch 01.** Every custom patch still has at least one surface upstream doesn't cover.

---

## Per-commit audit

### f2022065037 — dropping prompt-caching-scope-* header for v1/messages (#41)

- **files:** `litellm/llms/anthropic/experimental_pass_through/messages/transformation.py`, `litellm/types/llms/anthropic.py`, test file
- **intent:** Strip `prompt-caching-scope-*` headers from direct-anthropic v1/messages transformation for Claude Code compatibility.
- **upstream overlap:** `8363a26d2e` fixed only vertex_ai path. Our path (direct anthropic) is uncovered.
- **decision:** **REWORK (likely)** — `transformation.py` has 19 upstream commits; our hunk may conflict.
- **rationale:** Scope gap remains, so patch is needed. File churn will drive conflict surface.
- **replay plan:** Cherry-pick; if conflict, re-apply strip logic to current v1.83.3 transformation shape. Verify `types/llms/anthropic.py` type update still compiles (upstream's `8363a26d2e` also edited that file — expect small conflict).
- **verification:** MUST-SURVIVE item #20 (prompt-caching-scope header drop for Claude Code).
- **reviewer:** TBD

### 9cc84c4a1f — Dropping prompt caching scope header, vertex-ai partner-models anthropic (#42)

- **files:** `litellm/llms/vertex_ai/vertex_ai_partner_models/anthropic/transformation.py` (non-experimental variant)
- **intent:** Strip the same headers for vertex-ai anthropic partner-models non-experimental flow.
- **upstream overlap:** `8363a26d2e` patched the *experimental* variant in the same directory — **different file**. Our file is unpatched upstream.
- **decision:** **KEEP-AS-IS (likely)**
- **rationale:** Low upstream churn (5 commits) on our file. Surface should be clean.
- **replay plan:** Cherry-pick; expect clean apply.
- **verification:** MUST-SURVIVE item #20.
- **reviewer:** TBD

### df60009c8e — Dropping prompt caching scope header, proxy pre-call (#43)

- **files:** `litellm/proxy/litellm_pre_call_utils.py`
- **intent:** Strip headers at proxy pre-call layer (defense in depth — catches anything missed by provider-specific transformations).
- **upstream overlap:** None. Upstream did not touch proxy layer for this.
- **decision:** **REWORK**
- **rationale:** File has 51 upstream commits → conflict likely.
- **replay plan:** Cherry-pick; resolve by finding the current equivalent hook point in v1.83.3's `litellm_pre_call_utils.py`.
- **verification:** MUST-SURVIVE item #20.
- **reviewer:** TBD

### 01148e70ab — Dropping prompt caching scope header, follow-up (#44)

- **files:** `litellm/proxy/litellm_pre_call_utils.py`
- **intent:** Follow-up refinement of #43.
- **upstream overlap:** None.
- **decision:** **REWORK** (depends on #43 being applied first)
- **rationale:** Stacked on #43 — semantic intent is refinement.
- **replay plan:** Apply after #43. If both #43 and #44's diffs can be squashed mentally during resolution, do that to minimize noise.
- **verification:** MUST-SURVIVE item #20.
- **reviewer:** TBD

### 1a635e20e4 — Hotfix/reasoning field prod (#80)

- **files:** `litellm/litellm_core_utils/streaming_handler.py`, `litellm/llms/openai/chat/gpt_transformation.py`, `litellm/llms/openai/openai.py`, test file
- **intent:** Map `delta.reasoning` → `delta.reasoning_content` for OpenAI-compatible streaming providers (vLLM / GLM-5 / hosted_vllm) in the iterator + streaming handler.
- **upstream overlap:** `e48b7ae8f9` maps the same at the **Delta type level**. Broader reach — catches providers regardless of iterator. Upstream fix may make ours partially redundant, but not demonstrably equivalent on our specific iterator path.
- **decision:** **KEEP-AS-IS (likely)**, with post-replay verification that both fixes coexist cleanly (our mapping becomes an idempotent no-op if upstream's Delta-level fix fires first).
- **rationale:** streaming_handler.py has 34 upstream commits → possible conflict but the additions are new methods (`_map_reasoning_to_reasoning_content`), should be placement-only.
- **replay plan:** Cherry-pick. If upstream's Delta fix runs before our iterator fix, our code becomes a safety net. If there's a conflict, simplify by removing our code paths that are now covered by Delta-level fix.
- **verification:** smoke test — call vLLM/GLM-5 streaming endpoint with reasoning-capable model; verify `reasoning_content` appears (not `reasoning`). Not in the main MUST-SURVIVE list — add as new verification row.
- **reviewer:** TBD

### 730dddc2dd — Fix/content block issue (#134)

- **files:** `litellm/llms/anthropic/experimental_pass_through/adapters/transformation.py`
- **intent:** When a response has `reasoning_content` but no `thinking_blocks`, synthesize an anthropic `thinking` content block from `reasoning_content` (covers OpenRouter-style providers). Handles both non-streaming and streaming paths.
- **upstream overlap:** None applicable — upstream's `2c738cc939` content-block strip was REVERTED.
- **decision:** **REWORK (likely)**
- **rationale:** 23 upstream commits on this file; our additions (19 lines) land in two `elif` branches that may have shifted.
- **replay plan:** Cherry-pick; if conflict, re-apply the reasoning→thinking-block synthesis against v1.83.3's updated adapter.
- **verification:** smoke test — call an OpenRouter-backed model via the anthropic-compatible endpoint; verify thinking block appears in response.
- **reviewer:** TBD

---

## Batch summary

| # | SHA | Decision | Risk | Rationale |
|---|---|---|---|---|
| 1 | f2022065037 | REWORK | MED | Upstream covered only vertex_ai; direct-anthropic path still needs our fix |
| 2 | 9cc84c4a1f | KEEP-AS-IS | LOW | Non-experimental vertex-anthropic path untouched upstream |
| 3 | df60009c8e | REWORK | MED | Proxy pre-call layer has 51 upstream commits |
| 4 | 01148e70ab | REWORK | MED | Stacked on #43 |
| 5 | 1a635e20e4 | KEEP-AS-IS | LOW | Additive iterator-level fix; upstream's Delta-level fix is complementary |
| 6 | 730dddc2dd | REWORK | MED | 23 upstream commits on adapter file |

**No DROPs.** Upstream's prompt-caching-scope fix is scoped narrowly to vertex-ai experimental path; our fixes cover three additional surfaces (direct-anthropic, non-experimental vertex-anthropic, proxy pre-call). Upstream's reasoning-field fix is at a different abstraction layer. Content-block strip was reverted upstream.

## Replay notes

- Apply in chronological order per batch file.
- **`litellm/types/llms/anthropic.py` is edited by both our `#41` and upstream's `8363a26d2e`** — expect a small conflict resolved by merging both type additions.
- Add two new rows to MUST-SURVIVE verification list:
  - vLLM/GLM-5 streaming reasoning_content smoke test
  - OpenRouter→anthropic-adapter thinking-block smoke test
