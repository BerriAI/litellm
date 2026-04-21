# Batch 05 — Budgets (HIGH PRIORITY)

**Commits:** 5
**Scope:** User-budget restriction (multi-pod-safe), `FREE_MODELS` env-based budget bypass, related logging cleanup.
**Criticality:** MUST-SURVIVE items #1 (`FREE_MODELS` bypass) and #2 (budget duration enforcement) depend on this batch.

## Upstream landscape (v1.81.3 → v1.83.3)

| File | # upstream commits | Notes |
|---|---|---|
| `litellm/proxy/auth/auth_checks.py` | 44 | High churn — many new budget paths added: project checks, team_soft_budget, tag_max_budget, team_member_budget, tracer.trace wrappers |
| `litellm/proxy/auth/user_api_key_auth.py` | 57 | Very high — includes multi-pod spend counter wiring |
| `litellm/proxy/db/db_spend_update_writer.py` | 19 | Moderate |
| `litellm/proxy/hooks/proxy_track_cost_callback.py` | 19 | Moderate |

### Upstream equivalents found

- **`d533b432fd`** — `fix(proxy): enforce budget limits across multi-pod deployments via Redis-backed spend counters`. Introduces `spend_counter_cache` (Redis-backed DualCache), `increment_spend_counters()`, `get_current_spend()` helpers. **Covers: API keys, teams, team-members.** **Does NOT cover: user-budget checks.** User budget still reads `user_object.spend` at `auth_checks.py:558`.
- **`b0db75df1f`** — convert `max_budget` to `float` when set from env.
- **`28c77b48c9`** — fix `+Inf` user-budget metric.
- **`547e84418e` / `d9af321610`** — "Fix free models working from UI" — UI/Project-config path, not an env-var bypass mechanism. Upstream v1.83.3's `auth_checks.py:511` has only a comment referencing "free model" alongside a `skip_budget_checks` param; the `FREE_MODELS` env var does not exist upstream.

### Drop-gate verdicts

| Custom feature | Upstream equivalent? | Safe to DROP? |
|---|---|---|
| User-budget multi-pod restriction (#3e503c7347 + #7c2f182ac1) | Partial — upstream covers key/team/team-member. User-budget still reads stale cached spend. | **No.** Gap remains. |
| `FREE_MODELS` env-var bypass (#f59a60747e) | No — upstream's `skip_budget_checks` is a function parameter gated by project-config, not an env-var allowlist. | **No.** |
| Debug log removals (#e7135406e7, #46124d889f) | Depends on whether the debug lines they remove still exist post-rework | Possibly — if the debug lines no longer exist after REWORK of #7c2f182ac1, these become no-op and can be skipped. |

**No unconditional DROPs in batch 05.** #e7135406e7 and #46124d889f may become irrelevant after #7c2f182ac1 is reworked — will decide during replay.

---

## Per-commit audit

### 3e503c7347 — feature: user budget restriction

- **files:** `auth_checks.py` (+35/−7), `user_api_key_auth.py` (+3)
- **intent:** Enforce user-level `max_budget` at the auth layer (original implementation).
- **upstream overlap:** Upstream adds `_check_user_max_budget` family of checks with tracer wrappers, plus project-based `skip_budget_checks`. User-budget check still uses in-memory `user_object.spend`.
- **decision:** **REWORK**
- **rationale:** Insertion point in `auth_checks.py` has shifted significantly due to upstream's restructuring (project checks, tracer.trace blocks). Our 35-line hunk must be relocated to the correct section.
- **replay plan:** Cherry-pick; when conflict lands, find the function-equivalent block in v1.83.3's `_run_<x>_checks` series and re-apply our user-budget enforcement logic. Wrap in `tracer.trace("litellm.proxy.auth.common_checks.user_max_budget_check")` to match upstream style.
- **verification:** MUST-SURVIVE item #2 (budget duration daily/weekly/monthly reset).
- **reviewer:** TBD

### 7c2f182ac1 — cost restriction based on user spend and max_budget

- **files:** `auth_checks.py` (+82/−~), `db_spend_update_writer.py` (+121/−~), `proxy_track_cost_callback.py` (+76/−~)
- **intent:** Multi-pod-safe user budget enforcement — reconciles in-memory spend against DB/Redis state before enforcing.
- **upstream overlap:** `d533b432fd` adds exactly this for keys/teams/team-members (files: `auth_checks.py`, `user_api_key_auth.py`, `reset_budget_job.py`, `proxy_track_cost_callback.py`). User-scope absent from upstream's fix.
- **decision:** **REWORK — re-plumb onto upstream `spend_counter_cache`** ← *decision locked 2026-04-21 by @shriharsha*
- **rationale:** Highest-value rework in the upgrade. Upstream's `spend_counter_cache` infrastructure is exactly what we want to piggyback on. Carrying our parallel infrastructure would mean a large long-term diff; re-plumbing converts our patch into a thin user-scope extension over upstream's existing three scopes (key/team/team-member).
- **replay plan:** 
  1. Read upstream's `d533b432fd` end-to-end.
  2. Rewrite our hunks to call upstream's helpers with user-scope counter key.
  3. Our net changes should shrink significantly — effectively adding one more scope (user) to upstream's existing three (key/team/team-member).
- **verification:** MUST-SURVIVE item #2. Also: soak test 2-pod deployment, set `max_budget=100`, spend across both pods, verify enforcement at $100 not $200.
- **reviewer:** TBD. **This commit needs the most careful review in the entire upgrade.**

### e7135406e7 — removing debug print logs for cost restriction

- **files:** `db_spend_update_writer.py` (+2/−13), `proxy_track_cost_callback.py` (+1/−5)
- **intent:** Strip debug prints added during #7c2f182ac1 development.
- **upstream overlap:** The debug lines it removes only exist if #7c2f182ac1 is applied verbatim. After REWORK of #7c2f182ac1 (to use upstream helpers), these debug lines likely won't exist.
- **decision:** **CONDITIONAL — likely no-op after #7c2f182ac1 REWORK**
- **rationale:** If the REWORK of #7c2f182ac1 doesn't introduce the original debug prints, there's nothing to remove.
- **replay plan:** After #7c2f182ac1 replay, check whether the debug prints exist. If not, skip #e7135406e7 (record as DROP with rationale "no-op after upstream alignment of #7c2f182ac1").
- **verification:** n/a (log hygiene).
- **reviewer:** TBD

### 46124d889f — removed a debug log line

- **files:** `auth_checks.py` (+1/−1)
- **intent:** Single-line debug log removal.
- **upstream overlap:** Same as #e7135406e7 — depends on whether #3e503c7347's debug line survives REWORK.
- **decision:** **CONDITIONAL — likely no-op after #3e503c7347 REWORK**
- **replay plan:** Same pattern as #e7135406e7. Skip if the target line no longer exists.
- **verification:** n/a.
- **reviewer:** TBD

### f59a60747e — models endpoints budget bypass fix / FREE_MODELS (#106)

- **files:** `auth_checks.py` (+83/−30), `user_api_key_auth.py` (+5/−2)
- **intent:** `FREE_MODELS` env var — if the requested model is in the env-var list, bypass user budget enforcement. Also fixes `/models` endpoint to not require budget.
- **upstream overlap:** Upstream has `skip_budget_checks` param and project-based free-model config, but **no env-var-driven allowlist**. Our env-var contract with ops team cannot be dropped without coordinating a migration.
- **decision:** **REWORK**
- **rationale:** 83 lines into a high-churn file (`auth_checks.py` has 44 upstream commits, many reshaping the exact block our code lives in).
- **replay plan:** 
  1. Cherry-pick; massive conflict expected.
  2. Resolve by re-inserting FREE_MODELS env-var check as the first statement inside `common_checks` after arguments are bound, returning early when the model is in the allowlist (consistent with upstream's `skip_budget_checks` early-return pattern).
  3. Verify `/models` endpoint still returns full model list regardless of budget state.
- **verification:** MUST-SURVIVE item #1 — set `FREE_MODELS=gpt-3.5-turbo`, user with exhausted budget calls model, expect 200.
- **reviewer:** TBD

---

## Batch summary

| # | SHA | Decision | Risk |
|---|---|---|---|
| 1 | 3e503c7347 | REWORK | HIGH |
| 2 | 7c2f182ac1 | REWORK (re-plumb onto upstream spend_counter_cache) | **HIGHEST IN UPGRADE** |
| 3 | e7135406e7 | Conditional DROP (no-op after rework) | LOW |
| 4 | 46124d889f | Conditional DROP (no-op after rework) | LOW |
| 5 | f59a60747e | REWORK | HIGH |

**No unconditional DROPs.** Two conditional DROPs pending rework outcome of prior commits.

## Replay notes

- **This batch is the highest audit-intensity one.** Budget correctness is zero-tolerance — wrong answers manifest as either (a) dropped revenue (our free-models bypass fails silently) or (b) production budget bypass (user budget not enforced across pods).
- Strongly recommend reviewing #7c2f182ac1's upstream alignment with an engineer before replay.
- After replay, **do not skip the 2-pod soak verification** — this is the exact class of bug that caused upstream's `d533b432fd` in the first place.
- Consider opening upstream PR to add `spend:user:{user_id}` to upstream's `spend_counter_cache` — that would eliminate our long-term carried diff here.
