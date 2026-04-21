# Phase 2 Audit Summary — v1.83.3 Upgrade

**Status:** All 9 batches audited.
**Total custom commits:** 87 → 83 after DROP pairs.
**Reclassifications:** 1 (`3be74052a5` moved from batch 01 to batch 02 — GCS-logger fix misclassified).

## Decision breakdown

| Decision | Count | % |
|---|---|---|
| KEEP-AS-IS | 30 | 34% |
| REWORK | 51 | 59% |
| DROP (unconditional) | 4 | 5% (2 revert pairs) |
| Conditional DROP | 2 | 2% (no-ops after upstream refactors) |

## DROPs explained

| SHA | Reason | Audit doc |
|---|---|---|
| 20caa0aebb (#111) | Reverted in place by #115 — net zero diff | batch-06 |
| 596a3a3a5a (#115) | Reverts #111 — net zero diff | batch-06 |
| f7f8141eab (#46) | Reverted in place by #48 — net zero diff | batch-07 |
| 0930b8d771 (#48) | Reverts #46 — net zero diff | batch-07 |

**Verification command before Phase 3:**
```bash
git diff 20caa0aebb^ 596a3a3a5a | wc -l   # expect 0
git diff f7f8141eab^ 0930b8d771 | wc -l   # expect 0
```

If both are 0, DROPs are safe. If not, re-audit.

## Conditional DROPs

| SHA | Reason | Batch |
|---|---|---|
| e7135406e7 | Removes debug prints from 7c2f182ac1; after REWORK of 7c2f182ac1 onto upstream's spend_counter_cache, the debug lines likely don't exist. Skip if target lines absent. | 05 |
| 46124d889f | Single debug-log removal in auth_checks.py; likely no-op after #3e503c7347 REWORK. | 05 |
| 61fb9f323f | Removes 82 lines of duplicated spend-update code. May be no-op if upstream refactored the duplication. | 06 |

## Per-batch summary

| Batch | Commits | KEEP | REWORK | DROP | Risk | Replay order |
|---|---|---|---|---|---|---|
| 01 claude-anthropic-compat | 6 | 2 | 4 | 0 | MED | 1st |
| 09 build-ci-playground-misc | 12 | 4 | 8 | 0 | MED-HIGH | 2nd |
| 02 gcs-gcp-logging | 14 | 12 | 2 | 0 | LOW-MED | 3rd |
| 05 budgets | 5 | 0 | 3 | 0 + 2 conditional | **HIGHEST** | 4th |
| 08 admin-user-mgmt | 4 | 0 | 4 | 0 | MED-HIGH | 5th |
| 06 analytics-spend-failure | 13 | 1 | 9 | 2 + 1 conditional | HIGH | 6th |
| 07 ui | 10 | 3 | 5 | 2 | MED-HIGH | 7th |
| 04 rate-limit-concurrency | 6 | 1 | 5 | 0 | HIGH | 8th |
| 03 routing-vision | 17 | 7 | 10 | 0 | HIGH (highest file churn) | 9th |
| **Totals** | **87** | **30** | **50** | **4** + 3 conditional | | |

## Highest-risk files (by upstream churn)

| File | # upstream commits | Batches affected |
|---|---|---|
| `litellm/proxy/_types.py` | 145 | 03, 07, 08, 09 |
| `litellm/router.py` | 117 | 03 |
| `litellm/proxy/proxy_server.py` | 183 | 08, 09 |
| `litellm/proxy/utils.py` | 62 | 04, 06 |
| `ui/networking.tsx` | 93 | 06, 07 |
| `litellm/proxy/auth/user_api_key_auth.py` | 57 | 04, 05, 07 |
| `litellm/proxy/litellm_pre_call_utils.py` | 51 | 01, 03 |
| `litellm/proxy/auth/auth_checks.py` | 44 | 03, 05 |
| `model_prices_and_context_window.json` | 266 | 09 (mechanical) |
| `pyproject.toml` | 102 | 07, 09 |

## Key upstream features that overlap with ours (verified scope, not DROPs)

- **Multi-pod budget enforcement** (`d533b432fd`) — covers key/team/team-member but NOT user. Our user-budget fix still uniquely needed.
- **`prompt-caching-scope` header fix** (PR #20058) — vertex-ai experimental pass-through only. Our v1/messages + proxy-pre-call + non-experimental paths still uniquely needed.
- **Reasoning field → reasoning_content mapping** (`e48b7ae8f9`) — Delta type-level only. Our iterator-level fix complementary.
- **Sticky-sessions** (PR #21763) — session-id header-driven affinity. Our load-driven sticky-least-busy is different mechanism, both can coexist.
- **Agent-level budget + rate limiting** (`cf439c269c`, +141 lines in `parallel_request_limiter_v3.py`) — different scope from our MPR fixes. Our counter-leak/drift work still uniquely needed.
- **`/user/bulk_update`** — adjacent to our `/user/bulk_cost_update` but different endpoint name; dropping breaks clients.
- **Audit log S3 export** — additive to our audit-hook work, no replacement.

## Pre-Phase-3 checklist

Before running `git reset --hard v1.83.3-stable`:

1. [ ] Verify DROP revert pairs produce zero diff (commands above).
2. [ ] Assign reviewer handles to all `reviewer: TBD` rows in audit docs.
3. [ ] Confirm backup branches exist (`backup/pre-v1.83.3-upgrade-20260421` locally).
4. [ ] Dry-run `git cherry-pick --no-commit b95bd31c76` on a throwaway branch off `v1.83.3-stable` — gauge first-conflict surface for batch 04.
5. [ ] If dry-run shows >200 conflict lines on the first commit, plan to subdivide batch 04.
6. [ ] Read batch-05 audit end-to-end with a second engineer; budget correctness is zero-tolerance.

## Replay sequence (9 batches, 2–4 days wall-clock)

```
feature/upgrade-to-litellm-v1.83.3 (reset to v1.83.3-stable)
  ↓
Batch 01 (claude-compat, 6) → smoke lint-ruff → tag replay-checkpoint/01
  ↓
Batch 09 (build/ci/playground, 12) → build check → tag 02
  ↓
Batch 02 (gcs-gcp logging, 14) → tag 03
  ↓
Batch 05 (budgets, 5) → smoke budget test → tag 04  ⚠ highest-risk
  ↓
Batch 08 (admin-user-mgmt, 4) → tag 05
  ↓
Batch 06 (analytics, 13) → npm install + UI build → tag 06
  ↓
Batch 07 (UI, 10) → UI build + smoke → tag 07
  ↓
Batch 04 (rate-limit, 6) → MPR soak test → tag 08  ⚠ high-risk
  ↓
Batch 03 (routing-vision, 17) → routing tests → tag 09  ⚠ highest file churn
  ↓
Phase 4 verification (MUST-SURVIVE checklist)
  ↓
Phase 5 ship (squash-merge PR → main)
```

## Open items to discuss before Phase 3

- **Batch 05 / #7c2f182ac1:** Should we re-plumb our user-budget enforcement onto upstream's `spend_counter_cache` pattern (longer REWORK, cleaner long-term) or carry parallel infrastructure (shorter REWORK, bigger diff to maintain)?
- **Batch 03 / sticky-least-busy:** Should we adopt upstream's `deployment_affinity_check.py` pattern as a substrate, or keep our `router_strategy/sticky_least_busy*.py` as independent modules?
- **Batch 08 / #75 bulk cost update:** Consider renaming our endpoint or migrating callers to upstream's `/user/bulk_update` to reduce long-term divergence.
- **Batch 09 / playground #124:** 3 schema.prisma files changed — confirm migration timestamp doesn't clash with upstream's migrations in v1.83.3.
