# Batch 04 — Rate-limit & Concurrency

**Commits:** 6
**Scope:** Multi-instance max-parallel-requests rate-limiter (MPR), Redis counter leak/drift fixes, perf, metrics.
**Primary file:** `litellm/proxy/hooks/parallel_request_limiter_v3.py` (already on v3 upstream module — prior REWORK done).

## Upstream landscape (v1.81.3 → v1.83.3)

Commits touching files in this batch:

| File | # upstream commits | Notable |
|---|---|---|
| `parallel_request_limiter_v3.py` | 3 | `cac685014f` feat: deployment-default tpm/rpm (+8 lines); `cf439c269c` feat: agent-level budget + rate-limit (+141 lines, rewrites chunks of the limiter); `2e9f057fbd` style: black formatter (+39/−21) |
| `common_request_processing.py` | 54 | Heavy churn — guardrails, streaming, error handling refactors |
| `proxy/utils.py` | 62 | Heavy churn |
| `prometheus.py` | 23 | Org budget metrics added |
| `spend_management_endpoints.py` | 18 | Spend-log duration, model cost aliases |

**Counter-leak / counter-drift search in upstream commit messages:** zero matches. Upstream did not ship equivalents to our `#101`, `#102`, `#114`, `#125` fixes. **No DROP candidates in this batch.**

**Line-level divergence in `parallel_request_limiter_v3.py`:** ~681 lines between `main` and `v1.83.3-stable` — real REWORK surface, dominated by `cf439c269c` (agent feature) overlap risk.

---

## Per-commit audit

### b95bd31c76 — fix: multi instance max parallel requests rate limit fixes (#101)

- **files:** `docker-compose.yml` (+19), `parallel_request_limiter_v3.py` (+18/−13)
- **intent:** Multi-instance MPR correctness — docker-compose sets up Redis wiring for local multi-instance testing; limiter changes fix counter semantics across instances.
- **upstream in region:** `cac685014f`, `cf439c269c`, `2e9f057fbd` (see landscape table).
- **decision:** **REWORK**
- **rationale:** Agent-limiting rewrite (`cf439c269c`) changed adjacent functions; our hunk likely conflicts. Black formatter changed surrounding whitespace. `docker-compose.yml` is local-only, should be clean.
- **replay plan:** Expect conflict on limiter hunk. Resolve by: (1) locate the same semantic region in v1.83.3's agent-aware limiter, (2) re-apply our multi-instance fix against that region, (3) verify key-level / deployment-default / agent-level priority order is preserved.
- **verification:** MUST-SURVIVE item #8 (MPR soak test), #9 (Redis-only counter).
- **reviewer:** TBD

### 3727eebad7 — performance fixes for rpm and concurrency limit (#102)

- **files:** `get_litellm_params.py` (+2), `main.py` (+3), `common_request_processing.py` (+11), `parallel_request_limiter_v3.py` (+224/−~), `proxy/utils.py` (+34), `types/utils.py` (+1)
- **intent:** Avoid redundant Redis writes on counter inc/dec — perf improvement.
- **upstream in region:** v3 limiter agent rewrite, 54 commits to `common_request_processing.py`, 62 commits to `proxy/utils.py`.
- **decision:** **REWORK**
- **rationale:** Largest surface in the batch. Touches 6 files, 3 of which are high-churn upstream. `main.py` + `get_litellm_params.py` + `types/utils.py` hunks are small and likely clean; `common_request_processing.py`, `proxy/utils.py`, and v3 limiter are REWORK.
- **replay plan:** Cherry-pick and resolve piecewise. Expect heaviest conflict in v3 limiter (agent rewrite overlap). `proxy/utils.py` and `common_request_processing.py` hunks should be portable but need context-matching against upstream's new state.
- **verification:** MUST-SURVIVE item #8, #9.
- **reviewer:** TBD

### 9cdf6d7092 — fix: alternate approach for handling rate limit counter leak issues (#114)

- **files:** `common_request_processing.py` (+370/−~), `parallel_request_limiter_v3.py` (+338/−~), `proxy/utils.py` (+31/−~)
- **intent:** Second-attempt counter-leak fix after #101's original approach. Decouples counter decrement from request-lifecycle error paths.
- **upstream in region:** no counter-leak fix upstream. v3 limiter + adjacent files have agent/guardrail rewrites.
- **decision:** **REWORK**
- **rationale:** Largest single-commit diff in the batch. Because `#101` and `#114` are iterative fixes on the same bug, they depend on each other's state. Replay order matters: `#101` first, then `#102`, then `#114`.
- **replay plan:** Apply in strict chronological order (list order in batch file). After the v3 limiter REWORK on #101, #114 may apply more cleanly against the reworked base.
- **verification:** MUST-SURVIVE item #8.
- **reviewer:** TBD

### f3136f3eac — feat: metrics & UI for Redis counter verification (#116)

- **files:** `docker-compose.yml` (+1), `prometheus.py` (+14), `common_request_processing.py` (+9/−5), `parallel_request_limiter_v3.py` (+94/−~), `spend_management_endpoints.py` (+303, new endpoint), `ui/networking.tsx` (+60), `ui/view_logs/concurrent_request_logs.tsx` (+318, new component), `ui/view_logs/index.tsx` (+5)
- **intent:** Expose Prometheus metric + new admin endpoint + UI tab for observing the MPR counter state — diagnostic tooling for the leak/drift fixes.
- **upstream in region:** 23 prometheus commits (org budget metrics — same file, different metric additions). 18 spend_management_endpoints commits. v3 limiter churn. UI files moderately churned.
- **decision:** **REWORK**
- **rationale:** New code (`concurrent_request_logs.tsx`, new endpoint in `spend_management_endpoints.py`) is additive and likely clean. Prometheus additions may conflict with org-budget metrics ordering; UI `index.tsx` tab registration is a known conflict surface.
- **replay plan:** Apply new-file hunks cleanly. Re-insert prometheus metric alongside upstream's new org-budget metrics (no logical conflict — just ordering). Reconcile UI tab list.
- **verification:** MUST-SURVIVE item #10 (concurrent-requests filters + viewer perm).
- **reviewer:** TBD

### 29495ff0a6 — fix: log for metric emission in MPR rate-limit flow (#121)

- **files:** `parallel_request_limiter_v3.py` (+13/−1)
- **intent:** Observability-only — adds a log line when MPR metric is emitted.
- **upstream in region:** v3 limiter agent rewrite.
- **decision:** **KEEP-AS-IS (likely) — small REWORK possible**
- **rationale:** 13-line addition. If the log-insertion point is in a region upstream didn't touch, clean; otherwise minor context-conflict resolution.
- **replay plan:** Cherry-pick; if conflict, re-position the log statement in the equivalent emission branch.
- **verification:** observable in Prometheus logs during MPR soak test (MUST-SURVIVE #8).
- **reviewer:** TBD

### 61dda93e05 — fix: redis counter drift problem (#125)

- **files:** `common_request_processing.py` (+58/−10), `parallel_request_limiter_v3.py` (+1/−1)
- **intent:** Fix counter drift where error paths could leave Redis counters stale across instances.
- **upstream in region:** 54 commits to `common_request_processing.py`; v3 limiter churn.
- **decision:** **REWORK**
- **rationale:** `common_request_processing.py` is a heavy-churn file — 58 lines of our fix will land amidst 54 upstream changes. V3 limiter touch is a single-line fix likely clean.
- **replay plan:** Apply #125 after all prior batch-04 commits (chronological order). Re-validate the error-path invariant our fix depends on still holds in v1.83.3's version of `common_request_processing.py`.
- **verification:** MUST-SURVIVE item #8 (drift specifically — compare Redis counter to actual in-flight request count after long soak).
- **reviewer:** TBD

---

## Batch summary

| # | SHA | Decision | Risk |
|---|---|---|---|
| 1 | b95bd31c76 | REWORK | MED |
| 2 | 3727eebad7 | REWORK | HIGH |
| 3 | 9cdf6d7092 | REWORK | HIGH |
| 4 | f3136f3eac | REWORK | MED |
| 5 | 29495ff0a6 | KEEP-AS-IS (likely) | LOW |
| 6 | 61dda93e05 | REWORK | MED |

**No DROPs.** Counter-leak and counter-drift fixes remain uniquely ours; upstream shipped no equivalents between v1.81.3 and v1.83.3.

## Replay notes

- Strict chronological order (list in `.upgrade/batches/04-rate-limit-concurrency.txt`).
- Expect the heaviest conflicts on `cf439c269c`-overlapping regions of `parallel_request_limiter_v3.py` (agent feature integration).
- Dry-run suggestion: before Phase 3, do a single `git cherry-pick --no-commit b95bd31c76` on a throwaway branch off `v1.83.3-stable` to quantify the conflict surface. If `#101` alone has >200 conflict lines, we should break the batch into smaller replay units.
- After replay, run `tests/local_testing/test_parallel_request_limiter_v2.py` (or the upstream equivalent) plus our custom MPR soak.
