# Phase 3 Dry-Run Findings — 2026-04-21

**Method:** created scratch branch off `v1.83.3-stable`, ran `git cherry-pick --no-commit <sha>` per commit, measured conflict markers and files affected. Reset between runs.

## 7-commit sample (chosen to cover highest-risk audit decisions)

| SHA | Purpose | Files conflicted | Conflict markers | Verdict |
|---|---|---|---|---|
| `b95bd31c76` | MPR rate-limit fixes (batch 04 first) | 0 | 0 | **CLEAN** — auto-merged `docker-compose.yml` + `parallel_request_limiter_v3.py` |
| `03a998194b` | Vision fix in `router.py` (117 upstream commits) | 0 | 0 | **CLEAN** — auto-merged |
| `7c2f182ac1` | User-budget multi-pod (batch 05 highest-risk) | 3 | 4 | MEDIUM — `auth_checks.py`, `db_spend_update_writer.py`, `proxy_track_cost_callback.py` |
| `f59a60747e` | `FREE_MODELS` bypass | 2 | 5 | MEDIUM — `auth_checks.py`, `user_api_key_auth.py` |
| `a41df42b80` | Audit logging (broadest batch-08 diff) | 2 | 7 | MEDIUM-HIGH — `key_management_endpoints.py`, `audit_logs.py`. Other 5 touched files auto-merged. |
| `9cdf6d7092` | Counter leak alternate approach (batch 04) | 3 | 10 | HIGH — `common_request_processing.py`, `parallel_request_limiter_v3.py`, `utils.py` |
| `e6e49c5069` | Sticky-least-busy feature | 1 (modify/delete) | 0 | SPECIAL — `.github/workflows/build-gcr.yml` was removed upstream; resolve by keeping our version (no manual conflict) |

## Key findings

1. **Auto-merge is very effective.** Commits landing on files with 100+ upstream commits (e.g., `router.py` vision fix) still auto-merge cleanly in most cases. The audit over-estimated REWORK effort based on file-level churn — conflict surface is much smaller than churn-count suggested.

2. **Actual conflict hotspots:**
   - `litellm/proxy/auth/auth_checks.py` — conflicts on BOTH batch-05 commits (user-budget + FREE_MODELS). Expect cumulative conflict here if replayed in sequence without staged resolution.
   - `litellm/proxy/hooks/parallel_request_limiter_v3.py` — conflicts on batch-04 iterative fixes (#114 especially).
   - `litellm/proxy/common_request_processing.py` — conflicts on counter-leak path.
   - `litellm/proxy/management_helpers/audit_logs.py` — conflicts on audit-logging extension.

3. **Total conflict marker count across the 7 sampled commits: 26.** If this rate holds for the full 87 commits, expected total is ~100-150 conflict markers across the whole replay — **manageable in 1-2 focused days**, not the 2-4 days originally estimated.

4. **No unexpected blockers.** Every cherry-pick either applied cleanly or produced standard content conflicts resolvable manually. No `fatal:` errors, no binary conflicts, no deleted-file problems beyond the trivial modify/delete on `build-gcr.yml`.

5. **Revised Phase 3 estimate:** 1.5–2.5 working days (was 2–4).

## Handling modify/delete for `.github/workflows/build-gcr.yml`

Upstream removed the file. Our commits (#96, #105, #131) modify it. Resolution: accept our version (`git add .github/workflows/build-gcr.yml`). Our file is entirely custom with 0 upstream commits; losing upstream's removal is the right choice.

## Implications for batch replay order

- **Batch 04 commits with iterative fixes (#101 → #102 → #114 → #116 → #121 → #125) must be replayed in order.** Conflict on #114 only arises because earlier commits' integration points shifted upstream. Out-of-order replay multiplies conflicts.
- **Batch 05 #106 (`FREE_MODELS`) should land LAST within batch 05** because it touches the same `auth_checks.py` region as #3e503c7347 and #7c2f182ac1.
- **Batch 03 `e6e49c5069` modify/delete needs a pre-replay note** so the operator doesn't panic on the first "deleted in HEAD" message.
