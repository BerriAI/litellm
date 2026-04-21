# Batch 07 — UI (with some cross-cutting backend)

**Commits:** 10 (includes a revert pair — candidate for DROP-both)
**Scope:** Logs pagination, UI type fix, model-filter dropdown, message-filter callback, filter-state preservation, concurrent-requests tab viewer permission.

Note: Some batch-07 commits touch backend files (callbacks, auth, endpoints) despite the "UI" label — these get audited here for coherence but the replay may want to interleave with other batches.

## Upstream landscape (v1.81.3 → v1.83.3)

| File | # upstream commits | Notes |
|---|---|---|
| `ui/view_logs/index.tsx` | 31 | Moderate |
| `ui/view_logs/log_filter_logic.tsx` | 12 | Low-mod |
| `ui/molecules/filter.tsx` | 4 | Low |
| `ui/networking.tsx` | 93 | Very high |
| `ui/view_logs/concurrent_request_logs.tsx` | 0 | **Our file** |
| `litellm/proxy/_types.py` | 145 | Very high |
| `litellm/proxy/auth/user_api_key_auth.py` | 57 | High |
| `litellm/proxy/spend_tracking/spend_management_endpoints.py` | 18 | Moderate |
| `litellm/callbacks/message_filter.py` | 0 | **Our file** |
| `.github/workflows/test-*.yml` | 4–13 each | Low-mod |
| `gcs_logger.py` | ~2 semantic | Low |

### Upstream equivalents

Searched for: message filter callback, model filter dropdown in logs, filter-state preservation, concurrent-requests viewer permission. **Zero upstream matches.** No DROPs except revert pair.

---

## Per-commit audit

### 066d5d8026 — Logs pagination fix (#29)

- **files:** `view_logs/index.tsx` (31), `log_filter_logic.tsx` (12), test file
- **decision:** **REWORK**
- **rationale:** Moderate-churn files, pagination logic fix will need re-location.
- **verification:** MUST-SURVIVE smoke — UI logs view, page 2/3, verify filters persist.

### 80e242236e — build fix (#36) [actually a UI fix]

- **files:** `ui/user_agent_activity.tsx` (0 upstream — our file)
- **decision:** **KEEP-AS-IS**

### f7f8141eab — "temp logging change" (#46)

- **files:** `gcs_bucket/gcs_logger.py`
- **decision:** **DROP**
- **rationale:** Adds `"litellm_raw_payload": kwargs` as a temporary debug payload; #48 removes exactly this line. Purely a debug/temp change.

### 0930b8d771 — Revert "temp logging change" + "Add headers to error log metadata" (#48)

- **files:** `gcs_bucket/gcs_logger.py`
- **decision:** **KEEP-AS-IS**
- **rationale:** **NOT a pure revert.** The commit does two things:
  1. Reverts #46's `litellm_raw_payload` addition (undone).
  2. **Adds** `"headers": metadata.get("headers"),` to the error-log path (new feature).
  
  The second change is a real feature — headers in error log metadata — that we must keep. If we DROP both #46 and #48, we lose this feature.

#### Drop-gate check for #46 (only)

1. Behavior equivalence: `litellm_raw_payload: kwargs` was a temp debug payload, fully undone by the first hunk of #48.
2. No semantic regression: dropping #46 means the temp payload never existed, which is the target state.
3. Verification test: no user-facing behavior tied to the debug payload.
4. Human sign-off: TBD.

**Since we're keeping #48, #48 contributes both the revert-effect AND the headers-feature to the final tree. Cherry-picking #48 alone (and skipping #46) produces the correct end state.**

### 4d50cd5674 — chartOptions strict type checks (#55)

- **files:** `ui/view_logs/ErrorStatsTable.tsx` (our new file from batch-06 #47)
- **decision:** **KEEP-AS-IS**
- **rationale:** Our component.

### f8709b0065 — Message filter callback (#100)

- **files:** `litellm/callbacks/message_filter.py` (new file, ours only)
- **decision:** **KEEP-AS-IS**
- **rationale:** New backend callback module. Not in upstream.
- **verification:** MUST-SURVIVE item #19 (message filter UI wired to this backend).

### c1fefb1090 — models filter dropdown + viewer access (#103)

- **files:** 6 GitHub workflow files (4–13 upstream each), `_types.py` (145), `user_api_key_auth.py` (57), `spend_management_endpoints.py` (18), `package-lock.json`, `networking.tsx` (93), `view_logs/index.tsx` (31), `log_filter_logic.tsx` (12)
- **decision:** **REWORK (high effort)**
- **rationale:** Largest commit in batch 07. Auth changes (viewer access to filtered logs), workflow changes (CI), and UI changes (dropdown + filter logic). Cross-cuts very-high-churn files.
- **replay plan:**
  1. Workflows: low risk, apply first.
  2. Auth: re-apply viewer permission addition against upstream's v1.83.3 `user_api_key_auth.py` (57 commits including multi-pod spend counter wiring).
  3. `_types.py`: our enum/model additions into the new file layout.
  4. UI networking + view_logs: merge with upstream changes.
  5. Regenerate `package-lock.json`.
- **verification:** MUST-SURVIVE item #18 (models filter dropdown + viewer access).

### 2277c4aa4b — saving filter state before remounting (#109)

- **files:** `ui/molecules/filter.tsx` (4 upstream), `view_logs/index.tsx` (31)
- **decision:** **REWORK (low effort)**
- **rationale:** Small state-management fix; low conflict risk.

### 4a744f2144 — permission: concurrent-requests tab viewable for admin-viewer (#118)

- **files:** `litellm/proxy/_types.py` (145 upstream)
- **decision:** **REWORK**
- **rationale:** 1-line enum addition into a very-high-churn types file. Small diff but likely conflict on enum positioning.
- **verification:** MUST-SURVIVE item #10 (concurrent-requests filters + viewer perm).

### a6698e18db — concurrent-requests logs issues + filters (#126)

- **files:** `Dockerfile` (13), `gcp_logging_helpers/__init__.py` (ours), `gcp_logs_query.py` (ours), `spend_management_endpoints.py` (18), `pyproject.toml` (102), `networking.tsx` (93), `view_logs/concurrent_request_logs.tsx` (ours)
- **decision:** **REWORK**
- **rationale:** Cross-cutting. Dockerfile + pyproject.toml hunks will conflict with dependency bumps upstream.
- **replay plan:**
  1. Our new `gcp_logging_helpers` files apply clean.
  2. Dockerfile + pyproject.toml — reconcile version pins.
  3. `spend_management_endpoints.py` + `networking.tsx` — merge with upstream changes.
- **verification:** MUST-SURVIVE item #10.

---

## Batch summary

| # | SHA | Decision | Risk |
|---|---|---|---|
| 1 | 066d5d8026 | REWORK | MED |
| 2 | 80e242236e | KEEP-AS-IS | LOW |
| 3 | f7f8141eab | **DROP** (temp debug, pure no-op) | n/a |
| 4 | 0930b8d771 | **KEEP-AS-IS** (revert + real feature: headers in error log) | LOW |
| 5 | 4d50cd5674 | KEEP-AS-IS | LOW |
| 6 | f8709b0065 | KEEP-AS-IS | LOW |
| 7 | c1fefb1090 | REWORK | HIGH |
| 8 | 2277c4aa4b | REWORK | LOW |
| 9 | 4a744f2144 | REWORK | LOW |
| 10 | a6698e18db | REWORK | MED-HIGH |

**One DROP** (#46, pure temp debug change — removed by #48 which we keep). 4 KEEP-AS-IS, 5 REWORK.

## Replay notes

- #46 is DROP (pure debug) but #48 must be KEEP. Verified 2026-04-21: #48 reverts #46 AND introduces "headers in error log metadata" as a new feature. Skipping #48 would lose the feature.
- `c1fefb1090` is the most complex commit in this batch; allocate ~half a day to its resolution.
- `package-lock.json` conflicts in #103 and elsewhere: regenerate via `npm install` in the `ui/litellm-dashboard` directory post-cherry-pick.
