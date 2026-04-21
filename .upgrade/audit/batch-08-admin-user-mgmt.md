# Batch 08 — Admin / user management

**Commits:** 4
**Scope:** Bulk cost update endpoint, alternative team-update endpoint (avoiding premium-feature gate), user-delete allowlist, lightweight audit logging across user/key/team/model ops.

## Upstream landscape (v1.81.3 → v1.83.3)

| File | # upstream commits | Notes |
|---|---|---|
| `internal_user_endpoints.py` | 37 | Upstream added `/user/bulk_update` endpoint (lines ~1437 in v1.83.3) |
| `proxy_server.py` | 183 | Very high churn |
| `_types.py` | 145 | Very high churn |
| `key_management_endpoints.py` | 115 | High churn |
| `model_management_endpoints.py` | 25 | Moderate |
| `team_endpoints.py` | 36 | Moderate |
| `management_helpers/audit_logs.py` | 6 | Low — upstream added S3 export + mypy fixes |
| `custom_team_endpoints.py` | 0 | **New file — ours only** |
| `audit_log_endpoints.py` | 0 | **New file — ours only** |

### Upstream equivalents found

- **`/user/bulk_update`** exists upstream at `internal_user_endpoints.py:1437`. Different endpoint name from our `/user/bulk_cost_update`. Functionally adjacent but not drop-in.
- **Upstream audit-log work:** `764b96b1aa` S3 export, `76cff9ae0e` admin-viewer access to audit log endpoints, `4e32db2f11` audit log S3 export. All additive — none replace the write-side audit helpers our #129 adds.
- **No upstream equivalent** for `USER_DELETE_ALLOWED_USER_IDS`, alternative team-update endpoint, or our cost-specific bulk endpoint.

### Drop-gate verdicts

| Custom feature | Upstream equivalent? | Safe to DROP? |
|---|---|---|
| `/user/bulk_cost_update` (#75) | Adjacent — `/user/bulk_update` | No. Dropping breaks clients using our endpoint path; semantic regression. |
| Alternative team update (#97) | No | No. |
| USER_DELETE_ALLOWED_USER_IDS (#128) | No | No. |
| Audit logging hooks (#129) | Additive only — different scope | No. |

**No DROPs in batch 08.**

---

## Per-commit audit

### 9eaa3ea353 — bulk cost update endpoint (#75) (#79)

- **files:** `internal_user_endpoints.py` (+195 new, +56 modified), `test_internal_user_endpoints.py` (+412)
- **intent:** Add `/user/bulk_cost_update` endpoint — bulk-update `spend` across many users in one call.
- **upstream overlap:** Upstream shipped `/user/bulk_update` (general user-field bulk update). Different endpoint, adjacent purpose.
- **decision:** **REWORK**
- **rationale:** 37 upstream commits on `internal_user_endpoints.py`, including the `/user/bulk_update` addition near lines 1373–1631. Our endpoint must co-exist without name conflict.
- **replay plan:** Cherry-pick; expect conflicts on router registrations and imports. Ensure our endpoint is registered at a distinct path.
- **verification:** Smoke test — `POST /user/bulk_cost_update` with 10 users, verify spend field updated.
- **reviewer:** TBD

### 09d73a2528 — alternative team update endpoint (#97)

- **files:** `custom_team_endpoints.py` (+176 new file), `proxy_server.py` (+4)
- **intent:** New non-premium team-update endpoint at a separate router, bypassing premium-feature gate on the existing `/team/update`.
- **upstream overlap:** File is new, not in upstream. `proxy_server.py` has 183 upstream commits.
- **decision:** **KEEP-AS-IS (file) + REWORK (router registration)**
- **rationale:** New file applies clean. The 4-line registration in `proxy_server.py` will conflict due to high churn.
- **replay plan:** Cherry-pick new file clean; resolve the router-registration hunk manually (find current router-include section, add ours).
- **verification:** Smoke test — call the non-premium team update endpoint without a premium license; verify 200.
- **reviewer:** TBD

### 8bf7d3bb64 — USER_DELETE_ALLOWED_USER_IDS allowlist (#128)

- **files:** `_types.py` (+1), `internal_user_endpoints.py` (+21)
- **intent:** Allow non-admin users whose IDs appear in env-var `USER_DELETE_ALLOWED_USER_IDS` to call `/user/delete`.
- **upstream overlap:** None. `_types.py` has 145 upstream commits (very high churn).
- **decision:** **REWORK**
- **rationale:** Very small diff but lands in a high-churn file. Conflict likely on Pydantic model edits.
- **replay plan:** Cherry-pick; if `_types.py` conflicts, apply our 1-line addition manually to the correct model.
- **verification:** MUST-SURVIVE item #21 — non-admin in allowlist calls `/user/delete` → 200.
- **reviewer:** TBD

### a41df42b80 — lightweight audit logging (#129)

- **files:** `audit_log_endpoints.py` (+104 new), `internal_user_endpoints.py` (+46), `key_management_endpoints.py` (+112), `model_management_endpoints.py` (+55), `team_endpoints.py` (+62), `management_helpers/audit_logs.py` (+64), `proxy_server.py` (+4)
- **intent:** Audit-log hooks around create/update/delete for users, keys, teams, models. Tracks `before_value` and `_changed_by` fields.
- **upstream overlap:** `audit_logs.py` is extended upstream (S3 export, mypy fixes) — our changes are additive helpers. Endpoint files have high churn (key: 115, user: 37, team: 36, model: 25).
- **decision:** **REWORK**
- **rationale:** Broadest surface in the batch. Conflicts expected on every endpoint file and in `audit_logs.py`. Upstream already has some audit-log infrastructure; we must ensure our helpers coexist with upstream's S3-export machinery.
- **replay plan:** Cherry-pick; resolve piecewise:
  1. New file (`audit_log_endpoints.py`) clean.
  2. `audit_logs.py` — add our `before_value`/`_changed_by` helpers alongside upstream's S3-export functions.
  3. Endpoint files — re-apply our `_track_audit()` call sites against v1.83.3 endpoint signatures.
- **verification:** MUST-SURVIVE item #23 — trigger each of create/update/delete for user/key/team/model; verify audit log row written with `before_value` populated.
- **reviewer:** TBD

---

## Batch summary

| # | SHA | Decision | Risk |
|---|---|---|---|
| 1 | 9eaa3ea353 | REWORK | MED |
| 2 | 09d73a2528 | KEEP + REWORK (split) | LOW-MED |
| 3 | 8bf7d3bb64 | REWORK | LOW |
| 4 | a41df42b80 | REWORK | HIGH |

**No DROPs.** Every admin/user-mgmt patch has at least one surface upstream does not cover.

## Replay notes

- `#129` (audit logging) is the highest-risk commit in the batch — touches 7 files, all with upstream churn, and overlaps with upstream audit-log work. Expect this one to dominate the batch's replay time.
- After replay, consider opening an upstream PR to contribute `before_value` tracking to `audit_logs.py` — that would shrink the diff carried in future upgrades.
