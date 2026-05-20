# G4 — Regression-replay set for PR1 (Key Tier-1)

This directory documents the regression-replay verification for the
behavior-pinning suite. For each in-scope recent fix-PR touching
`litellm/proxy/management_endpoints/key_management_endpoints.py`, we record:

- which scenario(s) in the matrix catch the behavior the fix introduced,
- a RED-at-parent / GREEN-at-tip transcript for at least one canonical PR.

## Methodology

For a given fix-PR `<sha>` with parent `<parent>`:

```bash
# 1. Save current handler
cp litellm/proxy/management_endpoints/key_management_endpoints.py /tmp/key_mgmt.HEAD.py

# 2. Replace with the pre-fix version
git show <parent>:litellm/proxy/management_endpoints/key_management_endpoints.py \
  > litellm/proxy/management_endpoints/key_management_endpoints.py

# 3. Run the catching slice — expect RED on the scenarios that flip
DATABASE_URL=postgresql://litellm:litellm@localhost:5432/litellm_test \
  uv run --no-sync pytest tests/proxy_behavior/management/<test_file>.py -v

# 4. Restore + confirm GREEN
cp /tmp/key_mgmt.HEAD.py litellm/proxy/management_endpoints/key_management_endpoints.py
DATABASE_URL=postgresql://litellm:litellm@localhost:5432/litellm_test \
  uv run --no-sync pytest tests/proxy_behavior/management/<test_file>.py -v
```

When the fix changed multiple files, the in-place swap is constrained to the
handler module — if a referenced helper moved between modules the swap may
not run; in that case use `git worktree add` for a clean replay.

## Replay table

| # | Fix SHA | Subject (truncated) | Endpoint | Catching scenarios | Verified |
|---|---------|---------------------|----------|--------------------|----------|
| 1 | `c7c3df2b02` | extend /key/update admin check to non-budget fields | `/key/update` | `test_key_update.py::self/{team_admin,internal_user,owner,unrelated_same_org,cross_org_user,service_account}` — all 6 flip 200→403 between parent and HEAD | ✅ RED→GREEN below |
| 2 | `8bbc61e03c` | harden /key/update authorization checks | `/key/update` | `test_key_update.py::owner_target/*` — non-admins blocked from updating peers' keys | by-inspection (overlapping coverage with #1) |
| 3 | `1b2756811e` | close project hijacking and key org IDOR | `/key/update` (org_id field) | `test_key_update.py` matrix asserts no row mutation on denied responses (`row.models != [MARKER]`) | by-inspection |
| 4 | `c7c3df2b02` siblings: `f6cd0a827a` | /key/update returns 404 (not 401) for nonexistent body key | `/key/update` | NOT covered by current matrix — the matrix only exercises existing target keys. Future: add 404-on-missing scenarios. | gap (filed below) |
| 5 | `133471f882` | double-counting bug in org/team key limit checks on update | `/key/update` (counting) | NOT directly covered; matrix asserts status only, not counts. Future: budget/limit assertions. | gap (filed below) |
| 6 | `574633fcf1` | exclude budget_limits from deleted verification token | `/key/delete` | `test_key_delete.py` matrix verifies post-delete authentication fails — would catch a shape-of-delete change but not a budget_limits-specific bug | by-inspection (partial) |
| 7 | `db8ef44323` | enforce upperbound_key_generate_params on /key/regenerate | `/key/regenerate` | `test_key_regenerate.py` matrix asserts status; a regen that exceeds upperbound limits would surface IF the test bodies passed disallowed params. Currently they don't. | gap (filed below) |
| 8 | `2220f3076a` | tighten caller-permission checks on key route fields | multiple | spans /key/generate + /key/update; partial overlap with our `team_id`/`user_id` boundary scenarios | by-inspection |
| 9 | `5190bd07eb` | extend caller-permission to service-account | /key/generate, /key/service-account/generate | service_account actor IS in our matrix; some sub-scenarios overlap | by-inspection |
| 10 | `12005c4a02` | /key/aliases auth | `/key/aliases` | not Tier-1 (out of PR1 scope; PR3 territory) | out of scope |
| 11 | `818c097ca9` | Self-exclusion hash mismatch | `/key/update` | partial overlap | by-inspection |
| 12 | `daf7c0c3a8` | virtual keys team filter | `/key/list` | `test_key_list.py` filter param scenarios deferred; default-visibility scenarios already in matrix | gap (deferred) |

## Verified replay: `c7c3df2b02`

**Fix subject**: `fix(proxy): extend /key/update admin check to non-budget fields`
**Parent SHA**: `662d05531d`
**Catching slice**: `tests/proxy_behavior/management/test_key_update.py`

### Parent (pre-fix) — RED

```
FAILED tests/proxy_behavior/management/test_key_update.py::test_key_update_authz_matrix[self/team_admin]
FAILED tests/proxy_behavior/management/test_key_update.py::test_key_update_authz_matrix[self/internal_user]
FAILED tests/proxy_behavior/management/test_key_update.py::test_key_update_authz_matrix[self/owner]
FAILED tests/proxy_behavior/management/test_key_update.py::test_key_update_authz_matrix[self/unrelated_same_org]
FAILED tests/proxy_behavior/management/test_key_update.py::test_key_update_authz_matrix[self/cross_org_user]
FAILED tests/proxy_behavior/management/test_key_update.py::test_key_update_authz_matrix[self/service_account]
6 failed, 15 passed
```

Each of the 6 self-target scenarios returned `200` under the parent code (the
admin check only gated budget/spend changes, so a non-admin could rewrite
`models`) but is pinned at `403` post-fix.

### HEAD (post-fix) — GREEN

```
21 passed
```

The matrix correctly flipped from 6/21 RED to 21/21 GREEN solely on the
handler swap — confirming the suite catches the c7c3df2b02 behavior change.

## Identified coverage gaps (deferred to PR2/PR3)

These cells in the replay table call out genuine matrix gaps — the current
PR1 surface does not pin the specific behavior the fix introduced. Each is
worth a follow-up scenario:

- **404-on-missing-key** (rows 4, paralleling `f6cd0a827a` and
  `19efe556cb`) — add an explicit "actor calls /key/update with a body
  `key` that does not exist" scenario per endpoint, asserting 404.
- **Budget/limit counting bugs** (row 5, `133471f882`) — add scenarios that
  read the team/org spend rows after a denied /key/update and assert no
  counter movement.
- **Upperbound enforcement on /key/regenerate** (row 7, `db8ef44323`) —
  extend the regenerate matrix with at least one scenario that requests
  params exceeding the team's `upperbound_key_generate_params`.
- **`/key/list` filter-param view** (row 12, `daf7c0c3a8`) — PR1 only
  pins default-visibility. Filter combinations (`team_id=`,
  `include_team_keys=true`, etc.) belong in a follow-up.

Filed as TODOs rather than blocking PR1: the matrix shape is correct,
these are scope extensions.
