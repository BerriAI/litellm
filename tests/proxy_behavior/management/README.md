# Management-endpoint behavior-pinning suite

HTTP-boundary behavior tests for `litellm/proxy/management_endpoints/`, run
against a real Postgres. The auth layer runs for real — it is **never**
mocked. The goal is catching unintentional behavior changes (authorization,
cross-tenant isolation, budget bypass), not a coverage number.

Each test parametrizes `(actor, target, expected_status)` tuples and drives
the public HTTP endpoints. Status codes are pinned to **observed** handler
behavior; surprising results are surfaced in comments, not "fixed" in the
test. A scenario flipping is the signal — investigate the diff, don't loosen
the assertion.

## Scope

| Slice | Endpoints | Tests |
|-------|-----------|-------|
| Key Tier-1 (PR1) | `/key/generate`, `/key/info`, `/key/list`, `/key/update`, `/key/regenerate`, `/key/delete` | `test_key_*.py` |
| Team Tier-1 (PR2) | `/team/new`, `/team/info`, `/team/list`, `/team/update`, `/team/member_add`, `/team/member_delete`, `/team/member_update` | `test_team_*.py` |

## Layout

- `conftest.py` — session-scoped ASGI client + prisma; `create_scratch_key` /
  `create_scratch_team` helpers; the function-scoped `scratch` namespace and
  its prefix-truncate teardown.
- `actors.py` — the immutable read-world seed: 2 orgs, 3 teams, 9 role-scoped
  actors with real `hash_token`-ed keys.
- `test_no_management_imports.py` — codifies G3 (see below).
- `mutmut_triage/` — G5 survivor triage docs, one per slice.

## Running locally

BYO Postgres via `DATABASE_URL` — same three commands CI runs, verbatim:

```bash
# 1. Start Postgres (any reachable instance works)
docker run --rm -d --name litellm-test-pg \
  -e POSTGRES_USER=litellm -e POSTGRES_PASSWORD=litellm -e POSTGRES_DB=litellm_test \
  -p 5432:5432 postgres:14

# 2. Migrate the schema (one-time per fresh DB)
export DATABASE_URL=postgresql://litellm:litellm@localhost:5432/litellm_test
uv run prisma generate --schema litellm/proxy/schema.prisma
uv run prisma db push --schema litellm/proxy/schema.prisma --accept-data-loss

# 3. Run the suite
uv run pytest tests/proxy_behavior/management/
```

Single scenario, fast inner loop (the world seed is session-scoped, so
re-runs are sub-second):

```bash
uv run pytest tests/proxy_behavior/management/test_team_info.py -v
```

CI wires the same steps into `.github/workflows/test-unit-proxy-mgmt-behavior.yml`
(`on: pull_request`), so every PR proves the suite runs even if local dev
never happens.

## Exit gates

- **G1** — the new CI job is green on the PR, no skipped tests.
- **G2** — the job stays ≤ 10 min wall-clock. xdist is disabled (`workers: 0`)
  because the world seed is one shared Postgres state.
- **G3** — strict imports. Zero matches for either, enforced by
  `test_no_management_imports.py`:
  ```bash
  rg 'from litellm\.proxy\.management_endpoints' tests/proxy_behavior/
  rg 'mock.*user_api_key_auth|patch.*user_api_key_auth' tests/proxy_behavior/
  ```
- **G4** — regression replay. For each in-scope behavior-fix PR, reverting the
  fix turns a scenario RED; restoring it turns it GREEN. Reverse-apply the
  fix's source hunk, run the test, observe RED, `git checkout` the file,
  observe GREEN.
- **G5** — mutmut triage. Every surviving mutant in the slice's in-scope
  handler functions is classified killed or accepted-with-reason in
  `mutmut_triage/<pr_slug>.md`. The whole-folder run takes hours and is not on
  the per-PR critical path; it is a manual follow-up (`gh workflow run
  mutation-test.yml`).

## Deferred coverage gaps

- **`/team/update` org-relocation, allowed branch.** The relocation gate's
  *allow* path for a non-proxy-admin needs a caller who is org admin of both
  the source and destination org; no seeded actor is. The deny paths are
  pinned (`test_team_update.py::test_team_update_org_relocation_gate`).
- Key-slice gaps filed by PR1 (404-on-missing-key, regenerate upperbound,
  `/key/list` filter params) remain open for a later slice.
