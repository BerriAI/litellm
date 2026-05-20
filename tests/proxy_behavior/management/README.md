# Management-endpoint behavior-pinning suite

HTTP-boundary regression tests for
`litellm/proxy/management_endpoints/key_management_endpoints.py` (PR1 — Key
Tier-1). Runs against the real proxy app via in-process `httpx.ASGITransport`,
connected to a real Postgres pointed at by `DATABASE_URL`. **No mocks** —
auth runs, prisma runs, integrations run. Test bodies make HTTP calls and
assert at the API boundary.

The eventual goal (across PR1–PR3) is to pin every authorization /
cross-tenant / budget-bypass boundary on the key + team management
surfaces. PR1 covers six Tier-1 key endpoints (`/key/generate`, `/key/info`,
`/key/list`, `/key/update`, `/key/regenerate`, `/key/delete`). See the
[Notion plan](https://www.notion.so/36643b8acdab8128a581ced0f6a4744d) for
the full scope.

## Local repro

Identical to the three commands the CI workflow
(`.github/workflows/test-unit-proxy-mgmt-behavior.yml`, which delegates
to `_test-unit-services-base.yml`) runs:

```bash
# 1. Bring up Postgres
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

Whole-suite wall-time is ~6s on a warm cache (one ~1.5s session setup +
~0.01–0.04s per test). Re-running back-to-back produces identical pass
counts — the scratch-namespace teardown leaves no rows behind.

### Single scenario / inner loop

```bash
uv run pytest tests/proxy_behavior/management/test_key_update.py -k self/owner -v
```

## Layout

```
tests/proxy_behavior/management/
├── conftest.py                    # session ASGI client, world seed, scratch fixture
├── actors.py                      # 8-actor enum + seed_world() helper
├── test_smoke.py                  # liveness + key/generate de-risk smoke
├── test_world_seed.py             # every seeded actor key authenticates
├── test_scratch_teardown.py       # scratch namespace cleanup invariants
├── test_no_management_imports.py  # G3 — strict-import grep as a test
├── test_key_generate.py           # Slice 7 — actor × target matrix
├── test_key_info.py               # Slice 8
├── test_key_list.py               # Slice 9
├── test_key_update.py             # Slice 10
├── test_key_regenerate.py         # Slice 11
├── test_key_delete.py             # Slice 12
├── regression_replay/README.md    # G4 — fix-PR → catching-scenario mapping
└── mutmut_triage/pr1.md           # G5 — survivor classification protocol
```

## Conventions

- **Async fixture / loop scope.** `pyproject.toml` sets
  `asyncio_default_fixture_loop_scope = "session"`, but the default
  *test* loop scope is per-function. Add
  `pytestmark = pytest.mark.asyncio(loop_scope="session")` at the top
  of every test file so the AsyncClient and prisma connection (both
  session-scoped) share a loop with the test body.
- **Forbidden imports (G3).** No `from litellm.proxy.management_endpoints`,
  no `mock`/`patch` on `user_api_key_auth`. Enforced by
  `test_no_management_imports.py` as a pytest item.
- **Read-world vs scratch-world.** The `world` fixture seeds an immutable
  read-world under the `behavior-pin-` prefix; tests must not mutate
  those rows. The `scratch` fixture gives a per-test
  `scratch-<uuid>` prefix and tears down any row tagged with it.
  Write scenarios always tag their creates with `scratch.prefix`.
- **Behavior pinning, not behavior judging.** Expected status codes are
  pinned against current handler behavior. The suite's job is to make
  *changes* to that behavior visible — not to assert what the codes
  *should* be. Comments above each `_SCENARIOS` block call out
  surprising or potentially-buggy behaviors for human review.

## Gate evidence

PR1's evidence for each G1–G5 + PR1.M1–M3 gate lives in:

- **G1** — CI run on the PR's workflow `test-unit-proxy-mgmt-behavior` (green).
- **G2** — `pytest --durations=…` summary in the PR description (≤ 10 min).
- **G3** — `test_no_management_imports.py` is part of the suite itself.
- **G4** — `regression_replay/README.md`.
- **G5** — `mutmut_triage/pr1.md`, filled in after the first manually-triggered
  `mutation-test.yml` run.
- **PR1.M1** — total scenario count, this README's "Layout" section.
- **PR1.M2** — this README + the workflow YAML are the local-repro contract.
- **PR1.M3** — `mutmut_triage/pr1.md` "Baseline metrics" table.
