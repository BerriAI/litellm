# Budget Test Coverage Matrix

Maps every row of `BUDGET_CODE_MATRIX.md` (what LiteLLM implements) to its tests
and level, then marks the live e2e coverage this suite adds.

Levels: `unit` mocked (`AsyncMock` on `get_current_spend`/prisma); `router` live
router with fake deployments; `live-e2e` real proxy, real key/team, real requests
until blocked. Status: `covered` / `partial` / `gap`.

Pre-existing live coverage outside this suite:
- `tests/otel_tests/test_e2e_budgeting.py` - key + team enforcement, budget update.
- `tests/local_testing/test_router_budget_limiter.py` - provider / tag / deployment
  budgets at the router.

This suite (`tests/e2e/budgets/`) adds the missing live coverage and runs
on the shared lifecycle (every entity it creates is deleted on teardown).

---

## Per-entity enforcement

| Entity | Unit | Pre-existing live | This suite (live) | Status |
|--------|------|-------------------|-------------------|--------|
| API key | `test_budget_reservation.py`, `test_max_budget_limiter.py` | `otel_tests` | `test_budget_enforcement_e2e::test_key_budget_blocks` | **covered** |
| Team | `test_team_budget_limits.py` | `otel_tests` | (org test builds a team) | **covered** |
| Internal user | auth unit tests | - | `test_internal_user_budget_blocks` | **covered (new)** |
| Team member | `test_team_member_budget.py` | - | `test_team_member_budget_blocks` | **covered (new)** |
| End-user / customer | `test_custom_auth_end_user_budget.py` | - | `test_end_user_budget_blocks` | **covered (new)** |
| Organization | `test_organization_budget_enforcement.py` (flagged weak) | - | `test_organization_budget_blocks` | **covered (new)** |
| Tag (proxy-level) | - | router only | `test_tag_budget_e2e::test_tag_budget_blocks_tagged_requests` | **covered (new)** |
| Model-level (`model_max_budget`) | `test_unit_test_max_model_budget_limiter.py` | - | `test_model_max_budget_e2e::test_model_max_budget_isolates_per_model` | **covered (new)** |
| Provider (router) | `test_budget_limiter_hotpath.py` | `test_router_budget_limiter.py` | - | **covered** (router) |
| Global proxy (`litellm.max_budget`) | unit | - | - | **gap** (needs a config-level cap; not key-settable) |

## Budget mechanisms

| Mechanism | Unit | This suite (live) | Status |
|-----------|------|-------------------|--------|
| Pre-call reservation | `test_budget_reservation.py` | exercised by every enforcement test | **partial** |
| Soft budget / alerts | `SlackAlerting/test_budget_alert_types.py` | `test_soft_budget_e2e::test_soft_budget_does_not_block` | **covered (new)** (block-vs-alert; the alert side-effect itself stays unit) |
| Budget CRUD | `test_budget_endpoints.py` | `test_budget_crud_e2e` (roundtrip + delete) | **covered (new)** |
| Reset scheduling | `test_proxy_budget_reset.py` | `test_budget_crud_e2e::test_budget_duration_schedules_reset_on_key` | **covered (new)** (scheduling; actual zeroing is time-dependent -> unit) |
| Multi-window budgets | `test_multi_budget_windows.py` | - | **gap** (window setup is fiddly; left to unit for now) |
| Read budget+spend | `test_spend_management_endpoints.py` | `/key/info` asserted in CRUD + enforcement | **partial** |

## Remaining gaps (intentionally not live-tested)

- **Global proxy budget** (`litellm.max_budget`): set via proxy config, not a
  per-key API, so it needs a dedicated proxy boot with that config rather than a
  runtime-created entity. Out of scope for the per-entity suite.
- **Multi-window budgets**: the `budget_limits` list shape and per-window reset are
  covered by `test_multi_budget_windows.py` (unit); a live version would need to
  wait out a short window to see the reset, which is time-dependent.
- **Soft-budget alert delivery**: whether the Slack/email actually fires is not
  observable from the proxy API; unit tests own that. The live test pins the
  load-bearing behavior (soft does not block).
- **Reset zeroing after the window elapses**: time-dependent; unit tests own the
  reset-job logic. The live test pins that `budget_reset_at` is scheduled.

## This suite's files

| File | Covers |
|------|--------|
| `test_budget_enforcement_e2e.py` | key / internal-user / end-user / organization / team-member hard enforcement |
| `test_model_max_budget_e2e.py` | per-model caps isolate by model |
| `test_soft_budget_e2e.py` | soft budget alerts but does not block |
| `test_tag_budget_e2e.py` | proxy-level tag budget blocks tagged requests, spares others |
| `test_budget_crud_e2e.py` | `/budget/*` CRUD roundtrip + delete + `budget_reset_at` scheduling |

## Pattern + timing

Create the entity with a tiny `max_budget`, drive spend until a `budget_exceeded`
block. The enforcement helper is two-phase: a fast warmup (key/user/org/member/tag/
model block within ~2 calls off real-time counters), then a poll across the ~60s
batch-write window (end-user enforcement reads table spend that lags). Skip on a
non-budget error (provider down / key missing); fail if the budget is never
enforced. Chat tests use `gpt-5.5` (the model with a working key on the reference
proxy); swap the literal if your proxy differs.
