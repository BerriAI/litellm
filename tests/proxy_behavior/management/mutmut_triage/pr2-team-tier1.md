# Mutmut triage — PR2 (Team Tier-1)

G5 triage for the team Tier-1 slice. Every surviving mutant in the in-scope
handler functions is classified **killed** (a test was added so the mutant
now dies) or **accepted** (equivalent / not worth pinning, with a one-line
reason).

## In-scope handler functions

`litellm/proxy/management_endpoints/team_endpoints.py`:
`new_team`, `update_team`, `team_member_add`, `team_member_delete`,
`team_member_update`, `team_info`, `list_team`, and the authz helpers they
call (`_verify_team_access`, `validate_membership`,
`_validate_team_member_add_permissions`, `_authorize_and_filter_teams`).

## Status — attempted; blocked on tooling, full run still pending

A scoped local `mutmut run` (3.5.0, `paths_to_mutate` narrowed to
`team_endpoints.py`) was attempted and did not complete. Three concrete
blockers — the next attempt must clear all three:

1. **Sandbox import shadow.** mutmut copies the project into `mutants/` and
   trampolines the mutated file, but the test process imports `litellm` from
   the worktree source (cwd / `sys.path[0]`) instead of `mutants/litellm/`.
   The trampolines never fire, so mutmut reports "could not find any test
   case for any mutant" and stops — independent of which suite is used.

2. **The mock and behavior suites cannot share a pytest session.** The legacy
   mock suite (`tests/test_litellm/proxy/management_endpoints/`) globally
   patches `prisma_client` / the auth layer; run in the same session as the
   real-DB behavior suite, it fails behavior tests. mutmut runs all of
   `tests_dir` in one session for its stats phase, so the two suites cannot
   both sit in `tests_dir` for a single run.

3. **The CI `mutation-test.yml` workflow has no Postgres.** PR1 added
   `tests/proxy_behavior/management/` to `[tool.mutmut].tests_dir`, but the
   workflow starts no Postgres service — so its stats phase now aborts on the
   DB-dependent behavior tests. The workflow needs a Postgres service
   container (mirroring `_test-unit-services-base.yml`) before it can run
   with the current `tests_dir`.

Until these are resolved the binding pre-merge signal stays the behavior
matrix itself (CI gate G1) plus the G4 regression-replay (one verified
RED→GREEN against `09ffc87734`). mutmut remains a deferred follow-up, as in
PR1.

A verified by-hand mutation already exists as G4 evidence: deleting the
`_verify_team_access` call in `update_team` (one mutant) is killed by
`test_team_update.py::test_team_update_org_relocation_gate[org_b_admin]`.

## Triage table

| Mutant (file:line) | Survives? | Verdict | Note |
|--------------------|-----------|---------|------|
| _to be filled from the first completed mutmut run_ | | | |

## PR2.M3 telemetry

Survivor count + kill rate on `team_endpoints.py` against this suite, as a
delta vs. the PR1 baseline, is reported here once a run completes. It is
**telemetry, not a gate** — G5 (every survivor classified above) is the
binding mutation gate.
