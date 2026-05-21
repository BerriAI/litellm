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

## Status — run pending (manual follow-up)

The mutmut run is **deferred to a manual follow-up**, matching how PR1 (key
Tier-1) handled G5: a whole-folder `mutation-test.yml` run takes hours and is
not on the per-PR critical path. The fast pre-merge signal is the behavior
matrix itself (CI gate G1); mutmut directs *where to add the next pin*.

Run it with:

```bash
gh workflow run mutation-test.yml
```

`[tool.mutmut].tests_dir` in `pyproject.toml` already includes
`tests/proxy_behavior/management/`, so the team matrix contributes to the
mutation signal.

## Triage table

| Mutant (file:line) | Survives? | Verdict | Note |
|--------------------|-----------|---------|------|
| _to be filled from the first post-PR2 mutmut run_ | | | |

## PR2.M3 telemetry

Survivor count + kill rate on `team_endpoints.py` against this suite, as a
delta vs. the PR1 baseline, is reported here once the run completes. It is
**telemetry, not a gate** — G5 (every survivor classified above) is the
binding mutation gate.
