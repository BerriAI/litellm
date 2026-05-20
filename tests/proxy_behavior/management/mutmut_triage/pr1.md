# G5 — Mutmut survivor triage for PR1 (Key Tier-1)

PR1 wires the new behavior suite into `[tool.mutmut].tests_dir` in
`pyproject.toml`. This file records the **first manually-triggered mutmut
run** that exercises `litellm/proxy/management_endpoints/key_management_endpoints.py`
against the new suite, and classifies every surviving mutant inside the 6
Tier-1 handler functions:

  * `generate_key_fn` (and helpers `_common_key_generation_helper`,
    `key_generation_check`, `_team_key_generation_check`,
    `_team_key_operation_team_member_check`, `_get_user_in_team`)
  * `update_key_fn` (and `_check_key_admin_access` helper)
  * `regenerate_key_fn` (and `_get_and_validate_existing_key` helper)
  * `info_key_fn`
  * `list_keys`
  * `delete_key_fn`

## How to run

```bash
# Manual trigger of the workflow:
gh workflow run mutation-test.yml --ref litellm_/silly-wright-1b8559
gh run watch
```

Or locally (~hours, plan around it):

```bash
uv run --with mutmut==3.5.0 mutmut run
```

Once the run completes, download `mutation-report.md` from the workflow
artifact and:

  1. Filter the survivors to the 6 handler functions listed above.
  2. For each surviving mutant in those functions, add a row to the table
     below classifying it **killed** (write a new test → mutant dies →
     rerun confirms) or **accepted** with a one-line reason (equivalent
     mutation, defensive-only branch, unreachable under any realistic
     world-seed).

G5 is the binding gate for the mutation signal: **zero unreviewed
survivors** in the Tier-1 handler set. The aggregate kill rate is recorded
as telemetry under "Baseline metrics" below — never as a goalpost.

## Baseline metrics (filled after the first mutmut run lands)

| Metric | Value | Notes |
|--------|-------|-------|
| Surviving mutants (all paths_to_mutate) | _TBD_ | |
| Surviving mutants in Tier-1 handler funcs | _TBD_ | G5 gate target: 0 unreviewed |
| Killed mutants | _TBD_ | |
| Kill rate (killed / (killed + survived)) | _TBD_ | Telemetry only |
| Wall-clock minutes | _TBD_ | |

## Survivor triage

| # | Mutant location (file:line, op) | Classification | Reason / new test |
|---|---------------------------------|----------------|-------------------|
| _none yet — first mutmut run pending_ | | | |

## PR2 / PR3 delta

When PR2 (Team Tier-1) ships, append a new "Baseline metrics" block here
keyed on the PR — never overwrite. The headline number to report is the
**delta** vs. PR1's baseline, not the absolute kill rate.
