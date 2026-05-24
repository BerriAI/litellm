"""Static guardrails for the Agent Shin + Greptile workflow YAML files.

These workflows can post comments and close PRs/issues on
BerriAI/litellm, so the gating logic that decides "is this a real
close-on-fail run?" must fail-safe on any unexpected input. The risk
is mostly maintenance: someone edits the bash gate, drops a quote,
inverts a comparison, or uses `!= "false"` (which treats "True",
"yes", "1", and typos as enabling closure) and the regression isn't
caught until a real OSS contributor's PR gets auto-closed.

The tests below pin two invariants across every workflow that gates a
destructive `--close`:

  1. The gate uses the fail-safe `= "true"` comparison — not `!= "false"`,
     not `!= ""`. Only the literal string "true" should ever enable
     closure.
  2. The gate also requires `AGENT_SHIN_ENABLED = "true"` (or the
     scheduled-job equivalent) — disabling the variable must always
     force dry-run.

Static parsing of the YAML + bash text is the right level of test here:
the gating logic lives in a `run:` block, not in a Python module we can
import, and end-to-end testing a GitHub Actions workflow from CI is
infeasible. A YAML-level guardrail is exactly what would have caught
the original `!= "false"` regression at PR time.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# Map of workflow file -> the env var name that drives the destructive
# gate inside that workflow's `run:` block. Keeping this table explicit
# (rather than scraping every workflow file) means a new workflow file
# that bypasses the dry-run gating doesn't silently slip past this test.
DESTRUCTIVE_GATE_ENV: dict[str, str] = {
    "triage_pr_with_llm.yml": "DISPATCH_CLOSE",
    "triage_issue_with_llm.yml": "DISPATCH_CLOSE",
    "close_low_quality_prs.yml": "CLOSE_FLAG",
    "review_gate.yml": "CLOSE_FLAG",
    # The reconsider workflow has no per-run "really do it?" knob — its
    # only kill switch is `AGENT_SHIN_ENABLED`, which already serves as
    # both the destructive gate and the global enablement gate.
    "triage_reconsider.yml": "AGENT_SHIN_ENABLED",
}


def _load_workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS_DIR / name).read_text())


def _all_run_blocks(workflow: dict) -> list[str]:
    """Return every `run:` step's command text, joined."""
    commands: list[str] = []
    jobs = workflow.get("jobs") or {}
    for job in jobs.values():
        for step in job.get("steps", []) or []:
            if not isinstance(step, dict):
                continue
            run = step.get("run")
            if isinstance(run, str):
                commands.append(run)
    return commands


@pytest.mark.parametrize("workflow_file,env_var", sorted(DESTRUCTIVE_GATE_ENV.items()))
def test_should_use_failsafe_equals_true_comparison(
    workflow_file: str, env_var: str
) -> None:
    """The destructive `--close` gate must use `= "true"` (fail-safe), not
    `!= "false"` (which would treat "True", "yes", "1", or any typo as
    enabling closure).

    Both bare `${ENV_VAR}` and `${ENV_VAR:-false}` (with a default) are
    accepted forms — what matters is the comparison operator. The
    Greptile closer relies on an outer `AGENT_SHIN_ENABLED` gate so it
    can use the bare form; the Agent Shin workflows include `:-false`
    for defense in depth. Either is fine.
    """
    workflow = _load_workflow(workflow_file)
    text = "\n".join(_all_run_blocks(workflow))
    assert env_var in text, (
        f"{workflow_file} no longer references {env_var}; was the "
        "gating env var renamed without updating this test?"
    )
    accepted_patterns = (
        f'"${{{env_var}}}" = "true"',
        f'"${{{env_var}:-false}}" = "true"',
    )
    assert any(p in text for p in accepted_patterns), (
        f"{workflow_file} must gate the destructive --close flag on the "
        f'EXACT string "true" (one of: {accepted_patterns!r}). Mirror '
        'the Greptile closer pattern; do NOT use `!= "false"` which '
        'fail-opens on unknown values like "True", "yes", "1", or typos.'
    )
    forbidden_patterns = (
        f'"${{{env_var}}}" != "false"',
        f'"${{{env_var}:-false}}" != "false"',
        f'"${{{env_var}:-true}}" != "false"',
    )
    for forbidden in forbidden_patterns:
        assert forbidden not in text, (
            f"{workflow_file} uses the fail-open pattern {forbidden!r}. "
            'Switch to `= "true"` so unknown values stay dry-run.'
        )


@pytest.mark.parametrize("workflow_file", sorted(DESTRUCTIVE_GATE_ENV))
def test_should_require_agent_shin_enabled_for_close(workflow_file: str) -> None:
    """Every destructive gate must also gate on the global enablement
    variable, so flipping `AGENT_SHIN_ENABLED` off is a kill switch
    regardless of any per-run input.

    Two patterns are equally fine:
      - Positive: `[ "${AGENT_SHIN_ENABLED:-false}" = "true" ]` to enter
        the close branch (Agent Shin workflows).
      - Negative: `[ "${AGENT_SHIN_ENABLED:-false}" != "true" ]` then
        bail out / force dry-run (Greptile closer).

    What matters is that the comparison value is the literal "true";
    `!= "false"` or `= "1"` etc. would not be a true kill switch.
    """
    workflow = _load_workflow(workflow_file)
    text = "\n".join(_all_run_blocks(workflow))
    accepted_patterns = (
        '"${AGENT_SHIN_ENABLED:-false}" = "true"',
        '"${AGENT_SHIN_ENABLED:-false}" != "true"',
    )
    assert any(p in text for p in accepted_patterns), (
        f"{workflow_file} must gate destructive actions on "
        '`AGENT_SHIN_ENABLED = "true"` (or the inverted `!= "true"` '
        "guard that forces dry-run). Without this, an unset repo "
        "variable would not be treated as a kill switch."
    )
