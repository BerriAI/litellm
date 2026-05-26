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

# Workflows that consult a per-run user-input env var (e.g. CLOSE_FLAG,
# DISPATCH_CLOSE) before adding `--close`. Those gates must stay positive
# `= "true"` so typos like "True"/"yes"/"1" fail-closed to dry-run.
PER_RUN_GATE_ENV: dict[str, str] = {
    "triage_pr_with_llm.yml": "DISPATCH_CLOSE",
    "triage_issue_with_llm.yml": "DISPATCH_CLOSE",
    "close_low_quality_prs.yml": "CLOSE_FLAG",
    "review_gate.yml": "CLOSE_FLAG",
}

# Every workflow that can post comments or close PRs/issues must consult
# the `AGENT_SHIN_ENABLED` kill switch. Post-enactment the gate is
# inverted: bot is live by default, only forced into dry-run when the
# variable is literally "false". The reconsider workflow has no per-run
# knob — AGENT_SHIN_ENABLED IS its only gate — so it appears here but
# NOT in PER_RUN_GATE_ENV.
KILL_SWITCH_WORKFLOWS: tuple[str, ...] = (
    "triage_pr_with_llm.yml",
    "triage_issue_with_llm.yml",
    "close_low_quality_prs.yml",
    "review_gate.yml",
    "triage_reconsider.yml",
)


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


@pytest.mark.parametrize("workflow_file,env_var", sorted(PER_RUN_GATE_ENV.items()))
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


@pytest.mark.parametrize("workflow_file", sorted(KILL_SWITCH_WORKFLOWS))
def test_should_require_agent_shin_enabled_for_close(workflow_file: str) -> None:
    """Every destructive gate must consult ``AGENT_SHIN_ENABLED`` so the
    variable is a usable kill switch regardless of any per-run input.

    Post-enactment the variable is INVERTED — Agent Shin is live by default
    and the variable forces dry-run when explicitly set to ``"false"``. Two
    patterns are equally fine:

      - Positive enter-kill-switch branch:
        ``[ "${AGENT_SHIN_ENABLED:-true}" = "false" ]``
      - Negative bypass-kill-switch branch:
        ``[ "${AGENT_SHIN_ENABLED:-true}" != "false" ]``

    What matters is that the comparison value is the literal ``"false"``
    AND the default-when-unset is ``"true"`` (i.e. live). ``= "true"``
    against ``${AGENT_SHIN_ENABLED:-false}`` would re-introduce the
    pre-rollout dry-run-by-default semantics.
    """
    workflow = _load_workflow(workflow_file)
    text = "\n".join(_all_run_blocks(workflow))
    accepted_patterns = (
        '"${AGENT_SHIN_ENABLED:-true}" = "false"',
        '"${AGENT_SHIN_ENABLED:-true}" != "false"',
    )
    assert any(p in text for p in accepted_patterns), (
        f"{workflow_file} must gate destructive actions on the inverted "
        '`AGENT_SHIN_ENABLED` kill switch (`= "false"` or `!= "false"` '
        'against the `${AGENT_SHIN_ENABLED:-true}` default). Without this, '
        "an unset repo variable could be confused for an opt-in/opt-out."
    )
