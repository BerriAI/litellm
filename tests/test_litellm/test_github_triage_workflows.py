"""Static guardrails for the Agent Shin + Greptile workflow YAML files.

These workflows can post comments and close PRs/issues on
BerriAI/litellm, so the gating logic that decides "is this a real
close-on-fail run?" must fail-safe on any unexpected input. The risk
is mostly maintenance: someone edits the bash gate, drops a quote,
inverts a comparison, or uses `!= "false"` (which treats "True",
"yes", "1", and typos as enabling closure) and the regression isn't
caught until a real OSS contributor's PR gets auto-closed.

The tests below pin a set of invariants. The first two apply to every
workflow that gates a destructive `--close`:

  1. The gate uses the fail-safe `= "true"` comparison — not `!= "false"`,
     not `!= ""`. Only the literal string "true" should ever enable
     closure.
  2. The gate also requires `AGENT_SHIN_ENABLED = "true"` (or the
     scheduled-job equivalent) — disabling the variable must always
     force dry-run.

A third invariant covers every workflow that installs the OpenAI client.
These run with a write-scoped `GITHUB_TOKEN`, so a compromised package
release would execute in that context; the install must therefore come
from the hash-pinned `.github/scripts/triage-requirements.txt` via
`pip --require-hashes`, never a floating `pip install openai>=...`.

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
    "triage_issue_with_llm.yml": "DISPATCH_CLOSE",
    "close_low_quality_prs.yml": "CLOSE_FLAG",
    # The reconsider workflow has no per-run "really do it?" knob — its
    # only kill switch is `AGENT_SHIN_ENABLED`, which already serves as
    # both the destructive gate and the global enablement gate.
    "triage_reconsider.yml": "AGENT_SHIN_ENABLED",
}


# Privileged workflows that install the OpenAI client. They run with a
# write-scoped GITHUB_TOKEN, so the install must be hash-pinned: a poisoned
# release would otherwise execute in that context. A new workflow that
# installs the client must be added here and use the same pinned file.
LLM_CLIENT_INSTALLER_WORKFLOWS = (
    "triage_issue_with_llm.yml",
    "triage_reconsider.yml",
    "triage_rollout_heads_up.yml",
)

PINNED_INSTALL = "--require-hashes -r .github/scripts/triage-requirements.txt"
REQUIREMENTS_FILE = REPO_ROOT / ".github" / "scripts" / "triage-requirements.txt"


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
def test_should_use_failsafe_equals_true_comparison(workflow_file: str, env_var: str) -> None:
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
        f"{workflow_file} no longer references {env_var}; was the gating env var renamed without updating this test?"
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


@pytest.mark.parametrize("workflow_file", LLM_CLIENT_INSTALLER_WORKFLOWS)
def test_llm_client_install_is_hash_pinned(workflow_file: str) -> None:
    """Every privileged workflow installs the OpenAI client from the
    hash-pinned requirements file, never by floating version.

    A bare `pip install "openai>=1.40.0"` resolves to whatever PyPI serves
    at run time and executes during install/import while a write-scoped
    `GITHUB_TOKEN` is in scope, so a compromised release runs in a
    privileged context. This test fails if that floating form comes back or
    if the `--require-hashes` install is loosened.
    """
    blocks = _all_run_blocks(_load_workflow(workflow_file))
    assert PINNED_INSTALL in "\n".join(blocks), (
        f"{workflow_file} must install the client via `pip install "
        f"{PINNED_INSTALL}`; a floating install runs unverified code with a "
        "write-scoped token."
    )
    offenders = [b for b in blocks if "pip install" in b and "openai" in b]
    assert not offenders, (
        f"{workflow_file} installs openai by name ({offenders!r}); pin it "
        "through the hash-locked requirements file so the version and "
        "checksum are fixed."
    )


def test_triage_requirements_are_fully_hash_pinned() -> None:
    """The shared requirements file pins every package to an exact version
    with a sha256 hash, which is what `pip --require-hashes` enforces at
    install time. A loosened pin or a missing hash here would silently widen
    the supply-chain surface for all the installer workflows.
    """
    assert REQUIREMENTS_FILE.exists(), (
        f"the hash-pinned requirements file the triage workflows install from is missing at {REQUIREMENTS_FILE}"
    )
    joined = REQUIREMENTS_FILE.read_text().replace("\\\n", " ")
    entries = [line.strip() for line in joined.splitlines() if line.strip() and not line.strip().startswith("#")]
    assert any(e.split()[0].startswith("openai==") for e in entries), (
        "openai must be pinned to an exact version in the triage requirements"
    )
    for entry in entries:
        spec = entry.split()[0]
        assert "==" in spec, (
            f"requirement {spec!r} is not pinned to an exact version; "
            "--require-hashes needs every package pinned with =="
        )
        assert "--hash=sha256:" in entry, (
            f"requirement {spec!r} has no sha256 hash; every pin must carry "
            "checksums so --require-hashes can verify the download"
        )


def _heads_up_run_step() -> dict:
    workflow = _load_workflow("triage_rollout_heads_up.yml")
    for step in workflow["jobs"]["heads-up"]["steps"]:
        if isinstance(step.get("run"), str) and "triage_rollout_heads_up.py" in step["run"]:
            return step
    raise AssertionError("no run step invokes triage_rollout_heads_up.py")


def test_rollout_heads_up_push_trigger_never_posts() -> None:
    """Merging the heads-up script to staging must stay inert: the automatic
    push trigger only ever runs dry-run. The real one-shot sweep is a
    deliberate manual `workflow_dispatch` with `dry_run=false`, the sole path
    that adds `--close`.

    This guards the "inert by default" invariant for the one workflow that is
    intentionally not gated on AGENT_SHIN_ENABLED (it has to warn contributors
    before that flag flips on). A regression to auto-`--close`-on-push would
    post real comments on every push that touches the script.
    """
    run = _heads_up_run_step()["run"]
    assert '"${GITHUB_EVENT_NAME:-}" = "workflow_dispatch"' in run, (
        "the real (--close) run must be a manual workflow_dispatch, not the automatic push trigger"
    )
    assert '"${DRY_RUN_INPUT:-true}" = "false"' in run, (
        "the real run must require the dry_run input to be the exact string 'false' (fail-safe); any other value stays dry-run"
    )
    assert run.count("ARGS+=(--close)") == 1, (
        "--close must appear once, inside the manual real-run branch; a second occurrence means the push path posts real comments on merge"
    )


def test_rollout_heads_up_key_is_dispatch_gated() -> None:
    """OPENAI_API_KEY is exposed only on the manual dispatch (the real-run
    trigger), never unconditionally. The sibling triage workflows gate the key
    the same way; an unconditional `secrets.OPENAI_API_KEY` here would hand the
    key to the automatic push run, which must stay a no-op dry-run preview.
    """
    key_expr = (_heads_up_run_step().get("env") or {}).get("OPENAI_API_KEY", "")
    assert "github.event_name == 'workflow_dispatch'" in key_expr, (
        f"OPENAI_API_KEY must be gated on workflow_dispatch so the automatic push trigger gets no key; found: {key_expr!r}"
    )


def _reconsider_steps() -> list[dict]:
    workflow = _load_workflow("triage_reconsider.yml")
    return workflow["jobs"]["reconsider"]["steps"]


def _index_of_run_step(steps: list[dict], needle: str) -> int:
    for i, step in enumerate(steps):
        run = step.get("run")
        if isinstance(run, str) and needle in run:
            return i
    raise AssertionError(f"no run step contains {needle!r}")


def _reaction_steps(steps: list[dict], content: str) -> list[tuple[int, dict]]:
    return [
        (i, s)
        for i, s in enumerate(steps)
        if isinstance(s.get("run"), str) and f"content={content}" in s["run"] and "/reactions" in s["run"]
    ]


def test_reconsider_routes_open_prs_to_review_gate() -> None:
    """`@agent-shin reconsider` on an OPEN PR must run the review gate (label
    flip) rather than `--reconsider` (which skips non-closed items) — without
    this branch the advertised "reconsider can change the tag" flow silently
    does nothing.
    """
    steps = _reconsider_steps()
    run = steps[_index_of_run_step(steps, "triage_with_llm.py")]["run"]
    assert '"${STATE}" = "open"' in run, (
        "the reconsider run step must branch on the PR's open/closed state"
    )
    assert "--review-gate" in run, (
        "open-PR reconsiders must invoke the review gate so the label pair flips"
    )
    assert "--reconsider" in run, "closed PRs/issues must still use --reconsider"


def test_lite_mode_gates_are_exact_string_matches() -> None:
    """Lite mode must only engage on the EXACT string "lite" — any other value
    (typos, "Lite", "true") falls through to full behavior, mirroring the
    fail-safe `= "true"` convention used for AGENT_SHIN_ENABLED. Both the
    review gate and the reconsider workflow carry the branch.
    """
    for workflow_file in ("review_gate.yml", "triage_reconsider.yml"):
        text = "\n".join(_all_run_blocks(_load_workflow(workflow_file)))
        assert '"${AGENT_SHIN_MODE:-}" = "lite"' in text, (
            f"{workflow_file} must select lite mode with an exact-string "
            'comparison against "lite"'
        )
        assert "--notice-only" in text, (
            f"{workflow_file} must pass --notice-only when AGENT_SHIN_MODE=lite"
        )


class TestReconsiderReactions:
    """The reconsider workflow acknowledges the triggering comment with a 👀
    reaction the moment it accepts the trigger, and a 👍 once the run finishes,
    so the contributor gets feedback immediately instead of waiting on a cron.

    Both reactions are gated on `AGENT_SHIN_ENABLED == 'true'` so a dry-run
    leaves no visible trace, and both target the comment that fired the event
    (`github.event.comment.id`). The ordering (👀 before the triage run, 👍
    after) is the whole point — these tests fail if a refactor reorders the
    steps, drops a reaction, or stops gating them.
    """

    def test_eyes_reaction_is_posted_before_the_triage_run(self) -> None:
        steps = _reconsider_steps()
        run_idx = _index_of_run_step(steps, "triage_with_llm.py")
        eyes = _reaction_steps(steps, "eyes")
        assert len(eyes) == 1, "expected exactly one 👀 (eyes) reaction step"
        idx, step = eyes[0]
        assert idx < run_idx, "👀 must be posted BEFORE the slow triage run, not after"
        assert "github.event.comment.id" in (step.get("env") or {}).get("COMMENT_ID", ""), (
            "👀 must react to the comment that triggered the workflow"
        )
        assert "${COMMENT_ID}" in step["run"], "👀 must react to the triggering comment, not a hardcoded id"
        assert "vars.AGENT_SHIN_ENABLED == 'true'" in step["if"], (
            "👀 must be gated on AGENT_SHIN_ENABLED so dry-run stays inert"
        )

    def test_thumbs_up_reaction_is_posted_after_a_successful_run(self) -> None:
        steps = _reconsider_steps()
        run_idx = _index_of_run_step(steps, "triage_with_llm.py")
        thumbs = _reaction_steps(steps, "+1")
        assert len(thumbs) == 1, "expected exactly one 👍 (+1) reaction step"
        idx, step = thumbs[0]
        assert idx > run_idx, "👍 must come AFTER the triage run"
        assert "success()" in step["if"], "👍 must only fire when the reconsider run succeeded"
        assert "vars.AGENT_SHIN_ENABLED == 'true'" in step["if"], (
            "👍 must be gated on AGENT_SHIN_ENABLED so dry-run stays inert"
        )
