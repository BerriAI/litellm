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
