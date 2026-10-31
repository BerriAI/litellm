import os
import subprocess
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
import yaml
from pydantic import BaseModel, ConfigDict

CONFIG: Final = Path(__file__).resolve().parents[2] / ".circleci" / "config.yml"
PUBLIC_REPO: Final = "https://github.com/BerriAI/litellm.git"


class _Run(BaseModel):
    model_config = ConfigDict(frozen=True)
    command: str


class _Step(BaseModel):
    model_config = ConfigDict(frozen=True)
    run: _Run


class _Command(BaseModel):
    model_config = ConfigDict(frozen=True)
    steps: tuple[_Step]


class _Commands(BaseModel):
    model_config = ConfigDict(frozen=True)
    checkout_over_https: _Command


class _Config(BaseModel):
    model_config = ConfigDict(frozen=True)
    commands: _Commands


@dataclass(frozen=True, slots=True)
class _Upstream:
    url: str
    older: str
    newer: str


def _step_name(step: object) -> str:
    if isinstance(step, str):
        return step
    if isinstance(step, dict) and len(step) == 1:
        return str(next(iter(step)))
    return ""


def _step_names(node: object) -> Iterator[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("steps", "pre-steps", "post-steps") and isinstance(value, list):
                yield from map(_step_name, value)
            yield from _step_names(value)
    if isinstance(node, list):
        for item in node:
            yield from _step_names(item)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _commit(work: Path, message: str) -> str:
    _git(work, "commit", "-q", "--allow-empty", "-m", message)
    return _git(work, "rev-parse", "HEAD")


def _publish_two_commits(tmp_path: Path) -> _Upstream:
    remote: Final = tmp_path / "remote.git"
    work: Final = tmp_path / "work"
    work.mkdir()
    _git(tmp_path, "init", "-q", "--bare", str(remote))
    _git(work, "init", "-q", "-b", "main")
    _git(work, "config", "user.email", "t@t")
    _git(work, "config", "user.name", "t")
    older: Final = _commit(work, "pipeline commit")
    newer: Final = _commit(work, "main moved on")
    _git(work, "push", "-q", str(remote), "main")
    return _Upstream(url=remote.as_uri(), older=older, newer=newer)


def _run_checkout(tmp_path: Path, upstream: _Upstream, revisions: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    job: Final = tmp_path / "job"
    job.mkdir()
    command: Final = _Config.model_validate(yaml.safe_load(CONFIG.read_text())).commands.checkout_over_https
    return subprocess.run(
        ["bash", "-eo", "pipefail", "-c", command.steps[0].run.command],
        cwd=job,
        env={
            "PATH": os.environ["PATH"],
            "HOME": str(tmp_path),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_SSH_COMMAND": "false",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": f"url.{upstream.url}.insteadOf",
            "GIT_CONFIG_VALUE_0": PUBLIC_REPO,
            **revisions,
        },
        capture_output=True,
        text=True,
    )


def test_every_clone_goes_over_https_because_the_project_has_no_ssh_checkout_key() -> None:
    names: Final = tuple(_step_names(yaml.safe_load(CONFIG.read_text())))
    assert "checkout_over_https" in names
    assert "checkout" not in names


@pytest.mark.parametrize(
    "revisions",
    (
        {"CIRCLE_SHA1": "older"},
        {"CIRCLE_SHA1": "older", "REQUESTED_REVISION": ""},
        {"CIRCLE_SHA1": "newer", "REQUESTED_REVISION": "older"},
    ),
)
def test_a_job_checks_out_its_own_commit_from_the_public_https_url_after_main_moves(
    tmp_path: Path, revisions: Mapping[str, str]
) -> None:
    upstream: Final = _publish_two_commits(tmp_path)
    shas: Final = {"older": upstream.older, "newer": upstream.newer, "": ""}
    result: Final = _run_checkout(tmp_path, upstream, {name: shas[value] for name, value in revisions.items()})
    assert result.returncode == 0, result.stderr
    assert _git(tmp_path / "job", "rev-parse", "HEAD") == upstream.older


def test_a_revision_that_is_not_a_full_sha_is_refused_before_git_runs(tmp_path: Path) -> None:
    upstream: Final = _publish_two_commits(tmp_path)
    result: Final = _run_checkout(tmp_path, upstream, {"CIRCLE_SHA1": upstream.newer, "REQUESTED_REVISION": "main"})
    assert result.returncode != 0
    assert not (tmp_path / "job" / ".git").exists()
