import os
import shutil
import subprocess
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
import yaml
from pydantic import BaseModel, ConfigDict

CIRCLECI: Final = Path(__file__).resolve().parents[2] / ".circleci"
CONFIG: Final = CIRCLECI / "config.yml"
PUBLIC_REPO: Final = "https://github.com/BerriAI/litellm.git"
LINUX_SHELL: Final = ("bash", "-eo", "pipefail", "-c")
WINDOWS_SHELL: Final = ("bash", "-c")
PRODUCT_DIRS: Final = ("litellm", "enterprise", "litellm-proxy-extras")


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


class _NamedRun(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str = ""
    command: str


class _JobStep(BaseModel):
    model_config = ConfigDict(frozen=True)
    run: _NamedRun | str | None = None


class _Job(BaseModel):
    model_config = ConfigDict(frozen=True)
    steps: tuple[_JobStep | str, ...]


class _Jobs(BaseModel):
    model_config = ConfigDict(frozen=True)
    routing_parity: _Job


class _Config(BaseModel):
    model_config = ConfigDict(frozen=True)
    commands: _Commands
    jobs: _Jobs


class _TraceEvent(BaseModel):
    model_config = ConfigDict(frozen=True)
    event: str
    name: str = ""


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


def _config() -> _Config:
    return _Config.model_validate(yaml.safe_load(CONFIG.read_text()))


def _routing_parity_step(name: str) -> str:
    return next(
        step.run.command
        for step in _config().jobs.routing_parity.steps
        if isinstance(step, _JobStep) and isinstance(step.run, _NamedRun) and step.run.name == name
    )


def _fetches(trace: Path) -> int:
    events: Final = tuple(map(_TraceEvent.model_validate_json, trace.read_text().splitlines()))
    return sum(event.event == "cmd_name" and event.name == "fetch" for event in events)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _commit(work: Path, files: Mapping[str, str], message: str) -> str:
    for name, content in files.items():
        (work / name).parent.mkdir(parents=True, exist_ok=True)
        (work / name).write_text(content)
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "--allow-empty", "-m", message)
    return _git(work, "rev-parse", "HEAD")


def _new_upstream(tmp_path: Path) -> tuple[Path, Path]:
    remote: Final = tmp_path / "remote.git"
    work: Final = tmp_path / "work"
    work.mkdir()
    _git(tmp_path, "init", "-q", "--bare", str(remote))
    _git(remote, "config", "uploadpack.allowFilter", "true")
    _git(work, "init", "-q", "-b", "main")
    _git(work, "config", "user.email", "t@t")
    _git(work, "config", "user.name", "t")
    return remote, work


def _publish_two_commits(tmp_path: Path) -> _Upstream:
    remote, work = _new_upstream(tmp_path)
    older: Final = _commit(work, {"app.py": "x = 1\n"}, "pipeline commit")
    newer: Final = _commit(work, {"app.py": "x = 2\n"}, "main moved on")
    _git(work, "push", "-q", str(remote), "main")
    return _Upstream(url=remote.as_uri(), older=older, newer=newer)


def _publish_docs_only_pr(tmp_path: Path) -> _Upstream:
    remote, work = _new_upstream(tmp_path)
    main: Final = _commit(work, {"litellm_inference.py": "x\n"}, "base")
    _git(work, "checkout", "-q", "-b", "litellm_feature")
    head: Final = _commit(work, {"README.md": "# docs\n"}, "docs")
    _git(work, "push", "-q", str(remote), "main", "litellm_feature")
    return _Upstream(url=remote.as_uri(), older=main, newer=head)


def _product(version: str) -> dict[str, str]:
    return {f"{directory}/code.py": f"{version} {directory}\n" for directory in PRODUCT_DIRS}


def _publish_parity_base(tmp_path: Path, base_on_main: bool) -> _Upstream:
    remote, work = _new_upstream(tmp_path)
    fork_point: Final = _commit(work, _product("fork point"), "fork point")
    base: Final = _commit(work, _product("base"), "release")
    _git(work, "branch", "-q", "stable")
    _git(work, "reset", "-q", "--hard", base if base_on_main else fork_point)
    head: Final = _commit(work, _product("head"), "pipeline commit")
    _git(work, "push", "-q", str(remote), "main", "stable")
    return _Upstream(url=remote.as_uri(), older=base, newer=head)


def _stub(bin_dir: Path, name: str, script: str) -> None:
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / name).write_text(f"#!/usr/bin/env bash\n{script}\n")
    (bin_dir / name).chmod(0o755)


def _job_env(tmp_path: Path, upstream: _Upstream) -> dict[str, str]:
    return {
        "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_SSH_COMMAND": "false",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": f"url.{upstream.url}.insteadOf",
        "GIT_CONFIG_VALUE_0": PUBLIC_REPO,
    }


def _run_checkout(
    tmp_path: Path,
    upstream: _Upstream,
    revisions: Mapping[str, str],
    shell: tuple[str, ...] = LINUX_SHELL,
) -> subprocess.CompletedProcess[str]:
    job: Final = tmp_path / "job"
    job.mkdir()
    command: Final = _config().commands.checkout_over_https
    return subprocess.run(
        [*shell, command.steps[0].run.command],
        cwd=job,
        env={**_job_env(tmp_path, upstream), **revisions},
        capture_output=True,
        text=True,
    )


def test_every_clone_goes_over_https_so_no_job_needs_the_ssh_checkout_key() -> None:
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


@pytest.mark.parametrize("shell", (LINUX_SHELL, WINDOWS_SHELL), ids=("linux", "windows-bash-exe"))
def test_a_revision_that_is_not_a_full_sha_is_refused_before_git_runs(tmp_path: Path, shell: tuple[str, ...]) -> None:
    upstream: Final = _publish_two_commits(tmp_path)
    result: Final = _run_checkout(
        tmp_path, upstream, {"CIRCLE_SHA1": upstream.newer, "REQUESTED_REVISION": "main"}, shell
    )
    assert result.returncode != 0
    assert not (tmp_path / "job" / ".git").exists()


def test_a_docs_only_pr_job_still_skips_backend_work_after_the_checkout(tmp_path: Path) -> None:
    upstream: Final = _publish_docs_only_pr(tmp_path)
    _stub(tmp_path / "bin", "circleci-agent", 'echo "circleci-agent $*"')
    checkout: Final = _run_checkout(tmp_path, upstream, {"CIRCLE_SHA1": upstream.newer})
    assert checkout.returncode == 0, checkout.stderr
    path_filter: Final = subprocess.run(
        ["bash", str(CIRCLECI / "scripts" / "path_filter.sh"), "backend"],
        cwd=tmp_path / "job",
        env={**_job_env(tmp_path, upstream), "CIRCLE_PULL_REQUEST": "https://github.com/BerriAI/litellm/pull/1"},
        capture_output=True,
        text=True,
    )
    assert path_filter.returncode == 0, path_filter.stderr
    assert "circleci-agent step halt" in path_filter.stdout


@pytest.mark.parametrize("shell", (LINUX_SHELL, WINDOWS_SHELL), ids=("linux", "windows-bash-exe"))
@pytest.mark.parametrize(
    ("failing_step", "failures", "checkouts", "checked_out"),
    (("fetch", 2, 1, True), ("fetch", 3, 0, False), ("checkout", 2, 3, True), ("checkout", 3, 3, False)),
)
def test_a_failed_fetch_or_blob_download_is_retried_and_the_step_stops_on_its_error_once_retries_run_out(
    tmp_path: Path, shell: tuple[str, ...], failing_step: str, failures: int, checkouts: int, checked_out: bool
) -> None:
    upstream: Final = _publish_two_commits(tmp_path)
    calls: Final = tmp_path / "calls"
    _stub(tmp_path / "bin", "sleep", f'echo sleep >> "{calls}"')
    _stub(
        tmp_path / "bin",
        "git",
        f'''echo "$1" >> "{calls}"
if [ "$1" = {failing_step} ] && [ "$(grep -c '^{failing_step}$' "{calls}")" -le {failures} ]; then exit 128; fi
exec "{shutil.which("git")}" "$@"''',
    )
    result: Final = _run_checkout(tmp_path, upstream, {"CIRCLE_SHA1": upstream.older}, shell)
    assert (result.returncode == 0) is checked_out, result.stderr
    steps: Final = calls.read_text().splitlines()
    assert steps.count("fetch") == min(failures + 1, 3)
    assert steps.count("sleep") == steps.count("fetch") - 1
    assert steps.count("checkout") == checkouts
    head: Final = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path / "job", capture_output=True, text=True)
    assert (head.stdout.strip() == upstream.older) is checked_out


@pytest.mark.parametrize("base_on_main", (True, False), ids=("main-commit", "release-branch-commit"))
def test_routing_parity_downloads_the_base_files_in_one_fetch_instead_of_one_per_missing_file(
    tmp_path: Path, base_on_main: bool
) -> None:
    upstream: Final = _publish_parity_base(tmp_path, base_on_main)
    checkout: Final = _run_checkout(tmp_path, upstream, {"CIRCLE_SHA1": upstream.newer})
    assert checkout.returncode == 0, checkout.stderr
    job: Final = tmp_path / "job"
    (job / "litellm" / "rust_bridge").mkdir()
    (job / "litellm" / "rust_bridge" / "_native.abi3.so").write_text("built for this job\n")
    trace: Final = tmp_path / "trace.json"
    result: Final = subprocess.run(
        [*LINUX_SHELL, _routing_parity_step("Check out base product code")],
        cwd=job,
        env={**_job_env(tmp_path, upstream), "ROUTING_PARITY_BASE": upstream.older, "GIT_TRACE2_EVENT": str(trace)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert tuple((job / name).read_text() for name in _product("base")) == tuple(_product("base").values())
    assert _fetches(trace) == 1
    assert _git(job, "rev-parse", "HEAD") == upstream.newer
