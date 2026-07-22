"""Regression tests for CircleCI change-based job gating.

`.circleci/scripts/classify_changes.sh` is the pure decision function behind
`path_filter.sh`: given the list of files a PR changed (on stdin) and a job
category, it prints `run` or `skip`. The gating contract we lock in here:

  * docs-only changes (``*.md``, ``*.mdx``, ``docs/``) run nothing
  * client-only changes (``ui/``) run client jobs but skip backend jobs
  * any backend change runs both client and backend jobs

If this logic silently regresses, real test jobs get skipped, so these cases
are the guardrail against that.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / ".circleci" / "scripts"
SCRIPT = SCRIPTS_DIR / "classify_changes.sh"
PATH_FILTER = SCRIPTS_DIR / "path_filter.sh"


def classify(category: str, changed: list[str]) -> str:
    result = subprocess.run(
        ["bash", str(SCRIPT), category],
        input="\n".join(changed),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


DOCS = ["README.md", "docs/my_website/index.mdx", "litellm/anywhere.md"]
CLIENT = ["ui/litellm-dashboard/src/App.tsx"]
BACKEND = ["litellm/main.py"]


@pytest.mark.parametrize(
    "category,changed,expected",
    [
        # docs-only: skip everything
        ("backend", DOCS, "skip"),
        ("client", DOCS, "skip"),
        ("backend", [], "skip"),
        ("client", [], "skip"),
        # client-only: backend skips, client runs
        ("backend", CLIENT, "skip"),
        ("client", CLIENT, "run"),
        ("backend", CLIENT + DOCS, "skip"),
        ("client", CLIENT + DOCS, "run"),
        # any backend change: both run ("backend runs both")
        ("backend", BACKEND, "run"),
        ("client", BACKEND, "run"),
        ("backend", BACKEND + DOCS, "run"),
        ("client", BACKEND + DOCS, "run"),
        ("backend", BACKEND + CLIENT, "run"),
        ("client", BACKEND + CLIENT, "run"),
    ],
)
def test_classify_decisions(category: str, changed: list[str], expected: str) -> None:
    assert classify(category, changed) == expected


def test_markdown_under_ui_counts_as_client_not_docs() -> None:
    assert classify("client", ["ui/litellm-dashboard/README.md"]) == "run"
    assert classify("backend", ["ui/litellm-dashboard/README.md"]) == "skip"


def test_non_docs_directory_with_docs_in_name_is_backend() -> None:
    assert classify("backend", ["documentation_tests/foo.py"]) == "run"


def test_unknown_category_fails_open_to_run() -> None:
    assert classify("mystery", DOCS) == "run"


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _pr_repo(tmp_path: Path, feature_files: dict[str, str]) -> Path:
    """A repo whose HEAD is a feature branch off `main` with `feature_files` changed."""
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    work = tmp_path / "work"
    work.mkdir()
    _git(work, "init", "-q", "-b", "main")
    _git(work, "config", "user.email", "t@t")
    _git(work, "config", "user.name", "t")
    _git(work, "remote", "add", "origin", str(remote))
    (work / "litellm_core.py").write_text("x\n")
    _git(work, "add", ".")
    _git(work, "commit", "-qm", "base")
    _git(work, "push", "-q", "origin", "main")
    _git(work, "checkout", "-q", "-b", "litellm_feature")
    for rel, content in feature_files.items():
        target = work / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    _git(work, "add", "-A")
    _git(work, "commit", "-qm", "feature")
    return work


def _run_path_filter(work: Path, tmp_path: Path, category: str, scripts_dir: Path, is_pr: bool = True):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "circleci-agent"
    stub.write_text("#!/usr/bin/env bash\necho \"[stub] circleci-agent $*\"\nexit 0\n")
    stub.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env.pop("CIRCLE_PULL_REQUEST", None)
    if is_pr:
        env["CIRCLE_PULL_REQUEST"] = "https://github.com/x/y/pull/1"
    return subprocess.run(
        ["bash", str(scripts_dir / "path_filter.sh"), category],
        cwd=work,
        capture_output=True,
        text=True,
        env=env,
    )


def test_path_filter_halts_docs_only_pr(tmp_path: Path) -> None:
    work = _pr_repo(tmp_path, {"README.md": "# docs\n"})
    result = _run_path_filter(work, tmp_path, "backend", SCRIPTS_DIR)
    assert result.returncode == 0
    assert "circleci-agent step halt" in result.stdout


def test_path_filter_runs_backend_pr(tmp_path: Path) -> None:
    work = _pr_repo(tmp_path, {"litellm/new.py": "y\n"})
    result = _run_path_filter(work, tmp_path, "backend", SCRIPTS_DIR)
    assert result.returncode == 0
    assert "running job" in result.stdout
    assert "halt" not in result.stdout


def test_path_filter_fails_open_when_not_a_pr(tmp_path: Path) -> None:
    work = _pr_repo(tmp_path, {"README.md": "# docs\n"})
    result = _run_path_filter(work, tmp_path, "backend", SCRIPTS_DIR, is_pr=False)
    assert result.returncode == 0
    assert "not a pull request" in result.stdout
    assert "halt" not in result.stdout


def test_path_filter_fails_open_when_classifier_errors(tmp_path: Path) -> None:
    """Regression: a broken classifier must run the job, never silently halt it."""
    broken_scripts = tmp_path / "broken_scripts"
    broken_scripts.mkdir()
    shutil.copy(PATH_FILTER, broken_scripts / "path_filter.sh")
    (broken_scripts / "classify_changes.sh").write_text("#!/usr/bin/env bash\nexit 1\n")
    (broken_scripts / "classify_changes.sh").chmod(0o755)

    work = _pr_repo(tmp_path, {"README.md": "# docs\n"})
    result = _run_path_filter(work, tmp_path, "backend", broken_scripts)
    assert result.returncode == 0
    assert "classify_changes.sh failed" in result.stdout
    assert "halt" not in result.stdout
