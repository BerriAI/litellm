import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

ROOT: Final = Path(__file__).resolve().parents[2]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _commit(repo: Path, message: str) -> None:
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", message)


@pytest.fixture
def remote_and_clone(tmp_path: Path) -> tuple[Path, Path]:
    seed: Final = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init", "-q", "-b", "litellm_internal_staging")
    (seed / "scripts").mkdir()
    for name in (
        "default_branch.py",
        "budget_ratchet_check.py",
        "ruff_strict_gate.py",
        "type_discipline_gate.py",
        "test_quality_gate.py",
        "type_check_gate.py",
        "gate_slot_lock.py",
    ):
        shutil.copyfile(ROOT / "scripts" / name, seed / "scripts" / name)
    shutil.copyfile(ROOT / "Makefile", seed / "Makefile")
    (seed / "litellm").mkdir()
    (seed / "litellm" / "example.py").write_text("value = 0\n")
    (seed / "ruff-strict-budget.json").write_text('{"C901": {"limit": 1}}\n')
    _commit(seed, "staging base")
    _git(seed, "checkout", "-qb", "main")
    (seed / "litellm" / "example.py").write_text("value = 1\n")
    (seed / "ruff-strict-budget.json").write_text('{"C901": {"limit": 0}}\n')
    _commit(seed, "main base")
    remote: Final = tmp_path / "remote.git"
    _git(tmp_path, "clone", "-q", "--bare", str(seed), str(remote))
    _git(remote, "symbolic-ref", "HEAD", "refs/heads/litellm_internal_staging")
    repo: Final = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", "--single-branch", str(remote), str(repo))
    return remote, repo


def _resolve(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "default_branch.py"), *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


def _make(repo: Path, target: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["make", target, "LINT_DEP_INSTALL=", "LINT_DEP_BASE=", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        env={key: value for key, value in os.environ.items() if key != "BASE_REF"},
    )


def test_existing_single_branch_clone_follows_remote_switch(remote_and_clone: tuple[Path, Path]) -> None:
    remote, repo = remote_and_clone
    before: Final = _resolve(repo)
    assert before.returncode == 0, before.stderr
    assert before.stdout.strip() == "origin/litellm_internal_staging"
    _git(remote, "symbolic-ref", "HEAD", "refs/heads/main")
    after: Final = _resolve(repo)
    assert after.returncode == 0, after.stderr
    assert after.stdout.strip() == "origin/main"
    assert _git(repo, "rev-parse", "origin/main") == _git(remote, "rev-parse", "main")
    assert _git(repo, "symbolic-ref", "refs/remotes/origin/HEAD").endswith("/litellm_internal_staging")


@pytest.mark.parametrize("missing_head", [False, True])
def test_unverifiable_default_never_uses_cached_head(
    remote_and_clone: tuple[Path, Path],
    missing_head: bool,
) -> None:
    remote, repo = remote_and_clone
    if missing_head:
        _git(remote, "symbolic-ref", "HEAD", "refs/heads/missing")
    else:
        _git(repo, "remote", "set-url", "origin", str(remote / "missing"))
    result: Final = _resolve(repo)
    assert result.returncode != 0
    assert not result.stdout
    assert "explicit base ref" in result.stderr
    checked: Final = _make(repo, "lint-format-check-changed")
    assert checked.returncode != 0
    assert "No changed" not in checked.stdout


@pytest.mark.parametrize("base_ref", ["HEAD", "origin/litellm_internal_staging"])
def test_explicit_base_works_without_remote_access(
    remote_and_clone: tuple[Path, Path],
    base_ref: str,
) -> None:
    remote, repo = remote_and_clone
    _git(repo, "remote", "set-url", "origin", str(remote / "missing"))
    result: Final = _resolve(repo, "--base", base_ref)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == base_ref
    checked: Final = _make(repo, "lint-format-check-changed", f"BASE_REF={base_ref}")
    assert checked.returncode == 0, checked.stderr
    assert "No changed litellm Python files" in checked.stdout


def test_budget_ratchet_compares_against_new_default(remote_and_clone: tuple[Path, Path]) -> None:
    remote, repo = remote_and_clone
    _git(remote, "symbolic-ref", "HEAD", "refs/heads/main")
    resolved: Final = _resolve(repo)
    assert resolved.returncode == 0, resolved.stderr
    _git(repo, "checkout", "-qb", "litellm_feature", "origin/main")
    (repo / "ruff-strict-budget.json").write_text('{"C901": {"limit": 1}}\n')
    command: Final = [sys.executable, "scripts/budget_ratchet_check.py"]
    checked: Final = subprocess.run(command, cwd=repo, capture_output=True, text=True, check=False)
    assert checked.returncode == 1
    assert "limit raised 0 -> 1" in checked.stdout
    assert "base origin/main" in checked.stdout
    overridden: Final = subprocess.run(
        [*command, "--base", "origin/litellm_internal_staging"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert overridden.returncode == 0, overridden.stdout + overridden.stderr


def _freshness(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; "
            "from ci_cd.run_migration import _check_branch_freshness; "
            "_check_branch_freshness(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else None)",
            str(repo),
            *args,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_migration_freshness_refuses_stale_branch_after_switch(remote_and_clone: tuple[Path, Path]) -> None:
    remote, repo = remote_and_clone
    before: Final = _freshness(repo)
    assert before.returncode == 0, before.stderr
    assert "Branch freshness OK" in before.stdout
    _git(remote, "symbolic-ref", "HEAD", "refs/heads/main")
    after: Final = _freshness(repo)
    assert after.returncode == 3
    assert "1 commit(s) behind origin/main" in after.stderr
    overridden: Final = _freshness(repo, "litellm_internal_staging")
    assert overridden.returncode == 0, overridden.stderr
    _git(repo, "merge", "--ff-only", "origin/main")
    updated: Final = _freshness(repo)
    assert updated.returncode == 0, updated.stderr
    assert "up to date with origin/main" in updated.stdout


def test_migration_freshness_refuses_unavailable_remote(remote_and_clone: tuple[Path, Path]) -> None:
    remote, repo = remote_and_clone
    _git(repo, "remote", "set-url", "origin", str(remote / "missing"))
    result: Final = _freshness(repo)
    assert result.returncode == 3
    assert "Could not discover origin's default branch" in result.stderr
    explicit: Final = _freshness(repo, "litellm_internal_staging")
    assert explicit.returncode == 3
    assert "git fetch origin litellm_internal_staging" in explicit.stderr


@pytest.mark.parametrize(
    "gate",
    [
        "budget_ratchet_check",
        "ruff_strict_gate",
        "type_discipline_gate",
        "test_quality_gate",
        "type_check_gate",
    ],
)
def test_each_gate_refuses_an_unverifiable_default(remote_and_clone: tuple[Path, Path], gate: str) -> None:
    remote, repo = remote_and_clone
    _git(repo, "remote", "set-url", "origin", str(remote / "missing"))
    result: Final = subprocess.run(
        [sys.executable, f"scripts/{gate}.py"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Cannot verify the base branch against origin" in result.stderr


@pytest.mark.parametrize(
    "target", ["lint-format-check-changed", "lint-test-quality", "lint-test-quality-budget-update"]
)
def test_direct_make_target_fetches_default_once(remote_and_clone: tuple[Path, Path], target: str) -> None:
    _, repo = remote_and_clone
    trace: Final = repo.parent / "git-trace.jsonl"
    shutil.copyfile(ROOT / "scripts" / "check_test_quality.py", repo / "scripts" / "check_test_quality.py")
    shutil.copyfile(ROOT / "test-quality-budget.json", repo / "test-quality-budget.json")
    (repo / "tests").mkdir()
    result: Final = subprocess.run(
        ["make", "-o", "install-dev", target, "LINT_DEP_INSTALL=", "UV_RUN=env"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        env={**{key: value for key, value in os.environ.items() if key != "BASE_REF"}, "GIT_TRACE2_EVENT": str(trace)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    commands: Final = tuple(
        event["argv"][1:]
        for line in trace.read_text().splitlines()
        if (event := json.loads(line)).get("event") == "start"
    )
    assert sum(command[0] == "ls-remote" for command in commands) == 1
    assert sum(command[0] == "fetch" for command in commands) == 1
