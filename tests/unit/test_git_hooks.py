"""Tests for the repo's git hook scripts in ``.githooks/``.

The commit-msg hook enforces Conventional Commits. It is exercised as a
subprocess against representative valid / invalid inputs so that any future
regex change or accidental edit gets caught by ``make test-unit``.

The hook is a bash script; the test is skipped on Windows where ``bash`` may
not be on PATH.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HOOKS_DIR = _REPO_ROOT / ".githooks"
_COMMIT_MSG_HOOK = _HOOKS_DIR / "commit-msg"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None,
    reason="bash not available; git hook scripts are bash-based",
)


@pytest.fixture(autouse=True)
def _ensure_hooks_exist():
    assert _COMMIT_MSG_HOOK.exists(), f"missing hook: {_COMMIT_MSG_HOOK}"
    # Exec bit may be missing on a fresh clone on case-preserving filesystems;
    # the installer normalizes this, but the test shouldn't depend on having
    # run it.
    mode = _COMMIT_MSG_HOOK.stat().st_mode
    if not (mode & 0o100):
        _COMMIT_MSG_HOOK.chmod(mode | 0o755)


def _run_commit_msg(subject: str, tmp_path: Path) -> subprocess.CompletedProcess:
    msg_file = tmp_path / "COMMIT_EDITMSG"
    msg_file.write_text(subject + "\n", encoding="utf-8")
    return subprocess.run(
        ["bash", str(_COMMIT_MSG_HOOK), str(msg_file)],
        capture_output=True,
        text=True,
        check=False,
    )


# ----- commit-msg -----------------------------------------------------------


@pytest.mark.parametrize(
    "subject",
    [
        "feat(router): add weighted round-robin strategy",
        "fix(bedrock): decouple STS region from aws_region_name",
        "chore(deps): bump black to 26.3.1",
        "docs: rewrite contributing guide",
        "refactor!: drop Python 3.8 support",
        "feat(api,proxy)!: rename endpoint",
        "test: cover hook bypass list",
        "perf(streaming): avoid extra json parse",
        "revert: feat(router): add weighted round-robin",
    ],
)
def test_commit_msg_accepts_conventional_subjects(tmp_path, subject):
    result = _run_commit_msg(subject, tmp_path)
    assert result.returncode == 0, (
        f"hook rejected a valid subject:\n  subject: {subject!r}\n"
        f"  stderr: {result.stderr}"
    )


@pytest.mark.parametrize(
    "subject",
    [
        "add stuff",                          # no type
        "feat add router strategy",           # missing colon
        "feat:add router strategy",           # missing space after colon
        "feat():",                            # empty description
        "ux: thing",                          # unknown type
        "Feat(router): capital type",         # types are lowercase
        "feat(router):",                      # empty description
        # Description must start with a lowercase letter — kept in sync with
        # the CI workflow's subjectPattern so the local hook never accepts a
        # subject that CI will later reject.
        "feat: Add thing",
        "fix(router): Decouple something",
        "chore: BUMP deps",
        "feat: A",
    ],
)
def test_commit_msg_rejects_invalid_subjects(tmp_path, subject):
    result = _run_commit_msg(subject, tmp_path)
    assert result.returncode == 1, (
        f"hook accepted an invalid subject:\n  subject: {subject!r}\n"
        f"  stderr: {result.stderr}"
    )
    assert "Conventional Commits" in result.stderr


@pytest.mark.parametrize(
    "subject",
    [
        # Lowercase letter — the common case.
        "feat: lowercase start is fine",
        # The CI's `^(?![A-Z]).+$` rejects only uppercase A-Z, so digits and
        # symbols are still allowed; mirror that behavior here.
        "feat: 1-based indexing now works",
        "fix(deps): @types/node bump",
    ],
)
def test_commit_msg_accepts_non_uppercase_starts(tmp_path, subject):
    result = _run_commit_msg(subject, tmp_path)
    assert result.returncode == 0, (
        f"hook rejected a valid non-uppercase-start subject:\n"
        f"  subject: {subject!r}\n  stderr: {result.stderr}"
    )


@pytest.mark.parametrize(
    "subject",
    [
        "Merge branch 'main' into feature/foo",
        'Revert "feat(router): add weighted round-robin strategy"',
        "fixup! feat(router): add weighted round-robin strategy",
        "squash! feat(router): add weighted round-robin strategy",
        "amend! feat(router): add weighted round-robin strategy",
    ],
)
def test_commit_msg_passes_git_generated_messages(tmp_path, subject):
    result = _run_commit_msg(subject, tmp_path)
    assert result.returncode == 0, (
        f"hook should pass git-generated subject:\n  subject: {subject!r}\n"
        f"  stderr: {result.stderr}"
    )


def test_commit_msg_rejects_empty_message(tmp_path):
    result = _run_commit_msg("", tmp_path)
    assert result.returncode == 1
    assert "empty commit message" in result.stderr


def test_commit_msg_skips_comment_only_lines(tmp_path):
    # An all-comments file has no subject — should be rejected.
    msg_file = tmp_path / "COMMIT_EDITMSG"
    msg_file.write_text("# please enter a commit message\n# above this line\n", encoding="utf-8")
    result = subprocess.run(
        ["bash", str(_COMMIT_MSG_HOOK), str(msg_file)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "empty commit message" in result.stderr


def test_commit_msg_uses_first_non_comment_line(tmp_path):
    # Real git-generated COMMIT_EDITMSG has a status block prefixed with '#'
    # below the subject. Make sure leading comment lines are skipped too.
    msg_file = tmp_path / "COMMIT_EDITMSG"
    msg_file.write_text(
        "# On branch feature/foo\n"
        "\n"
        "feat(router): add weighted round-robin\n"
        "\n"
        "# Please enter the commit message...\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", str(_COMMIT_MSG_HOOK), str(msg_file)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
