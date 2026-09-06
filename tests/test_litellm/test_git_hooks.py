"""Tests for the repo's git hook scripts in ``.githooks/``.

The hooks enforce Conventional Commits 1.0.0 on the commit-msg path and
Conventional Branches on the pre-push path. Each hook is exercised here as a
subprocess against representative valid / invalid inputs so that any future
regex change or accidental edit gets caught by ``make test-unit``.

The hooks are POSIX-ish bash scripts; the test is skipped on Windows where
``bash`` may not be on PATH.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HOOKS_DIR = _REPO_ROOT / ".githooks"
_COMMIT_MSG_HOOK = _HOOKS_DIR / "commit-msg"
_PRE_PUSH_HOOK = _HOOKS_DIR / "pre-push"

_ZERO_OID = "0" * 40
_NONZERO_OID = "abc123abc123abc123abc123abc123abc123abc1"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None,
    reason="bash not available; git hook scripts are bash-based",
)


@pytest.fixture(autouse=True)
def _ensure_hooks_exist():
    assert _COMMIT_MSG_HOOK.exists(), f"missing hook: {_COMMIT_MSG_HOOK}"
    assert _PRE_PUSH_HOOK.exists(), f"missing hook: {_PRE_PUSH_HOOK}"
    # Exec bit may be missing on a fresh clone on case-preserving filesystems;
    # the installer normalizes this, but the test shouldn't depend on having
    # run it.
    for hook in (_COMMIT_MSG_HOOK, _PRE_PUSH_HOOK):
        mode = hook.stat().st_mode
        if not (mode & 0o100):
            hook.chmod(mode | 0o755)


def _run_commit_msg(subject: str, tmp_path: Path) -> subprocess.CompletedProcess:
    msg_file = tmp_path / "COMMIT_EDITMSG"
    msg_file.write_text(subject + "\n", encoding="utf-8")
    return subprocess.run(
        ["bash", str(_COMMIT_MSG_HOOK), str(msg_file)],
        capture_output=True,
        text=True,
        check=False,
    )


def _run_pre_push(stdin: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(_PRE_PUSH_HOOK)],
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )


def _ref_line(branch: str, local_oid: str = _NONZERO_OID, remote_oid: str = _ZERO_OID) -> str:
    ref = f"refs/heads/{branch}"
    return f"{ref} {local_oid} {ref} {remote_oid}\n"


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


# ----- pre-push -------------------------------------------------------------


@pytest.mark.parametrize(
    "branch",
    [
        "feature/weighted-round-robin",
        "bugfix/streaming-empty-chunks",
        "hotfix/auth-bypass",
        "release/v1.45.0",
        "chore/bump-deps",
        "feature/nested/path/ok",  # nested slashes after type are fine
    ],
)
def test_pre_push_accepts_conventional_branches(branch):
    result = _run_pre_push(_ref_line(branch))
    assert result.returncode == 0, (
        f"hook rejected a valid branch:\n  branch: {branch!r}\n"
        f"  stderr: {result.stderr}"
    )


@pytest.mark.parametrize(
    "branch",
    [
        "random-branch-name",
        "litellm_fix/optimize-streaming",  # legacy pattern is now rejected
        "ui/navbar-notifications",         # not in the allow list
        "feature/",                        # empty description
        "Feature/foo",                     # type is case-sensitive
        "feat/foo",                        # angular commit type, not branch type
    ],
)
def test_pre_push_rejects_non_conventional_branches(branch):
    result = _run_pre_push(_ref_line(branch))
    assert result.returncode == 1, (
        f"hook accepted an invalid branch:\n  branch: {branch!r}\n"
        f"  stderr: {result.stderr}"
    )
    assert "Conventional Branches" in result.stderr


@pytest.mark.parametrize(
    "branch",
    [
        "main",
        "litellm_internal_staging",
        "dependabot/github_actions/foo",
        "gh-readonly-queue/main/abc123",
    ],
)
def test_pre_push_bypasses_protected_branches(branch):
    result = _run_pre_push(_ref_line(branch))
    assert result.returncode == 0, (
        f"protected branch was rejected:\n  branch: {branch!r}\n"
        f"  stderr: {result.stderr}"
    )


def test_pre_push_skips_tag_pushes():
    line = f"refs/tags/v1 {_NONZERO_OID} refs/tags/v1 {_ZERO_OID}\n"
    result = _run_pre_push(line)
    assert result.returncode == 0, result.stderr


def test_pre_push_skips_branch_deletions():
    # local oid all zeros = deletion
    line = f"refs/heads/whatever {_ZERO_OID} refs/heads/whatever {_NONZERO_OID}\n"
    result = _run_pre_push(line)
    assert result.returncode == 0, result.stderr


def test_pre_push_fails_if_any_ref_is_invalid():
    # Mixed batch: one valid, one invalid — entire push should fail.
    stdin = _ref_line("feature/ok") + _ref_line("random-bad")
    result = _run_pre_push(stdin)
    assert result.returncode == 1
    assert "random-bad" in result.stderr


def test_pre_push_no_refs_passes():
    # Empty stdin (no refs being pushed) should pass.
    result = _run_pre_push("")
    assert result.returncode == 0, result.stderr
