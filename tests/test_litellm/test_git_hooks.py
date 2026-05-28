"""
Tests for the Conventional Commits / Conventional Branches git hooks shipped
in `.githooks/`.

We invoke each hook as a subprocess (the production code path) so the test
exercises exactly the bash that contributors will run locally.
"""

from __future__ import annotations

import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMIT_MSG_HOOK = REPO_ROOT / ".githooks" / "commit-msg"
PRE_PUSH_HOOK = REPO_ROOT / ".githooks" / "pre-push"

bash = shutil.which("bash")

pytestmark = pytest.mark.skipif(
    bash is None,
    reason="bash not available on this platform; git hooks require bash",
)


# --- helpers ---------------------------------------------------------------


def _run_commit_msg(message: str, tmp_path: Path) -> subprocess.CompletedProcess:
    """Run the commit-msg hook against `message` written to a temp file."""
    assert COMMIT_MSG_HOOK.exists(), f"missing hook: {COMMIT_MSG_HOOK}"
    # Make sure the hook is executable — some checkouts lose +x.
    mode = COMMIT_MSG_HOOK.stat().st_mode
    COMMIT_MSG_HOOK.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    msg_file = tmp_path / "COMMIT_EDITMSG"
    msg_file.write_text(message)
    return subprocess.run(
        [bash, str(COMMIT_MSG_HOOK), str(msg_file)],
        capture_output=True,
        text=True,
        timeout=10,
    )


def _run_pre_push(stdin: str) -> subprocess.CompletedProcess:
    """Run the pre-push hook with `stdin` as the git pre-push contract."""
    assert PRE_PUSH_HOOK.exists(), f"missing hook: {PRE_PUSH_HOOK}"
    mode = PRE_PUSH_HOOK.stat().st_mode
    PRE_PUSH_HOOK.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # The hook is invoked as `pre-push <remote_name> <remote_url>` by git;
    # we pass dummy args. The real signal is on stdin.
    return subprocess.run(
        [bash, str(PRE_PUSH_HOOK), "origin", "https://example.invalid/repo.git"],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=10,
    )


# --- commit-msg ------------------------------------------------------------


@pytest.mark.parametrize(
    "subject",
    [
        "feat: add cooldown",
        "fix: bedrock empty tool use",
        "docs: clarify proxy install",
        "chore(test): bump deps",
        "feat(router): add cooldown for upstream 5xx",
        "fix(bedrock-converse): handle empty toolUse blocks",
        "refactor(proxy/auth): split key cache",
        "perf(streaming)!: drop deprecated buffer path",
        "revert: feat(router): add cooldown for upstream 5xx",
        # pass-throughs
        "Merge branch 'main' into feat/x",
        "Merge pull request #28800 from foo/bar",
        "Revert \"feat: bad change\"",
        "fixup! feat: add cooldown",
        "squash! fix: bedrock",
        "amend! feat: add cooldown",
    ],
)
def test_commit_msg_accepts_valid(tmp_path, subject):
    r = _run_commit_msg(subject + "\n\nbody line\n", tmp_path)
    assert r.returncode == 0, f"expected accept, got rc={r.returncode}\nstderr={r.stderr}"


@pytest.mark.parametrize(
    "subject",
    [
        "add stuff",
        "WIP",
        "fix bedrock",  # missing colon
        "Feat: add cooldown",  # type must be lowercase
        "feat add cooldown",  # missing colon
        "feat(router) add cooldown",  # missing colon after scope
        "feat:",  # empty description
        "feat: ",  # empty (just whitespace) description
        "wibble: add stuff",  # unknown type
    ],
)
def test_commit_msg_rejects_invalid(tmp_path, subject):
    r = _run_commit_msg(subject + "\n", tmp_path)
    assert r.returncode == 1, (
        f"expected reject for {subject!r}, got rc={r.returncode}\n"
        f"stdout={r.stdout}\nstderr={r.stderr}"
    )
    assert "Conventional Commits" in r.stderr


def test_commit_msg_ignores_leading_comments_and_blank_lines(tmp_path):
    msg = (
        "# Please enter the commit message for your changes.\n"
        "# Lines starting with '#' will be ignored.\n"
        "\n"
        "feat(router): add cooldown\n"
        "\n"
        "body\n"
    )
    r = _run_commit_msg(msg, tmp_path)
    assert r.returncode == 0, r.stderr


def test_commit_msg_rejects_empty_message(tmp_path):
    r = _run_commit_msg("# only comments\n#another\n\n\n", tmp_path)
    assert r.returncode == 1
    assert "empty commit message" in r.stderr


# --- pre-push --------------------------------------------------------------


ZERO = "0" * 40
SHA = "deadbeefcafebabe1234567890abcdef12345678"


@pytest.mark.parametrize(
    "branch",
    [
        "feature/router-cooldown",
        "bugfix/bedrock-empty-tooluse",
        "hotfix/auth-401",
        "release/v1.86.0",
        "chore/bump-deps",
        # bypass list — `litellm_*` covers all long-lived internal branches
        "main",
        "litellm_internal_staging",
        "litellm_oss_agent_shin_daily_branch",
        "litellm_release_v1.86.0",
        "dependabot/pip/openai-1.2.3",
        "gh-readonly-queue/main/pr-123",
    ],
)
def test_pre_push_accepts_valid(branch):
    stdin = f"refs/heads/{branch} {SHA} refs/heads/{branch} {ZERO}\n"
    r = _run_pre_push(stdin)
    assert r.returncode == 0, f"expected accept for {branch!r}\nstderr={r.stderr}"


@pytest.mark.parametrize(
    "branch",
    [
        "random-branch",
        "Ishaan/foo",
        "feat/foo",  # `feat` is a commit type, not a branch type
        "fix/foo",
        "feature",  # missing `/<description>`
        "feature/",  # empty description
    ],
)
def test_pre_push_rejects_invalid(branch):
    stdin = f"refs/heads/{branch} {SHA} refs/heads/{branch} {ZERO}\n"
    r = _run_pre_push(stdin)
    assert r.returncode == 1, (
        f"expected reject for {branch!r}, got rc={r.returncode}\n"
        f"stdout={r.stdout}\nstderr={r.stderr}"
    )
    assert "Conventional Branches" in r.stderr


def test_pre_push_skips_tag_push():
    stdin = f"refs/tags/v1.0.0 {SHA} refs/tags/v1.0.0 {ZERO}\n"
    r = _run_pre_push(stdin)
    assert r.returncode == 0, r.stderr


def test_pre_push_skips_deletions():
    # Deletion: local sha is all zeros even with an invalid-looking branch name.
    stdin = f"refs/heads/not-a-valid-branch {ZERO} refs/heads/not-a-valid-branch {SHA}\n"
    r = _run_pre_push(stdin)
    assert r.returncode == 0, r.stderr


def test_pre_push_handles_multiple_refs():
    # One valid + one invalid → reject.
    stdin = (
        f"refs/heads/feature/ok {SHA} refs/heads/feature/ok {ZERO}\n"
        f"refs/heads/bad-branch {SHA} refs/heads/bad-branch {ZERO}\n"
    )
    r = _run_pre_push(stdin)
    assert r.returncode == 1
    assert "bad-branch" in r.stderr


def test_pre_push_no_input_is_ok():
    r = _run_pre_push("")
    assert r.returncode == 0, r.stderr
