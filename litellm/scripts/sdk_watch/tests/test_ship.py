import base64
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pytest
from conftest import git
from sdk_watch.demo import run_demo
from sdk_watch.implement import collect_patch
from sdk_watch.report import DemoVerdict, PrDraft
from sdk_watch.ship import Opened, Skipped, asset_path, is_allowed, ship

FIX = "def shout(text):\n    return text.upper()\n"
TEST = "from litellm.shout import shout\n\n\ndef test_shout():\n    assert shout('hi') == 'HI'\n"
SLOW_TEST = "import time\n\n\ndef test_shout():\n    time.sleep(60)\n"


@dataclass
class FakeGitHub:
    existing_branches: frozenset[str] = frozenset()
    opened: list[dict[str, str]] = field(default_factory=list)

    def first_pr(self, query: str) -> str | None:
        return None

    def branch_exists(self, branch: str) -> bool:
        return branch in self.existing_branches

    def create_pr(self, title: str, head: str, base: str, body: str) -> str:
        self.opened.append({"title": title, "head": head, "base": base, "body": body})
        return "https://github.com/BerriAI/litellm/pull/1"


def draft() -> PrDraft:
    return PrDraft(
        branch="litellm_sdk_watch_anthropic_0_105_0_shout",
        title="feat(anthropic): shout",
        sdk="anthropic",
        version="0.105.0",
        released=date(2026, 5, 28),
        release_url="https://github.com/anthropics/anthropic-sdk-python/releases/tag/v0.105.0",
        sdk_change="**api:** shouting",
        litellm_change="Upper-case the text",
        walkthrough="Walkthrough text",
        test_files=("tests/unit/test_shout.py",),
    )


def make_patch(repo: Path, tmp_path: Path, files: dict[str, str]) -> Path:
    for path, text in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)
    patch = tmp_path / "change.patch"
    patch.write_text(collect_patch(repo))
    return patch


@pytest.mark.parametrize(
    ("path", "allowed"),
    [
        ("litellm/llms/anthropic/chat/transformation.py", True),
        ("tests/unit/test_x.py", True),
        ("model_prices_and_context_window.json", True),
        ("litellm/scripts/sdk_watch/sdk_watch/ship.py", False),
        (".github/workflows/provider-sdk-watch.yml", False),
        ("pyproject.toml", False),
        ("litellm/../.github/workflows/x.yml", False),
    ],
)
def test_the_agent_may_only_change_litellm_code_tests_and_the_cost_map(path: str, allowed: bool) -> None:
    assert is_allowed(path) is allowed


def test_ships_a_verified_patch_as_bot_commits_whose_video_stays_out_of_the_merged_tree(
    repo: Path, tmp_path: Path
) -> None:
    patch = make_patch(repo, tmp_path, {"litellm/shout.py": FIX, "tests/unit/test_shout.py": TEST})
    gif = tmp_path / "demo.gif"
    gif.write_bytes(b"GIF89a")
    github = FakeGitHub()
    base = git(repo, "rev-parse", "HEAD").strip()

    result = ship(
        repo,
        "BerriAI/litellm",
        "litellm_release_line",
        base,
        patch,
        draft(),
        DemoVerdict(fails_before=True, passes_after=True),
        gif,
        github,
        "token",
        "https://run",
    )

    assert result == Opened(url="https://github.com/BerriAI/litellm/pull/1")
    pushed = git(repo, "ls-remote", "origin", f"refs/heads/{draft().branch}").split()[0]
    authors = git(repo, "log", "--format=%an <%ae>", f"{base}..{pushed}").split("\n")[:-1]
    assert set(authors) == {"github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>"}
    assert git(repo, "log", "--reverse", "--format=%s", f"{base}..{pushed}").split("\n")[0] == "feat(anthropic): shout"
    assert sorted(git(repo, "diff", "--name-only", base, pushed).split()) == [
        "litellm/shout.py",
        "tests/unit/test_shout.py",
    ]
    [opened] = github.opened
    assert (opened["head"], opened["base"]) == (draft().branch, "litellm_release_line")
    url = rf"https://github.com/BerriAI/litellm/blob/([0-9a-f]{{40}})/{re.escape(asset_path(draft()))}\?raw=true"
    [recording] = re.findall(url, opened["body"])
    assert git(repo, "merge-base", "--is-ancestor", recording, pushed) == ""
    assert git(repo, "show", f"{recording}:{asset_path(draft())}") == "GIF89a"
    assert "[insert handwritten description here]" in opened["body"]


def test_a_failed_push_keeps_the_token_out_of_the_error(repo: Path, tmp_path: Path) -> None:
    patch = make_patch(repo, tmp_path, {"litellm/shout.py": FIX, "tests/unit/test_shout.py": TEST})
    gif = tmp_path / "demo.gif"
    gif.write_bytes(b"GIF89a")
    base = git(repo, "rev-parse", "HEAD").strip()
    git(repo, "remote", "set-url", "origin", str(tmp_path / "missing.git"))
    token = "ghs_do_not_leak"

    with pytest.raises(subprocess.CalledProcessError) as failure:
        ship(
            repo,
            "BerriAI/litellm",
            "main",
            base,
            patch,
            draft(),
            DemoVerdict(fails_before=True, passes_after=True),
            gif,
            FakeGitHub(),
            token,
            "https://run",
        )

    rendered = f"{failure.value} {failure.value.cmd}"
    assert token not in rendered
    assert base64.b64encode(f"x-access-token:{token}".encode()).decode() not in rendered


@pytest.mark.parametrize(
    ("files", "named"),
    [
        ({"litellm/shout.py": FIX, ".github/workflows/x.yml": "on: push\n"}, ".github/workflows/x.yml"),
        ({"litellm/shout.py": FIX}, "tests/unit/test_shout.py"),
    ],
    ids=["touches-a-workflow", "leaves-out-its-tests"],
)
def test_refuses_a_patch_that_touches_a_workflow_or_leaves_out_its_tests_and_leaves_main_untouched(
    repo: Path, tmp_path: Path, files: dict[str, str], named: str
) -> None:
    patch = make_patch(repo, tmp_path, files)
    github = FakeGitHub()
    base = git(repo, "rev-parse", "HEAD").strip()

    result = ship(
        repo,
        "BerriAI/litellm",
        "main",
        base,
        patch,
        draft(),
        DemoVerdict(fails_before=True, passes_after=True),
        tmp_path / "none.gif",
        github,
        "token",
        "https://run",
    )

    assert isinstance(result, Skipped) and named in result.reason
    assert github.opened == []
    assert git(repo, "status", "--porcelain") == ""
    assert git(repo, "ls-remote", "origin", f"refs/heads/{draft().branch}") == ""


@pytest.mark.parametrize(
    ("verdict", "branches"),
    [
        (DemoVerdict(fails_before=False, passes_after=True), frozenset()),
        (DemoVerdict(fails_before=True, passes_after=False), frozenset()),
        (DemoVerdict(fails_before=True, passes_after=True), frozenset({draft().branch})),
    ],
    ids=["tests-pass-without-the-fix", "tests-fail-with-the-fix", "branch-already-exists"],
)
def test_opens_nothing_for_unproven_tests_a_failing_fix_or_a_rerun(
    repo: Path, tmp_path: Path, verdict: DemoVerdict, branches: frozenset[str]
) -> None:
    patch = make_patch(repo, tmp_path, {"litellm/shout.py": FIX, "tests/unit/test_shout.py": TEST})
    github = FakeGitHub(existing_branches=branches)

    result = ship(
        repo,
        "BerriAI/litellm",
        "main",
        git(repo, "rev-parse", "HEAD").strip(),
        patch,
        draft(),
        verdict,
        tmp_path / "none.gif",
        github,
        "token",
        "https://run",
    )

    assert isinstance(result, Skipped)
    assert github.opened == []


def test_refuses_to_ship_from_a_checkout_with_local_changes_and_keeps_them(repo: Path, tmp_path: Path) -> None:
    patch = make_patch(repo, tmp_path, {"litellm/shout.py": FIX, "tests/unit/test_shout.py": TEST})
    (repo / "notes.txt").write_text("work in progress")
    github = FakeGitHub()

    result = ship(
        repo,
        "BerriAI/litellm",
        "main",
        git(repo, "rev-parse", "HEAD").strip(),
        patch,
        draft(),
        DemoVerdict(fails_before=True, passes_after=True),
        tmp_path / "none.gif",
        github,
        "token",
        "https://run",
    )

    assert isinstance(result, Skipped)
    assert github.opened == []
    assert (repo / "notes.txt").read_text() == "work in progress"


def test_the_demo_shows_the_tests_failing_on_main_and_passing_with_the_fix(repo: Path, tmp_path: Path) -> None:
    patch = make_patch(repo, tmp_path, {"litellm/shout.py": FIX, "tests/unit/test_shout.py": TEST})
    lines = []

    verdict = run_demo(repo, patch, draft(), Path(sys.executable), lines.append, 900)

    assert verdict == DemoVerdict(fails_before=True, passes_after=True)
    assert lines[-1] == "# Before: failed. After: passed"
    assert git(repo, "status", "--porcelain") == ""
    assert json.loads(verdict.model_dump_json()) == {"fails_before": True, "passes_after": True}


def test_a_test_run_that_hangs_counts_as_failed_once_its_time_is_up(repo: Path, tmp_path: Path) -> None:
    patch = make_patch(repo, tmp_path, {"litellm/shout.py": FIX, "tests/unit/test_shout.py": SLOW_TEST})
    lines = []

    verdict = run_demo(repo, patch, draft(), Path(sys.executable), lines.append, 2)

    assert verdict == DemoVerdict(fails_before=True, passes_after=False)
    assert lines.count("# Timed out after 2 s") == 2
    assert git(repo, "status", "--porcelain") == ""
