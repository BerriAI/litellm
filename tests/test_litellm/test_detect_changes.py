"""Regression tests for the GitHub Actions change-based job gating.

`.github/scripts/detect_changes.sh` decides whether a pull request's jobs do
real work. It asks the API which files the pull request touches and hands them
to `classify_changes.sh` under one category. The contract locked in here:

  * a UI-only pull request skips backend jobs even when the checked-out merge
    ref carries backend commits from the base branch
  * the ui category is the mirror image: it skips when only backend files
    changed, so a backend-only PR stops building and unit-testing the dashboard
  * anything the classification cannot resolve (no pull request, an API
    failure, a truncated file list, a broken classifier) runs the job
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "detect_changes.sh"
CLASSIFIER = REPO_ROOT / ".circleci" / "scripts" / "classify_changes.sh"

UI_FILE = "ui/litellm-dashboard/src/components/Teams.tsx"
BACKEND_FILE = "litellm/proxy/proxy_server.py"


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _merge_ref_checkout(tmp_path: Path) -> Path:
    """A checkout shaped like `refs/pull/N/merge`: a UI-only branch merged into a
    base tip that has moved ahead by a backend commit since the branch was cut."""
    work = tmp_path / "work"
    work.mkdir()
    _git(work, "init", "-q", "-b", "main")
    _git(work, "config", "user.email", "t@t")
    _git(work, "config", "user.name", "t")
    (work / "seed.txt").write_text("seed\n")
    _git(work, "add", "-A")
    _git(work, "commit", "-qm", "base")

    _git(work, "checkout", "-q", "-b", "feature")
    ui = work / UI_FILE
    ui.parent.mkdir(parents=True, exist_ok=True)
    ui.write_text("export const Teams = () => null\n")
    _git(work, "add", "-A")
    _git(work, "commit", "-qm", "ui change")

    _git(work, "checkout", "-q", "main")
    backend = work / BACKEND_FILE
    backend.parent.mkdir(parents=True, exist_ok=True)
    backend.write_text("x = 1\n")
    _git(work, "add", "-A")
    _git(work, "commit", "-qm", "someone else's backend change")

    _git(work, "merge", "-q", "--no-ff", "-m", "Merge feature into main", "feature")
    return work


def _scripts_tree(tmp_path: Path, classifier_body: str | None = None) -> Path:
    """Copy the scripts into a throwaway tree, preserving their relative layout."""
    root = tmp_path / "tree"
    (root / ".github" / "scripts").mkdir(parents=True)
    (root / ".circleci" / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT, root / ".github" / "scripts" / SCRIPT.name)
    target = root / ".circleci" / "scripts" / CLASSIFIER.name
    if classifier_body is None:
        shutil.copy(CLASSIFIER, target)
    else:
        target.write_text(classifier_body)
    target.chmod(0o755)
    return root


def _run(
    tmp_path: Path,
    *,
    files: list[str],
    cwd: Path | None = None,
    pr_number: str = "37540",
    changed_file_count: str | None = None,
    gh_exit_code: int = 0,
    classifier_body: str | None = None,
    category: str | None = None,
) -> tuple[str, str]:
    """Run the script against a stubbed `gh`; returns (decision, stdout)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    listing = "".join(f"echo {f}\n" for f in files)
    stub = bin_dir / "gh"
    stub.write_text(f"#!/usr/bin/env bash\n{listing}exit {gh_exit_code}\n")
    stub.chmod(0o755)

    output_file = tmp_path / "github_output"
    output_file.write_text("")

    env = {k: v for k, v in os.environ.items() if k not in {"GH_TOKEN", "GITHUB_TOKEN"}}
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["GITHUB_OUTPUT"] = str(output_file)
    env["REPO"] = "BerriAI/litellm"
    env["PR_NUMBER"] = pr_number
    env["CHANGED_FILE_COUNT"] = changed_file_count if changed_file_count is not None else str(len(files))
    if category is not None:
        env["CATEGORY"] = category
    else:
        env.pop("CATEGORY", None)

    tree = _scripts_tree(tmp_path, classifier_body)
    result = subprocess.run(
        ["bash", str(tree / ".github" / "scripts" / SCRIPT.name)],
        cwd=cwd or tmp_path,
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return output_file.read_text().strip(), result.stdout


def test_ui_only_pr_skips_even_when_the_merge_ref_carries_backend_commits(tmp_path: Path) -> None:
    """The bug this replaces: diffing the checked-out merge ref against the event's
    base sha attributed the base branch's own backend commits to the pull request,
    so every UI-only PR ran the full backend suite."""
    work = _merge_ref_checkout(tmp_path)
    tracked = subprocess.run(
        ["git", "diff", "--name-only", "HEAD~2", "HEAD"],
        cwd=work,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert BACKEND_FILE in tracked, "the checkout must contain the base branch's backend commit"

    decision, _ = _run(tmp_path, files=[UI_FILE], cwd=work)
    assert decision == "decision=skip"


def test_backend_file_in_the_pr_runs(tmp_path: Path) -> None:
    decision, _ = _run(tmp_path, files=[UI_FILE, BACKEND_FILE])
    assert decision == "decision=run"


def test_docs_only_pr_skips(tmp_path: Path) -> None:
    decision, _ = _run(tmp_path, files=["README.md", "docs/my-website/index.mdx"])
    assert decision == "decision=skip"


def test_non_pull_request_event_runs(tmp_path: Path) -> None:
    decision, stdout = _run(tmp_path, files=[UI_FILE], pr_number="")
    assert decision == "decision=run"
    assert "not a pull_request event" in stdout


def test_api_failure_runs(tmp_path: Path) -> None:
    decision, stdout = _run(tmp_path, files=[], gh_exit_code=1)
    assert decision == "decision=run"
    assert "could not list the files" in stdout


def test_empty_file_list_runs(tmp_path: Path) -> None:
    decision, stdout = _run(tmp_path, files=[], changed_file_count="0")
    assert decision == "decision=run"
    assert "listed no files" in stdout


def test_pr_past_the_listing_ceiling_runs(tmp_path: Path) -> None:
    """The API caps its file listing, so a larger PR would be classified from a
    truncated set and could skip backend jobs it needs."""
    decision, stdout = _run(tmp_path, files=[UI_FILE], changed_file_count="3001")
    assert decision == "decision=run"
    assert "past the 3000-file listing ceiling" in stdout


def test_broken_classifier_runs(tmp_path: Path) -> None:
    decision, stdout = _run(
        tmp_path,
        files=[UI_FILE],
        classifier_body="#!/usr/bin/env bash\nexit 1\n",
    )
    assert decision == "decision=run"
    assert "classify_changes.sh failed" in stdout


def test_unexpected_classifier_output_runs(tmp_path: Path) -> None:
    decision, stdout = _run(
        tmp_path,
        files=[UI_FILE],
        classifier_body="#!/usr/bin/env bash\ncat >/dev/null\necho maybe\n",
    )
    assert decision == "decision=run"
    assert "unexpected decision: maybe" in stdout


def test_ui_category_skips_a_backend_only_pr(tmp_path: Path) -> None:
    """The dashboard build and its unit tests cannot be affected by a pull request
    that touches no `ui/` file, and the `client` category cannot express that
    because it deliberately runs whenever the backend changes."""
    decision, _ = _run(tmp_path, files=[BACKEND_FILE], category="ui")
    assert decision == "decision=skip"


def test_ui_category_runs_a_ui_only_pr(tmp_path: Path) -> None:
    decision, _ = _run(tmp_path, files=[UI_FILE], category="ui")
    assert decision == "decision=run"


def test_ui_category_runs_a_mixed_pr(tmp_path: Path) -> None:
    decision, _ = _run(tmp_path, files=[UI_FILE, BACKEND_FILE], category="ui")
    assert decision == "decision=run"


def test_absent_category_still_runs_a_backend_pr(tmp_path: Path) -> None:
    """Callers that pass no category keep the pre-existing backend behaviour."""
    assert _run(tmp_path, files=[BACKEND_FILE])[0] == "decision=run"


def test_absent_category_still_skips_a_ui_pr(tmp_path: Path) -> None:
    assert _run(tmp_path, files=[UI_FILE])[0] == "decision=skip"


def test_ui_category_fails_open_when_the_api_fails(tmp_path: Path) -> None:
    decision, stdout = _run(tmp_path, files=[], gh_exit_code=1, category="ui")
    assert decision == "decision=run"
    assert "detect-changes[ui]" in stdout


def test_ui_category_runs_when_the_ui_workflows_themselves_change(tmp_path: Path) -> None:
    """Without this the dashboard jobs would skip on the pull request that edits
    them, shipping a workflow change nothing ever exercised."""
    decision, _ = _run(tmp_path, files=[".github/workflows/test-litellm-ui-unit.yml"], category="ui")
    assert decision == "decision=run"
