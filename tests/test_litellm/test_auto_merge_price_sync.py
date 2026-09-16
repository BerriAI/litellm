"""Tests for .github/scripts/auto_merge_price_sync.py.

`evaluate` is pure: it takes the pull request plus the fetched facts and
returns a Verdict, so each gate is exercised by building inputs where exactly
one condition fails and asserting the matching hold reason. A merge verdict
is the thing that spends an unreviewed merge, so the defaults below are the
happy path that every case perturbs one part of.
"""

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO_ROOT / ".github" / "scripts" / "auto_merge_price_sync.py"
_spec = importlib.util.spec_from_file_location("auto_merge_price_sync", _MODULE_PATH)
merger = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = merger
_spec.loader.exec_module(merger)

HEAD_SHA: Final = "deadbeef" * 5
HEAD_DATE: Final = datetime(2026, 1, 10, tzinfo=timezone.utc)
ALLOWLIST: Final = frozenset({"berriai-litellm-provider-info-sync[bot]"})
COST_MAP_FILES: Final = ("model_prices_and_context_window.json",)


def _pr(**overrides: object) -> merger.PullRequest:
    base: Final = {
        "number": 1,
        "title": "sync prices",
        "author_login": "berriai-litellm-provider-info-sync[bot]",
        "state": "open",
        "draft": False,
        "mergeable": True,
        "mergeable_state": "clean",
        "head_sha": HEAD_SHA,
    }
    return merger.PullRequest(**{**base, **overrides})


def _greptile(score: int, updated_at: datetime) -> merger.IssueComment:
    return merger.IssueComment(
        author_login="greptile-apps[bot]",
        body=f"Confidence Score: {score}/5",
        updated_at=updated_at,
    )


def _bugbot(commit_id: str, body: str, submitted_at: datetime) -> merger.Review:
    return merger.Review(
        author_login="cursor[bot]",
        state="COMMENTED",
        body=body,
        commit_id=commit_id,
        submitted_at=submitted_at,
    )


def _inputs(**overrides: object) -> merger.EvaluationInputs:
    base: Final = {
        "pr": _pr(),
        "changed_files": COST_MAP_FILES,
        "required_contexts": frozenset({"build"}),
        "check_runs": (merger.CheckRun(name="build", status="completed", conclusion="success"),),
        "statuses": (),
        "comments": (_greptile(5, datetime(2026, 1, 11, tzinfo=timezone.utc)),),
        "reviews": (
            _bugbot(
                HEAD_SHA,
                "<!-- BUGBOT_REVIEW --> cursor bugbot found no new issues",
                datetime(2026, 1, 11, tzinfo=timezone.utc),
            ),
        ),
        "head_commit_date": HEAD_DATE,
        "self_check_name": "auto-merge-price-sync",
        "author_allowlist": ALLOWLIST,
    }
    return merger.EvaluationInputs(**{**base, **overrides})


def _evaluate(inputs: merger.EvaluationInputs) -> merger.Verdict:
    return merger.evaluate(inputs, classify=lambda files: "run")


def _holds(inputs: merger.EvaluationInputs, fragment: str) -> merger.Verdict:
    verdict: Final = _evaluate(inputs)
    assert not verdict.merge
    assert any(fragment in reason for reason in verdict.reasons), verdict.reasons
    return verdict


def test_happy_path_merges() -> None:
    verdict: Final = _evaluate(_inputs())
    assert verdict.merge
    assert verdict.reasons == ()


def test_non_allowlisted_author_holds() -> None:
    _holds(_inputs(pr=_pr(author_login="octocat")), "not in allowlist")


def test_closed_pr_holds() -> None:
    _holds(_inputs(pr=_pr(state="closed")), "pr not open")


def test_draft_pr_holds() -> None:
    _holds(_inputs(pr=_pr(draft=True)), "draft")


def test_unmergeable_pr_holds() -> None:
    _holds(_inputs(pr=_pr(mergeable=False)), "not mergeable")


def test_dirty_pr_holds() -> None:
    _holds(_inputs(pr=_pr(mergeable_state="dirty")), "merge conflicts")


def test_non_cost_map_files_hold() -> None:
    verdict: Final = merger.evaluate(_inputs(changed_files=("litellm/utils.py",)), classify=lambda files: "skip")
    assert not verdict.merge
    assert any("cost-map-only" in reason for reason in verdict.reasons)


def test_required_context_missing_holds() -> None:
    _holds(_inputs(check_runs=()), "required check 'build' not green")


def test_required_context_via_commit_status_passes() -> None:
    verdict: Final = _evaluate(
        _inputs(
            check_runs=(),
            statuses=(merger.CommitStatus(context="build", state="success"),),
        )
    )
    assert verdict.merge


def test_failing_check_run_holds() -> None:
    _holds(
        _inputs(
            check_runs=(
                merger.CheckRun(name="build", status="completed", conclusion="success"),
                merger.CheckRun(name="lint", status="completed", conclusion="failure"),
            )
        ),
        "check run 'lint' is completed/failure",
    )


def test_in_progress_check_run_holds() -> None:
    _holds(
        _inputs(
            check_runs=(
                merger.CheckRun(name="build", status="completed", conclusion="success"),
                merger.CheckRun(name="ui", status="in_progress", conclusion=None),
            )
        ),
        "check run 'ui'",
    )


def test_own_check_run_is_ignored() -> None:
    verdict: Final = _evaluate(
        _inputs(
            check_runs=(
                merger.CheckRun(name="build", status="completed", conclusion="success"),
                merger.CheckRun(name="auto-merge-price-sync", status="in_progress", conclusion=None),
            )
        )
    )
    assert verdict.merge


def test_pending_commit_status_holds() -> None:
    _holds(
        _inputs(statuses=(merger.CommitStatus(context="codecov", state="pending"),)),
        "commit status 'codecov' is pending",
    )


def test_greptile_missing_holds() -> None:
    _holds(_inputs(comments=()), "greptile score not available")


def test_greptile_four_of_five_holds() -> None:
    _holds(
        _inputs(comments=(_greptile(4, datetime(2026, 1, 11, tzinfo=timezone.utc)),)),
        "greptile score 4/5",
    )


def test_greptile_older_than_head_holds() -> None:
    _holds(
        _inputs(comments=(_greptile(5, datetime(2026, 1, 9, tzinfo=timezone.utc)),)),
        "older than head commit",
    )


def test_bugbot_missing_holds() -> None:
    _holds(_inputs(reviews=()), "bugbot review not available")


def test_bugbot_stale_marker_ignored() -> None:
    _holds(
        _inputs(
            reviews=(
                _bugbot(
                    HEAD_SHA,
                    "<!-- BUGBOT_REVIEW --><!-- BUGBOT_REVIEW_STALE --> cursor bugbot found no new issues",
                    datetime(2026, 1, 11, tzinfo=timezone.utc),
                ),
            )
        ),
        "bugbot review not available",
    )


def test_bugbot_old_commit_ignored() -> None:
    _holds(
        _inputs(
            reviews=(
                _bugbot(
                    "0" * 40,
                    "<!-- BUGBOT_REVIEW --> cursor bugbot found no new issues",
                    datetime(2026, 1, 11, tzinfo=timezone.utc),
                ),
            )
        ),
        "bugbot review not available",
    )


def test_bugbot_issues_found_holds() -> None:
    _holds(
        _inputs(
            reviews=(
                _bugbot(
                    HEAD_SHA,
                    "<!-- BUGBOT_REVIEW --> cursor bugbot found 2 new issues",
                    datetime(2026, 1, 11, tzinfo=timezone.utc),
                ),
            )
        ),
        "bugbot reported issues",
    )


def test_changes_requested_holds() -> None:
    _holds(
        _inputs(
            reviews=(
                _bugbot(
                    HEAD_SHA,
                    "<!-- BUGBOT_REVIEW --> cursor bugbot found no new issues",
                    datetime(2026, 1, 11, tzinfo=timezone.utc),
                ),
                merger.Review(
                    author_login="human-reviewer",
                    state="CHANGES_REQUESTED",
                    body="",
                    commit_id=HEAD_SHA,
                    submitted_at=datetime(2026, 1, 12, tzinfo=timezone.utc),
                ),
            )
        ),
        "changes requested by human-reviewer",
    )


def test_superseded_changes_requested_merges() -> None:
    verdict: Final = _evaluate(
        _inputs(
            reviews=(
                _bugbot(
                    HEAD_SHA,
                    "<!-- BUGBOT_REVIEW --> cursor bugbot found no new issues",
                    datetime(2026, 1, 12, tzinfo=timezone.utc),
                ),
                merger.Review(
                    author_login="human-reviewer",
                    state="CHANGES_REQUESTED",
                    body="",
                    commit_id=HEAD_SHA,
                    submitted_at=datetime(2026, 1, 11, tzinfo=timezone.utc),
                ),
                merger.Review(
                    author_login="human-reviewer",
                    state="APPROVED",
                    body="",
                    commit_id=HEAD_SHA,
                    submitted_at=datetime(2026, 1, 13, tzinfo=timezone.utc),
                ),
            )
        )
    )
    assert verdict.merge


def test_classifier_cost_map_set_runs() -> None:
    assert merger._classify(["model_prices_and_context_window.json", "tests/test_litellm/test_x.py"]) == "run"


def test_classifier_backend_file_skips() -> None:
    assert merger._classify(["model_prices_and_context_window.json", "litellm/main.py"]) == "skip"


def test_main_without_token_logs_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("GH_TOKEN", raising=False)
    assert merger.main() == 0
    assert "app credentials not configured" in capsys.readouterr().out
