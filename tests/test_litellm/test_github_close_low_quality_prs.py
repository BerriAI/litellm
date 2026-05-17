"""Unit tests for `.github/scripts/close_low_quality_prs.py`.

These exercise the pure logic (score extraction and per-PR evaluation) without
hitting GitHub. Network/CLI calls are stubbed via monkeypatch.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github"
    / "scripts"
    / "close_low_quality_prs.py"
)


@pytest.fixture(scope="module")
def closer_module():
    """Load the script as a module via its file path (it lives outside the package)."""
    spec = importlib.util.spec_from_file_location("close_low_quality_prs", SCRIPT_PATH)
    assert spec and spec.loader, f"Could not load spec for {SCRIPT_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["close_low_quality_prs"] = module
    spec.loader.exec_module(module)
    return module


def _greptile_comment(
    body: str,
    updated_at: str = "2026-05-10T00:00:00Z",
    login: str = "greptile-apps[bot]",
) -> dict:
    return {
        "user": {"login": login},
        "body": body,
        "created_at": updated_at,
        "updated_at": updated_at,
    }


class TestExtractGreptileScore:
    def test_should_extract_score_from_html_header(self, closer_module):
        comments = [
            _greptile_comment("<h3>Confidence Score: 3/5</h3>\nSome body text.")
        ]
        result = closer_module.extract_greptile_score(comments)
        assert result is not None
        score, _ = result
        assert score == 3

    def test_should_accept_both_greptile_login_variants(self, closer_module):
        # REST API form ("greptile-apps[bot]") and GraphQL form ("greptile-apps")
        for login in ("greptile-apps", "greptile-apps[bot]"):
            comments = [
                _greptile_comment("<h3>Confidence Score: 2/5</h3>", login=login)
            ]
            result = closer_module.extract_greptile_score(comments)
            assert result is not None, f"failed to detect score for login={login}"
            score, _ = result
            assert score == 2

    def test_should_extract_score_from_plain_text(self, closer_module):
        comments = [_greptile_comment("Confidence Score: 5/5 — looks good!")]
        result = closer_module.extract_greptile_score(comments)
        assert result is not None
        score, _ = result
        assert score == 5

    def test_should_tolerate_whitespace_and_case(self, closer_module):
        comments = [_greptile_comment("**confidence score : 2 / 5**")]
        result = closer_module.extract_greptile_score(comments)
        assert result is not None
        score, _ = result
        assert score == 2

    def test_should_pick_most_recent_comment_when_rereview_happens(self, closer_module):
        comments = [
            _greptile_comment(
                "Confidence Score: 2/5", updated_at="2026-05-01T00:00:00Z"
            ),
            _greptile_comment(
                "Confidence Score: 5/5", updated_at="2026-05-12T00:00:00Z"
            ),
        ]
        result = closer_module.extract_greptile_score(comments)
        assert result is not None
        score, _ = result
        assert score == 5

    def test_should_ignore_non_greptile_authors(self, closer_module):
        comments = [
            {
                "user": {"login": "some-human"},
                "body": "Confidence Score: 1/5",
                "created_at": "2026-05-12T00:00:00Z",
                "updated_at": "2026-05-12T00:00:00Z",
            }
        ]
        assert closer_module.extract_greptile_score(comments) is None

    def test_should_return_none_when_no_score_present(self, closer_module):
        comments = [_greptile_comment("Greptile summary without a score.")]
        assert closer_module.extract_greptile_score(comments) is None

    def test_should_return_none_for_empty_comments(self, closer_module):
        assert closer_module.extract_greptile_score([]) is None


class TestEvaluatePr:
    @pytest.fixture(autouse=True)
    def _now(self):
        return dt.datetime(2026, 5, 17, tzinfo=dt.timezone.utc)

    def _make_pr(
        self,
        *,
        number: int = 1,
        created_days_ago: int = 10,
        is_draft: bool = False,
        labels: list[str] | None = None,
    ) -> dict:
        created = dt.datetime(2026, 5, 17, tzinfo=dt.timezone.utc) - dt.timedelta(
            days=created_days_ago
        )
        return {
            "number": number,
            "title": f"PR #{number}",
            "createdAt": created.isoformat().replace("+00:00", "Z"),
            "isDraft": is_draft,
            "labels": [{"name": lbl} for lbl in (labels or [])],
            "author": {"login": "someone"},
            "url": f"https://example.com/pr/{number}",
        }

    def test_should_skip_drafts(self, closer_module, _now, monkeypatch):
        monkeypatch.setattr(
            closer_module,
            "fetch_pr_comments",
            lambda *a, **kw: pytest.fail("should not fetch comments for drafts"),
        )
        action, score, age = closer_module.evaluate_pr(
            self._make_pr(is_draft=True),
            now=_now,
            min_age_days=7,
            min_score=4,
            repo=None,
            optout_labels=set(),
        )
        assert action == "skip-draft"
        assert score is None and age is None

    def test_should_skip_optout_label_case_insensitive(
        self, closer_module, _now, monkeypatch
    ):
        monkeypatch.setattr(
            closer_module,
            "fetch_pr_comments",
            lambda *a, **kw: pytest.fail("should not fetch comments for opt-outs"),
        )
        action, _, _ = closer_module.evaluate_pr(
            self._make_pr(labels=["WIP"]),
            now=_now,
            min_age_days=7,
            min_score=4,
            repo=None,
            optout_labels={"wip"},
        )
        assert action == "skip-optout-label"

    def test_should_skip_too_young(self, closer_module, _now, monkeypatch):
        monkeypatch.setattr(
            closer_module,
            "fetch_pr_comments",
            lambda *a, **kw: pytest.fail("should not fetch comments for young PRs"),
        )
        action, _, age = closer_module.evaluate_pr(
            self._make_pr(created_days_ago=2),
            now=_now,
            min_age_days=7,
            min_score=4,
            repo=None,
            optout_labels=set(),
        )
        assert action == "skip-too-young"
        assert age == 2

    def test_should_skip_when_greptile_has_not_reviewed(
        self, closer_module, _now, monkeypatch
    ):
        monkeypatch.setattr(closer_module, "fetch_pr_comments", lambda *a, **kw: [])
        action, score, age = closer_module.evaluate_pr(
            self._make_pr(created_days_ago=10),
            now=_now,
            min_age_days=7,
            min_score=4,
            repo=None,
            optout_labels=set(),
        )
        assert action == "skip-no-greptile-score"
        assert score is None and age == 10

    def test_should_skip_when_score_meets_threshold(
        self, closer_module, _now, monkeypatch
    ):
        monkeypatch.setattr(
            closer_module,
            "fetch_pr_comments",
            lambda *a, **kw: [_greptile_comment("Confidence Score: 4/5")],
        )
        action, score, age = closer_module.evaluate_pr(
            self._make_pr(created_days_ago=10),
            now=_now,
            min_age_days=7,
            min_score=4,
            repo=None,
            optout_labels=set(),
        )
        assert action == "skip-score-ok"
        assert score == 4 and age == 10

    def test_should_close_when_old_and_low_score(
        self, closer_module, _now, monkeypatch
    ):
        monkeypatch.setattr(
            closer_module,
            "fetch_pr_comments",
            lambda *a, **kw: [_greptile_comment("Confidence Score: 3/5")],
        )
        action, score, age = closer_module.evaluate_pr(
            self._make_pr(created_days_ago=10),
            now=_now,
            min_age_days=7,
            min_score=4,
            repo=None,
            optout_labels=set(),
        )
        assert action == "close"
        assert score == 3 and age == 10

    def test_should_close_when_old_and_very_low_score(
        self, closer_module, _now, monkeypatch
    ):
        monkeypatch.setattr(
            closer_module,
            "fetch_pr_comments",
            lambda *a, **kw: [_greptile_comment("<h3>Confidence Score: 1/5</h3>")],
        )
        action, score, _ = closer_module.evaluate_pr(
            self._make_pr(created_days_ago=14),
            now=_now,
            min_age_days=7,
            min_score=4,
            repo=None,
            optout_labels=set(),
        )
        assert action == "close"
        assert score == 1


class TestHasOptoutLabel:
    def test_should_match_label_case_insensitively(self, closer_module):
        pr = {"labels": [{"name": "Do Not Close"}, {"name": "bug"}]}
        assert closer_module.has_optout_label(pr, {"do not close"}) is True

    def test_should_return_false_when_no_match(self, closer_module):
        pr = {"labels": [{"name": "bug"}, {"name": "enhancement"}]}
        assert closer_module.has_optout_label(pr, {"wip", "keep open"}) is False

    def test_should_handle_missing_labels(self, closer_module):
        assert closer_module.has_optout_label({}, {"wip"}) is False
