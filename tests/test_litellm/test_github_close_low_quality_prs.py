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


class TestFetchPrComments:
    """`fetch_pr_comments` must fail-safe so a transient `gh api` error on
    one PR doesn't abort the whole daily sweep mid-loop. Pinning the
    empty-list return matches the fail-safe pattern in
    `fetch_pr_author_association`.
    """

    def test_should_return_empty_list_when_gh_api_fails(
        self, closer_module, monkeypatch
    ):
        import subprocess

        def _failing_gh(*args, **kwargs):
            raise subprocess.CalledProcessError(1, ["gh", *args])

        monkeypatch.setattr(closer_module, "gh", _failing_gh)
        assert closer_module.fetch_pr_comments(123, repo="x/y") == []

    def test_should_return_empty_list_when_paginated_output_is_malformed(
        self, closer_module, monkeypatch
    ):
        monkeypatch.setattr(closer_module, "gh", lambda *a, **kw: "not json\n")
        assert closer_module.fetch_pr_comments(123, repo="x/y") == []


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

    @pytest.fixture(autouse=True)
    def _external_author(self, closer_module, monkeypatch):
        """Treat every test PR as external unless overridden."""
        monkeypatch.setattr(
            closer_module, "is_external_pr_author", lambda pr, repo: True
        )

    def test_should_close_drafts_when_score_low(self, closer_module, _now, monkeypatch):
        # Drafts are NOT a free pass — the open-PR queue should reflect any
        # PR that needs human attention regardless of draft status. Authors
        # who need a long-lived draft can use the `wip` opt-out label.
        monkeypatch.setattr(
            closer_module,
            "fetch_pr_comments",
            lambda *a, **kw: [_greptile_comment("Confidence Score: 2/5")],
        )
        action, score, age = closer_module.evaluate_pr(
            self._make_pr(is_draft=True, created_days_ago=0),
            now=_now,
            min_age_days=0,
            min_score=4,
            repo=None,
            optout_labels=set(),
        )
        assert action == "close"
        assert score == 2 and age == 0

    def test_should_close_brand_new_pr_when_min_age_zero(
        self, closer_module, _now, monkeypatch
    ):
        # `min_age_days=0` means no age filter — a freshly-opened PR is
        # eligible the moment Greptile scores it below threshold. This is
        # the new default behavior.
        monkeypatch.setattr(
            closer_module,
            "fetch_pr_comments",
            lambda *a, **kw: [_greptile_comment("Confidence Score: 1/5")],
        )
        action, score, age = closer_module.evaluate_pr(
            self._make_pr(created_days_ago=0),
            now=_now,
            min_age_days=0,
            min_score=4,
            repo=None,
            optout_labels=set(),
        )
        assert action == "close"
        assert score == 1 and age == 0

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

    def test_should_skip_too_young_when_min_age_set(
        self, closer_module, _now, monkeypatch
    ):
        # The min-age-days flag is now opt-in (default 0). When a maintainer
        # explicitly passes a positive value (e.g. for a backfill run that
        # wants to spare brand-new PRs), the skip-too-young path still works.
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

    def test_should_not_skip_when_min_age_is_zero(
        self, closer_module, _now, monkeypatch
    ):
        # With the new default min_age_days=0, even a 0-day-old PR is
        # evaluated. This test pins that behavior so future refactors don't
        # silently restore an age filter.
        monkeypatch.setattr(
            closer_module,
            "fetch_pr_comments",
            lambda *a, **kw: [_greptile_comment("Confidence Score: 5/5")],
        )
        action, score, age = closer_module.evaluate_pr(
            self._make_pr(created_days_ago=0),
            now=_now,
            min_age_days=0,
            min_score=4,
            repo=None,
            optout_labels=set(),
        )
        assert action == "skip-score-ok"
        assert score == 5 and age == 0

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

    def test_should_skip_internal_authors(self, closer_module, _now, monkeypatch):
        # Override the fixture for this one test.
        monkeypatch.setattr(
            closer_module, "is_external_pr_author", lambda pr, repo: False
        )
        monkeypatch.setattr(
            closer_module,
            "fetch_pr_comments",
            lambda *a, **kw: pytest.fail("should not fetch comments for internal"),
        )
        action, score, _ = closer_module.evaluate_pr(
            self._make_pr(created_days_ago=14),
            now=_now,
            min_age_days=7,
            min_score=4,
            repo=None,
            optout_labels=set(),
        )
        assert action == "skip-internal"
        assert score is None


class TestMainOptoutLabelDefault:
    """`--optout-label` must REPLACE the canonical defaults, not append."""

    def _patch_no_op(self, closer_module, monkeypatch):
        monkeypatch.setattr(closer_module, "fetch_open_prs", lambda repo: [])
        # `optout_labels` is captured indirectly via evaluate_pr; sniff the
        # set passed in by stubbing evaluate_pr.
        captured: dict = {}

        def fake_evaluate(pr, now, min_age_days, min_score, repo, optout_labels):
            captured["optout_labels"] = set(optout_labels)
            return ("skip-internal", None, None)

        monkeypatch.setattr(closer_module, "evaluate_pr", fake_evaluate)
        return captured

    def test_should_use_canonical_defaults_when_flag_omitted(
        self, closer_module, monkeypatch
    ):
        captured = self._patch_no_op(closer_module, monkeypatch)
        # No PRs -> capture won't fire; instead inject one synthetic PR via
        # fetch_open_prs so evaluate_pr is invoked at least once.
        monkeypatch.setattr(
            closer_module,
            "fetch_open_prs",
            lambda repo: [
                {
                    "number": 1,
                    "title": "p",
                    "createdAt": "2026-05-10T00:00:00Z",
                    "isDraft": True,
                    "labels": [],
                    "author": {"login": "x"},
                }
            ],
        )
        monkeypatch.setattr(sys, "argv", ["close_low_quality_prs.py"])
        rc = closer_module.main()
        assert rc == 0
        assert captured["optout_labels"] == set(closer_module.DEFAULT_OPTOUT_LABELS)

    def test_should_replace_defaults_when_flag_provided(
        self, closer_module, monkeypatch
    ):
        captured = self._patch_no_op(closer_module, monkeypatch)
        monkeypatch.setattr(
            closer_module,
            "fetch_open_prs",
            lambda repo: [
                {
                    "number": 1,
                    "title": "p",
                    "createdAt": "2026-05-10T00:00:00Z",
                    "isDraft": True,
                    "labels": [],
                    "author": {"login": "x"},
                }
            ],
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "close_low_quality_prs.py",
                "--optout-label",
                "hold",
                "--optout-label",
                "needs-discussion",
            ],
        )
        rc = closer_module.main()
        assert rc == 0
        # Crucially, none of the canonical defaults leak in.
        assert captured["optout_labels"] == {"hold", "needs-discussion"}
        for default in closer_module.DEFAULT_OPTOUT_LABELS:
            assert default not in captured["optout_labels"], default


class TestMainLimitFlag:
    """`--limit N` must cap closures in both dry-run and real mode."""

    def _patch_three_closeable_prs(self, closer_module, monkeypatch):
        prs = [
            {
                "number": i,
                "title": f"p{i}",
                "createdAt": "2026-05-10T00:00:00Z",
                "isDraft": False,
                "labels": [],
                "author": {"login": f"ext{i}"},
            }
            for i in (1, 2, 3)
        ]
        monkeypatch.setattr(closer_module, "fetch_open_prs", lambda repo: prs)
        monkeypatch.setattr(
            closer_module,
            "evaluate_pr",
            lambda pr, now, mad, ms, repo, ol: ("close", 2, 30),
        )
        called: list[int] = []

        def fake_close_pr(pr, **kwargs):
            called.append(pr["number"])

        monkeypatch.setattr(closer_module, "close_pr", fake_close_pr)
        return called

    def test_should_stop_at_limit_in_dry_run(self, closer_module, monkeypatch):
        called = self._patch_three_closeable_prs(closer_module, monkeypatch)
        monkeypatch.setattr(sys, "argv", ["close_low_quality_prs.py", "--limit", "2"])
        rc = closer_module.main()
        assert rc == 0
        assert len(called) == 2

    def test_should_stop_at_limit_when_closing(self, closer_module, monkeypatch):
        called = self._patch_three_closeable_prs(closer_module, monkeypatch)
        monkeypatch.setattr(
            sys, "argv", ["close_low_quality_prs.py", "--limit", "2", "--close"]
        )
        rc = closer_module.main()
        assert rc == 0
        assert len(called) == 2


class TestFetchOpenPrsLimitWarning:
    """`fetch_open_prs` must surface a warning when the gh CLI cap is hit."""

    def test_should_warn_when_at_cap(self, closer_module, monkeypatch, capsys):
        # Pretend `gh pr list --limit 1000` returned exactly 1000 PRs —
        # this is the silent-truncation case the warning is meant to catch.
        cap = closer_module.GH_PR_LIST_LIMIT
        synthetic = [{"number": i} for i in range(cap)]
        import json as _json

        monkeypatch.setattr(
            closer_module, "gh", lambda *a, **kw: _json.dumps(synthetic)
        )
        result = closer_module.fetch_open_prs(None)
        assert len(result) == cap
        captured = capsys.readouterr()
        # GitHub Actions `::warning::` annotations go to stderr by
        # convention; just check the marker appears somewhere visible.
        combined = captured.out + captured.err
        assert "::warning::" in combined
        assert str(cap) in combined

    def test_should_not_warn_when_under_cap(self, closer_module, monkeypatch, capsys):
        synthetic = [{"number": i} for i in range(5)]
        import json as _json

        monkeypatch.setattr(
            closer_module, "gh", lambda *a, **kw: _json.dumps(synthetic)
        )
        result = closer_module.fetch_open_prs(None)
        assert len(result) == 5
        captured = capsys.readouterr()
        assert "::warning::" not in (captured.out + captured.err)


class TestHasOptoutLabel:
    def test_should_match_label_case_insensitively(self, closer_module):
        pr = {"labels": [{"name": "Do Not Close"}, {"name": "bug"}]}
        assert closer_module.has_optout_label(pr, {"do not close"}) is True

    def test_should_return_false_when_no_match(self, closer_module):
        pr = {"labels": [{"name": "bug"}, {"name": "enhancement"}]}
        assert closer_module.has_optout_label(pr, {"wip", "keep open"}) is False

    def test_should_handle_missing_labels(self, closer_module):
        assert closer_module.has_optout_label({}, {"wip"}) is False
