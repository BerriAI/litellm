"""Unit tests for the `ready for review` label lifecycle (Agent Shin review gate).

Exercises `triage_with_llm.review_gate`, the state machine that keeps the
`ready for review` label in sync with whether a PR clears both the LLM rubric
and Greptile's confidence score:

  * pass (untagged)            -> add label + "ready for review" comment
  * pass (untagged, recovered) -> add label + "all clear again" comment
  * pass (already tagged)      -> noop
  * regress (tagged)           -> remove label + "what's missing" comment, stays open
  * fail (untagged, within 24h)-> one-time "what's missing" notice
  * fail (untagged, >24h)      -> close + comment
  * dry run (close=False)      -> would-* previews, no side effects
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / ".github" / "scripts" / "triage_with_llm.py"
)

NOW = dt.datetime(2026, 5, 24, 12, 0, 0, tzinfo=dt.timezone.utc)
JUST_NOW = "2026-05-24T11:00:00Z"  # 1h old -> within 24h grace
TWO_DAYS_AGO = "2026-05-22T11:00:00Z"  # >24h old -> past grace


@pytest.fixture(scope="module")
def triage_module():
    spec = importlib.util.spec_from_file_location("triage_with_llm", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["triage_with_llm"] = module
    spec.loader.exec_module(module)
    return module


class _Recorder:
    """Captures every gh mutation review_gate could fire, and fails loudly
    on the ones a given scenario forbids."""

    def __init__(self, triage_module, monkeypatch):
        self.comments: list[str] = []
        self.added: list[str] = []
        self.removed: list[str] = []
        self.closed: list[int] = []
        monkeypatch.setattr(
            triage_module,
            "post_comment",
            lambda repo, n, body: self.comments.append(body),
        )
        monkeypatch.setattr(
            triage_module,
            "add_label",
            lambda repo, n, label: self.added.append(label),
        )
        monkeypatch.setattr(
            triage_module,
            "remove_label",
            lambda repo, n, label: self.removed.append(label),
        )
        monkeypatch.setattr(
            triage_module,
            "close_pr",
            lambda repo, n: self.closed.append(n),
        )


def _make_pr(**overrides):
    base = {
        "number": 7,
        "title": "feat: do a thing",
        "body": "some body without a linked issue or QA proof",
        "state": "open",
        "author_association": "NONE",
        "user": {"login": "mateo-berri"},
        "labels": [],
        "created_at": JUST_NOW,
    }
    base.update(overrides)
    return base


def _pass(prompt):
    return '{"verdict": "pass", "missing": [], "explanation": "looks good"}'


def _fail(prompt):
    return (
        '{"verdict": "fail", "missing": ["QA proof", "expected vs. actual"],'
        ' "explanation": "thin description"}'
    )


def _gate(triage_module, **kwargs):
    """Call review_gate with safe defaults for the injectable hooks."""
    params = dict(
        repo="o/r",
        number=7,
        close=True,
        model="m",
        judge=_pass,
        greptile_score=None,
        comments=[],
        now=NOW,
    )
    params.update(kwargs)
    return triage_module.review_gate(**params)


class TestReviewGatePass:
    def test_pass_untagged_adds_label_and_ready_comment(
        self, triage_module, monkeypatch
    ):
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: _make_pr())
        rec = _Recorder(triage_module, monkeypatch)

        result = _gate(triage_module, judge=_pass, greptile_score=5)

        assert result["action"] == "labeled-ready"
        assert rec.added == [triage_module.READY_FOR_REVIEW_LABEL]
        assert rec.removed == [] and rec.closed == []
        assert len(rec.comments) == 1
        assert "ready for review" in rec.comments[0].lower()
        assert triage_module.READY_MARKER in rec.comments[0]
        assert "5/5" in rec.comments[0]

    def test_pass_already_tagged_is_noop(self, triage_module, monkeypatch):
        pr = _make_pr(labels=[{"name": "ready for review"}])
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        rec = _Recorder(triage_module, monkeypatch)

        result = _gate(triage_module, judge=_pass, greptile_score=5)

        assert result["action"] == "noop-passing"
        assert rec.added == [] and rec.removed == [] and rec.comments == []

    def test_pass_after_prior_regression_uses_all_clear_wording(
        self, triage_module, monkeypatch
    ):
        # A regression marker in history -> this is a recovery, not a first pass.
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: _make_pr())
        rec = _Recorder(triage_module, monkeypatch)
        prior = [
            {
                "user": {"login": "github-actions[bot]"},
                "body": triage_module.REGRESSED_MARKER,
            }
        ]

        result = _gate(triage_module, judge=_pass, greptile_score=5, comments=prior)

        assert result["action"] == "labeled-ready"
        assert "all clear" in rec.comments[0].lower()

    def test_linked_issue_passes_without_calling_judge(
        self, triage_module, monkeypatch
    ):
        pr = _make_pr(body="Fixes #4321\n\nbody")
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        rec = _Recorder(triage_module, monkeypatch)

        result = _gate(
            triage_module,
            judge=lambda p: pytest.fail("LLM must not be called for linked issue"),
            greptile_score=5,
        )
        assert result["action"] == "labeled-ready"
        assert rec.added == [triage_module.READY_FOR_REVIEW_LABEL]


class TestReviewGateRegression:
    def test_regression_removes_label_and_keeps_pr_open(
        self, triage_module, monkeypatch
    ):
        pr = _make_pr(labels=[{"name": "ready for review"}])
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        rec = _Recorder(triage_module, monkeypatch)

        result = _gate(triage_module, judge=_fail, greptile_score=5)

        assert result["action"] == "label-removed-regressed"
        assert rec.removed == [triage_module.READY_FOR_REVIEW_LABEL]
        assert rec.closed == []  # regression NEVER closes the PR
        assert triage_module.REGRESSED_MARKER in rec.comments[0]
        assert "QA proof" in rec.comments[0]
        # The state machine closes a still-failing PR `grace_days` after this
        # notice (default 24h); the comment must disclose that deadline rather
        # than implying the PR stays open indefinitely.
        assert "24 hours" in rec.comments[0]
        assert "auto-closed" in rec.comments[0]

    def test_regression_comment_discloses_grace_deadline(self, triage_module):
        one_day = triage_module.format_regression_comment(
            ["QA proof"], "needs work", grace_days=1
        )
        assert "24 hours" in one_day
        assert "auto-closed" in one_day

        three_days = triage_module.format_regression_comment(
            ["QA proof"], "needs work", grace_days=3
        )
        assert "3 days" in three_days
        assert "auto-closed" in three_days

    def test_greptile_drop_alone_triggers_regression(self, triage_module, monkeypatch):
        # Rubric still passes, but Greptile fell to 2/5 -> not passing.
        pr = _make_pr(labels=[{"name": "ready for review"}])
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        rec = _Recorder(triage_module, monkeypatch)

        result = _gate(triage_module, judge=_pass, greptile_score=2)

        assert result["action"] == "label-removed-regressed"
        assert rec.removed == [triage_module.READY_FOR_REVIEW_LABEL]
        assert "2/5" in rec.comments[0]

    def test_greptile_score_read_from_comments_when_not_injected(
        self, triage_module, monkeypatch
    ):
        pr = _make_pr(labels=[{"name": "ready for review"}])
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        rec = _Recorder(triage_module, monkeypatch)
        greptile = [
            {
                "user": {"login": "greptile-apps[bot]"},
                "body": "Confidence Score: 2/5",
                "created_at": "2026-05-24T10:00:00Z",
            }
        ]

        result = _gate(
            triage_module,
            judge=_pass,
            greptile_score=triage_module._UNSET,
            comments=greptile,
        )
        assert result["action"] == "label-removed-regressed"
        assert "2/5" in rec.comments[0]


class TestReviewGateGraceAndClose:
    def test_within_grace_posts_one_time_notice(self, triage_module, monkeypatch):
        monkeypatch.setattr(
            triage_module, "fetch_pr", lambda repo, n: _make_pr(created_at=JUST_NOW)
        )
        rec = _Recorder(triage_module, monkeypatch)

        result = _gate(triage_module, judge=_fail, greptile_score=None)

        assert result["action"] == "within-grace-notified"
        assert rec.closed == [] and rec.added == [] and rec.removed == []
        assert triage_module.WITHIN_GRACE_MARKER in rec.comments[0]
        assert "QA proof" in rec.comments[0]

    def test_within_grace_does_not_double_notify(self, triage_module, monkeypatch):
        monkeypatch.setattr(
            triage_module, "fetch_pr", lambda repo, n: _make_pr(created_at=JUST_NOW)
        )
        rec = _Recorder(triage_module, monkeypatch)
        prior = [
            {
                "user": {"login": "github-actions[bot]"},
                "body": triage_module.WITHIN_GRACE_MARKER,
            }
        ]

        result = _gate(triage_module, judge=_fail, greptile_score=None, comments=prior)

        assert result["action"] == "within-grace-already-notified"
        assert rec.comments == []

    def test_past_grace_closes_with_comment(self, triage_module, monkeypatch):
        monkeypatch.setattr(
            triage_module,
            "fetch_pr",
            lambda repo, n: _make_pr(created_at=TWO_DAYS_AGO),
        )
        rec = _Recorder(triage_module, monkeypatch)

        result = _gate(triage_module, judge=_fail, greptile_score=None)

        assert result["action"] == "closed"
        assert rec.closed == [7]
        assert len(rec.comments) == 1
        # The close comment must carry the reconsider provenance marker so
        # `was_closed_by_agent_shin` can later recognize this as an Agent Shin
        # close (and not some other workflow's `github-actions[bot]` close).
        assert triage_module.AGENT_SHIN_CLOSE_MARKER in rec.comments[0]

    def test_recent_regression_marker_blocks_close(self, triage_module, monkeypatch):
        """A failing PR with a fresh regression notice must NOT be closed —
        the contributor needs a window to address the regression."""
        monkeypatch.setattr(
            triage_module,
            "fetch_pr",
            lambda repo, n: _make_pr(created_at=TWO_DAYS_AGO),
        )
        rec = _Recorder(triage_module, monkeypatch)
        prior = [
            {
                "user": {"login": "github-actions[bot]"},
                "body": triage_module.REGRESSED_MARKER,
                # Posted just an hour before NOW -> well inside grace_days.
                "created_at": "2026-05-24T11:00:00Z",
            }
        ]

        result = _gate(triage_module, judge=_fail, greptile_score=None, comments=prior)

        assert result["action"] == "regressed-already-notified"
        assert rec.closed == [] and rec.comments == []

    def test_stale_regression_marker_allows_close(self, triage_module, monkeypatch):
        """Once grace_days have elapsed since the regression notice, the
        review gate must let the close path fire — otherwise PRs that were
        regressed and then abandoned stay open forever."""
        monkeypatch.setattr(
            triage_module,
            "fetch_pr",
            lambda repo, n: _make_pr(created_at=TWO_DAYS_AGO),
        )
        rec = _Recorder(triage_module, monkeypatch)
        prior = [
            {
                "user": {"login": "github-actions[bot]"},
                "body": triage_module.REGRESSED_MARKER,
                # Posted 30 days before NOW -> well past the default 1-day grace.
                "created_at": "2026-04-24T11:00:00Z",
            }
        ]

        result = _gate(triage_module, judge=_fail, greptile_score=None, comments=prior)

        assert result["action"] == "closed"
        assert rec.closed == [7]
        assert len(rec.comments) == 1

    def test_linked_issue_with_greptile_fail_uses_greptile_explanation(
        self, triage_module, monkeypatch
    ):
        """When the rubric short-circuits to pass (linked-issue regex) but
        Greptile dragged the PR under the bar, the close comment's
        explanation must describe the Greptile shortfall, not the
        misleading "LLM was not called" rubric placeholder."""
        pr = _make_pr(body="Fixes #4321\n\nbody", created_at=TWO_DAYS_AGO)
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        rec = _Recorder(triage_module, monkeypatch)

        result = _gate(
            triage_module,
            judge=lambda p: pytest.fail("LLM must not be called for linked issue"),
            greptile_score=2,
        )

        assert result["action"] == "closed"
        assert len(rec.comments) == 1
        body = rec.comments[0]
        assert "LLM was not called" not in body
        assert "Greptile" in body and "2/5" in body


class TestReviewGateDryRun:
    @pytest.mark.parametrize(
        "scenario,labels,judge,score,created,expected",
        [
            ("pass", [], _pass, 5, JUST_NOW, "would-label-ready"),
            (
                "regress",
                [{"name": "ready for review"}],
                _fail,
                5,
                JUST_NOW,
                "would-remove-label",
            ),
            ("within-grace", [], _fail, None, JUST_NOW, "would-notify-within-grace"),
            ("past-grace", [], _fail, None, TWO_DAYS_AGO, "would-close"),
        ],
    )
    def test_dry_run_previews_without_side_effects(
        self,
        triage_module,
        monkeypatch,
        scenario,
        labels,
        judge,
        score,
        created,
        expected,
    ):
        pr = _make_pr(labels=labels, created_at=created)
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        rec = _Recorder(triage_module, monkeypatch)

        result = _gate(triage_module, close=False, judge=judge, greptile_score=score)

        assert result["action"] == expected
        # Dry run touches nothing.
        assert rec.added == [] and rec.removed == [] and rec.closed == []
        assert rec.comments == []
        assert "comment" in result  # preview body still surfaced


class TestReviewGateGuards:
    def test_skips_internal_author(self, triage_module, monkeypatch):
        pr = _make_pr(author_association="MEMBER", user={"login": "krrish"})
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        result = _gate(
            triage_module,
            judge=lambda p: pytest.fail("no LLM for internal"),
            allowlist=frozenset(),
        )
        assert result["action"] == "skip-internal-author"

    def test_skips_closed_pr(self, triage_module, monkeypatch):
        pr = _make_pr(state="closed")
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        result = _gate(triage_module, judge=lambda p: pytest.fail("no LLM for closed"))
        assert result["action"] == "skip-not-open"

    def test_llm_error_is_non_destructive(self, triage_module, monkeypatch):
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: _make_pr())
        rec = _Recorder(triage_module, monkeypatch)

        def boom(prompt):
            raise RuntimeError("api down")

        result = _gate(triage_module, judge=boom, greptile_score=None)

        assert result["action"] == "skip-llm-error"
        assert rec.closed == [] and rec.added == [] and rec.removed == []

    def test_full_recovery_cycle(self, triage_module, monkeypatch):
        """pass -> regress -> recover, threading labels/comments like GitHub would."""
        state = {"labels": [], "comments": []}

        def fake_fetch(repo, n):
            return _make_pr(labels=list(state["labels"]), created_at=JUST_NOW)

        monkeypatch.setattr(triage_module, "fetch_pr", fake_fetch)
        monkeypatch.setattr(
            triage_module,
            "post_comment",
            lambda repo, n, body: state["comments"].append(
                {"user": {"login": "github-actions[bot]"}, "body": body}
            ),
        )
        monkeypatch.setattr(
            triage_module,
            "add_label",
            lambda repo, n, label: state["labels"].append({"name": label}),
        )
        monkeypatch.setattr(
            triage_module,
            "remove_label",
            lambda repo, n, label: state["labels"].clear(),
        )
        monkeypatch.setattr(
            triage_module, "close_pr", lambda repo, n: pytest.fail("must not close")
        )

        # 1) passes -> tagged
        r1 = _gate(
            triage_module, judge=_pass, greptile_score=5, comments=state["comments"]
        )
        assert r1["action"] == "labeled-ready"
        assert any(lbl["name"] == "ready for review" for lbl in state["labels"])

        # 2) regresses -> tag removed, comment posted, PR still open
        r2 = _gate(
            triage_module, judge=_fail, greptile_score=2, comments=state["comments"]
        )
        assert r2["action"] == "label-removed-regressed"
        assert state["labels"] == []

        # 3) fixed again -> "all clear" + tag back
        r3 = _gate(
            triage_module, judge=_pass, greptile_score=5, comments=state["comments"]
        )
        assert r3["action"] == "labeled-ready"
        assert any(lbl["name"] == "ready for review" for lbl in state["labels"])
        assert "all clear" in state["comments"][-1]["body"].lower()


class TestReviewGateAllowlist:
    """While the dogfood allowlist is active it is the sole author gate:
    only the named accounts pass, and for them the internal-author exemption
    is bypassed. Emptying it restores the normal internal-author skip."""

    def test_should_skip_author_not_on_allowlist(self, triage_module, monkeypatch):
        pr = _make_pr(user={"login": "random-oss-dev"})
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        rec = _Recorder(triage_module, monkeypatch)
        result = _gate(
            triage_module, judge=lambda p: pytest.fail("no LLM for non-allowlisted")
        )
        assert result["action"] == "skip-not-allowlisted"
        assert rec.added == [] and rec.comments == [] and rec.closed == []

    def test_should_act_on_allowlisted_internal_author(
        self, triage_module, monkeypatch
    ):
        pr = _make_pr(author_association="MEMBER", user={"login": "mateo-berri"})
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        rec = _Recorder(triage_module, monkeypatch)
        result = _gate(triage_module, judge=_pass, greptile_score=5)
        assert result["action"] == "labeled-ready"
        assert rec.added == [triage_module.READY_FOR_REVIEW_LABEL]

    def test_empty_allowlist_restores_internal_skip(self, triage_module, monkeypatch):
        pr = _make_pr(author_association="MEMBER", user={"login": "krrish"})
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        result = _gate(
            triage_module,
            judge=lambda p: pytest.fail("no LLM for internal"),
            allowlist=frozenset(),
        )
        assert result["action"] == "skip-internal-author"
