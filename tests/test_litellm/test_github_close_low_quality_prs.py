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
        author_login: str = "mateo-berri",
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
            "author": {"login": author_login},
            "url": f"https://example.com/pr/{number}",
        }

    @pytest.fixture(autouse=True)
    def _external_author(self, closer_module, monkeypatch):
        """Treat every test PR as external unless overridden."""
        monkeypatch.setattr(
            closer_module, "is_external_pr_author", lambda pr, repo: True
        )

    def test_should_warn_drafts_when_score_low_first_time(
        self, closer_module, _now, monkeypatch
    ):
        # Drafts are NOT a free pass — the open-PR queue should reflect any
        # PR that needs human attention regardless of draft status. Authors
        # who need a long-lived draft can use the `wip` opt-out label.
        # First run: warn the contributor (1-day grace), don't close yet.
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
        assert action == "warn-grace"
        assert score == 2 and age == 0

    def test_should_warn_brand_new_pr_when_min_age_zero(
        self, closer_module, _now, monkeypatch
    ):
        # `min_age_days=0` means no age filter — a freshly-opened PR is
        # eligible the moment Greptile scores it below threshold. The
        # first detection still goes through the warn-grace step rather
        # than closing immediately, giving the contributor 2 hours to
        # respond before the next run actually closes the PR.
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
        assert action == "warn-grace"
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

    def test_should_warn_when_old_and_low_score_no_prior_warning(
        self, closer_module, _now, monkeypatch
    ):
        # Even an old PR that still has no grace warning gets one on the
        # first eligible run — the daily cron is the natural cadence, so
        # an existing-but-never-warned PR enters the grace flow normally.
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
        assert action == "warn-grace"
        assert score == 3 and age == 10

    def test_should_close_when_grace_warning_aged_out_and_score_still_low(
        self, closer_module, _now, monkeypatch
    ):
        # Day-1 the closer posted a warning. Day-2 the PR still scores <4
        # AND the warning is older than `GRACE_PERIOD_SECONDS`, so the
        # action flips to `close`. This is the "grace expired" path.
        old_warning = {
            "user": {"login": "github-actions[bot]"},
            "body": (
                "you have 2 hours to fix this\n\n" + closer_module.GRACE_COMMENT_MARKER
            ),
            "created_at": (
                _now - dt.timedelta(seconds=closer_module.GRACE_PERIOD_SECONDS + 60)
            )
            .isoformat()
            .replace("+00:00", "Z"),
            "updated_at": "2026-05-15T00:00:00Z",
        }
        monkeypatch.setattr(
            closer_module,
            "fetch_pr_comments",
            lambda *a, **kw: [
                _greptile_comment(
                    "<h3>Confidence Score: 1/5</h3>",
                    updated_at="2026-05-15T00:00:00Z",
                ),
                old_warning,
            ],
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

    def test_should_skip_when_grace_warning_within_window(
        self, closer_module, _now, monkeypatch
    ):
        # Within the 2-hour grace window the closer must NOT close the
        # PR even if the score is still low. The warning is only an hour
        # old; give the contributor time to push fixes before destruction.
        recent_warning = {
            "user": {"login": "github-actions[bot]"},
            "body": "warning text\n\n" + closer_module.GRACE_COMMENT_MARKER,
            "created_at": (_now - dt.timedelta(hours=1))
            .isoformat()
            .replace("+00:00", "Z"),
        }
        monkeypatch.setattr(
            closer_module,
            "fetch_pr_comments",
            lambda *a, **kw: [
                _greptile_comment("Confidence Score: 2/5"),
                recent_warning,
            ],
        )
        action, score, _ = closer_module.evaluate_pr(
            self._make_pr(created_days_ago=10),
            now=_now,
            min_age_days=0,
            min_score=4,
            repo=None,
            optout_labels=set(),
        )
        assert action == "skip-in-grace-period"
        assert score == 2

    def test_should_warn_grace_for_swiftwinds_not_close_immediately(
        self, closer_module, _now, monkeypatch
    ):
        # Regression: SwiftWinds (the dogfood account) used to be in a
        # now-removed `IMMEDIATE_CLOSE_LOGINS` bypass that closed on first
        # detection. It must now follow the SAME grace path as every other
        # external author: warn first, close only after the window elapses.
        monkeypatch.setattr(
            closer_module,
            "fetch_pr_comments",
            lambda *a, **kw: [_greptile_comment("Confidence Score: 1/5")],
        )
        action, score, _ = closer_module.evaluate_pr(
            self._make_pr(created_days_ago=0, author_login="SwiftWinds"),
            now=_now,
            min_age_days=0,
            min_score=4,
            repo=None,
            optout_labels=set(),
        )
        assert action == "warn-grace"
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
            self._make_pr(created_days_ago=14, author_login="krrishdholakia"),
            now=_now,
            min_age_days=7,
            min_score=4,
            repo=None,
            optout_labels=set(),
            allowlist=frozenset(),
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


class TestSecondsSinceLastGraceWarning:
    """Grace-period detection: only counts comments by the bot identity
    that contain the shared `GRACE_COMMENT_MARKER`."""

    def _make_marker_comment(
        self,
        closer_module,
        *,
        login: str = "github-actions[bot]",
        created_at: str = "2026-05-16T00:00:00Z",
        include_marker: bool = True,
    ) -> dict:
        body = "warning text"
        if include_marker:
            body += "\n\n" + closer_module.GRACE_COMMENT_MARKER
        return {
            "user": {"login": login},
            "body": body,
            "created_at": created_at,
        }

    def test_should_return_none_when_no_marker_comment(self, closer_module):
        comments = [
            {
                "user": {"login": "github-actions[bot]"},
                "body": "Some other bot comment",
                "created_at": "2026-05-16T00:00:00Z",
            }
        ]
        assert closer_module.seconds_since_last_grace_warning(comments) is None

    def test_should_return_none_for_empty(self, closer_module):
        assert closer_module.seconds_since_last_grace_warning([]) is None

    def test_should_ignore_non_bot_comments_with_marker(self, closer_module):
        # If a curious user quotes the marker in a comment, we must NOT
        # treat it as a bot warning. The grace timer would then never fire.
        comments = [
            self._make_marker_comment(closer_module, login="random-user"),
        ]
        assert closer_module.seconds_since_last_grace_warning(comments) is None

    def test_should_pick_latest_marker_comment(self, closer_module):
        # When multiple grace warnings exist (e.g. a re-open cycle), use
        # the most recent one to compute the age.
        comments = [
            self._make_marker_comment(closer_module, created_at="2026-05-15T00:00:00Z"),
            self._make_marker_comment(closer_module, created_at="2026-05-16T23:00:00Z"),
        ]
        now = dt.datetime(2026, 5, 17, 0, 0, 0, tzinfo=dt.timezone.utc)
        age = closer_module.seconds_since_last_grace_warning(comments, now=now)
        # 1h = 3600s
        assert age == 3600.0


class TestGraceWarningCommentText:
    """Pin the user-facing language in the grace warning comment so the
    grace-window and `@greptileai still works after close` promises
    don't get accidentally dropped in a future refactor.
    """

    def test_should_state_grace_window(self, closer_module):
        body = closer_module.format_grace_warning_comment(score=2, threshold=4)
        # The user's PR explicitly said "specify in the comment" — pin
        # that the grace window appears in the comment.
        assert "2 hours" in body

    def test_should_mention_agent_shin_reconsider(self, closer_module):
        body = closer_module.format_grace_warning_comment(score=2, threshold=4)
        assert "@agent-shin reconsider" in body

    def test_should_promise_greptileai_works_after_close(self, closer_module):
        body = closer_module.format_grace_warning_comment(score=2, threshold=4)
        assert "@greptileai" in body
        assert "even after the PR is closed" in body

    def test_should_carry_grace_marker(self, closer_module):
        # The marker is what `seconds_since_last_grace_warning` greps for
        # to detect a prior warning — dropping it would silently break
        # the cooldown.
        body = closer_module.format_grace_warning_comment(score=2, threshold=4)
        assert closer_module.GRACE_COMMENT_MARKER in body

    def test_close_comment_should_mention_greptileai_post_close(self, closer_module):
        # The close comment should ALSO point at the @greptileai post-close
        # re-review path so contributors see the same options whether they
        # read the warning or only catch the close comment.
        body = closer_module.format_close_comment(score=2, threshold=4)
        assert "@greptileai" in body
        assert "even after the PR is closed" in body

    def test_close_comment_should_advertise_reconsider(self, closer_module):
        body = closer_module.format_close_comment(score=2, threshold=4)
        assert "@agent-shin reconsider" in body

    def test_close_comment_should_carry_agent_shin_close_marker(self, closer_module):
        # The close comment advertises `@agent-shin reconsider`, and the
        # reconsider reopen guard (`was_closed_by_agent_shin`) only treats a
        # PR as Agent-Shin-closed when the close comment carries this marker.
        # Dropping it silently breaks the advertised recovery path for every
        # PR closed by this daily sweep.
        body = closer_module.format_close_comment(score=2, threshold=4)
        assert closer_module.AGENT_SHIN_CLOSE_MARKER in body

    def test_close_comment_should_state_score_and_threshold(self, closer_module):
        body = closer_module.format_close_comment(score=1, threshold=4)
        assert "1/5" in body
        assert "4/5" in body


class TestHasOptoutLabel:
    def test_should_match_label_case_insensitively(self, closer_module):
        pr = {"labels": [{"name": "Do Not Close"}, {"name": "bug"}]}
        assert closer_module.has_optout_label(pr, {"do not close"}) is True

    def test_should_return_false_when_no_match(self, closer_module):
        pr = {"labels": [{"name": "bug"}, {"name": "enhancement"}]}
        assert closer_module.has_optout_label(pr, {"wip", "keep open"}) is False

    def test_should_handle_missing_labels(self, closer_module):
        assert closer_module.has_optout_label({}, {"wip"}) is False


class TestListOpenItemsNoCap:
    """The bulk sweeps must fetch the ENTIRE open backlog.

    Regression guard for the old hard-coded ``--limit 1000``: gh lists
    newest-first, so a low cap silently dropped the *oldest* PRs/issues —
    exactly the stale ones a low-quality sweep exists to catch.
    """

    @staticmethod
    def _shared(closer_module):
        # `closer_module` loading puts `.github/scripts` on sys.path and
        # imports agent_shin_shared, so it's already in sys.modules.
        import agent_shin_shared

        return agent_shin_shared

    def _capture_gh_args(self, closer_module, monkeypatch, *, returns="[]"):
        shared = self._shared(closer_module)
        captured: dict = {}

        def fake_gh(*args):
            captured["args"] = args
            return returns

        # `list_open_items` looks up `gh` in agent_shin_shared's namespace.
        monkeypatch.setattr(shared, "gh", fake_gh)
        return shared, captured

    def test_list_open_items_passes_no_cap_limit_not_1000(
        self, closer_module, monkeypatch
    ):
        shared, captured = self._capture_gh_args(closer_module, monkeypatch)
        shared.list_open_items("pr", repo="o/r", fields="number,title")
        args = captured["args"]
        assert "--limit" in args
        limit_value = args[args.index("--limit") + 1]
        assert limit_value == str(shared.GH_LIST_ALL_LIMIT)
        assert limit_value != "1000"
        # A meaningful ceiling: comfortably above any realistic open backlog.
        assert shared.GH_LIST_ALL_LIMIT >= 100_000

    def test_list_open_items_uses_dedicated_command_state_and_fields(
        self, closer_module, monkeypatch
    ):
        shared, captured = self._capture_gh_args(closer_module, monkeypatch)
        shared.list_open_items("issue", repo="o/r", fields="number")
        args = captured["args"]
        assert args[0] == "issue" and args[1] == "list"
        assert args[args.index("--state") + 1] == "open"
        assert args[args.index("--json") + 1] == "number"
        assert tuple(args[-2:]) == ("--repo", "o/r")

    def test_list_open_items_omits_repo_when_none(self, closer_module, monkeypatch):
        shared, captured = self._capture_gh_args(closer_module, monkeypatch)
        shared.list_open_items("pr", repo=None, fields="number")
        assert "--repo" not in captured["args"]

    def test_list_open_items_parses_json_array(self, closer_module, monkeypatch):
        shared, _ = self._capture_gh_args(
            closer_module, monkeypatch, returns='[{"number": 1}, {"number": 2}]'
        )
        items = shared.list_open_items("pr", repo=None, fields="number")
        assert [i["number"] for i in items] == [1, 2]

    def test_list_open_items_rejects_unknown_kind(self, closer_module):
        shared = self._shared(closer_module)
        with pytest.raises(ValueError):
            shared.list_open_items("both", repo="o/r", fields="number")

    def test_fetch_open_prs_delegates_with_no_cap(self, closer_module, monkeypatch):
        shared, captured = self._capture_gh_args(closer_module, monkeypatch)
        closer_module.fetch_open_prs("o/r")
        args = captured["args"]
        assert args[0] == "pr"
        assert args[args.index("--limit") + 1] == str(shared.GH_LIST_ALL_LIMIT)
        # Still requests every field downstream evaluate_pr / labels logic needs.
        assert "createdAt" in args[args.index("--json") + 1]


class TestEvaluatePrAllowlist:
    """While the dogfood allowlist is active `evaluate_pr` only acts on the
    named accounts and bypasses the external-only restriction for them.
    Emptying it restores the internal-author skip."""

    @pytest.fixture(autouse=True)
    def _now(self):
        return dt.datetime(2026, 5, 17, tzinfo=dt.timezone.utc)

    def _make_pr(self, *, author_login: str, created_days_ago: int = 10) -> dict:
        created = dt.datetime(2026, 5, 17, tzinfo=dt.timezone.utc) - dt.timedelta(
            days=created_days_ago
        )
        return {
            "number": 1,
            "title": "PR #1",
            "createdAt": created.isoformat().replace("+00:00", "Z"),
            "isDraft": False,
            "labels": [],
            "author": {"login": author_login},
            "url": "https://example.com/pr/1",
        }

    def test_should_skip_author_not_on_allowlist(
        self, closer_module, _now, monkeypatch
    ):
        monkeypatch.setattr(
            closer_module,
            "fetch_pr_comments",
            lambda *a, **kw: pytest.fail("must not fetch comments for non-allowlisted"),
        )
        action, score, _ = closer_module.evaluate_pr(
            self._make_pr(author_login="random-oss-dev"),
            now=_now,
            min_age_days=0,
            min_score=4,
            repo=None,
            optout_labels=set(),
        )
        assert action == "skip-not-allowlisted"
        assert score is None

    def test_should_act_on_allowlisted_internal_author(
        self, closer_module, _now, monkeypatch
    ):
        monkeypatch.setattr(
            closer_module, "is_external_pr_author", lambda pr, repo: False
        )
        monkeypatch.setattr(
            closer_module,
            "fetch_pr_comments",
            lambda *a, **kw: [_greptile_comment("Confidence Score: 2/5")],
        )
        action, score, _ = closer_module.evaluate_pr(
            self._make_pr(author_login="mateo-berri", created_days_ago=0),
            now=_now,
            min_age_days=0,
            min_score=4,
            repo=None,
            optout_labels=set(),
        )
        assert action == "warn-grace"
        assert score == 2

    def test_empty_allowlist_restores_internal_skip(
        self, closer_module, _now, monkeypatch
    ):
        monkeypatch.setattr(
            closer_module, "is_external_pr_author", lambda pr, repo: False
        )
        monkeypatch.setattr(
            closer_module,
            "fetch_pr_comments",
            lambda *a, **kw: pytest.fail("must not fetch comments for internal"),
        )
        action, score, _ = closer_module.evaluate_pr(
            self._make_pr(author_login="krrishdholakia"),
            now=_now,
            min_age_days=0,
            min_score=4,
            repo=None,
            optout_labels=set(),
            allowlist=frozenset(),
        )
        assert action == "skip-internal"

    def test_allowlist_constant_is_the_two_dogfood_accounts(self, closer_module):
        assert closer_module.ALLOWLIST_LOGINS == frozenset(
            {"mateo-berri", "swiftwinds"}
        )


class TestDryRunGateOnClose:
    """Regression: the daily sweep is dry-run unless `--close` is passed
    (the workflow only adds it when `AGENT_SHIN_ENABLED=true`). A closeable
    PR (low score, grace window elapsed) must be DETECTED and reported as
    "would close", but the dry run must never make a real GitHub mutation,
    so merging Agent Shin stays inert by default."""

    def _closeable_pr(self) -> dict:
        return {
            "number": 7,
            "title": "thin PR",
            "createdAt": "2026-05-10T00:00:00Z",
            "isDraft": False,
            "labels": [],
            "author": {"login": "SwiftWinds"},
            "url": "https://example.com/pr/7",
        }

    def test_dry_run_sweep_detects_but_does_not_close(
        self, closer_module, monkeypatch, capsys
    ):
        aged_out_warning = {
            "user": {"login": "github-actions[bot]"},
            "body": "warned\n\n" + closer_module.GRACE_COMMENT_MARKER,
            # Far enough in the past that it's aged out regardless of
            # GRACE_PERIOD_SECONDS, since main() pins `now` to real time.
            "created_at": "2020-01-01T00:00:00Z",
        }
        monkeypatch.setattr(
            closer_module, "fetch_open_prs", lambda repo: [self._closeable_pr()]
        )
        monkeypatch.setattr(
            closer_module,
            "fetch_pr_comments",
            lambda *a, **kw: [
                _greptile_comment("Confidence Score: 1/5"),
                aged_out_warning,
            ],
        )
        # Any real GitHub mutation during a dry run is the bug under test.
        monkeypatch.setattr(
            closer_module,
            "gh",
            lambda *a, **kw: pytest.fail(f"dry run must not call gh: {a}"),
        )
        monkeypatch.setattr(sys, "argv", ["close_low_quality_prs.py"])

        rc = closer_module.main()

        assert rc == 0
        # The PR is detected as closeable, just not acted on.
        assert "Total would close: 1" in capsys.readouterr().out
