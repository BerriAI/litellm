"""Unit tests for `.github/scripts/triage_with_llm.py` (Agent Shin)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / ".github" / "scripts" / "triage_with_llm.py"
)


@pytest.fixture(scope="module")
def triage_module():
    spec = importlib.util.spec_from_file_location("triage_with_llm", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["triage_with_llm"] = module
    spec.loader.exec_module(module)
    return module


class TestIsInternalContributor:
    @pytest.mark.parametrize("association", ["OWNER", "MEMBER", "COLLABORATOR"])
    def test_should_mark_org_associations_as_internal(self, triage_module, association):
        item = {
            "author_association": association,
            "user": {"login": "krrishdholakia"},
        }
        assert triage_module.is_internal_contributor(item) is True

    @pytest.mark.parametrize(
        "association",
        ["CONTRIBUTOR", "FIRST_TIME_CONTRIBUTOR", "FIRST_TIMER", "NONE"],
    )
    def test_should_mark_outside_associations_as_external(
        self, triage_module, association
    ):
        item = {
            "author_association": association,
            "user": {"login": "random-oss-dev"},
        }
        assert triage_module.is_internal_contributor(item) is False

    @pytest.mark.parametrize(
        "item",
        [
            {"author_association": "", "user": {"login": "random-oss-dev"}},
            {"user": {"login": "random-oss-dev"}},  # association field absent
        ],
    )
    def test_should_fail_safe_when_author_association_is_missing(
        self, triage_module, item
    ):
        # Fail-safe: an empty/missing association must never make a PR
        # eligible for the destructive close path. Treat as internal (skip).
        assert triage_module.is_internal_contributor(item) is True

    @pytest.mark.parametrize(
        "login",
        ["dependabot[bot]", "greptile-apps[bot]", "dependabot", "github-actions"],
    )
    def test_should_skip_bot_accounts_regardless_of_association(
        self, triage_module, login
    ):
        item = {"author_association": "NONE", "user": {"login": login}}
        assert triage_module.is_internal_contributor(item) is True


class TestHasLinkedIssue:
    @pytest.mark.parametrize(
        "body",
        [
            "Fixes #1234",
            "closes #1",
            "Resolves #99",
            "fix #42 — this addresses the regression",
            "Closes https://github.com/BerriAI/litellm/issues/27000",
            "Resolved https://github.com/BerriAI/litellm/issues/27001",
        ],
    )
    def test_should_detect_common_link_phrases(self, triage_module, body):
        assert triage_module.has_linked_issue(body) is True

    @pytest.mark.parametrize(
        "body",
        [
            "",
            "Some change",
            # Casual mentions must NOT auto-pass — they should fall through to
            # the LLM judge so the stricter "not a passing mention" rule fires.
            "See #1234",
            "see #1234 for context",
            "ref #1234",
            "Refs https://github.com/BerriAI/litellm/issues/27000",
            "this addresses #1234",
        ],
    )
    def test_should_not_auto_pass_casual_mentions(self, triage_module, body):
        assert triage_module.has_linked_issue(body) is False

    def test_should_not_detect_when_only_html_comment_template(self, triage_module):
        body = "<!-- e.g. Fixes #1234 -->"
        assert triage_module.has_linked_issue(body) is False


class TestStripHtmlComments:
    def test_should_remove_single_line_comments(self, triage_module):
        text = "before <!-- placeholder --> after"
        assert "placeholder" not in triage_module.strip_html_comments(text)

    def test_should_remove_multiline_comments(self, triage_module):
        text = "kept\n<!--\nlots of placeholder text\nFixes #1\n-->\nkept2"
        cleaned = triage_module.strip_html_comments(text)
        assert "Fixes #1" not in cleaned
        assert "kept" in cleaned and "kept2" in cleaned

    def test_should_handle_none(self, triage_module):
        assert triage_module.strip_html_comments(None) == ""


class TestCloseCommentText:
    """Pin the user-facing language in close comments so changes are intentional."""

    def test_pr_close_comment_should_recommend_new_pr_primarily(self, triage_module):
        body = triage_module.format_pr_close_comment(
            {"verdict": "fail", "missing": ["QA proof"], "explanation": "thin"}
        )
        # Primary path: open a new PR (because OSS authors can't reopen a
        # bot-closed PR). Secondary path: `@agent-shin reconsider`.
        assert "Open a new PR" in body
        assert "@agent-shin reconsider" in body
        # Old advice that no longer works for OSS contributors must NOT
        # appear (they can't reopen a PR closed by a bot/maintainer).
        assert "Reopen the PR" not in body

    def test_reopen_comment_should_carry_reconsider_marker(self, triage_module):
        # The marker is what the rate-limit guard greps for to detect a
        # prior reconsider verdict on the same PR. If the marker ever
        # gets dropped from this comment, the cooldown silently breaks
        # and a contributor can spam `@agent-shin reconsider` to burn
        # LLM budget.
        body = triage_module.format_reopen_comment("pr")
        assert triage_module.RECONSIDER_COMMENT_MARKER in body

    def test_still_failing_comment_should_carry_reconsider_marker(self, triage_module):
        body = triage_module.format_reconsider_still_failing_comment(
            "pr",
            {"verdict": "fail", "missing": ["QA proof"], "explanation": "thin"},
        )
        assert triage_module.RECONSIDER_COMMENT_MARKER in body

    def test_pr_close_comment_should_not_promise_automatic_reopen_on_open(
        self, triage_module
    ):
        # The previous comment said "I'll re-evaluate automatically" — that
        # only worked because the author could reopen, which they often
        # can't. The new wording must point them at the comment trigger or
        # a new PR instead.
        body = triage_module.format_pr_close_comment(
            {"verdict": "fail", "missing": [], "explanation": ""}
        )
        assert "I'll re-evaluate automatically" not in body

    def test_issue_close_comment_should_use_reconsider_trigger(self, triage_module):
        # OSS authors have read access, which only lets them reopen issues
        # they closed themselves; they CANNOT reopen an issue a maintainer or
        # bot closed. So the recovery path is `@agent-shin reconsider` (the
        # bot reopens), exactly like the PR path. If this regresses to "reopen
        # it yourself", contributors hit a dead end on bot-closed issues.
        body = triage_module.format_issue_close_comment(
            {"verdict": "fail", "missing": ["repro"], "explanation": "thin"}
        )
        assert "@agent-shin reconsider" in body

    def test_pr_close_comment_should_link_blog_explainer(self, triage_module):
        # The blog post is the canonical public explanation of what the bot
        # checks and why. Every action-required bot comment must link to it
        # so contributors landing on a bot-closed PR can self-serve context
        # without pinging a maintainer.
        body = triage_module.format_pr_close_comment(
            {"verdict": "fail", "missing": [], "explanation": ""}
        )
        assert "https://docs.litellm.ai/blog/agent-shin-triage" in body

    def test_issue_close_comment_should_link_blog_explainer(self, triage_module):
        body = triage_module.format_issue_close_comment(
            {"verdict": "fail", "missing": [], "explanation": ""}
        )
        assert "https://docs.litellm.ai/blog/agent-shin-triage" in body

    def test_pr_close_comment_should_flag_mocked_tests_as_insufficient_proof(
        self, triage_module
    ):
        # The PR rubric was tightened to require end-to-end QA proof and
        # explicitly exclude mocked-dependency unit tests. The user-facing
        # close comment must say so — otherwise contributors will keep
        # re-submitting "pytest passed (mocks)" runs and getting closed
        # again with no explanation of why.
        body = triage_module.format_pr_close_comment(
            {"verdict": "fail", "missing": [], "explanation": ""}
        )
        assert "end-to-end qa proof" in body.lower()
        assert "mock" in body.lower()

    def test_all_agent_shin_comments_should_use_bullet_train_emoji(self, triage_module):
        # The bullet train (🚅) is Agent Shin's symbol, matching the LiteLLM
        # logo; the previous wave (👋) was generic and didn't match the bot's
        # identity. Every action-required comment the bot can post must use the
        # bullet train so the contributor recognizes who's writing without
        # reading the signoff.
        verdict = {"verdict": "fail", "missing": [], "explanation": ""}
        comments = {
            "pr_close": triage_module.format_pr_close_comment(verdict),
            "issue_close": triage_module.format_issue_close_comment(verdict),
            "pr_grace": triage_module.format_grace_warning_pr_comment(verdict),
            "issue_grace": triage_module.format_grace_warning_issue_comment(verdict),
            "within_grace": triage_module.format_within_grace_comment(
                [], "", grace_days=1
            ),
        }
        for name, body in comments.items():
            assert "🚅" in body, f"{name} comment is missing the bullet train emoji"
            assert "👋" not in body, f"{name} comment still uses the old wave emoji"

    def test_pr_close_comment_should_show_what_pr_got_right(self, triage_module):
        # The user explicitly asked for a "things you got right" section so
        # the comment doesn't read as pure rejection. When the judge confirms
        # a field is present (e.g. linked_issue), the bullet for it MUST
        # appear in the close comment.
        body = triage_module.format_pr_close_comment(
            {
                "verdict": "fail",
                "linked_issue": True,
                "has_problem_description": True,
                "has_expected_vs_actual": False,
                "has_qa_proof": False,
                "missing": ["QA proof"],
                "explanation": "no proof",
            }
        )
        assert "What you got right" in body
        # The two present fields surface as ✅ bullets; the two absent
        # fields do not get a ✅ bullet (the QA-proof rubric block still
        # mentions the concept, but only the affirmed fields get checkmarks).
        assert "- ✅ Linked a related GitHub issue" in body
        assert "- ✅ Clear problem description" in body
        assert "- ✅ Expected vs. actual behavior" not in body
        assert "- ✅ End-to-end QA proof" not in body

    def test_pr_close_comment_should_omit_present_section_when_nothing_present(
        self, triage_module
    ):
        # If the judge says nothing is present (every flag False), the
        # "what you got right" block is skipped entirely — better to omit
        # than to render "What you got right: (nothing)".
        body = triage_module.format_pr_close_comment(
            {
                "verdict": "fail",
                "linked_issue": False,
                "has_problem_description": False,
                "has_expected_vs_actual": False,
                "has_qa_proof": False,
                "missing": [],
                "explanation": "",
            }
        )
        assert "What you got right" not in body

    def test_issue_close_comment_should_show_what_issue_got_right(self, triage_module):
        # `has_expected_vs_actual` is present, the end-to-end bug evidence is
        # not: the "what you got right" block must surface the former and omit
        # the latter (no "✅ (nothing)"-style noise for absent items).
        body = triage_module.format_issue_close_comment(
            {
                "verdict": "fail",
                "kind": "bug",
                "has_repro": False,
                "has_expected_vs_actual": True,
                "missing": ["end-to-end evidence of the bug"],
                "explanation": "no repro shown",
            }
        )
        assert "What you got right" in body
        assert "Expected vs. actual behavior" in body
        assert "- ✅ End-to-end evidence of the bug" not in body

    def test_close_comments_should_use_softer_park_for_later_framing(
        self, triage_module
    ):
        # User feedback: the messaging shouldn't feel like punishment. The
        # comment must explicitly frame close as a "park this for later," not
        # a rejection, and ground that in the queue-hygiene reason.
        for body in (
            triage_module.format_pr_close_comment(
                {"verdict": "fail", "missing": [], "explanation": ""}
            ),
            triage_module.format_issue_close_comment(
                {"verdict": "fail", "missing": [], "explanation": ""}
            ),
        ):
            assert "park this for later" in body
            assert (
                "not a rejection" in body
                or "isn't a rejection" in body
                or ("isn't us saying" in body)
            )

    def test_only_close_comments_carry_the_agent_shin_close_marker(self, triage_module):
        # The reconsider reopen guard keys off AGENT_SHIN_CLOSE_MARKER to tell
        # an Agent Shin close from a same-identity close by another workflow.
        # That only works if the marker is stamped on the close comments and
        # NOT on the grace warnings (which don't close anything).
        verdict = {"verdict": "fail", "missing": [], "explanation": ""}
        marker = triage_module.AGENT_SHIN_CLOSE_MARKER
        assert marker in triage_module.format_pr_close_comment(verdict)
        assert marker in triage_module.format_issue_close_comment(verdict)
        assert marker not in triage_module.format_grace_warning_pr_comment(verdict)
        assert marker not in triage_module.format_grace_warning_issue_comment(verdict)


class TestWasClosedByAgentShin:
    """Bot-closed guard: only Agent Shin's own closures are reopen candidates."""

    @staticmethod
    def _stub_close_event(
        triage_module,
        monkeypatch,
        *,
        actor: str | None,
        closed_at: object = "now",
    ):
        """Stub the most recent `closed` event used by the guard.

        `actor` is the login that closed the item. `closed_at` defaults
        to "now" so the marker comment (stubbed at 42s ago) reads as
        recent enough relative to the close; tests can pass a concrete
        ``datetime`` to simulate older closes (e.g. the stale-marker
        regression scenario).
        """
        import datetime as real_dt

        if closed_at == "now":
            closed_at = real_dt.datetime.now(real_dt.timezone.utc)
        monkeypatch.setattr(
            triage_module,
            "fetch_last_close_event",
            lambda repo, n: (actor, closed_at),
        )

    @staticmethod
    def _stub_close_marker_present(
        triage_module, monkeypatch, *, present: bool, age_seconds: float = 42.0
    ):
        """Stub the Agent Shin close-comment marker lookup.

        `was_closed_by_agent_shin` requires the closing actor AND a
        recent Agent Shin close comment; these tests pin the latter so
        they exercise the actor half in isolation.
        """
        monkeypatch.setattr(
            triage_module,
            "seconds_since_last_agent_shin_close",
            lambda *a, **kw: age_seconds if present else None,
        )

    def test_should_return_true_when_bot_closed_and_close_comment_present(
        self, triage_module, monkeypatch
    ):
        self._stub_close_event(triage_module, monkeypatch, actor="github-actions[bot]")
        self._stub_close_marker_present(triage_module, monkeypatch, present=True)
        assert triage_module.was_closed_by_agent_shin("o/r", 1) is True

    def test_should_return_false_when_bot_closed_but_no_agent_shin_comment(
        self, triage_module, monkeypatch
    ):
        # The `github-actions[bot]` identity is shared across workflows. A
        # stale/duplicate sweep closing under that identity must NOT let
        # @agent-shin reconsider reopen the item: without an Agent Shin close
        # comment the guard fails closed.
        self._stub_close_event(triage_module, monkeypatch, actor="github-actions[bot]")
        self._stub_close_marker_present(triage_module, monkeypatch, present=False)
        assert triage_module.was_closed_by_agent_shin("o/r", 1) is False

    def test_should_return_false_when_last_close_actor_is_maintainer(
        self, triage_module, monkeypatch
    ):
        # A maintainer closed it (e.g. duplicate, security, design). The
        # bot must refuse to reopen on @agent-shin reconsider even if an
        # earlier Agent Shin close comment is still on the thread.
        self._stub_close_event(triage_module, monkeypatch, actor="krrishdholakia")
        self._stub_close_marker_present(triage_module, monkeypatch, present=True)
        assert triage_module.was_closed_by_agent_shin("o/r", 1) is False

    def test_should_fail_closed_when_no_close_event(self, triage_module, monkeypatch):
        # If the events API returns nothing (network blip, repo permission
        # quirk), the guard must fail-closed: refuse to reopen rather than
        # assume the bot did it.
        self._stub_close_event(triage_module, monkeypatch, actor=None, closed_at=None)
        self._stub_close_marker_present(triage_module, monkeypatch, present=True)
        assert triage_module.was_closed_by_agent_shin("o/r", 1) is False

    def test_should_fail_closed_when_close_event_has_no_timestamp(
        self, triage_module, monkeypatch
    ):
        # Without a usable close timestamp the guard cannot prove the
        # marker comment belongs to the latest close; fail-closed.
        self._stub_close_event(
            triage_module, monkeypatch, actor="github-actions[bot]", closed_at=None
        )
        self._stub_close_marker_present(triage_module, monkeypatch, present=True)
        assert triage_module.was_closed_by_agent_shin("o/r", 1) is False

    def test_should_return_false_when_marker_predates_latest_close(
        self, triage_module, monkeypatch
    ):
        # Regression for the stale-marker bug: Agent Shin closed once
        # (marker stamped), reconsider reopened, and a different workflow
        # later closed under the same bot identity without stamping the
        # marker. The old marker is still on the thread but does NOT
        # belong to the latest close, so reconsider must not reopen.
        import datetime as real_dt

        now = real_dt.datetime.now(real_dt.timezone.utc)
        # Latest close happened a minute ago.
        self._stub_close_event(
            triage_module,
            monkeypatch,
            actor="github-actions[bot]",
            closed_at=now - real_dt.timedelta(seconds=60),
        )
        # The most recent Agent Shin marker is from an hour ago (a prior
        # closed/reopened cycle), which is well outside the skew window.
        self._stub_close_marker_present(
            triage_module, monkeypatch, present=True, age_seconds=3600.0
        )
        assert triage_module.was_closed_by_agent_shin("o/r", 1) is False

    def test_should_respect_bot_login_override_via_env(
        self, triage_module, monkeypatch
    ):
        # Operators wiring Agent Shin to a PAT (instead of GITHUB_TOKEN)
        # can override the expected bot login via env. The guard must
        # respect the override so non-default deployments still work.
        monkeypatch.setenv("AGENT_SHIN_BOT_LOGIN", "my-bot")
        self._stub_close_marker_present(triage_module, monkeypatch, present=True)
        self._stub_close_event(triage_module, monkeypatch, actor="my-bot")
        assert triage_module.was_closed_by_agent_shin("o/r", 1) is True
        # Default "github-actions[bot]" should NOT match when env is set.
        self._stub_close_event(triage_module, monkeypatch, actor="github-actions[bot]")
        assert triage_module.was_closed_by_agent_shin("o/r", 1) is False


class TestSecondsSinceLastAgentShinClose:
    """Close-provenance lookup: detects the bot's own auto-close marker."""

    def _make_comment(self, *, login: str, body: str) -> dict:
        return {
            "user": {"login": login},
            "body": body,
            "created_at": "2026-05-18T05:00:00Z",
        }

    def test_should_return_none_when_bot_never_closed(self, triage_module, monkeypatch):
        # Comments exist, but none is an Agent Shin close — e.g. only a grace
        # warning, or a close by another workflow with no Agent Shin comment.
        comments = [
            self._make_comment(login="outside-dev", body="any update?"),
            self._make_comment(
                login="github-actions[bot]",
                body=triage_module.format_grace_warning_pr_comment(
                    {"verdict": "fail", "missing": [], "explanation": ""}
                ),
            ),
        ]
        monkeypatch.setattr(
            triage_module, "_iter_paginated_json", lambda *a, **kw: iter(comments)
        )
        assert triage_module.seconds_since_last_agent_shin_close("o/r", 1) is None

    def test_should_detect_bot_close_comment(self, triage_module, monkeypatch):
        comments = [
            self._make_comment(
                login="github-actions[bot]",
                body=triage_module.format_pr_close_comment(
                    {"verdict": "fail", "missing": [], "explanation": ""}
                ),
            ),
        ]
        monkeypatch.setattr(
            triage_module, "_iter_paginated_json", lambda *a, **kw: iter(comments)
        )
        assert triage_module.seconds_since_last_agent_shin_close("o/r", 1) is not None

    def test_should_ignore_non_bot_comment_quoting_marker(
        self, triage_module, monkeypatch
    ):
        # A contributor quoting the hidden marker (GitHub "Quote reply"
        # preserves HTML comments) must not be mistaken for a bot close.
        comments = [
            self._make_comment(
                login="curious-user",
                body=f"what is this? {triage_module.AGENT_SHIN_CLOSE_MARKER}",
            ),
        ]
        monkeypatch.setattr(
            triage_module, "_iter_paginated_json", lambda *a, **kw: iter(comments)
        )
        assert triage_module.seconds_since_last_agent_shin_close("o/r", 1) is None


class TestSecondsSinceLastReconsiderVerdict:
    """Rate-limit guard: detects the bot's own reconsider verdict marker."""

    def _make_comment(
        self, *, login: str, body: str, created_at: str | None = "2026-05-18T05:00:00Z"
    ) -> dict:
        comment: dict = {"user": {"login": login}, "body": body}
        if created_at is not None:
            comment["created_at"] = created_at
        return comment

    def test_should_return_none_when_no_bot_reconsider_comments(
        self, triage_module, monkeypatch
    ):
        # An issue with chatter from other users but no bot reconsider
        # verdict must not be rate-limited.
        comments = [
            self._make_comment(login="outside-dev", body="ping?"),
            self._make_comment(
                login="github-actions[bot]", body="some other bot message"
            ),
        ]
        monkeypatch.setattr(
            triage_module, "_iter_paginated_json", lambda *a, **kw: iter(comments)
        )
        assert triage_module.seconds_since_last_reconsider_verdict("o/r", 1) is None

    def test_should_pick_latest_bot_reconsider_marker(self, triage_module, monkeypatch):
        # When multiple reconsider verdicts exist, return the AGE of the
        # most recent one. Using a frozen reference helps pin the math.
        comments = [
            self._make_comment(
                login="github-actions[bot]",
                body="old verdict " + triage_module.RECONSIDER_COMMENT_MARKER,
                created_at="2026-05-18T04:00:00Z",
            ),
            self._make_comment(
                login="github-actions[bot]",
                body="newer verdict " + triage_module.RECONSIDER_COMMENT_MARKER,
                created_at="2026-05-18T04:55:00Z",
            ),
        ]
        monkeypatch.setattr(
            triage_module, "_iter_paginated_json", lambda *a, **kw: iter(comments)
        )

        # Freeze "now" via a tiny shim on the module's `dt` import.
        import datetime as real_dt

        class FrozenDateTime(real_dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return real_dt.datetime(2026, 5, 18, 5, 0, 0, tzinfo=tz)

        frozen_module = type(triage_module.dt)("datetime")
        frozen_module.datetime = FrozenDateTime
        frozen_module.timezone = real_dt.timezone
        monkeypatch.setattr(triage_module, "dt", frozen_module)

        age = triage_module.seconds_since_last_reconsider_verdict("o/r", 1)
        # newer verdict is 5 minutes (300 seconds) before "now"
        assert age == 300.0

    def test_should_ignore_non_bot_comments_with_marker(
        self, triage_module, monkeypatch
    ):
        # A user comment that happens to quote the marker (e.g. in
        # a "what does this hidden marker do?" question) must NOT count.
        # The rate-limit guard only trusts comments authored by the bot.
        comments = [
            self._make_comment(
                login="curious-user",
                body=f"Saw this marker: {triage_module.RECONSIDER_COMMENT_MARKER}",
            ),
        ]
        monkeypatch.setattr(
            triage_module, "_iter_paginated_json", lambda *a, **kw: iter(comments)
        )
        assert triage_module.seconds_since_last_reconsider_verdict("o/r", 1) is None

    def test_should_ignore_bot_comments_without_marker(
        self, triage_module, monkeypatch
    ):
        # The bot posts other things too (Agent Shin close comments,
        # CI status, etc.) — only the reconsider-verdict marker should
        # arm the cooldown.
        comments = [
            self._make_comment(
                login="github-actions[bot]",
                body="Agent Shin closed this PR (no marker)",
            ),
        ]
        monkeypatch.setattr(
            triage_module, "_iter_paginated_json", lambda *a, **kw: iter(comments)
        )
        assert triage_module.seconds_since_last_reconsider_verdict("o/r", 1) is None


class TestParseVerdict:
    def test_should_parse_plain_json(self, triage_module):
        raw = '{"verdict": "pass", "missing": []}'
        assert triage_module.parse_verdict(raw)["verdict"] == "pass"

    def test_should_strip_markdown_fence(self, triage_module):
        raw = '```json\n{"verdict": "fail", "missing": ["foo"]}\n```'
        result = triage_module.parse_verdict(raw)
        assert result["verdict"] == "fail"
        assert result["missing"] == ["foo"]

    def test_should_extract_embedded_json_from_prose(self, triage_module):
        raw = 'Here you go: {"verdict": "pass", "missing": []}\nThanks.'
        assert triage_module.parse_verdict(raw)["verdict"] == "pass"

    def test_should_raise_for_unparseable_text(self, triage_module):
        with pytest.raises(ValueError):
            triage_module.parse_verdict("not even close to json")

    def test_should_raise_for_empty(self, triage_module):
        with pytest.raises(ValueError):
            triage_module.parse_verdict("")


class TestBuildPrompts:
    def test_should_include_pr_title_and_body(self, triage_module):
        prompt = triage_module.build_pr_prompt(
            title="Add foo", body="<!-- comment --> Real body"
        )
        assert "Add foo" in prompt
        assert "Real body" in prompt
        assert "comment" not in prompt  # HTML comments are stripped

    def test_should_show_empty_marker_for_empty_pr_body(self, triage_module):
        prompt = triage_module.build_pr_prompt(title="t", body="<!-- nothing -->")
        assert "(empty)" in prompt

    def test_should_include_issue_title_and_body(self, triage_module):
        prompt = triage_module.build_issue_prompt(title="Bug", body="repro here")
        assert "Bug" in prompt
        assert "repro here" in prompt

    def test_issue_bug_rubric_requires_end_to_end_evidence_and_drops_pass_bias(
        self, triage_module
    ):
        # The bug bar was tightened: a report needs the "before" half shown
        # end-to-end (video / screenshot / real command output), prose-only
        # repro steps no longer pass, and the old "bias toward PASS" leniency
        # is gone. If any of these regress, the judge silently goes soft on
        # undemonstrated bug reports again.
        prompt = triage_module.build_issue_prompt(title="t", body="x")
        normalized = " ".join(prompt.split())
        assert "Bias toward PASS when the issue has structure" not in normalized
        assert "END-TO-END EVIDENCE OF THE BUG" in normalized
        assert "Do not bias toward PASS" in normalized
        # The three accepted forms of the "before" demonstration must be named.
        assert "screen recording / video" in normalized
        assert "screenshot of the bug" in normalized
        assert "mocked or stubbed" in normalized
        # Prose-only steps are explicitly insufficient now.
        assert "steps to reproduce" in normalized

    def test_should_not_crash_when_pr_body_contains_curly_braces(self, triage_module):
        """User-supplied content with `{` / `}` must NOT be re-parsed by
        `str.format()`. `format` only scans the template literal for
        replacement fields; values being substituted in are inserted as
        plain strings, so a body like `{"foo": "bar"}` or `{unmatched`
        cannot blow up the script. Pinning this here so a future
        "improvement" to the templating doesn't reintroduce a crash on
        every PR that quotes JSON.
        """
        for body in (
            'Here is some JSON: {"foo": "bar", "n": 1}',
            "Half a brace { left dangling, and a stray }",
            "Format-spec-looking thing: {0}, {name:>10}, {!r}",
            "Nested {a: {b: c}} braces",
        ):
            pr_prompt = triage_module.build_pr_prompt(title="t", body=body)
            issue_prompt = triage_module.build_issue_prompt(title="t", body=body)
            assert body in pr_prompt
            assert body in issue_prompt

    def test_should_not_crash_when_pr_title_contains_curly_braces(self, triage_module):
        title = "Fix bug in {0:>10} format-spec handling"
        pr_prompt = triage_module.build_pr_prompt(title=title, body="x")
        issue_prompt = triage_module.build_issue_prompt(title=title, body="x")
        assert title in pr_prompt
        assert title in issue_prompt

    def test_should_preserve_template_indentation_with_multiline_body(
        self, triage_module
    ):
        """`textwrap.dedent` runs on the static template *before* user
        content is interpolated, so a multi-line body (whose 2nd+ lines
        start at column 0) cannot defeat the common-indent computation
        and leave 8-space indentation on every template line. Pin the
        dedented shape so the rendered prompt stays consistent for the
        LLM judge.
        """
        body = "first line\nsecond line at column 0\nthird line at column 0"
        for builder in (
            triage_module.build_pr_prompt,
            triage_module.build_issue_prompt,
        ):
            prompt = builder(title="t", body=body)
            # Template lines should NOT carry the 8 leading spaces from
            # the source-file indentation of the triple-quoted string.
            assert "        You are " not in prompt
            assert 'You are "Agent Shin"' in prompt
            assert body in prompt


class TestMainModelDefault:
    """`--model` falls back to DEFAULT_MODEL even when TRIAGE_MODEL is empty."""

    def _stub_triage(self, triage_module, monkeypatch):
        captured: dict = {}

        def fake_triage(**kwargs):
            captured.update(kwargs)
            return {
                "kind": kwargs["kind"],
                "number": kwargs["number"],
                "title": "",
                "author": "x",
                "author_association": "NONE",
                "state": "open",
                "action": "skip-no-llm-key",
            }

        monkeypatch.setattr(triage_module, "triage", fake_triage)
        return captured

    def test_should_fall_back_to_default_when_triage_model_env_empty(
        self, triage_module, monkeypatch
    ):
        captured = self._stub_triage(triage_module, monkeypatch)
        monkeypatch.setenv("TRIAGE_MODEL", "")
        monkeypatch.setattr(
            sys,
            "argv",
            ["triage_with_llm.py", "--repo", "o/r", "--pr", "1"],
        )
        rc = triage_module.main()
        assert rc == 0
        assert captured["model"] == triage_module.DEFAULT_MODEL

    def test_should_respect_explicit_triage_model_env(self, triage_module, monkeypatch):
        captured = self._stub_triage(triage_module, monkeypatch)
        monkeypatch.setenv("TRIAGE_MODEL", "gpt-4o-mini")
        monkeypatch.setattr(
            sys,
            "argv",
            ["triage_with_llm.py", "--repo", "o/r", "--pr", "1"],
        )
        rc = triage_module.main()
        assert rc == 0
        assert captured["model"] == "gpt-4o-mini"


class TestCallLlmJudge:
    """call_llm_judge sets gpt-5 specific kwargs correctly."""

    def _stub_openai(self, monkeypatch, captured: dict):
        """Install a fake `openai.OpenAI` client into sys.modules.

        The fake client records the kwargs passed to chat.completions.create
        and returns a minimal response object whose .choices[0].message.content
        is "ok".
        """
        import types

        class FakeMessage:
            content = '{"verdict": "pass"}'

        class FakeChoice:
            message = FakeMessage()

        class FakeResponse:
            choices = [FakeChoice()]

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return FakeResponse()

        class FakeChat:
            completions = FakeCompletions()

        class FakeClient:
            def __init__(self, api_key, base_url=None):
                captured["__client_kwargs__"] = {
                    "api_key": api_key,
                    "base_url": base_url,
                }
                self.chat = FakeChat()

        fake_module = types.ModuleType("openai")
        fake_module.OpenAI = FakeClient
        monkeypatch.setitem(sys.modules, "openai", fake_module)

    def test_should_set_reasoning_effort_none_for_gpt5_family(
        self, triage_module, monkeypatch
    ):
        captured: dict = {}
        self._stub_openai(monkeypatch, captured)
        triage_module.call_llm_judge(
            "prompt", model="gpt-5.4-mini", api_key="sk-test", base_url=None
        )
        assert captured["model"] == "gpt-5.4-mini"
        assert captured["temperature"] == 0
        assert captured["extra_body"] == {"reasoning_effort": "none"}

    def test_should_set_reasoning_effort_for_capitalized_or_dated_gpt5(
        self, triage_module, monkeypatch
    ):
        for model in ("GPT-5.4-mini", "gpt-5.4-mini-2026-03-17", "gpt-5"):
            captured: dict = {}
            self._stub_openai(monkeypatch, captured)
            triage_module.call_llm_judge(
                "prompt", model=model, api_key="sk-test", base_url=None
            )
            assert captured["extra_body"] == {"reasoning_effort": "none"}, model

    def test_should_omit_reasoning_effort_for_non_gpt5(
        self, triage_module, monkeypatch
    ):
        captured: dict = {}
        self._stub_openai(monkeypatch, captured)
        triage_module.call_llm_judge(
            "prompt", model="gpt-4o-mini", api_key="sk-test", base_url=None
        )
        assert "extra_body" not in captured

    def test_should_pass_base_url_when_provided(self, triage_module, monkeypatch):
        captured: dict = {}
        self._stub_openai(monkeypatch, captured)
        triage_module.call_llm_judge(
            "p",
            model="gpt-5.4-mini",
            api_key="sk-test",
            base_url="https://proxy.example.com/v1",
        )
        assert (
            captured["__client_kwargs__"]["base_url"] == "https://proxy.example.com/v1"
        )


class TestTriageOrchestration:
    """End-to-end-ish tests that mock both gh fetchers and the LLM."""

    def _make_pr(self, **overrides):
        base = {
            "number": 1,
            "title": "PR title",
            "body": "PR body",
            "state": "open",
            "author_association": "NONE",
            "user": {"login": "mateo-berri"},
        }
        base.update(overrides)
        return base

    def test_should_skip_internal_author(self, triage_module, monkeypatch):
        pr = self._make_pr(
            author_association="MEMBER", user={"login": "krrishdholakia"}
        )
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)

        def boom(*a, **kw):
            pytest.fail("LLM should not be called for internal authors")

        result = triage_module.triage(
            repo="o/r",
            kind="pr",
            number=1,
            close=True,
            model="m",
            judge=boom,
            allowlist=frozenset(),
        )
        assert result["action"] == "skip-internal-author"

    def test_should_skip_closed_pr(self, triage_module, monkeypatch):
        pr = self._make_pr(state="closed")
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        result = triage_module.triage(
            repo="o/r",
            kind="pr",
            number=1,
            close=True,
            model="m",
            judge=lambda p: pytest.fail("should not run on closed PRs"),
        )
        assert result["action"] == "skip-not-open"

    def test_should_short_circuit_on_linked_issue(self, triage_module, monkeypatch):
        pr = self._make_pr(body="Fixes #1234\n\nFoo bar")
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        result = triage_module.triage(
            repo="o/r",
            kind="pr",
            number=1,
            close=True,
            model="m",
            judge=lambda p: pytest.fail("LLM should not be called"),
        )
        assert result["action"] == "pass-linked-issue"
        assert result["verdict"]["verdict"] == "pass"

    def test_should_not_short_circuit_on_casual_mention(
        self, triage_module, monkeypatch
    ):
        # "See #1234" is a passing mention, not a closing keyword. The LLM
        # must get a chance to apply the stricter rubric. With no prior
        # grace warning, the first failing verdict triggers the warning
        # path (`would-warn-grace` in dry-run).
        pr = self._make_pr(body="See #1234 for context. No QA proof here.")
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        self._stub_grace_no_warning(triage_module, monkeypatch)
        called = {"judge": False}

        def judge(prompt):
            called["judge"] = True
            return json.dumps(
                {"verdict": "fail", "missing": ["QA proof"], "explanation": "thin."}
            )

        result = triage_module.triage(
            repo="o/r",
            kind="pr",
            number=1,
            close=False,
            model="m",
            judge=judge,
        )
        assert called["judge"] is True
        assert result["action"] == "would-warn-grace"

    def test_should_return_pass_llm_when_judge_passes(self, triage_module, monkeypatch):
        pr = self._make_pr(body="Long body, no linked issue.")
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        captured = {}

        def judge(prompt):
            captured["prompt"] = prompt
            return json.dumps({"verdict": "pass", "missing": [], "explanation": "ok"})

        result = triage_module.triage(
            repo="o/r", kind="pr", number=1, close=True, model="m", judge=judge
        )
        assert result["action"] == "pass-llm"
        assert "Long body" in captured["prompt"]

    def test_should_return_would_close_in_dry_run_after_grace_aged_out(
        self, triage_module, monkeypatch
    ):
        # When the grace warning has already aged out (>= GRACE_PERIOD_SECONDS)
        # AND the rubric still fails, the dry-run preview returns
        # `would-close` so a step-summary writer can render the close
        # comment without touching GitHub state.
        pr = self._make_pr(body="just a sentence.")
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        self._stub_grace_aged_out(triage_module, monkeypatch)

        def fake_post(*a, **kw):
            pytest.fail("should not post comments in dry-run")

        def fake_close(*a, **kw):
            pytest.fail("should not close in dry-run")

        monkeypatch.setattr(triage_module, "post_comment", fake_post)
        monkeypatch.setattr(triage_module, "close_pr", fake_close)

        verdict = {
            "verdict": "fail",
            "missing": ["problem description", "QA proof"],
            "explanation": "Body is one sentence.",
        }
        result = triage_module.triage(
            repo="o/r",
            kind="pr",
            number=1,
            close=False,
            model="m",
            judge=lambda p: json.dumps(verdict),
        )
        assert result["action"] == "would-close"
        assert result["verdict"]["missing"] == ["problem description", "QA proof"]

    def test_should_post_comment_and_close_after_grace_window(
        self, triage_module, monkeypatch
    ):
        # The "real close" path: --close passed AND the grace warning has
        # aged out AND the rubric still fails. The bot posts the close
        # comment and closes the PR.
        pr = self._make_pr(body="just a sentence.")
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        self._stub_grace_aged_out(triage_module, monkeypatch)
        posted = {}
        closed = {}
        monkeypatch.setattr(
            triage_module,
            "post_comment",
            lambda repo, n, body: posted.update({"repo": repo, "n": n, "body": body}),
        )
        monkeypatch.setattr(
            triage_module,
            "close_pr",
            lambda repo, n: closed.update({"repo": repo, "n": n}),
        )

        verdict = {
            "verdict": "fail",
            "missing": ["QA proof"],
            "explanation": "Body too thin.",
        }
        result = triage_module.triage(
            repo="o/r",
            kind="pr",
            number=42,
            close=True,
            model="m",
            judge=lambda p: json.dumps(verdict),
        )
        assert result["action"] == "closed"
        assert posted["n"] == 42 and closed["n"] == 42
        assert "Agent Shin" in posted["body"]
        assert "QA proof" in posted["body"]

    def test_should_skip_on_llm_error_in_close_mode(self, triage_module, monkeypatch):
        pr = self._make_pr(body="something.")
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        monkeypatch.setattr(
            triage_module,
            "post_comment",
            lambda *a, **kw: pytest.fail("must not comment on LLM error"),
        )
        monkeypatch.setattr(
            triage_module,
            "close_pr",
            lambda *a, **kw: pytest.fail("must not close on LLM error"),
        )

        def broken_judge(prompt):
            raise RuntimeError("upstream 500")

        result = triage_module.triage(
            repo="o/r",
            kind="pr",
            number=1,
            close=True,
            model="m",
            judge=broken_judge,
        )
        assert result["action"] == "skip-llm-error"
        assert "upstream 500" in result["error"]

    def test_should_skip_open_pr_in_reconsider_mode(self, triage_module, monkeypatch):
        # Reconsider only makes sense on a CLOSED PR — running it on an open
        # one is a no-op (the regular triage flow already evaluated it).
        pr = self._make_pr(state="open")
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        result = triage_module.triage(
            repo="o/r",
            kind="pr",
            number=1,
            close=False,
            model="m",
            judge=lambda p: pytest.fail("should not run on open PR in reconsider"),
            reconsider=True,
        )
        assert result["action"] == "skip-not-closed"

    @staticmethod
    def _stub_reconsider_guards(triage_module, monkeypatch):
        """Default reconsider-guard stubs: pretend bot closed + no cooldown.

        The new safety guards (`was_closed_by_agent_shin`,
        `seconds_since_last_reconsider_verdict`) hit the GitHub API in
        production. Tests that exercise the reconsider happy path stub
        them to "yes the bot closed it, no recent reconsider comment"
        so the test stays focused on its actual assertion.
        """
        monkeypatch.setattr(
            triage_module, "was_closed_by_agent_shin", lambda *a, **kw: True
        )
        monkeypatch.setattr(
            triage_module,
            "seconds_since_last_reconsider_verdict",
            lambda *a, **kw: None,
        )

    @staticmethod
    def _stub_grace_aged_out(triage_module, monkeypatch):
        """Pretend the grace warning has aged out.

        For tests that exercise the post-grace close path. Set the age
        to twice the grace window so a future tweak to
        `GRACE_PERIOD_SECONDS` doesn't accidentally make the stub fall
        back inside the window.
        """
        monkeypatch.setattr(
            triage_module,
            "seconds_since_last_grace_warning",
            lambda *a, **kw: triage_module.GRACE_PERIOD_SECONDS * 2,
        )

    @staticmethod
    def _stub_grace_no_warning(triage_module, monkeypatch):
        """Pretend no grace warning has been posted yet (first detection)."""
        monkeypatch.setattr(
            triage_module,
            "seconds_since_last_grace_warning",
            lambda *a, **kw: None,
        )

    def test_should_reopen_on_reconsider_pass(self, triage_module, monkeypatch):
        # Reconsider on a closed PR with a passing verdict -> reopen + post a
        # friendly "re-evaluated" comment. close=True is the production path
        # (the workflow only adds --close when AGENT_SHIN_ENABLED=true).
        pr = self._make_pr(
            state="closed", body="Updated body with QA proof + screenshots."
        )
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        self._stub_reconsider_guards(triage_module, monkeypatch)
        posted = {}
        reopened = {}
        monkeypatch.setattr(
            triage_module,
            "post_comment",
            lambda repo, n, body: posted.update({"n": n, "body": body}),
        )
        monkeypatch.setattr(
            triage_module,
            "reopen_pr",
            lambda repo, n: reopened.update({"n": n}),
        )
        # close_pr / close_issue MUST NOT fire in reconsider mode.
        monkeypatch.setattr(
            triage_module,
            "close_pr",
            lambda *a, **kw: pytest.fail("must not close on reconsider pass"),
        )

        result = triage_module.triage(
            repo="o/r",
            kind="pr",
            number=42,
            close=True,
            model="m",
            judge=lambda p: json.dumps(
                {"verdict": "pass", "missing": [], "explanation": "ok now"}
            ),
            reconsider=True,
        )
        assert result["action"] == "reopened"
        assert reopened["n"] == 42
        assert posted["n"] == 42
        assert "reopened" in posted["body"].lower()

    def test_should_dry_run_reconsider_pass_when_close_false(
        self, triage_module, monkeypatch
    ):
        # Reconsider must honor `close=False` (dry-run) just like the
        # regular triage flow. A local invocation of
        # `python triage_with_llm.py --reconsider --pr N` (no --close)
        # must NOT post a comment or reopen the PR — it should return
        # `would-reopen` so the operator can preview the outcome.
        pr = self._make_pr(
            state="closed", body="Updated body with QA proof + screenshots."
        )
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        self._stub_reconsider_guards(triage_module, monkeypatch)
        monkeypatch.setattr(
            triage_module,
            "post_comment",
            lambda *a, **kw: pytest.fail("must not post comment in dry-run reconsider"),
        )
        monkeypatch.setattr(
            triage_module,
            "reopen_pr",
            lambda *a, **kw: pytest.fail("must not reopen PR in dry-run reconsider"),
        )

        result = triage_module.triage(
            repo="o/r",
            kind="pr",
            number=42,
            close=False,
            model="m",
            judge=lambda p: json.dumps(
                {"verdict": "pass", "missing": [], "explanation": "ok now"}
            ),
            reconsider=True,
        )
        assert result["action"] == "would-reopen"
        # The previewed comment body is still returned so a step-summary
        # writer can render exactly what would have been posted.
        assert "reopened" in result["comment"].lower()

    def test_should_post_still_failing_on_reconsider_fail(
        self, triage_module, monkeypatch
    ):
        pr = self._make_pr(state="closed", body="still empty")
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        self._stub_reconsider_guards(triage_module, monkeypatch)
        posted = {}
        monkeypatch.setattr(
            triage_module,
            "post_comment",
            lambda repo, n, body: posted.update({"n": n, "body": body}),
        )
        # Neither reopen nor close should fire when reconsider verdict is fail.
        monkeypatch.setattr(
            triage_module,
            "reopen_pr",
            lambda *a, **kw: pytest.fail("must not reopen on fail"),
        )
        monkeypatch.setattr(
            triage_module,
            "close_pr",
            lambda *a, **kw: pytest.fail("must not close again on reconsider fail"),
        )

        verdict = {
            "verdict": "fail",
            "missing": ["QA proof"],
            "explanation": "Still no QA proof.",
        }
        result = triage_module.triage(
            repo="o/r",
            kind="pr",
            number=42,
            close=True,
            model="m",
            judge=lambda p: json.dumps(verdict),
            reconsider=True,
        )
        assert result["action"] == "reconsider-still-failing"
        assert posted["n"] == 42
        assert "QA proof" in posted["body"]

    def test_should_not_reopen_on_reconsider_with_ambiguous_verdict(
        self, triage_module, monkeypatch
    ):
        # Regression: only an explicit `pass` verdict reopens. Missing,
        # empty, or unexpected verdict strings ("failed", "", garbage)
        # must fall through to the still-failing branch rather than
        # reopen a PR the rubric did not actually clear.
        pr = self._make_pr(state="closed", body="still empty")
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        self._stub_reconsider_guards(triage_module, monkeypatch)
        posted = {}
        monkeypatch.setattr(
            triage_module,
            "post_comment",
            lambda repo, n, body: posted.update({"body": body}),
        )
        monkeypatch.setattr(
            triage_module,
            "reopen_pr",
            lambda *a, **kw: pytest.fail("must not reopen on ambiguous verdict"),
        )

        for ambiguous in ("", "failed", "needs-info", "unknown"):
            posted.clear()
            result = triage_module.triage(
                repo="o/r",
                kind="pr",
                number=42,
                close=True,
                model="m",
                judge=lambda p, v=ambiguous: json.dumps(
                    {"verdict": v, "missing": [], "explanation": "weird"}
                ),
                reconsider=True,
            )
            assert result["action"] == "reconsider-still-failing", ambiguous
            assert "body" in posted, ambiguous

    def test_should_dry_run_reconsider_fail_when_close_false(
        self, triage_module, monkeypatch
    ):
        # Mirror dry-run behavior for the FAIL branch — `close=False`
        # must NOT post the "still failing" comment.
        pr = self._make_pr(state="closed", body="still empty")
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        self._stub_reconsider_guards(triage_module, monkeypatch)
        monkeypatch.setattr(
            triage_module,
            "post_comment",
            lambda *a, **kw: pytest.fail(
                "must not post still-failing comment in dry-run"
            ),
        )

        verdict = {
            "verdict": "fail",
            "missing": ["QA proof"],
            "explanation": "Still no QA proof.",
        }
        result = triage_module.triage(
            repo="o/r",
            kind="pr",
            number=42,
            close=False,
            model="m",
            judge=lambda p: json.dumps(verdict),
            reconsider=True,
        )
        assert result["action"] == "would-reconsider-still-failing"
        assert "QA proof" in result["comment"]

    def test_should_reopen_on_reconsider_with_linked_issue_short_circuit(
        self, triage_module, monkeypatch
    ):
        # The linked-issue short-circuit also has to honor reconsider mode:
        # if the contributor edited the body to add `Fixes #1234`, the regex
        # path should reopen the PR without calling the LLM.
        pr = self._make_pr(state="closed", body="Fixes #1234\n\nAddresses the bug.")
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        self._stub_reconsider_guards(triage_module, monkeypatch)
        posted = {}
        reopened = {}
        monkeypatch.setattr(
            triage_module,
            "post_comment",
            lambda repo, n, body: posted.update({"body": body}),
        )
        monkeypatch.setattr(
            triage_module,
            "reopen_pr",
            lambda repo, n: reopened.update({"n": n}),
        )

        result = triage_module.triage(
            repo="o/r",
            kind="pr",
            number=55,
            close=True,
            model="m",
            judge=lambda p: pytest.fail("LLM must not run when linked-issue matches"),
            reconsider=True,
        )
        assert result["action"] == "reopened"
        assert reopened["n"] == 55
        assert "reopened" in posted["body"].lower()

    def test_should_dry_run_reconsider_with_linked_issue_when_close_false(
        self, triage_module, monkeypatch
    ):
        # Linked-issue short-circuit must ALSO honor dry-run.
        pr = self._make_pr(state="closed", body="Fixes #1234\n\nAddresses the bug.")
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        self._stub_reconsider_guards(triage_module, monkeypatch)
        monkeypatch.setattr(
            triage_module,
            "post_comment",
            lambda *a, **kw: pytest.fail("must not post in dry-run"),
        )
        monkeypatch.setattr(
            triage_module,
            "reopen_pr",
            lambda *a, **kw: pytest.fail("must not reopen in dry-run"),
        )

        result = triage_module.triage(
            repo="o/r",
            kind="pr",
            number=55,
            close=False,
            model="m",
            judge=lambda p: pytest.fail("LLM must not run when linked-issue matches"),
            reconsider=True,
        )
        assert result["action"] == "would-reopen"

    def test_should_skip_internal_in_reconsider_mode(self, triage_module, monkeypatch):
        # Internal authors are exempt from triage in both regular and
        # reconsider mode — Agent Shin should never reopen one of their PRs
        # automatically, in case a maintainer closed it intentionally.
        pr = self._make_pr(
            state="closed",
            author_association="MEMBER",
            user={"login": "krrishdholakia"},
        )
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        monkeypatch.setattr(
            triage_module,
            "reopen_pr",
            lambda *a, **kw: pytest.fail("must not reopen for internal author"),
        )
        result = triage_module.triage(
            repo="o/r",
            kind="pr",
            number=1,
            close=False,
            model="m",
            judge=lambda p: pytest.fail("LLM must not run for internal author"),
            reconsider=True,
            allowlist=frozenset(),
        )
        assert result["action"] == "skip-internal-author"

    def test_should_skip_reconsider_when_not_bot_closed(
        self, triage_module, monkeypatch
    ):
        # SECURITY: `@agent-shin reconsider` must NOT reopen a PR/issue
        # that a MAINTAINER closed for non-rubric reasons (e.g. duplicate,
        # design rejection, security report). Only PRs closed by the bot
        # itself should ever be candidates for the reconsider reopen path.
        pr = self._make_pr(state="closed", body="something.")
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        monkeypatch.setattr(
            triage_module, "was_closed_by_agent_shin", lambda *a, **kw: False
        )
        # Even though there's no rate-limit conflict, the bot-closed guard
        # alone is sufficient to block. The LLM judge must never run on a
        # maintainer-closed PR.
        monkeypatch.setattr(
            triage_module,
            "seconds_since_last_reconsider_verdict",
            lambda *a, **kw: None,
        )
        monkeypatch.setattr(
            triage_module,
            "post_comment",
            lambda *a, **kw: pytest.fail("must not comment on maintainer-closed PR"),
        )
        monkeypatch.setattr(
            triage_module,
            "reopen_pr",
            lambda *a, **kw: pytest.fail("must not reopen maintainer-closed PR"),
        )

        result = triage_module.triage(
            repo="o/r",
            kind="pr",
            number=1,
            close=True,
            model="m",
            judge=lambda p: pytest.fail("LLM must not run before bot-closed guard"),
            reconsider=True,
        )
        assert result["action"] == "skip-not-bot-closed"

    def test_should_rate_limit_repeated_reconsider_triggers(
        self, triage_module, monkeypatch
    ):
        # COST CONTROL: each `@agent-shin reconsider` event burns CI
        # minutes + an OpenAI API call. If the bot already posted a
        # reconsider verdict within the cooldown window
        # (RECONSIDER_RATE_LIMIT_SECONDS), refuse to run again. This
        # bounds the damage from a contributor spamming the trigger.
        pr = self._make_pr(state="closed", body="something with new edits.")
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        monkeypatch.setattr(
            triage_module, "was_closed_by_agent_shin", lambda *a, **kw: True
        )
        # Pretend the bot posted a reconsider verdict 1 second ago.
        monkeypatch.setattr(
            triage_module,
            "seconds_since_last_reconsider_verdict",
            lambda *a, **kw: 1.0,
        )
        monkeypatch.setattr(
            triage_module,
            "post_comment",
            lambda *a, **kw: pytest.fail("must not comment during cooldown"),
        )
        monkeypatch.setattr(
            triage_module,
            "reopen_pr",
            lambda *a, **kw: pytest.fail("must not reopen during cooldown"),
        )

        result = triage_module.triage(
            repo="o/r",
            kind="pr",
            number=1,
            close=True,
            model="m",
            judge=lambda p: pytest.fail("LLM must not run during cooldown"),
            reconsider=True,
        )
        assert result["action"] == "skip-rate-limited"
        assert result["rate_limit_age_seconds"] == 1.0
        assert (
            result["rate_limit_window_seconds"]
            == triage_module.RECONSIDER_RATE_LIMIT_SECONDS
        )

    def test_should_allow_reconsider_after_cooldown_window(
        self, triage_module, monkeypatch
    ):
        # The cooldown is a window, not a one-shot lock — once
        # RECONSIDER_RATE_LIMIT_SECONDS has elapsed since the last bot
        # verdict, a fresh `@agent-shin reconsider` is allowed through.
        pr = self._make_pr(state="closed", body="updated with screenshots now.")
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        monkeypatch.setattr(
            triage_module, "was_closed_by_agent_shin", lambda *a, **kw: True
        )
        # Last reconsider was 1 hour ago — well outside the 10-min window.
        monkeypatch.setattr(
            triage_module,
            "seconds_since_last_reconsider_verdict",
            lambda *a, **kw: 3600.0,
        )
        posted = {}
        reopened = {}
        monkeypatch.setattr(
            triage_module,
            "post_comment",
            lambda repo, n, body: posted.update({"n": n, "body": body}),
        )
        monkeypatch.setattr(
            triage_module,
            "reopen_pr",
            lambda repo, n: reopened.update({"n": n}),
        )

        result = triage_module.triage(
            repo="o/r",
            kind="pr",
            number=1,
            close=True,
            model="m",
            judge=lambda p: json.dumps(
                {"verdict": "pass", "missing": [], "explanation": "ok"}
            ),
            reconsider=True,
        )
        assert result["action"] == "reopened"
        assert reopened["n"] == 1

    def test_should_reopen_issue_on_reconsider_pass(self, triage_module, monkeypatch):
        issue = {
            "number": 7,
            "title": "Bug: now with repro",
            "body": "## Repro\n```bash\ncurl ...\n```\n\nExpected X, got Y.",
            "state": "closed",
            "author_association": "NONE",
            "user": {"login": "mateo-berri"},
        }
        monkeypatch.setattr(triage_module, "fetch_issue", lambda repo, n: issue)
        self._stub_reconsider_guards(triage_module, monkeypatch)
        posted = {}
        reopened = {}
        monkeypatch.setattr(
            triage_module,
            "post_comment",
            lambda repo, n, body: posted.update({"body": body}),
        )
        monkeypatch.setattr(
            triage_module,
            "reopen_issue",
            lambda repo, n: reopened.update({"n": n}),
        )

        result = triage_module.triage(
            repo="o/r",
            kind="issue",
            number=7,
            close=True,
            model="m",
            judge=lambda p: json.dumps(
                {"verdict": "pass", "missing": [], "explanation": "now reproducible"}
            ),
            reconsider=True,
        )
        assert result["action"] == "reopened"
        assert reopened["n"] == 7
        assert "reopened" in posted["body"].lower()

    def test_should_triage_issues_kind(self, triage_module, monkeypatch):
        issue = {
            "number": 7,
            "title": "Bug: X is broken",
            "body": "no detail",
            "state": "open",
            "author_association": "NONE",
            "user": {"login": "mateo-berri"},
        }
        monkeypatch.setattr(triage_module, "fetch_issue", lambda repo, n: issue)
        # Grace already aged out -> close path. (Issues use the same
        # GRACE_COMMENT_MARKER detection as PRs.)
        self._stub_grace_aged_out(triage_module, monkeypatch)
        closed = {}
        posted = {}
        monkeypatch.setattr(
            triage_module,
            "post_comment",
            lambda repo, n, body: posted.update(body=body),
        )
        monkeypatch.setattr(
            triage_module, "close_issue", lambda repo, n: closed.update(n=n)
        )

        verdict = {
            "verdict": "fail",
            "kind": "bug",
            "has_repro": False,
            "missing": ["reproduction", "expected vs. actual"],
            "explanation": "No repro provided.",
        }
        result = triage_module.triage(
            repo="o/r",
            kind="issue",
            number=7,
            close=True,
            model="m",
            judge=lambda p: json.dumps(verdict),
        )
        assert result["action"] == "closed"
        assert closed["n"] == 7
        assert "reproduction" in posted["body"]

    # ---- Grace-period flow ------------------------------------------------

    def test_should_post_grace_warning_on_first_failing_run_in_close_mode(
        self, triage_module, monkeypatch
    ):
        # First low-quality detection -> bot posts a warning comment with
        # the GRACE_COMMENT_MARKER. The PR must NOT be closed yet.
        pr = self._make_pr(body="just a sentence.")
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        self._stub_grace_no_warning(triage_module, monkeypatch)
        posted = {}
        monkeypatch.setattr(
            triage_module,
            "post_comment",
            lambda repo, n, body: posted.update({"n": n, "body": body}),
        )
        monkeypatch.setattr(
            triage_module,
            "close_pr",
            lambda *a, **kw: pytest.fail("must not close on first detection"),
        )

        verdict = {
            "verdict": "fail",
            "missing": ["QA proof"],
            "explanation": "Body too thin.",
        }
        result = triage_module.triage(
            repo="o/r",
            kind="pr",
            number=42,
            close=True,
            model="m",
            judge=lambda p: json.dumps(verdict),
        )
        assert result["action"] == "warned-grace"
        assert posted["n"] == 42
        # Pin the user-facing language pieces the user explicitly asked for.
        assert "2 hours" in posted["body"]
        assert "@agent-shin reconsider" in posted["body"]
        assert "@greptileai" in posted["body"]
        assert "even after the PR is closed" in posted["body"]
        assert triage_module.GRACE_COMMENT_MARKER in posted["body"]

    def test_should_skip_close_inside_grace_window(self, triage_module, monkeypatch):
        # A warning was posted recently; do nothing on this run regardless
        # of close=True. The next run after `GRACE_PERIOD_SECONDS` elapses
        # is the one that flips to actual close.
        pr = self._make_pr(body="just a sentence.")
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        monkeypatch.setattr(
            triage_module,
            "seconds_since_last_grace_warning",
            lambda *a, **kw: 60.0,
        )
        monkeypatch.setattr(
            triage_module,
            "post_comment",
            lambda *a, **kw: pytest.fail("must not comment during grace window"),
        )
        monkeypatch.setattr(
            triage_module,
            "close_pr",
            lambda *a, **kw: pytest.fail("must not close during grace window"),
        )

        verdict = {
            "verdict": "fail",
            "missing": ["QA proof"],
            "explanation": "Body too thin.",
        }
        result = triage_module.triage(
            repo="o/r",
            kind="pr",
            number=42,
            close=True,
            model="m",
            judge=lambda p: json.dumps(verdict),
        )
        assert result["action"] == "skip-in-grace-period"
        assert result["grace_age_seconds"] == 60.0
        assert result["grace_period_seconds"] == triage_module.GRACE_PERIOD_SECONDS

    def test_should_dry_run_grace_warning_when_close_false(
        self, triage_module, monkeypatch
    ):
        # In dry-run mode the FIRST failing detection returns
        # `would-warn-grace` (with the previewed comment body) and never
        # touches GitHub state. Lets a local operator preview the
        # warning before flipping --close on.
        pr = self._make_pr(body="thin")
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        self._stub_grace_no_warning(triage_module, monkeypatch)
        monkeypatch.setattr(
            triage_module,
            "post_comment",
            lambda *a, **kw: pytest.fail("must not post in dry-run grace warn"),
        )

        verdict = {
            "verdict": "fail",
            "missing": ["QA proof"],
            "explanation": "thin",
        }
        result = triage_module.triage(
            repo="o/r",
            kind="pr",
            number=1,
            close=False,
            model="m",
            judge=lambda p: json.dumps(verdict),
        )
        assert result["action"] == "would-warn-grace"
        assert "2 hours" in result["comment"]

    def test_should_warn_grace_for_swiftwinds_not_close_instantly(
        self, triage_module, monkeypatch
    ):
        # Regression: SwiftWinds (the dogfood account) used to be in a
        # now-removed `IMMEDIATE_CLOSE_LOGINS` bypass that skipped the grace
        # window and closed on first detection. It must follow the SAME
        # grace path as every other author: warn first, close only after the
        # window elapses. A re-added instant-close bypass would call
        # close_pr here and fail the test.
        pr = self._make_pr(body="just a sentence.", user={"login": "SwiftWinds"})
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        self._stub_grace_no_warning(triage_module, monkeypatch)
        posted = {}
        monkeypatch.setattr(
            triage_module,
            "post_comment",
            lambda repo, n, body: posted.update({"n": n, "body": body}),
        )
        monkeypatch.setattr(
            triage_module,
            "close_pr",
            lambda *a, **kw: pytest.fail(
                "SwiftWinds must not close on first detection; it gets the grace window"
            ),
        )

        verdict = {
            "verdict": "fail",
            "missing": ["QA proof"],
            "explanation": "Body too thin.",
        }
        result = triage_module.triage(
            repo="o/r",
            kind="pr",
            number=99,
            close=True,
            model="m",
            judge=lambda p: json.dumps(verdict),
        )
        assert result["action"] == "warned-grace"
        assert "2 hours" in posted["body"]


class TestGraceWarningCommentText:
    """Pin the user-facing promises in the grace warning so a future
    refactor can't silently drop them."""

    def test_pr_grace_warning_should_state_grace_window(self, triage_module):
        body = triage_module.format_grace_warning_pr_comment(
            {"verdict": "fail", "missing": ["QA proof"], "explanation": "thin"}
        )
        # The user explicitly asked: "specify in the comment" the grace window.
        assert "2 hours" in body

    def test_pr_grace_warning_should_mention_reconsider_during_grace(
        self, triage_module
    ):
        body = triage_module.format_grace_warning_pr_comment(
            {"verdict": "fail", "missing": [], "explanation": ""}
        )
        assert "@agent-shin reconsider" in body

    def test_pr_grace_warning_should_promise_greptileai_works_post_close(
        self, triage_module
    ):
        body = triage_module.format_grace_warning_pr_comment(
            {"verdict": "fail", "missing": [], "explanation": ""}
        )
        # Per user: comment should state @greptileai works even after close.
        assert "@greptileai" in body
        assert "even after the PR is closed" in body

    def test_pr_grace_warning_should_carry_grace_marker(self, triage_module):
        # The marker is what `seconds_since_last_grace_warning` greps for
        # on subsequent runs to detect that a warning has been posted.
        # Dropping it would silently break the close-after-grace path.
        body = triage_module.format_grace_warning_pr_comment(
            {"verdict": "fail", "missing": [], "explanation": ""}
        )
        assert triage_module.GRACE_COMMENT_MARKER in body

    def test_issue_grace_warning_should_carry_grace_marker(self, triage_module):
        body = triage_module.format_grace_warning_issue_comment(
            {"verdict": "fail", "missing": [], "explanation": ""}
        )
        assert triage_module.GRACE_COMMENT_MARKER in body
        assert "2 hours" in body
        # OSS authors can't reopen a bot-closed issue, so recovery is
        # `@agent-shin reconsider` (the bot reopens), like the PR path.
        assert "@agent-shin reconsider" in body

    def test_pr_close_comment_should_promise_greptileai_works_post_close(
        self, triage_module
    ):
        # The standard close comment must ALSO point at @greptileai so
        # contributors see the same options whether they read the warning
        # or only catch the close comment.
        body = triage_module.format_pr_close_comment(
            {"verdict": "fail", "missing": [], "explanation": ""}
        )
        assert "@greptileai" in body
        assert "even after the PR is closed" in body

    def test_pr_grace_warning_should_not_prompt_reconsider_during_grace_window(
        self, triage_module
    ):
        # Per user feedback: during the 24h grace window, the contributor
        # should just update the PR description. Asking them to also comment
        # "@agent-shin reconsider" right away adds a step they don't need —
        # the bot re-checks automatically on the next sweep. The reconsider
        # trigger is reserved for the post-close recovery path.
        #
        # We pin this by checking that the grace section explicitly tells
        # the contributor they don't need to ping the bot during the grace
        # window. The presence of "@agent-shin reconsider" elsewhere in the
        # comment (as the post-close path) is fine and required by other
        # tests.
        body = triage_module.format_grace_warning_pr_comment(
            {"verdict": "fail", "missing": [], "explanation": ""}
        )
        assert "No need to ping" in body or "no need to ping" in body

    def test_grace_warnings_should_show_what_got_right(self, triage_module):
        # The "What you got right" section must appear in the grace warning
        # too, not only the close comment — the contributor sees the warning
        # first and that's their best chance to know what to keep.
        pr_body = triage_module.format_grace_warning_pr_comment(
            {
                "verdict": "fail",
                "linked_issue": True,
                "has_problem_description": True,
                "has_expected_vs_actual": True,
                "has_qa_proof": False,
                "missing": ["QA proof"],
                "explanation": "thin",
            }
        )
        assert "What you got right" in pr_body
        assert "Linked a related GitHub issue" in pr_body

        issue_body = triage_module.format_grace_warning_issue_comment(
            {
                "verdict": "fail",
                "kind": "feature",
                "has_motivation_example": True,
                "missing": ["concrete description"],
                "explanation": "vague",
            }
        )
        assert "What you got right" in issue_body
        assert "Motivation and concrete example" in issue_body

    def test_grace_warnings_should_use_softer_park_for_later_framing(
        self, triage_module
    ):
        # Same softer-framing pin as the close comment, but for the warning
        # — the contributor's first contact with the bot must not read as a
        # hard deadline / ultimatum.
        for body in (
            triage_module.format_grace_warning_pr_comment(
                {"verdict": "fail", "missing": [], "explanation": ""}
            ),
            triage_module.format_grace_warning_issue_comment(
                {"verdict": "fail", "missing": [], "explanation": ""}
            ),
        ):
            assert "park this for later" in body
            assert (
                "not a rejection" in body
                or "isn't a rejection" in body
                or ("isn't us saying" in body)
            )


class TestSecondsSinceLastGraceWarning:
    """Mirror of TestSecondsSinceLastReconsiderVerdict for the new helper.
    Both helpers share `_seconds_since_latest_marker_comment` underneath
    so the parsing logic is exercised either way; these tests pin the
    grace-marker-specific behavior."""

    def _make_comment(
        self,
        *,
        login: str,
        body: str,
        created_at: str | None = "2026-05-18T05:00:00Z",
    ) -> dict:
        comment: dict = {"user": {"login": login}, "body": body}
        if created_at is not None:
            comment["created_at"] = created_at
        return comment

    def test_should_return_none_when_no_grace_marker(self, triage_module, monkeypatch):
        comments = [
            self._make_comment(
                login="github-actions[bot]",
                body="Some other bot message",
            ),
            self._make_comment(login="random-user", body="ping?"),
        ]
        monkeypatch.setattr(
            triage_module, "_iter_paginated_json", lambda *a, **kw: iter(comments)
        )
        assert triage_module.seconds_since_last_grace_warning("o/r", 1) is None

    def test_should_ignore_non_bot_comments_with_marker(
        self, triage_module, monkeypatch
    ):
        # A user who quotes the marker in a question must NOT be treated
        # as the bot warning; otherwise the close-after-grace path would
        # never fire because the timer keeps resetting.
        comments = [
            self._make_comment(
                login="random-user",
                body=f"What is {triage_module.GRACE_COMMENT_MARKER}?",
            )
        ]
        monkeypatch.setattr(
            triage_module, "_iter_paginated_json", lambda *a, **kw: iter(comments)
        )
        assert triage_module.seconds_since_last_grace_warning("o/r", 1) is None

    def test_should_pick_latest_grace_marker(self, triage_module, monkeypatch):
        comments = [
            self._make_comment(
                login="github-actions[bot]",
                body="old warning " + triage_module.GRACE_COMMENT_MARKER,
                created_at="2026-05-18T03:00:00Z",
            ),
            self._make_comment(
                login="github-actions[bot]",
                body="newer warning " + triage_module.GRACE_COMMENT_MARKER,
                created_at="2026-05-18T04:55:00Z",
            ),
        ]
        monkeypatch.setattr(
            triage_module, "_iter_paginated_json", lambda *a, **kw: iter(comments)
        )

        import datetime as real_dt

        class FrozenDateTime(real_dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return real_dt.datetime(2026, 5, 18, 5, 0, 0, tzinfo=tz)

        frozen_module = type(triage_module.dt)("datetime")
        frozen_module.datetime = FrozenDateTime
        frozen_module.timezone = real_dt.timezone
        monkeypatch.setattr(triage_module, "dt", frozen_module)

        age = triage_module.seconds_since_last_grace_warning("o/r", 1)
        # Newer warning is 5 minutes (300s) before "now".
        assert age == 300.0


class TestTriageAllowlist:
    """The dogfood allowlist gates `triage`: while non-empty it is the sole
    author filter (only the named accounts are acted on) and it bypasses the
    internal-author exemption for them, so a maintainer can dogfood on their
    own org account. Emptying it restores the internal-author skip."""

    def _make_pr(self, **overrides):
        base = {
            "number": 1,
            "title": "PR title",
            "body": "Body with no linked issue and no QA proof.",
            "state": "open",
            "author_association": "NONE",
            "user": {"login": "mateo-berri"},
        }
        base.update(overrides)
        return base

    def test_should_skip_author_not_on_allowlist(self, triage_module, monkeypatch):
        pr = self._make_pr(user={"login": "random-oss-dev"})
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        result = triage_module.triage(
            repo="o/r",
            kind="pr",
            number=1,
            close=True,
            model="m",
            judge=lambda p: pytest.fail("LLM must not run for non-allowlisted author"),
        )
        assert result["action"] == "skip-not-allowlisted"

    def test_should_act_on_allowlisted_internal_author(
        self, triage_module, monkeypatch
    ):
        pr = self._make_pr(author_association="MEMBER", user={"login": "mateo-berri"})
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        result = triage_module.triage(
            repo="o/r",
            kind="pr",
            number=1,
            close=True,
            model="m",
            judge=lambda p: json.dumps(
                {"verdict": "pass", "missing": [], "explanation": "ok"}
            ),
        )
        assert result["action"] == "pass-llm"

    def test_empty_allowlist_restores_internal_skip(self, triage_module, monkeypatch):
        pr = self._make_pr(
            author_association="MEMBER", user={"login": "krrishdholakia"}
        )
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        result = triage_module.triage(
            repo="o/r",
            kind="pr",
            number=1,
            close=True,
            model="m",
            judge=lambda p: pytest.fail("LLM must not run for internal author"),
            allowlist=frozenset(),
        )
        assert result["action"] == "skip-internal-author"

    def test_allowlist_constant_is_the_two_dogfood_accounts(self, triage_module):
        assert triage_module.ALLOWLIST_LOGINS == frozenset(
            {"mateo-berri", "swiftwinds"}
        )
        for login in triage_module.ALLOWLIST_LOGINS:
            assert login == login.lower(), login
