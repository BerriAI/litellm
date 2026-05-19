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
        body = triage_module.format_issue_close_comment(
            {"verdict": "fail", "missing": ["repro"], "explanation": "thin"}
        )
        assert "@agent-shin reconsider" in body
        assert "Reopen the issue" not in body


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


class TestWasAutoClosedByAgentShin:
    """Provenance check that gates reconsider's reopen path."""

    @staticmethod
    def _install(monkeypatch, triage_module, events, comments):
        monkeypatch.setattr(triage_module, "fetch_issue_events", lambda repo, n: events)
        monkeypatch.setattr(
            triage_module, "fetch_issue_comments", lambda repo, n: comments
        )

    def test_should_return_true_when_latest_close_was_bot_with_marker(
        self, triage_module, monkeypatch
    ):
        events = [
            {
                "event": "closed",
                "actor": {"login": "github-actions[bot]"},
                "created_at": "2025-01-01T00:00:10Z",
            }
        ]
        comments = [
            {
                "user": {"login": "github-actions[bot]"},
                "created_at": "2025-01-01T00:00:05Z",
                "body": (
                    "👋 Hi, thanks for the PR! I'm **Agent Shin**, the automated "
                    "triage bot for this repository.\n\nThis PR is being **auto-closed**..."
                ),
            }
        ]
        self._install(monkeypatch, triage_module, events, comments)
        assert triage_module.was_auto_closed_by_agent_shin("o/r", 1) is True

    def test_should_return_false_when_no_close_event(self, triage_module, monkeypatch):
        self._install(monkeypatch, triage_module, [], [])
        assert triage_module.was_auto_closed_by_agent_shin("o/r", 1) is False

    def test_should_ignore_non_bot_author_with_marker(self, triage_module, monkeypatch):
        # A contributor pasting the marker into a manual comment must NOT
        # be treated as proof Agent Shin closed the PR. Even if a bot did
        # the most recent close, the marker comment must be authored by
        # that same bot login — not by the human.
        events = [
            {
                "event": "closed",
                "actor": {"login": "github-actions[bot]"},
                "created_at": "2025-01-01T00:00:10Z",
            }
        ]
        comments = [
            {
                "user": {"login": "outside-dev"},
                "created_at": "2025-01-01T00:00:05Z",
                "body": "I'm **Agent Shin**, just kidding — please reconsider this.",
            }
        ]
        self._install(monkeypatch, triage_module, events, comments)
        assert triage_module.was_auto_closed_by_agent_shin("o/r", 1) is False

    def test_should_ignore_bot_comment_without_marker(self, triage_module, monkeypatch):
        # Other bots (codecov, cla-assistant, etc.) post on every PR; their
        # presence must not satisfy the provenance check.
        events = [
            {
                "event": "closed",
                "actor": {"login": "github-actions[bot]"},
                "created_at": "2025-01-01T00:00:10Z",
            }
        ]
        comments = [
            {
                "user": {"login": "codecov[bot]"},
                "created_at": "2025-01-01T00:00:01Z",
                "body": "## Codecov Report ...",
            },
            {
                "user": {"login": "greptile-apps[bot]"},
                "created_at": "2025-01-01T00:00:02Z",
                "body": "Confidence Score: 2/5",
            },
        ]
        self._install(monkeypatch, triage_module, events, comments)
        assert triage_module.was_auto_closed_by_agent_shin("o/r", 1) is False

    def test_should_anchor_on_most_recent_close_event(self, triage_module, monkeypatch):
        # Agent Shin auto-closed first, contributor commented after; the
        # bot-authored marker comment is anywhere in the timeline.
        events = [
            {
                "event": "labeled",
                "actor": {"login": "krrishdholakia"},
                "created_at": "2025-01-01T00:00:01Z",
            },
            {
                "event": "closed",
                "actor": {"login": "github-actions[bot]"},
                "created_at": "2025-01-01T00:00:10Z",
            },
        ]
        comments = [
            {
                "user": {"login": "codecov[bot]"},
                "created_at": "2025-01-01T00:00:02Z",
                "body": "## Codecov Report",
            },
            {
                "user": {"login": "github-actions[bot]"},
                "created_at": "2025-01-01T00:00:05Z",
                "body": "I'm **Agent Shin**, the automated triage bot ...",
            },
            {
                "user": {"login": "outside-dev"},
                "created_at": "2025-01-01T00:00:15Z",
                "body": "Replying after auto-close ...",
            },
        ]
        self._install(monkeypatch, triage_module, events, comments)
        assert triage_module.was_auto_closed_by_agent_shin("o/r", 1) is True

    def test_should_refuse_when_maintainer_re_closed_after_agent_shin(
        self, triage_module, monkeypatch
    ):
        # Agent Shin auto-closed, the PR was reopened, then a maintainer
        # closed it again (e.g. as a duplicate). `@agent-shin reconsider`
        # must NOT override the maintainer's later closure even though the
        # historical Agent Shin marker comment still exists.
        events = [
            {
                "event": "closed",
                "actor": {"login": "github-actions[bot]"},
                "created_at": "2025-01-01T00:00:10Z",
            },
            {
                "event": "reopened",
                "actor": {"login": "github-actions[bot]"},
                "created_at": "2025-01-01T00:00:20Z",
            },
            {
                "event": "closed",
                "actor": {"login": "krrishdholakia"},
                "created_at": "2025-01-01T00:00:30Z",
            },
        ]
        comments = [
            {
                "user": {"login": "github-actions[bot]"},
                "created_at": "2025-01-01T00:00:05Z",
                "body": "I'm **Agent Shin**, the automated triage bot ...",
            }
        ]
        self._install(monkeypatch, triage_module, events, comments)
        assert triage_module.was_auto_closed_by_agent_shin("o/r", 1) is False

    def test_should_refuse_when_unrelated_bot_re_closed_after_agent_shin(
        self, triage_module, monkeypatch
    ):
        # Agent Shin closed, the PR was reopened, then a different bot
        # (stale, etc.) closed it. The marker comment is from
        # `github-actions[bot]` but the most recent closer is
        # `stale[bot]`, so the logins don't match -> refuse to reopen.
        events = [
            {
                "event": "closed",
                "actor": {"login": "github-actions[bot]"},
                "created_at": "2025-01-01T00:00:10Z",
            },
            {
                "event": "reopened",
                "actor": {"login": "github-actions[bot]"},
                "created_at": "2025-01-01T00:00:20Z",
            },
            {
                "event": "closed",
                "actor": {"login": "stale[bot]"},
                "created_at": "2025-01-01T00:00:30Z",
            },
        ]
        comments = [
            {
                "user": {"login": "github-actions[bot]"},
                "created_at": "2025-01-01T00:00:05Z",
                "body": "I'm **Agent Shin**, the automated triage bot ...",
            }
        ]
        self._install(monkeypatch, triage_module, events, comments)
        assert triage_module.was_auto_closed_by_agent_shin("o/r", 1) is False

    def test_should_refuse_when_stale_workflow_re_closed_after_agent_shin(
        self, triage_module, monkeypatch
    ):
        # The repo's `actions/stale` workflow uses `secrets.GITHUB_TOKEN`,
        # so its closes are attributed to `github-actions[bot]` — the same
        # login as Agent Shin. A historical Agent Shin marker comment
        # from an earlier close cycle must NOT satisfy the provenance check
        # for a later stale-initiated close (which never posts that marker).
        events = [
            {
                "event": "closed",
                "actor": {"login": "github-actions[bot]"},
                "created_at": "2025-01-01T00:00:10Z",
            },
            {
                "event": "reopened",
                "actor": {"login": "github-actions[bot]"},
                "created_at": "2025-01-01T00:00:20Z",
            },
            {
                "event": "closed",
                "actor": {"login": "github-actions[bot]"},
                "created_at": "2025-04-01T00:00:00Z",
            },
        ]
        comments = [
            {
                "user": {"login": "github-actions[bot]"},
                "created_at": "2025-01-01T00:00:05Z",
                "body": "I'm **Agent Shin**, the automated triage bot ...",
            },
            {
                "user": {"login": "github-actions[bot]"},
                "created_at": "2025-03-25T00:00:00Z",
                "body": "This pull request has been automatically marked as stale...",
            },
        ]
        self._install(monkeypatch, triage_module, events, comments)
        assert triage_module.was_auto_closed_by_agent_shin("o/r", 1) is False

    def test_should_accept_marker_from_current_cycle_after_reopen(
        self, triage_module, monkeypatch
    ):
        # Two clean Agent Shin cycles: closed, reconsider→reopened, closed
        # again with a new marker comment posted in the current cycle.
        # The second-cycle marker is what proves provenance.
        events = [
            {
                "event": "closed",
                "actor": {"login": "github-actions[bot]"},
                "created_at": "2025-01-01T00:00:10Z",
            },
            {
                "event": "reopened",
                "actor": {"login": "github-actions[bot]"},
                "created_at": "2025-01-01T00:00:20Z",
            },
            {
                "event": "closed",
                "actor": {"login": "github-actions[bot]"},
                "created_at": "2025-01-01T00:00:40Z",
            },
        ]
        comments = [
            {
                "user": {"login": "github-actions[bot]"},
                "created_at": "2025-01-01T00:00:05Z",
                "body": "I'm **Agent Shin**, the automated triage bot ...",
            },
            {
                "user": {"login": "github-actions[bot]"},
                "created_at": "2025-01-01T00:00:35Z",
                "body": "I'm **Agent Shin** — still missing X ...",
            },
        ]
        self._install(monkeypatch, triage_module, events, comments)
        assert triage_module.was_auto_closed_by_agent_shin("o/r", 1) is True


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
            "user": {"login": "outside-dev"},
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
        # must get a chance to apply the stricter rubric.
        pr = self._make_pr(body="See #1234 for context. No QA proof here.")
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
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
        assert result["action"] == "would-close"

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

    def test_should_return_would_close_in_dry_run(self, triage_module, monkeypatch):
        pr = self._make_pr(body="just a sentence.")
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)

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

    def test_should_post_comment_and_close_when_close_enabled(
        self, triage_module, monkeypatch
    ):
        pr = self._make_pr(body="just a sentence.")
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
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

    def test_should_reopen_on_reconsider_pass(self, triage_module, monkeypatch):
        # Reconsider on a closed PR with a passing verdict -> reopen + post a
        # friendly "re-evaluated" comment.
        pr = self._make_pr(
            state="closed", body="Updated body with QA proof + screenshots."
        )
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        # Provenance: the PR was auto-closed by Agent Shin (a bot-authored
        # auto-close comment exists), so reconsider is allowed to reopen.
        monkeypatch.setattr(
            triage_module, "was_auto_closed_by_agent_shin", lambda repo, n: True
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

    def test_should_post_still_failing_on_reconsider_fail(
        self, triage_module, monkeypatch
    ):
        pr = self._make_pr(state="closed", body="still empty")
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        monkeypatch.setattr(
            triage_module, "was_auto_closed_by_agent_shin", lambda repo, n: True
        )
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

    def test_should_reopen_on_reconsider_with_linked_issue_short_circuit(
        self, triage_module, monkeypatch
    ):
        # The linked-issue short-circuit also has to honor reconsider mode:
        # if the contributor edited the body to add `Fixes #1234`, the regex
        # path should reopen the PR without calling the LLM.
        pr = self._make_pr(state="closed", body="Fixes #1234\n\nAddresses the bug.")
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        monkeypatch.setattr(
            triage_module, "was_auto_closed_by_agent_shin", lambda repo, n: True
        )
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
        # Internal check must fire *before* provenance, so the provenance
        # helper should never be invoked for an internal author.
        monkeypatch.setattr(
            triage_module,
            "was_auto_closed_by_agent_shin",
            lambda *a, **kw: pytest.fail("must not check provenance for internal"),
        )
        result = triage_module.triage(
            repo="o/r",
            kind="pr",
            number=1,
            close=True,
            model="m",
            judge=lambda p: pytest.fail("LLM must not run for internal author"),
            reconsider=True,
        )
        assert result["action"] == "skip-internal-author"

    def test_should_skip_reconsider_when_not_bot_closed(
        self, triage_module, monkeypatch
    ):
        # A maintainer-closed PR (no Agent Shin auto-close comment) must
        # never be reopened by `@agent-shin reconsider`, regardless of how
        # good the LLM verdict would be. Otherwise the original author
        # could polish the description and silently override a maintainer's
        # "closed as duplicate / out of scope" decision.
        pr = self._make_pr(state="closed", body="Fixes #1234")
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        monkeypatch.setattr(
            triage_module, "was_auto_closed_by_agent_shin", lambda repo, n: False
        )
        monkeypatch.setattr(
            triage_module,
            "post_comment",
            lambda *a, **kw: pytest.fail("must not comment when not bot-closed"),
        )
        monkeypatch.setattr(
            triage_module,
            "reopen_pr",
            lambda *a, **kw: pytest.fail("must not reopen when not bot-closed"),
        )
        result = triage_module.triage(
            repo="o/r",
            kind="pr",
            number=1,
            close=True,
            model="m",
            judge=lambda p: pytest.fail("LLM must not run when not bot-closed"),
            reconsider=True,
        )
        assert result["action"] == "skip-not-bot-closed"

    def test_should_skip_reconsider_issue_when_not_bot_closed(
        self, triage_module, monkeypatch
    ):
        # Same provenance gate for issues: only Agent Shin auto-closed
        # issues are eligible for reopen-on-reconsider.
        issue = {
            "number": 7,
            "title": "Bug",
            "body": "Repro: curl ...",
            "state": "closed",
            "author_association": "NONE",
            "user": {"login": "outside"},
        }
        monkeypatch.setattr(triage_module, "fetch_issue", lambda repo, n: issue)
        monkeypatch.setattr(
            triage_module, "was_auto_closed_by_agent_shin", lambda repo, n: False
        )
        monkeypatch.setattr(
            triage_module,
            "reopen_issue",
            lambda *a, **kw: pytest.fail("must not reopen maintainer-closed issue"),
        )
        result = triage_module.triage(
            repo="o/r",
            kind="issue",
            number=7,
            close=True,
            model="m",
            judge=lambda p: pytest.fail("LLM must not run for non-bot-closed issue"),
            reconsider=True,
        )
        assert result["action"] == "skip-not-bot-closed"

    def test_should_preview_reopen_in_reconsider_dry_run(
        self, triage_module, monkeypatch
    ):
        # When `close=False` and `reconsider=True`, a passing verdict must
        # produce a `would-reopen` preview WITHOUT posting a comment or
        # reopening — same dry-run pattern as `would-close` in regular mode.
        pr = self._make_pr(state="closed", body="Now with screenshots + repro.")
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        monkeypatch.setattr(
            triage_module, "was_auto_closed_by_agent_shin", lambda repo, n: True
        )
        monkeypatch.setattr(
            triage_module,
            "post_comment",
            lambda *a, **kw: pytest.fail("dry-run must not post"),
        )
        monkeypatch.setattr(
            triage_module,
            "reopen_pr",
            lambda *a, **kw: pytest.fail("dry-run must not reopen"),
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
        # The preview should include the comment body the bot WOULD post
        # (useful for $GITHUB_STEP_SUMMARY).
        assert "reopened" in result["comment"].lower()

    def test_should_preview_reopen_in_reconsider_dry_run_linked_issue(
        self, triage_module, monkeypatch
    ):
        # The linked-issue short-circuit also has to honor dry-run in
        # reconsider mode — no LLM call AND no destructive side effects.
        pr = self._make_pr(state="closed", body="Fixes #1234\n\nDetails.")
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        monkeypatch.setattr(
            triage_module, "was_auto_closed_by_agent_shin", lambda repo, n: True
        )
        monkeypatch.setattr(
            triage_module,
            "post_comment",
            lambda *a, **kw: pytest.fail("dry-run must not post"),
        )
        monkeypatch.setattr(
            triage_module,
            "reopen_pr",
            lambda *a, **kw: pytest.fail("dry-run must not reopen"),
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
        assert "reopened" in result["comment"].lower()

    def test_should_preview_still_failing_in_reconsider_dry_run(
        self, triage_module, monkeypatch
    ):
        # When `close=False` and the verdict is fail, the dry-run preview
        # must say `would-leave-closed-still-failing` and not post the
        # "still failing" comment.
        pr = self._make_pr(state="closed", body="still thin")
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        monkeypatch.setattr(
            triage_module, "was_auto_closed_by_agent_shin", lambda repo, n: True
        )
        monkeypatch.setattr(
            triage_module,
            "post_comment",
            lambda *a, **kw: pytest.fail("dry-run must not post"),
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
        assert result["action"] == "would-leave-closed-still-failing"
        assert "QA proof" in result["comment"]

    def test_should_reopen_issue_on_reconsider_pass(self, triage_module, monkeypatch):
        issue = {
            "number": 7,
            "title": "Bug: now with repro",
            "body": "## Repro\n```bash\ncurl ...\n```\n\nExpected X, got Y.",
            "state": "closed",
            "author_association": "NONE",
            "user": {"login": "outside"},
        }
        monkeypatch.setattr(triage_module, "fetch_issue", lambda repo, n: issue)
        monkeypatch.setattr(
            triage_module, "was_auto_closed_by_agent_shin", lambda repo, n: True
        )
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

    def test_should_reopen_before_posting_on_reconsider_pass(
        self, triage_module, monkeypatch
    ):
        # A failed reopen call must not leave a misleading "we reopened it"
        # comment on a still-closed PR. Pin the order: reopen happens first;
        # if reopen raises, post_comment must not run.
        pr = self._make_pr(state="closed", body="Updated body with QA proof.")
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        monkeypatch.setattr(
            triage_module, "was_auto_closed_by_agent_shin", lambda repo, n: True
        )
        monkeypatch.setattr(
            triage_module,
            "post_comment",
            lambda *a, **kw: pytest.fail("must not post comment if reopen fails"),
        )

        def boom(*a, **kw):
            raise RuntimeError("reopen 422")

        monkeypatch.setattr(triage_module, "reopen_pr", boom)
        with pytest.raises(RuntimeError):
            triage_module.triage(
                repo="o/r",
                kind="pr",
                number=42,
                close=True,
                model="m",
                judge=lambda p: json.dumps(
                    {"verdict": "pass", "missing": [], "explanation": "ok"}
                ),
                reconsider=True,
            )

    def test_should_reopen_before_posting_on_reconsider_linked_issue(
        self, triage_module, monkeypatch
    ):
        # Same ordering invariant for the linked-issue short-circuit.
        pr = self._make_pr(state="closed", body="Fixes #1234")
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: pr)
        monkeypatch.setattr(
            triage_module, "was_auto_closed_by_agent_shin", lambda repo, n: True
        )
        monkeypatch.setattr(
            triage_module,
            "post_comment",
            lambda *a, **kw: pytest.fail("must not post comment if reopen fails"),
        )

        def boom(*a, **kw):
            raise RuntimeError("reopen 422")

        monkeypatch.setattr(triage_module, "reopen_pr", boom)
        with pytest.raises(RuntimeError):
            triage_module.triage(
                repo="o/r",
                kind="pr",
                number=55,
                close=True,
                model="m",
                judge=lambda p: pytest.fail("LLM must not run for linked-issue path"),
                reconsider=True,
            )

    def test_should_reopen_before_posting_on_reconsider_issue_pass(
        self, triage_module, monkeypatch
    ):
        # Same ordering invariant for issues (reopen_issue, not reopen_pr).
        issue = {
            "number": 7,
            "title": "Bug",
            "body": "Repro: curl ...",
            "state": "closed",
            "author_association": "NONE",
            "user": {"login": "outside"},
        }
        monkeypatch.setattr(triage_module, "fetch_issue", lambda repo, n: issue)
        monkeypatch.setattr(
            triage_module, "was_auto_closed_by_agent_shin", lambda repo, n: True
        )
        monkeypatch.setattr(
            triage_module,
            "post_comment",
            lambda *a, **kw: pytest.fail("must not post comment if reopen fails"),
        )

        def boom(*a, **kw):
            raise RuntimeError("reopen 422")

        monkeypatch.setattr(triage_module, "reopen_issue", boom)
        with pytest.raises(RuntimeError):
            triage_module.triage(
                repo="o/r",
                kind="issue",
                number=7,
                close=True,
                model="m",
                judge=lambda p: json.dumps(
                    {"verdict": "pass", "missing": [], "explanation": "ok"}
                ),
                reconsider=True,
            )

    def test_should_triage_issues_kind(self, triage_module, monkeypatch):
        issue = {
            "number": 7,
            "title": "Bug: X is broken",
            "body": "no detail",
            "state": "open",
            "author_association": "NONE",
            "user": {"login": "outside"},
        }
        monkeypatch.setattr(triage_module, "fetch_issue", lambda repo, n: issue)
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
