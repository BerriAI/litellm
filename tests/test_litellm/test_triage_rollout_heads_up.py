"""Unit tests for the one-shot 7-day heads-up sweep.

Exercises:

  * The ``_agent_shin_actions`` dry-run wrappers — each ``maybe_*`` helper
    must call the real underlying mutation iff ``dry_run=False``, and log to
    stdout otherwise.
  * ``triage_rollout_heads_up._would_be_closed`` — the predicate that
    decides "would the future bot close this?" for both PRs and issues.
  * ``triage_rollout_heads_up._process_one`` — the per-item processor:
    skip when state != open, skip internal authors, skip already-notified
    items, post heads-up on failing items, leave passing items alone.
  * ``triage_rollout_heads_up.run`` — the sweep loop end-to-end, in both
    dry-run and real modes, with the comment-posting injected so we never
    talk to GitHub.

Every test stubs out ``gh()`` and the GitHub mutations; nothing in this file
ever shells out.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / ".github" / "scripts"


@pytest.fixture(scope="module")
def triage_module():
    """Load triage_with_llm under its canonical name so the sibling modules
    can `from triage_with_llm import ...`."""
    spec = importlib.util.spec_from_file_location(
        "triage_with_llm", _SCRIPTS_DIR / "triage_with_llm.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["triage_with_llm"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def actions_module(triage_module):
    spec = importlib.util.spec_from_file_location(
        "_agent_shin_actions", _SCRIPTS_DIR / "_agent_shin_actions.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["_agent_shin_actions"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def heads_up_module(triage_module, actions_module):
    spec = importlib.util.spec_from_file_location(
        "triage_rollout_heads_up", _SCRIPTS_DIR / "triage_rollout_heads_up.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["triage_rollout_heads_up"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# _agent_shin_actions: the dry-run wrappers


class TestActionsDryRun:
    """Each maybe_* helper must NOT hit GitHub in dry-run, and MUST hit it
    in real mode. The whole rollout's safety story rests on this."""

    def test_maybe_post_comment_dry_run_logs_only(
        self, actions_module, triage_module, monkeypatch, capsys
    ):
        called = []
        monkeypatch.setattr(
            triage_module,
            "post_comment",
            lambda *a, **k: called.append((a, k)),
        )
        actions_module.maybe_post_comment("o/r", 7, "hello", dry_run=True)
        assert called == []
        assert "[DRY RUN] comment o/r#7" in capsys.readouterr().out

    def test_maybe_post_comment_real_run_calls_through(
        self, actions_module, triage_module, monkeypatch
    ):
        called = []
        monkeypatch.setattr(
            triage_module,
            "post_comment",
            lambda repo, n, body: called.append((repo, n, body)),
        )
        actions_module.maybe_post_comment("o/r", 7, "hello", dry_run=False)
        assert called == [("o/r", 7, "hello")]


# ---------------------------------------------------------------------------
# _would_be_closed predicate


class TestWouldBeClosed:
    def test_pr_passing_returns_false(self, heads_up_module):
        assert (
            heads_up_module._would_be_closed(
                "pr", {"passing": True, "action": "noop-passing"}
            )
            is False
        )

    def test_pr_failing_returns_true(self, heads_up_module):
        assert (
            heads_up_module._would_be_closed(
                "pr",
                {
                    "passing": False,
                    "action": "would-close",
                    "verdict": {"verdict": "fail"},
                },
            )
            is True
        )

    def test_pr_skipped_returns_false(self, heads_up_module):
        # passing is None for skip paths (internal-author, llm-error, etc.)
        assert (
            heads_up_module._would_be_closed("pr", {"action": "skip-internal-author"})
            is False
        )

    def test_issue_pass_returns_false(self, heads_up_module):
        assert (
            heads_up_module._would_be_closed(
                "issue", {"action": "pass-llm", "verdict": {"verdict": "pass"}}
            )
            is False
        )

    def test_issue_fail_returns_true(self, heads_up_module):
        assert (
            heads_up_module._would_be_closed(
                "issue", {"action": "would-close", "verdict": {"verdict": "fail"}}
            )
            is True
        )

    def test_issue_missing_verdict_returns_false(self, heads_up_module):
        # Skip paths don't surface a verdict; treat as "won't close".
        assert (
            heads_up_module._would_be_closed("issue", {"action": "skip-not-open"})
            is False
        )


# ---------------------------------------------------------------------------
# Comment formatter — wording sanity checks


class TestHeadsUpCommentBody:
    def test_pr_comment_contains_cutoff_rubric_marker(self, heads_up_module):
        body = heads_up_module.format_heads_up_comment(
            kind="pr",
            verdict={"verdict": "fail", "missing": ["QA proof"], "explanation": "thin"},
            greptile_score=3,
            cutoff=dt.date(2026, 6, 1),
        )
        assert "Monday, June 1, 2026" in body  # cutoff readable
        assert "09:00 UTC" in body  # deadline is timezone-explicit
        assert "we'll close it" in body  # hard deadline, not a passive notice
        assert "2-hour lifetime" in body  # post-rollout steady state
        assert "Greptile" in body and "3/5" in body  # specific shortfall
        assert "QA proof" in body  # missing piece surfaced
        assert "PR *description*" in body  # description-only note
        assert heads_up_module.HEADS_UP_MARKER in body  # idempotency marker

    def test_issue_comment_uses_reconsider_recovery_path(self, heads_up_module):
        # OSS authors can't reopen an issue the bot closed (read access only
        # lets them reopen issues they closed themselves), so the heads-up
        # recovery path is `@agent-shin reconsider`, not self-reopen.
        body = heads_up_module.format_heads_up_comment(
            kind="issue",
            verdict={"verdict": "fail", "missing": ["repro"], "explanation": ""},
            greptile_score=None,
            cutoff=dt.date(2026, 6, 1),
        )
        assert "@agent-shin reconsider" in body
        assert heads_up_module.HEADS_UP_MARKER in body

    def test_empty_missing_uses_fallback_copy(self, heads_up_module):
        body = heads_up_module.format_heads_up_comment(
            kind="pr",
            verdict={"verdict": "fail", "missing": [], "explanation": ""},
            greptile_score=None,
            cutoff=dt.date(2026, 6, 1),
        )
        assert "couldn't articulate" in body
        # Make sure the fallback didn't leave us with a broken sentence.
        assert "specific missing piece" in body


# ---------------------------------------------------------------------------
# _process_one — per-item dispatch


def _stub_fetchers(heads_up_module, triage_module, *, item):
    """Monkeypatch fetch_pr and fetch_issue (both in triage_with_llm and the
    re-imported names in heads_up_module) to return `item`."""
    return [
        (triage_module, "fetch_pr", lambda repo, n: item),
        (triage_module, "fetch_issue", lambda repo, n: item),
        (heads_up_module, "fetch_pr", lambda repo, n: item),
        (heads_up_module, "fetch_issue", lambda repo, n: item),
    ]


class TestProcessOne:
    """Per-item processing: the right skip reason fires for each scenario,
    and the heads-up only goes out when the rubric is genuinely failing."""

    @pytest.fixture
    def patch_env(self, heads_up_module, triage_module, monkeypatch):
        """Helper that returns a callable to install a PR/issue body, suppress
        marker checks, and stub the comment poster."""
        posts = []
        monkeypatch.setattr(
            heads_up_module,
            "maybe_post_comment",
            lambda repo, n, body, *, dry_run: posts.append((repo, n, body, dry_run)),
        )
        monkeypatch.setattr(heads_up_module, "_has_heads_up_marker", lambda item: False)
        monkeypatch.setattr(
            heads_up_module, "_comments_have_marker", lambda repo, n: False
        )

        def _install(item):
            for mod, name, fn in _stub_fetchers(
                heads_up_module, triage_module, item=item
            ):
                monkeypatch.setattr(mod, name, fn)

        return _install, posts

    def test_skip_closed_pr(self, heads_up_module, patch_env):
        install, posts = patch_env
        install(
            {"state": "closed", "user": {"login": "ext"}, "author_association": "NONE"}
        )
        r = heads_up_module._process_one(
            repo="o/r",
            kind="pr",
            number=7,
            model="m",
            cutoff=dt.date(2026, 6, 1),
            dry_run=True,
        )
        assert r["action"] == "skip-not-open"
        assert posts == []

    def test_skip_internal_pr(self, heads_up_module, patch_env):
        install, posts = patch_env
        install(
            {
                "state": "open",
                "user": {"login": "krrishdholakia"},
                "author_association": "MEMBER",
                "body": "",
                "labels": [],
                "created_at": "2026-05-25T00:00:00Z",
            }
        )
        r = heads_up_module._process_one(
            repo="o/r",
            kind="pr",
            number=7,
            model="m",
            cutoff=dt.date(2026, 6, 1),
            dry_run=True,
            allowlist=frozenset(),
        )
        assert r["action"] == "skip-internal-author"
        assert posts == []

    def test_skip_passing_pr(self, heads_up_module, patch_env, monkeypatch):
        install, posts = patch_env
        install(
            {
                "state": "open",
                "user": {"login": "mateo-berri"},
                "author_association": "NONE",
                "body": "Fixes #123 — clean fix with a passing rubric.",
                "labels": [],
                "created_at": "2026-05-25T00:00:00Z",
            }
        )
        monkeypatch.setattr(
            heads_up_module,
            "_evaluate_pr",
            lambda **kwargs: {
                "action": "noop-passing",
                "passing": True,
                "verdict": {"verdict": "pass"},
                "greptile_score": 5,
            },
        )
        r = heads_up_module._process_one(
            repo="o/r",
            kind="pr",
            number=7,
            model="m",
            cutoff=dt.date(2026, 6, 1),
            dry_run=True,
        )
        assert r["action"] == "skip-passing"
        assert posts == []

    def test_failing_pr_posts_heads_up_dry_run(
        self, heads_up_module, patch_env, monkeypatch, capsys
    ):
        install, posts = patch_env
        install(
            {
                "state": "open",
                "user": {"login": "mateo-berri"},
                "author_association": "NONE",
                "body": "thin",
                "labels": [],
                "created_at": "2026-05-25T00:00:00Z",
            }
        )
        monkeypatch.setattr(
            heads_up_module,
            "_evaluate_pr",
            lambda **kwargs: {
                "action": "would-close",
                "passing": False,
                "verdict": {
                    "verdict": "fail",
                    "missing": ["QA proof"],
                    "explanation": "PR body is one line.",
                },
                "greptile_score": 3,
            },
        )
        r = heads_up_module._process_one(
            repo="o/r",
            kind="pr",
            number=7,
            model="m",
            cutoff=dt.date(2026, 6, 1),
            dry_run=True,
        )
        assert r["action"] == "would-post-heads-up"
        assert posts == [("o/r", 7, posts[0][2], True)]  # tuple shape preserved
        assert "QA proof" in posts[0][2]
        assert heads_up_module.HEADS_UP_MARKER in posts[0][2]

    def test_failing_issue_posts_heads_up_real_run(
        self, heads_up_module, patch_env, monkeypatch
    ):
        install, posts = patch_env
        install(
            {
                "state": "open",
                "user": {"login": "mateo-berri"},
                "author_association": "NONE",
                "body": "X is broken",
                "labels": [],
                "created_at": "2026-05-25T00:00:00Z",
            }
        )
        monkeypatch.setattr(
            heads_up_module,
            "_evaluate_issue",
            lambda **kwargs: {
                "action": "would-close",
                "verdict": {
                    "verdict": "fail",
                    "missing": ["reproduction"],
                    "explanation": "too thin",
                },
            },
        )
        r = heads_up_module._process_one(
            repo="o/r",
            kind="issue",
            number=42,
            model="m",
            cutoff=dt.date(2026, 6, 1),
            dry_run=False,
        )
        assert r["action"] == "heads-up-posted"
        assert len(posts) == 1
        _, n, _, dry = posts[0]
        assert n == 42 and dry is False

    def test_already_notified_is_skipped(self, heads_up_module, patch_env, monkeypatch):
        install, posts = patch_env
        install(
            {
                "state": "open",
                "user": {"login": "mateo-berri"},
                "author_association": "NONE",
                "body": "thin",
                "labels": [],
                "created_at": "2026-05-25T00:00:00Z",
            }
        )
        # Override the marker check for this scenario only.
        monkeypatch.setattr(
            heads_up_module, "_comments_have_marker", lambda repo, n: True
        )
        r = heads_up_module._process_one(
            repo="o/r",
            kind="pr",
            number=7,
            model="m",
            cutoff=dt.date(2026, 6, 1),
            dry_run=True,
        )
        assert r["action"] == "skip-already-notified"
        assert posts == []

    def test_ignore_existing_marker_forces_post(
        self, heads_up_module, patch_env, monkeypatch
    ):
        install, posts = patch_env
        install(
            {
                "state": "open",
                "user": {"login": "mateo-berri"},
                "author_association": "NONE",
                "body": "thin",
                "labels": [],
                "created_at": "2026-05-25T00:00:00Z",
            }
        )
        monkeypatch.setattr(
            heads_up_module, "_comments_have_marker", lambda repo, n: True
        )
        monkeypatch.setattr(
            heads_up_module,
            "_evaluate_pr",
            lambda **kwargs: {
                "action": "would-close",
                "passing": False,
                "verdict": {"verdict": "fail", "missing": ["X"], "explanation": ""},
                "greptile_score": None,
            },
        )
        r = heads_up_module._process_one(
            repo="o/r",
            kind="pr",
            number=7,
            model="m",
            cutoff=dt.date(2026, 6, 1),
            dry_run=True,
            skip_marker_check=True,
        )
        assert r["action"] == "would-post-heads-up"


# ---------------------------------------------------------------------------
# run() — sweep loop


class TestRun:
    """End-to-end the sweep loop with a tiny fake repo: 1 passing PR, 1
    failing PR, 1 passing issue, 1 failing issue."""

    @pytest.fixture
    def configured(self, heads_up_module, triage_module, monkeypatch):
        posts = []
        monkeypatch.setattr(
            heads_up_module,
            "maybe_post_comment",
            lambda repo, n, body, *, dry_run: posts.append((n, dry_run, body)),
        )
        monkeypatch.setattr(heads_up_module, "_has_heads_up_marker", lambda item: False)
        monkeypatch.setattr(
            heads_up_module, "_comments_have_marker", lambda repo, n: False
        )

        def fake_list(repo, kind):
            return [1, 2] if kind == "pr" else [101, 102]

        monkeypatch.setattr(heads_up_module, "_list_open_numbers", fake_list)

        def make_item(login="mateo-berri"):
            return {
                "state": "open",
                "user": {"login": login},
                "author_association": "NONE",
                "body": "thin",
                "labels": [],
                "created_at": "2026-05-25T00:00:00Z",
            }

        monkeypatch.setattr(heads_up_module, "fetch_pr", lambda repo, n: make_item())
        monkeypatch.setattr(heads_up_module, "fetch_issue", lambda repo, n: make_item())
        monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: make_item())
        monkeypatch.setattr(triage_module, "fetch_issue", lambda repo, n: make_item())

        def pr_eval(*, number, **kwargs):
            if number == 1:
                return {
                    "action": "noop-passing",
                    "passing": True,
                    "verdict": {"verdict": "pass"},
                }
            return {
                "action": "would-close",
                "passing": False,
                "verdict": {"verdict": "fail", "missing": ["m"], "explanation": ""},
                "greptile_score": 2,
            }

        def issue_eval(*, number, **kwargs):
            if number == 101:
                return {"action": "pass-llm", "verdict": {"verdict": "pass"}}
            return {
                "action": "would-close",
                "verdict": {"verdict": "fail", "missing": ["repro"], "explanation": ""},
            }

        monkeypatch.setattr(heads_up_module, "_evaluate_pr", pr_eval)
        monkeypatch.setattr(heads_up_module, "_evaluate_issue", issue_eval)
        return posts

    def test_dry_run_posts_nothing_but_logs_both_would_posts(
        self, heads_up_module, configured, capsys
    ):
        results = heads_up_module.run(
            repo="o/r",
            close=False,
            cutoff=dt.date(2026, 6, 1),
            model="m",
        )
        actions = [r["action"] for r in results]
        assert actions.count("would-post-heads-up") == 2
        assert actions.count("skip-passing") == 2
        assert all(dry for _, dry, _ in configured)  # every post was dry-run

    def test_real_run_posts_two_comments(self, heads_up_module, configured):
        results = heads_up_module.run(
            repo="o/r",
            close=True,
            cutoff=dt.date(2026, 6, 1),
            model="m",
        )
        assert sum(1 for r in results if r["action"] == "heads-up-posted") == 2
        # Two real-run posts: one failing PR (#2), one failing issue (#102).
        real_posts = [n for n, dry, _ in configured if dry is False]
        assert sorted(real_posts) == [2, 102]

    def test_kinds_filter_skips_issues(self, heads_up_module, configured):
        results = heads_up_module.run(
            repo="o/r",
            close=False,
            cutoff=dt.date(2026, 6, 1),
            model="m",
            kinds=("pr",),
        )
        assert {r["kind"] for r in results} == {"pr"}

    def test_only_numbers_restricts_sweep(self, heads_up_module, configured):
        results = heads_up_module.run(
            repo="o/r",
            close=False,
            cutoff=dt.date(2026, 6, 1),
            model="m",
            only_numbers={"pr": [2], "issue": [101]},
        )
        assert sorted((r["kind"], r["number"]) for r in results) == [
            ("issue", 101),
            ("pr", 2),
        ]


class TestListOpenNumbersNoCap:
    """`_list_open_numbers` must sweep the WHOLE backlog, not a capped page.

    Regression guard: the rollout sweep is one-shot, so any item it misses
    here never gets a heads-up before the bot starts auto-closing.
    """

    def test_delegates_to_list_open_items_with_no_cap(
        self, heads_up_module, monkeypatch
    ):
        import agent_shin_shared

        captured: dict = {}

        def fake_gh(*args):
            captured["args"] = args
            return '[{"number": 5}, {"number": 9}]'

        monkeypatch.setattr(agent_shin_shared, "gh", fake_gh)
        numbers = heads_up_module._list_open_numbers("o/r", "issue")
        assert numbers == [5, 9]
        args = captured["args"]
        assert args[0] == "issue"
        assert args[args.index("--limit") + 1] == str(
            agent_shin_shared.GH_LIST_ALL_LIMIT
        )
        assert "1000" not in args
