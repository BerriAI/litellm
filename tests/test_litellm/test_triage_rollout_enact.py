"""Unit tests for the day-7 enactment sweep.

Covers:

  * ``_fake_now`` — the time-travel context manager. Patches
    ``triage_with_llm.dt`` so ``datetime.now()`` returns a pinned value,
    and restores the original module on exit (even on exception).
  * ``_apply_pr_result`` / ``_apply_issue_result`` — dispatch tables that
    turn a review_gate/triage verdict into one or two ``maybe_*`` calls.
    These are where the dry-run boolean actually reaches the side-effect
    wrappers, so a regression here breaks the entire enactment.
  * ``_process_one`` — per-item skip cases (not-open, internal author).
  * ``run`` — sweep loop end-to-end with stubbed evaluators, exercising
    both dry-run and real modes plus the time-travel offset.

Nothing in this file ever shells out — gh / openai / GitHub mutations are
stubbed end-to-end.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / ".github" / "scripts"


def _load(name: str, filename: str | None = None):
    spec = importlib.util.spec_from_file_location(
        name, _SCRIPTS_DIR / (filename or f"{name}.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def shared_module():
    return _load("agent_shin_shared")


@pytest.fixture(scope="module")
def triage_module(shared_module):
    return _load("triage_with_llm")


@pytest.fixture(scope="module")
def actions_module(triage_module):
    return _load("_agent_shin_actions")


@pytest.fixture(scope="module")
def enact_module(triage_module, actions_module, shared_module):
    return _load("triage_rollout_enact")


# ---------------------------------------------------------------------------
# _fake_now context manager


class TestFakeNow:
    def test_patches_dt_now_inside_context(self, enact_module, triage_module):
        when = dt.datetime(2099, 1, 1, tzinfo=dt.timezone.utc)
        with enact_module._fake_now(when):
            assert triage_module.dt.datetime.now(dt.timezone.utc) == when

    def test_restores_dt_on_exit(self, enact_module, triage_module):
        original = triage_module.dt
        with enact_module._fake_now(dt.datetime(2099, 1, 1, tzinfo=dt.timezone.utc)):
            pass
        assert triage_module.dt is original

    def test_restores_dt_on_exception(self, enact_module, triage_module):
        original = triage_module.dt
        with pytest.raises(RuntimeError):
            with enact_module._fake_now(
                dt.datetime(2099, 1, 1, tzinfo=dt.timezone.utc)
            ):
                raise RuntimeError("boom")
        assert triage_module.dt is original


# ---------------------------------------------------------------------------
# _apply_pr_result and _apply_issue_result


@pytest.fixture
def recorder(enact_module, monkeypatch):
    """Replace every maybe_* in the enactment module with a recorder so we
    can assert the exact sequence and dry-run flag of each side effect."""
    calls: list[dict] = []

    def make(name):
        def _stub(*args, dry_run=None, **kw):
            calls.append({"name": name, "args": args, "kw": kw, "dry_run": dry_run})

        return _stub

    for name in (
        "maybe_post_comment",
        "maybe_close_pr",
        "maybe_close_issue",
        "maybe_add_label",
        "maybe_remove_label",
    ):
        monkeypatch.setattr(enact_module, name, make(name))
    return calls


class TestApplyPRResult:
    def _apply(self, enact_module, action, *, dry_run, comment="body", **extra):
        return enact_module._apply_pr_result(
            repo="o/r",
            number=7,
            result={"action": action, "comment": comment, **extra},
            dry_run=dry_run,
        )

    def test_noop_passing_does_nothing(self, enact_module, recorder):
        r = self._apply(enact_module, "noop-passing", dry_run=True)
        assert r["result"] == "noop"
        assert recorder == []

    def test_skip_internal_author_is_noop(self, enact_module, recorder):
        r = self._apply(enact_module, "skip-internal-author", dry_run=True)
        assert r["result"] == "noop"
        assert recorder == []

    def test_skip_no_llm_key_records_unavailable(self, enact_module, recorder):
        r = self._apply(enact_module, "skip-no-llm-key", dry_run=True, error=None)
        assert r["result"] == "noop-llm-unavailable"
        assert recorder == []

    def test_would_label_ready_posts_then_labels(self, enact_module, recorder):
        r = self._apply(enact_module, "would-label-ready", dry_run=True)
        assert r["result"] == "labeled-ready"
        assert [c["name"] for c in recorder] == [
            "maybe_post_comment",
            "maybe_add_label",
        ]
        # Both maybe_* calls must carry the same dry_run boolean.
        assert all(c["dry_run"] is True for c in recorder)

    def test_would_close_real_run_posts_then_closes(self, enact_module, recorder):
        r = self._apply(enact_module, "would-close", dry_run=False)
        assert r["result"] == "closed"
        assert [c["name"] for c in recorder] == [
            "maybe_post_comment",
            "maybe_close_pr",
        ]
        assert all(c["dry_run"] is False for c in recorder)

    def test_would_remove_label_posts_and_removes(self, enact_module, recorder):
        r = self._apply(enact_module, "would-remove-label", dry_run=True)
        assert r["result"] == "label-removed-regressed"
        names = [c["name"] for c in recorder]
        assert names.count("maybe_remove_label") == 1
        assert names.count("maybe_post_comment") == 1

    def test_within_grace_only_warns_no_close(self, enact_module, recorder):
        r = self._apply(enact_module, "would-notify-within-grace", dry_run=False)
        assert r["result"] == "warned-within-grace"
        assert [c["name"] for c in recorder] == ["maybe_post_comment"]
        # Crucially, NO maybe_close_pr — the contributor still has time.
        assert not any(c["name"] == "maybe_close_pr" for c in recorder)

    def test_unknown_action_is_safe_noop(self, enact_module, recorder):
        r = self._apply(enact_module, "future-action-we-dont-handle", dry_run=False)
        assert r["result"] == "noop-unknown-action"
        assert recorder == []


class TestApplyIssueResult:
    def _apply(self, enact_module, action, *, dry_run, verdict=None):
        return enact_module._apply_issue_result(
            repo="o/r",
            number=42,
            result={
                "action": action,
                "verdict": verdict or {"missing": ["X"], "explanation": ""},
            },
            dry_run=dry_run,
        )

    def test_pass_llm_is_noop(self, enact_module, recorder):
        r = self._apply(enact_module, "pass-llm", dry_run=True)
        assert r["result"] == "noop"
        assert recorder == []

    def test_would_warn_grace_posts_warning(self, enact_module, recorder):
        r = self._apply(enact_module, "would-warn-grace", dry_run=True)
        assert r["result"] == "warned-within-grace"
        assert [c["name"] for c in recorder] == ["maybe_post_comment"]

    def test_in_grace_already_warned_is_noop(self, enact_module, recorder):
        r = self._apply(enact_module, "skip-in-grace-period", dry_run=False)
        assert r["result"] == "noop-already-warned"
        assert recorder == []

    def test_would_close_real_posts_then_closes(self, enact_module, recorder):
        r = self._apply(enact_module, "would-close", dry_run=False)
        assert r["result"] == "closed"
        names = [c["name"] for c in recorder]
        assert names == ["maybe_post_comment", "maybe_close_issue"]
        assert all(c["dry_run"] is False for c in recorder)


# ---------------------------------------------------------------------------
# _process_one — skip cases


class TestProcessOneSkips:
    @pytest.fixture
    def patched(self, enact_module, triage_module, monkeypatch):
        def install(*, item):
            monkeypatch.setattr(enact_module, "fetch_pr", lambda repo, n: item)
            monkeypatch.setattr(enact_module, "fetch_issue", lambda repo, n: item)
            monkeypatch.setattr(triage_module, "fetch_pr", lambda repo, n: item)
            monkeypatch.setattr(triage_module, "fetch_issue", lambda repo, n: item)

        return install

    def test_closed_pr_is_skipped(self, enact_module, patched):
        patched(item={"state": "closed", "user": {"login": "outside-dev"}})
        r = enact_module._process_one(
            repo="o/r",
            kind="pr",
            number=7,
            model="m",
            dry_run=True,
            current_time=dt.datetime.now(dt.timezone.utc),
        )
        assert r["result"] == "skip-not-open"

    def test_internal_author_is_skipped(self, enact_module, patched):
        patched(
            item={
                "state": "open",
                "user": {"login": "mateo-berri"},
                "author_association": "MEMBER",
            }
        )
        r = enact_module._process_one(
            repo="o/r",
            kind="pr",
            number=7,
            model="m",
            dry_run=True,
            current_time=dt.datetime.now(dt.timezone.utc),
        )
        assert r["result"] == "skip-internal-author"


# ---------------------------------------------------------------------------
# run() — end-to-end sweep with stubbed evaluators


class TestRun:
    @pytest.fixture
    def env(self, enact_module, triage_module, monkeypatch):
        posts: list[dict] = []

        def record(name):
            def _stub(*args, dry_run=None, **kw):
                posts.append({"name": name, "args": args, "kw": kw, "dry_run": dry_run})

            return _stub

        for name in (
            "maybe_post_comment",
            "maybe_close_pr",
            "maybe_close_issue",
            "maybe_add_label",
            "maybe_remove_label",
        ):
            monkeypatch.setattr(enact_module, name, record(name))

        def list_open(repo, kind):
            return [1, 2] if kind == "pr" else [101]

        monkeypatch.setattr(enact_module, "_list_open_numbers", list_open)

        def item(_repo, _n):
            return {
                "state": "open",
                "user": {"login": "outside-dev"},
                "author_association": "NONE",
            }

        monkeypatch.setattr(enact_module, "fetch_pr", item)
        monkeypatch.setattr(enact_module, "fetch_issue", item)
        monkeypatch.setattr(triage_module, "fetch_pr", item)
        monkeypatch.setattr(triage_module, "fetch_issue", item)

        # Pretend review_gate / triage have already been called and pass
        # back canned would-* verdicts that exercise both branches.
        seen_now: list[dt.datetime] = []

        def fake_evaluate_pr(*, repo, number, model, current_time, judge=None):
            seen_now.append(current_time)
            if number == 1:
                return {
                    "action": "would-label-ready",
                    "comment": "ready!",
                    "passing": True,
                }
            return {
                "action": "would-close",
                "comment": "closing!",
                "passing": False,
                "verdict": {"verdict": "fail", "missing": ["X"]},
            }

        def fake_evaluate_issue(*, repo, number, model, current_time, judge=None):
            seen_now.append(current_time)
            return {
                "action": "would-warn-grace",
                "verdict": {"verdict": "fail", "missing": ["repro"]},
            }

        monkeypatch.setattr(enact_module, "_evaluate_pr", fake_evaluate_pr)
        monkeypatch.setattr(enact_module, "_evaluate_issue", fake_evaluate_issue)
        return posts, seen_now

    def test_dry_run_passes_dry_true_through_to_wrappers(self, enact_module, env):
        posts, _ = env
        clock = dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc)
        enact_module.run(repo="o/r", close=False, model="m", current_time=clock)
        assert posts, "dry-run should still call wrappers (they self-gate)"
        assert all(p["dry_run"] is True for p in posts)

    def test_real_run_passes_dry_false_to_wrappers(self, enact_module, env):
        posts, _ = env
        clock = dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc)
        enact_module.run(repo="o/r", close=True, model="m", current_time=clock)
        assert all(p["dry_run"] is False for p in posts)

    def test_current_time_threads_through_to_evaluators(self, enact_module, env):
        _, seen_now = env
        clock = dt.datetime(2099, 1, 1, tzinfo=dt.timezone.utc)
        enact_module.run(repo="o/r", close=False, model="m", current_time=clock)
        # PRs and issue all evaluated against the same future clock.
        assert seen_now and all(t == clock for t in seen_now)

    def test_kinds_filter(self, enact_module, env):
        _, seen_now = env
        clock = dt.datetime.now(dt.timezone.utc)
        results = enact_module.run(
            repo="o/r",
            close=False,
            model="m",
            current_time=clock,
            kinds=("issue",),
        )
        assert {r["kind"] for r in results} == {"issue"}

    def test_only_numbers_restricts_sweep(self, enact_module, env):
        _, _ = env
        clock = dt.datetime.now(dt.timezone.utc)
        results = enact_module.run(
            repo="o/r",
            close=False,
            model="m",
            current_time=clock,
            only_numbers={"pr": [2]},
        )
        prs = [r for r in results if r["kind"] == "pr"]
        assert [r["number"] for r in prs] == [2]
