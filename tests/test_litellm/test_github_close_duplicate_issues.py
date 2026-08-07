"""Unit tests for `.github/scripts/close_duplicate_issues.py`.

These exercise the pure logic (issue fetching and duplicate detection) without
hitting GitHub. The `gh` CLI wrapper is stubbed via monkeypatch.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github"
    / "scripts"
    / "close_duplicate_issues.py"
)


@pytest.fixture(scope="module")
def dup_module():
    """Load the script as a module via its file path (it lives outside the package)."""
    spec = importlib.util.spec_from_file_location("close_duplicate_issues", SCRIPT_PATH)
    assert spec and spec.loader, f"Could not load spec for {SCRIPT_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["close_duplicate_issues"] = module
    spec.loader.exec_module(module)
    return module


class TestFetchOpenIssues:
    def test_should_flatten_all_pages_from_slurp_output(self, dup_module, monkeypatch):
        pages = [
            [{"number": 1, "title": "a"}, {"number": 2, "title": "b"}],
            [{"number": 3, "title": "c"}],
        ]
        monkeypatch.setattr(dup_module, "gh", lambda *a: json.dumps(pages))
        issues = dup_module.fetch_open_issues("owner/repo")
        assert [i["number"] for i in issues] == [1, 2, 3]

    def test_should_exclude_pull_requests(self, dup_module, monkeypatch):
        pages = [
            [
                {"number": 1, "title": "real issue"},
                {"number": 2, "title": "a PR", "pull_request": {"url": "x"}},
            ]
        ]
        monkeypatch.setattr(dup_module, "gh", lambda *a: json.dumps(pages))
        issues = dup_module.fetch_open_issues("owner/repo")
        assert [i["number"] for i in issues] == [1]

    def test_should_pass_slurp_flag_to_gh(self, dup_module, monkeypatch):
        captured: dict = {}

        def fake_gh(*args):
            captured["args"] = args
            return json.dumps([[]])

        monkeypatch.setattr(dup_module, "gh", fake_gh)
        dup_module.fetch_open_issues("owner/repo")
        assert "--paginate" in captured["args"]
        assert "--slurp" in captured["args"]

    @pytest.mark.parametrize("sep", ["\u2028", "\u2029", "\x85"])
    def test_should_not_break_on_unicode_line_separators_in_body(
        self, dup_module, monkeypatch, sep
    ):
        # Regression for #32639: gh emits the GitHub API JSON with Unicode
        # line separators (\u2028, \u2029, \x85) left literal inside issue
        # bodies. The old parser split the payload on str.splitlines(), which
        # breaks on those code points, fragmenting the otherwise-valid JSON
        # array and raising "Unterminated string" from json.loads.
        pages = [
            [
                {"number": 1, "title": "keep me", "body": f"line one{sep}line two"},
                {"number": 2, "title": "keep me too", "body": "plain"},
            ]
        ]
        raw = json.dumps(pages, ensure_ascii=False)
        assert sep in raw
        assert len(raw.splitlines()) > 1
        assert json.loads(raw) == pages
        monkeypatch.setattr(dup_module, "gh", lambda *a: raw)
        issues = dup_module.fetch_open_issues("owner/repo")
        assert [i["number"] for i in issues] == [1, 2]
        assert issues[0]["body"] == f"line one{sep}line two"


class TestFindDuplicate:
    def test_should_match_titles_above_threshold_ignoring_bug_prefix(self, dup_module):
        issue = {"number": 10, "title": "[Bug]: proxy crashes on startup"}
        candidates = [{"number": 3, "title": "Proxy crashes on startup"}]
        dup = dup_module.find_duplicate(issue, candidates, threshold=0.85)
        assert dup is not None and dup["number"] == 3

    def test_should_not_match_below_threshold(self, dup_module):
        issue = {"number": 10, "title": "streaming responses are truncated"}
        candidates = [{"number": 3, "title": "add support for new provider"}]
        assert dup_module.find_duplicate(issue, candidates, threshold=0.85) is None

    def test_should_skip_self(self, dup_module):
        issue = {"number": 10, "title": "same title"}
        candidates = [{"number": 10, "title": "same title"}]
        assert dup_module.find_duplicate(issue, candidates, threshold=0.85) is None
