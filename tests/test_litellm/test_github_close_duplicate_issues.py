"""Unit tests for `.github/scripts/close_duplicate_issues.py`.

Focused on the parsing/matching pure logic. Network/CLI calls are stubbed via
dependency injection where needed.
"""

from __future__ import annotations

import importlib.util
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
def dedup_module():
    spec = importlib.util.spec_from_file_location("close_duplicate_issues", SCRIPT_PATH)
    assert spec and spec.loader, f"Could not load spec for {SCRIPT_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["close_duplicate_issues"] = module
    spec.loader.exec_module(module)
    return module


class TestParsePaginatedIssues:
    """`parse_paginated_issues` must handle the newline-delimited JSON that
    `gh api --paginate --jq '.[] | ...'` emits. The prior implementation split
    on newlines and then tried to `json.loads` a single 15MB concatenated blob,
    which failed intermittently in CI with `Unterminated string...` when the
    payload was truncated or the arrays were not merged into a single one.
    Regression: parsing must be per-line and must skip PRs.
    """

    def test_parses_newline_delimited_json_and_filters_prs(self, dedup_module):
        raw = (
            '{"number": 1, "title": "issue A", "is_pr": false}\n'
            '{"number": 2, "title": "PR B", "is_pr": true}\n'
            '{"number": 3, "title": "issue C", "is_pr": false}\n'
        )
        result = dedup_module.parse_paginated_issues(raw)
        assert [i["number"] for i in result] == [1, 3]
        assert all(not i["is_pr"] for i in result)

    def test_skips_blank_and_whitespace_only_lines(self, dedup_module):
        raw = (
            "\n"
            '{"number": 10, "title": "first", "is_pr": false}\n'
            "   \n"
            "\t\n"
            '{"number": 11, "title": "second", "is_pr": false}\n'
        )
        result = dedup_module.parse_paginated_issues(raw)
        assert [i["number"] for i in result] == [10, 11]

    def test_returns_empty_list_for_empty_input(self, dedup_module):
        assert dedup_module.parse_paginated_issues("") == []
        assert dedup_module.parse_paginated_issues("   \n\n") == []

    def test_preserves_unicode_titles(self, dedup_module):
        raw = (
            '{"number": 42, "title": "Ünïcödé\\u2014 issue", "is_pr": false}\n'
            '{"number": 43, "title": "second — em dash", "is_pr": false}\n'
        )
        result = dedup_module.parse_paginated_issues(raw)
        assert result[0]["title"] == "Ünïcödé— issue"
        assert result[1]["title"] == "second — em dash"

    def test_parses_scale_output_line_by_line(self, dedup_module):
        raw = "\n".join(
            f'{{"number": {n}, "title": "issue {n}", "is_pr": {"true" if n % 5 == 0 else "false"}}}'
            for n in range(1, 501)
        )
        result = dedup_module.parse_paginated_issues(raw)
        assert len(result) == 400
        assert result[0]["number"] == 1
        assert result[-1]["number"] == 499

    def test_broken_line_surfaces_a_clear_error_not_a_silent_dropped_record(
        self, dedup_module
    ):
        raw = (
            '{"number": 1, "title": "ok", "is_pr": false}\n'
            '{"number": 2, "title": "trunc\n'
        )
        import json as _json

        with pytest.raises(_json.JSONDecodeError):
            dedup_module.parse_paginated_issues(raw)


class TestFindDuplicate:
    def _issue(self, number: int, title: str) -> dict:
        return {"number": number, "title": title, "is_pr": False}

    def test_returns_first_candidate_above_threshold(self, dedup_module):
        target = self._issue(100, "Proxy hangs on Redis timeout")
        candidates = [
            self._issue(50, "Something totally unrelated"),
            self._issue(60, "Proxy hangs on Redis timeout"),
            self._issue(70, "Different bug"),
        ]
        dup = dedup_module.find_duplicate(target, candidates, threshold=0.85)
        assert dup is not None
        assert dup["number"] == 60

    def test_ignores_self(self, dedup_module):
        target = self._issue(1, "identical title")
        candidates = [self._issue(1, "identical title")]
        assert dedup_module.find_duplicate(target, candidates, threshold=0.85) is None

    def test_returns_none_when_no_candidate_above_threshold(self, dedup_module):
        target = self._issue(10, "Some very unique title")
        candidates = [
            self._issue(1, "Totally different topic here"),
            self._issue(2, "Nothing alike whatsoever"),
        ]
        assert dedup_module.find_duplicate(target, candidates, threshold=0.85) is None

    def test_normalization_strips_bug_prefix(self, dedup_module):
        target = self._issue(10, "[Bug]: Auth flow broken")
        candidates = [self._issue(5, "auth flow broken")]
        dup = dedup_module.find_duplicate(target, candidates, threshold=0.85)
        assert dup is not None
        assert dup["number"] == 5


class TestNormalizeTitle:
    def test_strips_common_prefixes(self, dedup_module):
        cases = [
            ("Bug: broken", "broken"),
            ("[Feature Request] add flag", "add flag"),
            ("[Docs] fix typo", "fix typo"),
            ("Enhancement: add opt", "add opt"),
            ("Question: how do I", "how do i"),
        ]
        for raw, expected in cases:
            assert dedup_module.normalize_title(raw) == expected, f"input={raw!r}"

    def test_collapses_whitespace_and_lowercases(self, dedup_module):
        assert (
            dedup_module.normalize_title("  MULTI   space\t\tTitle ")
            == "multi space title"
        )
