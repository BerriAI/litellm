"""Tests for the duplicate-issue close script.

Run with: python -m pytest .github/scripts/test_close_duplicate_issues.py
"""

import json
import os
import sys
import types
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(__file__))

import close_duplicate_issues as mod


def _make_completed_process(stdout: str):
    return types.SimpleNamespace(stdout=stdout, returncode=0, stderr="")


def test_fetch_open_issues_handles_unicode_line_separator():
    """Regression: U+2028 / U+2029 inside JSON strings must not break parsing.

    Python's str.splitlines() treats these characters as line breaks, but they
    are valid inside JSON string values without escaping, so any line-based
    parsing of gh's compact JSON output would split mid-string.
    """
    issues = [
        {
            "number": 1,
            "title": "title with line\u2028separator",
            "body": "paragraph\u2029separator and line\u2028break",
        },
        {"number": 2, "title": "plain", "body": "plain"},
    ]
    raw = json.dumps(issues, ensure_ascii=False)
    assert "\u2028" in raw and "\u2029" in raw

    with patch.object(mod.subprocess, "run", return_value=_make_completed_process(raw)):
        result = mod.fetch_open_issues("owner/repo")

    assert [i["number"] for i in result] == [1, 2]
    assert result[0]["title"] == "title with line\u2028separator"


def test_fetch_open_issues_filters_pull_requests():
    payload = [
        {"number": 1, "title": "real issue"},
        {"number": 2, "title": "a PR", "pull_request": {"url": "..."}},
    ]
    raw = json.dumps(payload)

    with patch.object(mod.subprocess, "run", return_value=_make_completed_process(raw)):
        result = mod.fetch_open_issues("owner/repo")

    assert [i["number"] for i in result] == [1]


def test_fetch_open_issues_empty_response():
    with patch.object(mod.subprocess, "run", return_value=_make_completed_process("")):
        assert mod.fetch_open_issues("owner/repo") == []

    with patch.object(
        mod.subprocess, "run", return_value=_make_completed_process("[]")
    ):
        assert mod.fetch_open_issues("owner/repo") == []


def test_find_duplicate_normalizes_prefixes_and_case():
    target = {"number": 10, "title": "[Bug] Crash on startup"}
    older = [
        {"number": 1, "title": "feature request: add streaming"},
        {"number": 2, "title": "crash on startup"},
    ]
    dup = mod.find_duplicate(target, older, threshold=0.85)
    assert dup is not None
    assert dup["number"] == 2


def test_find_duplicate_returns_none_below_threshold():
    target = {"number": 10, "title": "Streaming hangs on Anthropic"}
    older = [{"number": 1, "title": "Completely unrelated topic"}]
    assert mod.find_duplicate(target, older, threshold=0.85) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
