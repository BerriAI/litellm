"""Unit tests for `.github/scripts/close_duplicate_issues.py`."""

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
def dedupe_module():
    spec = importlib.util.spec_from_file_location("close_duplicate_issues", SCRIPT_PATH)
    assert spec and spec.loader, f"Could not load spec for {SCRIPT_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["close_duplicate_issues"] = module
    spec.loader.exec_module(module)
    return module


def test_parse_concatenated_json_joins_paginated_arrays(dedupe_module):
    page_one = json.dumps([{"number": 1, "title": "a"}, {"number": 2, "title": "b"}])
    page_two = json.dumps([{"number": 3, "title": "c"}])
    issues = dedupe_module.parse_concatenated_json(page_one + page_two)
    assert [i["number"] for i in issues] == [1, 2, 3]


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\x85"])
def test_parse_concatenated_json_survives_unicode_line_breaks_in_bodies(
    dedupe_module, separator
):
    body = f"first{separator}second"
    page_one = json.dumps([{"number": 1, "title": "a", "body": body}])
    page_two = json.dumps([{"number": 2, "title": "b", "body": "plain"}])
    issues = dedupe_module.parse_concatenated_json(page_one + page_two)
    assert [i["number"] for i in issues] == [1, 2]
    assert issues[0]["body"] == body
