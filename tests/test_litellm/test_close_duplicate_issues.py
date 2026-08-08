"""Unit tests for `.github/scripts/close_duplicate_issues.py`."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "close_duplicate_issues.py"


@pytest.fixture(scope="module")
def duplicate_issues_module():
    spec = importlib.util.spec_from_file_location("close_duplicate_issues", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fetch_open_issues_preserves_unicode_line_separators(duplicate_issues_module, monkeypatch):
    raw = '[{"number": 1, "body": "line\u2028separator"}, {"number": 2, "body": "paragraph\u2029separator"}]'
    monkeypatch.setattr(duplicate_issues_module, "gh", lambda *args: raw)

    assert duplicate_issues_module.fetch_open_issues("BerriAI/litellm") == [
        {"number": 1, "body": "line\u2028separator"},
        {"number": 2, "body": "paragraph\u2029separator"},
    ]
