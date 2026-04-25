"""Tests for the `compat_result` fixture's tagged-union validation.

The conftest's `pytest_runtest_makereport` hook is exercised end-to-end by
the matrix-builder golden-file tests (which consume a results.json that
the harness would produce). Here we just test the input-validation
contract on `CompatResult.set()`.
"""

from __future__ import annotations

import pytest

from tests.claude_code.conftest import CompatResult


def test_set_pass_is_accepted():
    r = CompatResult()
    r.set({"status": "pass"})
    assert r.value == {"status": "pass"}


def test_set_fail_requires_error():
    r = CompatResult()
    with pytest.raises(ValueError, match="requires 'error'"):
        r.set({"status": "fail"})


def test_set_fail_with_error_is_accepted():
    r = CompatResult()
    r.set({"status": "fail", "error": "boom"})
    assert r.value == {"status": "fail", "error": "boom"}


def test_set_not_applicable_requires_reason():
    r = CompatResult()
    with pytest.raises(ValueError, match="requires 'reason'"):
        r.set({"status": "not_applicable"})


def test_set_not_applicable_with_reason_is_accepted():
    r = CompatResult()
    r.set({"status": "not_applicable", "reason": "Bedrock has no /thinking"})
    assert r.value == {"status": "not_applicable", "reason": "Bedrock has no /thinking"}


def test_set_not_tested_is_accepted():
    r = CompatResult()
    r.set({"status": "not_tested"})
    assert r.value == {"status": "not_tested"}


def test_set_rejects_unknown_status():
    r = CompatResult()
    with pytest.raises(ValueError, match="status must be one of"):
        r.set({"status": "maybe"})


def test_set_rejects_non_dict():
    r = CompatResult()
    with pytest.raises(TypeError):
        r.set("pass")  # type: ignore[arg-type]


def test_set_copies_input():
    """Mutating the dict after set() must not change the stored value."""
    r = CompatResult()
    payload = {"status": "fail", "error": "x"}
    r.set(payload)
    payload["error"] = "mutated"
    assert r.value["error"] == "x"
