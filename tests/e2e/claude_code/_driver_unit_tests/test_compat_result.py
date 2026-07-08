"""Tests for the `compat_result` fixture's tagged-union validation.

The conftest's `pytest_runtest_makereport` hook is exercised end-to-end by
the matrix-builder golden-file tests (which consume a results.json that
the harness would produce). Here we just test the input-validation
contract on `CompatResult.set()`.
"""

from __future__ import annotations

import pytest

from claude_code.conftest import CompatResult


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


# ---------------------------------------------------------------------------
# add() / collected()
#
# When a single test exercises three Claude tiers in parallel, each tier
# needs its own row in the results artifact so the matrix builder can
# apply its "all three must pass" aggregation. `add()` is the per-tier
# recorder; `collected()` is what the conftest hook reads.
# ---------------------------------------------------------------------------


def test_add_appends_each_call_to_values():
    r = CompatResult()
    r.add({"status": "pass"})
    r.add({"status": "fail", "error": "bad"})
    assert r.values == [
        {"status": "pass"},
        {"status": "fail", "error": "bad"},
    ]


def test_add_validates_like_set():
    """The add() and set() validators are the same; both must reject bad payloads."""
    r = CompatResult()
    with pytest.raises(ValueError, match="requires 'error'"):
        r.add({"status": "fail"})
    with pytest.raises(ValueError, match="requires 'reason'"):
        r.add({"status": "not_applicable"})
    with pytest.raises(ValueError, match="status must be one of"):
        r.add({"status": "maybe"})
    with pytest.raises(TypeError):
        r.add("pass")  # type: ignore[arg-type]


def test_add_copies_input():
    """Same defensive copy contract as set()."""
    r = CompatResult()
    payload = {"status": "fail", "error": "x"}
    r.add(payload)
    payload["error"] = "mutated"
    assert r.values[0]["error"] == "x"


def test_collected_returns_values_when_added():
    r = CompatResult()
    r.add({"status": "pass"})
    r.add({"status": "pass"})
    assert r.collected() == [{"status": "pass"}, {"status": "pass"}]


def test_collected_returns_single_value_when_only_set_called():
    """Legacy single-result tests should still surface their one outcome."""
    r = CompatResult()
    r.set({"status": "pass"})
    assert r.collected() == [{"status": "pass"}]


def test_collected_prefers_added_values_over_set_value():
    """If both are populated, the per-tier list wins — that's the multi-model shape."""
    r = CompatResult()
    r.set({"status": "pass"})
    r.add({"status": "fail", "error": "tier-2 broke"})
    assert r.collected() == [{"status": "fail", "error": "tier-2 broke"}]


def test_collected_returns_empty_when_nothing_reported():
    assert CompatResult().collected() == []
