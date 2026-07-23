"""Tests for processing.py pure transforms: reconstruction, extractors, cap, verdict apply."""

import pytest

from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence.chunked_evaluation import (
    SegmentVerdict,
)
from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence.exceptions import (
    WonderFenceBlockedError,
    WonderFenceScanBudgetExceeded,
)
from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence.processing import (
    JOINER,
    apply_response_verdicts,
    check_scan_budget,
    function_definition_segments,
    reconstruct,
    tool_definition_segments,
)


def _block(detections=None, correlation_ids=None):
    return SegmentVerdict("BLOCK", None, detections or [], correlation_ids or [])


# --------------- reconstruct (masked-join alignment) ---------------


def test_reconstruct_no_change_round_trips():
    parts = ["alpha", "beta", "gamma"]
    assert reconstruct(parts, JOINER.join(parts)) == parts


def test_reconstruct_masks_a_middle_part():
    parts = ["alpha", "sensitive", "gamma"]
    masked = JOINER.join(["alpha", "[REDACTED]", "gamma"])
    assert reconstruct(parts, masked) == ["alpha", "[REDACTED]", "gamma"]


def test_reconstruct_mask_at_part_start():
    parts = ["alpha", "beta", "gamma"]
    masked = JOINER.join(["[X]lpha", "beta", "gamma"])
    assert reconstruct(parts, masked) == ["[X]lpha", "beta", "gamma"]


def test_reconstruct_handles_a_part_that_itself_contains_newline():
    """A message part can itself contain the joiner char; alignment is
    structural, not a naive split on '\\n', so this still reconstructs."""
    parts = ["line1\nline1b", "second"]
    masked = JOINER.join(["line1\n[REDACTED]", "second"])
    assert reconstruct(parts, masked) == ["line1\n[REDACTED]", "second"]


def test_reconstruct_fails_closed_when_mask_spans_a_joiner():
    """If the mask swallows a joiner (parts merged), reconstruction must fail
    closed (None) rather than misassign redacted text to the wrong message."""
    parts = ["alpha", "beta", "gamma"]
    merged = "alphaXXXbeta\ngamma"  # joiner between alpha|beta is gone
    assert reconstruct(parts, merged) is None


def test_reconstruct_empty_parts_is_empty_list():
    assert reconstruct([], "") == []


# --------------- check_scan_budget (total-work cap) ---------------


def test_check_scan_budget_passes_within_limits():
    check_scan_budget(["a", "b", "c"], max_scan_chars=100, max_scan_segments=100)


def test_check_scan_budget_rejects_too_many_segments():
    with pytest.raises(WonderFenceScanBudgetExceeded) as exc:
        check_scan_budget(["x"] * 11, max_scan_chars=10_000, max_scan_segments=10)
    assert exc.value.detail["limit"] == "max_scan_segments"
    assert exc.value.detail["max_scan_segments"] == 10


def test_check_scan_budget_rejects_too_many_chars():
    with pytest.raises(WonderFenceScanBudgetExceeded) as exc:
        check_scan_budget(["x" * 50, "y" * 60], max_scan_chars=100, max_scan_segments=100)
    assert exc.value.detail["limit"] == "max_scan_chars"
    assert exc.value.detail["chars"] == 110


def test_check_scan_budget_none_limits_disable_the_cap():
    check_scan_budget(["x" * 10_000] * 100, max_scan_chars=None, max_scan_segments=None)


# --------------- tool/function definition extractors (detection-only, list of texts) ---------------


def test_tool_definition_segments_extracts_description_and_param_descriptions():
    inputs = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "TOP_DESC",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string", "description": "PARAM_DESC"}},
                    },
                },
            }
        ]
    }
    assert set(tool_definition_segments(inputs)) == {"TOP_DESC", "PARAM_DESC"}


def test_tool_definition_segments_ignores_non_dict_tools_and_blank_descriptions():
    inputs = {
        "tools": [
            "not-a-dict",
            {"type": "function", "function": {"name": "f", "description": "   "}},
            {"type": "function"},
        ]
    }
    assert tool_definition_segments(inputs) == []


def test_function_definition_segments_extracts_descriptions():
    request_data = {
        "functions": [
            {
                "name": "weather",
                "description": "TOP_DESC",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "description": "PARAM_DESC"}},
                },
            },
            "not-a-dict",
            {"name": "f", "description": "   "},
        ]
    }
    assert set(function_definition_segments(request_data)) == {"TOP_DESC", "PARAM_DESC"}


def test_function_definition_segments_empty_when_absent():
    assert function_definition_segments({"model": "gpt-4"}) == []


# --------------- apply_response_verdicts ---------------


def test_response_block_verdict_raises_with_aggregated_detections():
    inputs = {"texts": ["bad", "ok"]}
    d = {"policy_name": "p"}
    verdicts = [_block(detections=[d], correlation_ids=["c1"]), SegmentVerdict("", None, [], [])]
    with pytest.raises(WonderFenceBlockedError) as exc:
        apply_response_verdicts(inputs, verdicts, [], [], "gn", "blocked!")
    assert exc.value.detail["error"] == "blocked!"
    assert exc.value.detail["action"] == "BLOCK"
    assert exc.value.detail["detections"] == [d]
    assert exc.value.detail["wonderfence_correlation_id"] == "c1"


def test_response_mask_writes_to_the_mapped_text_index_only():
    inputs = {"texts": ["keep", "MASK_ME", "keep2"]}
    verdicts = [
        SegmentVerdict("", None, [], []),
        SegmentVerdict("MASK", "[R]", [], []),
        SegmentVerdict("", None, [], []),
    ]
    out = apply_response_verdicts(inputs, verdicts, [], [], "gn", "blocked!")
    assert out["texts"] == ["keep", "[R]", "keep2"]


def test_response_mask_writes_tool_call_arguments_in_place():
    inputs = {
        "texts": ["ok"],
        "tool_calls": [{"function": {"arguments": '{"x": "secret"}'}}],
    }
    out = apply_response_verdicts(
        inputs,
        [SegmentVerdict("", None, [], [])],
        [0],
        [SegmentVerdict("MASK", '{"x": "[R]"}', [], [])],
        "gn",
        "blocked!",
    )
    assert out["tool_calls"][0]["function"]["arguments"] == '{"x": "[R]"}'


def test_response_detect_and_no_action_leave_texts_unchanged():
    inputs = {"texts": ["a", "b"]}
    verdicts = [SegmentVerdict("DETECT", None, [], []), SegmentVerdict("", None, [], [])]
    out = apply_response_verdicts(inputs, verdicts, [], [], "gn", "blocked!")
    assert out["texts"] == ["a", "b"]
