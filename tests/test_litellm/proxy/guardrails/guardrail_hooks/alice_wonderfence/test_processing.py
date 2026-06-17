"""Tests for processing.py pure transforms: verdict apply."""

import pytest

from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence.chunked_evaluation import (
    SegmentVerdict,
)
from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence.exceptions import (
    WonderFenceBlockedError,
)
from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence.processing import (
    apply_verdicts,
)


def _block(detections=None, correlation_ids=None):
    return SegmentVerdict("BLOCK", None, detections or [], correlation_ids or [])


def test_block_verdict_raises_with_aggregated_detections():
    inputs = {"texts": ["bad", "ok"]}
    d = {"policy_name": "p"}
    verdicts = [
        _block(detections=[d], correlation_ids=["c1"]),
        SegmentVerdict("", None, [], []),
    ]
    with pytest.raises(WonderFenceBlockedError) as exc:
        apply_verdicts(inputs, [0, 1], verdicts, "gn", "blocked!")
    assert exc.value.detail["error"] == "blocked!"
    assert exc.value.detail["action"] == "BLOCK"
    assert exc.value.detail["detections"] == [d]
    assert exc.value.detail["wonderfence_correlation_id"] == "c1"


def test_mask_writes_to_the_mapped_text_index_only():
    inputs = {"texts": ["keep", "MASK_ME", "keep2"]}
    verdicts = [SegmentVerdict("MASK", "[R]", [], [])]
    out = apply_verdicts(inputs, [1], verdicts, "gn", "blocked!")
    assert out["texts"] == ["keep", "[R]", "keep2"]


def test_detect_and_no_action_leave_texts_unchanged():
    inputs = {"texts": ["a", "b"]}
    verdicts = [
        SegmentVerdict("DETECT", None, [], []),
        SegmentVerdict("", None, [], []),
    ]
    out = apply_verdicts(inputs, [0, 1], verdicts, "gn", "blocked!")
    assert out["texts"] == ["a", "b"]


# --------------- tool_definition_segments ---------------


def test_tool_definition_segments_extracts_description_and_param_descriptions():
    from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence.processing import (
        _set_by_path,
        tool_definition_segments,
    )

    inputs = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "TOP_DESC",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "PARAM_DESC"}
                        },
                    },
                },
            }
        ]
    }
    paths, segments = tool_definition_segments(inputs)
    assert set(segments) == {"TOP_DESC", "PARAM_DESC"}
    # each path round-trips: writing via the path updates the right slot
    for path, text in zip(paths, segments):
        _set_by_path(inputs["tools"], path, f"<{text}>")
    fn = inputs["tools"][0]["function"]
    assert fn["description"] == "<TOP_DESC>"
    assert fn["parameters"]["properties"]["city"]["description"] == "<PARAM_DESC>"


def test_tool_definition_segments_ignores_non_dict_tools_and_blank_descriptions():
    inputs = {
        "tools": [
            "not-a-dict",
            {"type": "function", "function": {"name": "f", "description": "   "}},
            {"type": "function"},
        ]
    }
    from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence.processing import (
        tool_definition_segments,
    )

    paths, segments = tool_definition_segments(inputs)
    assert segments == []
