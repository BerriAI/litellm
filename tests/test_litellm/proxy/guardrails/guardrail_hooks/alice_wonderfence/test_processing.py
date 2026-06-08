"""Tests for processing.py pure transforms: user-text mapping and verdict apply."""

import pytest

from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence.chunked_evaluation import (
    SegmentVerdict,
)
from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence.exceptions import (
    WonderFenceBlockedError,
)
from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence.processing import (
    apply_verdicts,
    request_user_text_indices,
)

# ----------------------------- request_user_text_indices -----------------------------


def test_only_user_string_messages_are_indexed():
    messages = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    assert request_user_text_indices(messages, ["a", "b", "c"]) == [0, 2]


def test_system_message_excluded_even_when_present_in_texts():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    assert request_user_text_indices(messages, ["sys", "hi"]) == [1]


def test_list_content_yields_one_index_per_text_part():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "x"},
                {"type": "image_url", "image_url": {"url": "http://img"}},
                {"type": "text", "text": "y"},
            ],
        },
    ]
    # texts flattens to the two text parts (image contributes no text entry)
    assert request_user_text_indices(messages, ["x", "y"]) == [0, 1]


def test_absent_structured_messages_scans_all_indices():
    assert request_user_text_indices(None, ["a", "b", "c"]) == [0, 1, 2]


def test_count_mismatch_falls_back_to_scanning_all():
    """If the replayed flatten count diverges from len(texts), over-scan rather
    than risk mis-mapping a mask onto the wrong slot."""
    messages = [{"role": "user", "content": "a"}]
    assert request_user_text_indices(messages, ["a", "b"]) == [0, 1]


# ----------------------------- apply_verdicts -----------------------------


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
