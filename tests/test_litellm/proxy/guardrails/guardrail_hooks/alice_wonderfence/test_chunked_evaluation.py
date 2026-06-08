"""Tests for the WonderFence-agnostic chunk + parallel-evaluate + aggregate unit."""

import asyncio
from unittest.mock import Mock

import pytest

from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence.chunked_evaluation import (
    MAX_PROMPT_CHARS,
    SegmentVerdict,
    _split_text,
    evaluate_segments,
)


def _result(action, action_text=None, detections=None, correlation_id=None):
    r = Mock()
    r.action = action
    r.action_text = action_text
    r.detections = detections or []
    r.correlation_id = correlation_id
    return r


# ----------------------------- _split_text -----------------------------


def test_split_short_text_is_single_chunk():
    assert _split_text("hello world", 10000) == ["hello world"]


def test_split_long_text_is_lossless():
    text = " ".join(f"word{i}" for i in range(5000))
    chunks = _split_text(text, 100)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks) == text


def test_split_force_splits_oversized_single_token():
    text = "x" * 250
    chunks = _split_text(text, 100)
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks) == text


# ----------------------------- evaluate_segments alignment -----------------------------


@pytest.mark.asyncio
async def test_verdicts_align_one_to_one_with_segments():
    actions = {"a": "BLOCK", "b": "MASK", "c": ""}

    async def evaluate(text):
        return _result(
            actions[text], action_text="[M]" if actions[text] == "MASK" else None
        )

    verdicts = await evaluate_segments(["a", "b", "c"], evaluate)
    assert [v.action for v in verdicts] == ["BLOCK", "MASK", ""]
    assert isinstance(verdicts[0], SegmentVerdict)


@pytest.mark.asyncio
async def test_mask_verdict_carries_masked_text():
    async def evaluate(text):
        return _result("MASK", action_text="[REDACTED]")

    verdicts = await evaluate_segments(["secret"], evaluate)
    assert verdicts[0].action == "MASK"
    assert verdicts[0].masked_text == "[REDACTED]"


# ----------------------------- chunking precedence -----------------------------


@pytest.mark.asyncio
async def test_block_in_non_first_chunk_blocks_whole_segment():
    """A segment split into chunks where only a later chunk trips BLOCK must
    still produce a BLOCK verdict; the old last-only path never saw earlier text."""
    segment = ("safe " * 30) + "TRIPWIRE"

    async def evaluate(text):
        return _result("BLOCK" if "TRIPWIRE" in text else "")

    verdicts = await evaluate_segments([segment], evaluate, max_chars=50)
    assert verdicts[0].action == "BLOCK"


@pytest.mark.asyncio
async def test_mask_rejoins_per_chunk_action_text_into_full_segment():
    segment = ("ab " * 60).strip()
    chunks = _split_text(segment, 50)
    assert len(chunks) > 1

    async def evaluate(text):
        return _result("MASK", action_text=f"<{text}>")

    verdicts = await evaluate_segments([segment], evaluate, max_chars=50)
    assert verdicts[0].action == "MASK"
    assert verdicts[0].masked_text == "".join(f"<{c}>" for c in chunks)


@pytest.mark.asyncio
async def test_unmasked_chunks_fall_back_to_original_text_on_rejoin():
    segment = " ".join(f"w{i}" for i in range(40))
    chunks = _split_text(segment, 20)
    assert len(chunks) > 1

    async def evaluate(text):
        return _result("MASK" if text == chunks[0] else "", action_text="[X]")

    verdicts = await evaluate_segments([segment], evaluate, max_chars=20)
    expected = "[X]" + "".join(chunks[1:])
    assert verdicts[0].masked_text == expected


@pytest.mark.asyncio
async def test_block_beats_mask_within_segment():
    chunks_seen = []

    async def evaluate(text):
        chunks_seen.append(text)
        return _result("BLOCK" if "B" in text else "MASK", action_text="[m]")

    segment = "aaa B"
    verdicts = await evaluate_segments([segment], evaluate, max_chars=2)
    assert verdicts[0].action == "BLOCK"


# ----------------------------- aggregation of detections/correlation ids -----------------------------


@pytest.mark.asyncio
async def test_block_verdict_aggregates_detections_and_correlation_ids():
    d1, d2 = Mock(), Mock()

    async def evaluate(text):
        if "x" in text:
            return _result("BLOCK", detections=[d1], correlation_id="c1")
        return _result("BLOCK", detections=[d2], correlation_id="c2")

    verdicts = await evaluate_segments(["x", "y"], evaluate)
    assert verdicts[0].detections == [d1]
    assert verdicts[0].correlation_ids == ["c1"]
    assert verdicts[1].correlation_ids == ["c2"]


# ----------------------------- concurrency cap -----------------------------


@pytest.mark.asyncio
async def test_evaluations_run_in_parallel_under_a_cap():
    state = {"current": 0, "max_seen": 0}

    async def evaluate(text):
        state["current"] += 1
        state["max_seen"] = max(state["max_seen"], state["current"])
        await asyncio.sleep(0.01)
        state["current"] -= 1
        return _result("")

    segments = [f"s{i}" for i in range(12)]
    await evaluate_segments(segments, evaluate, max_concurrency=3)
    assert state["max_seen"] > 1, "evaluations did not run concurrently"
    assert state["max_seen"] <= 3, "concurrency cap exceeded"


def test_max_prompt_chars_is_positive():
    assert isinstance(MAX_PROMPT_CHARS, int) and MAX_PROMPT_CHARS > 0
