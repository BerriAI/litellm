"""Tests for the WonderFence-agnostic chunk + parallel-evaluate + aggregate unit."""

import asyncio
from unittest.mock import Mock

import pytest

from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence.chunked_evaluation import (
    MAX_PROMPT_CHARS,
    SegmentVerdict,
    WindowConfig,
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
        return _result(actions[text], action_text="[M]" if actions[text] == "MASK" else None)

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
async def test_mask_rejoins_masked_owned_regions_into_full_segment():
    """A multi-chunk segment where the service redacts a token in one chunk (a
    real span substitution that preserves surrounding bytes) rejoins into the
    fully masked segment. overlap=0 keeps the chunks disjoint for a clean check;
    the per-chunk (original, masked) pairs are carried on the verdict."""
    segment = " ".join(f"w{i}" for i in range(40)) + " SECRET " + " ".join(f"v{i}" for i in range(40))

    async def evaluate(text):
        return _result("MASK", action_text=text.replace("SECRET", "[X]")) if "SECRET" in text else _result("")

    chunks = _split_text(segment, 20)
    assert len(chunks) > 1
    verdicts = await evaluate_segments([segment], evaluate, max_chars=20, windows=WindowConfig(overlap=0))
    assert verdicts[0].action == "MASK"
    assert verdicts[0].masked_text == segment.replace("SECRET", "[X]")
    assert verdicts[0].masked_chunks is not None
    assert "".join(o for o, _ in verdicts[0].masked_chunks) == segment


@pytest.mark.asyncio
async def test_unmasked_chunks_keep_original_text_on_rejoin():
    """Chunks the service did not mask contribute their original owned text
    verbatim; only the masked chunk changes."""
    segment = " ".join(f"w{i}" for i in range(40))

    async def evaluate(text):
        return _result("MASK", action_text=text.replace("w0", "[X]")) if "w0 " in text else _result("")

    verdicts = await evaluate_segments([segment], evaluate, max_chars=20, windows=WindowConfig(overlap=0))
    assert verdicts[0].action == "MASK"
    assert verdicts[0].masked_text == segment.replace("w0", "[X]", 1)


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


# ----------------------------- seam overlap (detection / masking across chunk splits) -----------------------------


@pytest.mark.asyncio
async def test_block_phrase_split_across_chunk_seam_is_detected():
    """A blocked phrase straddling an owned-region seam is caught because the
    next chunk carries an overlap prefix from the previous owned region, so one
    scan sees the phrase whole even though neither disjoint owned region does."""
    segment = "aaaaa BLOCK ME zzzzz"
    chunks = _split_text(segment, 12)
    assert len(chunks) > 1
    assert all("BLOCK ME" not in c for c in chunks)

    async def evaluate(text):
        return _result("BLOCK" if "BLOCK ME" in text else "")

    verdicts = await evaluate_segments([segment], evaluate, max_chars=12, windows=WindowConfig(overlap=6))
    assert verdicts[0].action == "BLOCK"


@pytest.mark.asyncio
async def test_no_overlap_lets_seam_phrase_evade():
    """Control: with overlap disabled there is no prefix, so the same straddling
    phrase is seen by neither owned region -- demonstrating what the overlap closes."""
    segment = "aaaaa BLOCK ME zzzzz"

    async def evaluate(text):
        return _result("BLOCK" if "BLOCK ME" in text else "")

    verdicts = await evaluate_segments([segment], evaluate, max_chars=12, windows=WindowConfig(overlap=0))
    assert verdicts[0].action == ""


@pytest.mark.asyncio
async def test_single_chunk_segment_evaluates_once():
    calls = []

    async def evaluate(text):
        calls.append(text)
        return _result("")

    await evaluate_segments(["short benign text"], evaluate, max_chars=10000)
    assert calls == ["short benign text"]


@pytest.mark.asyncio
async def test_mask_straddling_a_seam_fails_closed_as_block():
    """A MASK whose redaction reaches into a chunk's overlap prefix (content
    straddling, or within `overlap` of, an owned-region seam) cannot be stitched
    without double-counting the overlap, so it fails closed as BLOCK rather than
    leak the un-redacted half. This is the preempt for the seam-mask leak."""
    segment = "aaaaa SECRET HERE zzzzz"
    chunks = _split_text(segment, 12)
    assert len(chunks) > 1

    async def evaluate(text):
        # The chunk that sees "SECRET HERE" whole (via its overlap prefix) masks
        # it; the redaction lands in the prefix bytes -> fail closed.
        return _result("MASK", action_text=text.replace("SECRET HERE", "[X]")) if "SECRET HERE" in text else _result("")

    verdicts = await evaluate_segments([segment], evaluate, max_chars=12, windows=WindowConfig(overlap=6))
    assert verdicts[0].action == "BLOCK"


@pytest.mark.asyncio
async def test_independent_segments_are_not_concatenated():
    """Segments are scanned independently (no cross-segment window): a phrase
    split across two segments is NOT joined. On the request side, message parts
    are joined into one document *before* reaching here; the response side has
    independent choices that the model never concatenates."""

    async def evaluate(text):
        return _result("BLOCK" if "BLOCKME" in text else "")

    verdicts = await evaluate_segments(["BLOCK", "ME"], evaluate)
    assert [v.action for v in verdicts] == ["", ""]
