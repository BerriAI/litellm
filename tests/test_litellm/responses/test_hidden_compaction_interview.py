"""
Hidden test set for the /v1/responses compaction V0 interview exercise.
NOT included in the candidate's starter branch.
Run against the candidate's submitted implementation.
"""
import json
import os
from typing import Any, Dict

import pytest

import litellm
from litellm.types.llms.openai import ResponsesAPIResponse

# ---------------------------------------------------------------------------
# T0 — Baseline: no context_management param → normal response shape
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_baseline_no_compaction_param():
    """Baseline: no context_management → normal response, no compaction fields, output[0] is a message"""
    response = await litellm.aresponses(
        model="groq/llama-3.3-70b-versatile",
        input=[
            {"type": "message", "role": "user", "content": "What is the capital of France?"},
        ],
        store=False,
        max_output_tokens=50,
    )
    assert response is not None
    assert response.output is not None
    assert len(response.output) > 0
    first_item = response.output[0]
    first_type = first_item.get("type") if isinstance(first_item, dict) else getattr(first_item, "type", None)
    assert first_type == "message", f"Expected first output item to be 'message', got '{first_type}'"
    assert not any(
        (item.get("type") if isinstance(item, dict) else getattr(item, "type", None)) == "compaction"
        for item in response.output
    )


# ---------------------------------------------------------------------------
# T1 — No-op when conversation is under the compact_threshold (Case 1)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compaction_noop_under_threshold():
    """Case 1: small conversation + high threshold → no compaction item in output"""
    response = await litellm.aresponses(
        model="groq/llama-3.3-70b-versatile",
        input=[
            {"type": "message", "role": "user", "content": "What is 2+2?"},
        ],
        context_management=[{"type": "compaction", "compact_threshold": 200000}],
        store=False,
        max_output_tokens=50,
    )
    assert response is not None
    output_types = [
        item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        for item in response.output
    ]
    assert "compaction" not in output_types, (
        "Compaction item should NOT appear when conversation is under the threshold"
    )
    assert "message" in output_types


# ---------------------------------------------------------------------------
# T2 — Compaction triggers when over threshold (Case 2)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compaction_triggers_over_threshold():
    """Case 2: large conversation, low threshold → compaction item with encrypted_content in output"""
    fat_message = "word " * 5000
    response = await litellm.aresponses(
        model="groq/llama-3.3-70b-versatile",
        input=[
            {"type": "message", "role": "user", "content": "Start of conversation."},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": fat_message}],
            },
            {"type": "message", "role": "user", "content": "Now answer: what is 1+1?"},
        ],
        context_management=[{"type": "compaction", "compact_threshold": 100}],
        store=False,
        max_output_tokens=50,
    )
    assert response is not None
    output_by_type: Dict[str, Any] = {}
    for item in response.output:
        t = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        if t and t not in output_by_type:
            output_by_type[t] = item

    assert "compaction" in output_by_type, "Compaction item missing from output"
    cmp_item = output_by_type["compaction"]
    enc = cmp_item.get("encrypted_content") if isinstance(cmp_item, dict) else getattr(cmp_item, "encrypted_content", None)
    assert enc, "Compaction item must have non-empty encrypted_content"
    cmp_id = cmp_item.get("id") if isinstance(cmp_item, dict) else getattr(cmp_item, "id", None)
    assert cmp_id, "Compaction item must have an id"
    assert "message" in output_by_type, "Model response missing from output"


# ---------------------------------------------------------------------------
# T2b — Compaction item is FIRST in output array (ordering)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compaction_item_is_first_in_output():
    """Compaction item must be output[0]; model message must follow"""
    fat_message = "word " * 5000
    response = await litellm.aresponses(
        model="groq/llama-3.3-70b-versatile",
        input=[
            {"type": "message", "role": "user", "content": "Start."},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": fat_message}],
            },
            {"type": "message", "role": "user", "content": "Continue."},
        ],
        context_management=[{"type": "compaction", "compact_threshold": 100}],
        store=False,
        max_output_tokens=50,
    )
    first_item = response.output[0]
    first_type = first_item.get("type") if isinstance(first_item, dict) else getattr(first_item, "type", None)
    assert first_type == "compaction", (
        f"Expected compaction at output[0], got: {first_type}"
    )
    remaining_types = [
        item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        for item in response.output[1:]
    ]
    assert "message" in remaining_types, "Model response missing after compaction item"


# ---------------------------------------------------------------------------
# T3 — Encrypted content survives a round-trip (Case 2 → 3)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_encrypted_content_round_trip():
    """encrypted_content from turn 1 can be sent back as input in turn 2 without error"""
    fat_message = "word " * 5000

    resp1 = await litellm.aresponses(
        model="groq/llama-3.3-70b-versatile",
        input=[
            {"type": "message", "role": "user", "content": "Let's talk about debugging."},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": fat_message}],
            },
            {"type": "message", "role": "user", "content": "Continue."},
        ],
        context_management=[{"type": "compaction", "compact_threshold": 100}],
        store=False,
        max_output_tokens=50,
    )
    cmp_items = [
        item for item in resp1.output
        if (item.get("type") if isinstance(item, dict) else getattr(item, "type", None)) == "compaction"
    ]
    assert cmp_items, "Expected compaction in turn 1"
    cmp_item = cmp_items[0]
    if not isinstance(cmp_item, dict):
        cmp_item = dict(cmp_item)

    resp2 = await litellm.aresponses(
        model="groq/llama-3.3-70b-versatile",
        input=[
            cmp_item,
            {"type": "message", "role": "user", "content": "What were we talking about?"},
        ],
        context_management=[{"type": "compaction", "compact_threshold": 200000}],
        store=False,
        max_output_tokens=100,
    )
    assert resp2 is not None
    assert resp2.output is not None
    assert len(resp2.output) > 0


# ---------------------------------------------------------------------------
# T3b — LLM receives DECRYPTED content on second turn (decrypt-before-LLM)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_receives_decrypted_content_on_second_turn():
    """
    The LLM must receive the decoded conversation, not the raw encrypted blob.
    We embed a known fact in the compacted history, then ask about it in turn 2.
    If the LLM can answer, decryption happened correctly.
    """
    fat_prefix = "word " * 3000
    known_fact = "The answer to the secret question is BANANA."

    resp1 = await litellm.aresponses(
        model="groq/llama-3.3-70b-versatile",
        input=[
            {"type": "message", "role": "user", "content": f"Remember this: {known_fact}"},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": fat_prefix + " I have noted that."}],
            },
            {"type": "message", "role": "user", "content": "Good, keep that in mind."},
        ],
        context_management=[{"type": "compaction", "compact_threshold": 100}],
        store=False,
        max_output_tokens=100,
    )
    cmp_items = [
        item for item in resp1.output
        if (item.get("type") if isinstance(item, dict) else getattr(item, "type", None)) == "compaction"
    ]
    assert cmp_items, "Expected compaction to trigger in turn 1"
    cmp_item = cmp_items[0]
    if not isinstance(cmp_item, dict):
        cmp_item = dict(cmp_item)

    resp2 = await litellm.aresponses(
        model="groq/llama-3.3-70b-versatile",
        input=[
            cmp_item,
            {"type": "message", "role": "user", "content": "What is the answer to the secret question?"},
        ],
        context_management=[{"type": "compaction", "compact_threshold": 200000}],
        store=False,
        max_output_tokens=100,
    )
    assert resp2 is not None
    text_output = ""
    for item in resp2.output:
        item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        if item_type == "message":
            content_list = item.get("content") if isinstance(item, dict) else getattr(item, "content", [])
            if isinstance(content_list, list):
                for block in content_list:
                    if isinstance(block, dict) and block.get("type") == "output_text":
                        text_output += block.get("text", "")
            elif isinstance(content_list, str):
                text_output += content_list

    assert "BANANA" in text_output.upper(), (
        f"LLM didn't receive decrypted context. Got: '{text_output[:200]}'"
    )


# ---------------------------------------------------------------------------
# T4 — No recompaction when compacted + new messages stay under threshold (Case 3)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_recompaction_when_under_threshold():
    """Case 3: compaction item + small new message, total under threshold → no new compaction"""
    import base64

    fake_summary = "This is a summary of the previous conversation about debugging."
    fake_encrypted = base64.b64encode(fake_summary.encode()).decode()
    prior_cmp = {
        "type": "compaction",
        "id": "cmp_prior_001",
        "encrypted_content": fake_encrypted,
    }

    response = await litellm.aresponses(
        model="groq/llama-3.3-70b-versatile",
        input=[
            prior_cmp,
            {"type": "message", "role": "user", "content": "What is 3+3?"},
        ],
        context_management=[{"type": "compaction", "compact_threshold": 200000}],
        store=False,
        max_output_tokens=50,
    )
    assert response is not None
    new_cmp_items = [
        item for item in response.output
        if (item.get("type") if isinstance(item, dict) else getattr(item, "type", None)) == "compaction"
        and (item.get("id") if isinstance(item, dict) else getattr(item, "id", None)) != "cmp_prior_001"
    ]
    assert not new_cmp_items, (
        "Should NOT re-compact when compacted context + new messages are under threshold"
    )
    output_types = [
        item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        for item in response.output
    ]
    assert "message" in output_types


# ---------------------------------------------------------------------------
# T7 — Absolute no-op when context_management is absent (regression guard)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_context_management_is_noop():
    """No context_management param → response has no compaction fields anywhere"""
    response = await litellm.aresponses(
        model="groq/llama-3.3-70b-versatile",
        input=[{"type": "message", "role": "user", "content": "What is the capital of France?"}],
        store=False,
        max_output_tokens=50,
    )
    assert response is not None
    output_types = [
        item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        for item in response.output
    ]
    assert "compaction" not in output_types


# ---------------------------------------------------------------------------
# T8 — compact_threshold = 0 handled gracefully (no unhandled crash)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compact_threshold_zero_handled_gracefully():
    """compact_threshold=0 should not raise an unhandled exception"""
    try:
        response = await litellm.aresponses(
            model="groq/llama-3.3-70b-versatile",
            input=[{"type": "message", "role": "user", "content": "Hello."}],
            context_management=[{"type": "compaction", "compact_threshold": 0}],
            store=False,
            max_output_tokens=50,
        )
        assert response is not None
    except Exception as e:
        err = str(e).lower()
        assert "compact_threshold" in err or "threshold" in err or "compaction" in err, (
            f"Expected a clear validation error about compact_threshold, got: {e}"
        )
