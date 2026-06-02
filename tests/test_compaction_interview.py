import base64
import os

import pytest

import litellm

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY required",
)

MODEL = os.getenv("ANTHROPIC_COMPACTION_MODEL", "claude-haiku-4-5-20251001")
FAT = "word " * 5000


def _types(response):
    out = []
    for item in response.output or []:
        out.append(item.get("type") if isinstance(item, dict) else getattr(item, "type", None))
    return out


def _compaction_item(response):
    for item in response.output or []:
        t = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        if t == "compaction":
            return item if isinstance(item, dict) else dict(item)
    return None


# --- Required matrix (cases 1–3) ---


@pytest.mark.asyncio
async def test_compaction_noop_under_threshold():
    """Case 1: under threshold → no compaction."""
    response = await litellm.aresponses(
        model=MODEL,
        input=[
            {"type": "message", "role": "user", "content": "Help me debug a production incident."},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "What symptoms are you seeing?"}],
            },
            {
                "type": "message",
                "role": "user",
                "content": "We are seeing intermittent 502s from one provider path.",
            },
        ],
        context_management=[{"type": "compaction", "compact_threshold": 200000}],
        store=False,
        max_output_tokens=200,
    )
    assert response is not None
    assert "compaction" not in _types(response)


@pytest.mark.asyncio
async def test_compaction_triggers_over_threshold():
    """Case 2: over threshold → compaction item with encrypted_content."""
    response = await litellm.aresponses(
        model=MODEL,
        input=[
            {"type": "message", "role": "user", "content": "Help me debug a production incident."},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": FAT}]},
            {
                "type": "message",
                "role": "user",
                "content": "We are seeing intermittent 502s from one provider path.",
            },
        ],
        context_management=[{"type": "compaction", "compact_threshold": 100}],
        store=False,
        max_output_tokens=200,
    )
    assert response is not None
    cmp = _compaction_item(response)
    assert cmp is not None
    assert cmp.get("encrypted_content")
    assert "message" in _types(response)


@pytest.mark.asyncio
async def test_no_recompaction_when_under_threshold():
    """Case 3: prior compaction + short follow-up → no new compaction id."""
    prior_cmp = {
        "type": "compaction",
        "id": "cmp_prior_001",
        "encrypted_content": base64.b64encode(b"Previous conversation summary.").decode(),
    }
    response = await litellm.aresponses(
        model=MODEL,
        input=[
            prior_cmp,
            {"type": "message", "role": "user", "content": "What was the main issue?"},
        ],
        context_management=[{"type": "compaction", "compact_threshold": 200000}],
        store=False,
        max_output_tokens=100,
    )
    assert response is not None
    new_cmp = [
        i
        for i in response.output
        if (i.get("type") if isinstance(i, dict) else getattr(i, "type", None)) == "compaction"
        and (i.get("id") if isinstance(i, dict) else getattr(i, "id", None)) != "cmp_prior_001"
    ]
    assert not new_cmp


# --- A few extras from the "additional cases" list ---


@pytest.mark.asyncio
async def test_no_context_management_param_is_noop():
    """No param → plain response, no compaction."""
    response = await litellm.aresponses(
        model=MODEL,
        input=[{"type": "message", "role": "user", "content": "What is the capital of France?"}],
        store=False,
        max_output_tokens=50,
    )
    assert response is not None
    assert "compaction" not in _types(response)


@pytest.mark.asyncio
async def test_encrypted_content_round_trip():
    """Turn 1 compaction → turn 2 input with same item → assistant message."""
    resp1 = await litellm.aresponses(
        model=MODEL,
        input=[
            {"type": "message", "role": "user", "content": "Help me debug a production incident."},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": FAT}]},
            {
                "type": "message",
                "role": "user",
                "content": "We are seeing intermittent 502s from one provider path.",
            },
        ],
        context_management=[{"type": "compaction", "compact_threshold": 100}],
        store=False,
        max_output_tokens=120,
    )
    cmp = _compaction_item(resp1)
    assert cmp is not None

    resp2 = await litellm.aresponses(
        model=MODEL,
        input=[cmp, {"type": "message", "role": "user", "content": "What were we discussing?"}],
        context_management=[{"type": "compaction", "compact_threshold": 200000}],
        store=False,
        max_output_tokens=100,
    )
    assert resp2 is not None
    assert "message" in _types(resp2)
