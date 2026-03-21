"""
Tests for fix/presidio-pii-roundtrip-v2

Verifies PII round-trip masking/unmasking:
1. anonymize_text stores correct original values (position fix)
2. apply_guardrail unmask path restores PII tokens
3. apply_guardrail strips OpenAI-converted keys
4. apply_guardrail stores pii_tokens in request_data
5. _unmask_pii_text handles exact match, stripped brackets, truncated tokens
6. async_post_call_success_hook handles Anthropic native dict responses
7. Streaming SSE unmask with carry buffer

Related issue: https://github.com/BerriAI/litellm/issues/22821
"""

import json
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.guardrails.guardrail_hooks.presidio import (
    _OPTIONAL_PresidioPIIMasking,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def guardrail():
    """Create a Presidio guardrail instance with mock_testing=True."""
    return _OPTIONAL_PresidioPIIMasking(
        mock_testing=True,
        output_parse_pii=True,
    )


# ---------------------------------------------------------------------------
# _unmask_pii_text tests
# ---------------------------------------------------------------------------


def test_unmask_pii_text_exact_match():
    """Exact token in text is replaced with original value."""
    pii_tokens = {"<PERSON_abc123>": "Alice Smith"}
    result = _OPTIONAL_PresidioPIIMasking._unmask_pii_text(
        "Hello <PERSON_abc123>, welcome!", pii_tokens
    )
    assert result == "Hello Alice Smith, welcome!"


def test_unmask_pii_text_multiple_tokens():
    """Multiple different tokens are all replaced."""
    pii_tokens = {
        "<PERSON_abc123>": "Alice",
        "<PHONE_NUMBER_def456>": "555-0100",
    }
    result = _OPTIONAL_PresidioPIIMasking._unmask_pii_text(
        "Call <PERSON_abc123> at <PHONE_NUMBER_def456>", pii_tokens
    )
    assert result == "Call Alice at 555-0100"


def test_unmask_pii_text_stripped_brackets():
    """FALLBACK 1: LLM stripped angle brackets from token."""
    pii_tokens = {"<PERSON_abc123>": "Bob Jones"}
    result = _OPTIONAL_PresidioPIIMasking._unmask_pii_text(
        "Hello PERSON_abc123, welcome!", pii_tokens
    )
    assert result == "Hello Bob Jones, welcome!"


def test_unmask_pii_text_truncated_token():
    """FALLBACK 2: Token truncated at end of text by max_tokens."""
    pii_tokens = {"<COMPANY_NAME_abcdef123456>": "Acme Corp"}
    # Token is cut off at the end of the text
    truncated = "Working at <COMPANY_NAME_abcde"
    result = _OPTIONAL_PresidioPIIMasking._unmask_pii_text(truncated, pii_tokens)
    assert result == "Working at Acme Corp"


def test_unmask_pii_text_no_match_passthrough():
    """Text without any PII tokens passes through unchanged."""
    pii_tokens = {"<PERSON_abc123>": "Alice"}
    result = _OPTIONAL_PresidioPIIMasking._unmask_pii_text(
        "Hello world, no PII here!", pii_tokens
    )
    assert result == "Hello world, no PII here!"


# ---------------------------------------------------------------------------
# apply_guardrail tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_guardrail_unmask_path(guardrail):
    """input_type='response' with pii_tokens unmasks texts."""
    request_data = {
        "metadata": {"pii_tokens": {"<PERSON_abc123>": "Alice Smith"}},
    }
    inputs = {"texts": ["Hello <PERSON_abc123>, how are you?"]}

    result = await guardrail.apply_guardrail(
        inputs=inputs,
        request_data=request_data,
        input_type="response",
    )

    assert result["texts"] == ["Hello Alice Smith, how are you?"]


@pytest.mark.asyncio
async def test_apply_guardrail_unmask_no_tokens(guardrail):
    """input_type='response' with empty pii_tokens passes through."""
    request_data = {"metadata": {"pii_tokens": {}}}
    inputs = {"texts": ["Hello world"]}

    result = await guardrail.apply_guardrail(
        inputs=inputs,
        request_data=request_data,
        input_type="response",
    )

    assert result["texts"] == ["Hello world"]


@pytest.mark.asyncio
async def test_apply_guardrail_strips_openai_keys(guardrail):
    """After masking, OpenAI-converted keys are stripped from return."""
    inputs = {
        "texts": ["no pii here"],
        "tools": [{"type": "function", "name": "test"}],
        "structured_messages": [{"role": "user"}],
        "model": "gpt-4",
        "images": [],
    }

    # Mock check_pii to avoid calling Presidio
    async def _passthrough(text, output_parse_pii, presidio_config, request_data):
        return text

    guardrail.check_pii = _passthrough

    result = await guardrail.apply_guardrail(
        inputs=inputs,
        request_data={},
        input_type="request",
    )

    assert "tools" not in result
    assert "structured_messages" not in result
    assert "model" not in result
    assert "images" not in result
    assert "texts" in result


@pytest.mark.asyncio
async def test_apply_guardrail_stores_pii_tokens_via_check_pii(guardrail):
    """check_pii stores pii_tokens in request_data['metadata']['pii_tokens']."""
    # Mock check_pii to simulate anonymize_text storing a token
    async def _mock_check_pii(text, output_parse_pii, presidio_config, request_data):
        if "metadata" not in request_data:
            request_data["metadata"] = {}
        if "pii_tokens" not in request_data["metadata"]:
            request_data["metadata"]["pii_tokens"] = {}
        request_data["metadata"]["pii_tokens"]["<PERSON_abc>"] = "Alice"
        return "Hello <PERSON_abc>"

    guardrail.check_pii = _mock_check_pii

    inputs = {"texts": ["Hello Alice"]}
    # request_data must be non-empty (truthy) so `request_data or {}`
    # returns the same dict object rather than a new empty one.
    request_data = {"model": "test"}

    await guardrail.apply_guardrail(
        inputs=inputs,
        request_data=request_data,
        input_type="request",
    )

    assert "metadata" in request_data
    assert "pii_tokens" in request_data["metadata"]
    assert request_data["metadata"]["pii_tokens"] == {"<PERSON_abc>": "Alice"}


# ---------------------------------------------------------------------------
# async_post_call_success_hook: Anthropic native dict
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_call_success_hook_anthropic_native_dict(guardrail):
    """Anthropic native dict response (type:'message') gets unmasked."""
    guardrail.pii_tokens = {"<PERSON_abc123>": "Alice Smith"}
    guardrail.output_parse_pii = True

    response = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello <PERSON_abc123>!"}],
        "model": "claude-sonnet-4-6",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    data = {"metadata": {"pii_tokens": {"<PERSON_abc123>": "Alice Smith"}}}
    from litellm.proxy._types import UserAPIKeyAuth

    user_api_key = UserAPIKeyAuth(api_key="test_key")

    result = await guardrail.async_post_call_success_hook(
        data=data,
        user_api_key_dict=user_api_key,
        response=response,
    )

    assert result["content"][0]["text"] == "Hello Alice Smith!"


# ---------------------------------------------------------------------------
# Streaming SSE tests
# ---------------------------------------------------------------------------


def _make_sse_event(event_type: str, data: dict) -> bytes:
    """Helper to create an SSE event as bytes."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode("utf-8")


def _make_text_delta_event(text: str, index: int = 0) -> bytes:
    """Helper to create a text_delta SSE event."""
    return _make_sse_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": text},
        },
    )


async def _async_iter(items):
    """Convert a list to an async iterator."""
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_streaming_anthropic_sse_unmask(guardrail):
    """PII tokens in SSE text_delta events are unmasked."""
    guardrail.pii_tokens = {"<PERSON_abc123>": "Alice Smith"}
    guardrail.output_parse_pii = True

    chunks = [
        _make_sse_event("message_start", {"type": "message_start"}),
        _make_text_delta_event("Hello <PERSON_abc123>!"),
        _make_sse_event("message_stop", {"type": "message_stop"}),
    ]

    request_data = {"metadata": {"pii_tokens": {"<PERSON_abc123>": "Alice Smith"}}}
    collected = b""

    async for chunk in guardrail.async_post_call_streaming_iterator_hook(
        user_api_key_dict=MagicMock(),
        response=_async_iter(chunks),
        request_data=request_data,
    ):
        if isinstance(chunk, bytes):
            collected += chunk

    decoded = collected.decode("utf-8")
    assert "Alice Smith" in decoded
    assert "<PERSON_abc123>" not in decoded


@pytest.mark.asyncio
async def test_streaming_sse_split_token_carry_buffer(guardrail):
    """Token split across two SSE events is correctly reassembled via carry buffer."""
    guardrail.pii_tokens = {"<PERSON_abc123>": "Alice Smith"}
    guardrail.output_parse_pii = True

    # Split the token across two text_delta events
    chunks = [
        _make_text_delta_event("Hello <PERSO"),
        _make_text_delta_event("N_abc123>!"),
        _make_sse_event("message_stop", {"type": "message_stop"}),
    ]

    request_data = {"metadata": {"pii_tokens": {"<PERSON_abc123>": "Alice Smith"}}}
    collected = b""

    async for chunk in guardrail.async_post_call_streaming_iterator_hook(
        user_api_key_dict=MagicMock(),
        response=_async_iter(chunks),
        request_data=request_data,
    ):
        if isinstance(chunk, bytes):
            collected += chunk

    decoded = collected.decode("utf-8")
    assert "Alice Smith" in decoded
    assert "<PERSON_abc123>" not in decoded


@pytest.mark.asyncio
async def test_streaming_sse_passthrough_no_pii_tokens(guardrail):
    """When pii_tokens is empty, SSE bytes pass through unchanged."""
    guardrail.pii_tokens = {}
    guardrail.output_parse_pii = True

    original_text = "Hello world, no PII!"
    chunks = [
        _make_text_delta_event(original_text),
        _make_sse_event("message_stop", {"type": "message_stop"}),
    ]

    request_data = {}
    collected = b""

    async for chunk in guardrail.async_post_call_streaming_iterator_hook(
        user_api_key_dict=MagicMock(),
        response=_async_iter(chunks),
        request_data=request_data,
    ):
        if isinstance(chunk, bytes):
            collected += chunk

    decoded = collected.decode("utf-8")
    assert original_text in decoded
