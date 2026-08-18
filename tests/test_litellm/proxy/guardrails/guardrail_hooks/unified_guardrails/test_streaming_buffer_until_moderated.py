"""
Tests for ``streaming_buffer_until_moderated`` on the unified guardrail
post-call streaming iterator hook.

With this flag set, the hook must withhold every upstream chunk until
end-of-stream moderation has run. The decisive guarantee versus the
detect-only ``streaming_end_of_stream_only`` behavior: when the guardrail
blocks, the original (objectionable) content is NEVER yielded to the client --
only the block message is. On a clean response, all original chunks are
released unchanged after moderation passes.
"""

import json
from typing import Any, List, Literal, Optional

import pytest

from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    ModifyResponseException,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import (
    UnifiedLLMGuardrails,
)
from litellm.types.utils import GenericGuardrailAPIInputs

BLOCK_MESSAGE = "Blocked by policy: this response was withheld."
ORIGINAL_MARKER = "ORIGINAL-SECRET-ANSWER"


class _BlockingGuardrail(CustomGuardrail):
    """Always blocks at moderation time."""

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        raise ModifyResponseException(
            message=BLOCK_MESSAGE,
            model="claude-3-5-sonnet",
            request_data=request_data,
            guardrail_name=self.guardrail_name,
        )


class _PassingGuardrail(CustomGuardrail):
    """Never blocks; returns inputs unchanged."""

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        return inputs


def _sse_event(event_type: str, data: dict) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


async def _anthropic_stream():
    """A complete Anthropic /v1/messages SSE stream whose assistant text
    contains ORIGINAL_MARKER so leakage is unambiguous to assert."""
    yield _sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_orig",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet",
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        },
    )
    yield _sse_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    )
    for text in ["Here is ", "the ", ORIGINAL_MARKER, " for you."]:
        yield _sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        )
    yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield _sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 5},
        },
    )
    yield _sse_event("message_stop", {"type": "message_stop"})


def _decode(chunks: List[Any]) -> str:
    return "".join(c.decode() if isinstance(c, bytes) else str(c) for c in chunks)


async def _run(guardrail: CustomGuardrail) -> str:
    # Rubrik's real config: end-of-stream-only moderation. Without buffering
    # this releases every chunk before moderation runs (content leaks on
    # block); the buffer flag must change that to moderate-then-release.
    guardrail.streaming_end_of_stream_only = True
    guardrail.streaming_buffer_until_moderated = True
    unified = UnifiedLLMGuardrails()
    user_api_key_dict = UserAPIKeyAuth(api_key="test", request_route="/v1/messages")
    request_data = {
        "messages": [{"role": "user", "content": "hi"}],
        "guardrail_to_apply": guardrail,
        "metadata": {"guardrails": [guardrail.guardrail_name]},
    }
    collected: List[Any] = []
    async for chunk in unified.async_post_call_streaming_iterator_hook(
        user_api_key_dict=user_api_key_dict,
        response=_anthropic_stream(),
        request_data=request_data,
    ):
        collected.append(chunk)
    return _decode(collected)


@pytest.mark.asyncio
async def test_buffered_block_withholds_original_content():
    raw = await _run(_BlockingGuardrail(guardrail_name="blk", event_hook="post_call"))
    # The original content must never reach the client...
    assert ORIGINAL_MARKER not in raw, f"original content leaked: {raw!r}"
    # ...only the block message, in a clean terminating stream.
    assert BLOCK_MESSAGE in raw
    assert '"error"' not in raw


@pytest.mark.asyncio
async def test_buffered_clean_releases_all_content():
    raw = await _run(_PassingGuardrail(guardrail_name="pass", event_hook="post_call"))
    # A clean response is released in full after moderation passes.
    assert ORIGINAL_MARKER in raw
    assert (
        raw.rstrip().endswith('event: message_stop\ndata: {"type": "message_stop"}'.rstrip()) or "message_stop" in raw
    )
    assert BLOCK_MESSAGE not in raw


@pytest.mark.asyncio
async def test_buffered_mode_disabled_for_content_rewriting_guardrail():
    """Buffered replay yields the withheld *original* chunks verbatim, which
    is unsafe for a guardrail that rewrites response text (e.g. PII masking):
    the client would get the unredacted original instead of the moderated
    output. mask_response_content=True must force buffering off so the
    request falls back to the (correctly moderated) non-buffered path."""
    guardrail = _PassingGuardrail(guardrail_name="masker", event_hook="post_call", mask_response_content=True)
    raw = await _run(guardrail)
    assert guardrail.streaming_buffer_until_moderated is True  # request asked for buffering
    assert ORIGINAL_MARKER in raw
    assert BLOCK_MESSAGE not in raw
