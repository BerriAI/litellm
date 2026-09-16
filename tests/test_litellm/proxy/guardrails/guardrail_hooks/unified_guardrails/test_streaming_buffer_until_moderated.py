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
from typing import Any, AsyncGenerator, List, Literal, Optional

import pytest

from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    ModifyResponseException,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import (
    UnifiedLLMGuardrails,
)
from litellm.types.utils import Delta, GenericGuardrailAPIInputs, ModelResponseStream, StreamingChoices

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


class _CountingPassingGuardrail(_PassingGuardrail):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scan_count = 0

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        self.scan_count += 1
        return inputs


class _SecondScanBlockingGuardrail(_CountingPassingGuardrail):
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        self.scan_count += 1
        if self.scan_count == 2:
            raise ModifyResponseException(
                message=BLOCK_MESSAGE,
                model="gpt-4",
                request_data=request_data,
                guardrail_name=self.guardrail_name,
            )
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


def _chat_chunk(content: str = "", finish_reason: str | None = None) -> ModelResponseStream:
    return ModelResponseStream(
        id="chatcmpl-windowed",
        created=1724900000,
        model="gpt-4",
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(role="assistant", content=content),
                finish_reason=finish_reason,
            )
        ],
    )


async def _windowed_chat_stream(
    yielded_count: List[int], collected: List[Any], content_chunks: List[str]
) -> AsyncGenerator[ModelResponseStream, None]:
    for content in content_chunks:
        yielded_count.append(len(collected))
        yield _chat_chunk(content)
    yielded_count.append(len(collected))
    yield _chat_chunk(finish_reason="stop")


async def _run_windowed(
    guardrail: CustomGuardrail,
    content_chunks: List[str],
    end_of_stream_only: bool = False,
) -> tuple[List[Any], List[int]]:
    guardrail.streaming_buffer_until_moderated = True
    guardrail.streaming_buffer_release_on_scan = True
    guardrail.streaming_end_of_stream_only = end_of_stream_only
    guardrail.streaming_sampling_rate = 2
    unified = UnifiedLLMGuardrails()
    user_api_key_dict = UserAPIKeyAuth(api_key="test", request_route="/v1/chat/completions")
    request_data = {
        "messages": [{"role": "user", "content": "hi"}],
        "guardrail_to_apply": guardrail,
        "metadata": {"guardrails": [guardrail.guardrail_name]},
    }
    collected: List[Any] = []
    yielded_count: List[int] = []
    async for chunk in unified.async_post_call_streaming_iterator_hook(
        user_api_key_dict=user_api_key_dict,
        response=_windowed_chat_stream(yielded_count, collected, content_chunks),
        request_data=request_data,
    ):
        collected.append(chunk)
    return collected, yielded_count


def _chat_text(chunks: List[Any]) -> str:
    return "".join(
        choice.delta.content or ""
        for chunk in chunks
        if isinstance(chunk, ModelResponseStream)
        for choice in chunk.choices
    )


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
async def test_windowed_buffer_releases_after_each_passing_scan():
    guardrail = _CountingPassingGuardrail(guardrail_name="windowed-pass", event_hook="post_call")
    content_chunks = ["one ", "two ", "three ", "four ", "five ", "six "]

    collected, yielded_count = await _run_windowed(guardrail, content_chunks)

    assert yielded_count[2] >= 2
    assert yielded_count == [0, 0, 2, 2, 4, 4, 6]
    assert _chat_text(collected) == "".join(content_chunks)
    assert guardrail.scan_count > 1


@pytest.mark.asyncio
async def test_windowed_buffer_drops_blocked_window():
    guardrail = _SecondScanBlockingGuardrail(guardrail_name="windowed-block", event_hook="post_call")
    content_chunks = ["one ", "two ", "MARKER ", "four ", "five ", "six "]

    collected, _ = await _run_windowed(guardrail, content_chunks)
    raw = _decode(collected)

    assert _chat_text(collected) == "one two "
    assert "MARKER" not in raw
    assert BLOCK_MESSAGE in raw
    assert '"error"' not in raw


@pytest.mark.asyncio
async def test_windowed_buffer_with_explicit_end_of_stream_only_stays_fully_buffered():
    guardrail = _CountingPassingGuardrail(guardrail_name="windowed-eos", event_hook="post_call")
    content_chunks = ["one ", "two ", "three ", "four ", "five ", "six "]

    collected, yielded_count = await _run_windowed(guardrail, content_chunks, end_of_stream_only=True)

    assert yielded_count == [0, 0, 0, 0, 0, 0, 0]
    assert _chat_text(collected) == "".join(content_chunks)
    assert guardrail.scan_count == 1


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
