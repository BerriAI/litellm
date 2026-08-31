"""
Regression tests for blocking an OpenAI-format streaming response from the
unified guardrail post-call streaming iterator hook.

When a guardrail's ``apply_guardrail`` raises ``ModifyResponseException``
while (or at the end of) a chat completions or Responses API stream is being
relayed, the hook must emit a well-formed SSE termination sequence carrying
the block message - NOT a bare ``data: {"error": ...}`` blob that surfaces as
an HTTP 500 error frame and truncates the stream.
"""

import json
from typing import Any, AsyncGenerator, Dict, Literal, Optional, Tuple, Union

import pytest

from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    ModifyResponseException,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import (
    UnifiedLLMGuardrails,
)
from litellm.types.utils import (
    Delta,
    GenericGuardrailAPIInputs,
    ModelResponseStream,
    StreamingChoices,
)

BLOCK_MESSAGE = "This response was replaced by policy."

JsonPayload = Dict[str, object]
StreamChunk = Union[ModelResponseStream, JsonPayload, bytes]


class _BlockingGuardrail(CustomGuardrail):
    """Mock guardrail that always blocks response scans by raising ModifyResponseException."""

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        raise ModifyResponseException(
            message=BLOCK_MESSAGE,
            model="gpt-5.4-mini",
            request_data=request_data,
            guardrail_name=self.guardrail_name,
        )


def _chat_chunk(delta: Delta, finish_reason: Optional[str] = None) -> ModelResponseStream:
    return ModelResponseStream(
        id="chatcmpl-live",
        created=1724900000,
        model="gpt-5.4-mini",
        choices=[StreamingChoices(index=0, delta=delta, finish_reason=finish_reason)],
    )


async def _chat_stream(end: bool) -> AsyncGenerator[ModelResponseStream, None]:
    yield _chat_chunk(Delta(role="assistant", content="This "))
    for text in ["is ", "the ", "original ", "answer."]:
        yield _chat_chunk(Delta(content=text))
    if end:
        yield _chat_chunk(Delta(), finish_reason="stop")


async def _responses_stream(end: bool) -> AsyncGenerator[JsonPayload, None]:
    original_text = "This is the original answer."
    response_envelope = {"id": "resp_live", "model": "gpt-5.4-mini", "status": "in_progress", "output": []}
    yield {"type": "response.created", "response": response_envelope}
    yield {"type": "response.in_progress", "response": response_envelope}
    yield {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {"id": "msg_orig", "type": "message", "role": "assistant", "content": []},
    }
    yield {
        "type": "response.content_part.added",
        "item_id": "msg_orig",
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "output_text", "text": "", "annotations": []},
    }
    for delta in ["This ", "is ", "the ", "original ", "answer."]:
        yield {
            "type": "response.output_text.delta",
            "item_id": "msg_orig",
            "output_index": 0,
            "content_index": 0,
            "delta": delta,
        }
    yield {
        "type": "response.output_text.done",
        "item_id": "msg_orig",
        "output_index": 0,
        "content_index": 0,
        "text": original_text,
    }
    if end:
        yield {
            "type": "response.completed",
            "response": {
                "id": "resp_live",
                "model": "gpt-5.4-mini",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_orig",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": original_text, "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 7, "output_tokens": 21, "total_tokens": 28},
            },
        }


async def _run_hook(
    route: str,
    stream: AsyncGenerator[Union[ModelResponseStream, JsonPayload], None],
    sampling_rate: int = 1,
    end_of_stream_only: bool = False,
    buffer_until_moderated: bool = False,
) -> Tuple[StreamChunk, ...]:
    guardrail = _BlockingGuardrail(guardrail_name="test-blocking-guardrail", event_hook="post_call")
    guardrail.streaming_sampling_rate = sampling_rate
    guardrail.streaming_end_of_stream_only = end_of_stream_only
    guardrail.streaming_buffer_until_moderated = buffer_until_moderated

    unified_guardrail = UnifiedLLMGuardrails()
    user_api_key_dict = UserAPIKeyAuth(api_key="test", request_route=route)
    request_data = {
        "messages": [{"role": "user", "content": "hi"}],
        "guardrail_to_apply": guardrail,
        "metadata": {"guardrails": ["test-blocking-guardrail"]},
    }

    return tuple(
        [
            chunk
            async for chunk in unified_guardrail.async_post_call_streaming_iterator_hook(
                user_api_key_dict=user_api_key_dict,
                response=stream,
                request_data=request_data,
            )
        ]
    )


def _sse_payloads(collected: Tuple[StreamChunk, ...]) -> Tuple[JsonPayload, ...]:
    return tuple(
        json.loads(line[len("data:") :].strip())
        for chunk in collected
        if isinstance(chunk, bytes)
        for block in chunk.decode().split("\n\n")
        for line in block.strip().split("\n")
        if line.startswith("data:")
    )


def _assert_no_error_frame(collected: Tuple[StreamChunk, ...]) -> None:
    raw = "".join(chunk.decode() for chunk in collected if isinstance(chunk, bytes))
    assert '"error"' not in raw, f"unexpected error blob in stream: {raw!r}"


@pytest.mark.asyncio
async def test_chat_pre_stream_block_emits_standalone_completion():
    """Block on the first chunk: a standalone completion opens with a role delta
    and ends with finish_reason content_filter."""
    collected = await _run_hook("/v1/chat/completions", _chat_stream(end=False))
    _assert_no_error_frame(collected)
    payloads = _sse_payloads(collected)
    assert payloads, "no block SSE chunks were emitted"
    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant", "content": BLOCK_MESSAGE}
    assert payloads[-1]["choices"][0]["finish_reason"] == "content_filter"


@pytest.mark.asyncio
async def test_chat_mid_stream_block_continues_the_completion():
    """Regression for the LIT-6496 500 error frame: after chunks were already
    forwarded, the block continues the same completion id and terminates with
    finish_reason content_filter instead of raising into an error blob."""
    collected = await _run_hook("/v1/chat/completions", _chat_stream(end=False), sampling_rate=5)
    _assert_no_error_frame(collected)
    forwarded = [chunk for chunk in collected if isinstance(chunk, ModelResponseStream)]
    assert forwarded, "original chunks should have streamed before the block"
    payloads = _sse_payloads(collected)
    assert payloads, "no block SSE chunks were emitted"
    assert all(payload["id"] == "chatcmpl-live" for payload in payloads), (
        "block chunks must continue the in-progress completion, not start a new one"
    )
    assert payloads[0]["choices"][0]["delta"] == {"content": BLOCK_MESSAGE}
    assert payloads[-1]["choices"][0]["finish_reason"] == "content_filter"


@pytest.mark.asyncio
async def test_chat_end_of_stream_block_terminates_cleanly():
    collected = await _run_hook("/v1/chat/completions", _chat_stream(end=True), end_of_stream_only=True)
    _assert_no_error_frame(collected)
    payloads = _sse_payloads(collected)
    assert BLOCK_MESSAGE in json.dumps(payloads)
    assert payloads[-1]["choices"][0]["finish_reason"] == "content_filter"


@pytest.mark.asyncio
async def test_responses_buffered_block_emits_full_event_sequence():
    """Buffered moderation blocks before anything streams: a complete synthetic
    Responses stream from response.created through response.completed carrying
    the block message, with the original content never released."""
    collected = await _run_hook("/v1/responses", _responses_stream(end=True), buffer_until_moderated=True)
    _assert_no_error_frame(collected)
    assert not [chunk for chunk in collected if isinstance(chunk, dict)], (
        "buffered original chunks must never be released after a block"
    )
    payloads = _sse_payloads(collected)
    event_types = [payload["type"] for payload in payloads]
    assert event_types[0] == "response.created"
    assert "response.output_text.delta" in event_types
    assert event_types[-1] == "response.completed"
    completed = payloads[-1]["response"]
    assert completed["status"] == "completed"
    assert completed["output"][0]["content"][0]["text"] == BLOCK_MESSAGE
    assert completed["usage"] == {"input_tokens": 7, "output_tokens": 21, "total_tokens": 28}
    assert "original answer" not in json.dumps(payloads)


@pytest.mark.asyncio
async def test_responses_mid_stream_block_continues_the_response():
    """Regression for the LIT-6496 500 error frame: after events were already
    forwarded, the block appends a new output item under the same response id
    and closes with response.completed - never a second response.created."""
    collected = await _run_hook("/v1/responses", _responses_stream(end=False))
    _assert_no_error_frame(collected)
    forwarded_types = [chunk["type"] for chunk in collected if isinstance(chunk, dict)]
    assert "response.created" in forwarded_types, "original events should have streamed before the block"
    payloads = _sse_payloads(collected)
    assert payloads, "no block SSE chunks were emitted"
    block_types = [payload["type"] for payload in payloads]
    assert "response.created" not in block_types, "a mid-stream block must not restart the response"
    assert block_types[0] == "response.output_item.added"
    assert block_types[-1] == "response.completed"
    assert payloads[0]["output_index"] == 1, "the block item must continue after the original output item"
    completed = payloads[-1]["response"]
    assert completed["id"] == "resp_live"
    assert completed["output"][0]["content"][0]["text"] == BLOCK_MESSAGE


@pytest.mark.asyncio
async def test_responses_end_of_stream_block_reports_original_usage():
    collected = await _run_hook("/v1/responses", _responses_stream(end=True), end_of_stream_only=True)
    _assert_no_error_frame(collected)
    forwarded_types = [chunk["type"] for chunk in collected if isinstance(chunk, dict)]
    assert "response.completed" not in forwarded_types, (
        "the original terminal event must be withheld and replaced by the block sequence"
    )
    payloads = _sse_payloads(collected)
    completed = payloads[-1]["response"]
    assert payloads[-1]["type"] == "response.completed"
    assert completed["id"] == "resp_live"
    assert completed["output"][0]["content"][0]["text"] == BLOCK_MESSAGE
    assert completed["usage"] == {"input_tokens": 7, "output_tokens": 21, "total_tokens": 28}
