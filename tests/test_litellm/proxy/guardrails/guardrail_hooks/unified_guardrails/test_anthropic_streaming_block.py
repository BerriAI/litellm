"""
Regression tests for blocking an Anthropic streaming response from the
unified guardrail post-call streaming iterator hook.

When a guardrail's ``apply_guardrail`` raises ``ModifyResponseException`` while
(or at the end of) an Anthropic ``/v1/messages`` stream is being relayed, the
hook must emit a well-formed Anthropic SSE termination sequence carrying the
block message - NOT a bare ``data: {"error": ...}`` blob that truncates the
stream and causes the Anthropic SDK parser to discard the response.
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


class _BlockingGuardrail(CustomGuardrail):
    """Mock guardrail that always blocks by raising ModifyResponseException."""

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


def _sse_event(event_type: str, data: dict) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


async def _anthropic_stream(end: bool):
    """Yield Anthropic SSE byte chunks. If end=True, include a terminating
    message_delta (stop_reason set) so the hook's end-of-stream path runs."""
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
    for text in ["This ", "is ", "the ", "original ", "answer."]:
        yield _sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        )
    if end:
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
    parts = []
    for chunk in chunks:
        parts.append(chunk.decode() if isinstance(chunk, bytes) else str(chunk))
    return "".join(parts)


def _parse_sse_event_types(raw: str) -> List[str]:
    event_types = []
    for block in raw.split("\n\n"):
        for line in block.strip().split("\n"):
            if line.startswith("data:"):
                payload = line[len("data:") :].strip()
                try:
                    event_types.append(json.loads(payload).get("type"))
                except json.JSONDecodeError:
                    pass
    return event_types


async def _run_hook(end: bool, sampling_rate: int = 1, end_of_stream_only: bool = False) -> str:
    guardrail = _BlockingGuardrail(guardrail_name="test-blocking-guardrail", event_hook="post_call")
    # sampling_rate controls how many chunks are forwarded before the block
    # fires: 1 blocks on the first chunk (nothing sent yet); >1 forwards earlier
    # chunks first, exercising the mid-stream "continue the message" path.
    guardrail.streaming_sampling_rate = sampling_rate
    guardrail.streaming_end_of_stream_only = end_of_stream_only

    unified_guardrail = UnifiedLLMGuardrails()
    user_api_key_dict = UserAPIKeyAuth(api_key="test", request_route="/v1/messages")
    request_data = {
        "messages": [{"role": "user", "content": "hi"}],
        "guardrail_to_apply": guardrail,
        "metadata": {"guardrails": ["test-blocking-guardrail"]},
    }

    collected: List[Any] = []
    async for chunk in unified_guardrail.async_post_call_streaming_iterator_hook(
        user_api_key_dict=user_api_key_dict,
        response=_anthropic_stream(end=end),
        request_data=request_data,
    ):
        collected.append(chunk)
    return _decode(collected)


def _assert_clean_block_termination(raw: str) -> None:
    # No bare error blob that would truncate the stream.
    assert '"error"' not in raw, f"unexpected error blob in stream: {raw!r}"
    # The block message is delivered as assistant text.
    assert BLOCK_MESSAGE in raw, f"block message missing from stream: {raw!r}"
    # A complete, parseable Anthropic SSE termination sequence is present.
    event_types = _parse_sse_event_types(raw)
    assert "message_start" in event_types
    assert "content_block_delta" in event_types
    # Exactly one message_start: a block must never inject a second
    # message envelope into an already-started stream (clients reject it).
    assert event_types.count("message_start") == 1, f"expected a single message_start, got: {event_types}"
    assert event_types[-1] == "message_stop", f"stream did not end cleanly: {event_types}"
    # message_delta carries a stop_reason.
    assert any('"stop_reason"' in block and "message_delta" in block for block in raw.split("\n\n"))


def _parse_sse_payloads(raw: str) -> List[dict]:
    payloads = []
    for block in raw.split("\n\n"):
        for line in block.strip().split("\n"):
            if line.startswith("data:"):
                payload = line[len("data:") :].strip()
                try:
                    parsed = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    payloads.append(parsed)
    return payloads


@pytest.mark.asyncio
async def test_mid_stream_block_emits_clean_anthropic_sse():
    """Per-chunk block: a clean SSE termination with the block message, no error blob."""
    raw = await _run_hook(end=False)
    _assert_clean_block_termination(raw)


@pytest.mark.asyncio
async def test_end_of_stream_block_emits_clean_anthropic_sse():
    """End-of-stream block: same clean SSE termination guarantees."""
    raw = await _run_hook(end=True)
    _assert_clean_block_termination(raw)


@pytest.mark.asyncio
async def test_mid_stream_block_after_prior_chunks_continues_message():
    """Regression: when real chunks were already forwarded (sampling_rate>1),
    the block must continue the in-progress message, not start a second one."""
    raw = await _run_hook(end=False, sampling_rate=5)
    # Some original content was forwarded before the block...
    assert "message_start" in raw
    # ...and the block continues that same message (single message_start) with
    # the block message appended, ending cleanly.
    _assert_clean_block_termination(raw)


@pytest.mark.asyncio
async def test_end_of_stream_only_block_does_not_append_after_message_stop():
    raw = await _run_hook(end=True, end_of_stream_only=True)
    event_types = _parse_sse_event_types(raw)
    message_delta_usages = [
        payload.get("usage", {}).get("output_tokens")
        for payload in _parse_sse_payloads(raw)
        if payload.get("type") == "message_delta"
    ]

    assert BLOCK_MESSAGE in raw
    assert event_types.count("message_stop") == 1
    assert event_types[-1] == "message_stop"
    assert message_delta_usages[-1] == 5


def test_blocked_stream_reports_usage_from_original_chunks():
    from litellm.integrations.custom_guardrail import ModifyResponseException
    from litellm.llms.anthropic.chat.guardrail_translation.handler import (
        AnthropicMessagesHandler,
    )
    from litellm.llms.base_llm.guardrail_translation.utils import (
        blocked_response_usage,
    )

    original_chunks: List[Any] = [
        _sse_event(
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
                    "usage": {"input_tokens": 12, "output_tokens": 0},
                },
            },
        ),
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
    ]
    seen_chunks = [
        _sse_event(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        )
    ]
    exc = ModifyResponseException(
        message=BLOCK_MESSAGE,
        model="claude-3-5-sonnet",
        request_data={},
        guardrail_name="g",
        original_response=original_chunks,
    )

    usage = blocked_response_usage(original_chunks)
    raw = b"".join(
        AnthropicMessagesHandler().build_block_sse_chunks(exc, stream_started=True, responses_so_far=seen_chunks)
    ).decode()
    message_delta_usages = [
        payload.get("usage", {}).get("output_tokens")
        for payload in _parse_sse_payloads(raw)
        if payload.get("type") == "message_delta"
    ]

    assert usage == {"input_tokens": 12, "output_tokens": 5}
    assert message_delta_usages[-1] == 5


class TestContentBlockState:
    """`_content_block_state` must reflect the true open/last block index across
    the two chunk formats the stream can carry (multi-event bytes, parsed dict),
    so a mid-stream block closes/opens the right indices."""

    def _handler(self):
        from litellm.llms.anthropic.chat.guardrail_translation.handler import (
            AnthropicMessagesHandler,
        )

        return AnthropicMessagesHandler()

    def test_multi_event_bytes_chunk_is_fully_parsed(self):
        # One item bundles start(0) + delta + stop(0): the block is already
        # closed, so open_index is None (not 0) and max_index is 0.
        bundled = (
            _sse_event(
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            )
            + _sse_event(
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
            )
            + _sse_event("content_block_stop", {"type": "content_block_stop", "index": 0})
        )
        open_index, max_index = self._handler()._content_block_state([bundled])
        assert open_index is None
        assert max_index == 0

    def test_open_block_across_separate_chunks(self):
        chunks = [
            _sse_event(
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            ),
            _sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}),
            _sse_event(
                "content_block_start",
                {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
            ),
        ]
        open_index, max_index = self._handler()._content_block_state(chunks)
        assert open_index == 1
        assert max_index == 1

    def test_dict_format_chunks_are_parsed(self):
        # The backwards-compat parsed-dict format must be understood too.
        chunks = [
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "x"}},
        ]
        open_index, max_index = self._handler()._content_block_state(chunks)
        assert open_index == 0
        assert max_index == 0

    def test_continuation_closes_open_block_and_appends_after_it(self):
        from litellm.integrations.custom_guardrail import ModifyResponseException

        handler = self._handler()
        # Client has seen an open text block at index 0.
        seen = [
            _sse_event(
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            ),
            _sse_event(
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "partial"}},
            ),
        ]
        exc = ModifyResponseException(
            message=BLOCK_MESSAGE, model="claude-3-5-sonnet", request_data={}, guardrail_name="g"
        )
        raw = b"".join(handler.build_block_sse_chunks(exc, stream_started=True, responses_so_far=seen)).decode()
        events = _parse_sse_event_types(raw)
        # No new message envelope, closes block 0, appends block text at index 1.
        assert "message_start" not in events
        assert events == [
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        assert BLOCK_MESSAGE in raw
        assert '"index": 1' in raw
