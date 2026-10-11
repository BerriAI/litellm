import json
from collections.abc import AsyncIterator, Iterable

import pytest

from litellm.proxy.pass_through_endpoints.injected_stream_usage import (
    stream_usage_relay,
    without_injected_usage_frames,
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
FINGERPRINT = "fp-vllm-test"


def _frame(payload: dict) -> bytes:
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


def _chunk(identity: str, **fields) -> dict:
    return {"id": identity, "object": "chat.completion.chunk", "created": 1, "model": MODEL, **fields}


CONTENT = _frame(_chunk("chatcmpl-1", choices=[{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}]))
FINISH = _frame(_chunk("chatcmpl-1", choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}]))
USAGE = _frame(
    _chunk(
        "chatcmpl-1",
        choices=[],
        usage={"prompt_tokens": 34, "completion_tokens": 4, "total_tokens": 38},
        system_fingerprint=FINGERPRINT,
    )
)
DONE = b"data: [DONE]\n\n"


async def _pieces(pieces: Iterable[bytes]) -> AsyncIterator[bytes]:
    for piece in pieces:
        yield piece


async def _relayed(pieces: Iterable[bytes]) -> bytes:
    return b"".join([piece async for piece in without_injected_usage_frames(_pieces(pieces))])


def _payloads(relayed: bytes) -> list[dict]:
    return [json.loads(line[len("data: ") :]) for line in relayed.decode().split("\n\n") if line.startswith("data: {")]


class TestStreamUsageRelay:
    def test_a_chat_stream_nobody_asked_usage_for_gets_it_injected_and_stripped(self):
        relay = stream_usage_relay(
            endpoint="chat/completions",
            request_body={"model": MODEL, "messages": [], "stream": True},
            general_settings={},
            supports_stream_options=lambda: True,
        )

        assert relay.body == {"model": MODEL, "messages": [], "stream": True, "stream_options": {"include_usage": True}}
        assert relay.strips_usage is True

    def test_a_chat_stream_the_caller_asked_usage_for_is_relayed_untouched(self):
        body = {"model": MODEL, "messages": [], "stream": True, "stream_options": {"include_usage": True}}

        relay = stream_usage_relay(
            endpoint="v1/chat/completions", request_body=body, general_settings={}, supports_stream_options=lambda: True
        )

        assert relay.body == body
        assert relay.strips_usage is False

    def test_a_deployment_without_stream_options_support_is_left_alone(self):
        body = {"model": MODEL, "messages": [], "stream": True}

        relay = stream_usage_relay(
            endpoint="chat/completions", request_body=body, general_settings={}, supports_stream_options=lambda: False
        )

        assert relay.body == body
        assert relay.strips_usage is False

    def test_a_text_completion_stream_is_injected_only_when_usage_is_always_included(self):
        body = {"model": MODEL, "prompt": "Hi", "stream": True}

        by_default = stream_usage_relay(
            endpoint="completions", request_body=body, general_settings={}, supports_stream_options=lambda: True
        )
        always = stream_usage_relay(
            endpoint="v1/completions",
            request_body=body,
            general_settings={"always_include_stream_usage": True},
            supports_stream_options=lambda: True,
        )

        assert by_default.body == body
        assert always.body == {**body, "stream_options": {"include_usage": True}}
        assert (by_default.strips_usage, always.strips_usage) == (False, False)

    def test_a_non_completion_endpoint_is_relayed_untouched(self):
        body = {"model": MODEL, "input": "Hi", "stream": True}

        relay = stream_usage_relay(
            endpoint="embeddings", request_body=body, general_settings={}, supports_stream_options=lambda: True
        )

        assert relay.body == body
        assert relay.strips_usage is False


class TestWithoutInjectedUsageFrames:
    @pytest.mark.asyncio
    async def test_the_usage_frame_is_dropped_and_its_fingerprint_moves_onto_the_finish_frame(self):
        relayed = await _relayed([CONTENT, FINISH, USAGE, DONE])

        payloads = _payloads(relayed)
        assert [payload["choices"] for payload in payloads] == [
            [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}],
            [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        ]
        assert "usage" not in relayed.decode()
        assert payloads[1]["system_fingerprint"] == FINGERPRINT
        assert relayed.endswith(DONE)

    @pytest.mark.asyncio
    async def test_frames_split_across_upstream_reads_relay_the_same_bytes(self):
        whole = CONTENT + FINISH + USAGE + DONE

        assert await _relayed([whole[index : index + 1] for index in range(len(whole))]) == await _relayed([whole])

    @pytest.mark.asyncio
    async def test_a_stream_without_a_usage_frame_still_relays_its_finish_frame(self):
        assert await _relayed([CONTENT, FINISH, DONE]) == CONTENT + FINISH + DONE

    @pytest.mark.asyncio
    async def test_a_finish_frame_already_fingerprinted_is_not_held(self):
        stamped_finish = _frame(
            _chunk(
                "chatcmpl-1", choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}], system_fingerprint="fp-0"
            )
        )

        relayed = await _relayed([CONTENT, stamped_finish, USAGE, DONE])

        assert relayed == CONTENT + stamped_finish + DONE

    @pytest.mark.asyncio
    async def test_an_unterminated_tail_is_flushed_when_the_upstream_closes(self):
        assert await _relayed([CONTENT, b'data: {"partial": true']) == CONTENT + b'data: {"partial": true'
