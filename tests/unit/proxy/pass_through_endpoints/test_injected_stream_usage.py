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


def _content(index: int, **fields) -> bytes:
    return _frame(
        _chunk("chatcmpl-1", choices=[{"index": index, "delta": {"content": "Hi"}, "finish_reason": None}], **fields)
    )


def _finish(index: int, **fields) -> bytes:
    return _frame(_chunk("chatcmpl-1", choices=[{"index": index, "delta": {}, "finish_reason": "stop"}], **fields))


USAGE_COUNTS = {"prompt_tokens": 34, "completion_tokens": 4, "total_tokens": 38}
CONTENT = _content(0)
FINISH = _finish(0)
USAGE = _frame(_chunk("chatcmpl-1", choices=[], usage=USAGE_COUNTS, system_fingerprint=FINGERPRINT))
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

    def test_a_text_completion_stream_gets_usage_injected_and_stripped_like_a_chat_stream(self):
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

        assert by_default.body == always.body == {**body, "stream_options": {"include_usage": True}}
        assert (by_default.strips_usage, always.strips_usage) == (True, True)

    def test_turning_stream_usage_off_relays_the_stream_untouched(self):
        body = {"model": MODEL, "messages": [], "stream": True}

        relay = stream_usage_relay(
            endpoint="chat/completions",
            request_body=body,
            general_settings={"always_include_stream_usage": False},
            supports_stream_options=lambda: True,
        )

        assert relay.body == body
        assert relay.strips_usage is False

    @pytest.mark.parametrize(
        "stream_options",
        [True, "include_usage", ["include_usage"], {"include_usage": "true"}, {"include_usage": 1}],
    )
    def test_stream_options_the_proxy_cannot_read_reach_the_server_as_the_caller_sent_them(self, stream_options):
        body = {"model": MODEL, "messages": [], "stream": True, "stream_options": stream_options}

        relay = stream_usage_relay(
            endpoint="chat/completions", request_body=body, general_settings={}, supports_stream_options=lambda: True
        )

        assert relay.body == body
        assert relay.strips_usage is False

    def test_continuous_usage_stats_alone_is_not_turned_into_per_chunk_usage(self):
        body = {"model": MODEL, "messages": [], "stream": True, "stream_options": {"continuous_usage_stats": True}}

        relay = stream_usage_relay(
            endpoint="chat/completions", request_body=body, general_settings={}, supports_stream_options=lambda: True
        )

        assert relay.body == {**body, "stream_options": {"include_usage": True}}
        assert relay.strips_usage is True

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
    @pytest.mark.parametrize("frame_end", [b"\n\n", b"\r\n\r\n", b"\r\r"])
    async def test_frames_split_across_upstream_reads_relay_the_same_bytes(self, frame_end):
        whole = (CONTENT + FINISH + USAGE + DONE).replace(b"\n\n", frame_end)

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
    async def test_every_finish_frame_gets_the_fingerprint_when_the_stream_has_several_choices(self):
        relayed = await _relayed([_content(0), _content(1), _finish(0), _finish(1), USAGE, DONE])

        payloads = _payloads(relayed)
        finishes = {
            payload["choices"][0]["index"]: payload.get("system_fingerprint")
            for payload in payloads
            if payload["choices"][0]["finish_reason"]
        }
        assert finishes == {0: FINGERPRINT, 1: FINGERPRINT}
        assert "usage" not in relayed.decode()

    @pytest.mark.asyncio
    async def test_a_choice_that_finishes_while_another_still_streams_keeps_its_place(self):
        last_token_finish = _frame(
            _chunk("chatcmpl-1", choices=[{"index": 0, "delta": {"content": "!"}, "finish_reason": "stop"}])
        )
        upstream = [_content(0), _content(1), last_token_finish, _content(1), _content(1), _finish(1), USAGE, DONE]

        relayed = await _relayed(upstream)

        assert [payload["choices"] for payload in _payloads(relayed)] == [
            payload["choices"] for payload in _payloads(b"".join(upstream[:-2]))
        ]
        assert _payloads(relayed)[-1]["system_fingerprint"] == FINGERPRINT

    @pytest.mark.asyncio
    async def test_a_server_that_forces_usage_onto_every_chunk_is_relayed_untouched(self):
        forced = [
            _content(0, usage=USAGE_COUNTS),
            _finish(0, usage=USAGE_COUNTS),
            USAGE,
            DONE,
        ]

        assert await _relayed(forced) == b"".join(forced)

    @pytest.mark.asyncio
    async def test_closing_the_relay_early_closes_the_upstream(self):
        closed = []

        async def upstream() -> AsyncIterator[bytes]:
            try:
                yield CONTENT
                yield FINISH
            finally:
                closed.append(True)

        relay = without_injected_usage_frames(upstream())
        assert await anext(relay) == CONTENT
        await relay.aclose()

        assert closed == [True]

    @pytest.mark.asyncio
    async def test_an_unterminated_tail_is_flushed_when_the_upstream_closes(self):
        assert await _relayed([CONTENT, b'data: {"partial": true']) == CONTENT + b'data: {"partial": true'
