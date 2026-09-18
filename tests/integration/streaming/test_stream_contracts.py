import asyncio
import json
import threading
import uuid
from typing import Final

import pytest
from hypothesis import Phase, example, given, settings, strategies as st
import openai
from openai import OpenAI

from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, wire_server


def frame(identity: str, delta: dict, *, finish: str | None = None) -> bytes:
    value: Final = {"id": identity, "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return b"data: " + json.dumps(value, ensure_ascii=False).encode() + b"\n\n"


def text_stream(identity: str) -> tuple[bytes, ...]:
    usage: Final = {"id": identity, "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini", "choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}}
    return (frame(identity, {"role": "assistant", "content": "Hello "}), frame(identity, {"content": "雪 café"}), frame(identity, {}, finish="stop"), b"data: " + json.dumps(usage).encode() + b"\n\n", b"data: [DONE]\n\n")


@pytest.mark.covers("other.streaming.byte_partitions.preserve_text_identity_and_usage")
def test_generated_tcp_partitions_preserve_unicode_text_identity_and_final_usage() -> None:
    import litellm

    body: Final = b"".join(text_stream("stream-partition-control"))

    @settings(max_examples=20, deadline=None, database=None, phases=(Phase.explicit, Phase.generate, Phase.shrink))
    @example(cuts=tuple(range(1, len(body))))
    @example(cuts=())
    @given(cuts=st.lists(st.integers(min_value=1, max_value=len(body) - 1), max_size=35, unique=True).map(tuple))
    def check(cuts: tuple[int, ...]) -> None:
        boundaries: Final = (0, *sorted(cuts), len(body))
        pieces: Final = tuple(body[left:right] for left, right in zip(boundaries, boundaries[1:]))
        with wire_server(lambda request: Reply(content_type="text/event-stream", chunks=pieces)) as wire:
            stream: Final = litellm.completion(model="openai/gpt-4o-mini", api_base=wire.url + "/v1", api_key="synthetic-stream-key", messages=[{"role": "user", "content": "partition control"}], stream=True, stream_options={"include_usage": True}, timeout=5, num_retries=0)
            try:
                chunks: Final = tuple(stream)
            finally:
                asyncio.run(stream.aclose())
            assert "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices) == "Hello 雪 café"
            assert {chunk.id for chunk in chunks} == {"stream-partition-control"}
            assert [choice.finish_reason for chunk in chunks for choice in chunk.choices if choice.finish_reason] == ["stop"]
            usages: Final = tuple(chunk.usage for chunk in chunks if getattr(chunk, "usage", None) is not None)
            assert len(usages) == 1
            assert usages[0].prompt_tokens == 11 and usages[0].completion_tokens == 4
            assert len(wire.drain()) == 1

    check()


@pytest.mark.covers("other.streaming.tools.fragmented_calls_keep_independent_arguments")
def test_fragmented_tool_names_and_arguments_keep_each_call_identity() -> None:
    import litellm

    identity: Final = "stream-tools-control"
    deltas: Final = (
        {"role": "assistant", "tool_calls": [{"index": 0, "id": "call-add", "type": "function", "function": {"name": "ad", "arguments": ""}}, {"index": 1, "id": "call-multiply", "type": "function", "function": {"name": "multi", "arguments": ""}}]},
        {"tool_calls": [{"index": 1, "function": {"name": "ply", "arguments": '{"x":3,'}}, {"index": 0, "function": {"arguments": '{"x":1,'}}]},
        {"tool_calls": [{"index": 0, "function": {"name": "d", "arguments": '"y":2}'}}, {"index": 1, "function": {"arguments": '"y":4}'}}]},
    )
    frames: Final = (*tuple(frame(identity, delta) for delta in deltas), frame(identity, {}, finish="tool_calls"), b"data: [DONE]\n\n")
    with wire_server(lambda request: Reply(content_type="text/event-stream", chunks=frames)) as wire:
        stream: Final = litellm.completion(model="openai/gpt-4o-mini", api_base=wire.url + "/v1", api_key="synthetic-stream-key", messages=[{"role": "user", "content": "tool control"}], stream=True, timeout=5, num_retries=0)
        try:
            chunks: Final = tuple(stream)
        finally:
            asyncio.run(stream.aclose())
        events: Final = tuple((choice.index, tool) for chunk in chunks for choice in chunk.choices for tool in (choice.delta.tool_calls or ()))
        for index, name, call_id, arguments in ((0, "add", "call-add", {"x": 1, "y": 2}), (1, "multiply", "call-multiply", {"x": 3, "y": 4})):
            selected: Final = tuple(tool for choice, tool in events if (choice, tool.index) == (0, index))
            assert "".join(tool.id or "" for tool in selected) == call_id
            assert "".join(tool.function.name or "" for tool in selected) == name
            assert json.loads("".join(tool.function.arguments or "" for tool in selected)) == arguments
        assert {tool.index for _, tool in events} == {0, 1}
        assert [choice.finish_reason for chunk in chunks for choice in chunk.choices if choice.finish_reason] == ["tool_calls"]
        assert len(wire.drain()) == 1


@pytest.mark.covers("other.streaming.usage.client_visibility_preserves_persisted_accounting")
def test_proxy_stream_usage_visibility_keeps_exact_persisted_charge(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        for include in (None, False, True):
            identity: Final = "stream-usage-" + uuid.uuid4().hex
            with wire_server(lambda request, identity=identity: Reply(content_type="text/event-stream", chunks=text_stream(identity))) as wire:
                model: Final = scenario.model(api_base=wire.url + "/v1", input_cost_per_token=0.001, output_cost_per_token=0.002)
                with OpenAI(api_key=gateway.key, base_url=str(gateway.client.base_url), timeout=5, max_retries=0) as client:
                    stream: Final = client.chat.completions.create(model=model, messages=[{"role": "user", "content": identity}], stream=True, **({} if include is None else {"stream_options": {"include_usage": include}}))
                    with stream:
                        chunks: Final = tuple(stream)
                assert "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices) == "Hello 雪 café"
                assert {chunk.id for chunk in chunks} == {identity}
                usages: Final = tuple(chunk.usage for chunk in chunks if chunk.usage is not None)
                assert len(usages) == (1 if include else 0)
                if include:
                    assert usages[0].prompt_tokens == 11 and usages[0].completion_tokens == 4
                requests: Final = wire.drain()
                assert len(requests) == 1
                assert json.loads(requests[0].body)["stream_options"]["include_usage"] is True
                rows: Final = eventually(lambda identity=identity: read_rows('SELECT spend, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (identity,)), lambda values: len(values) == 1, seconds=70)
                assert rows[0]["prompt_tokens"] == 11 and rows[0]["completion_tokens"] == 4
                assert float(rows[0]["spend"]) == pytest.approx(0.019)


@pytest.mark.covers("other.streaming.failure.truncated_transport_raises_and_control_recovers")
def test_truncated_http_stream_is_an_error_and_next_stream_succeeds() -> None:
    import litellm

    for truncated in (True, False):
        with wire_server(lambda request, truncated=truncated: Reply(content_type="text/event-stream", chunks=text_stream("stream-truncated"), abort_after=1 if truncated else None)) as wire:
            stream: Final = litellm.completion(model="openai/gpt-4o-mini", api_base=wire.url + "/v1", api_key="synthetic-stream-key", messages=[{"role": "user", "content": "truncation control"}], stream=True, timeout=5, num_retries=0)
            try:
                if truncated:
                    with pytest.raises(litellm.exceptions.MidStreamFallbackError, match="incomplete chunked read") as failure:
                        tuple(stream)
                    assert isinstance(failure.value.original_exception, litellm.APIConnectionError)
                    assert failure.value.generated_content == "Hello "
                    assert failure.value.is_pre_first_chunk is False
                else:
                    chunks: Final = tuple(stream)
                    assert "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices) == "Hello 雪 café"
                    assert any(choice.finish_reason == "stop" for chunk in chunks for choice in chunk.choices)
            finally:
                asyncio.run(stream.aclose())
            assert len(wire.drain()) == 1


@pytest.mark.covers("other.streaming.cancellation.closes_actual_provider_connection")
def test_client_cancellation_releases_the_actual_provider_connection() -> None:
    import litellm

    gate: Final = threading.Event()
    frames: Final = (frame("stream-cancel", {"role": "assistant", "content": "first"}), b":" + b"x" * 4_000_000 + b"\n\n", b"data: [DONE]\n\n")
    with wire_server(lambda request: Reply(content_type="text/event-stream", chunks=frames, gate_after_first=gate)) as wire:
        stream: Final = litellm.completion(model="openai/gpt-4o-mini", api_base=wire.url + "/v1", api_key="synthetic-stream-key", messages=[{"role": "user", "content": "cancellation control"}], stream=True, timeout=5, num_retries=0)
        try:
            first: Final = next(stream)
            assert first.choices[0].delta.content == "first"
        finally:
            try:
                asyncio.run(stream.aclose())
            finally:
                gate.set()
        assert wire.disconnected.get(timeout=5) == "/v1/chat/completions"
        assert len(wire.drain()) == 1


def responses_frame(value: dict) -> bytes:
    return b"event: " + value["type"].encode() + b"\ndata: " + json.dumps(value, ensure_ascii=False).encode() + b"\n\n"


def responses_stream_cut_by(error: dict) -> tuple[bytes, ...]:
    created: Final = {"id": "resp-cut", "object": "response", "created_at": 1, "model": "gpt-5", "status": "in_progress", "output": [], "parallel_tool_calls": False, "tool_choice": "auto", "tools": []}
    return (
        responses_frame({"type": "response.created", "sequence_number": 0, "response": created}),
        responses_frame({"type": "response.output_text.delta", "sequence_number": 1, "item_id": "msg-cut", "output_index": 0, "content_index": 0, "delta": "Hello "}),
        responses_frame({"type": "error", "sequence_number": 2, "error": error}),
    )


@pytest.mark.covers("other.streaming.failure.in_stream_error_code_outranks_generic_type")
def test_responses_in_stream_error_classifies_by_code_not_generic_type() -> None:
    """A gateway cuts a Responses stream in transit and reports it with a transport code the status table
    does not know (`request_timeout`) under the generic type `invalid_request_error`. The request was
    accepted and two events were already delivered, so the caller must not be told its request was
    malformed: an unrecognised code is an unknown condition, and an unknown condition is retriable. A code
    the table does know as a client fault still decides, under the same generic type."""
    import litellm

    cut: Final = {"type": "invalid_request_error", "code": "request_timeout", "message": "stream disconnected before completion: stream closed before response.completed", "param": None}
    too_long: Final = {"type": "invalid_request_error", "code": "context_length_exceeded", "message": "too long", "param": None}
    for error, status in ((cut, 500), (too_long, 400)):
        with wire_server(lambda request, error=error: Reply(content_type="text/event-stream", chunks=responses_stream_cut_by(error))) as wire:
            stream: Final = litellm.responses(model="openai/gpt-5", api_base=wire.url + "/v1", api_key="synthetic-stream-key", input="in-stream error control", stream=True, timeout=5, num_retries=0)
            with pytest.raises(openai.APIError) as failure:
                tuple(stream)
            assert failure.value.status_code == status
            if status == 500:
                assert isinstance(failure.value, litellm.exceptions.MidStreamFallbackError)
                assert isinstance(failure.value.original_exception, litellm.InternalServerError)
                assert failure.value.generated_content == "Hello "
                assert failure.value.is_pre_first_chunk is False
                assert "invalid_request_error" not in str(failure.value)
            else:
                assert isinstance(failure.value, litellm.BadRequestError)
                assert not isinstance(failure.value, litellm.exceptions.MidStreamFallbackError)
                assert (failure.value.body["type"], failure.value.body["code"]) == (error["type"], error["code"])
            assert error["message"] in str(failure.value)
            assert len(wire.drain()) == 1
