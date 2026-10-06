import asyncio
import json
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from queue import SimpleQueue
from typing import Final

import pytest
import yaml
from hypothesis import Phase, example, given, settings
from hypothesis import strategies as st
from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from openai import OpenAI
from pydantic import JsonValue, TypeAdapter

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def frame(identity: str, delta: dict, *, finish: str | None = None) -> bytes:
    value: Final = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return b"data: " + json.dumps(value, ensure_ascii=False).encode() + b"\n\n"


def text_stream(identity: str) -> tuple[bytes, ...]:
    usage: Final = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [],
        "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
    }
    return (
        frame(identity, {"role": "assistant", "content": "Hello "}),
        frame(identity, {"content": "雪 café"}),
        frame(identity, {}, finish="stop"),
        b"data: " + json.dumps(usage).encode() + b"\n\n",
        b"data: [DONE]\n\n",
    )


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
            stream: Final = litellm.completion(
                model="openai/gpt-4o-mini",
                api_base=wire.url + "/v1",
                api_key="synthetic-stream-key",
                messages=[{"role": "user", "content": "partition control"}],
                stream=True,
                stream_options={"include_usage": True},
                timeout=5,
                num_retries=0,
            )
            try:
                chunks: Final = tuple(stream)
            finally:
                asyncio.run(stream.aclose())
            assert (
                "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices) == "Hello 雪 café"
            )
            assert {chunk.id for chunk in chunks} == {"stream-partition-control"}
            assert [choice.finish_reason for chunk in chunks for choice in chunk.choices if choice.finish_reason] == [
                "stop"
            ]
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
        {
            "role": "assistant",
            "tool_calls": [
                {"index": 0, "id": "call-add", "type": "function", "function": {"name": "ad", "arguments": ""}},
                {"index": 1, "id": "call-multiply", "type": "function", "function": {"name": "multi", "arguments": ""}},
            ],
        },
        {
            "tool_calls": [
                {"index": 1, "function": {"name": "ply", "arguments": '{"x":3,'}},
                {"index": 0, "function": {"arguments": '{"x":1,'}},
            ]
        },
        {
            "tool_calls": [
                {"index": 0, "function": {"name": "d", "arguments": '"y":2}'}},
                {"index": 1, "function": {"arguments": '"y":4}'}},
            ]
        },
    )
    frames: Final = (
        *tuple(frame(identity, delta) for delta in deltas),
        frame(identity, {}, finish="tool_calls"),
        b"data: [DONE]\n\n",
    )
    with wire_server(lambda request: Reply(content_type="text/event-stream", chunks=frames)) as wire:
        stream: Final = litellm.completion(
            model="openai/gpt-4o-mini",
            api_base=wire.url + "/v1",
            api_key="synthetic-stream-key",
            messages=[{"role": "user", "content": "tool control"}],
            stream=True,
            timeout=5,
            num_retries=0,
        )
        try:
            chunks: Final = tuple(stream)
        finally:
            asyncio.run(stream.aclose())
        events: Final = tuple(
            (choice.index, tool)
            for chunk in chunks
            for choice in chunk.choices
            for tool in (choice.delta.tool_calls or ())
        )
        for index, name, call_id, arguments in (
            (0, "add", "call-add", {"x": 1, "y": 2}),
            (1, "multiply", "call-multiply", {"x": 3, "y": 4}),
        ):
            selected: Final = tuple(tool for choice, tool in events if (choice, tool.index) == (0, index))
            assert "".join(tool.id or "" for tool in selected) == call_id
            assert "".join(tool.function.name or "" for tool in selected) == name
            assert json.loads("".join(tool.function.arguments or "" for tool in selected)) == arguments
        assert {tool.index for _, tool in events} == {0, 1}
        assert [choice.finish_reason for chunk in chunks for choice in chunk.choices if choice.finish_reason] == [
            "tool_calls"
        ]
        assert len(wire.drain()) == 1


@pytest.mark.covers("other.streaming.usage.client_visibility_preserves_persisted_accounting")
def test_proxy_stream_usage_visibility_keeps_exact_persisted_charge(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        for include in (None, False, True):
            identity: Final = "stream-usage-" + uuid.uuid4().hex
            with wire_server(
                lambda request, identity=identity: Reply(content_type="text/event-stream", chunks=text_stream(identity))
            ) as wire:
                model: Final = scenario.model(
                    api_base=wire.url + "/v1", input_cost_per_token=0.001, output_cost_per_token=0.002
                )
                with OpenAI(
                    api_key=gateway.key, base_url=str(gateway.client.base_url), timeout=5, max_retries=0
                ) as client:
                    stream: Final = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": identity}],
                        stream=True,
                        **({} if include is None else {"stream_options": {"include_usage": include}}),
                    )
                    with stream:
                        chunks: Final = tuple(stream)
                assert (
                    "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices)
                    == "Hello 雪 café"
                )
                assert {chunk.id for chunk in chunks} == {identity}
                usages: Final = tuple(chunk.usage for chunk in chunks if chunk.usage is not None)
                assert len(usages) == (1 if include else 0)
                if include:
                    assert usages[0].prompt_tokens == 11 and usages[0].completion_tokens == 4
                requests: Final = wire.drain()
                assert len(requests) == 1
                assert json.loads(requests[0].body)["stream_options"]["include_usage"] is True
                rows: Final = eventually(
                    lambda identity=identity: read_rows(
                        'SELECT spend, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                        (identity,),
                    ),
                    lambda values: len(values) == 1,
                    seconds=70,
                )
                assert rows[0]["prompt_tokens"] == 11 and rows[0]["completion_tokens"] == 4
                assert float(rows[0]["spend"]) == pytest.approx(0.019)


@pytest.mark.covers("other.streaming.messages_bridge.empty_choices_usage_chunk_completes_stream")
def test_messages_stream_completes_through_trailing_empty_choices_usage_chunk(gateway: Gateway) -> None:
    identity: Final = "messages-empty-choices-" + uuid.uuid4().hex
    metadata: Final = (
        b"data: "
        + json.dumps(
            {
                "id": identity,
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [],
                "prompt_filter_results": [{"prompt_index": 0, "content_filter_results": {}}],
            },
            ensure_ascii=False,
        ).encode()
        + b"\n\n"
    )
    frames: Final = (metadata, *text_stream(identity))
    with (
        wire_server(lambda request: Reply(content_type="text/event-stream", chunks=frames)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model="azure/gpt-4o-mini", api_base=wire.url + "/v1")
        with gateway.client.stream(
            "POST",
            "/v1/messages",
            json={
                "model": model,
                "max_tokens": 64,
                "stream": True,
                "messages": [{"role": "user", "content": identity}],
            },
            headers={"Authorization": f"Bearer {gateway.key}"},
        ) as response:
            assert response.status_code == 200, response.read().decode()
            events: Final = tuple(
                json.loads(line.removeprefix("data: ")) for line in response.iter_lines() if line.startswith("data: ")
            )
    assert tuple(event["type"] for event in events) == (
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ), f"observed events: {events!r}"
    assert (
        "".join(event["delta"]["text"] for event in events if event["type"] == "content_block_delta") == "Hello 雪 café"
    )
    message_delta: Final = next(event for event in events if event["type"] == "message_delta")
    assert message_delta["usage"] == {"input_tokens": 11, "output_tokens": 4}
    requests: Final = wire.drain()
    assert len(requests) == 1
    outbound: Final = json.loads(requests[0].body)
    assert outbound["stream"] is True and outbound["stream_options"] == {"include_usage": True}, (
        f"observed outbound body: {outbound!r}"
    )


def reasoning_first_stream(identity: str) -> tuple[bytes, ...]:
    usage: Final = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [],
        "usage": {"prompt_tokens": 11, "completion_tokens": 6, "total_tokens": 17},
    }
    return (
        frame(identity, {"role": "assistant", "content": None, "reasoning_content": "Let me "}),
        frame(identity, {"content": None, "reasoning_content": "think."}),
        frame(identity, {"content": "Hello "}),
        frame(identity, {"content": "there"}),
        frame(identity, {}, finish="stop"),
        b"data: " + json.dumps(usage).encode() + b"\n\n",
        b"data: [DONE]\n\n",
    )


@pytest.mark.covers("streaming.messages_bridge.reasoning_content_only_chunks_open_a_thinking_block_first")
def test_messages_stream_opens_thinking_block_at_index_zero_for_reasoning_content_only_chunks(
    gateway: Gateway,
) -> None:
    identity: Final = "messages-reasoning-first-" + uuid.uuid4().hex
    with (
        wire_server(
            lambda request: Reply(content_type="text/event-stream", chunks=reasoning_first_stream(identity))
        ) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model="hosted_vllm/reasoning-model", api_base=wire.url + "/v1")
        with gateway.client.stream(
            "POST",
            "/v1/messages",
            json={
                "model": model,
                "max_tokens": 64,
                "stream": True,
                "messages": [{"role": "user", "content": identity}],
            },
            headers={"Authorization": f"Bearer {gateway.key}"},
        ) as response:
            text: Final = response.read().decode()
    assert response.status_code == 200, text
    assert response.headers["content-type"].startswith("text/event-stream"), text
    events: Final = tuple(json.loads(line) for line in sse_data_lines(text))
    blocks: Final = tuple(
        (event["index"], event.get("content_block") or event["delta"])
        for event in events
        if event["type"] in ("content_block_start", "content_block_delta")
    )
    assert blocks == (
        (0, {"type": "thinking", "thinking": "", "signature": ""}),
        (0, {"type": "thinking_delta", "thinking": "Let me "}),
        (0, {"type": "thinking_delta", "thinking": "think."}),
        (1, {"type": "text", "text": ""}),
        (1, {"type": "text_delta", "text": "Hello "}),
        (1, {"type": "text_delta", "text": "there"}),
    ), text
    assert tuple(event["type"] for event in events) == (
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ), text
    message_delta: Final = next(event for event in events if event["type"] == "message_delta")
    assert message_delta["usage"] == {"input_tokens": 11, "output_tokens": 6}, text
    requests: Final = wire.drain()
    assert len(requests) == 1
    outbound: Final = json.loads(requests[0].body)
    assert outbound["stream"] is True and outbound["messages"] == [{"role": "user", "content": identity}], outbound


@pytest.mark.covers("other.streaming.responses_bridge.empty_choices_chunks_complete_stream")
def test_responses_stream_completes_through_empty_choices_metadata_and_usage_chunks(gateway: Gateway) -> None:
    identity: Final = "responses-empty-choices-" + uuid.uuid4().hex
    metadata: Final = (
        b"data: "
        + json.dumps(
            {
                "id": identity,
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [],
                "prompt_filter_results": [{"prompt_index": 0, "content_filter_results": {}}],
            },
            ensure_ascii=False,
        ).encode()
        + b"\n\n"
    )
    frames: Final = (metadata, *text_stream(identity))
    with (
        wire_server(lambda request: Reply(content_type="text/event-stream", chunks=frames)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model="deepseek/gpt-4o-mini", api_base=wire.url + "/v1")
        with gateway.client.stream(
            "POST",
            "/v1/responses",
            json={"model": model, "input": identity, "stream": True},
            headers={"Authorization": f"Bearer {gateway.key}"},
        ) as response:
            assert response.status_code == 200, response.read().decode()
            events: Final = tuple(
                json.loads(line.removeprefix("data: "))
                for line in response.iter_lines()
                if line.startswith("data: ") and line != "data: [DONE]"
            )
    assert (
        "".join(event["delta"] for event in events if event["type"] == "response.output_text.delta") == "Hello 雪 café"
    ), f"observed events: {events!r}"
    assert tuple(event["type"] for event in events if event["type"] != "response.output_text.delta") == (
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ), f"observed events: {events!r}"
    assert events[-1]["type"] == "response.completed"
    assert events[-1]["response"]["usage"] == {
        "input_tokens": 11,
        "output_tokens": 4,
        "output_tokens_details": {"reasoning_tokens": 0, "text_tokens": 4},
        "total_tokens": 15,
    }
    requests: Final = wire.drain()
    assert len(requests) == 1
    outbound: Final = json.loads(requests[0].body)
    assert outbound["stream"] is True and outbound["stream_options"] == {"include_usage": True}, (
        f"observed outbound body: {outbound!r}"
    )


def provider_cost_object_stream(identity: str, total_cost: float) -> tuple[bytes, ...]:
    cost: Final = {
        "input_tokens_cost": 0.0001,
        "output_tokens_cost": 0.0002,
        "request_cost": 0.012,
        "total_cost": total_cost,
    }
    usage: Final = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "sonar",
        "choices": [],
        "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15, "cost": cost},
    }
    return (
        frame(identity, {"role": "assistant", "content": "Hello "}),
        frame(identity, {"content": "from search"}),
        frame(identity, {}, finish="stop"),
        b"data: " + json.dumps(usage).encode() + b"\n\n",
        b"data: [DONE]\n\n",
    )


def sse_data_lines(text: str) -> tuple[str, ...]:
    return tuple(line.removeprefix("data: ") for line in text.splitlines() if line.startswith("data: "))


def _native_anthropic_message(model: str, identity: str, text: str) -> dict[str, object]:
    return {
        "id": identity,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 11, "output_tokens": 4},
    }


def _native_responses_response(model: str, identity: str, text: str) -> dict[str, object]:
    return {
        "id": identity,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": model,
        "output": [
            {
                "id": "msg_" + identity,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15},
    }


def _native_sse_event(event: dict[str, object]) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


def _native_anthropic_stream(model: str, identity: str, text: str) -> tuple[bytes, ...]:
    return (
        _native_sse_event(
            {
                "type": "message_start",
                "message": {
                    **_native_anthropic_message(model, identity, ""),
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 11, "output_tokens": 0},
                },
            }
        ),
        _native_sse_event({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        _native_sse_event(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            }
        ),
        _native_sse_event({"type": "content_block_stop", "index": 0}),
        _native_sse_event(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 4},
            }
        ),
        _native_sse_event({"type": "message_stop"}),
    )


def _native_responses_stream(model: str, identity: str, text: str) -> tuple[bytes, ...]:
    response: Final = _native_responses_response(model, identity, text)
    return (
        _native_sse_event(
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {**response, "status": "in_progress", "output": []},
            }
        ),
        _native_sse_event(
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "item_id": "msg_" + identity,
                "output_index": 0,
                "content_index": 0,
                "delta": text,
            }
        ),
        _native_sse_event({"type": "response.completed", "sequence_number": 2, "response": response}),
    )


@pytest.mark.covers("other.streaming.usage.provider_cost_object_completes_stream_and_bills_total_cost")
def test_perplexity_stream_with_cost_breakdown_object_completes_and_bills_total_cost(gateway: Gateway) -> None:
    identity: Final = "stream-cost-object-" + uuid.uuid4().hex
    total_cost: Final = 0.0123
    with (
        gateway.scenario() as scenario,
        wire_server(
            lambda request: Reply(
                content_type="text/event-stream", chunks=provider_cost_object_stream(identity, total_cost)
            )
        ) as wire,
    ):
        model: Final = scenario.model(model="perplexity/sonar", api_base=wire.url + "/v1")
        with gateway.client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": identity}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            headers={"Authorization": f"Bearer {gateway.key}"},
        ) as response:
            text: Final = response.read().decode()
        assert response.status_code == 200, text
        lines: Final = sse_data_lines(text)
        assert lines[-1] == "[DONE]", text
        events: Final = tuple(json.loads(line) for line in lines[:-1])
        assert [event for event in events if "error" in event] == [], text
        assert (
            "".join(choice["delta"].get("content") or "" for event in events for choice in event["choices"])
            == "Hello from search"
        ), text
        assert [
            choice.get("finish_reason")
            for event in events
            for choice in event["choices"]
            if choice.get("finish_reason")
        ] == ["stop"], text
        usages: Final = tuple(event["usage"] for event in events if event.get("usage") is not None)
        assert len(usages) == 1, text
        assert (usages[0]["prompt_tokens"], usages[0]["completion_tokens"], usages[0]["total_tokens"]) == (11, 4, 15), (
            text
        )
        requests: Final = wire.drain()
        assert len(requests) == 1
        outbound: Final = json.loads(requests[0].body)
        assert outbound["model"] == "sonar" and outbound["stream"] is True, outbound
        assert outbound["messages"] == [{"role": "user", "content": identity}], outbound
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT spend, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                (identity,),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert (rows[0]["prompt_tokens"], rows[0]["completion_tokens"]) == (11, 4)
        assert float(rows[0]["spend"]) == pytest.approx(total_cost)


@pytest.mark.covers(
    "other.streaming.fallback.empty_leading_chunk_then_disconnect_streams_fallback_with_usage_and_spend"
)
def test_primary_stream_with_empty_first_chunk_then_disconnect_falls_back_and_bills_the_fallback(
    gateway: Gateway,
    tmp_path: Path,
) -> None:
    identity: Final = "stream-empty-fallback-" + uuid.uuid4().hex
    empty_first: Final = (
        b"data: "
        + json.dumps(
            {
                "id": identity + "-primary",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [],
                "usage": {"prompt_tokens": 11, "completion_tokens": 0, "total_tokens": 11},
            }
        ).encode()
        + b"\n\n"
    )
    with (
        wire_server(
            lambda request: Reply(
                content_type="text/event-stream",
                chunks=(empty_first, b":" + b"x" * 4_000_000 + b"\n\n", empty_first),
                abort_after=2,
            )
        ) as primary,
        wire_server(lambda request: Reply(content_type="text/event-stream", chunks=text_stream(identity))) as fallback,
    ):
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["model_list"] = [
            {
                "model_name": name,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "synthetic-fallback-key",
                    "api_base": server.url + "/v1",
                    "input_cost_per_token": 0.001,
                    "output_cost_per_token": 0.002,
                },
            }
            for name, server in (("primary", primary), ("fallback", fallback))
        ]
        config["router_settings"] = {
            "num_retries": 0,
            "disable_cooldowns": True,
            "fallbacks": [{"primary": ["fallback"]}],
        }
        path: Final = tmp_path / "fallbacks.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate:
            body: Final = {
                "model": "primary",
                "messages": [{"role": "user", "content": identity}],
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            with candidate.client.stream(
                "POST", "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {candidate.key}"}
            ) as response:
                lines: Final = tuple(line for line in response.iter_lines() if line.startswith("data:"))
            assert response.status_code == 200, lines
            assert lines[-1] == "data: [DONE]", lines
            events: Final = tuple(json.loads(line.removeprefix("data:")) for line in lines[:-1])
            assert all("error" not in event for event in events), lines
            assert (
                "".join(choice["delta"].get("content") or "" for event in events for choice in event["choices"])
                == "Hello 雪 café"
            ), lines
            usages: Final = tuple(event["usage"] for event in events if event.get("usage") is not None)
            assert (usages[-1]["prompt_tokens"], usages[-1]["completion_tokens"]) == (11, 4), lines
            assert tuple(
                json.loads(request.body)["messages"]
                for request in primary.drain()
                if request.target.endswith("/chat/completions")
            ) == (body["messages"],)
            assert tuple(
                json.loads(request.body)["messages"]
                for request in fallback.drain()
                if request.target.endswith("/chat/completions")
            ) == (body["messages"],)
            rows: Final = eventually(
                lambda: read_rows(
                    'SELECT spend, prompt_tokens, completion_tokens, status FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                    (identity,),
                ),
                lambda values: len(values) == 1,
                seconds=70,
            )
            assert (rows[0]["prompt_tokens"], rows[0]["completion_tokens"], rows[0]["status"]) == (11, 4, "success"), (
                rows
            )
            assert float(rows[0]["spend"]) == pytest.approx(0.019), rows


@pytest.mark.parametrize(
    ("route", "failure_mode"),
    (
        ("messages", "non-streaming"),
        ("messages", "before-first-byte"),
        ("messages", "mid-stream"),
        ("messages", "request-body-fallbacks"),
        ("responses", "non-streaming"),
        ("responses", "before-first-byte"),
        ("responses", "mid-stream"),
        ("responses", "request-body-fallbacks"),
    ),
)
def test_native_messages_and_responses_routes_fall_back_and_bill_the_fallback(
    gateway: Gateway,
    tmp_path: Path,
    route: str,
    failure_mode: str,
) -> None:
    identity: Final = f"native-fallback-{route}-{failure_mode}-{uuid.uuid4().hex}"
    is_messages: Final = route == "messages"
    path: Final = "/v1/messages" if is_messages else "/v1/responses"
    primary_model: Final = "claude-sonnet-4-5-20250929" if is_messages else "gpt-5.6"
    fallback_model: Final = "claude-opus-4-5-20251101" if is_messages else "gpt-5.6"
    stream: Final = failure_mode in ("before-first-byte", "mid-stream")
    client_body: Final = (
        {
            "model": "primary",
            "max_tokens": 12,
            "messages": [{"role": "user", "content": identity}],
            "stream": stream,
            **({"fallbacks": ["fallback"]} if failure_mode == "request-body-fallbacks" else {}),
        }
        if is_messages
        else {
            "model": "primary",
            "input": identity,
            "stream": stream,
            **({"fallbacks": ["fallback"]} if failure_mode == "request-body-fallbacks" else {}),
        }
    )
    primary_body: Final = (
        {
            "model": primary_model,
            "max_tokens": 12,
            "messages": [{"role": "user", "content": identity}],
            "stream": stream,
        }
        if is_messages
        else {"model": primary_model, "input": identity, "stream": stream}
    )
    fallback_body: Final = {**primary_body, "model": fallback_model}
    call_order: Final[SimpleQueue[str]] = SimpleQueue()
    received: Final[SimpleQueue[tuple[str, Request]]] = SimpleQueue()
    fallback_text: Final = f"fallback response {identity}"
    primary_text: Final = f"primary response {identity}"

    def response_for(model: str, text: str) -> Reply:
        if is_messages:
            if stream:
                return Reply(content_type="text/event-stream", chunks=_native_anthropic_stream(model, identity, text))
            return Reply(body=json.dumps(_native_anthropic_message(model, identity, text)).encode())
        if stream:
            return Reply(content_type="text/event-stream", chunks=_native_responses_stream(model, identity, text))
        return Reply(body=json.dumps(_native_responses_response(model, identity, text)).encode())

    def upstream(name: str, model: str) -> Callable[[Request], Reply]:
        def handle(request: Request) -> Reply:
            if request.method == "GET":
                return Reply(body=b'{"object":"list","data":[]}')
            assert request.method == "POST" and request.target == path, (request.method, request.target)
            expected_body: Final = primary_body if name == "primary" else fallback_body
            assert json.loads(request.body) == expected_body, request.body.decode()
            if is_messages:
                assert request.headers.get("x-api-key") == "synthetic-native-fallback-key", request.headers
                assert request.headers.get("anthropic-version") == "2023-06-01", request.headers
                assert "authorization" not in request.headers, request.headers
            else:
                assert request.headers.get("authorization") == "Bearer synthetic-native-fallback-key", request.headers
            call_order.put(name)
            received.put((name, request))
            if name == "primary" and failure_mode == "mid-stream":
                primary_events: Final = (
                    _native_anthropic_stream(primary_model, identity + "-primary", primary_text)
                    if is_messages
                    else _native_responses_stream(primary_model, identity + "-primary", primary_text)
                )
                first_event: Final = primary_events[0]
                second_event: Final = (
                    primary_events[1]
                    if is_messages
                    else _native_sse_event(
                        {
                            "type": "response.in_progress",
                            "sequence_number": 1,
                            "response": {
                                **_native_responses_response(primary_model, identity + "-primary", primary_text),
                                "status": "in_progress",
                                "output": [],
                            },
                        }
                    )
                )
                return Reply(
                    content_type="text/event-stream",
                    chunks=(first_event, second_event, first_event),
                    abort_after=2,
                    pause_between_chunks=0.05,
                )
            if name == "primary" and failure_mode in (
                "non-streaming",
                "before-first-byte",
                "request-body-fallbacks",
            ):
                return Reply(
                    status=500,
                    body=b'{"error":{"message":"synthetic primary failure","type":"server_error","code":"internal_error"}}',
                )
            return response_for(model, fallback_text if name == "fallback" else primary_text)

        return handle

    with (
        wire_server(upstream("primary", primary_model)) as primary,
        wire_server(upstream("fallback", fallback_model)) as fallback,
    ):
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["model_list"] = [
            {
                "model_name": model_group,
                "litellm_params": {
                    "model": f"{'anthropic' if is_messages else 'openai'}/{model}",
                    "api_key": "synthetic-native-fallback-key",
                    "api_base": server.url if is_messages else server.url + "/v1",
                    "input_cost_per_token": 0.001,
                    "output_cost_per_token": 0.002,
                },
            }
            for model_group, model, server in (
                ("primary", primary_model, primary),
                ("fallback", fallback_model, fallback),
            )
        ]
        config["router_settings"] = {
            "num_retries": 0,
            "disable_cooldowns": True,
            "fallbacks": [] if failure_mode == "request-body-fallbacks" else [{"primary": ["fallback"]}],
        }
        proxy_config: Final = tmp_path / f"native-{route}-{failure_mode}.yaml"
        proxy_config.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=proxy_config) as candidate:
            request_headers: Final = {
                "x-litellm-call-id": identity,
                **(
                    {"x-api-key": candidate.key, "anthropic-version": "2023-06-01"}
                    if is_messages
                    else {"Authorization": f"Bearer {candidate.key}"}
                ),
            }
            with candidate.client.stream("POST", path, json=client_body, headers=request_headers) as response:
                response_text: Final = response.read().decode()
            assert response.status_code == 200, response_text
            assert response.headers.get("x-litellm-attempted-fallbacks") == "1", response_text
            assert tuple(call_order.get_nowait() for _ in range(call_order.qsize())) == ("primary", "fallback")
            assert tuple(
                (name, request.method, request.target, json.loads(request.body))
                for name, request in (received.get_nowait() for _ in range(received.qsize()))
            ) == (("primary", "POST", path, primary_body), ("fallback", "POST", path, fallback_body))
            if stream:
                event_lines: Final = tuple(line for line in sse_data_lines(response_text) if line != "[DONE]")
                events: Final = tuple(JSON_OBJECT.validate_json(line) for line in event_lines)
                event_types: Final = tuple(event["type"] for event in events)
                expected_types: Final = (
                    (
                        "message_start",
                        "content_block_start",
                        "content_block_delta",
                        "content_block_stop",
                        "message_delta",
                        "message_stop",
                    )
                    if is_messages
                    else ("response.created", "response.output_text.delta", "response.completed")
                )
                assert event_types.count("message_start" if is_messages else "response.created") == 1, response_text
                assert event_types.count("message_stop" if is_messages else "response.completed") == 1, response_text
                assert event_types[-1] == ("message_stop" if is_messages else "response.completed"), response_text
                assert event_types == expected_types, response_text
                output_text: Final = (
                    "".join(
                        string_value(object_value(event["delta"])["text"])
                        for event in events
                        if event["type"] == "content_block_delta"
                    )
                    if is_messages
                    else "".join(
                        string_value(event["delta"])
                        for event in events
                        if event["type"] == "response.output_text.delta"
                    )
                )
                assert output_text == fallback_text, response_text
            else:
                response_body: Final = JSON_OBJECT.validate_json(response.content)
                expected_response: Final = (
                    _native_anthropic_message(fallback_model, identity, fallback_text)
                    if is_messages
                    else {
                        "id": response_body["id"],
                        "created_at": 1,
                        "error": None,
                        "incomplete_details": None,
                        "instructions": None,
                        "metadata": None,
                        "model": fallback_model,
                        "object": "response",
                        "output": [
                            {
                                "id": "msg_" + identity,
                                "content": [
                                    {
                                        "annotations": [],
                                        "text": fallback_text,
                                        "type": "output_text",
                                        "logprobs": None,
                                    }
                                ],
                                "role": "assistant",
                                "status": "completed",
                                "type": "message",
                                "phase": None,
                            }
                        ],
                        "parallel_tool_calls": None,
                        "temperature": None,
                        "tool_choice": None,
                        "tools": None,
                        "top_p": None,
                        "max_output_tokens": None,
                        "previous_response_id": None,
                        "reasoning": None,
                        "status": "completed",
                        "text": None,
                        "truncation": None,
                        "usage": {
                            "input_tokens": 11,
                            "input_tokens_details": None,
                            "output_tokens": 4,
                            "output_tokens_details": None,
                            "total_tokens": 15,
                            "cost": None,
                        },
                        "user": None,
                        "store": None,
                    }
                )
                assert response_body == expected_response, response_text
                if not is_messages:
                    assert response_body["id"].startswith("resp_"), response_text
            rows: Final = eventually(
                lambda: read_rows(
                    "SELECT model, model_group, spend, prompt_tokens, completion_tokens, status "
                    'FROM "LiteLLM_SpendLogs" WHERE request_id=%s OR litellm_call_id=%s',
                    (identity, identity),
                ),
                lambda values: len(values) == 1,
                seconds=70,
            )
            assert rows[0]["model_group"] == "fallback", rows
            assert rows[0]["status"] == "success", rows
            assert (rows[0]["prompt_tokens"], rows[0]["completion_tokens"]) == (11, 4), rows
            assert float(rows[0]["spend"]) == pytest.approx(0.019), rows


@pytest.mark.covers("other.streaming.failure.truncated_transport_raises_and_control_recovers")
def test_truncated_http_stream_is_an_error_and_next_stream_succeeds() -> None:
    import litellm

    for truncated in (True, False):
        with wire_server(
            lambda request, truncated=truncated: Reply(
                content_type="text/event-stream",
                chunks=text_stream("stream-truncated"),
                abort_after=1 if truncated else None,
            )
        ) as wire:
            stream: Final = litellm.completion(
                model="openai/gpt-4o-mini",
                api_base=wire.url + "/v1",
                api_key="synthetic-stream-key",
                messages=[{"role": "user", "content": "truncation control"}],
                stream=True,
                timeout=5,
                num_retries=0,
            )
            try:
                if truncated:
                    with pytest.raises(
                        litellm.exceptions.MidStreamFallbackError, match="incomplete chunked read"
                    ) as failure:
                        tuple(stream)
                    assert isinstance(failure.value.original_exception, litellm.APIConnectionError)
                    assert failure.value.generated_content == "Hello "
                    assert failure.value.is_pre_first_chunk is False
                else:
                    chunks: Final = tuple(stream)
                    assert (
                        "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices)
                        == "Hello 雪 café"
                    )
                    assert any(choice.finish_reason == "stop" for chunk in chunks for choice in chunk.choices)
            finally:
                asyncio.run(stream.aclose())
            assert len(wire.drain()) == 1


@pytest.mark.covers("other.streaming.cancellation.closes_actual_provider_connection")
def test_client_cancellation_releases_the_actual_provider_connection() -> None:
    import litellm

    gate: Final = threading.Event()
    frames: Final = (
        frame("stream-cancel", {"role": "assistant", "content": "first"}),
        b":" + b"x" * 4_000_000 + b"\n\n",
        b"data: [DONE]\n\n",
    )
    with wire_server(
        lambda request: Reply(content_type="text/event-stream", chunks=frames, gate_after_first=gate)
    ) as wire:
        stream: Final = litellm.completion(
            model="openai/gpt-4o-mini",
            api_base=wire.url + "/v1",
            api_key="synthetic-stream-key",
            messages=[{"role": "user", "content": "cancellation control"}],
            stream=True,
            timeout=5,
            num_retries=0,
        )
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
