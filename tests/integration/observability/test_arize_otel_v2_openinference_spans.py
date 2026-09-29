from __future__ import annotations

import asyncio
import json
import uuid
from itertools import chain
from pathlib import Path
from typing import Final

import anthropic
import httpx
import openai
import pytest
from _openinference_support import (
    CHAT_TOOLS,
    RESPONSES_TOOLS,
    Rig,
    _anthropic_stream_response,
    _assert_chat_request,
    _assert_messages_request,
    _assert_responses_request,
    _assert_tool_span,
    _chat_cache_hit_caller_stream,
    _chat_caller_response,
    _chat_caller_stream,
    _chat_plain_response,
    _chat_response,
    _chat_stream_response,
    _chat_tool_call,
    _json_messages,
    _json_object,
    _matching_marker_span,
    _messages_caller_response,
    _messages_caller_stream,
    _messages_caller_stream_response,
    _normalize_chat_caller_stream,
    _normalize_responses_caller_body,
    _normalize_responses_caller_stream,
    _response_tool_calls,
    _responses_caller_response,
    _responses_caller_stream,
    _responses_response,
    _responses_stream_response,
    _rig,
)
from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from pydantic import JsonValue


def _chat_upstream(request: Request) -> None:
    _assert_chat_request(request, messages=[{"role": "user", "content": "weather in Paris?"}])


def _responses_upstream(request: Request, marker: str, *, stream: bool = False) -> None:
    _assert_responses_request(request, marker=marker, stream=stream)


def _messages_upstream(request: Request, marker: str, *, stream: bool = False) -> None:
    _assert_messages_request(request, marker=marker, stream=stream)


def _messages_response(identity: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": identity,
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5-5",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_" + identity,
                        "name": "lookup_weather",
                        "input": {"city": "Paris"},
                    }
                ],
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": 11, "output_tokens": 4},
            }
        ).encode()
    )


def _assert_tool_span_for_marker(attributes: dict[str, str], marker: str, *, content: str | None = None) -> None:
    _assert_tool_span(
        attributes,
        marker=marker,
        output=[
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": "call_" + marker,
                        "type": "function",
                        "function": {"name": "lookup_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            }
        ],
        calls=[("call_" + marker, "lookup_weather", {"city": "Paris"})],
        metadata={"trace_marker": marker},
        baggage={"trace_marker": marker},
    )


def _chat_request(
    proxy: Gateway, model: str, marker: str, *, stream: bool = False, no_cache: bool = True
) -> httpx.Response:
    return proxy.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "weather in Paris?"}],
            "tools": CHAT_TOOLS,
            "tool_choice": {"type": "function", "function": {"name": "lookup_weather"}},
            "metadata": {"trace_marker": marker},
            **({"stream": True} if stream else {}),
            **({"cache": {"no-cache": True}} if no_cache else {}),
        },
    )


def _responses_request(proxy: Gateway, model: str, marker: str) -> httpx.Response:
    return proxy.request(
        "POST",
        "/v1/responses",
        {
            "model": model,
            "input": "weather in Paris?",
            "tools": RESPONSES_TOOLS,
            "tool_choice": {"type": "function", "name": "lookup_weather"},
            "metadata": {"trace_marker": marker},
            "cache": {"no-cache": True},
        },
    )


def _messages_request(proxy: Gateway, model: str, marker: str) -> httpx.Response:
    return proxy.request(
        "POST",
        "/v1/messages",
        {
            "model": model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "weather in Paris?"}],
            "tools": [
                {
                    "name": "lookup_weather",
                    "description": "Get weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ],
            "tool_choice": {"type": "auto"},
            "metadata": {"trace_marker": marker},
            "cache": {"no-cache": True},
        },
    )


def _openai_client(proxy: Gateway) -> openai.OpenAI:
    return openai.OpenAI(base_url=str(proxy.client.base_url) + "/v1", api_key=proxy.key, max_retries=0)


def _async_openai_client(proxy: Gateway) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(base_url=str(proxy.client.base_url) + "/v1", api_key=proxy.key, max_retries=0)


def test_arize_otel_v2_a1_chat_sync_sdk(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "a1-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        _chat_upstream(request)
        body: Final = _json_object(request.body)
        assert body == {
            "messages": [{"role": "user", "content": "weather in Paris?"}],
            "model": "gpt-4o-mini",
            "tool_choice": {"type": "function", "function": {"name": "lookup_weather"}},
            "tools": CHAT_TOOLS,
        }, body
        return _chat_response(marker)

    with _rig(gateway, tmp_path, upstream) as rig:
        client: Final = _openai_client(rig.proxy)
        response: Final = client.chat.completions.create(
            model=rig.model,
            messages=[{"role": "user", "content": "weather in Paris?"}],
            tools=CHAT_TOOLS,
            tool_choice={"type": "function", "function": {"name": "lookup_weather"}},
            extra_body={"metadata": {"trace_marker": marker}, "cache": {"no-cache": True}},
        )
        assert response.id == marker, response
        assert response.model_dump(mode="json", exclude_unset=True) == _chat_caller_response(
            _chat_response(marker), rig.model
        ), response
        calls: Final = response.choices[0].message.tool_calls
        assert calls is not None and len(calls) == 1, response
        assert (calls[0].id, calls[0].function.name, calls[0].function.arguments) == (
            f"call_{marker}",
            "lookup_weather",
            '{"city": "Paris"}',
        ), response
        _assert_tool_span_for_marker(_matching_marker_span(rig.destination, marker), marker)


def test_arize_otel_v2_a2_chat_async_sdk(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "a2-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        _chat_upstream(request)
        return _chat_response(marker)

    async def call() -> None:
        with _rig(gateway, tmp_path, upstream) as rig:
            client: Final = _async_openai_client(rig.proxy)
            response: Final = await client.chat.completions.create(
                model=rig.model,
                messages=[{"role": "user", "content": "weather in Paris?"}],
                tools=CHAT_TOOLS,
                tool_choice={"type": "function", "function": {"name": "lookup_weather"}},
                extra_body={"metadata": {"trace_marker": marker}, "cache": {"no-cache": True}},
            )
            assert response.id == marker, response
            assert response.model_dump(mode="json", exclude_unset=True) == _chat_caller_response(
                _chat_response(marker), rig.model
            ), response
            calls: Final = response.choices[0].message.tool_calls
            assert calls is not None and len(calls) == 1, response
            assert (calls[0].id, calls[0].function.name, calls[0].function.arguments) == (
                f"call_{marker}",
                "lookup_weather",
                '{"city": "Paris"}',
            ), response
            _assert_tool_span_for_marker(_matching_marker_span(rig.destination, marker), marker)

    asyncio.run(call())


def test_arize_otel_v2_a5_responses_sync_sdk(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "a5-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        _responses_upstream(request, marker)
        return _responses_response(marker)

    with _rig(gateway, tmp_path, upstream) as rig:
        client: Final = _openai_client(rig.proxy)
        response: Final = client.responses.create(
            model=rig.model,
            input="weather in Paris?",
            tools=RESPONSES_TOOLS,
            tool_choice={"type": "function", "name": "lookup_weather"},
            extra_body={"metadata": {"trace_marker": marker}, "cache": {"no-cache": True}},
        )
        assert response.id.startswith("resp_"), response
        assert _normalize_responses_caller_body(response.model_dump(mode="json", exclude_unset=True)) == (
            _responses_caller_response(_responses_response(marker), rig.model)
        ), response
        assert (response.status, response.model) == ("completed", rig.model), response
        assert (
            response.output[0].type,
            response.output[0].call_id,
            response.output[0].name,
            response.output[0].arguments,
        ) == ("function_call", f"call_{marker}", "lookup_weather", '{"city": "Paris"}'), response
        _assert_tool_span_for_marker(_matching_marker_span(rig.destination, marker), marker)


def test_arize_otel_v2_a6_responses_async_streaming_sdk(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "a6-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        _responses_upstream(request, marker, stream=True)
        return _responses_stream_response(marker, (_chat_tool_call(marker),))

    async def call() -> None:
        with _rig(gateway, tmp_path, upstream) as rig:
            client: Final = _async_openai_client(rig.proxy)
            stream: Final = await client.responses.create(
                model=rig.model,
                input="weather in Paris?",
                tools=RESPONSES_TOOLS,
                tool_choice={"type": "function", "name": "lookup_weather"},
                stream=True,
                extra_body={"metadata": {"trace_marker": marker}, "cache": {"no-cache": True}},
            )
            events: Final = tuple([event async for event in stream])
            assert events[-1].type == "response.completed", events
            assert events[-1].response.id.startswith("resp_"), events[-1]
            assert _normalize_responses_caller_stream(
                tuple(event.model_dump(mode="json", exclude_unset=True) for event in events)
            ) == _responses_caller_stream(_responses_stream_response(marker, (_chat_tool_call(marker),)), rig.model), (
                events
            )
            call: Final = events[-1].response.output[0]
            assert (call.type, call.call_id, call.name, call.arguments) == (
                "function_call",
                f"call_{marker}",
                "lookup_weather",
                '{"city": "Paris"}',
            ), events[-1]
            _assert_tool_span_for_marker(_matching_marker_span(rig.destination, marker), marker)

    asyncio.run(call())


def test_arize_otel_v2_a7_messages_sync_sdk(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "a7-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        _messages_upstream(request, marker)
        return _messages_response(marker)

    with _rig(gateway, tmp_path, upstream, model_name="anthropic/claude-opus-5-5", api_base_suffix="") as rig:
        client: Final = anthropic.Anthropic(
            base_url=str(rig.proxy.client.base_url), api_key=rig.proxy.key, max_retries=0
        )
        response: Final = client.messages.create(
            model=rig.model,
            max_tokens=64,
            messages=[{"role": "user", "content": "weather in Paris?"}],
            tools=[
                {
                    "name": "lookup_weather",
                    "description": "Get weather",
                    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
                }
            ],
            tool_choice={"type": "auto"},
            metadata={"trace_marker": marker},
        )
        assert response.id == marker, response
        assert response.model_dump(mode="json", exclude_unset=True) == _messages_caller_response(
            _messages_response(marker), rig.model
        ), response
        call: Final = response.content[0]
        assert (call.type, call.id, call.name, call.input) == (
            "tool_use",
            f"call_{marker}",
            "lookup_weather",
            {"city": "Paris"},
        ), response
        _assert_tool_span_for_marker(_matching_marker_span(rig.destination, marker), marker)


def test_arize_otel_v2_a3_chat_streaming(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "a3-" + uuid.uuid4().hex
    stream_reply: Final = _chat_stream_response(marker, (_chat_tool_call(marker),), include_usage=False)

    def upstream(request: Request) -> Reply:
        _assert_chat_request(
            request,
            messages=[{"role": "user", "content": "weather in Paris?"}],
            stream=True,
            stream_options={"include_usage": False},
        )
        return stream_reply

    with _rig(gateway, tmp_path, upstream, general_settings={"always_include_stream_usage": False}) as rig:
        client: Final = _openai_client(rig.proxy)
        stream: Final = client.chat.completions.create(
            model=rig.model,
            messages=[{"role": "user", "content": "weather in Paris?"}],
            tools=CHAT_TOOLS,
            tool_choice={"type": "function", "function": {"name": "lookup_weather"}},
            stream=True,
            stream_options={"include_usage": False},
            extra_body={"metadata": {"trace_marker": marker}, "cache": {"no-cache": True}},
        )
        chunks: Final = tuple(stream)
        assert chunks[0].id == marker and chunks[-1].id == marker, chunks
        assert _normalize_chat_caller_stream(
            tuple(chunk.model_dump(mode="json", exclude_unset=True) for chunk in chunks)
        ) == _chat_caller_stream(stream_reply, rig.model), chunks
        assert chunks[-1].choices[0].finish_reason == "tool_calls", chunks
        tool_call_deltas: Final = tuple(
            chain.from_iterable(chunk.choices[0].delta.tool_calls or () for chunk in chunks)
        )
        assert len(tool_call_deltas) == 2, chunks
        assert (
            tool_call_deltas[0].id,
            tool_call_deltas[0].function.name,
            tool_call_deltas[1].function.arguments,
        ) == (f"call_{marker}", "lookup_weather", '{"city": "Paris"}'), chunks
        _assert_tool_span_for_marker(_matching_marker_span(rig.destination, marker), marker)


def test_arize_otel_v2_a4_chat_async_streaming(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "a4-" + uuid.uuid4().hex
    stream_reply: Final = _chat_stream_response(marker, (_chat_tool_call(marker),), include_usage=False)

    def upstream(request: Request) -> Reply:
        _assert_chat_request(
            request,
            messages=[{"role": "user", "content": "weather in Paris?"}],
            stream=True,
            stream_options={"include_usage": False},
        )
        return stream_reply

    async def call() -> None:
        with _rig(gateway, tmp_path, upstream, general_settings={"always_include_stream_usage": False}) as rig:
            client: Final = _async_openai_client(rig.proxy)
            stream: Final = await client.chat.completions.create(
                model=rig.model,
                messages=[{"role": "user", "content": "weather in Paris?"}],
                tools=CHAT_TOOLS,
                tool_choice={"type": "function", "function": {"name": "lookup_weather"}},
                stream=True,
                stream_options={"include_usage": False},
                extra_body={"metadata": {"trace_marker": marker}, "cache": {"no-cache": True}},
            )
            chunks: Final = tuple([chunk async for chunk in stream])
            assert chunks[0].id == marker and chunks[-1].id == marker, chunks
            assert _normalize_chat_caller_stream(
                tuple(chunk.model_dump(mode="json", exclude_unset=True) for chunk in chunks)
            ) == _chat_caller_stream(stream_reply, rig.model), chunks
            assert chunks[-1].choices[0].finish_reason == "tool_calls", chunks
            tool_call_deltas: Final = tuple(
                chain.from_iterable(chunk.choices[0].delta.tool_calls or () for chunk in chunks)
            )
            assert len(tool_call_deltas) == 2, chunks
            assert (
                tool_call_deltas[0].id,
                tool_call_deltas[0].function.name,
                tool_call_deltas[1].function.arguments,
            ) == (f"call_{marker}", "lookup_weather", '{"city": "Paris"}'), chunks
            _assert_tool_span_for_marker(_matching_marker_span(rig.destination, marker), marker)

    asyncio.run(call())


def test_arize_otel_v2_a8_messages_streaming(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "a8-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        _messages_upstream(request, marker, stream=True)
        return _anthropic_stream_response(marker)

    async def call() -> None:
        with _rig(gateway, tmp_path, upstream, model_name="anthropic/claude-opus-5-5", api_base_suffix="") as rig:
            client: Final = anthropic.AsyncAnthropic(
                base_url=str(rig.proxy.client.base_url), api_key=rig.proxy.key, max_retries=0
            )
            async with client.messages.stream(
                model=rig.model,
                max_tokens=64,
                messages=[{"role": "user", "content": "weather in Paris?"}],
                tools=[
                    {
                        "name": "lookup_weather",
                        "description": "Get weather",
                        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
                    }
                ],
                tool_choice={"type": "auto"},
                metadata={"trace_marker": marker},
            ) as stream:
                events: Final = tuple([event async for event in stream])
                response: Final = await stream.get_final_message()
            assert events[-1].type == "message_stop", events
            assert response.id == marker, response
            assert response.model_dump(mode="json", exclude_unset=True) == _messages_caller_stream_response(
                _messages_response(marker), rig.model
            ), response
            assert tuple(event.model_dump(mode="json", exclude_unset=True) for event in events) == (
                _messages_caller_stream(
                    _anthropic_stream_response(marker),
                    rig.model,
                    final_message=_messages_caller_stream_response(_messages_response(marker), rig.model),
                )
            ), events
            call: Final = response.content[0]
            assert (call.type, call.id, call.name, call.input) == (
                "tool_use",
                f"call_{marker}",
                "lookup_weather",
                {"city": "Paris"},
            ), response
            _assert_tool_span_for_marker(_matching_marker_span(rig.destination, marker), marker, content="")

    asyncio.run(call())


def test_arize_otel_v2_llm_span_carries_openinference_tool_calls_and_metadata(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "a9-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        _chat_upstream(request)
        return _chat_response(marker)

    with _rig(gateway, tmp_path, upstream) as rig:
        response: Final = _chat_request(rig.proxy, rig.model, marker)
        assert response.status_code == 200, response.text
        assert _json_object(response.content) == _chat_caller_response(_chat_response(marker), rig.model), response.text
        attributes: Final = _matching_marker_span(rig.destination, marker)
        _assert_tool_span_for_marker(attributes, marker)


def test_arize_otel_v2_responses_span_carries_openinference_tool_calls_and_metadata(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "a10-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        _responses_upstream(request, marker)
        return _responses_response(marker)

    with _rig(gateway, tmp_path, upstream) as rig:
        response: Final = _responses_request(rig.proxy, rig.model, marker)
        assert response.status_code == 200, response.text
        response_body: Final = _json_object(response.content)
        assert _normalize_responses_caller_body(response_body) == _responses_caller_response(
            _responses_response(marker), rig.model
        ), response.text
        _assert_tool_span_for_marker(_matching_marker_span(rig.destination, marker), marker)


def test_arize_otel_v2_a11_parallel_output_tool_calls(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "a11-" + uuid.uuid4().hex
    calls: Final = _response_tool_calls(marker, ("Paris", "Berlin"))

    def upstream(request: Request) -> Reply:
        _assert_chat_request(request, messages=[{"role": "user", "content": "weather in Paris?"}])
        return _chat_response(marker, calls)

    with _rig(gateway, tmp_path, upstream) as rig:
        response: Final = _chat_request(rig.proxy, rig.model, marker)
        assert response.status_code == 200, response.text
        assert _json_object(response.content) == _chat_caller_response(_chat_response(marker, calls), rig.model), (
            response.text
        )
        attributes: Final = _matching_marker_span(rig.destination, marker)
        expected_calls: Final = (
            ("call_" + marker + "-Paris", "lookup_weather", {"city": "Paris"}),
            ("call_" + marker + "-Berlin", "lookup_weather", {"city": "Berlin"}),
        )
        _assert_tool_span(
            attributes,
            marker=marker,
            output=[{"role": "assistant", "content": None, "tool_calls": calls}],
            calls=expected_calls,
            metadata={"trace_marker": marker},
            baggage={"trace_marker": marker},
        )


def test_arize_otel_v2_a12_plain_text_has_metadata_without_tool_calls(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "a12-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        _assert_chat_request(
            request,
            messages=[{"role": "user", "content": "weather in Paris?"}],
            include_tools=False,
        )
        return _chat_plain_response(marker, "The weather is clear")

    with _rig(gateway, tmp_path, upstream) as rig:
        response: Final = rig.proxy.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": rig.model,
                "messages": [{"role": "user", "content": "weather in Paris?"}],
                "metadata": {"trace_marker": marker},
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        assert _json_object(response.content) == _chat_caller_response(
            _chat_plain_response(marker, "The weather is clear"), rig.model
        ), response.text
        attributes: Final = _matching_marker_span(rig.destination, marker)
        assert _json_object(attributes["metadata"].encode()) == {"trace_marker": marker}, attributes
        assert attributes["litellm.metadata.trace_marker"] == marker, attributes
        assert not any(".tool_calls." in key for key in attributes), attributes
        assert _json_messages(attributes["output.value"]) == [
            {"role": "assistant", "content": "The weather is clear"}
        ], attributes


def test_arize_otel_v2_a13_multiturn_input_tool_calls(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "a13-" + uuid.uuid4().hex
    messages: Final = [
        {"role": "user", "content": "weather in Paris?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-prior",
                    "type": "function",
                    "function": {"name": "lookup_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-prior", "content": '{"temperature": 20}'},
    ]
    upstream_messages: Final = [
        {key: value for key, value in message.items() if value is not None} for message in messages
    ]

    def upstream(request: Request) -> Reply:
        _assert_chat_request(request, messages=upstream_messages)
        return _chat_response(marker)

    with _rig(gateway, tmp_path, upstream) as rig:
        response: Final = rig.proxy.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": rig.model,
                "messages": messages,
                "tools": CHAT_TOOLS,
                "tool_choice": {"type": "function", "function": {"name": "lookup_weather"}},
                "metadata": {"trace_marker": marker},
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        assert _json_object(response.content) == _chat_caller_response(_chat_response(marker), rig.model), response.text
        attributes: Final = _matching_marker_span(rig.destination, marker)
        _assert_tool_span_for_marker(attributes, marker)
        assert not any(key.startswith("llm.input_messages.") and ".tool_calls." in key for key in attributes), (
            attributes
        )
        assert tuple(attributes[f"llm.input_messages.{index}.message.role"] for index in range(3)) == (
            "user",
            "assistant",
            "tool",
        ), attributes
        assert tuple(attributes[f"llm.input_messages.{index}.message.content"] for index in (0, 2)) == (
            "weather in Paris?",
            '{"temperature": 20}',
        ), attributes
        expected_input_value: Final = [
            messages[0],
            messages[1],
            {"role": "tool", "content": '{"temperature": 20}'},
        ]
        assert _json_messages(attributes["input.value"]) == expected_input_value, attributes


def test_arize_otel_v2_a14_two_choices_each_with_tool_calls(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "a14-" + uuid.uuid4().hex
    first_call: Final = _response_tool_calls(marker, ("Paris",))[0]
    second_call: Final = _response_tool_calls(marker, ("Berlin",))[0]

    def choice(index: int, call: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return {
            "index": index,
            "finish_reason": "tool_calls",
            "message": {"role": "assistant", "content": None, "tool_calls": [call]},
        }

    expected_response: Final = Reply(
        body=json.dumps(
            {
                "id": marker,
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [choice(0, first_call), choice(1, second_call)],
                "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
            }
        ).encode()
    )

    def upstream(request: Request) -> Reply:
        _assert_chat_request(
            request,
            messages=[{"role": "user", "content": "weather in Paris and Berlin?"}],
            n=2,
        )
        return expected_response

    with _rig(gateway, tmp_path, upstream) as rig:
        response: Final = rig.proxy.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": rig.model,
                "messages": [{"role": "user", "content": "weather in Paris and Berlin?"}],
                "tools": CHAT_TOOLS,
                "tool_choice": {"type": "function", "function": {"name": "lookup_weather"}},
                "n": 2,
                "metadata": {"trace_marker": marker},
                "cache": {"no-cache": True},
            },
        )
        assert response.status_code == 200, response.text
        assert _json_object(response.content) == _chat_caller_response(expected_response, rig.model), response.text
        attributes: Final = _matching_marker_span(rig.destination, marker)
        assert _json_messages(attributes["output.value"]) == [
            {"role": "assistant", "content": None, "tool_calls": [first_call]},
            {"role": "assistant", "content": None, "tool_calls": [second_call]},
        ], attributes
        assert attributes["llm.output_messages.0.message.tool_calls.0.tool_call.id"] == str(first_call["id"]), (
            attributes
        )
        assert attributes["llm.output_messages.1.message.tool_calls.0.tool_call.id"] == str(second_call["id"]), (
            attributes
        )
        assert attributes["llm.output_messages.0.message.tool_calls.0.tool_call.function.name"] == "lookup_weather", (
            attributes
        )
        assert (
            attributes["llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments"] == '{"city": "Paris"}'
        ), attributes
        assert attributes["llm.output_messages.1.message.tool_calls.0.tool_call.function.name"] == "lookup_weather", (
            attributes
        )
        assert (
            attributes["llm.output_messages.1.message.tool_calls.0.tool_call.function.arguments"]
            == '{"city": "Berlin"}'
        ), attributes
        assert _json_object(attributes["metadata"].encode()) == {"trace_marker": marker}, attributes
        assert attributes["litellm.metadata.trace_marker"] == marker, attributes


def _cache_call(
    rig: Rig, surface: str, marker: str, *, cache_hit: bool = False
) -> tuple[tuple[str, str, str], str, httpx.Headers]:
    match surface:
        case "chat":
            client: Final = _openai_client(rig.proxy)
            raw: Final = client.chat.completions.with_raw_response.create(
                model=rig.model,
                messages=[{"role": "user", "content": marker}],
                tools=CHAT_TOOLS,
                tool_choice={"type": "function", "function": {"name": "lookup_weather"}},
                extra_body={"metadata": {"trace_marker": marker}, "cache": {"no-cache": False}},
            )
            response: Final = raw.parse()
            assert response.model_dump(mode="json", exclude_unset=True) == _chat_caller_response(
                _chat_response(marker), rig.model
            ), response
            assert response.choices[0].message.tool_calls is not None, response
            call: Final = response.choices[0].message.tool_calls[0]
            return (call.id, call.function.name, call.function.arguments), response.id, raw.headers
        case "chat-stream":
            client: Final = _openai_client(rig.proxy)
            with client.chat.completions.with_streaming_response.create(
                model=rig.model,
                messages=[{"role": "user", "content": marker}],
                tools=CHAT_TOOLS,
                tool_choice={"type": "function", "function": {"name": "lookup_weather"}},
                stream=True,
                stream_options={"include_usage": False},
                extra_body={"metadata": {"trace_marker": marker}, "cache": {"no-cache": False}},
            ) as raw:
                chunks: Final = tuple(raw.parse())
                expected_chunks: Final = (
                    _chat_cache_hit_caller_stream(marker, rig.model, (_chat_tool_call(marker),))
                    if cache_hit
                    else _chat_caller_stream(
                        _chat_stream_response(marker, (_chat_tool_call(marker),), include_usage=False), rig.model
                    )
                )
                assert (
                    _normalize_chat_caller_stream(
                        tuple(chunk.model_dump(mode="json", exclude_unset=True) for chunk in chunks)
                    )
                    == expected_chunks
                ), chunks
                calls: Final = tuple(chain.from_iterable(chunk.choices[0].delta.tool_calls or () for chunk in chunks))
                if cache_hit:
                    assert len(calls) == 1, chunks
                    call: Final = calls[0]
                    assert (
                        call.id is not None and call.function.name is not None and call.function.arguments is not None
                    ), chunks
                    return (
                        (call.id, call.function.name, call.function.arguments),
                        chunks[0].id,
                        raw.headers,
                    )
                assert len(calls) == 2, chunks
                call: Final = calls[0]
                arguments_call: Final = calls[1]
                assert (
                    call.id is not None
                    and call.function.name is not None
                    and arguments_call.function.arguments is not None
                ), chunks
                return (
                    (call.id, call.function.name, arguments_call.function.arguments),
                    chunks[0].id,
                    raw.headers,
                )
        case "responses":
            client: Final = _openai_client(rig.proxy)
            raw: Final = client.responses.with_raw_response.create(
                model=rig.model,
                input=marker,
                tools=RESPONSES_TOOLS,
                tool_choice={"type": "function", "name": "lookup_weather"},
                extra_body={"metadata": {"trace_marker": marker}, "cache": {"no-cache": False}},
            )
            response: Final = raw.parse()
            assert _normalize_responses_caller_body(response.model_dump(mode="json", exclude_unset=True)) == (
                _responses_caller_response(_responses_response(marker), rig.model)
            ), response
            call: Final = response.output[0]
            assert call.type == "function_call", response
            return (call.call_id, call.name, call.arguments), response.id, raw.headers
        case "messages":
            client: Final = anthropic.Anthropic(
                base_url=str(rig.proxy.client.base_url), api_key=rig.proxy.key, max_retries=0
            )
            raw: Final = client.messages.with_raw_response.create(
                model=rig.model,
                max_tokens=64,
                messages=[{"role": "user", "content": marker}],
                tools=[
                    {
                        "name": "lookup_weather",
                        "description": "Get weather",
                        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
                    }
                ],
                tool_choice={"type": "auto"},
                extra_body={"cache": {"no-cache": False}, "metadata": {"trace_marker": marker}},
            )
            response: Final = raw.parse()
            assert response.model_dump(mode="json", exclude_unset=True) == _messages_caller_response(
                _messages_response(marker), rig.model
            ), response
            call: Final = response.content[0]
            assert call.type == "tool_use", response
            return (call.id, call.name, json.dumps(call.input)), response.id, raw.headers
        case _:
            raise AssertionError(f"Unknown cache surface: {surface}")


@pytest.mark.parametrize("surface", ("chat", "chat-stream", "responses", "messages"))
def test_arize_otel_v2_a_cache(surface: str, gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = f"a-cache-{surface}-" + uuid.uuid4().hex

    def upstream(request: Request) -> Reply:
        body: Final = _json_object(request.body)
        if surface in ("chat", "chat-stream"):
            _assert_chat_request(
                request,
                messages=[{"role": "user", "content": marker}],
                stream=True if surface == "chat-stream" else None,
                stream_options={"include_usage": False} if surface == "chat-stream" else None,
            )
            return (
                _chat_stream_response(marker, (_chat_tool_call(marker),), include_usage=False)
                if surface == "chat-stream"
                else _chat_response(marker)
            )
        if surface == "responses":
            assert body.get("metadata") == {"trace_marker": marker}, body
            _assert_responses_request(request, marker=marker, input_value=marker)
            return _responses_response(marker)
        _assert_messages_request(request, marker=marker, prompt=marker)
        return _messages_response(marker)

    with _rig(
        gateway,
        tmp_path,
        upstream,
        general_settings={"always_include_stream_usage": False} if surface == "chat-stream" else None,
        model_name="anthropic/claude-opus-5-5" if surface == "messages" else "gpt-4o-mini",
        api_base_suffix="" if surface == "messages" else "/v1",
    ) as rig:
        expected_arguments: Final = '{"city": "Paris"}'
        first: Final = _cache_call(rig, surface, marker)
        assert first[0] == (f"call_{marker}", "lookup_weather", expected_arguments), first
        assert first[1].startswith("resp_") if surface == "responses" else first[1] == marker, first
        assert not first[2].get("x-litellm-cache-key"), first[2]
        rig.destination.drain()
        second: Final = _cache_call(rig, surface, marker, cache_hit=True)
        assert second[0] == first[0], second
        assert second[1].startswith("resp_") if surface == "responses" else second[1] == first[1], second
        forwarded: Final = tuple(
            request for request in rig.provider.drain() if request.method == "POST" and marker.encode() in request.body
        )
        assert len(forwarded) == 1, forwarded
        if surface == "messages":
            assert not second[2].get("x-litellm-cache-key"), second[2]
        else:
            assert second[2].get("x-litellm-cache-key"), second[2]
        _assert_tool_span_for_marker(_matching_marker_span(rig.destination, marker), marker)
        forwarded_body: Final = _json_object(forwarded[0].body)
        if surface in ("chat", "chat-stream"):
            assert "metadata" not in forwarded_body, forwarded[0]
        elif surface == "responses":
            assert forwarded_body["metadata"] == {"trace_marker": marker}, forwarded[0]
        else:
            assert forwarded_body["metadata"] == {}, forwarded[0]
