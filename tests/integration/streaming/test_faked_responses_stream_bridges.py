import json
import uuid
from collections.abc import Iterator, Sequence
from typing import Final

import anthropic
import openai
from integration._support.client import JSON_OBJECT, Gateway, list_value, object_value, string_value
from integration._support.wire import wire_server
from integration.streaming.faked_responses_case import (
    ANSWER,
    ANTHROPIC_TOOL_PARAMS,
    ANTHROPIC_TOOLS,
    ARGUMENTS,
    AZURE_API_VERSION,
    CALL_ID,
    CHAT_TOOL_PARAMS,
    CHAT_TOOLS,
    NO_CACHE,
    QUESTION,
    SECOND_ARGUMENTS,
    SUMMARIES,
    answering,
    assert_one_unstreamed_upstream_call,
    faked_azure_model,
    faked_model,
    function_call_item,
    message_item,
    reasoning_item,
    spend_statuses,
    stream_events,
)
from openai.types.chat import ChatCompletionChunk
from pydantic import JsonValue

_OPENAI_TARGET: Final = "/v1/responses"
_AZURE_TARGET: Final = f"/openai/responses?api-version={AZURE_API_VERSION}"
_OPENAI_BRIDGE: Final = "openai/responses/faked-stream-backend"


def _two_calls_and_text(identity: str) -> tuple[dict[str, JsonValue], ...]:
    return (
        reasoning_item(identity),
        message_item(identity),
        function_call_item(identity),
        function_call_item(identity, SECOND_ARGUMENTS, "get_forecast"),
    )


def _call_and_text(identity: str) -> tuple[dict[str, JsonValue], ...]:
    return (reasoning_item(identity), message_item(identity), function_call_item(identity))


def _openai(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(base_url=str(gateway.client.base_url) + "/v1", api_key=gateway.key, max_retries=0)


def _async_openai(gateway: Gateway) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(base_url=str(gateway.client.base_url) + "/v1", api_key=gateway.key, max_retries=0)


def _anthropic(gateway: Gateway) -> anthropic.Anthropic:
    return anthropic.Anthropic(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0)


def _async_anthropic(gateway: Gateway) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0)


def _tool_call_fragments(chunks: Sequence[ChatCompletionChunk]) -> Iterator[tuple[int, str | None, str]]:
    for chunk in chunks:
        for choice in chunk.choices:
            for call in choice.delta.tool_calls or ():
                yield (
                    call.index,
                    call.function.name if call.function else None,
                    (call.function.arguments or "" if call.function else ""),
                )


def _arguments_by_index(chunks: Sequence[ChatCompletionChunk]) -> dict[int, str]:
    fragments: Final = tuple(_tool_call_fragments(chunks))
    return {
        index: "".join(arguments for fragment_index, _, arguments in fragments if fragment_index == index)
        for index in sorted({index for index, _, _ in fragments})
    }


def _names_by_index(chunks: Sequence[ChatCompletionChunk]) -> dict[int, str]:
    return {index: name for index, name, _ in _tool_call_fragments(chunks) if name}


def _function_arguments(call: JsonValue) -> JsonValue:
    return object_value(object_value(call)["function"])["arguments"]


def _contents(chunks: Sequence[ChatCompletionChunk]) -> Iterator[str]:
    for chunk in chunks:
        for choice in chunk.choices:
            yield choice.delta.content or ""


def _chat_text(chunks: Sequence[ChatCompletionChunk]) -> str:
    return "".join(_contents(chunks))


def test_chat_stream_through_the_openai_responses_bridge_sends_tool_arguments_once(gateway: Gateway) -> None:
    with wire_server(answering(_two_calls_and_text)) as wire, gateway.scenario() as scenario:
        model: Final = faked_model(scenario, wire, _OPENAI_BRIDGE)
        raw: Final = _openai(gateway).chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "user", "content": f"{QUESTION} {uuid.uuid4().hex}"}],
            tools=CHAT_TOOL_PARAMS,
            stream=True,
            extra_body={"cache": NO_CACHE},
        )
        chunks: Final = list(raw.parse())
        assert _arguments_by_index(chunks) == {0: ARGUMENTS, 1: SECOND_ARGUMENTS}, _arguments_by_index(chunks)
        assert _names_by_index(chunks) == {0: "get_weather", 1: "get_forecast"}, _names_by_index(chunks)
        assert _chat_text(chunks) == ANSWER
        assert len({chunk.id for chunk in chunks}) == 1, {chunk.id for chunk in chunks}
        assert_one_unstreamed_upstream_call(wire, _OPENAI_TARGET)
        assert spend_statuses(raw.headers[CALL_ID]) == ["success"]


async def test_async_chat_stream_through_the_azure_responses_bridge_sends_tool_arguments_once(gateway: Gateway) -> None:
    with wire_server(answering(_call_and_text)) as wire, gateway.scenario() as scenario:
        model: Final = faked_azure_model(scenario, wire)
        stream: Final = await _async_openai(gateway).chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"{QUESTION} {uuid.uuid4().hex}"}],
            tools=CHAT_TOOL_PARAMS,
            stream=True,
            extra_body={"cache": NO_CACHE},
        )
        chunks: Final = [chunk async for chunk in stream]
        assert _arguments_by_index(chunks) == {0: ARGUMENTS}, _arguments_by_index(chunks)
        assert json.loads(_arguments_by_index(chunks)[0]) == json.loads(ARGUMENTS)
        assert _chat_text(chunks) == ANSWER
        assert_one_unstreamed_upstream_call(wire, _AZURE_TARGET)


def test_chat_without_stream_through_the_bridge_returns_the_upstream_values(gateway: Gateway) -> None:
    with wire_server(answering(_two_calls_and_text)) as wire, gateway.scenario() as scenario:
        model: Final = faked_model(scenario, wire, _OPENAI_BRIDGE)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"{QUESTION} {uuid.uuid4().hex}"}],
                "tools": CHAT_TOOLS,
                "cache": NO_CACHE,
            },
        )
        assert response.status_code == 200, response.text
        choice: Final = object_value(list_value(JSON_OBJECT.validate_json(response.content)["choices"])[0])
        message: Final = object_value(choice["message"])
        assert message["content"] == ANSWER, message
        assert [_function_arguments(call) for call in list_value(message["tool_calls"])] == [
            ARGUMENTS,
            SECOND_ARGUMENTS,
        ]
        assert_one_unstreamed_upstream_call(wire, _OPENAI_TARGET)


def test_messages_stream_through_the_azure_chat_bridge_gives_claude_tool_input_once(gateway: Gateway) -> None:
    with wire_server(answering(_call_and_text)) as wire, gateway.scenario() as scenario:
        model: Final = faked_azure_model(scenario, wire)
        with _anthropic(gateway).messages.stream(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": f"{QUESTION} {uuid.uuid4().hex}"}],
            tools=ANTHROPIC_TOOL_PARAMS,
            extra_body={"cache": NO_CACHE},
        ) as stream:
            partial_json: Final = "".join(
                event.delta.partial_json
                for event in stream
                if event.type == "content_block_delta" and event.delta.type == "input_json_delta"
            )
            final: Final = stream.get_final_message()
        assert partial_json == ARGUMENTS
        tool_uses: Final = [block for block in final.content if block.type == "tool_use"]
        assert [(block.name, block.input) for block in tool_uses] == [("get_weather", json.loads(ARGUMENTS))]
        assert "".join(block.text for block in final.content if block.type == "text") == ANSWER
        assert_one_unstreamed_upstream_call(wire, _AZURE_TARGET)


async def test_async_messages_stream_through_the_openai_responses_bridge_keeps_summaries_apart(
    gateway: Gateway,
) -> None:
    with wire_server(answering(_call_and_text)) as wire, gateway.scenario() as scenario:
        model: Final = faked_model(scenario, wire)
        async with _async_anthropic(gateway).messages.stream(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": f"{QUESTION} {uuid.uuid4().hex}"}],
            tools=ANTHROPIC_TOOL_PARAMS,
            extra_body={"cache": NO_CACHE},
        ) as stream:
            final: Final = await stream.get_final_message()
        thinking: Final = "".join(block.thinking for block in final.content if block.type == "thinking")
        first, second = SUMMARIES
        assert thinking.startswith(first) and thinking.endswith(second), thinking
        separator: Final = thinking[len(first) : len(thinking) - len(second)]
        assert separator and separator.strip() == "", repr(thinking)
        tool_uses: Final = [block for block in final.content if block.type == "tool_use"]
        assert [(block.name, block.input) for block in tool_uses] == [("get_weather", json.loads(ARGUMENTS))]
        assert "".join(block.text for block in final.content if block.type == "text") == ANSWER
        assert_one_unstreamed_upstream_call(wire, _OPENAI_TARGET)


def test_raw_messages_stream_through_the_openai_responses_bridge_logs_once(gateway: Gateway) -> None:
    with wire_server(answering(_call_and_text)) as wire, gateway.scenario() as scenario:
        model: Final = faked_model(scenario, wire)
        call_id, events = stream_events(
            gateway,
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 256,
                "messages": [{"role": "user", "content": f"{QUESTION} {uuid.uuid4().hex}"}],
                "tools": ANTHROPIC_TOOLS,
                "cache": NO_CACHE,
            },
        )
        deltas: Final = [object_value(event["delta"]) for event in events if event["type"] == "content_block_delta"]
        assert "".join(string_value(delta.get("partial_json", "")) for delta in deltas) == ARGUMENTS
        assert "".join(string_value(delta.get("text", "")) for delta in deltas) == ANSWER
        assert_one_unstreamed_upstream_call(wire, _OPENAI_TARGET)
        assert spend_statuses(call_id) == ["success"]


def test_messages_without_stream_through_the_bridge_return_the_upstream_values(gateway: Gateway) -> None:
    with wire_server(answering(_call_and_text)) as wire, gateway.scenario() as scenario:
        model: Final = faked_model(scenario, wire)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 256,
                "messages": [{"role": "user", "content": f"{QUESTION} {uuid.uuid4().hex}"}],
                "tools": ANTHROPIC_TOOLS,
                "cache": NO_CACHE,
            },
        )
        assert response.status_code == 200, response.text
        content: Final = tuple(map(object_value, list_value(JSON_OBJECT.validate_json(response.content)["content"])))
        assert [block["input"] for block in content if block["type"] == "tool_use"] == [json.loads(ARGUMENTS)]
        assert "".join(string_value(block["text"]) for block in content if block["type"] == "text") == ANSWER
        assert_one_unstreamed_upstream_call(wire, _OPENAI_TARGET)
