import itertools
import json
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Final

import anthropic
import openai
import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "llama3-prompt-tools"
_API_KEY: Final = "synthetic-ollama-key"
_INSTRUCTION: Final = (
    'To call a function, reply with JSON ONLY in this format {"name": "function_name", '
    '"arguments":{"argument_name": "argument_value"}}. Once a function result answers the request, '
    "reply to the user in plain text instead of calling a function again. "
    "The following functions are available to you:"
)
_QUESTION: Final = "What is the weather in Paris?"
_RESULT: Final = "Paris: 22 degrees Celsius, clear skies"
_ANSWER: Final = "Paris is 22 degrees Celsius with clear skies."
_CALL_ID: Final = "call_prompt_tools_1"
_ARGUMENTS: Final[dict[str, JsonValue]] = {"city": "Paris"}
_CALL_JSON: Final = json.dumps({"name": "get_weather", "arguments": _ARGUMENTS})
_CALL_JSON_FIELDS: Final = frozenset({"get_weather"})
_PARAMETERS: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}
_WEATHER_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "function": {"name": "get_weather", "description": "Weather for a city", "parameters": _PARAMETERS},
}
_TIME_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "function": {"name": "get_time", "description": "Local time for a city", "parameters": _PARAMETERS},
}
_ANTHROPIC_TOOL: Final[dict[str, JsonValue]] = {
    "name": "get_weather",
    "description": "Weather for a city",
    "input_schema": _PARAMETERS,
}
_RESPONSES_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "name": "get_weather",
    "description": "Weather for a city",
    "parameters": _PARAMETERS,
}
_NO_CACHE: Final[dict[str, JsonValue]] = {"no-cache": True}
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_RAW_ANTHROPIC_EVENTS: Final = frozenset(
    {"message_start", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"}
)


def _generate_reply(text: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "model": _BACKEND,
                "created_at": "2026-10-07T00:00:00Z",
                "response": text,
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 30,
                "eval_count": 12,
            }
        ).encode()
    )


def _streamed_reply(text: str) -> Reply:
    pieces: Final = tuple(text[index : index + 7] for index in range(0, len(text), 7))
    lines: Final = tuple(
        json.dumps({"model": _BACKEND, "created_at": "2026-10-07T00:00:00Z", "response": piece, "done": False}).encode()
        + b"\n"
        for piece in pieces
    )
    final: Final = (
        json.dumps(
            {
                "model": _BACKEND,
                "created_at": "2026-10-07T00:00:00Z",
                "response": "",
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 30,
                "eval_count": 12,
            }
        ).encode()
        + b"\n"
    )
    return Reply(content_type="application/x-ndjson", chunks=(*lines, final))


def _is_generate(request: Request) -> bool:
    return (request.method, request.target) == ("POST", "/api/generate")


@contextmanager
def _ollama_server(respond: Callable[[Request], Reply]) -> Iterator[Wire]:
    with wire_server(lambda request: respond(request) if _is_generate(request) else Reply(body=b"{}")) as wire:
        yield wire


def _generate_calls(wire: Wire) -> tuple[Request, ...]:
    return tuple(request for request in wire.drain() if _is_generate(request))


def _only_generate(wire: Wire) -> dict[str, JsonValue]:
    received: Final = _generate_calls(wire)
    assert len(received) == 1, [(request.method, request.target) for request in received]
    assert received[0].headers["authorization"] == f"Bearer {_API_KEY}"
    return _JSON_OBJECT.validate_json(received[0].body)


def _prompt_of(body: dict[str, JsonValue]) -> str:
    assert body["model"] == _BACKEND
    assert body["format"] == "json"
    assert "tools" not in body and "messages" not in body, sorted(body)
    prompt: Final = body["prompt"]
    assert isinstance(prompt, str)
    return prompt


def _assert_instructed_once(prompt: str, *tool_names: str) -> None:
    assert prompt.count(_INSTRUCTION) == 1, prompt
    assert prompt.count("### System:") == 1, prompt
    for name in tool_names:
        assert f"'name': '{name}'" in prompt, prompt


def _assert_tool_turn(prompt: str, result: str = _RESULT) -> None:
    _assert_instructed_once(prompt, "get_weather")
    assert f"### User:\n{_QUESTION}\n\n### Assistant:\n{_CALL_JSON}\n\n### User:\n{result}\n\n" in prompt, prompt


def _spend_row(identity: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT model_group, status, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
            (identity,),
        ),
        lambda found: len(found) == 1,
        seconds=70,
    )
    return rows[0]


def _model_spend_rows(model: str, expected: int) -> list[dict[str, JsonValue]]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id, status, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE model_group=%s',
            (model,),
        ),
        lambda found: len(found) >= expected,
        seconds=70,
    )
    assert len({row["request_id"] for row in rows}) == len(rows) == expected, rows
    return rows


def _billed(model: str) -> dict[str, JsonValue]:
    return {"model_group": model, "status": "success", "prompt_tokens": 30, "completion_tokens": 12}


def _openai_client(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(base_url=str(gateway.client.base_url) + "/v1", api_key=gateway.key, max_retries=0)


def _async_openai_client(gateway: Gateway) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(base_url=str(gateway.client.base_url) + "/v1", api_key=gateway.key, max_retries=0)


def _anthropic_client(gateway: Gateway) -> anthropic.Anthropic:
    return anthropic.Anthropic(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0)


def _async_anthropic_client(gateway: Gateway) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0)


def _first_turn() -> list[dict[str, JsonValue]]:
    return [{"role": "user", "content": _QUESTION}]


def _second_turn(result: JsonValue = _RESULT) -> list[dict[str, JsonValue]]:
    return [
        {"role": "user", "content": _QUESTION},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": _CALL_ID, "type": "function", "function": {"name": "get_weather", "arguments": json.dumps(_ARGUMENTS)}}
            ],
        },
        {"role": "tool", "tool_call_id": _CALL_ID, "content": result},
    ]


def _anthropic_second_turn() -> list[dict[str, JsonValue]]:
    return [
        {"role": "user", "content": _QUESTION},
        {"role": "assistant", "content": [{"type": "tool_use", "id": _CALL_ID, "name": "get_weather", "input": _ARGUMENTS}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": _CALL_ID, "content": _RESULT}]},
    ]


def _responses_second_turn() -> list[dict[str, JsonValue]]:
    return [
        {"role": "user", "content": _QUESTION},
        {"type": "function_call", "call_id": _CALL_ID, "name": "get_weather", "arguments": json.dumps(_ARGUMENTS)},
        {"type": "function_call_output", "call_id": _CALL_ID, "output": _RESULT},
    ]


def _post(gateway: Gateway, path: str, body: dict[str, JsonValue], key: str | None = None) -> tuple[int, str]:
    response: Final = gateway.request("POST", path, {**body, "cache": _NO_CACHE}, key=key)
    return response.status_code, response.text


def _post_chat(
    gateway: Gateway, model: str, messages: Sequence[dict[str, JsonValue]], **extra: JsonValue
) -> dict[str, JsonValue]:
    code, text = _post(gateway, "/v1/chat/completions", {"model": model, "messages": list(messages), **extra})
    assert code == 200, text
    return _JSON_OBJECT.validate_json(text)


def test_openai_sdk_tool_request_reaches_ollama_as_an_instructed_prompt(gateway: Gateway) -> None:
    with _ollama_server(lambda _: _generate_reply(_CALL_JSON)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        completion: Final = _openai_client(gateway).chat.completions.create(
            model=model,
            messages=_first_turn(),  # pyright: ignore[reportArgumentType]  # plain JSON messages
            tools=[_WEATHER_TOOL],  # pyright: ignore[reportArgumentType]  # plain JSON tool
            extra_body={"cache": _NO_CACHE},
        )
        choice: Final = completion.choices[0]
        assert choice.finish_reason == "tool_calls"
        assert choice.message.tool_calls is not None and len(choice.message.tool_calls) == 1
        call: Final = choice.message.tool_calls[0]
        assert call.type == "function"
        assert call.function.name == "get_weather"
        assert json.loads(call.function.arguments) == _ARGUMENTS
        assert completion.usage is not None and (completion.usage.prompt_tokens, completion.usage.completion_tokens) == (30, 12)
        body: Final = _only_generate(wire)
        assert body["stream"] is False
        prompt: Final = _prompt_of(body)
        _assert_instructed_once(prompt, "get_weather")
        assert f"### User:\n{_QUESTION}\n\n" in prompt, prompt
        assert "Weather for a city" in prompt, prompt
        assert _spend_row(completion.id) == _billed(model)


def test_openai_sdk_tool_result_turn_gets_a_plain_text_answer(gateway: Gateway) -> None:
    with _ollama_server(lambda _: _generate_reply(_ANSWER)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        completion: Final = _openai_client(gateway).chat.completions.create(
            model=model,
            messages=_second_turn(),  # pyright: ignore[reportArgumentType]  # plain JSON messages
            tools=[_WEATHER_TOOL],  # pyright: ignore[reportArgumentType]  # plain JSON tool
            extra_body={"cache": _NO_CACHE},
        )
        choice: Final = completion.choices[0]
        assert choice.finish_reason == "stop"
        assert choice.message.content == _ANSWER
        assert choice.message.tool_calls is None
        _assert_tool_turn(_prompt_of(_only_generate(wire)))
        assert _spend_row(completion.id) == _billed(model)


async def test_async_openai_sdk_stream_flushes_the_held_tool_call_once(gateway: Gateway) -> None:
    with _ollama_server(lambda _: _streamed_reply(_CALL_JSON)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        stream: Final = await _async_openai_client(gateway).chat.completions.create(
            model=model,
            messages=_first_turn(),  # pyright: ignore[reportArgumentType]  # plain JSON messages
            tools=[_WEATHER_TOOL],  # pyright: ignore[reportArgumentType]  # plain JSON tool
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"cache": _NO_CACHE},
        )
        chunks: Final = [chunk async for chunk in stream]
        assert {chunk.id for chunk in chunks} == {chunks[0].id}
        deltas: Final = [call for chunk in chunks for choice in chunk.choices for call in (choice.delta.tool_calls or [])]
        assert len(deltas) == 1, deltas
        assert deltas[0].function is not None and deltas[0].function.name == "get_weather"
        assert json.loads(deltas[0].function.arguments or "") == _ARGUMENTS
        assert [choice.finish_reason for chunk in chunks for choice in chunk.choices if choice.finish_reason] == ["tool_calls"]
        usages: Final = [chunk.usage for chunk in chunks if chunk.usage is not None]
        assert [(usage.prompt_tokens, usage.completion_tokens) for usage in usages] == [(30, 12)]
        body: Final = _only_generate(wire)
        assert body["stream"] is True
        _assert_instructed_once(_prompt_of(body), "get_weather")
        assert _spend_row(chunks[0].id) == _billed(model)


async def test_async_openai_sdk_stream_answers_the_tool_result_in_plain_text(gateway: Gateway) -> None:
    with _ollama_server(lambda _: _streamed_reply(_ANSWER)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        stream: Final = await _async_openai_client(gateway).chat.completions.create(
            model=model,
            messages=_second_turn(),  # pyright: ignore[reportArgumentType]  # plain JSON messages
            tools=[_WEATHER_TOOL],  # pyright: ignore[reportArgumentType]  # plain JSON tool
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"cache": _NO_CACHE},
        )
        chunks: Final = [chunk async for chunk in stream]
        assert "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices) == _ANSWER
        assert [call for chunk in chunks for choice in chunk.choices for call in (choice.delta.tool_calls or [])] == []
        assert [choice.finish_reason for chunk in chunks for choice in chunk.choices if choice.finish_reason] == ["stop"]
        body: Final = _only_generate(wire)
        assert body["stream"] is True
        _assert_tool_turn(_prompt_of(body))
        assert _spend_row(chunks[0].id) == _billed(model)


def test_anthropic_sdk_tool_request_comes_back_as_tool_use(gateway: Gateway) -> None:
    with _ollama_server(lambda _: _generate_reply(_CALL_JSON)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        message: Final = _anthropic_client(gateway).messages.create(
            model=model,
            max_tokens=64,
            messages=_first_turn(),  # pyright: ignore[reportArgumentType]  # plain JSON messages
            tools=[_ANTHROPIC_TOOL],  # pyright: ignore[reportArgumentType]  # plain JSON tool
            extra_body={"cache": _NO_CACHE},
        )
        assert message.stop_reason == "tool_use"
        assert [block.type for block in message.content] == ["tool_use"]
        block: Final = message.content[0]
        assert block.type == "tool_use"
        assert block.name == "get_weather"
        assert block.input == _ARGUMENTS
        assert (message.usage.input_tokens, message.usage.output_tokens) == (30, 12)
        prompt: Final = _prompt_of(_only_generate(wire))
        _assert_instructed_once(prompt, "get_weather")
        assert f"### User:\n{_QUESTION}\n\n" in prompt, prompt
        assert _spend_row(message.id) == _billed(model)


def test_anthropic_sdk_tool_result_turn_ends_with_text(gateway: Gateway) -> None:
    with _ollama_server(lambda _: _generate_reply(_ANSWER)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        message: Final = _anthropic_client(gateway).messages.create(
            model=model,
            max_tokens=64,
            messages=_anthropic_second_turn(),  # pyright: ignore[reportArgumentType]  # plain JSON messages
            tools=[_ANTHROPIC_TOOL],  # pyright: ignore[reportArgumentType]  # plain JSON tool
            extra_body={"cache": _NO_CACHE},
        )
        assert message.stop_reason == "end_turn"
        assert [(block.type, getattr(block, "text", None)) for block in message.content] == [("text", _ANSWER)]
        _assert_tool_turn(_prompt_of(_only_generate(wire)))
        assert _spend_row(message.id) == _billed(model)


async def test_async_anthropic_sdk_stream_emits_the_tool_use_block(gateway: Gateway) -> None:
    with _ollama_server(lambda _: _streamed_reply(_CALL_JSON)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        async with _async_anthropic_client(gateway).messages.stream(
            model=model,
            max_tokens=64,
            messages=_first_turn(),  # pyright: ignore[reportArgumentType]  # plain JSON messages
            tools=[_ANTHROPIC_TOOL],  # pyright: ignore[reportArgumentType]  # plain JSON tool
            extra_body={"cache": _NO_CACHE},
        ) as stream:
            events: Final = [event async for event in stream if event.type in _RAW_ANTHROPIC_EVENTS]
            final: Final = await stream.get_final_message()
        starts: Final = [event for event in events if event.type == "content_block_start"]
        assert [event.content_block.type for event in starts] == ["tool_use"], [event.type for event in events]
        assert any(
            event.type == "content_block_delta" and event.delta.type == "input_json_delta" for event in events
        ), [event.type for event in events]
        assert final.stop_reason == "tool_use"
        block: Final = final.content[0]
        assert block.type == "tool_use" and block.name == "get_weather" and block.input == _ARGUMENTS
        body: Final = _only_generate(wire)
        assert body["stream"] is True
        _assert_instructed_once(_prompt_of(body), "get_weather")
        assert _spend_row(final.id) == _billed(model)


async def test_async_anthropic_sdk_stream_answers_the_tool_result_in_text(gateway: Gateway) -> None:
    with _ollama_server(lambda _: _streamed_reply(_ANSWER)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        async with _async_anthropic_client(gateway).messages.stream(
            model=model,
            max_tokens=64,
            messages=_anthropic_second_turn(),  # pyright: ignore[reportArgumentType]  # plain JSON messages
            tools=[_ANTHROPIC_TOOL],  # pyright: ignore[reportArgumentType]  # plain JSON tool
            extra_body={"cache": _NO_CACHE},
        ) as stream:
            texts: Final = [event.text async for event in stream if event.type == "text"]
            final: Final = await stream.get_final_message()
        assert "".join(texts) == _ANSWER
        assert final.stop_reason == "end_turn"
        assert [(block.type, getattr(block, "text", None)) for block in final.content] == [("text", _ANSWER)]
        body: Final = _only_generate(wire)
        assert body["stream"] is True
        _assert_tool_turn(_prompt_of(body))
        assert _spend_row(final.id) == _billed(model)


def test_openai_sdk_responses_tool_request_comes_back_as_a_function_call(gateway: Gateway) -> None:
    with _ollama_server(lambda _: _generate_reply(_CALL_JSON)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = _openai_client(gateway).responses.create(
            model=model,
            input=_QUESTION,
            tools=[_RESPONSES_TOOL],  # pyright: ignore[reportArgumentType]  # plain JSON tool
            store=False,
            extra_body={"cache": _NO_CACHE},
        )
        assert [item.type for item in response.output] == ["function_call"]
        item: Final = response.output[0]
        assert item.type == "function_call"
        assert item.name == "get_weather"
        assert json.loads(item.arguments) == _ARGUMENTS
        assert response.usage is not None and (response.usage.input_tokens, response.usage.output_tokens) == (30, 12)
        prompt: Final = _prompt_of(_only_generate(wire))
        _assert_instructed_once(prompt, "get_weather")
        assert f"### User:\n{_QUESTION}\n\n" in prompt, prompt
        assert _model_spend_rows(model, 1)[0]["status"] == "success"


def test_openai_sdk_responses_function_output_turn_gets_a_message(gateway: Gateway) -> None:
    with _ollama_server(lambda _: _generate_reply(_ANSWER)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        response: Final = _openai_client(gateway).responses.create(
            model=model,
            input=_responses_second_turn(),  # pyright: ignore[reportArgumentType]  # plain JSON input items
            tools=[_RESPONSES_TOOL],  # pyright: ignore[reportArgumentType]  # plain JSON tool
            store=False,
            extra_body={"cache": _NO_CACHE},
        )
        assert [item.type for item in response.output] == ["message"]
        assert response.output_text == _ANSWER
        _assert_tool_turn(_prompt_of(_only_generate(wire)))
        rows: Final = _model_spend_rows(model, 1)
        assert (rows[0]["status"], rows[0]["prompt_tokens"], rows[0]["completion_tokens"]) == ("success", 30, 12)


async def test_async_openai_sdk_responses_stream_emits_the_function_call_item(gateway: Gateway) -> None:
    with _ollama_server(lambda _: _streamed_reply(_CALL_JSON)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        stream: Final = await _async_openai_client(gateway).responses.create(
            model=model,
            input=_QUESTION,
            tools=[_RESPONSES_TOOL],  # pyright: ignore[reportArgumentType]  # plain JSON tool
            store=False,
            stream=True,
            extra_body={"cache": _NO_CACHE},
        )
        events: Final = [event async for event in stream]
        done_items: Final = [event.item for event in events if event.type == "response.output_item.done"]
        assert [item.type for item in done_items] == ["function_call"], [event.type for event in events]
        item: Final = done_items[0]
        assert item.type == "function_call" and item.name == "get_weather"
        assert json.loads(item.arguments) == _ARGUMENTS
        completed: Final = [event for event in events if event.type == "response.completed"]
        assert len(completed) == 1
        final_calls: Final = [item for item in completed[0].response.output if item.type == "function_call"]
        assert [(item.name, json.loads(item.arguments)) for item in final_calls] == [("get_weather", _ARGUMENTS)]
        assert completed[0].response.output_text == ""
        body: Final = _only_generate(wire)
        assert body["stream"] is True
        _assert_instructed_once(_prompt_of(body), "get_weather")
        assert _model_spend_rows(model, 1)[0]["status"] == "success"


async def test_async_openai_sdk_responses_stream_answers_the_function_output_in_text(gateway: Gateway) -> None:
    with _ollama_server(lambda _: _streamed_reply(_ANSWER)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        stream: Final = await _async_openai_client(gateway).responses.create(
            model=model,
            input=_responses_second_turn(),  # pyright: ignore[reportArgumentType]  # plain JSON input items
            tools=[_RESPONSES_TOOL],  # pyright: ignore[reportArgumentType]  # plain JSON tool
            store=False,
            stream=True,
            extra_body={"cache": _NO_CACHE},
        )
        events: Final = [event async for event in stream]
        assert "".join(event.delta for event in events if event.type == "response.output_text.delta") == _ANSWER
        completed: Final = [event for event in events if event.type == "response.completed"]
        assert len(completed) == 1 and completed[0].response.output_text == _ANSWER
        assert [event.type for event in events if event.type == "response.output_item.done"] == ["response.output_item.done"]
        body: Final = _only_generate(wire)
        assert body["stream"] is True
        _assert_tool_turn(_prompt_of(body))
        assert _model_spend_rows(model, 1)[0]["status"] == "success"


def test_legacy_functions_param_is_instructed_the_same_way(gateway: Gateway) -> None:
    with _ollama_server(lambda _: _generate_reply(_CALL_JSON)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        function: Final = _WEATHER_TOOL["function"]
        payload: Final = _post_chat(gateway, model, _first_turn(), functions=[function])
        choices: Final = payload["choices"]
        assert isinstance(choices, list) and len(choices) == 1
        assert _CALL_JSON_FIELDS <= set(json.dumps(choices[0]).split('"')), choices
        prompt: Final = _prompt_of(_only_generate(wire))
        _assert_instructed_once(prompt, "get_weather")
        assert "Weather for a city" in prompt, prompt
        identity: Final = payload["id"]
        assert isinstance(identity, str)
        assert _spend_row(identity) == _billed(model)


def test_string_system_message_keeps_one_system_section_with_the_instruction(gateway: Gateway) -> None:
    system: Final = f"You are a terse weather bot {uuid.uuid4().hex}."
    with _ollama_server(lambda _: _generate_reply(_CALL_JSON)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        _post_chat(gateway, model, [{"role": "system", "content": system}, *_first_turn()], tools=[_WEATHER_TOOL])
        prompt: Final = _prompt_of(_only_generate(wire))
        _assert_instructed_once(prompt, "get_weather")
        assert f"### System:\n{system} {_INSTRUCTION}\n" in prompt, prompt


def test_list_system_message_keeps_one_system_section_with_the_instruction(gateway: Gateway) -> None:
    system: Final = f"You are a terse weather bot {uuid.uuid4().hex}."
    with _ollama_server(lambda _: _generate_reply(_CALL_JSON)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        _post_chat(
            gateway,
            model,
            [{"role": "system", "content": [{"type": "text", "text": system}]}, *_first_turn()],
            tools=[_WEATHER_TOOL],
        )
        prompt: Final = _prompt_of(_only_generate(wire))
        _assert_instructed_once(prompt, "get_weather")
        section: Final = prompt.split("### System:\n", 1)[1]
        assert section.startswith(system), section
        assert section.count(_INSTRUCTION) == 1, section


def test_two_tools_are_both_listed_under_one_instruction(gateway: Gateway) -> None:
    with _ollama_server(lambda _: _generate_reply(_CALL_JSON)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        _post_chat(gateway, model, _first_turn(), tools=[_WEATHER_TOOL, _TIME_TOOL])
        prompt: Final = _prompt_of(_only_generate(wire))
        _assert_instructed_once(prompt, "get_weather", "get_time")
        assert prompt.index("'name': 'get_weather'") < prompt.index("'name': 'get_time'"), prompt


def test_unauthenticated_tool_request_never_reaches_ollama(gateway: Gateway) -> None:
    with _ollama_server(lambda _: _generate_reply(_CALL_JSON)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        code, text = _post(
            gateway,
            "/v1/chat/completions",
            {"model": model, "messages": _first_turn(), "tools": [_WEATHER_TOOL]},
            key=f"sk-not-a-key-{uuid.uuid4().hex}",
        )
        assert code == 401, text
        assert _generate_calls(wire) == ()


def test_ollama_model_not_found_reaches_the_caller_after_one_attempt(gateway: Gateway) -> None:
    message: Final = f"model '{_BACKEND}' not found {uuid.uuid4().hex}"
    reply: Final = Reply(status=404, body=json.dumps({"error": message}).encode())
    with _ollama_server(lambda _: reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        code, text = _post(gateway, "/v1/chat/completions", {"model": model, "messages": _first_turn(), "tools": [_WEATHER_TOOL]})
        assert code == 404, text
        assert message in text, text
        _assert_instructed_once(_prompt_of(_only_generate(wire)), "get_weather")


def test_ollama_server_error_on_the_tool_result_turn_does_not_take_the_deployment_down(gateway: Gateway) -> None:
    attempts: Final = itertools.count()
    failure: Final = f"internal failure {uuid.uuid4().hex}"

    def respond(_: Request) -> Reply:
        if next(attempts) == 0:
            return Reply(status=500, body=json.dumps({"error": failure}).encode())
        return _generate_reply(_ANSWER)

    with _ollama_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        code, text = _post(gateway, "/v1/chat/completions", {"model": model, "messages": _second_turn(), "tools": [_WEATHER_TOOL]})
        assert code == 500, text
        assert failure in text, text
        payload: Final = _post_chat(gateway, model, _second_turn(), tools=[_WEATHER_TOOL])
        choices: Final = payload["choices"]
        assert isinstance(choices, list) and len(choices) == 1
        assert json.dumps(choices[0]).count(_ANSWER) == 1, choices
        prompts: Final = [_prompt_of(_JSON_OBJECT.validate_json(request.body)) for request in _generate_calls(wire)]
        assert len(prompts) == 2, prompts
        for prompt in prompts:
            _assert_tool_turn(prompt)


@pytest.mark.parametrize(
    ("result", "forwarded"),
    [
        pytest.param("", None, id="empty-string-drops-the-section"),
        pytest.param("r" * 5120, "r" * 5120, id="5kb-string-forwarded-intact"),
        pytest.param(
            [{"type": "text", "text": "Paris: 22 degrees"}, {"type": "text", "text": "clear skies"}],
            "Paris: 22 degreesclear skies",
            id="text-parts-joined",
        ),
    ],
)
def test_tool_result_content_shapes_reach_the_prompt(gateway: Gateway, result: JsonValue, forwarded: str | None) -> None:
    with _ollama_server(lambda _: _generate_reply(_ANSWER)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        payload: Final = _post_chat(gateway, model, _second_turn(result), tools=[_WEATHER_TOOL])
        assert json.dumps(payload["choices"]).count(_ANSWER) == 1, payload
        prompt: Final = _prompt_of(_only_generate(wire))
        _assert_instructed_once(prompt, "get_weather")
        if forwarded is None:
            assert f"### User:\n{_QUESTION}\n\n### Assistant:\n{_CALL_JSON}\n\n### System:\n" in prompt, prompt
        else:
            _assert_tool_turn(prompt, forwarded)


def test_the_same_tool_result_twice_is_forwarded_twice_under_one_instruction(gateway: Gateway) -> None:
    with _ollama_server(lambda _: _generate_reply(_ANSWER)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        messages: Final = [*_second_turn(), {"role": "tool", "tool_call_id": _CALL_ID, "content": _RESULT}]
        _post_chat(gateway, model, messages, tools=[_WEATHER_TOOL])
        prompt: Final = _prompt_of(_only_generate(wire))
        _assert_instructed_once(prompt, "get_weather")
        assert prompt.count(_RESULT) == 2, prompt
        assert prompt.count("### User:") == 2 and prompt.count("### Assistant:") == 1, prompt


def test_a_second_function_call_after_the_result_is_surfaced_as_tool_calls(gateway: Gateway) -> None:
    with _ollama_server(lambda _: _generate_reply(_CALL_JSON)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        completion: Final = _openai_client(gateway).chat.completions.create(
            model=model,
            messages=_second_turn(),  # pyright: ignore[reportArgumentType]  # plain JSON messages
            tools=[_WEATHER_TOOL],  # pyright: ignore[reportArgumentType]  # plain JSON tool
            extra_body={"cache": _NO_CACHE},
        )
        choice: Final = completion.choices[0]
        assert choice.finish_reason == "tool_calls"
        assert choice.message.tool_calls is not None and [call.function.name for call in choice.message.tool_calls] == [
            "get_weather"
        ]
        _assert_tool_turn(_prompt_of(_only_generate(wire)))
        assert _spend_row(completion.id) == _billed(model)


def test_a_non_function_json_answer_is_returned_as_text(gateway: Gateway) -> None:
    answer: Final = json.dumps({"city": "Paris", "temperature_c": 22, "sky": "clear"})
    with _ollama_server(lambda _: _generate_reply(answer)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        completion: Final = _openai_client(gateway).chat.completions.create(
            model=model,
            messages=_second_turn(),  # pyright: ignore[reportArgumentType]  # plain JSON messages
            tools=[_WEATHER_TOOL],  # pyright: ignore[reportArgumentType]  # plain JSON tool
            extra_body={"cache": _NO_CACHE},
        )
        choice: Final = completion.choices[0]
        assert choice.finish_reason == "stop"
        assert choice.message.tool_calls is None
        assert choice.message.content is not None and json.loads(choice.message.content) == json.loads(answer)
        _assert_tool_turn(_prompt_of(_only_generate(wire)))
        assert _spend_row(completion.id) == _billed(model)


def test_int_tool_result_content_fails_in_the_response_body_and_leaves_the_deployment_serving(gateway: Gateway) -> None:
    with _ollama_server(lambda _: _generate_reply(_ANSWER)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        code, text = _post(gateway, "/v1/chat/completions", {"model": model, "messages": _second_turn(22), "tools": [_WEATHER_TOOL]})
        assert code >= 400, text
        error: Final = _JSON_OBJECT.validate_json(text)["error"]
        assert isinstance(error, dict) and isinstance(error["message"], str) and error["message"], text
        assert _generate_calls(wire) == ()
        payload: Final = _post_chat(gateway, model, _second_turn(), tools=[_WEATHER_TOOL])
        assert json.dumps(payload["choices"]).count(_ANSWER) == 1, payload
        _assert_tool_turn(_prompt_of(_only_generate(wire)))


def test_ollama_chat_keeps_native_tools_and_gets_no_instruction(gateway: Gateway) -> None:
    reply: Final = Reply(
        body=json.dumps(
            {
                "model": _BACKEND,
                "created_at": "2026-10-07T00:00:00Z",
                "message": {"role": "assistant", "content": _ANSWER},
                "done": True,
                "prompt_eval_count": 30,
                "eval_count": 12,
            }
        ).encode()
    )
    with wire_server(lambda _: reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama_chat/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        completion: Final = _openai_client(gateway).chat.completions.create(
            model=model,
            messages=_second_turn(),  # pyright: ignore[reportArgumentType]  # plain JSON messages
            tools=[_WEATHER_TOOL],  # pyright: ignore[reportArgumentType]  # plain JSON tool
            extra_body={"cache": _NO_CACHE},
        )
        assert completion.choices[0].message.content == _ANSWER
        received: Final = tuple(request for request in wire.drain() if request.method == "POST")
        assert [request.target for request in received] == ["/api/chat"]
        body: Final = _JSON_OBJECT.validate_json(received[0].body)
        assert "prompt" not in body and "format" not in body, sorted(body)
        tools: Final = body["tools"]
        assert isinstance(tools, list) and len(tools) == 1
        messages: Final = body["messages"]
        assert isinstance(messages, list) and [item["role"] for item in messages if isinstance(item, dict)] == [
            "user",
            "assistant",
            "tool",
        ]
        assert "function_name" not in received[0].body.decode(), received[0].body
        assert _spend_row(completion.id) == _billed(model)


def test_identical_uncached_tool_result_turns_are_each_forwarded_and_billed_once(gateway: Gateway) -> None:
    with _ollama_server(lambda _: _generate_reply(_ANSWER)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        first: Final = _post_chat(gateway, model, _second_turn(), tools=[_WEATHER_TOOL])
        second: Final = _post_chat(gateway, model, _second_turn(), tools=[_WEATHER_TOOL])
        assert first["id"] != second["id"], (first["id"], second["id"])
        prompts: Final = [_prompt_of(_JSON_OBJECT.validate_json(request.body)) for request in _generate_calls(wire)]
        assert len(prompts) == 2, prompts
        for prompt in prompts:
            _assert_tool_turn(prompt)
        for payload in (first, second):
            identity: Final = payload["id"]
            assert isinstance(identity, str)
            assert _spend_row(identity) == _billed(model)


def test_the_cell_deployment_is_gone_after_its_scenario(gateway: Gateway) -> None:
    with _ollama_server(lambda _: _generate_reply(_CALL_JSON)) as wire:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(model=f"ollama/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
            _post_chat(gateway, model, _first_turn(), tools=[_WEATHER_TOOL])
            _only_generate(wire)
        listed: Final = eventually(
            lambda: [entry["model_name"] for entry in _deployments(gateway) if isinstance(entry, dict)],
            lambda names: model not in names,
            seconds=70,
        )
        assert model not in listed


def _deployments(gateway: Gateway) -> list[JsonValue]:
    data: Final = gateway.get("/model/info")["data"]
    assert isinstance(data, list)
    return data
