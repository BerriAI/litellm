import asyncio
import json
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal
from urllib.parse import unquote

import httpx
import pytest
from integration._support import claude_code as cc
from integration._support import responses_vendor as rv
from integration._support.anthropic_thinking import BEDROCK_MODEL, EVENT_STREAM, Event, aws_chunks, sse_chunks
from integration._support.client import Gateway, Scenario
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_CLAUDE: Final = cc.OPUS
_REGEX_TOOL_TYPE: Final = "tool_search_tool_regex_20251119"
_REGEX_TOOL_NAME: Final = "tool_search_tool_regex"
_REGEX_TOOL: Final[Mapping[str, JsonValue]] = MappingProxyType({"type": _REGEX_TOOL_TYPE, "name": _REGEX_TOOL_NAME})
_BEDROCK_REGEX_TOOL: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"type": _REGEX_TOOL_NAME, "name": _REGEX_TOOL_NAME}
)
_SEARCH_ARGUMENTS: Final = '{"pattern": "weather|stock"}'
_SEARCH_OUTPUT: Final = '{"tools": ["get_weather"]}'
_CACHE_BUST: Final[Mapping[str, JsonValue]] = MappingProxyType({"cache": {"no-cache": True}})
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_ITEMS: Final = TypeAdapter(list[dict[str, JsonValue]])
_EMPTY_TURN: Final[Mapping[str, JsonValue]] = MappingProxyType({"role": "user", "content": ""})
_EARLIER_PAIRS: Final = 200
_WEATHER_PARAMETERS: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}

Provider = Literal["anthropic", "bedrock_invoke"]
Answer = Literal["unanswered", "answered"]


@dataclass(frozen=True, slots=True)
class _Call:
    provider: Provider
    answer: Answer
    stream: bool
    marker: str


@dataclass(frozen=True, slots=True)
class _Served:
    call: _Call
    status: int
    text: str


def _prompt(marker: str) -> str:
    return f"Weather in Paris and the ACME price marker-{marker}"


def _message(marker: str) -> dict[str, JsonValue]:
    return {
        "id": f"msg_{marker}",
        "type": "message",
        "role": "assistant",
        "model": _CLAUDE,
        "content": [{"type": "text", "text": f"tomorrow is sunny marker-{marker}"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 20, "output_tokens": 7},
    }


def _events(marker: str) -> tuple[Event, ...]:
    opened: Final = {**_message(marker), "content": [], "stop_reason": None}
    return (
        {"type": "message_start", "message": opened},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": f"tomorrow is sunny marker-{marker}"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 7},
        },
        {"type": "message_stop"},
    )


def _anthropic_error(status: int, message: str) -> Reply:
    body: Final = {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
    return Reply(status=status, body=json.dumps(body).encode())


def _block_ids(turn: Mapping[str, JsonValue], block_type: str, key: str) -> frozenset[str]:
    content: Final = turn.get("content")
    if not isinstance(content, list):
        return frozenset()
    blocks: Final = _ITEMS.validate_python(content)
    return frozenset(str(block[key]) for block in blocks if block.get("type") == block_type)


def _mismatch(messages: Sequence[Mapping[str, JsonValue]], index: int) -> frozenset[str]:
    following: Final = messages[index + 1] if index + 1 < len(messages) else _EMPTY_TURN
    return _block_ids(messages[index], "tool_use", "id") ^ _block_ids(following, "tool_result", "tool_use_id")


def _unpaired(messages: Sequence[Mapping[str, JsonValue]]) -> frozenset[str]:
    assistant_turns: Final = [index for index, turn in enumerate(messages) if turn["role"] == "assistant"]
    return frozenset().union(*(_mismatch(messages, index) for index in assistant_turns))


def _vendor(request: Request) -> Reply:
    if request.method != "POST":
        return Reply()
    body: Final = _JSON_OBJECT.validate_json(request.body)
    marker: Final = rv.newest_marker(request.body.decode()) or "health"
    unpaired: Final = _unpaired(_ITEMS.validate_python(body.get("messages") or []))
    if unpaired:
        return _anthropic_error(400, f"messages: unpaired tool ids {sorted(unpaired)} marker-{marker}")
    if request.target.endswith("/invoke-with-response-stream"):
        return Reply(content_type=EVENT_STREAM, chunks=aws_chunks(_events(marker)))
    if body.get("stream") is True:
        return Reply(content_type="text/event-stream", chunks=sse_chunks(_events(marker)))
    return Reply(body=json.dumps(_message(marker)).encode())


def _deployment(scenario: Scenario, provider: Provider, wire_url: str) -> str:
    match provider:
        case "anthropic":
            return scenario.model(model=f"anthropic/{_CLAUDE}", api_base=wire_url, api_key=cc.ANTHROPIC_API_KEY)
        case "bedrock_invoke":
            return scenario.model(
                model=f"bedrock/invoke/{BEDROCK_MODEL}",
                api_base=wire_url,
                aws_access_key_id="AKIASCRIPTEDPROVIDER",
                aws_secret_access_key="scripted-secret",
                aws_region_name="us-east-1",
                aws_bedrock_runtime_endpoint=wire_url,
            )


def _target(provider: Provider, stream: bool) -> str:
    match provider:
        case "anthropic":
            return "/v1/messages"
        case "bedrock_invoke":
            return f"/model/{BEDROCK_MODEL}/invoke-with-response-stream" if stream else f"/model/{BEDROCK_MODEL}/invoke"


def _search_call(marker: str) -> dict[str, JsonValue]:
    return {
        "id": f"srvtoolu_{marker}",
        "type": "function",
        "function": {"name": _REGEX_TOOL_NAME, "arguments": _SEARCH_ARGUMENTS},
    }


def _weather_call(marker: str) -> dict[str, JsonValue]:
    return {
        "id": f"toolu_{marker}",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
    }


def _tool_message(tool_call_id: JsonValue, content: str) -> dict[str, JsonValue]:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def _history(marker: str, answers: Sequence[Mapping[str, JsonValue]]) -> list[JsonValue]:
    return [
        {"role": "user", "content": _prompt(marker)},
        {
            "role": "assistant",
            "content": f"searching marker-{marker}",
            "tool_calls": [_search_call(marker), _weather_call(marker)],
        },
        *(dict(answer) for answer in answers),
        _tool_message(f"toolu_{marker}", "22C"),
        {"role": "user", "content": f"And tomorrow? marker-{marker}"},
    ]


def _answers(marker: str, answer: Answer) -> tuple[dict[str, JsonValue], ...]:
    return (_tool_message(f"srvtoolu_{marker}", _SEARCH_OUTPUT),) if answer == "answered" else ()


def _chat_tools() -> list[JsonValue]:
    return [
        dict(_REGEX_TOOL),
        {
            "type": "function",
            "function": {"name": "get_weather", "description": "Weather for a city", "parameters": _WEATHER_PARAMETERS},
            "defer_loading": True,
        },
    ]


def _chat_body(model: str, marker: str, messages: Sequence[JsonValue], stream: bool) -> dict[str, JsonValue]:
    return {"model": model, "messages": list(messages), "tools": _chat_tools(), "stream": stream, **_CACHE_BUST}


def _raw(gateway: Gateway, path: str, body: Mapping[str, JsonValue]) -> httpx.Response:
    with httpx.Client(base_url=str(gateway.client.base_url).rstrip("/"), trust_env=False, timeout=60) as client:
        headers: Final = {"Authorization": f"Bearer {gateway.key}"}
        with client.stream("POST", path, json=dict(body), headers=headers) as response:
            response.read()
            return response


def _assert_served(response: httpx.Response, marker: str, stream: bool) -> None:
    assert response.status_code == 200, response.text
    if stream:
        assert "data: [DONE]" in response.text and f"marker-{marker}" in response.text, response.text
        return
    choice: Final = _ITEMS.validate_python(_JSON_OBJECT.validate_json(response.content)["choices"])[0]
    assert f"marker-{marker}" in str(_JSON_OBJECT.validate_python(choice["message"])["content"]), response.text


def _only_request(wire: Wire) -> tuple[Request, dict[str, JsonValue]]:
    received: Final = wire.drain()
    assert len(received) == 1, [(request.method, request.target) for request in received]
    return received[0], _JSON_OBJECT.validate_json(received[0].body)


def _assistant_tool_uses(messages: Sequence[Mapping[str, JsonValue]]) -> list[tuple[str, str]]:
    (assistant,) = [turn for turn in messages if turn["role"] == "assistant"]
    blocks: Final = _ITEMS.validate_python(assistant["content"])
    return sorted((str(block["id"]), str(block["name"])) for block in blocks if block["type"] == "tool_use")


def _tool_results(messages: Sequence[Mapping[str, JsonValue]]) -> Iterator[str]:
    for turn in messages:
        if turn["role"] != "user" or not isinstance(turn["content"], list):
            continue
        yield from (
            str(block["tool_use_id"])
            for block in _ITEMS.validate_python(turn["content"])
            if block["type"] == "tool_result"
        )


def _assert_tools_forwarded(provider: Provider, body: Mapping[str, JsonValue]) -> None:
    tools: Final = _ITEMS.validate_python(body["tools"])
    assert tools[0] == dict(_BEDROCK_REGEX_TOOL if provider == "bedrock_invoke" else _REGEX_TOOL), tools
    assert [(tool["name"], tool.get("defer_loading")) for tool in tools[1:]] == [("get_weather", True)], tools


def _assert_wire(call: _Call, request: Request, body: Mapping[str, JsonValue]) -> None:
    assert unquote(request.target) == _target(call.provider, call.stream), request.target
    _assert_tools_forwarded(call.provider, body)
    messages: Final = _ITEMS.validate_python(body["messages"])
    weather: Final = (f"toolu_{call.marker}", "get_weather")
    search: Final = (f"srvtoolu_{call.marker}", _REGEX_TOOL_NAME)
    match call.answer:
        case "unanswered":
            assert "srvtoolu_" not in json.dumps(messages), messages
            assert _assistant_tool_uses(messages) == [weather], messages
            assert sorted(_tool_results(messages)) == [weather[0]], messages
        case "answered":
            assert _assistant_tool_uses(messages) == [search, weather], messages
            assert sorted(_tool_results(messages)) == [search[0], weather[0]], messages


@pytest.mark.parametrize("stream", [False, True], ids=["sync", "stream"])
@pytest.mark.parametrize("answer", ["unanswered", "answered"])
@pytest.mark.parametrize("provider", ["anthropic", "bedrock_invoke"])
def test_a_replayed_search_call_is_dropped_only_when_nothing_answers_it(
    gateway: Gateway, provider: Provider, answer: Answer, stream: bool
) -> None:
    call: Final = _Call(provider, answer, stream, uuid.uuid4().hex)
    with wire_server(_vendor) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, provider, wire.url)
        history: Final = _history(call.marker, _answers(call.marker, answer))
        response: Final = _raw(gateway, "/v1/chat/completions", _chat_body(model, call.marker, history, stream))
        _assert_served(response, call.marker, stream)
        request, body = _only_request(wire)
        _assert_wire(call, request, body)


def _earlier_pairs(marker: str) -> Iterator[JsonValue]:
    for index in range(_EARLIER_PAIRS):
        call_id = f"toolu_{marker}_{index}"
        call: JsonValue = {
            "id": call_id,
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
        }
        yield {"role": "assistant", "content": f"step {index}", "tool_calls": [call]}
        yield _tool_message(call_id, "22C")


def _long_history(marker: str, answers: Sequence[Mapping[str, JsonValue]]) -> tuple[JsonValue, ...]:
    first, *rest = _history(marker, answers)
    return (first, *_earlier_pairs(marker), *rest)


def _last_assistant_tool_uses(messages: Sequence[Mapping[str, JsonValue]]) -> tuple[tuple[str, str], ...]:
    last: Final = [turn for turn in messages if turn["role"] == "assistant"][-1]
    blocks: Final = _ITEMS.validate_python(last["content"])
    return tuple(sorted((str(block["id"]), str(block["name"])) for block in blocks if block["type"] == "tool_use"))


@pytest.mark.parametrize("answer", ["unanswered", "answered"])
def test_a_search_call_after_two_hundred_answered_tool_turns_is_paired_only_by_its_own_answers(
    gateway: Gateway, answer: Answer
) -> None:
    call: Final = _Call("anthropic", answer, False, uuid.uuid4().hex)
    with wire_server(_vendor) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url)
        history: Final = _long_history(call.marker, _answers(call.marker, answer))
        response: Final = _raw(gateway, "/v1/chat/completions", _chat_body(model, call.marker, history, False))
        _assert_served(response, call.marker, False)
        request, body = _only_request(wire)
        assert request.target == "/v1/messages", request.target
        _assert_tools_forwarded("anthropic", body)
        messages: Final = _ITEMS.validate_python(body["messages"])
        earlier: Final = [f"toolu_{call.marker}_{index}" for index in range(_EARLIER_PAIRS)]
        weather: Final = (f"toolu_{call.marker}", "get_weather")
        search: Final = (f"srvtoolu_{call.marker}", _REGEX_TOOL_NAME)
        match answer:
            case "unanswered":
                assert "srvtoolu_" not in json.dumps(messages), messages[-4:]
                assert _last_assistant_tool_uses(messages) == (weather,), messages[-4:]
                assert sorted(_tool_results(messages)) == sorted([*earlier, weather[0]]), messages[-4:]
            case "answered":
                assert _last_assistant_tool_uses(messages) == (search, weather), messages[-4:]
                assert sorted(_tool_results(messages)) == sorted([*earlier, search[0], weather[0]]), messages[-4:]


def test_a_function_role_answer_keeps_the_replayed_search_call(gateway: Gateway) -> None:
    call: Final = _Call("anthropic", "answered", False, uuid.uuid4().hex)
    answer: Final[dict[str, JsonValue]] = {
        "role": "function",
        "name": _REGEX_TOOL_NAME,
        "tool_call_id": f"srvtoolu_{call.marker}",
        "content": _SEARCH_OUTPUT,
    }
    with wire_server(_vendor) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url)
        history: Final = _history(call.marker, (answer,))
        response: Final = _raw(gateway, "/v1/chat/completions", _chat_body(model, call.marker, history, False))
        _assert_served(response, call.marker, False)
        request, body = _only_request(wire)
        _assert_wire(call, request, body)


def test_a_duplicated_answer_keeps_the_search_call_with_every_result(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    repeated: Final = _tool_message(f"srvtoolu_{marker}", _SEARCH_OUTPUT)
    with wire_server(_vendor) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url)
        history: Final = _history(marker, (repeated, repeated))
        response: Final = _raw(gateway, "/v1/chat/completions", _chat_body(model, marker, history, False))
        _assert_served(response, marker, False)
        request, body = _only_request(wire)
        assert request.target == "/v1/messages", request.target
        messages: Final = _ITEMS.validate_python(body["messages"])
        assert _assistant_tool_uses(messages) == [
            (f"srvtoolu_{marker}", _REGEX_TOOL_NAME),
            (f"toolu_{marker}", "get_weather"),
        ]
        assert sorted(_tool_results(messages)) == [f"srvtoolu_{marker}", f"srvtoolu_{marker}", f"toolu_{marker}"], (
            messages
        )


@pytest.mark.parametrize("tool_call_id", ["", "x" * 5000], ids=["empty-string", "five-kilobyte-string"])
def test_an_answer_whose_id_matches_nothing_leaves_the_search_call_dropped(gateway: Gateway, tool_call_id: str) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_vendor) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url)
        history: Final = _history(marker, (_tool_message(tool_call_id, _SEARCH_OUTPUT),))
        response: Final = _raw(gateway, "/v1/chat/completions", _chat_body(model, marker, history, False))
        assert response.status_code == 400 and "unpaired tool ids" in response.text, response.text
        _, body = _only_request(wire)
        messages: Final = _ITEMS.validate_python(body["messages"])
        assert "srvtoolu_" not in json.dumps(messages), messages
        assert _assistant_tool_uses(messages) == [(f"toolu_{marker}", "get_weather")], messages


@pytest.mark.parametrize("tool_call_id", [5, ["srvtoolu_"]], ids=["int", "list"])
def test_a_non_string_answer_id_is_refused_with_400_before_the_wire_and_the_proxy_keeps_serving(
    gateway: Gateway, tool_call_id: JsonValue
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_vendor) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, "anthropic", wire.url)
        history: Final = _history(marker, (_tool_message(tool_call_id, _SEARCH_OUTPUT),))
        refused: Final = _raw(gateway, "/v1/chat/completions", _chat_body(model, marker, history, False))
        assert refused.status_code == 400, refused.text
        assert wire.drain() == (), refused.text
        follow_up: Final = uuid.uuid4().hex
        history_served: Final = _history(follow_up, _answers(follow_up, "answered"))
        served: Final = _raw(gateway, "/v1/chat/completions", _chat_body(model, follow_up, history_served, False))
        _assert_served(served, follow_up, False)
        assert len(wire.drain()) == 1


async def _send(client: httpx.AsyncClient, key: str, model: str, call: _Call) -> _Served:
    body: Final = _chat_body(model, call.marker, _history(call.marker, _answers(call.marker, call.answer)), call.stream)
    async with client.stream(
        "POST", "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"}
    ) as response:
        raw: Final = await response.aread()
    return _Served(call, response.status_code, raw.decode())


async def _burst(
    base_url: str, key: str, models: Mapping[Provider, str], calls: Sequence[_Call]
) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(base_url=base_url, timeout=60, trust_env=False) as client:
        return tuple(await asyncio.gather(*(_send(client, key, models[call.provider], call) for call in calls)))


def _burst_calls(count: int) -> tuple[_Call, ...]:
    providers: Final[tuple[Provider, ...]] = ("anthropic", "bedrock_invoke")
    answers: Final[tuple[Answer, ...]] = ("unanswered", "answered")
    return tuple(
        _Call(providers[index % 2], answers[(index // 2) % 2], index % 4 >= 2, uuid.uuid4().hex)
        for index in range(count)
    )


def _marked(received: Sequence[Request], marker: str) -> tuple[Request, dict[str, JsonValue]]:
    (request,) = [request for request in received if marker in request.body.decode()]
    return request, _JSON_OBJECT.validate_json(request.body)


def test_a_mixed_burst_of_replays_pairs_every_answered_call_and_drops_every_unanswered_one(gateway: Gateway) -> None:
    calls: Final = _burst_calls(12)
    with wire_server(_vendor) as anthropic_wire, wire_server(_vendor) as bedrock_wire, gateway.scenario() as scenario:
        models: Final[dict[Provider, str]] = {
            "anthropic": _deployment(scenario, "anthropic", anthropic_wire.url),
            "bedrock_invoke": _deployment(scenario, "bedrock_invoke", bedrock_wire.url),
        }
        served: Final = asyncio.run(_burst(str(gateway.client.base_url).rstrip("/"), gateway.key, models, calls))
        assert [item.status for item in served] == [200] * len(calls), [
            (item.status, item.text[:200]) for item in served
        ]
        received: Final = {"anthropic": anthropic_wire.drain(), "bedrock_invoke": bedrock_wire.drain()}
        assert {provider: len(posts) for provider, posts in received.items()} == {"anthropic": 6, "bedrock_invoke": 6}
        for item in served:
            assert f"marker-{item.call.marker}" in item.text, item.text
            request, body = _marked(received[item.call.provider], item.call.marker)
            _assert_wire(item.call, request, body)
