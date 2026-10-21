import asyncio
import uuid
from collections.abc import Iterable, Mapping, Sequence
from itertools import chain
from typing import Final, Literal, TypeAlias, cast

import httpx
import pytest
from anthropic import Anthropic
from integration._support import prompt_cache_breakpoint as pcb
from integration._support import responses_vendor as rv
from integration._support import tool_search as ts
from integration._support.client import Gateway, Scenario, eventually, string_value
from integration._support.wire import Wire, wire_server
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolUnionParam
from pydantic import JsonValue

from tests.integration.caching.response_cache_case import await_entry, redis_store

pytestmark: Final = pytest.mark.timeout(120)

Client: TypeAlias = Literal["httpx", "openai_sync", "openai_async"]
_ANTHROPIC_TOOL: Final[dict[str, JsonValue]] = {
    "name": ts.WEATHER,
    "description": "Look up get_weather",
    "input_schema": ts.PARAMETERS,
}


def _base_url(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


def _sdk_body(body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {key: value for key, value in body.items() if key not in ("model", "messages", "tools", "stream")}


def _sync_sdk(gateway: Gateway, body: Mapping[str, JsonValue]) -> tuple[object, ...]:
    messages: Final = cast(Iterable[ChatCompletionMessageParam], body["messages"])
    tools: Final = cast(Iterable[ChatCompletionToolUnionParam], body["tools"])
    model: Final = string_value(body["model"])
    with OpenAI(api_key=gateway.key, base_url=f"{_base_url(gateway)}/v1", max_retries=0) as client:
        if body.get("stream") is True:
            chunks: Final = client.chat.completions.create(
                model=model, messages=messages, tools=tools, stream=True, extra_body=_sdk_body(body)
            )
            return tuple(chunk.model_dump() for chunk in chunks)
        completion: Final = client.chat.completions.create(
            model=model, messages=messages, tools=tools, stream=False, extra_body=_sdk_body(body)
        )
        return (completion.model_dump(),)


async def _async_sdk(gateway: Gateway, body: Mapping[str, JsonValue]) -> tuple[object, ...]:
    messages: Final = cast(Iterable[ChatCompletionMessageParam], body["messages"])
    tools: Final = cast(Iterable[ChatCompletionToolUnionParam], body["tools"])
    model: Final = string_value(body["model"])
    async with AsyncOpenAI(api_key=gateway.key, base_url=f"{_base_url(gateway)}/v1", max_retries=0) as client:
        if body.get("stream") is True:
            chunks: Final = await client.chat.completions.create(
                model=model, messages=messages, tools=tools, stream=True, extra_body=_sdk_body(body)
            )
            return tuple([chunk.model_dump() async for chunk in chunks])
        completion: Final = await client.chat.completions.create(
            model=model, messages=messages, tools=tools, stream=False, extra_body=_sdk_body(body)
        )
        return (completion.model_dump(),)


def _send(gateway: Gateway, client: Client, body: Mapping[str, JsonValue]) -> tuple[dict[str, JsonValue], ...]:
    match client:
        case "httpx":
            response: Final = gateway.request("POST", "/v1/chat/completions", dict(body))
            assert response.status_code == 200, response.text
            if body.get("stream") is True:
                return ts.chunks_of(response.text)
            return (rv.JSON_OBJECT.validate_json(response.text),)
        case "openai_sync":
            return tuple(rv.JSON_OBJECT.validate_python(item) for item in _sync_sdk(gateway, body))
        case "openai_async":
            return tuple(rv.JSON_OBJECT.validate_python(item) for item in asyncio.run(_async_sdk(gateway, body)))


def _assistant(served: Sequence[Mapping[str, JsonValue]], stream: bool) -> dict[str, JsonValue]:
    if stream:
        return ts.assistant_from_stream(served)
    choices: Final = rv.ITEMS.validate_python(served[0]["choices"])
    return rv.JSON_OBJECT.validate_python(choices[0]["message"])


def _sent_message(message: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    kept: Final = ("role", "content", "tool_calls", "provider_specific_fields", "reasoning_items")
    return {key: value for key, value in message.items() if key in kept and value is not None}


def _search_items(marker: str) -> list[JsonValue]:
    return [*ts.searched(marker, "a", ts.WEATHER)]


def _replay(marker: str) -> list[dict[str, JsonValue]]:
    return [*ts.searched(marker, "a", ts.WEATHER), ts.reasoning(marker, "b"), ts.replayed_call(marker)]


def _body(
    model: str, messages: Sequence[JsonValue], *, stream: bool = False, **extra: JsonValue
) -> dict[str, JsonValue]:
    return {
        "model": model,
        "messages": list(messages),
        "tools": ts.chat_tools(ts.WEATHER),
        "stream": stream,
        **pcb.NO_CACHE,
        **extra,
    }


def _user(marker: str, *words: str) -> dict[str, JsonValue]:
    return {"role": "user", "content": " ".join((pcb.prompt(marker), *words))}


def _two_turns(
    gateway: Gateway, wire: Wire, model: str, client: Client, stream: bool, marker: str
) -> tuple[dict[str, JsonValue], list[dict[str, JsonValue]], tuple[str, str]]:
    user: Final = _user(marker)
    first: Final = _send(gateway, client, _body(model, [user], stream=stream))
    assistant: Final = _assistant(first, stream)
    pcb.posted(wire, marker)
    history: Final = [user, _sent_message(assistant), ts.tool_result(marker)]
    second: Final = _send(gateway, client, _body(model, history, stream=stream))
    answered: Final = _assistant(second, stream)
    assert answered["content"] == rv.answer(marker), answered
    identities: Final = (string_value(first[0]["id"]), string_value(second[0]["id"]))
    return assistant, ts.replayed(pcb.posted(wire, marker)), identities


def _assert_turn_one(assistant: Mapping[str, JsonValue], marker: str) -> None:
    assert ts.provider_fields(assistant).get("tool_search_items") == _search_items(marker), assistant
    assert [call["id"] for call in ts.tool_calls(assistant)] == [ts.call_id(marker)], assistant
    assert ts.namespaces(assistant) == [ts.WEATHER], assistant


def _assert_spend(model: str, response_ids: Sequence[str]) -> None:
    rows: Final = eventually(
        lambda: tuple(ts.spend_rows_of(model, response_id) for response_id in response_ids),
        lambda found: all(found),
        seconds=70,
    )
    assert [[row["status"] for row in found] for found in rows] == [["success"]] * len(response_ids), rows


@pytest.mark.parametrize("client", ["httpx", "openai_sync", "openai_async"])
def test_chat_bridge_returns_and_replays_tool_search_items_non_streaming(gateway: Gateway, client: Client) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(ts.scripted) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=pcb.MODEL, api_base=f"{wire.url}/v1")
        assistant, replayed, response_ids = _two_turns(gateway, wire, model, client, False, marker)
        _assert_turn_one(assistant, marker)
        assert replayed == _replay(marker)
        _assert_spend(model, response_ids)


@pytest.mark.parametrize("client", ["httpx", "openai_async"])
def test_chat_bridge_streams_tool_search_items_and_replays_the_rebuilt_message(
    gateway: Gateway, client: Client
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(ts.scripted) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=pcb.MODEL, api_base=f"{wire.url}/v1")
        assistant, replayed, response_ids = _two_turns(gateway, wire, model, client, True, marker)
        _assert_turn_one(assistant, marker)
        assert replayed == _replay(marker)
        _assert_spend(model, response_ids)


def _history(marker: str, **assistant_fields: JsonValue) -> list[JsonValue]:
    call: Final[dict[str, JsonValue]] = {
        "id": ts.call_id(marker),
        "type": "function",
        "function": {"name": ts.WEATHER, "arguments": '{"city": "Paris"}'},
    }
    return [
        _user(marker),
        {"role": "assistant", "content": None, "tool_calls": [call], **assistant_fields},
        ts.tool_result(marker),
    ]


def _replayed_history(marker: str) -> list[JsonValue]:
    return _history(
        marker,
        tool_calls=[
            {
                "id": ts.call_id(marker),
                "type": "function",
                "function": {"name": ts.WEATHER, "arguments": '{"city": "Paris"}'},
                "provider_specific_fields": {"namespace": ts.WEATHER},
            }
        ],
        provider_specific_fields={"tool_search_items": _search_items(marker)},
        reasoning_items=[ts.reasoning(marker, "a"), ts.reasoning(marker, "b")],
    )


def _replayed_upstream(
    gateway: Gateway,
    scenario: Scenario,
    wire: Wire,
    marker: str,
    body: Mapping[str, JsonValue],
    **model_params: JsonValue,
) -> list[dict[str, JsonValue]]:
    model: Final = scenario.model(model=pcb.MODEL, api_base=f"{wire.url}/v1", **model_params)
    response: Final = gateway.request("POST", "/v1/chat/completions", {**body, "model": model})
    assert response.status_code == 200, response.text
    assert ts.first_message(response.text)["content"] == rv.answer(marker), response.text
    return ts.replayed(pcb.posted(wire, marker))


def test_turn_without_tool_search_strips_search_items_but_keeps_namespace_and_reasoning(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(ts.scripted) as wire, gateway.scenario() as scenario:
        body: Final = {
            **_body("", _replayed_history(marker)),
            "tools": [{"type": "function", "function": ts.function(ts.WEATHER)}],
        }
        assert _replayed_upstream(gateway, scenario, wire, marker, body) == [
            ts.reasoning(marker, "a"),
            ts.reasoning(marker, "b"),
            ts.replayed_call(marker),
        ]


def test_tool_search_only_in_extra_body_tools_still_replays(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(ts.scripted) as wire, gateway.scenario() as scenario:
        body: Final = {
            **_body("", _replayed_history(marker)),
            "tools": [ts.deferred(ts.WEATHER)],
            "extra_body": {"tools": [ts.TOOL_SEARCH, ts.loaded(ts.WEATHER)]},
        }
        assert _replayed_upstream(gateway, scenario, wire, marker, body) == _replay(marker)


def test_extra_body_tools_without_tool_search_win_over_tools_and_skip_the_replay(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(ts.scripted) as wire, gateway.scenario() as scenario:
        body: Final = {
            **_body("", _replayed_history(marker)),
            "extra_body": {"tools": [{"type": "function", **ts.function(ts.WEATHER)}]},
        }
        assert _replayed_upstream(gateway, scenario, wire, marker, body) == [
            ts.reasoning(marker, "a"),
            ts.reasoning(marker, "b"),
            ts.replayed_call(marker),
        ]


def test_deployment_tool_search_replays_items_the_request_did_not_ask_for(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(ts.scripted) as wire, gateway.scenario() as scenario:
        body: Final = {**_body("", _replayed_history(marker)), "tools": [ts.deferred(ts.WEATHER)]}
        replayed: Final = _replayed_upstream(gateway, scenario, wire, marker, body, tools=[ts.TOOL_SEARCH])
        assert replayed == _replay(marker)


_MALFORMED_ITEMS: Final[dict[str, JsonValue]] = {
    "int": 7,
    "string": "tool_search_call",
    "empty-object": {},
    "empty-list": [],
    "null": None,
    "junk-list": [1, "x", None, {"type": "message", "content": "hi"}, {"no": "type"}],
    "5kb-string": "x" * 5120,
}


@pytest.mark.parametrize("items", list(_MALFORMED_ITEMS.values()), ids=list(_MALFORMED_ITEMS))
def test_malformed_tool_search_items_are_ignored_without_a_server_error(gateway: Gateway, items: JsonValue) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(ts.scripted) as wire, gateway.scenario() as scenario:
        body: Final = _body("", _history(marker, provider_specific_fields={"tool_search_items": items}))
        assert _replayed_upstream(gateway, scenario, wire, marker, body) == [
            {k: v for k, v in ts.replayed_call(marker).items() if k != "namespace"}
        ]


def test_search_output_with_non_list_tools_and_items_repeated_on_two_turns_replay_verbatim(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    odd_output: Final = {**ts.search_output(marker, []), "tools": "not-a-list"}
    items: Final[list[JsonValue]] = [ts.search_call(marker), odd_output]
    first_call: Final = {
        "id": ts.call_id(marker),
        "type": "function",
        "function": {"name": ts.WEATHER, "arguments": "{}"},
    }
    messages: Final[list[JsonValue]] = [
        _user(marker),
        {"role": "assistant", "content": "one", "provider_specific_fields": {"tool_search_items": items}},
        {"role": "user", "content": "again"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [first_call],
            "provider_specific_fields": {"tool_search_items": items},
        },
        ts.tool_result(marker),
    ]
    with wire_server(ts.scripted) as wire, gateway.scenario() as scenario:
        replayed: Final = _replayed_upstream(gateway, scenario, wire, marker, _body("", messages))
        assert replayed == [
            ts.search_call(marker),
            odd_output,
            ts.search_call(marker),
            odd_output,
            {"type": "function_call", "call_id": ts.call_id(marker), "name": ts.WEATHER, "arguments": "{}"},
        ]


_NAMESPACES: Final[dict[str, JsonValue]] = {"int": 3, "list": ["get_weather"], "empty": "", "5kb": "n" * 5120}


@pytest.mark.parametrize("value", list(_NAMESPACES.values()), ids=list(_NAMESPACES))
def test_malformed_namespace_is_dropped_and_a_long_one_is_forwarded(gateway: Gateway, value: JsonValue) -> None:
    marker: Final = uuid.uuid4().hex
    call: Final[dict[str, JsonValue]] = {
        "id": ts.call_id(marker),
        "type": "function",
        "function": {"name": ts.WEATHER, "arguments": '{"city": "Paris"}'},
        "provider_specific_fields": {"namespace": value},
    }
    with wire_server(ts.scripted) as wire, gateway.scenario() as scenario:
        body: Final = _body("", _history(marker, tool_calls=[call]))
        bare: Final = {k: v for k, v in ts.replayed_call(marker).items() if k != "namespace"}
        expected: Final = {**bare, "namespace": value} if isinstance(value, str) and value else bare
        assert _replayed_upstream(gateway, scenario, wire, marker, body) == [expected]


def test_upstream_rejecting_the_replay_reaches_the_caller(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(ts.scripted) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=pcb.MODEL, api_base=f"{wire.url}/v1")
        messages: Final = [_user(marker, ts.REJECT_REPLAY), *_replayed_history(marker)[1:]]
        response: Final = gateway.request("POST", "/v1/chat/completions", {**_body(model, messages)})
        assert response.status_code == 400, response.text
        assert ts.REJECTED in response.text, response.text
        assert ts.replayed(pcb.posted(wire, marker)) == _replay(marker)
        healthy: Final = gateway.request("POST", "/v1/chat/completions", _body(model, [_user(uuid.uuid4().hex)]))
        assert healthy.status_code == 200, healthy.text


@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
def test_turn_that_only_searched_returns_the_items_and_continues_the_search(gateway: Gateway, stream: bool) -> None:
    marker: Final = uuid.uuid4().hex
    user: Final = _user(marker, ts.ONLY_SEARCH)
    with wire_server(ts.scripted) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=pcb.MODEL, api_base=f"{wire.url}/v1")
        first: Final = _send(gateway, "httpx", _body(model, [user], stream=stream))
        assistant: Final = _assistant(first, stream)
        choices: Final = chain.from_iterable(rv.ITEMS.validate_python(chunk["choices"]) for chunk in first)
        finish: Final = [choice["finish_reason"] for choice in choices if choice.get("finish_reason")]
        assert finish == ["stop"], first
        assert assistant.get("content") in (None, ""), assistant
        assert ts.provider_fields(assistant).get("tool_search_items") == _search_items(marker), assistant
        assert "tool_calls" not in assistant or not assistant["tool_calls"], assistant
        pcb.posted(wire, marker)
        second: Final = _send(gateway, "httpx", _body(model, [user, _sent_message(assistant)], stream=stream))
        assert _assistant(second, stream)["content"] == rv.answer(marker), second
        assert ts.replayed(pcb.posted(wire, marker)) == _search_items(marker)


def test_two_searches_in_two_turns_replay_in_order(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    user: Final = _user(marker, ts.TWO_SEARCHES)
    with wire_server(ts.scripted) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=pcb.MODEL, api_base=f"{wire.url}/v1")
        tools: Final = ts.chat_tools(ts.WEATHER, ts.TIME)
        first: Final = _assistant(_send(gateway, "httpx", {**_body(model, [user]), "tools": tools}), False)
        pcb.posted(wire, marker)
        history: Final[list[JsonValue]] = [user, _sent_message(first), ts.tool_result(marker)]
        second: Final = _assistant(_send(gateway, "httpx", {**_body(model, history), "tools": tools}), False)
        assert ts.provider_fields(second).get("tool_search_items") == [*ts.searched(marker, "c", ts.TIME)], second
        assert ts.namespaces(second) == [ts.TIME], second
        pcb.posted(wire, marker)
        final: Final = [*history, _sent_message(second), ts.tool_result(marker, ts.TIME)]
        third: Final = _assistant(_send(gateway, "httpx", {**_body(model, final), "tools": tools}), False)
        assert third["content"] == rv.answer(marker), third
        assert ts.replayed(pcb.posted(wire, marker)) == [
            *_replay(marker),
            *ts.searched(marker, "c", ts.TIME),
            ts.replayed_call(marker, ts.TIME),
        ]


def _cached_twin(
    gateway: Gateway, body: Mapping[str, JsonValue], stream: bool
) -> tuple[httpx.Response, httpx.Response]:
    first: Final = gateway.request("POST", "/v1/chat/completions", dict(body))
    assert first.status_code == 200, first.text
    first_frame: Final = ts.chunks_of(first.text)[0] if stream else rv.JSON_OBJECT.validate_json(first.text)
    first_id: Final = string_value(first_frame["id"])
    await_entry(redis_store(), first_id)
    second: Final = gateway.request("POST", "/v1/chat/completions", dict(body))
    assert second.status_code == 200, second.text
    assert second.headers.get("x-litellm-cache-key"), dict(second.headers)
    return first, second


@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
def test_cache_hit_turn_carries_the_items_and_replays_them(gateway: Gateway, stream: bool) -> None:
    marker: Final = uuid.uuid4().hex
    user: Final = _user(marker)
    with wire_server(ts.scripted) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=pcb.MODEL, api_base=f"{wire.url}/v1")
        cached_body: Final = {
            key: value for key, value in _body(model, [user], stream=stream).items() if key != "cache"
        }
        first, hit = _cached_twin(gateway, cached_body, stream)
        pcb.posted(wire, marker)
        served: Final = ts.chunks_of(hit.text) if stream else (rv.JSON_OBJECT.validate_json(hit.text),)
        assistant: Final = _assistant(served, stream)
        _assert_turn_one(assistant, marker)
        second: Final = _send(gateway, "httpx", _body(model, [user, _sent_message(assistant), ts.tool_result(marker)]))
        assert _assistant(second, False)["content"] == rv.answer(marker), second
        assert ts.replayed(pcb.posted(wire, marker)) == _replay(marker)
        first_frames: Final = ts.chunks_of(first.text) if stream else (rv.JSON_OBJECT.validate_json(first.text),)
        assert served[0]["id"] == first_frames[0]["id"]


def test_messages_endpoint_sees_the_same_tool_use_on_both_sides(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(ts.scripted) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=pcb.MODEL, api_base=f"{wire.url}/v1")
        with Anthropic(api_key=gateway.key, base_url=_base_url(gateway), max_retries=0) as client:
            message: Final = client.messages.create(
                model=model,
                max_tokens=64,
                messages=[{"role": "user", "content": pcb.prompt(marker)}],
                tools=[_ANTHROPIC_TOOL],  # pyright: ignore[reportArgumentType]  # JSON tool definition
            )
            pcb.posted(wire, marker)
            uses: Final = [block for block in message.content if block.type == "tool_use"]
            assert [(block.name, block.input) for block in uses] == [(ts.WEATHER, {"city": "Paris"})], message
            assert "tool_search" not in message.model_dump_json(), message.model_dump_json()
            with client.messages.stream(
                model=model,
                max_tokens=64,
                messages=[{"role": "user", "content": pcb.prompt(marker)}],
                tools=[_ANTHROPIC_TOOL],  # pyright: ignore[reportArgumentType]  # JSON tool definition
            ) as stream:
                streamed: Final = stream.get_final_message()
            pcb.posted(wire, marker)
            streamed_uses: Final = [block for block in streamed.content if block.type == "tool_use"]
            assert [(block.name, block.input) for block in streamed_uses] == [(ts.WEATHER, {"city": "Paris"})]
            assert "tool_search" not in streamed.model_dump_json(), streamed.model_dump_json()


def test_messages_endpoint_turn_that_only_searched_is_answered(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(ts.scripted) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=pcb.MODEL, api_base=f"{wire.url}/v1")
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": f"{pcb.prompt(marker)} {ts.ONLY_SEARCH}"}],
                "tools": [_ANTHROPIC_TOOL],
            },
        )
        assert response.status_code == 200, response.text
        body: Final = rv.JSON_OBJECT.validate_json(response.text)
        assert body["stop_reason"] == "end_turn", body
        assert [block for block in rv.ITEMS.validate_python(body["content"]) if block.get("type") == "tool_use"] == []
        pcb.posted(wire, marker)


def test_native_responses_forwards_tool_search_items_and_namespace_verbatim(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    history: Final[list[JsonValue]] = [
        {"type": "message", "role": "user", "content": pcb.prompt(marker)},
        *ts.searched(marker, "a", ts.WEATHER),
        ts.replayed_call(marker),
        {"type": "function_call_output", "call_id": ts.call_id(marker), "output": "sunny"},
    ]
    with wire_server(ts.scripted) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=pcb.MODEL, api_base=f"{wire.url}/v1")
        response: Final = gateway.request(
            "POST",
            "/v1/responses",
            {"model": model, "input": history, "tools": [ts.TOOL_SEARCH, ts.loaded(ts.WEATHER)], **pcb.NO_CACHE},
        )
        assert response.status_code == 200, response.text
        assert rv.answer(marker) in response.text, response.text
        assert pcb.input_items(pcb.posted(wire, marker)) == history
