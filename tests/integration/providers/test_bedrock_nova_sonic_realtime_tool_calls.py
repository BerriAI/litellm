"""Bedrock Nova Sonic realtime tool calls through a scripted h2 event-stream peer (LIT-9465).

Nova Sonic returns a tool call as a role-TOOL contentStart/toolUse/contentEnd triple with the
arguments as a JSON string in ``content`` and no ASSISTANT block before it, and emits one such
triple per call including the two-call turn Nova 2 Sonic produces per question. The cells here
serve those triples from ``_support/bedrock_realtime_peer.py`` (the transport the handler really
dials: ``AWSCRTHTTPClient`` -> TLS ALPN h2 -> invoke-with-bidirectional-stream) and assert both
sides: the OpenAI realtime events the websocket client receives, and the toolSpec / toolResult
documents the proxy forwarded to the peer.

The fallback cells cover a websocket that outlives one provider attempt. A refused stream keeps the
client's opening message for the fallback, which gets it once whether it dials Bedrock again or
an OpenAI realtime deployment, and once Bedrock accepted a stream no later attempt may open a second
provider session on that websocket.

The file boots its own proxy instead of the shared gateway so it can pass ``SSL_CERT_FILE``
pointing at the peer's self-signed cert, which is the only way the CRT h2 client trusts it.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import pytest
import websockets
import yaml
from integration._support.bedrock_realtime_peer import (
    BedrockEvent,
    BedrockRealtimePeer,
    OpeningEnd,
    Refusal,
    StreamFailure,
    Turn,
    assistant_text,
    bedrock_realtime_peer,
    tool_use,
    usage,
    user_transcript,
)
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.tls import write_self_signed_cert
from integration._support.upstream import delete_scenario, register_scenario
from integration.cost_calculation.cost_tracking_case import RealtimeResponse
from pydantic import JsonValue
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

DEPLOYMENT: Final = "bedrock/amazon.nova-2-sonic-v1:0"
WEATHER_PARAMETERS: Final = {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}
WEATHER_FUNCTION: Final = {
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": WEATHER_PARAMETERS,
}
NESTED_TOOL: Final = {"type": "function", "function": WEATHER_FUNCTION}
FLAT_TOOL: Final = {"type": "function", **WEATHER_FUNCTION}
TOOL_RESULT_OUTPUT: Final = '{"temperature_f":72,"conditions":"sunny"}'
QUESTION: Final = "What is the weather in New York right now?"
ANSWER: Final = "It is 72 and sunny"
INSTRUCTIONS: Final = "Answer every weather question in one short sentence."
OPENAI_VOICE_MODEL: Final = "gpt-realtime-2.1"
BEDROCK_REPLAY_GROUPS: Final = MappingProxyType(
    {
        "system_prompt": ("sonic-refused-session-update-first", "sonic-replay-session-update-first"),
        "user_input": ("sonic-refused-text-first", "sonic-replay-text-first"),
    }
)
OPENAI_REPLAY_GROUPS: Final = ("sonic-refused-openai-fallback", "openai-replay-target")
OPENAI_COMMITTED_GROUPS: Final = ("sonic-committed-openai-fallback", "openai-committed-target")
FALLBACK_GROUPS: Final = (*BEDROCK_REPLAY_GROUPS.values(), OPENAI_REPLAY_GROUPS, OPENAI_COMMITTED_GROUPS)
SCRIPTED_DONE: Final[dict[str, JsonValue]] = {
    "type": "response.done",
    "event_id": "evt_$UNIQUE_ID",
    "response": {
        "id": "resp_$UNIQUE_ID",
        "object": "realtime.response",
        "status": "completed",
        "output": [],
        "usage": {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
    },
}


@pytest.fixture(scope="module")
def cert(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return write_self_signed_cert(tmp_path_factory.mktemp("bedrock-sonic-cert"))


def _fallback_config(directory: Path) -> Path:
    stock: Final = JSON_OBJECT.validate_python(
        yaml.safe_load((Path(__file__).resolve().parents[1] / "proxy_config.yaml").read_text())
    )
    config: Final = {
        **stock,
        "router_settings": {
            **object_value(stock["router_settings"]),
            "fallbacks": [{source: [target]} for source, target in FALLBACK_GROUPS],
        },
    }
    path: Final = directory / "bedrock-sonic-fallbacks.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture(scope="module")
def candidate(tmp_path_factory: pytest.TempPathFactory, cert: tuple[Path, Path]) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("bedrock-sonic")
    with gateway_from_environment() as base:
        with owned_proxy(base, directory, {"SSL_CERT_FILE": str(cert[0])}, config=_fallback_config(directory)) as proxy:
            yield proxy


def _deployment(gateway: Gateway, scenario: Scenario, peer: BedrockRealtimePeer) -> str:
    return _bedrock_deployment(gateway, scenario, peer, f"bedrock-sonic-{uuid.uuid4().hex[:12]}")


def _bedrock_deployment(gateway: Gateway, scenario: Scenario, peer: BedrockRealtimePeer, name: str) -> str:
    created: Final = gateway.post(
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {
                "model": DEPLOYMENT,
                "aws_region_name": "us-east-1",
                "aws_access_key_id": "integration-dummy-key",
                "aws_secret_access_key": "integration-dummy-secret",
                "api_base": peer.api_base,
            },
            "model_info": {"mode": "realtime"},
        },
    )
    scenario.cleanups.callback(scenario.delete_model, string_value(object_value(created["model_info"])["id"]))
    return name


def _openai_deployment(gateway: Gateway, scenario: Scenario, name: str) -> str:
    scenario_id: Final = f"bedrock-sonic-fallback-{uuid.uuid4().hex[:12]}"
    handle: Final = register_scenario(
        scenario_id, RealtimeResponse(content_type="application/x-realtime", events=(SCRIPTED_DONE,))
    )
    scenario.cleanups.callback(delete_scenario, handle)
    created: Final = gateway.post(
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {
                "model": f"openai/{OPENAI_VOICE_MODEL}",
                "api_key": scenario_id,
                "api_base": f"{gateway.upstream_url}/v1",
            },
            "model_info": {"mode": "realtime"},
        },
    )
    scenario.cleanups.callback(scenario.delete_model, string_value(object_value(created["model_info"])["id"]))
    return scenario_id


def _upstream_requests(gateway: Gateway, scenario_id: str) -> tuple[dict[str, JsonValue], ...]:
    with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream:
        observations: Final = JSON_OBJECT.validate_json(upstream.get("/__observations").content)
    requests: Final = observations["requests"]
    assert isinstance(requests, list), observations
    return tuple(
        request
        for request in map(object_value, requests)
        if request["authorization"] == f"Bearer {scenario_id}" or request["api_key"] == scenario_id
    )


def _upstream_frames(requests: tuple[dict[str, JsonValue], ...]) -> tuple[dict[str, JsonValue], ...]:
    return tuple(object_value(request["body"]) for request in requests if request["method"] == "WEBSOCKET_FRAME")


def _session_update(tool: Mapping[str, JsonValue]) -> str:
    return json.dumps(
        {
            "type": "session.update",
            "session": {
                "instructions": "Always use the get_weather tool for weather questions.",
                "tools": [tool],
            },
        }
    )


def _user_text() -> str:
    return json.dumps(
        {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": QUESTION}],
            },
        }
    )


def _instructions_update() -> str:
    return json.dumps({"type": "session.update", "session": {"instructions": INSTRUCTIONS}})


_RESPONSE_CREATE: Final = json.dumps({"type": "response.create"})


def _tool_result(call_id: str) -> str:
    return json.dumps(
        {
            "type": "conversation.item.create",
            "item": {"type": "function_call_output", "call_id": call_id, "output": TOOL_RESULT_OUTPUT},
        }
    )


Event = dict[str, JsonValue]
EventPredicate = Callable[[Event, tuple[Event, ...]], bool]


async def _events_until(
    socket: ClientConnection, predicate: EventPredicate, *, timeout: float = 60
) -> tuple[Event, ...]:
    collected: Final[list[Event]] = []
    deadline: Final = time.monotonic() + timeout
    while True:
        try:
            raw: Final = await asyncio.wait_for(socket.recv(), deadline - time.monotonic())
        except (TimeoutError, asyncio.TimeoutError):
            raise AssertionError(f"timed out waiting; types={[event.get('type') for event in collected]}")
        event: Final = JSON_OBJECT.validate_json(raw)
        collected.append(event)
        if predicate(event, tuple(collected)):
            return tuple(collected)


def _is_done(events: tuple[Event, ...]) -> bool:
    return any(event.get("type") == "response.done" for event in events)


def _function_calls(events: tuple[Event, ...]) -> tuple[Event, ...]:
    return tuple(event for event in events if event.get("type") == "response.function_call_arguments.done")


def _response_dones(events: tuple[Event, ...]) -> tuple[Event, ...]:
    return tuple(event for event in events if event.get("type") == "response.done")


def _done_ids(events: tuple[Event, ...]) -> tuple[str, ...]:
    return tuple(string_value(object_value(event["response"])["id"]) for event in _response_dones(events))


def _tail_types(events: tuple[Event, ...]) -> tuple[str, ...]:
    """Event types from the first response.created onward."""
    start: Final = next(index for index, event in enumerate(events) if event.get("type") == "response.created")
    return tuple(string_value(event["type"]) for event in events[start:])


@asynccontextmanager
async def _realtime(gateway: Gateway, model: str, key: str) -> AsyncIterator[ClientConnection]:
    proxy: Final = str(gateway.client.base_url).rstrip("/").replace("http://", "ws://")
    async with websockets.connect(
        f"{proxy}/v1/realtime?model={model}", additional_headers={"Authorization": f"Bearer {key}"}
    ) as socket:
        yield socket


async def _open_session(socket: ClientConnection, tool: Mapping[str, JsonValue]) -> None:
    created: Final = JSON_OBJECT.validate_json(await socket.recv())
    assert created.get("type") == "session.created", created
    await socket.send(_session_update(tool))
    await socket.send(_user_text())
    await socket.send(_RESPONSE_CREATE)


async def _one_call_session(gateway: Gateway, model: str, key: str, tool: Mapping[str, JsonValue]) -> tuple[Event, ...]:
    async with _realtime(gateway, model, key) as socket:
        await _open_session(socket, tool)
        return await _events_until(socket, lambda _event, events: _is_done(events))


def test_nested_tool_call_before_assistant_block(candidate: Gateway, cert: tuple[Path, Path]) -> None:
    turn: Final = Turn(
        "user_input", user_transcript("ready") + tool_use("tooluse_A", "get_weather", '{"location":"New York"}')
    )
    with bedrock_realtime_peer((turn,), cert) as peer, candidate.scenario() as scenario:
        model: Final = _deployment(candidate, scenario, peer)
        key: Final = scenario.key()
        events: Final = asyncio.run(_one_call_session(candidate, model, key, NESTED_TOOL))
        assert _tail_types(events) == (
            "response.created",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.done",
        ), events
        (delta,) = tuple(event for event in events if event.get("type") == "response.function_call_arguments.delta")
        assert string_value(delta["delta"]) == '{"location":"New York"}'
        (call,) = _function_calls(events)
        assert string_value(call["call_id"]) == "tooluse_A"
        assert string_value(call["name"]) == "get_weather"
        assert string_value(call["arguments"]) == '{"location":"New York"}'
        (done,) = _response_dones(events)
        output: Final = object_value(done["response"])["output"]
        assert isinstance(output, list) and len(output) == 1, done
        item: Final = object_value(output[0])
        assert string_value(item["type"]) == "function_call"
        assert string_value(item["call_id"]) == "tooluse_A"
        assert string_value(item["arguments"]) == '{"location":"New York"}'
        specs: Final = eventually(lambda: peer.tool_specs(), lambda values: len(values) == 1, seconds=10)
        assert string_value(specs[0]["name"]) == "get_weather"
        assert json.loads(string_value(object_value(specs[0]["inputSchema"])["json"])) == WEATHER_PARAMETERS


def test_flat_tool_shape(candidate: Gateway, cert: tuple[Path, Path]) -> None:
    turn: Final = Turn(
        "user_input", user_transcript("ready") + tool_use("tooluse_B", "get_weather", '{"location":"New York"}')
    )
    with bedrock_realtime_peer((turn,), cert) as peer, candidate.scenario() as scenario:
        model: Final = _deployment(candidate, scenario, peer)
        key: Final = scenario.key()
        events: Final = asyncio.run(_one_call_session(candidate, model, key, FLAT_TOOL))
        (call,) = _function_calls(events)
        assert string_value(call["call_id"]) == "tooluse_B"
        assert string_value(call["name"]) == "get_weather"
        specs: Final = eventually(lambda: peer.tool_specs(), lambda values: len(values) == 1, seconds=10)
        assert string_value(specs[0]["name"]) == "get_weather"
        assert string_value(specs[0]["description"]) == "Get current weather for a city"
        assert json.loads(string_value(object_value(specs[0]["inputSchema"])["json"])) == WEATHER_PARAMETERS


async def _tool_result_round_trip(gateway: Gateway, model: str, key: str) -> tuple[Event, ...]:
    async with _realtime(gateway, model, key) as socket:
        await _open_session(socket, NESTED_TOOL)
        first: Final = await _events_until(socket, lambda _event, events: _is_done(events))
        (call,) = _function_calls(first)
        await socket.send(_tool_result(string_value(call["call_id"])))
        await socket.send(_RESPONSE_CREATE)
        second: Final = await _events_until(socket, lambda _event, events: _is_done(events))
        return first + second


def test_tool_result_round_trip(candidate: Gateway, cert: tuple[Path, Path]) -> None:
    turns: Final = (
        Turn("user_input", user_transcript("ready") + tool_use("tooluse_A", "get_weather", '{"location":"New York"}')),
        Turn("tool_result", assistant_text("It is 72 and sunny") + (usage(30, 12),)),
    )
    with bedrock_realtime_peer(turns, cert) as peer, candidate.scenario() as scenario:
        model: Final = _deployment(candidate, scenario, peer)
        key: Final = scenario.key()
        events: Final = asyncio.run(_tool_result_round_trip(candidate, model, key))
        dones: Final = _response_dones(events)
        assert len(dones) == 2, events
        first_id, second_id = _done_ids(events)
        assert first_id != second_id
        second_created: Final = tuple(event for event in events if event.get("type") == "response.created")
        assert len(second_created) == 2 and object_value(second_created[1]["response"])["id"] == second_id
        text: Final = "".join(
            string_value(event["delta"]) for event in events if event.get("type") == "response.text.delta"
        )
        assert "72" in text
        last_output: Final = object_value(dones[-1]["response"])["output"]
        assert tuple(object_value(item).get("type") for item in last_output) == ("message",)
        (answer_item,) = last_output
        assert object_value(answer_item)["content"] == [{"type": "text", "text": "It is 72 and sunny"}]
        results: Final = eventually(lambda: peer.tool_results(), lambda values: len(values) == 1, seconds=10)
        assert results == (("tooluse_A", TOOL_RESULT_OUTPUT),)


async def _two_call_session(gateway: Gateway, model: str, key: str) -> tuple[Event, ...]:
    async with _realtime(gateway, model, key) as socket:
        await _open_session(socket, NESTED_TOOL)
        events: Final[tuple[Event, ...]] = await _events_until(
            socket, lambda _event, collected: len(_response_dones(collected)) == 2
        )
        for call in _function_calls(events):
            await socket.send(_tool_result(string_value(call["call_id"])))
        after_answer: Final = await _events_until(socket, lambda _event, collected: _is_done(collected))
        return events + after_answer


def test_two_tool_calls_one_turn(candidate: Gateway, cert: tuple[Path, Path]) -> None:
    args: Final = '{"location":"New York"}'
    turns: Final = (
        Turn(
            "user_input",
            user_transcript("ready")
            + tool_use("tooluse_A", "get_weather", args)
            + tool_use("tooluse_B", "get_weather", args),
        ),
        Turn("tool_result", assistant_text("It is 72 and sunny") + (usage(40, 15),)),
    )
    with bedrock_realtime_peer(turns, cert) as peer, candidate.scenario() as scenario:
        model: Final = _deployment(candidate, scenario, peer)
        key: Final = scenario.key()
        events: Final = asyncio.run(_two_call_session(candidate, model, key))
        calls: Final = _function_calls(events)
        assert len(calls) == 2, events
        assert tuple(string_value(call["call_id"]) for call in calls) == ("tooluse_A", "tooluse_B")
        assert tuple(string_value(call["arguments"]) for call in calls) == (args, args)
        item_ids: Final = tuple(string_value(call["item_id"]) for call in calls)
        response_ids: Final = tuple(string_value(call["response_id"]) for call in calls)
        assert len(set(item_ids)) == 2 and len(set(response_ids)) == 2
        first_dones: Final = _response_dones(events)[:2]
        for done, call_id in zip(first_dones, ("tooluse_A", "tooluse_B")):
            output: Final = object_value(done["response"])["output"]
            assert isinstance(output, list) and len(output) == 1, done
            assert string_value(object_value(output[0])["call_id"]) == call_id
        results: Final = eventually(lambda: peer.tool_results(), lambda values: len(values) == 2, seconds=10)
        assert tuple(result[0] for result in results) == ("tooluse_A", "tooluse_B")
        assert all(result[1] == TOOL_RESULT_OUTPUT for result in results)
        text: Final = "".join(
            string_value(event["delta"]) for event in events if event.get("type") == "response.text.delta"
        )
        assert "72" in text


def test_tool_call_inside_active_assistant_response(candidate: Gateway, cert: tuple[Path, Path]) -> None:
    start, output, _end = assistant_text("It is 72 and sunny")
    turn: Final = Turn(
        "user_input",
        (start, output) + tool_use("tooluse_C", "get_weather", '{"location":"New York"}') + (_end,),
    )
    with bedrock_realtime_peer((turn,), cert) as peer, candidate.scenario() as scenario:
        model: Final = _deployment(candidate, scenario, peer)
        key: Final = scenario.key()
        events: Final = asyncio.run(_one_call_session(candidate, model, key, NESTED_TOOL))
        created: Final = tuple(event for event in events if event.get("type") == "response.created")
        assert len(created) == 1, events
        response_id: Final = string_value(object_value(created[0]["response"])["id"])
        (call,) = _function_calls(events)
        assert string_value(call["call_id"]) == "tooluse_C"
        assert string_value(call["response_id"]) == response_id
        assert string_value(call["arguments"]) == '{"location":"New York"}'
        assert call["output_index"] == 1, events
        assert tuple(string_value(e["text"]) for e in events if e.get("type") == "response.text.done") == (
            "It is 72 and sunny",
        ), events
        message_done_indexes: Final = tuple(
            i
            for i, event in enumerate(events)
            if event.get("type") == "response.output_item.done" and object_value(event["item"]).get("type") == "message"
        )
        response_done_indexes: Final = tuple(
            i for i, event in enumerate(events) if event.get("type") == "response.done"
        )
        assert len(response_done_indexes) == 1, events
        assert len(message_done_indexes) == 1 and response_done_indexes[0] > message_done_indexes[0], events


@pytest.mark.parametrize(
    "content",
    ["", None, "not json", '{"location":"' + "x" * 5000 + '"}'],
    ids=["empty", "missing", "not-json", "five-kb"],
)
def test_malformed_tool_content(candidate: Gateway, cert: tuple[Path, Path], content: str | None) -> None:
    turn: Final = Turn("user_input", tool_use("tooluse_M", "get_weather", content))
    with bedrock_realtime_peer((turn,), cert) as peer, candidate.scenario() as scenario:
        model: Final = _deployment(candidate, scenario, peer)
        key: Final = scenario.key()
        events: Final = asyncio.run(_one_call_session(candidate, model, key, NESTED_TOOL))
        (call,) = _function_calls(events)
        assert string_value(call["arguments"]) == (content if content is not None else "{}"), events


def _of_type(events: tuple[Event, ...], event_type: str) -> tuple[Event, ...]:
    return tuple(event for event in events if event.get("type") == event_type)


def _bedrock_fields(events: tuple[BedrockEvent, ...], name: str) -> tuple[dict[str, JsonValue], ...]:
    bodies: Final = tuple(object_value(event["event"]) for event in events)
    return tuple(object_value(body[name]) for body in bodies if name in body)


def _text_inputs(events: tuple[BedrockEvent, ...], role: str) -> tuple[str, ...]:
    roles: Final = MappingProxyType(
        {string_value(start["contentName"]): start.get("role") for start in _bedrock_fields(events, "contentStart")}
    )
    return tuple(
        string_value(text["content"])
        for text in _bedrock_fields(events, "textInput")
        if roles.get(string_value(text["contentName"])) == role
    )


def _answer_turn() -> Turn:
    return Turn("user_input", assistant_text(ANSWER) + (usage(30, 12),))


def _answer_text(events: tuple[Event, ...]) -> str:
    return "".join(string_value(event["delta"]) for event in _of_type(events, "response.text.delta"))


async def _opened_with(socket: ClientConnection, opening: tuple[str, ...], stop: frozenset[str]) -> tuple[Event, ...]:
    created: Final = JSON_OBJECT.validate_json(await socket.recv())
    for message in opening:
        await socket.send(message)
    return (created,) + await _events_until(socket, lambda event, _events: event.get("type") in stop)


async def _session(
    gateway: Gateway, model: str, key: str, opening: tuple[str, ...], stop: frozenset[str]
) -> tuple[Event, ...]:
    async with _realtime(gateway, model, key) as socket:
        return await _opened_with(socket, opening, stop)


async def _session_until_closed(
    gateway: Gateway, model: str, key: str, opening: tuple[str, ...]
) -> tuple[tuple[Event, ...], int | None]:
    async with _realtime(gateway, model, key) as socket:
        events: Final = await _opened_with(socket, opening, frozenset(("error", "session.created")))
        if events[-1].get("type") != "error":
            return events, None
        try:
            await asyncio.wait_for(socket.recv(), 30)
        except ConnectionClosed:
            return events, socket.close_code
        return events, None


@pytest.mark.parametrize("after", ["system_prompt", "user_input"], ids=["session-update-first", "text-first"])
def test_refused_opening_is_replayed_on_the_bedrock_fallback(
    candidate: Gateway, cert: tuple[Path, Path], after: OpeningEnd
) -> None:
    source, target = BEDROCK_REPLAY_GROUPS[after]
    update_first: Final = after == "system_prompt"
    opening: Final = ((_instructions_update(),) if update_first else ()) + (_user_text(), _RESPONSE_CREATE)
    opening_role, opening_text = ("SYSTEM", INSTRUCTIONS) if update_first else ("USER", QUESTION)
    with (
        bedrock_realtime_peer((), cert, refusal=Refusal(after)) as refusing,
        bedrock_realtime_peer((_answer_turn(),), cert) as healthy,
        candidate.scenario() as scenario,
    ):
        _bedrock_deployment(candidate, scenario, refusing, source)
        _bedrock_deployment(candidate, scenario, healthy, target)
        events: Final = asyncio.run(
            _session(
                candidate, source, scenario.key(), opening, frozenset(("response.done", "error", "session.created"))
            )
        )
        assert len(_of_type(events, "session.created")) == 1, events
        assert len(_of_type(events, "session.updated")) == int(update_first), events
        assert _of_type(events, "error") == (), events
        assert len(_response_dones(events)) == 1, events
        assert _answer_text(events) == ANSWER, events
        (replayed,) = healthy.streams()
        assert _text_inputs(replayed, "USER") == (QUESTION,), replayed
        assert _text_inputs(replayed, "SYSTEM").count(INSTRUCTIONS) == int(update_first), replayed
        refused: Final = refusing.streams()
        assert refused and all(opening_text in _text_inputs(stream, opening_role) for stream in refused), refused


def test_refused_text_opening_is_replayed_on_the_openai_fallback(candidate: Gateway, cert: tuple[Path, Path]) -> None:
    source, target = OPENAI_REPLAY_GROUPS
    with (
        bedrock_realtime_peer((), cert, refusal=Refusal("user_input")) as refusing,
        candidate.scenario() as scenario,
    ):
        _bedrock_deployment(candidate, scenario, refusing, source)
        scenario_id: Final = _openai_deployment(candidate, scenario, target)
        events: Final = asyncio.run(
            _session(
                candidate,
                source,
                scenario.key(),
                (_user_text(), _RESPONSE_CREATE),
                frozenset(("response.done", "error")),
            )
        )
        assert _of_type(events, "error") == (), events
        assert len(_response_dones(events)) == 1, events
        requests: Final = _upstream_requests(candidate, scenario_id)
        assert len(tuple(request for request in requests if request["method"] == "WEBSOCKET")) == 1, requests
        assert _upstream_frames(requests) == (json.loads(_user_text()), json.loads(_RESPONSE_CREATE)), requests


def test_committed_session_failure_closes_the_socket_without_another_provider_stream(
    candidate: Gateway, cert: tuple[Path, Path]
) -> None:
    source, target = OPENAI_COMMITTED_GROUPS
    with (
        bedrock_realtime_peer((_answer_turn(),), cert, failure=StreamFailure()) as committed,
        candidate.scenario() as scenario,
    ):
        _bedrock_deployment(candidate, scenario, committed, source)
        scenario_id: Final = _openai_deployment(candidate, scenario, target)
        events, close_code = asyncio.run(
            _session_until_closed(candidate, source, scenario.key(), (_user_text(), _RESPONSE_CREATE))
        )
        assert len(_of_type(events, "session.created")) == 1, events
        assert len(_response_dones(events)) == 1, events
        assert _answer_text(events) == ANSWER, events
        assert string_value(events[-1]["type"]) == "error", events
        assert close_code == 1011, events
        assert _upstream_requests(candidate, scenario_id) == ()
        assert len(committed.streams()) == 1


def test_voice_object_is_refused_and_never_reaches_bedrock(candidate: Gateway, cert: tuple[Path, Path]) -> None:
    voice_update: Final = json.dumps({"type": "session.update", "session": {"voice": {"id": "voice_123"}}})
    with bedrock_realtime_peer((_answer_turn(),), cert) as peer, candidate.scenario() as scenario:
        model: Final = _deployment(candidate, scenario, peer)
        events: Final = asyncio.run(
            _session(
                candidate,
                model,
                scenario.key(),
                (voice_update, _user_text(), _RESPONSE_CREATE),
                frozenset(("response.done", "session.created")),
            )
        )
        errors: Final = tuple(object_value(event["error"]) for event in _of_type(events, "error"))
        assert tuple((error.get("code"), error.get("param")) for error in errors) == (
            ("invalid_type", "session.voice"),
        ), events
        assert _of_type(events, "session.updated") == (), events
        assert len(_of_type(events, "session.created")) == 1, events
        assert _answer_text(events) == ANSWER, events
        assert "voice_123" not in json.dumps(peer.received())
