"""Bedrock Nova Sonic realtime tool calls through a scripted h2 event-stream peer (LIT-9465).

Nova Sonic returns a tool call as a role-TOOL contentStart/toolUse/contentEnd triple with the
arguments as a JSON string in ``content`` and no ASSISTANT block before it, and emits one such
triple per call including the two-call turn Nova 2 Sonic produces per question. The cells here
serve those triples from ``_support/bedrock_realtime_peer.py`` (the transport the handler really
dials: ``AWSCRTHTTPClient`` -> TLS ALPN h2 -> invoke-with-bidirectional-stream) and assert both
sides: the OpenAI realtime events the websocket client receives, and the toolSpec / toolResult
documents the proxy forwarded to the peer.

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
from hashlib import sha256
from pathlib import Path
from typing import Final

import pytest
import websockets
from pydantic import JsonValue
from websockets.asyncio.client import ClientConnection

from tests.integration._support.bedrock_realtime_peer import (
    BedrockRealtimePeer,
    Turn,
    assistant_text,
    bedrock_realtime_peer,
    tool_use,
    usage,
    user_transcript,
)
from tests.integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from tests.integration._support.database import read_rows
from tests.integration._support.process import owned_proxy
from tests.integration._support.tls import write_self_signed_cert

pytestmark: Final = pytest.mark.timeout(180)

DEPLOYMENT: Final = "bedrock/amazon.nova-2-sonic-v1:0"
WEATHER_PARAMETERS: Final = {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}
NESTED_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": WEATHER_PARAMETERS,
    },
}
FLAT_TOOL: Final = {
    "type": "function",
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": WEATHER_PARAMETERS,
}
TOOL_RESULT_OUTPUT: Final = '{"temperature_f":72,"conditions":"sunny"}'
SPEND_SQL: Final = 'SELECT call_type FROM "LiteLLM_SpendLogs" WHERE api_key = %s'


@pytest.fixture(scope="module")
def cert(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return write_self_signed_cert(tmp_path_factory.mktemp("bedrock-sonic-cert"))


@pytest.fixture(scope="module")
def candidate(tmp_path_factory: pytest.TempPathFactory, cert: tuple[Path, Path]) -> Iterator[Gateway]:
    with gateway_from_environment() as base:
        with owned_proxy(base, tmp_path_factory.mktemp("bedrock-sonic"), {"SSL_CERT_FILE": str(cert[0])}) as proxy:
            yield proxy


def _deployment(gateway: Gateway, scenario: Scenario, peer: BedrockRealtimePeer) -> str:
    name: Final = f"bedrock-sonic-{uuid.uuid4().hex[:12]}"
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
                "content": [{"type": "input_text", "text": "What is the weather in New York right now?"}],
            },
        }
    )


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
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.done",
        ), events
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
        assert last_output == []
        results: Final = eventually(lambda: peer.tool_results(), lambda values: len(values) == 1, seconds=10)
        assert results == (("tooluse_A", TOOL_RESULT_OUTPUT),)
        rows: Final = eventually(
            lambda: read_rows(SPEND_SQL, (sha256(key.encode()).hexdigest(),)),
            lambda values: len(values) >= 1,
            seconds=70,
        )
        assert any(row["call_type"] == "_arealtime" for row in rows), rows


async def _two_call_session(gateway: Gateway, model: str, key: str) -> tuple[Event, ...]:
    async with _realtime(gateway, model, key) as socket:
        await _open_session(socket, NESTED_TOOL)
        events: Final[tuple[dict[str, JsonValue], ...]] = await _events_until(
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
        healthy_turn: Final = Turn(
            "user_input", user_transcript("ready") + tool_use("tooluse_H", "get_weather", '{"location":"Boston"}')
        )
        with bedrock_realtime_peer((healthy_turn,), cert) as second_peer:
            healthy_model: Final = _deployment(candidate, scenario, second_peer)
            healthy_key: Final = scenario.key()
            second: Final = asyncio.run(_one_call_session(candidate, healthy_model, healthy_key, NESTED_TOOL))
            (healthy_call,) = _function_calls(second)
            assert string_value(healthy_call["arguments"]) == '{"location":"Boston"}'
