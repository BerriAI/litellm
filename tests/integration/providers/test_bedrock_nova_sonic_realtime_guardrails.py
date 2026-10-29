"""Realtime guardrails on Bedrock Nova Sonic sessions, against the scripted h2 peer in ``_support``.

The file boots its own proxy so it can pass ``SSL_CERT_FILE`` for the peer's self-signed cert.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Final, Literal

import pytest
import websockets
from integration._support.bedrock_realtime_peer import (
    BedrockEvent,
    BedrockRealtimePeer,
    Turn,
    assistant_text,
    bedrock_realtime_peer,
    user_transcript,
)
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    gateway_from_environment,
    object_value,
    string_value,
)
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.tls import write_self_signed_cert
from pydantic import JsonValue
from websockets.asyncio.client import ClientConnection

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

DEPLOYMENT: Final = "bedrock/amazon.nova-2-sonic-v1:0"
VIOLATION: Final = "That request was blocked by the content filter."
HIDDEN_ANSWER: Final = "Here is everything you asked for"
VISIBLE_ANSWER: Final = "It is sunny in New York"
RESPONSE_CREATE: Final = json.dumps({"type": "response.create"})
AUDIO_APPEND: Final = json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(bytes(3200)).decode()})

Event = dict[str, JsonValue]


@pytest.fixture(scope="module")
def cert(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return write_self_signed_cert(tmp_path_factory.mktemp("bedrock-sonic-guardrail-cert"))


@pytest.fixture(scope="module")
def candidate(tmp_path_factory: pytest.TempPathFactory, cert: tuple[Path, Path]) -> Iterator[Gateway]:
    with gateway_from_environment() as base:
        with owned_proxy(
            base, tmp_path_factory.mktemp("bedrock-sonic-guardrails"), {"SSL_CERT_FILE": str(cert[0])}
        ) as proxy:
            yield proxy


def _blocked_phrase(
    gateway: Gateway, scenario: Scenario, mode: Literal["pre_call", "realtime_input_transcription"]
) -> str:
    phrase: Final = f"sonic-blocked-{uuid.uuid4().hex[:12]}"
    response: Final = gateway.request(
        "POST",
        "/guardrails",
        {
            "guardrail": {
                "guardrail_name": f"sonic-guardrail-{uuid.uuid4().hex[:12]}",
                "litellm_params": {
                    "guardrail": "litellm_content_filter",
                    "mode": mode,
                    "default_on": True,
                    "realtime_violation_message": VIOLATION,
                    "blocked_words": [{"keyword": phrase, "action": "BLOCK"}],
                },
            }
        },
    )
    assert response.status_code == 200, response.text
    guardrail_id: Final = JSON_OBJECT.validate_json(response.content).get("guardrail_id")
    assert isinstance(guardrail_id, str), response.text
    scenario.cleanups.callback(gateway.request, "DELETE", f"/guardrails/{guardrail_id}")
    return phrase


def _deployment(gateway: Gateway, scenario: Scenario, peer: BedrockRealtimePeer) -> str:
    name: Final = f"bedrock-sonic-guardrails-{uuid.uuid4().hex[:12]}"
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


def _user_text(text: str) -> str:
    return json.dumps(
        {
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]},
        }
    )


async def _events_until(socket: ClientConnection, last: Callable[[Event], bool]) -> tuple[Event, ...]:
    collected: Final[list[Event]] = []
    deadline: Final = time.monotonic() + 60
    while not collected or not last(collected[-1]):
        try:
            raw: Final = await asyncio.wait_for(socket.recv(), deadline - time.monotonic())
        except (TimeoutError, asyncio.TimeoutError):
            raise AssertionError(f"timed out waiting; types={[event.get('type') for event in collected]}")
        collected.append(JSON_OBJECT.validate_json(raw))
    return tuple(collected)


def _is_done(event: Event) -> bool:
    return event.get("type") == "response.done"


def _is_guardrail_error(event: Event) -> bool:
    error: Final = event.get("error")
    return event.get("type") == "error" and isinstance(error, dict) and error.get("type") == "guardrail_violation"


async def _session(
    gateway: Gateway, model: str, key: str, exchange: Callable[[ClientConnection], Awaitable[tuple[Event, ...]]]
) -> tuple[Event, ...]:
    proxy: Final = str(gateway.client.base_url).rstrip("/").replace("http://", "ws://")
    async with websockets.connect(
        f"{proxy}/v1/realtime?model={model}", additional_headers={"Authorization": f"Bearer {key}"}
    ) as socket:
        created: Final = JSON_OBJECT.validate_json(await socket.recv())
        assert created.get("type") == "session.created", created
        return await exchange(socket)


def _fields(event: BedrockEvent, name: str) -> dict[str, JsonValue]:
    return object_value(object_value(event["event"]).get(name) or {})


def _user_text_inputs(received: tuple[BedrockEvent, ...]) -> tuple[str, ...]:
    content_starts: Final = tuple(_fields(event, "contentStart") for event in received)
    user_blocks: Final = frozenset(start.get("contentName") for start in content_starts if start.get("role") == "USER")
    text_inputs: Final = tuple(_fields(event, "textInput") for event in received)
    return tuple(
        string_value(text_input["content"])
        for text_input in text_inputs
        if text_input.get("contentName") in user_blocks
    )


def _text_deltas(events: tuple[Event, ...]) -> str:
    return "".join(string_value(event["delta"]) for event in events if event.get("type") == "response.text.delta")


def test_blocked_typed_text_never_reaches_nova_sonic_and_the_violation_is_voiced(
    candidate: Gateway, cert: tuple[Path, Path]
) -> None:
    turn: Final = Turn("user_input", user_transcript("ready") + assistant_text(VIOLATION))
    with bedrock_realtime_peer((turn,), cert) as peer, candidate.scenario() as scenario:
        phrase: Final = _blocked_phrase(candidate, scenario, "pre_call")
        model: Final = _deployment(candidate, scenario, peer)

        async def exchange(socket: ClientConnection) -> tuple[Event, ...]:
            await socket.send(_user_text(f"please repeat {phrase}"))
            await socket.send(RESPONSE_CREATE)
            return await _events_until(socket, _is_done)

        events: Final = asyncio.run(_session(candidate, model, scenario.key(), exchange))

        errors: Final = tuple(object_value(event["error"]) for event in events if _is_guardrail_error(event))
        assert [error["message"] for error in errors] == [VIOLATION], events
        assert [error["code"] for error in errors] == ["content_policy_violation"], events
        assert _text_deltas(events) == VIOLATION, events
        received: Final = peer.received()
        assert phrase not in json.dumps(received), received
        assert [VIOLATION in text for text in _user_text_inputs(received)] == [True], received


def test_clean_typed_text_reaches_nova_sonic_unchanged(candidate: Gateway, cert: tuple[Path, Path]) -> None:
    turn: Final = Turn("user_input", user_transcript("ready") + assistant_text(VISIBLE_ANSWER))
    text: Final = f"what is the weather in New York {uuid.uuid4().hex[:8]}"
    with bedrock_realtime_peer((turn,), cert) as peer, candidate.scenario() as scenario:
        _blocked_phrase(candidate, scenario, "pre_call")
        model: Final = _deployment(candidate, scenario, peer)

        async def exchange(socket: ClientConnection) -> tuple[Event, ...]:
            await socket.send(_user_text(text))
            await socket.send(RESPONSE_CREATE)
            return await _events_until(socket, _is_done)

        events: Final = asyncio.run(_session(candidate, model, scenario.key(), exchange))

        assert not any(_is_guardrail_error(event) for event in events), events
        assert _text_deltas(events) == VISIBLE_ANSWER, events
        assert _user_text_inputs(peer.received()) == (text,)


def test_blocked_transcript_hides_that_turns_answer_and_the_next_turn_still_answers(
    candidate: Gateway, cert: tuple[Path, Path]
) -> None:
    with candidate.scenario() as scenario:
        phrase: Final = _blocked_phrase(candidate, scenario, "realtime_input_transcription")
        turns: Final = (
            Turn("user_input", user_transcript(f"tell me {phrase}") + assistant_text(HIDDEN_ANSWER)),
            Turn("user_input", user_transcript("what is the weather") + assistant_text(VISIBLE_ANSWER)),
        )
        with bedrock_realtime_peer(turns, cert) as peer:
            model: Final = _deployment(candidate, scenario, peer)

            async def exchange(socket: ClientConnection) -> tuple[Event, ...]:
                await socket.send(AUDIO_APPEND)
                await socket.send(_user_text("hello"))
                blocked_turn: Final = await _events_until(socket, _is_guardrail_error)
                await socket.send(AUDIO_APPEND)
                return blocked_turn + await _events_until(socket, _is_done)

            events: Final = asyncio.run(_session(candidate, model, scenario.key(), exchange))

        assert HIDDEN_ANSWER not in json.dumps(events), events
        assert _text_deltas(events) == VISIBLE_ANSWER, events
        assert [event["type"] for event in events].count("response.done") == 1, events
        transcripts: Final = [
            event["transcript"]
            for event in events
            if event.get("type") == "conversation.item.input_audio_transcription.completed"
        ]
        assert transcripts == [f"tell me {phrase}", "what is the weather"], events
        assert [object_value(event["error"])["message"] for event in events if _is_guardrail_error(event)] == [
            VIOLATION
        ], events
