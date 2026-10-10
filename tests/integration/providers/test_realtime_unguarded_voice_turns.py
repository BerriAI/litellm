"""Without a transcript guardrail the upstream receives exactly the client's realtime frames.

The proxy sends its own ``response.create`` after a voice transcript only when a ``realtime_input_transcription``
guardrail turned the provider's VAD auto-response off. Without one the client owns its turns: one
``response.create`` per turn reaches the upstream, carrying the client's own metadata, and no bare one from the
proxy. The scripted upstream answers one event per ``input_audio_buffer.commit`` or ``response.create``, so a
proxy-sent ``response.create`` shows up both as an extra frame and as a reply the client never asked for.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import uuid
from collections import Counter
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from dataclasses import dataclass
from hashlib import sha256
from itertools import chain
from pathlib import Path
from typing import Final

import httpx
import psutil
import pytest
import websockets
from openai import AsyncOpenAI, OpenAI
from openai.resources.realtime.realtime import AsyncRealtimeConnection, RealtimeConnection
from openai.types.realtime import RealtimeSessionCreateRequestParam
from pydantic import JsonValue
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed

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
from tests.integration._support.process import (
    OwnedProxy,
    UpstreamSlot,
    graceful_stop_seconds,
    owned_proxy_process,
    owned_upstream,
)
from tests.integration._support.upstream import ScenarioHandle, delete_scenario, register_scenario
from tests.integration.cost_calculation.cost_tracking_case import RealtimeResponse

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

WORKERS: Final = 2
BURST: Final = 20
OPEN_SESSIONS: Final = 6
VOICE_MODEL: Final = "gpt-realtime-2.1"
XAI_VOICE_MODEL: Final = "grok-voice"
LIVE_MODEL: Final = "gemini-3.8-live"
LIVE_LOCATION: Final = "us-central1"
AZURE_API_VERSION: Final = "2025-04-01-preview"
TRANSCRIPT: Final = "what is the weather like today"
REALTIME_HOOK: Final = "realtime_input_transcription"
PROMPT_GUARDRAIL: Final = "prompt-filter"
PROMPT_WORD: Final = "pineapple"
MIXED_GUARDRAIL: Final = "prompt-and-transcript-filter"
MIXED_WORD: Final = "anchovy"
TRANSCRIPT_COMPLETED: Final = "conversation.item.input_audio_transcription.completed"
RESPONSE_DONE: Final = "response.done"
HANDSHAKE_REFUSED: Final = "Upstream realtime handshake rejected with HTTP 403"
REFUSAL_CLOSE: Final = 1008
CREATED_SECONDS: Final = 60.0
STEP_SECONDS: Final = 30.0
FIVE_KB: Final = "x" * 5120
PCM_200MS_24KHZ: Final = base64.b64encode(bytes(9600)).decode()
SPEND_SQL: Final = 'SELECT call_type FROM "LiteLLM_SpendLogs" WHERE api_key = %s'
ROUTES: Final = ("/v1/realtime", "/realtime", "/openai/v1/realtime")
BETA_HEADERS: Final = {"OpenAI-Beta": "realtime=v1"}
COMMIT: Final[dict[str, JsonValue]] = {"type": "input_audio_buffer.commit"}
APPEND: Final[dict[str, JsonValue]] = {"type": "input_audio_buffer.append", "audio": PCM_200MS_24KHZ}
SDK_PUSH_TO_TALK: Final[RealtimeSessionCreateRequestParam] = {
    "type": "realtime",
    "audio": {"input": {"turn_detection": None}},
}
PUSH_TO_TALK: Final = JSON_OBJECT.validate_python(SDK_PUSH_TO_TALK)
BETA_PUSH_TO_TALK: Final[dict[str, JsonValue]] = {"turn_detection": None}
VOICE_UPDATES: Final = (
    pytest.param({"type": "realtime", "instructions": "answer briefly"}, id="turn-detection-unset"),
    pytest.param(
        {"type": "realtime", "audio": {"input": {"turn_detection": {"type": "server_vad", "create_response": False}}}},
        id="server-vad-without-auto-response",
    ),
    pytest.param(
        {
            "type": "realtime",
            "audio": {"input": {"turn_detection": {"type": "semantic_vad", "create_response": False}}},
        },
        id="semantic-vad-without-auto-response",
    ),
)
TRANSCRIPT_FIELDS: Final = (
    pytest.param({}, id="missing"),
    pytest.param({"transcript": ""}, id="empty"),
    pytest.param({"transcript": FIVE_KB}, id="five-kilobytes"),
    pytest.param({"transcript": None}, id="null"),
    pytest.param({"transcript": 123}, id="int"),
    pytest.param({"transcript": ["a"]}, id="list"),
    pytest.param({"transcript": {"text": "a"}}, id="object"),
)
VOICE_UPDATE: Final[dict[str, JsonValue]] = {"type": "realtime", "instructions": "answer briefly"}
VOICE_UPDATE_FORWARDED: Final[dict[str, JsonValue]] = {
    "type": "realtime",
    "instructions": "answer briefly",
    "audio": {"input": {"turn_detection": {"create_response": False}}},
}
TURN_DETECTION_TYPE_MISSING: Final = "Missing required parameter: 'session.audio.input.turn_detection.type'."
INJECTED_UPDATE: Final[dict[str, JsonValue]] = {
    "type": "realtime",
    "audio": {"input": {"turn_detection": {"type": "server_vad", "create_response": False}}},
}
PROXY_RESPONSE_CREATE: Final[dict[str, JsonValue]] = {"type": "response.create"}


@dataclass(frozen=True, slots=True)
class Session:
    events: tuple[dict[str, JsonValue], ...]

    @property
    def types(self) -> tuple[str, ...]:
        return tuple(string_value(event["type"]) for event in self.events)

    @property
    def transcript_fields(self) -> tuple[dict[str, JsonValue], ...]:
        completed: Final = tuple(event for event in self.events if event["type"] == TRANSCRIPT_COMPLETED)
        return tuple({key: value for key, value in event.items() if key == "transcript"} for event in completed)

    @property
    def response_ids(self) -> tuple[str, ...]:
        done: Final = tuple(event for event in self.events if event["type"] == RESPONSE_DONE)
        return tuple(string_value(object_value(event["response"])["id"]) for event in done)

    @property
    def error_messages(self) -> tuple[str, ...]:
        errors: Final = tuple(event for event in self.events if event["type"] == "error")
        return tuple(string_value(object_value(event["error"])["message"]) for event in errors)

    @property
    def close_code(self) -> JsonValue:
        closes: Final = tuple(event for event in self.events if event["type"] == "closed")
        return closes[-1]["code"] if closes else None


@dataclass(frozen=True, slots=True)
class Step:
    frames: tuple[dict[str, JsonValue], ...]
    until: str


@dataclass(frozen=True, slots=True)
class Outage:
    turns: tuple[Session, ...]
    observed: tuple[dict[str, JsonValue], ...]
    held: tuple[Session, ...]


def _ws_base(http_url: str) -> str:
    return http_url.replace("https://", "wss://").replace("http://", "ws://")


def _gateway_url(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


def _owned_url(owned: OwnedProxy) -> str:
    return _gateway_url(owned.gateway)


def _transcript_event(fields: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "type": TRANSCRIPT_COMPLETED,
        "event_id": "evt_$UNIQUE_ID",
        "item_id": "item_$UNIQUE_ID",
        "content_index": 0,
        **fields,
        "usage": {"type": "duration", "seconds": 1.5},
    }


def _done_event() -> dict[str, JsonValue]:
    return {
        "type": RESPONSE_DONE,
        "event_id": "evt_$UNIQUE_ID",
        "response": {
            "id": "resp_$UNIQUE_ID",
            "object": "realtime.response",
            "status": "completed",
            "output": [],
            "usage": {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
        },
    }


def _turns(fields: tuple[Mapping[str, JsonValue], ...]) -> RealtimeResponse:
    events: Final = tuple(chain.from_iterable((_transcript_event(turn), _done_event()) for turn in fields))
    return RealtimeResponse(content_type="application/x-realtime", events=events)


def _spoken(transcript: str = TRANSCRIPT, *, turns: int = 1) -> RealtimeResponse:
    return _turns(({"transcript": transcript},) * turns)


def _live_turn() -> RealtimeResponse:
    return RealtimeResponse(
        content_type="application/x-realtime",
        events=(
            {"serverContent": {"inputTranscription": {"text": TRANSCRIPT}}},
            {"serverContent": {"modelTurn": {"parts": [{"text": "scripted $REQUEST_ID"}]}}},
            {
                "serverContent": {"turnComplete": True},
                "usageMetadata": {"promptTokenCount": 7, "responseTokenCount": 5, "totalTokenCount": 12},
            },
        ),
    )


def _scripted(scenario: Scenario, response: RealtimeResponse, *, control_url: str | None = None) -> ScenarioHandle:
    scenario_id: Final = f"realtime-unguarded-{uuid.uuid4().hex[:12]}"
    handle: Final = (
        register_scenario(scenario_id, response)
        if control_url is None
        else register_scenario(scenario_id, response, control_url=control_url)
    )
    scenario.cleanups.callback(delete_scenario, handle)
    return handle


def _openai_deployment(scenario: Scenario, scenario_id: str) -> str:
    return scenario.model(model=f"openai/{VOICE_MODEL}", api_key=scenario_id)


def _named_deployment(creator: Gateway, scenario: Scenario, scenario_id: str) -> str:
    name: Final = f"realtime-unguarded-owned-{uuid.uuid4().hex[:8]}"
    created: Final = creator.post(
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {
                "model": f"openai/{VOICE_MODEL}",
                "api_key": scenario_id,
                "api_base": f"{creator.upstream_url}/v1",
            },
            "model_info": {},
        },
    )
    scenario.cleanups.callback(scenario.delete_model, string_value(object_value(created["model_info"])["id"]))
    return name


def _live_credentials(upstream_url: str) -> str:
    return json.dumps(
        {
            "type": "external_account",
            "audience": "synthetic-vertex-audience",
            "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
            "token_url": f"{upstream_url}/_oauth/token",
            "credential_source": {"url": f"{upstream_url}/health"},
        }
    )


def _live_deployment(gateway: Gateway, scenario: Scenario, project: str) -> str:
    name: Final = f"vertex-live-unguarded-{uuid.uuid4().hex[:12]}"
    upstream: Final = gateway.upstream_url.rstrip("/")
    created: Final = gateway.post(
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {
                "model": f"vertex_ai/{LIVE_MODEL}",
                "vertex_project": project,
                "vertex_credentials": _live_credentials(upstream),
                "vertex_location": LIVE_LOCATION,
                "api_base": upstream,
            },
            "model_info": {"mode": "realtime"},
        },
    )
    scenario.cleanups.callback(scenario.delete_model, string_value(object_value(created["model_info"])["id"]))
    return name


def _client_response_create(label: str) -> dict[str, JsonValue]:
    return {"type": "response.create", "response": {"metadata": {"turn": label}}}


def _update_frame(update: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {"type": "session.update", "session": update}


def _turn(label: str) -> tuple[Step, Step]:
    return Step((COMMIT,), TRANSCRIPT_COMPLETED), Step((_client_response_create(label),), RESPONSE_DONE)


def _client_turns(update: dict[str, JsonValue], labels: tuple[str, ...]) -> tuple[Step, ...]:
    return (Step((_update_frame(update),), "session.updated"), *chain.from_iterable(map(_turn, labels)))


def _client_frames(update: dict[str, JsonValue], labels: tuple[str, ...]) -> tuple[dict[str, JsonValue], ...]:
    turns: Final = chain.from_iterable((COMMIT, _client_response_create(label)) for label in labels)
    return (_update_frame(update), *turns)


def _close_code(closed: ConnectionClosed) -> int:
    return 1006 if closed.rcvd is None else closed.rcvd.code


async def _next_event(socket: ClientConnection, deadline: float) -> dict[str, JsonValue] | None:
    remaining: Final = deadline - asyncio.get_running_loop().time()
    try:
        return JSON_OBJECT.validate_json(await asyncio.wait_for(socket.recv(), max(remaining, 0.01)))
    except TimeoutError:
        return None


async def _until(socket: ClientConnection, until: str) -> AsyncIterator[dict[str, JsonValue]]:
    deadline: Final = asyncio.get_running_loop().time() + STEP_SECONDS
    while (event := await _next_event(socket, deadline)) is not None:
        yield event
        if event.get("type") == until:
            return
    yield {"type": "timeout"}


async def _steps_after_created(
    socket: ClientConnection, steps: tuple[Step, ...]
) -> AsyncIterator[dict[str, JsonValue]]:
    for step in steps:
        for frame in step.frames:
            await socket.send(json.dumps(frame))
        async for event in _until(socket, step.until):
            yield event
            if event["type"] == "timeout":
                return


async def _stepped(socket: ClientConnection, steps: tuple[Step, ...]) -> AsyncIterator[dict[str, JsonValue]]:
    try:
        first: Final = JSON_OBJECT.validate_json(await asyncio.wait_for(socket.recv(), CREATED_SECONDS))
        yield first
        if first.get("type") != "session.created":
            async for message in socket:
                yield JSON_OBJECT.validate_json(message)
            return
        async for event in _steps_after_created(socket, steps):
            yield event
    except ConnectionClosed as closed:
        yield {"type": "closed", "code": _close_code(closed)}


def _headers(key: str, extra: Mapping[str, str] | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}", **(extra or {})}


async def _driven(url: str, key: str, steps: tuple[Step, ...], headers: Mapping[str, str] | None) -> Session:
    async with websockets.connect(url, additional_headers=_headers(key, headers)) as socket:
        return Session(tuple([event async for event in _stepped(socket, steps)]))


def _session_url(proxy_url: str, query: str, path: str = "/v1/realtime") -> str:
    return f"{_ws_base(proxy_url)}{path}?{query}"


def _drive(
    proxy_url: str,
    query: str,
    key: str,
    steps: tuple[Step, ...],
    *,
    path: str = "/v1/realtime",
    headers: Mapping[str, str] | None = None,
) -> Session:
    return asyncio.run(_driven(_session_url(proxy_url, query, path), key, steps, headers))


def _observed_all(upstream_url: str) -> tuple[dict[str, JsonValue], ...]:
    with httpx.Client(
        base_url=upstream_url, timeout=5, trust_env=False, verify=not upstream_url.startswith("https://")
    ) as upstream:
        observations: Final = JSON_OBJECT.validate_json(upstream.get("/__observations").content)
    requests: Final = observations["requests"]
    assert isinstance(requests, list), observations
    return tuple(map(object_value, requests))


def _belongs(request: dict[str, JsonValue], scenario_id: str) -> bool:
    return request["authorization"] == f"Bearer {scenario_id}" or request["api_key"] == scenario_id


def _observed(upstream_url: str, scenario_id: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(request for request in _observed_all(upstream_url) if _belongs(request, scenario_id))


def _upgrades(observed: tuple[dict[str, JsonValue], ...]) -> tuple[tuple[JsonValue, JsonValue], ...]:
    upgrades: Final = tuple(request for request in observed if request["method"] == "WEBSOCKET")
    return tuple((request["path"], object_value(request["body"]).get("query")) for request in upgrades)


def _sent(observed: tuple[dict[str, JsonValue], ...]) -> tuple[dict[str, JsonValue], ...]:
    return tuple(object_value(request["body"]) for request in observed if request["method"] == "WEBSOCKET_FRAME")


def _frame_counts(frames: Iterator[dict[str, JsonValue]] | tuple[dict[str, JsonValue], ...]) -> Counter[str]:
    return Counter(json.dumps(frame, sort_keys=True) for frame in frames)


def _spend_rows(key: str, count: int) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows(SPEND_SQL, (sha256(key.encode()).hexdigest(),)),
        lambda rows: len(rows) == count,
        seconds=70,
    )


def _assert_billed(key: str, sessions: int) -> None:
    rows: Final = _spend_rows(key, sessions)
    assert [row["call_type"] for row in rows] == ["_arealtime"] * sessions, rows


def _assert_turns_answered(session: Session, scenario_id: str, fields: tuple[Mapping[str, JsonValue], ...]) -> None:
    assert session.types == (
        "session.created",
        "session.updated",
        *(TRANSCRIPT_COMPLETED, RESPONSE_DONE) * len(fields),
    ), session
    assert session.transcript_fields == fields, session
    assert session.error_messages == (), session
    assert len(set(session.response_ids)) == len(fields), session
    assert all(response_id.startswith(f"resp_{scenario_id}-") for response_id in session.response_ids), session


def _assert_client_turns(
    session: Session,
    observed: tuple[dict[str, JsonValue], ...],
    scenario_id: str,
    update: dict[str, JsonValue],
    labels: tuple[str, ...],
    *,
    fields: tuple[Mapping[str, JsonValue], ...] | None = None,
) -> None:
    _assert_turns_answered(session, scenario_id, fields or ({"transcript": TRANSCRIPT},) * len(labels))
    assert _sent(observed) == _client_frames(update, labels), _sent(observed)


def _content_filter(name: str, mode: JsonValue, *, default_on: bool, word: str) -> dict[str, JsonValue]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "litellm_content_filter",
            "mode": mode,
            "default_on": default_on,
            "blocked_words": [{"keyword": word, "action": "BLOCK"}],
        },
    }


def _write_config(directory: Path, ca_bundle: Path) -> Path:
    config: Final = directory / f"realtime_unguarded_{uuid.uuid4().hex[:8]}.yaml"
    config.write_text(
        json.dumps(
            {
                "guardrails": [
                    _content_filter(PROMPT_GUARDRAIL, "pre_call", default_on=True, word=PROMPT_WORD),
                    _content_filter(MIXED_GUARDRAIL, ["pre_call", REALTIME_HOOK], default_on=False, word=MIXED_WORD),
                ],
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "store_model_in_db": True,
                    "proxy_batch_write_at": 1,
                    "proxy_batch_polling_interval": 1,
                },
                "litellm_settings": {
                    "ssl_verify": str(ca_bundle),
                    "enable_redis_auth_cache": True,
                    "cache": True,
                    "cache_params": {"type": "redis", "host": "os.environ/REDIS_HOST", "port": "os.environ/REDIS_PORT"},
                },
                "router_settings": {"disable_cooldowns": True},
            }
        )
    )
    return config


def _ca_bundle(slot: UpstreamSlot) -> Path:
    assert slot.certificate is not None, "the TLS upstream carries its own certificate"
    return slot.certificate.certificate


@pytest.fixture(scope="module")
def tls_slot(tmp_path_factory: pytest.TempPathFactory) -> Iterator[UpstreamSlot]:
    with owned_upstream(tmp_path_factory.mktemp("unguarded-tls-upstream"), tls=True) as slot:
        yield slot


@pytest.fixture(scope="module")
def tls_upstream(tls_slot: UpstreamSlot) -> str:
    return tls_slot.url


@pytest.fixture(scope="module")
def guard_proxy(tmp_path_factory: pytest.TempPathFactory, tls_slot: UpstreamSlot) -> Iterator[OwnedProxy]:
    directory: Final = tmp_path_factory.mktemp("realtime-unguarded-proxy")
    ca_bundle: Final = _ca_bundle(tls_slot)
    with (
        gateway_from_environment() as rig,
        owned_proxy_process(
            rig,
            directory,
            {"DATABASE_URL": os.environ["DATABASE_URL"], "SSL_VERIFY": str(ca_bundle)},
            config=_write_config(directory, ca_bundle),
            workers=WORKERS,
        ) as owned,
    ):
        yield owned


@pytest.mark.parametrize("path", ROUTES)
def test_push_to_talk_turn_reaches_the_upstream_as_the_clients_own_frames(gateway: Gateway, path: str) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _spoken())
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        session: Final = _drive(
            _gateway_url(gateway), f"model={model}", key, _client_turns(PUSH_TO_TALK, ("1",)), path=path
        )
        observed: Final = _observed(gateway.upstream_url, handle.scenario_id)
        _assert_client_turns(session, observed, handle.scenario_id, PUSH_TO_TALK, ("1",))
        assert _upgrades(observed) == (("/v1/realtime", [["model", VOICE_MODEL]]),), observed
        _assert_billed(key, 1)


@pytest.mark.parametrize("update", VOICE_UPDATES)
def test_voice_session_turn_carries_only_the_clients_response_create(
    gateway: Gateway, update: dict[str, JsonValue]
) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _spoken())
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        session: Final = _drive(_gateway_url(gateway), f"model={model}", key, _client_turns(update, ("1",)))
        observed: Final = _observed(gateway.upstream_url, handle.scenario_id)
        _assert_client_turns(session, observed, handle.scenario_id, update, ("1",))
        _assert_billed(key, 1)


def test_beta_protocol_push_to_talk_turn_reaches_the_upstream_as_the_clients_own_frames(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _spoken())
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        session: Final = _drive(
            _gateway_url(gateway),
            f"model={model}",
            key,
            _client_turns(BETA_PUSH_TO_TALK, ("1",)),
            headers=BETA_HEADERS,
        )
        observed: Final = _observed(gateway.upstream_url, handle.scenario_id)
        _assert_client_turns(session, observed, handle.scenario_id, BETA_PUSH_TO_TALK, ("1",))
        _assert_billed(key, 1)


def test_two_turns_with_the_same_transcript_get_one_reply_each(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _spoken(turns=2))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        session: Final = _drive(_gateway_url(gateway), f"model={model}", key, _client_turns(PUSH_TO_TALK, ("1", "2")))
        observed: Final = _observed(gateway.upstream_url, handle.scenario_id)
        _assert_client_turns(session, observed, handle.scenario_id, PUSH_TO_TALK, ("1", "2"))
        _assert_billed(key, 1)


async def _sdk_async_events(connection: AsyncRealtimeConnection, until: str) -> AsyncIterator[dict[str, JsonValue]]:
    async for event in connection:
        yield JSON_OBJECT.validate_python(event.model_dump())
        if event.type == until:
            return


def _sdk_sync_events(connection: RealtimeConnection, until: str) -> Iterator[dict[str, JsonValue]]:
    for event in connection:
        yield JSON_OBJECT.validate_python(event.model_dump())
        if event.type == until:
            return


async def _sdk_async_turn(proxy_url: str, key: str, model: str) -> Session:
    client: Final = AsyncOpenAI(api_key=key, base_url=f"{proxy_url}/v1", websocket_base_url=f"{_ws_base(proxy_url)}/v1")
    async with client.realtime.connect(model=model) as connection:
        created: Final = [event async for event in _sdk_async_events(connection, "session.created")]
        await connection.session.update(session=SDK_PUSH_TO_TALK)
        updated: Final = [event async for event in _sdk_async_events(connection, "session.updated")]
        await connection.input_audio_buffer.commit()
        transcribed: Final = [event async for event in _sdk_async_events(connection, TRANSCRIPT_COMPLETED)]
        await connection.response.create(response={"metadata": {"turn": "1"}})
        answered: Final = [event async for event in _sdk_async_events(connection, RESPONSE_DONE)]
        return Session((*created, *updated, *transcribed, *answered))


def _sdk_sync_turn(proxy_url: str, key: str, model: str) -> Session:
    client: Final = OpenAI(api_key=key, base_url=f"{proxy_url}/v1", websocket_base_url=f"{_ws_base(proxy_url)}/v1")
    with client.realtime.connect(model=model) as connection:
        created: Final = tuple(_sdk_sync_events(connection, "session.created"))
        connection.session.update(session=SDK_PUSH_TO_TALK)
        updated: Final = tuple(_sdk_sync_events(connection, "session.updated"))
        connection.input_audio_buffer.commit()
        transcribed: Final = tuple(_sdk_sync_events(connection, TRANSCRIPT_COMPLETED))
        connection.response.create(response={"metadata": {"turn": "1"}})
        answered: Final = tuple(_sdk_sync_events(connection, RESPONSE_DONE))
        return Session((*created, *updated, *transcribed, *answered))


@pytest.mark.parametrize("client", ["async", "sync"])
def test_openai_sdk_push_to_talk_turn_reaches_the_upstream_as_the_clients_own_frames(
    gateway: Gateway, client: str
) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _spoken())
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        session: Final = (
            asyncio.run(_sdk_async_turn(_gateway_url(gateway), key, model))
            if client == "async"
            else _sdk_sync_turn(_gateway_url(gateway), key, model)
        )
        observed: Final = _observed(gateway.upstream_url, handle.scenario_id)
        _assert_client_turns(session, observed, handle.scenario_id, PUSH_TO_TALK, ("1",))
        _assert_billed(key, 1)


def test_azure_push_to_talk_turn_reaches_the_upstream_as_the_clients_own_frames(
    guard_proxy: OwnedProxy, tls_upstream: str
) -> None:
    with guard_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _spoken(), control_url=tls_upstream)
        key: Final = scenario.key()
        model: Final = scenario.model(
            model=f"azure/{VOICE_MODEL}",
            api_key=handle.scenario_id,
            api_base=_ws_base(tls_upstream),
            api_version=AZURE_API_VERSION,
        )
        session: Final = _drive(_owned_url(guard_proxy), f"model={model}", key, _client_turns(PUSH_TO_TALK, ("1",)))
        observed: Final = _observed(tls_upstream, handle.scenario_id)
        _assert_client_turns(session, observed, handle.scenario_id, PUSH_TO_TALK, ("1",))
        assert _upgrades(observed) == (("/openai/v1/realtime", [["model", VOICE_MODEL]]),), observed
        _assert_billed(key, 1)


def test_xai_push_to_talk_turn_reaches_the_upstream_as_the_clients_own_frames(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _spoken())
        key: Final = scenario.key()
        model: Final = scenario.model(model=f"xai/{XAI_VOICE_MODEL}", api_key=handle.scenario_id)
        session: Final = _drive(_gateway_url(gateway), f"model={model}", key, _client_turns(PUSH_TO_TALK, ("1",)))
        observed: Final = _observed(gateway.upstream_url, handle.scenario_id)
        _assert_client_turns(session, observed, handle.scenario_id, PUSH_TO_TALK, ("1",))
        assert _upgrades(observed) == (("/v1/realtime", [["model", XAI_VOICE_MODEL]]),), observed
        _assert_billed(key, 1)


def test_prompt_guardrail_alone_leaves_the_turn_to_the_client_when_the_transcript_has_its_word(
    guard_proxy: OwnedProxy,
) -> None:
    transcript: Final = f"please add {PROMPT_WORD} to the order"
    with guard_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _spoken(transcript))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        session: Final = _drive(_owned_url(guard_proxy), f"model={model}", key, _client_turns(PUSH_TO_TALK, ("1",)))
        observed: Final = _observed(guard_proxy.gateway.upstream_url, handle.scenario_id)
        _assert_client_turns(
            session, observed, handle.scenario_id, PUSH_TO_TALK, ("1",), fields=({"transcript": transcript},)
        )
        _assert_billed(key, 1)


def test_vertex_live_voice_turn_sends_the_upstream_only_the_setup_and_the_audio(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _live_turn())
        key: Final = scenario.key()
        model: Final = _live_deployment(gateway, scenario, handle.scenario_id)
        session: Final = _drive(_gateway_url(gateway), f"model={model}", key, (Step((APPEND,), RESPONSE_DONE),))
        observed: Final = _observed(gateway.upstream_url, handle.scenario_id)
        assert session.types[0] == "session.created", session
        assert session.types[-1] == RESPONSE_DONE, session
        assert session.transcript_fields == ({"transcript": TRANSCRIPT},), session
        assert session.error_messages == (), session
        assert len(session.response_ids) == 1, session
        sent: Final = _sent(observed)
        assert [sorted(frame) for frame in sent] == [["setup"], ["realtimeInput"]], sent
        audio: Final = object_value(object_value(sent[1]["realtimeInput"])["audio"])
        assert audio["data"] == PCM_200MS_24KHZ, audio
        _assert_billed(key, 1)


@pytest.mark.parametrize("fields", TRANSCRIPT_FIELDS)
def test_transcript_field_shape_is_relayed_and_only_the_clients_response_create_follows(
    gateway: Gateway, fields: dict[str, JsonValue]
) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _turns((fields,)))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        session: Final = _drive(_gateway_url(gateway), f"model={model}", key, _client_turns(PUSH_TO_TALK, ("1",)))
        observed: Final = _observed(gateway.upstream_url, handle.scenario_id)
        _assert_client_turns(session, observed, handle.scenario_id, PUSH_TO_TALK, ("1",), fields=(fields,))
        _assert_billed(key, 1)


def test_refused_upstream_upgrade_reaches_the_client_and_the_next_session_keeps_its_own_turns(
    gateway: Gateway,
) -> None:
    with gateway.scenario() as scenario:
        unregistered: Final = f"realtime-unguarded-unknown-{uuid.uuid4().hex[:12]}"
        handle: Final = _scripted(scenario, _spoken())
        key: Final = scenario.key()
        refused_model: Final = _openai_deployment(scenario, unregistered)
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        refused: Final = _drive(
            _gateway_url(gateway), f"model={refused_model}", key, _client_turns(PUSH_TO_TALK, ("1",))
        )
        assert refused.types == ("error", "closed"), refused
        assert refused.error_messages == (HANDSHAKE_REFUSED,), refused
        assert refused.close_code == REFUSAL_CLOSE, refused
        refused_observed: Final = _observed(gateway.upstream_url, unregistered)
        assert _upgrades(refused_observed) == (("/v1/realtime", [["model", VOICE_MODEL]]),), refused_observed
        assert _sent(refused_observed) == (), refused_observed
        session: Final = _drive(_gateway_url(gateway), f"model={model}", key, _client_turns(PUSH_TO_TALK, ("1",)))
        observed: Final = _observed(gateway.upstream_url, handle.scenario_id)
        _assert_client_turns(session, observed, handle.scenario_id, PUSH_TO_TALK, ("1",))
        _assert_billed(key, 1)


def _labels(count: int) -> tuple[str, ...]:
    return tuple(str(index) for index in range(count))


def _every_client_frame(labels: tuple[str, ...]) -> Iterator[dict[str, JsonValue]]:
    return chain.from_iterable(_client_frames(PUSH_TO_TALK, (label,)) for label in labels)


async def _concurrent_turns(url: str, key: str, labels: tuple[str, ...]) -> tuple[Session, ...]:
    sessions: Final = (_driven(url, key, _client_turns(PUSH_TO_TALK, (label,)), None) for label in labels)
    return tuple(await asyncio.gather(*sessions))


def test_concurrent_sessions_each_get_only_their_own_response_create(gateway: Gateway) -> None:
    labels: Final = _labels(BURST)
    with gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _spoken())
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        sessions: Final = asyncio.run(
            _concurrent_turns(_session_url(_gateway_url(gateway), f"model={model}"), key, labels)
        )
        for session in sessions:
            _assert_turns_answered(session, handle.scenario_id, ({"transcript": TRANSCRIPT},))
        assert len({response_id for session in sessions for response_id in session.response_ids}) == BURST, sessions
        sent: Final = _sent(_observed(gateway.upstream_url, handle.scenario_id))
        assert _frame_counts(sent) == _frame_counts(_every_client_frame(labels)), sent
        _assert_billed(key, BURST)


async def _frames_until_closed(socket: ClientConnection) -> AsyncIterator[dict[str, JsonValue]]:
    try:
        async for message in socket:
            yield JSON_OBJECT.validate_json(message)
    except ConnectionClosed as closed:
        yield {"type": "closed", "code": _close_code(closed)}


async def _turn_then_hold(url: str, key: str, label: str, turned: asyncio.Queue[Session]) -> Session:
    async with websockets.connect(url, additional_headers=_headers(key, None)) as socket:
        turn: Final = Session(tuple([event async for event in _stepped(socket, _client_turns(PUSH_TO_TALK, (label,)))]))
        await turned.put(turn)
        return Session(tuple([frame async for frame in _frames_until_closed(socket)]))


async def _drain(turned: asyncio.Queue[Session], count: int) -> tuple[Session, ...]:
    return tuple([await turned.get() for _ in range(count)])


async def _burst_through_outage(
    url: str,
    proxy_url: str,
    key: str,
    observe: Callable[[], tuple[dict[str, JsonValue], ...]],
    stop_upstream: Callable[[], None],
) -> Outage:
    turned: Final[asyncio.Queue[Session]] = asyncio.Queue()
    holders: Final = tuple(asyncio.ensure_future(_turn_then_hold(url, key, label, turned)) for label in _labels(BURST))
    turns: Final = await asyncio.wait_for(_drain(turned, BURST), 120)
    observed: Final = await asyncio.to_thread(observe)
    await asyncio.to_thread(stop_upstream)
    async with httpx.AsyncClient(base_url=proxy_url, timeout=15, trust_env=False) as client:
        liveliness: Final = await client.get("/health/liveliness")
    assert liveliness.status_code == 200, liveliness.text
    return Outage(turns, observed, tuple(await asyncio.wait_for(asyncio.gather(*holders), 90)))


def test_upstream_outage_mid_burst_closes_every_session_after_only_the_clients_frames(
    gateway: Gateway, tmp_path: Path
) -> None:
    with gateway.scenario() as scenario, owned_upstream(tmp_path) as slot:
        scenario_id: Final = f"realtime-unguarded-outage-{uuid.uuid4().hex[:12]}"
        register_scenario(scenario_id, _spoken(), control_url=slot.url)
        key: Final = scenario.key()
        model: Final = scenario.model(model=f"openai/{VOICE_MODEL}", api_key=scenario_id, api_base=slot.url)
        proxy_url: Final = _gateway_url(gateway)
        outage: Final = asyncio.run(
            _burst_through_outage(
                _session_url(proxy_url, f"model={model}"),
                proxy_url,
                key,
                lambda: _observed(slot.url, scenario_id),
                slot.stop,
            )
        )
        for turn in outage.turns:
            _assert_turns_answered(turn, scenario_id, ({"transcript": TRANSCRIPT},))
        assert _frame_counts(_sent(outage.observed)) == _frame_counts(_every_client_frame(_labels(BURST))), outage
        assert [session.types for session in outage.held] == [("error", "closed")] * BURST, outage.held
        slot.start()
        register_scenario(scenario_id, _spoken(), control_url=slot.url)
        recovered: Final = _drive(proxy_url, f"model={model}", key, _client_turns(PUSH_TO_TALK, ("recovered",)))
        _assert_client_turns(recovered, _observed(slot.url, scenario_id), scenario_id, PUSH_TO_TALK, ("recovered",))
        _assert_billed(key, BURST + 1)


def _spawned_worker(child: psutil.Process) -> bool:
    try:
        return any("multiprocessing.spawn" in part for part in child.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def _workers(root: psutil.Process) -> tuple[psutil.Process, ...]:
    return tuple(
        sorted((child for child in root.children() if _spawned_worker(child)), key=lambda process: process.pid)
    )


async def _turn_or_close(socket: ClientConnection, label: str) -> Session:
    try:
        return Session(
            tuple([event async for event in _steps_after_created(socket, _client_turns(PUSH_TO_TALK, (label,)))])
        )
    except ConnectionClosed as closed:
        return Session(({"type": "closed", "code": _close_code(closed)},))


async def _open_sessions(url: str, key: str) -> tuple[ClientConnection, ...]:
    sockets: Final = tuple(
        [await websockets.connect(url, additional_headers=_headers(key, None)) for _ in range(OPEN_SESSIONS)]
    )
    created: Final = await asyncio.wait_for(asyncio.gather(*(socket.recv() for socket in sockets)), 60)
    assert [JSON_OBJECT.validate_json(event).get("type") for event in created] == ["session.created"] * OPEN_SESSIONS
    return sockets


async def _sessions_through_worker_kill(url: str, key: str, root: psutil.Process) -> tuple[Session, ...]:
    sockets: Final = await _open_sessions(url, key)
    try:
        workers: Final = _workers(root)
        assert len(workers) == WORKERS, [process.pid for process in workers]
        workers[0].kill()
        await asyncio.to_thread(workers[0].wait, 10)
        turns: Final = (_turn_or_close(socket, label) for socket, label in zip(sockets, _labels(OPEN_SESSIONS)))
        return tuple(await asyncio.wait_for(asyncio.gather(*turns), 90))
    finally:
        await asyncio.gather(*(socket.close() for socket in sockets))


def _served(sessions: tuple[Session, ...]) -> tuple[str, ...]:
    labels: Final = _labels(OPEN_SESSIONS)
    return tuple(label for label, session in zip(labels, sessions) if session.types[-1:] == (RESPONSE_DONE,))


def test_worker_kill_leaves_the_surviving_sessions_with_only_their_own_frames(gateway: Gateway, tmp_path: Path) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _spoken())
        fresh: Final = _scripted(scenario, _spoken())
        key: Final = scenario.key()
        fresh_key: Final = scenario.key()
        with owned_proxy_process(gateway, tmp_path, {}, workers=WORKERS) as owned:
            model: Final = _named_deployment(owned.gateway, scenario, handle.scenario_id)
            fresh_model: Final = _named_deployment(owned.gateway, scenario, fresh.scenario_id)
            root: Final = psutil.Process(owned.process.pid)
            killed_pid: Final = _workers(root)[0].pid
            sessions: Final = asyncio.run(
                _sessions_through_worker_kill(_session_url(_owned_url(owned), f"model={model}"), key, root)
            )
            served: Final = _served(sessions)
            assert [session.types for session in sessions] == [
                ("session.updated", TRANSCRIPT_COMPLETED, RESPONSE_DONE) if label in served else ("closed",)
                for label in _labels(OPEN_SESSIONS)
            ], sessions
            sent: Final = _sent(_observed(gateway.upstream_url, handle.scenario_id))
            assert _frame_counts(sent) == _frame_counts(_every_client_frame(served)), sent
            with httpx.Client(base_url=_owned_url(owned), timeout=15, trust_env=False) as client:
                readiness: Final = client.get("/health/readiness")
            assert readiness.status_code == 200, readiness.text
            eventually(
                lambda: tuple(process.pid for process in _workers(root)),
                lambda pids: len(pids) == WORKERS and killed_pid not in pids,
                seconds=60,
            )
            fresh_session: Final = _drive(
                _owned_url(owned), f"model={fresh_model}", fresh_key, _client_turns(PUSH_TO_TALK, ("fresh",))
            )
            _assert_client_turns(
                fresh_session,
                _observed(gateway.upstream_url, fresh.scenario_id),
                fresh.scenario_id,
                PUSH_TO_TALK,
                ("fresh",),
            )
            _assert_billed(fresh_key, 1)
            _assert_billed(key, len(served))


def test_list_mode_transcript_guardrail_still_turns_auto_response_off_and_answers_the_turn(
    guard_proxy: OwnedProxy,
) -> None:
    with guard_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _spoken())
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        session: Final = _drive(
            _owned_url(guard_proxy),
            f"model={model}&guardrails={MIXED_GUARDRAIL}",
            key,
            (Step((_update_frame(VOICE_UPDATE), COMMIT), RESPONSE_DONE),),
        )
        observed: Final = _observed(guard_proxy.gateway.upstream_url, handle.scenario_id)
        assert session.types == (
            "session.created",
            "session.updated",
            "error",
            TRANSCRIPT_COMPLETED,
            RESPONSE_DONE,
        ), session
        assert session.transcript_fields == ({"transcript": TRANSCRIPT},), session
        assert session.error_messages == (TURN_DETECTION_TYPE_MISSING,), session
        assert _sent(observed) == (
            _update_frame(INJECTED_UPDATE),
            _update_frame(VOICE_UPDATE_FORWARDED),
            COMMIT,
            PROXY_RESPONSE_CREATE,
        ), _sent(observed)
        _assert_billed(key, 1)
