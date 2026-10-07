"""Transcript guardrails leave transcription-only realtime sessions alone.

With a ``realtime_input_transcription`` guardrail configured, the proxy disables the provider's VAD auto-response
by injecting ``turn_detection.create_response: false`` into ``session.update`` frames so the guardrail can gate
every assistant turn. A transcription session has no assistant turn and the vendors reject those updates, so the
injection is skipped when the route intent or a backend session event says the session is transcription-only,
while a client frame alone never flips a voice session into one. Every row runs against the scripted upstream,
which answers the injected and rewritten updates the way the vendors do.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import psutil
import pytest
import websockets
from openai import AsyncOpenAI, OpenAI
from openai.resources.realtime.realtime import AsyncRealtimeConnection, RealtimeConnection
from openai.types.realtime import RealtimeTranscriptionSessionCreateRequestParam
from pydantic import JsonValue
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed, InvalidStatus

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
    owned_proxy_process,
    owned_upstream,
    stop_root_process,
)
from tests.integration._support.upstream import ScenarioHandle, delete_scenario, register_scenario
from tests.integration.cost_calculation.cost_tracking_case import RealtimeResponse

RecordProperty = Callable[[str, object], None]

pytestmark: Final = pytest.mark.timeout(180)

WORKERS: Final = 2
BURST: Final = 20
OPEN_SESSIONS: Final = 6
BLOCKED_WORD: Final = "pineapple"
CLEAN_TRANSCRIPT: Final = "the weather is fine today"
BLOCKED_TRANSCRIPT: Final = f"please add {BLOCKED_WORD} to the order"
TRANSCRIBE_MODEL: Final = "gpt-live-transcribe"
VOICE_MODEL: Final = "gpt-realtime-2"
WHISPER_DEFAULT: Final = "gpt-realtime-whisper"
MUSE_MODEL: Final = "muse-voice-transcribe-1.0"
TRANSCRIPT_GUARDRAIL: Final = "transcript-filter"
OPTIN_GUARDRAIL: Final = "transcript-filter-optin"
PROMPT_GUARDRAIL: Final = "prompt-filter"
TRANSCRIPTION: Final = "transcription"
TRANSCRIPTION_QUERY: Final = f"intent={TRANSCRIPTION}"
OPTIN_QUERY: Final = f"guardrails={OPTIN_GUARDRAIL}"
BETA_HEADERS: Final = {"OpenAI-Beta": "realtime=v1"}
CREATED_TYPES: Final = frozenset({"session.created", "transcription_session.created"})
TRANSCRIPT_COMPLETED: Final = "conversation.item.input_audio_transcription.completed"
RESPONSE_DONE: Final = "response.done"
GUARDRAIL_VIOLATION: Final = ("guardrail_violation", "content_policy_violation")
MISSING_TURN_DETECTION_TYPE: Final = ("invalid_request_error", "missing_required_parameter")
INVALID_SESSION_VALUE: Final = ("invalid_request_error", "invalid_value")
FIVE_KB: Final = "x" * 5120
UNAUTHENTICATED_STATUS: Final = 403
UNKNOWN_MODEL_CLOSE: Final = 1011
AUDIO_SECONDS: Final = 1.5
MUSE_AUDIO_MS: Final = 1500
PCM_200MS_24KHZ: Final = base64.b64encode(bytes(9600)).decode()
MUSE_PACKET_BYTES: Final = 3840
MUSE_REMAINDER_BYTES: Final = 1920
SPEND_SQL: Final = 'SELECT call_type FROM "LiteLLM_SpendLogs" WHERE api_key = %s'
SDK_TRANSCRIPTION_SESSION: Final[RealtimeTranscriptionSessionCreateRequestParam] = {
    "type": "transcription",
    "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}, "transcription": {"model": TRANSCRIBE_MODEL}}},
}
GA_TRANSCRIPTION_UPDATE: Final = JSON_OBJECT.validate_python(SDK_TRANSCRIPTION_SESSION)
BETA_TRANSCRIPTION_UPDATE: Final[dict[str, JsonValue]] = {
    "input_audio_format": "pcm16",
    "input_audio_transcription": {"model": TRANSCRIBE_MODEL},
}
VOICE_UPDATE: Final[dict[str, JsonValue]] = {"type": "realtime", "instructions": "answer briefly"}
VOICE_UPDATE_FORWARDED: Final[dict[str, JsonValue]] = {
    "type": "realtime",
    "instructions": "answer briefly",
    "audio": {"input": {"turn_detection": {"create_response": False}}},
}
RE_ENABLE_UPDATE: Final[dict[str, JsonValue]] = {
    "type": "realtime",
    "audio": {"input": {"turn_detection": {"type": "server_vad", "create_response": True}}},
}
RE_ENABLE_FORWARDED: Final[dict[str, JsonValue]] = {
    "type": "realtime",
    "audio": {"input": {"turn_detection": {"type": "server_vad", "create_response": False}}},
}
CLIENT_DECLARED_TRANSCRIPTION: Final[dict[str, JsonValue]] = {
    "type": "transcription",
    "audio": {"input": {"turn_detection": {"type": "server_vad"}}},
}
CLIENT_DECLARED_TRANSCRIPTION_FORWARDED: Final[dict[str, JsonValue]] = {
    "type": "transcription",
    "audio": {"input": {"turn_detection": {"create_response": False}}},
}
TURN_DETECTION_TYPE_MISSING: Final = "Missing required parameter: 'session.audio.input.turn_detection.type'."
GA_INJECTED_UPDATE: Final[dict[str, JsonValue]] = {
    "type": "realtime",
    "audio": {"input": {"turn_detection": {"type": "server_vad", "create_response": False}}},
}
BETA_INJECTED_UPDATE: Final[dict[str, JsonValue]] = {"turn_detection": {"type": "server_vad", "create_response": False}}
COMMIT: Final[dict[str, JsonValue]] = {"type": "input_audio_buffer.commit"}
APPEND: Final[dict[str, JsonValue]] = {"type": "input_audio_buffer.append", "audio": PCM_200MS_24KHZ}
PUSH_TO_TALK: Final = "PUSH_TO_TALK"
ENDPOINTING: Final = "ENDPOINTING"
END_STREAM: Final[dict[str, JsonValue]] = {"type": "endStream"}


@dataclass(frozen=True, slots=True)
class Session:
    events: tuple[dict[str, JsonValue], ...]
    refused: int | None

    @property
    def types(self) -> tuple[str, ...]:
        return tuple(string_value(event["type"]) for event in self.events)

    @property
    def close_code(self) -> int | None:
        closes: Final = tuple(event for event in self.events if event["type"] == "closed")
        return _integer(closes[-1]["code"]) if closes else None

    @property
    def session_type(self) -> JsonValue:
        return object_value(self.events[0]["session"]).get("type")

    @property
    def transcripts(self) -> tuple[str, ...]:
        completed: Final = tuple(event for event in self.events if event["type"] == TRANSCRIPT_COMPLETED)
        return tuple(string_value(event["transcript"]) for event in completed)

    @property
    def errors(self) -> tuple[tuple[str, str], ...]:
        errors: Final = tuple(object_value(event["error"]) for event in self.events if event["type"] == "error")
        return tuple((string_value(error["type"]), string_value(error.get("code") or "")) for error in errors)

    @property
    def error_messages(self) -> tuple[str, ...]:
        errors: Final = tuple(event for event in self.events if event["type"] == "error")
        return tuple(string_value(object_value(event["error"])["message"]) for event in errors)


def _integer(value: JsonValue) -> int:
    assert isinstance(value, int), value
    return value


def _ws_base(http_url: str) -> str:
    return http_url.replace("https://", "wss://").replace("http://", "ws://")


def _proxy_url() -> str:
    return os.environ["INTEGRATION_PROXY_URL"].rstrip("/")


def _owned_url(owned: OwnedProxy) -> str:
    return str(owned.gateway.client.base_url).rstrip("/")


def _transcript_event(transcript: str) -> dict[str, JsonValue]:
    return {
        "type": TRANSCRIPT_COMPLETED,
        "event_id": "evt_$UNIQUE_ID",
        "item_id": "item_$UNIQUE_ID",
        "content_index": 0,
        "transcript": transcript,
        "usage": {"type": "duration", "seconds": AUDIO_SECONDS},
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


def _transcription_scenario(transcript: str, *, repeats: int = 1) -> RealtimeResponse:
    return RealtimeResponse(
        content_type="application/x-realtime",
        events=(_transcript_event(transcript),),
        session_type=TRANSCRIPTION,
        created_repeats=repeats,
    )


def _older_transcription_scenario(transcript: str) -> RealtimeResponse:
    return RealtimeResponse(
        content_type="application/x-realtime",
        events=(_transcript_event(transcript),),
        session_type=TRANSCRIPTION,
        created_event="transcription_session.created",
    )


def _voice_scenario(transcript: str) -> RealtimeResponse:
    return RealtimeResponse(
        content_type="application/x-realtime", events=(_transcript_event(transcript), _done_event())
    )


def _muse_scenario(transcript: str) -> RealtimeResponse:
    return RealtimeResponse(
        content_type="application/x-realtime",
        events=(
            {"type": "audioProgress", "audioProcessedMs": MUSE_AUDIO_MS},
            {"type": "speechStart", "turnId": "turn_$REQUEST_ID"},
            {"type": "transcript", "turnId": "turn_$REQUEST_ID", "transcript": transcript, "final": True},
        ),
    )


def _scripted(scenario: Scenario, response: RealtimeResponse, *, control_url: str | None = None) -> ScenarioHandle:
    scenario_id: Final = f"realtime-guard-{uuid.uuid4().hex[:12]}"
    handle: Final = (
        register_scenario(scenario_id, response)
        if control_url is None
        else register_scenario(scenario_id, response, control_url=control_url)
    )
    scenario.cleanups.callback(delete_scenario, handle)
    return handle


def _openai_deployment(scenario: Scenario, scenario_id: str, *, model: str = TRANSCRIBE_MODEL) -> str:
    return scenario.model(model=f"openai/{model}", api_key=scenario_id)


def _azure_deployment(scenario: Scenario, scenario_id: str, tls_upstream: str) -> str:
    return scenario.model(
        model=f"azure/{TRANSCRIBE_MODEL}",
        api_key=scenario_id,
        api_base=_ws_base(tls_upstream),
        api_version="2025-04-01-preview",
    )


def _with_transcription_model(update: dict[str, JsonValue], model: str) -> dict[str, JsonValue]:
    audio_input: Final = object_value(object_value(update["audio"])["input"])
    return {**update, "audio": {"input": {**audio_input, "transcription": {"model": model}}}}


def _muse_deployment(scenario: Scenario, scenario_id: str, api_base: str) -> str:
    return scenario.model(model=f"meta/{MUSE_MODEL}", api_key=scenario_id, api_base=api_base)


def _named_deployment(creator: Gateway, scenario: Scenario, name: str, model: str, scenario_id: str) -> str:
    created: Final = creator.post(
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {"model": model, "api_key": scenario_id, "api_base": f"{creator.upstream_url}/v1"},
            "model_info": {},
        },
    )
    scenario.cleanups.callback(scenario.delete_model, string_value(object_value(created["model_info"])["id"]))
    return name


def _close_code(closed: ConnectionClosed) -> int:
    return 1006 if closed.rcvd is None else closed.rcvd.code


async def _next_event(socket: ClientConnection, deadline: float) -> dict[str, JsonValue] | None:
    remaining: Final = deadline - asyncio.get_running_loop().time()
    try:
        return JSON_OBJECT.validate_json(await asyncio.wait_for(socket.recv(), max(remaining, 0.01)))
    except TimeoutError:
        return None


async def _events(
    socket: ClientConnection, frames: tuple[dict[str, JsonValue], ...], until: frozenset[str], seconds: float
) -> AsyncIterator[dict[str, JsonValue]]:
    deadline: Final = asyncio.get_running_loop().time() + seconds
    try:
        first: Final = JSON_OBJECT.validate_json(await asyncio.wait_for(socket.recv(), seconds))
        yield first
        if first.get("type") not in CREATED_TYPES:
            async for message in socket:
                yield JSON_OBJECT.validate_json(message)
            return
        for frame in frames:
            await socket.send(json.dumps(frame))
        while True:
            event: Final = await _next_event(socket, deadline)
            if event is None:
                yield {"type": "timeout"}
                return
            yield event
            if event.get("type") in until:
                return
    except ConnectionClosed as closed:
        yield {"type": "closed", "code": _close_code(closed)}


async def _session(
    ws_base: str,
    path: str,
    query: str,
    key: str | None,
    frames: tuple[dict[str, JsonValue], ...],
    *,
    until: frozenset[str],
    headers: Mapping[str, str] | None = None,
    seconds: float = 60,
) -> Session:
    request_headers: Final = {**({} if key is None else {"Authorization": f"Bearer {key}"}), **(headers or {})}
    try:
        async with websockets.connect(f"{ws_base}{path}?{query}", additional_headers=request_headers) as socket:
            return Session(tuple([event async for event in _events(socket, frames, until, seconds)]), None)
    except InvalidStatus as refusal:
        return Session((), refusal.response.status_code)


def _transcribe(
    ws_base: str,
    query: str,
    key: str | None,
    *,
    path: str = "/v1/realtime",
    update: JsonValue = GA_TRANSCRIPTION_UPDATE,
    headers: Mapping[str, str] | None = None,
) -> Session:
    frames: Final[tuple[dict[str, JsonValue], ...]] = ({"type": "session.update", "session": update}, COMMIT)
    until: Final = frozenset({TRANSCRIPT_COMPLETED})
    return asyncio.run(_session(ws_base, path, query, key, frames, until=until, headers=headers))


def _talk(
    ws_base: str,
    query: str,
    key: str,
    frames: tuple[dict[str, JsonValue], ...],
    *,
    until: str = RESPONSE_DONE,
    seconds: float = 60,
) -> Session:
    return asyncio.run(_session(ws_base, "/v1/realtime", query, key, frames, until=frozenset({until}), seconds=seconds))


def _voice_frames(*updates: dict[str, JsonValue]) -> tuple[dict[str, JsonValue], ...]:
    return (*({"type": "session.update", "session": update} for update in updates), COMMIT)


def _muse_update(turn_detection: JsonValue) -> dict[str, JsonValue]:
    return {
        "type": "transcription",
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "transcription": {"model": MUSE_MODEL},
                "turn_detection": turn_detection,
            }
        },
    }


def _muse_frames(turn_detection: JsonValue) -> tuple[dict[str, JsonValue], ...]:
    return ({"type": "session.update", "session": _muse_update(turn_detection)}, APPEND, COMMIT)


def _observed_all(upstream_url: str) -> tuple[dict[str, JsonValue], ...]:
    with httpx.Client(
        base_url=upstream_url, timeout=5, trust_env=False, verify=not upstream_url.startswith("https://")
    ) as upstream:
        return tuple(map(object_value, upstream.get("/__observations").json()["requests"]))


def _belonging(observed: tuple[dict[str, JsonValue], ...], scenario_id: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(request for request in observed if _belongs(request, scenario_id))


def _observed(upstream_url: str, scenario_id: str) -> tuple[dict[str, JsonValue], ...]:
    return _belonging(_observed_all(upstream_url), scenario_id)


def _belongs(request: dict[str, JsonValue], scenario_id: str) -> bool:
    return request["authorization"] == f"Bearer {scenario_id}" or request["api_key"] == scenario_id


def _upgrades(observed: tuple[dict[str, JsonValue], ...]) -> tuple[dict[str, JsonValue], ...]:
    return tuple(object_value(request["body"]) for request in observed if request["method"] == "WEBSOCKET")


def _sent(observed: tuple[dict[str, JsonValue], ...]) -> tuple[dict[str, JsonValue], ...]:
    return tuple(object_value(request["body"]) for request in observed if request["method"] == "WEBSOCKET_FRAME")


def _sent_types(observed: tuple[dict[str, JsonValue], ...]) -> tuple[JsonValue, ...]:
    return tuple(frame.get("type") for frame in _sent(observed))


def _session_updates(observed: tuple[dict[str, JsonValue], ...]) -> tuple[JsonValue, ...]:
    return tuple(frame.get("session") for frame in _sent(observed) if frame.get("type") == "session.update")


def _query(upgrade: dict[str, JsonValue]) -> JsonValue:
    return upgrade["query"]


def _spend_rows(key: str, count: int) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows(SPEND_SQL, (sha256(key.encode()).hexdigest(),)),
        lambda rows: len(rows) == count,
        seconds=70,
    )


def _content_filter(name: str, mode: str, *, default_on: bool) -> dict[str, JsonValue]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "litellm_content_filter",
            "mode": mode,
            "default_on": default_on,
            "blocked_words": [{"keyword": BLOCKED_WORD, "action": "BLOCK"}],
        },
    }


def _write_config(directory: Path, ca_bundle: Path) -> Path:
    config: Final = directory / f"realtime_guardrails_{uuid.uuid4().hex[:8]}.yaml"
    config.write_text(
        json.dumps(
            {
                "guardrails": [
                    _content_filter(TRANSCRIPT_GUARDRAIL, "realtime_input_transcription", default_on=True),
                    _content_filter(OPTIN_GUARDRAIL, "realtime_input_transcription", default_on=False),
                    _content_filter(PROMPT_GUARDRAIL, "pre_call", default_on=True),
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


def _overrides(ca_bundle: Path) -> dict[str, str]:
    return {"DATABASE_URL": os.environ["DATABASE_URL"], "SSL_VERIFY": str(ca_bundle)}


def _ca_bundle(slot: UpstreamSlot) -> Path:
    assert slot.certificate is not None, "the TLS upstream carries its own certificate"
    return slot.certificate.certificate


@pytest.fixture(scope="module")
def tls_slot(tmp_path_factory: pytest.TempPathFactory) -> Iterator[UpstreamSlot]:
    with owned_upstream(tmp_path_factory.mktemp("tls-upstream"), tls=True) as slot:
        yield slot


@pytest.fixture(scope="module")
def tls_upstream(tls_slot: UpstreamSlot) -> str:
    return tls_slot.url


@pytest.fixture(scope="module")
def guardrail_proxy(tmp_path_factory: pytest.TempPathFactory, tls_slot: UpstreamSlot) -> Iterator[OwnedProxy]:
    directory: Final = tmp_path_factory.mktemp("realtime-guardrail-proxy")
    ca_bundle: Final = _ca_bundle(tls_slot)
    with (
        gateway_from_environment() as rig,
        owned_proxy_process(
            rig, directory, _overrides(ca_bundle), config=_write_config(directory, ca_bundle), workers=WORKERS
        ) as owned,
    ):
        yield owned


def _assert_transcription_left_alone(
    session: Session, observed: tuple[dict[str, JsonValue], ...], transcript: str, *, update: JsonValue
) -> None:
    assert session.types == ("session.created", "session.updated", TRANSCRIPT_COMPLETED), session
    assert session.session_type == TRANSCRIPTION, session
    assert session.transcripts == (transcript,), session
    assert _sent_types(observed) == ("session.update", "input_audio_buffer.commit"), _sent(observed)
    assert _session_updates(observed) == (update,), _session_updates(observed)


@pytest.mark.parametrize("path", ["/v1/realtime", "/realtime", "/openai/v1/realtime"])
def test_transcription_session_update_reaches_the_upstream_verbatim(guardrail_proxy: OwnedProxy, path: str) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _transcription_scenario(CLEAN_TRANSCRIPT))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        session: Final = _transcribe(
            _ws_base(_owned_url(guardrail_proxy)), f"model={model}&{TRANSCRIPTION_QUERY}", key, path=path
        )
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        _assert_transcription_left_alone(session, observed, CLEAN_TRANSCRIPT, update=GA_TRANSCRIPTION_UPDATE)
        assert [_query(upgrade) for upgrade in _upgrades(observed)] == [
            [["model", TRANSCRIBE_MODEL], ["intent", TRANSCRIPTION]]
        ]
        rows: Final = _spend_rows(key, 1)
        assert rows[0]["call_type"] == "_arealtime", rows


async def _sdk_async_events(connection: AsyncRealtimeConnection) -> AsyncIterator[dict[str, JsonValue]]:
    async for event in connection:
        yield JSON_OBJECT.validate_python(event.model_dump())
        if event.type == TRANSCRIPT_COMPLETED:
            return


def _sdk_sync_events(connection: RealtimeConnection) -> Iterator[dict[str, JsonValue]]:
    for event in connection:
        yield JSON_OBJECT.validate_python(event.model_dump())
        if event.type == TRANSCRIPT_COMPLETED:
            return


async def _sdk_async_transcription(proxy_url: str, key: str, model: str) -> Session:
    client: Final = AsyncOpenAI(api_key=key, base_url=f"{proxy_url}/v1", websocket_base_url=f"{_ws_base(proxy_url)}/v1")
    async with client.realtime.connect(model=model, extra_query={"intent": TRANSCRIPTION}) as connection:
        await connection.session.update(session=SDK_TRANSCRIPTION_SESSION)
        await connection.input_audio_buffer.commit()
        return Session(tuple([event async for event in _sdk_async_events(connection)]), None)


def _sdk_sync_transcription(proxy_url: str, key: str, model: str) -> Session:
    client: Final = OpenAI(api_key=key, base_url=f"{proxy_url}/v1", websocket_base_url=f"{_ws_base(proxy_url)}/v1")
    with client.realtime.connect(model=model, extra_query={"intent": TRANSCRIPTION}) as connection:
        connection.session.update(session=SDK_TRANSCRIPTION_SESSION)
        connection.input_audio_buffer.commit()
        return Session(tuple(_sdk_sync_events(connection)), None)


@pytest.mark.parametrize("client", ["async", "sync"])
def test_openai_sdk_transcription_session_gets_its_transcript(guardrail_proxy: OwnedProxy, client: str) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _transcription_scenario(CLEAN_TRANSCRIPT))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        session: Final = (
            asyncio.run(_sdk_async_transcription(_owned_url(guardrail_proxy), key, model))
            if client == "async"
            else _sdk_sync_transcription(_owned_url(guardrail_proxy), key, model)
        )
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        assert session.types == ("session.created", "session.updated", TRANSCRIPT_COMPLETED), session
        assert session.transcripts == (CLEAN_TRANSCRIPT,), session
        assert _sent_types(observed) == ("session.update", "input_audio_buffer.commit"), _sent(observed)
        assert _session_updates(observed) == (GA_TRANSCRIPTION_UPDATE,), _session_updates(observed)


def test_beta_protocol_transcription_session_update_reaches_the_upstream_verbatim(guardrail_proxy: OwnedProxy) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _transcription_scenario(CLEAN_TRANSCRIPT))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        session: Final = _transcribe(
            _ws_base(_owned_url(guardrail_proxy)),
            f"model={model}&{TRANSCRIPTION_QUERY}",
            key,
            update=BETA_TRANSCRIPTION_UPDATE,
            headers=BETA_HEADERS,
        )
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        assert session.transcripts == (CLEAN_TRANSCRIPT,), session
        assert session.errors == (), session
        assert _sent_types(observed) == ("session.update", "input_audio_buffer.commit"), _sent(observed)
        assert _session_updates(observed) == (BETA_TRANSCRIPTION_UPDATE,), _session_updates(observed)


def test_intent_without_model_routes_to_the_whisper_default_without_an_injected_update(
    guardrail_proxy: OwnedProxy,
) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _transcription_scenario(CLEAN_TRANSCRIPT))
        key: Final = scenario.key()
        _named_deployment(
            guardrail_proxy.gateway, scenario, WHISPER_DEFAULT, f"openai/{WHISPER_DEFAULT}", handle.scenario_id
        )
        session: Final = _transcribe(_ws_base(_owned_url(guardrail_proxy)), TRANSCRIPTION_QUERY, key)
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        forwarded: Final = _with_transcription_model(GA_TRANSCRIPTION_UPDATE, WHISPER_DEFAULT)
        _assert_transcription_left_alone(session, observed, CLEAN_TRANSCRIPT, update=forwarded)
        assert [_query(upgrade) for upgrade in _upgrades(observed)] == [[["intent", TRANSCRIPTION]]]


def test_azure_transcription_session_update_reaches_the_upstream_verbatim(
    guardrail_proxy: OwnedProxy, tls_upstream: str
) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _transcription_scenario(CLEAN_TRANSCRIPT), control_url=tls_upstream)
        key: Final = scenario.key()
        model: Final = _azure_deployment(scenario, handle.scenario_id, tls_upstream)
        session: Final = _transcribe(_ws_base(_owned_url(guardrail_proxy)), f"model={model}&{TRANSCRIPTION_QUERY}", key)
        observed: Final = _observed(tls_upstream, handle.scenario_id)
        _assert_transcription_left_alone(session, observed, CLEAN_TRANSCRIPT, update=GA_TRANSCRIPTION_UPDATE)
        upgrades: Final = tuple(request for request in observed if request["method"] == "WEBSOCKET")
        assert [upgrade["path"] for upgrade in upgrades] == ["/openai/v1/realtime"], upgrades
        assert [_query(object_value(upgrade["body"])) for upgrade in upgrades] == [[["intent", TRANSCRIPTION]]], (
            upgrades
        )


def test_voice_session_keeps_the_injected_update_and_the_clean_transcript_flows(guardrail_proxy: OwnedProxy) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _voice_scenario(CLEAN_TRANSCRIPT))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id, model=VOICE_MODEL)
        session: Final = _talk(
            _ws_base(_owned_url(guardrail_proxy)), f"model={model}", key, _voice_frames(VOICE_UPDATE)
        )
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        assert session.types == (
            "session.created",
            "session.updated",
            "error",
            TRANSCRIPT_COMPLETED,
            RESPONSE_DONE,
        ), session
        assert session.errors == (MISSING_TURN_DETECTION_TYPE,), session
        assert session.error_messages == (TURN_DETECTION_TYPE_MISSING,), session
        assert _sent_types(observed) == (
            "session.update",
            "session.update",
            "input_audio_buffer.commit",
            "response.create",
        ), _sent(observed)
        assert _session_updates(observed) == (GA_INJECTED_UPDATE, VOICE_UPDATE_FORWARDED), _session_updates(observed)
        rows: Final = _spend_rows(key, 1)
        assert rows[0]["call_type"] == "_arealtime", rows


def test_voice_session_blocked_transcript_is_refused_through_the_backend(guardrail_proxy: OwnedProxy) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _voice_scenario(BLOCKED_TRANSCRIPT))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id, model=VOICE_MODEL)
        session: Final = _talk(
            _ws_base(_owned_url(guardrail_proxy)), f"model={model}", key, _voice_frames(VOICE_UPDATE)
        )
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        assert session.transcripts == (BLOCKED_TRANSCRIPT,), session
        assert session.errors == (MISSING_TURN_DETECTION_TYPE, GUARDRAIL_VIOLATION), session
        assert session.types[-1] == RESPONSE_DONE, session
        assert _sent_types(observed) == (
            "session.update",
            "session.update",
            "input_audio_buffer.commit",
            "response.cancel",
            "conversation.item.create",
            "response.create",
        ), _sent(observed)
        assert _session_updates(observed) == (GA_INJECTED_UPDATE, VOICE_UPDATE_FORWARDED), _session_updates(observed)


def test_voice_session_cannot_re_enable_auto_response_with_a_later_update(guardrail_proxy: OwnedProxy) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _voice_scenario(CLEAN_TRANSCRIPT))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id, model=VOICE_MODEL)
        frames: Final = _voice_frames(VOICE_UPDATE, RE_ENABLE_UPDATE)
        session: Final = _talk(_ws_base(_owned_url(guardrail_proxy)), f"model={model}", key, frames)
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        assert session.types[-1] == RESPONSE_DONE, session
        assert session.errors == (MISSING_TURN_DETECTION_TYPE,), session
        assert _session_updates(observed) == (
            GA_INJECTED_UPDATE,
            VOICE_UPDATE_FORWARDED,
            RE_ENABLE_FORWARDED,
        ), _session_updates(observed)


def test_client_declared_transcription_type_does_not_bypass_the_voice_guardrail(guardrail_proxy: OwnedProxy) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _voice_scenario(BLOCKED_TRANSCRIPT))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id, model=VOICE_MODEL)
        frames: Final = _voice_frames(CLIENT_DECLARED_TRANSCRIPTION)
        session: Final = _talk(_ws_base(_owned_url(guardrail_proxy)), f"model={model}", key, frames, seconds=20)
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        assert session.transcripts == (BLOCKED_TRANSCRIPT,), session
        assert session.errors == (MISSING_TURN_DETECTION_TYPE, GUARDRAIL_VIOLATION), session
        assert session.error_messages[0] == TURN_DETECTION_TYPE_MISSING, session
        assert session.types[-1] == RESPONSE_DONE, session
        assert _sent_types(observed) == (
            "session.update",
            "session.update",
            "input_audio_buffer.commit",
            "response.cancel",
            "conversation.item.create",
            "response.create",
        ), _sent(observed)
        assert _session_updates(observed) == (
            GA_INJECTED_UPDATE,
            CLIENT_DECLARED_TRANSCRIPTION_FORWARDED,
        ), _session_updates(observed)


def test_backend_session_created_typed_transcription_skips_the_injection_on_the_raw_path(
    guardrail_proxy: OwnedProxy,
) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _transcription_scenario(CLEAN_TRANSCRIPT))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        session: Final = _transcribe(_ws_base(_owned_url(guardrail_proxy)), f"model={model}", key)
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        _assert_transcription_left_alone(session, observed, CLEAN_TRANSCRIPT, update=GA_TRANSCRIPTION_UPDATE)
        assert [_query(upgrade) for upgrade in _upgrades(observed)] == [[["model", TRANSCRIBE_MODEL]]]


def _muse_handshake(observed: tuple[dict[str, JsonValue], ...]) -> dict[str, JsonValue]:
    upgrades: Final = _upgrades(observed)
    assert len(upgrades) == 1, upgrades
    return upgrades[0]


def _muse_audio_frames(observed: tuple[dict[str, JsonValue], ...]) -> tuple[dict[str, JsonValue], ...]:
    return _sent(observed)


def _muse_session(
    guardrail_proxy: OwnedProxy,
    scenario: Scenario,
    tls_upstream: str,
    transcript: str,
    query: str,
    turn_detection: JsonValue,
) -> tuple[Session, tuple[dict[str, JsonValue], ...]]:
    handle: Final = _scripted(scenario, _muse_scenario(transcript), control_url=tls_upstream)
    key: Final = scenario.key()
    model: Final = _muse_deployment(scenario, handle.scenario_id, tls_upstream)
    until: Final = TRANSCRIPT_COMPLETED if BLOCKED_WORD not in transcript else "error"
    session: Final = _talk(
        _ws_base(_owned_url(guardrail_proxy)), f"model={model}{query}", key, _muse_frames(turn_detection), until=until
    )
    return session, _observed(tls_upstream, handle.scenario_id)


def test_muse_push_to_talk_transcription_session_keeps_push_to_talk(
    guardrail_proxy: OwnedProxy, tls_upstream: str
) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        session, observed = _muse_session(
            guardrail_proxy, scenario, tls_upstream, CLEAN_TRANSCRIPT, f"&{TRANSCRIPTION_QUERY}", None
        )
        assert session.session_type == TRANSCRIPTION, session
        assert session.transcripts == (CLEAN_TRANSCRIPT,), session
        assert session.errors == (), session
        assert _muse_handshake(observed)["mode"] == PUSH_TO_TALK, _muse_handshake(observed)
        assert _muse_audio_frames(observed) == (
            {"binary_bytes": MUSE_PACKET_BYTES},
            {"binary_bytes": MUSE_PACKET_BYTES},
            {"binary_bytes": MUSE_REMAINDER_BYTES},
            END_STREAM,
        ), _muse_audio_frames(observed)


def test_muse_transcription_session_blocked_transcript_reaches_the_client_as_a_violation(
    guardrail_proxy: OwnedProxy, tls_upstream: str
) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        session, observed = _muse_session(
            guardrail_proxy, scenario, tls_upstream, BLOCKED_TRANSCRIPT, f"&{TRANSCRIPTION_QUERY}", None
        )
        assert session.transcripts == (BLOCKED_TRANSCRIPT,), session
        assert session.errors == (GUARDRAIL_VIOLATION,), session
        assert _muse_handshake(observed)["mode"] == PUSH_TO_TALK, _muse_handshake(observed)
        assert _muse_audio_frames(observed)[-1] == END_STREAM, _muse_audio_frames(observed)


def test_muse_server_vad_transcription_session_keeps_endpointing(
    guardrail_proxy: OwnedProxy, tls_upstream: str
) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        session, observed = _muse_session(
            guardrail_proxy, scenario, tls_upstream, CLEAN_TRANSCRIPT, f"&{TRANSCRIPTION_QUERY}", {"type": "server_vad"}
        )
        assert session.transcripts == (CLEAN_TRANSCRIPT,), session
        assert session.errors == (), session
        assert _muse_handshake(observed)["mode"] == ENDPOINTING, _muse_handshake(observed)
        assert _muse_audio_frames(observed) == (
            {"binary_bytes": MUSE_PACKET_BYTES},
            {"binary_bytes": MUSE_PACKET_BYTES},
            {"binary_bytes": MUSE_REMAINDER_BYTES},
        ), _muse_audio_frames(observed)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "a push-to-talk Muse session reached by model alone is forced to ENDPOINTING: the first-update injection "
        "runs before the backend session event flags the session as transcription-only"
    ),
)
def test_muse_push_to_talk_session_without_intent_keeps_push_to_talk(
    guardrail_proxy: OwnedProxy, tls_upstream: str, record_property: RecordProperty
) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        session, observed = _muse_session(guardrail_proxy, scenario, tls_upstream, BLOCKED_TRANSCRIPT, "", None)
        record_property("muse_handshake_mode_without_intent", _muse_handshake(observed)["mode"])
        assert session.transcripts == (BLOCKED_TRANSCRIPT,), session
        assert session.errors == (GUARDRAIL_VIOLATION,), session
        assert _muse_handshake(observed)["mode"] == PUSH_TO_TALK, _muse_handshake(observed)


def test_muse_session_without_intent_still_delivers_the_transcript_and_the_verdict(
    guardrail_proxy: OwnedProxy, tls_upstream: str, record_property: RecordProperty
) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        session, observed = _muse_session(guardrail_proxy, scenario, tls_upstream, BLOCKED_TRANSCRIPT, "", None)
        record_property("muse_handshake_mode_without_intent", _muse_handshake(observed)["mode"])
        assert session.session_type == TRANSCRIPTION, session
        assert session.transcripts == (BLOCKED_TRANSCRIPT,), session
        assert session.errors == (GUARDRAIL_VIOLATION,), session
        assert _muse_handshake(observed)["model"] == MUSE_MODEL, _muse_handshake(observed)


def test_without_a_guardrail_the_proxy_sends_no_session_update_of_its_own(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        transcription: Final = _scripted(scenario, _transcription_scenario(BLOCKED_TRANSCRIPT))
        voice: Final = _scripted(scenario, _voice_scenario(BLOCKED_TRANSCRIPT))
        key: Final = scenario.key()
        transcribe_model: Final = _openai_deployment(scenario, transcription.scenario_id)
        voice_model: Final = _openai_deployment(scenario, voice.scenario_id, model=VOICE_MODEL)
        transcribed: Final = _transcribe(_ws_base(_proxy_url()), f"model={transcribe_model}&{TRANSCRIPTION_QUERY}", key)
        talked: Final = _talk(_ws_base(_proxy_url()), f"model={voice_model}", key, _voice_frames(VOICE_UPDATE))
        observed: Final = _observed_all(gateway.upstream_url)
        transcription_observed: Final = _belonging(observed, transcription.scenario_id)
        voice_observed: Final = _belonging(observed, voice.scenario_id)
        _assert_transcription_left_alone(
            transcribed, transcription_observed, BLOCKED_TRANSCRIPT, update=GA_TRANSCRIPTION_UPDATE
        )
        assert talked.types == ("session.created", "session.updated", TRANSCRIPT_COMPLETED, RESPONSE_DONE), talked
        assert talked.errors == (), talked
        assert _session_updates(voice_observed) == (VOICE_UPDATE,), _session_updates(voice_observed)


def _assert_voice_session_ungated(session: Session, observed: tuple[dict[str, JsonValue], ...]) -> None:
    assert session.types == ("session.created", "session.updated", TRANSCRIPT_COMPLETED, RESPONSE_DONE), session
    assert session.transcripts == (BLOCKED_TRANSCRIPT,), session
    assert session.errors == (), session
    assert _session_updates(observed) == (VOICE_UPDATE,), _session_updates(observed)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the realtime route hands the transcript guardrail only the request's guardrails list, so the key and "
        "team metadata that opts out of a default-on guardrail never reaches it"
    ),
)
def test_key_opted_out_of_the_transcript_guardrail_is_not_gated_by_the_prompt_guardrail(
    guardrail_proxy: OwnedProxy,
) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _voice_scenario(BLOCKED_TRANSCRIPT))
        key: Final = scenario.key(metadata={"opted_out_global_guardrails": [TRANSCRIPT_GUARDRAIL]})
        model: Final = _openai_deployment(scenario, handle.scenario_id, model=VOICE_MODEL)
        session: Final = _talk(
            _ws_base(_owned_url(guardrail_proxy)), f"model={model}", key, _voice_frames(VOICE_UPDATE)
        )
        _assert_voice_session_ungated(session, _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the realtime route hands the transcript guardrail only the request's guardrails list, so the key and "
        "team metadata that opts out of a default-on guardrail never reaches it"
    ),
)
def test_team_opted_out_of_the_transcript_guardrail_is_not_gated(guardrail_proxy: OwnedProxy) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _voice_scenario(BLOCKED_TRANSCRIPT))
        team: Final = scenario.team(metadata={"opted_out_global_guardrails": [TRANSCRIPT_GUARDRAIL]})
        key: Final = scenario.key(team_id=team)
        model: Final = _openai_deployment(scenario, handle.scenario_id, model=VOICE_MODEL)
        session: Final = _talk(
            _ws_base(_owned_url(guardrail_proxy)), f"model={model}", key, _voice_frames(VOICE_UPDATE)
        )
        _assert_voice_session_ungated(session, _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id))


def test_opt_in_guardrail_gates_a_voice_session_on_an_opted_out_key(guardrail_proxy: OwnedProxy) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _voice_scenario(BLOCKED_TRANSCRIPT))
        key: Final = scenario.key(metadata={"opted_out_global_guardrails": [TRANSCRIPT_GUARDRAIL]})
        model: Final = _openai_deployment(scenario, handle.scenario_id, model=VOICE_MODEL)
        session: Final = _talk(
            _ws_base(_owned_url(guardrail_proxy)), f"model={model}&{OPTIN_QUERY}", key, _voice_frames(VOICE_UPDATE)
        )
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        assert session.transcripts == (BLOCKED_TRANSCRIPT,), session
        assert session.errors == (MISSING_TURN_DETECTION_TYPE, GUARDRAIL_VIOLATION), session
        assert _session_updates(observed) == (GA_INJECTED_UPDATE, VOICE_UPDATE_FORWARDED), _session_updates(observed)


def test_opt_in_guardrail_leaves_a_transcription_session_alone_on_an_opted_out_key(guardrail_proxy: OwnedProxy) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _transcription_scenario(CLEAN_TRANSCRIPT))
        key: Final = scenario.key(metadata={"opted_out_global_guardrails": [TRANSCRIPT_GUARDRAIL]})
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        session: Final = _transcribe(
            _ws_base(_owned_url(guardrail_proxy)), f"model={model}&{TRANSCRIPTION_QUERY}&{OPTIN_QUERY}", key
        )
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        _assert_transcription_left_alone(session, observed, CLEAN_TRANSCRIPT, update=GA_TRANSCRIPTION_UPDATE)


def test_guardrails_list_names_the_three_configured_guardrails(guardrail_proxy: OwnedProxy) -> None:
    listed: Final = guardrail_proxy.gateway.get("/guardrails/list")
    names: Final = {string_value(object_value(entry)["guardrail_name"]) for entry in _list(listed["guardrails"])}
    assert names == {TRANSCRIPT_GUARDRAIL, OPTIN_GUARDRAIL, PROMPT_GUARDRAIL}, listed


def _list(value: JsonValue) -> list[JsonValue]:
    assert isinstance(value, list), value
    return value


MALFORMED_SESSIONS: Final = (
    pytest.param(5, id="integer"),
    pytest.param(["transcription"], id="list"),
    pytest.param("", id="empty"),
    pytest.param(FIVE_KB, id="five_kilobytes"),
)


@pytest.mark.parametrize("malformed", MALFORMED_SESSIONS)
def test_malformed_session_update_is_relayed_and_the_session_keeps_serving(
    guardrail_proxy: OwnedProxy, malformed: JsonValue
) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _transcription_scenario(CLEAN_TRANSCRIPT))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        frames: Final[tuple[dict[str, JsonValue], ...]] = (
            {"type": "session.update", "session": malformed},
            {"type": "session.update", "session": GA_TRANSCRIPTION_UPDATE},
            COMMIT,
        )
        session: Final = _talk(
            _ws_base(_owned_url(guardrail_proxy)),
            f"model={model}&{TRANSCRIPTION_QUERY}",
            key,
            frames,
            until=TRANSCRIPT_COMPLETED,
        )
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        assert session.types == ("session.created", "error", "session.updated", TRANSCRIPT_COMPLETED), session
        assert session.errors == (INVALID_SESSION_VALUE,), session
        assert session.transcripts == (CLEAN_TRANSCRIPT,), session
        assert _session_updates(observed) == (malformed, GA_TRANSCRIPTION_UPDATE), _session_updates(observed)


def test_session_update_sent_twice_is_echoed_twice(guardrail_proxy: OwnedProxy) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _transcription_scenario(CLEAN_TRANSCRIPT))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        frames: Final[tuple[dict[str, JsonValue], ...]] = (
            {"type": "session.update", "session": GA_TRANSCRIPTION_UPDATE},
            {"type": "session.update", "session": GA_TRANSCRIPTION_UPDATE},
            COMMIT,
        )
        session: Final = _talk(
            _ws_base(_owned_url(guardrail_proxy)),
            f"model={model}&{TRANSCRIPTION_QUERY}",
            key,
            frames,
            until=TRANSCRIPT_COMPLETED,
        )
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        assert session.types == ("session.created", "session.updated", "session.updated", TRANSCRIPT_COMPLETED), session
        assert _session_updates(observed) == (GA_TRANSCRIPTION_UPDATE, GA_TRANSCRIPTION_UPDATE), _session_updates(
            observed
        )


def test_duplicate_backend_session_created_never_injects(guardrail_proxy: OwnedProxy) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _transcription_scenario(CLEAN_TRANSCRIPT, repeats=2))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        session: Final = _transcribe(_ws_base(_owned_url(guardrail_proxy)), f"model={model}&{TRANSCRIPTION_QUERY}", key)
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        assert session.types == ("session.created", "session.created", "session.updated", TRANSCRIPT_COMPLETED), session
        assert session.transcripts == (CLEAN_TRANSCRIPT,), session
        assert _session_updates(observed) == (GA_TRANSCRIPTION_UPDATE,), _session_updates(observed)


def test_older_transcription_session_created_event_skips_the_injection(guardrail_proxy: OwnedProxy) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _older_transcription_scenario(CLEAN_TRANSCRIPT))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        session: Final = _transcribe(_ws_base(_owned_url(guardrail_proxy)), f"model={model}", key)
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        assert session.types == ("transcription_session.created", "session.updated", TRANSCRIPT_COMPLETED), session
        assert session.transcripts == (CLEAN_TRANSCRIPT,), session
        assert _session_updates(observed) == (GA_TRANSCRIPTION_UPDATE,), _session_updates(observed)


def test_unauthenticated_upgrade_is_refused_and_the_next_key_still_connects(guardrail_proxy: OwnedProxy) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _transcription_scenario(CLEAN_TRANSCRIPT))
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        refused: Final = _transcribe(
            _ws_base(_owned_url(guardrail_proxy)), f"model={model}&{TRANSCRIPTION_QUERY}", None
        )
        assert refused.refused == UNAUTHENTICATED_STATUS, refused
        assert _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id) == ()
        session: Final = _transcribe(
            _ws_base(_owned_url(guardrail_proxy)), f"model={model}&{TRANSCRIPTION_QUERY}", scenario.key()
        )
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        _assert_transcription_left_alone(session, observed, CLEAN_TRANSCRIPT, update=GA_TRANSCRIPTION_UPDATE)


def test_unknown_model_is_rejected_before_any_upstream_upgrade(guardrail_proxy: OwnedProxy) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _transcription_scenario(CLEAN_TRANSCRIPT))
        key: Final = scenario.key()
        missing: Final = f"realtime-guard-missing-{uuid.uuid4().hex[:12]}"
        rejected: Final = _transcribe(
            _ws_base(_owned_url(guardrail_proxy)), f"model={missing}&{TRANSCRIPTION_QUERY}", key
        )
        assert rejected.types == ("error", "closed"), rejected
        assert rejected.close_code == UNKNOWN_MODEL_CLOSE, rejected
        assert "Invalid model" in rejected.error_messages[0], rejected
        assert _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id) == ()


def test_repeated_transcription_sessions_write_one_spend_row_each(guardrail_proxy: OwnedProxy) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _transcription_scenario(CLEAN_TRANSCRIPT))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        sessions: Final = tuple(
            _transcribe(_ws_base(_owned_url(guardrail_proxy)), f"model={model}&{TRANSCRIPTION_QUERY}", key)
            for _ in range(3)
        )
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        assert [session.transcripts for session in sessions] == [(CLEAN_TRANSCRIPT,)] * 3, sessions
        assert _session_updates(observed) == (GA_TRANSCRIPTION_UPDATE,) * 3, _session_updates(observed)
        rows: Final = _spend_rows(key, 3)
        assert {str(row["call_type"]) for row in rows} == {"_arealtime"}, rows


async def _hold_until_closed(ws_base: str, query: str, key: str, opened: asyncio.Queue[str]) -> Session:
    async with websockets.connect(
        f"{ws_base}/v1/realtime?{query}", additional_headers={"Authorization": f"Bearer {key}"}
    ) as socket:
        created: Final = JSON_OBJECT.validate_json(await socket.recv())
        assert created.get("type") == "session.created", created
        await opened.put(string_value(object_value(created["session"])["id"]))
        return Session(tuple([frame async for frame in _frames_until_closed(socket)]), None)


async def _frames_until_closed(socket: ClientConnection) -> AsyncIterator[dict[str, JsonValue]]:
    try:
        async for message in socket:
            yield JSON_OBJECT.validate_json(message)
    except ConnectionClosed as closed:
        yield {"type": "closed", "code": _close_code(closed)}


def _relays_the_upstream_close(session: Session) -> bool:
    return f"upstream websocket closed with code {session.close_code}" in session.error_messages[0]


async def _drain(opened: asyncio.Queue[str], count: int) -> tuple[str, ...]:
    return tuple([await opened.get() for _ in range(count)])


async def _burst_through_outage(
    ws_base: str, proxy_url: str, model: str, key: str, stop_upstream: Callable[[], None]
) -> tuple[Session, ...]:
    opened: Final[asyncio.Queue[str]] = asyncio.Queue()
    query: Final = f"model={model}&{TRANSCRIPTION_QUERY}"
    holders: Final = tuple(asyncio.ensure_future(_hold_until_closed(ws_base, query, key, opened)) for _ in range(BURST))
    opened_sessions: Final = await asyncio.wait_for(_drain(opened, BURST), 60)
    assert len(opened_sessions) == BURST, opened_sessions
    await asyncio.to_thread(stop_upstream)
    async with httpx.AsyncClient(base_url=proxy_url, timeout=15, trust_env=False) as client:
        liveliness: Final = await client.get("/health/liveliness")
    assert liveliness.status_code == 200, liveliness.text
    return tuple(await asyncio.wait_for(asyncio.gather(*holders), 90))


@pytest.mark.timeout(240)
def test_upstream_outage_closes_every_open_transcription_session_and_recovers_without_an_injection(
    guardrail_proxy: OwnedProxy, tmp_path: Path, record_property: RecordProperty
) -> None:
    with guardrail_proxy.gateway.scenario() as scenario, owned_upstream(tmp_path) as slot:
        scenario_id: Final = f"realtime-guard-outage-{uuid.uuid4().hex[:12]}"
        register_scenario(scenario_id, _transcription_scenario(CLEAN_TRANSCRIPT), control_url=slot.url)
        key: Final = scenario.key()
        model: Final = scenario.model(model=f"openai/{TRANSCRIBE_MODEL}", api_key=scenario_id, api_base=slot.url)
        proxy_url: Final = _owned_url(guardrail_proxy)
        held: Final = asyncio.run(_burst_through_outage(_ws_base(proxy_url), proxy_url, model, key, slot.stop))
        record_property("close_codes_during_upstream_outage", sorted(session.close_code or 0 for session in held))
        assert [session.types for session in held] == [("error", "closed")] * BURST, held
        assert all(_relays_the_upstream_close(session) for session in held), held
        assert len({session.close_code for session in held}) == 1, held
        slot.start()
        register_scenario(scenario_id, _transcription_scenario(CLEAN_TRANSCRIPT), control_url=slot.url)
        recovered: Final = _transcribe(_ws_base(proxy_url), f"model={model}&{TRANSCRIPTION_QUERY}", key)
        _assert_transcription_left_alone(
            recovered, _observed(slot.url, scenario_id), CLEAN_TRANSCRIPT, update=GA_TRANSCRIPTION_UPDATE
        )
        rows: Final = _spend_rows(key, BURST + 1)
        assert {str(row["call_type"]) for row in rows} == {"_arealtime"}, rows


def _spawned_worker(child: psutil.Process) -> bool:
    try:
        return any("multiprocessing.spawn" in part for part in child.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def _workers(root: psutil.Process) -> tuple[psutil.Process, ...]:
    return tuple(
        sorted((child for child in root.children() if _spawned_worker(child)), key=lambda process: process.pid)
    )


@dataclass(frozen=True, slots=True)
class KillOutcome:
    served: int
    closed: int
    killed_pid: int


async def _one_transcript_or_close(socket: ClientConnection) -> bool:
    try:
        await socket.send(json.dumps({"type": "session.update", "session": GA_TRANSCRIPTION_UPDATE}))
        await socket.send(json.dumps(COMMIT))
        async for message in socket:
            if JSON_OBJECT.validate_json(message).get("type") == TRANSCRIPT_COMPLETED:
                return True
    except ConnectionClosed:
        return False
    raise AssertionError("session ended without a transcript or a close frame")


async def _first_frames(sockets: tuple[ClientConnection, ...]) -> tuple[dict[str, JsonValue], ...]:
    return tuple([JSON_OBJECT.validate_json(await socket.recv()) for socket in sockets])


async def _open_sessions(ws_base: str, model: str, key: str) -> tuple[ClientConnection, ...]:
    query: Final = f"model={model}&{TRANSCRIPTION_QUERY}"
    headers: Final = {"Authorization": f"Bearer {key}"}
    sockets: Final = tuple(
        [
            await websockets.connect(f"{ws_base}/v1/realtime?{query}", additional_headers=headers)
            for _ in range(OPEN_SESSIONS)
        ]
    )
    created: Final = await asyncio.wait_for(_first_frames(sockets), 60)
    assert [event.get("type") for event in created] == ["session.created"] * OPEN_SESSIONS, created
    return sockets


async def _sessions_through_worker_kill(ws_base: str, model: str, key: str, root: psutil.Process) -> KillOutcome:
    sockets: Final = await _open_sessions(ws_base, model, key)
    try:
        workers: Final = _workers(root)
        assert len(workers) == WORKERS, [process.pid for process in workers]
        victim: Final = workers[0]
        victim.kill()
        await asyncio.to_thread(victim.wait, 10)
        served: Final = await asyncio.wait_for(
            asyncio.gather(*(_one_transcript_or_close(socket) for socket in sockets)), 60
        )
        return KillOutcome(served.count(True), served.count(False), victim.pid)
    finally:
        await asyncio.gather(*(socket.close() for socket in sockets))


async def _await_closes(sockets: tuple[ClientConnection, ...]) -> tuple[int, ...]:
    async def one(socket: ClientConnection) -> int:
        try:
            unexpected: Final = await socket.recv()
        except ConnectionClosed as closed:
            return _close_code(closed)
        raise AssertionError(f"the proxy shutdown did not close the session: {unexpected!r}")

    return tuple(await asyncio.wait_for(asyncio.gather(*(one(socket) for socket in sockets)), 90))


async def _sessions_through_proxy_shutdown(
    ws_base: str, model: str, key: str, shutdown: Callable[[], bool]
) -> tuple[int, ...]:
    sockets: Final = await _open_sessions(ws_base, model, key)
    stopped: Final = await asyncio.to_thread(shutdown)
    assert stopped, "the owned proxy root did not stop within the graceful window"
    return await _await_closes(sockets)


@pytest.mark.timeout(480)
def test_worker_kill_then_proxy_restart_keep_transcription_sessions_uninjected(
    gateway: Gateway, tls_slot: UpstreamSlot, tmp_path: Path, record_property: RecordProperty
) -> None:
    ca_bundle: Final = _ca_bundle(tls_slot)
    config: Final = _write_config(tmp_path, ca_bundle)
    with gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _transcription_scenario(CLEAN_TRANSCRIPT))
        key: Final = scenario.key()
        with owned_proxy_process(gateway, tmp_path, _overrides(ca_bundle), config=config, workers=WORKERS) as owned:
            owned_url: Final = _owned_url(owned)
            model: Final = _named_deployment(
                owned.gateway,
                scenario,
                f"realtime-guard-owned-{uuid.uuid4().hex[:8]}",
                f"openai/{TRANSCRIBE_MODEL}",
                handle.scenario_id,
            )
            root: Final = psutil.Process(owned.process.pid)
            outcome: Final = asyncio.run(_sessions_through_worker_kill(_ws_base(owned_url), model, key, root))
            record_property(
                "worker_kill", {"served": outcome.served, "closed": outcome.closed, "killed": outcome.killed_pid}
            )
            assert outcome.served + outcome.closed == OPEN_SESSIONS, outcome
            with httpx.Client(base_url=owned_url, timeout=15, trust_env=False) as fresh:
                readiness: Final = fresh.get("/health/readiness")
            assert readiness.status_code == 200, readiness.text
            respawned: Final = eventually(
                lambda: tuple(process.pid for process in _workers(root)),
                lambda pids: len(pids) == WORKERS and outcome.killed_pid not in pids,
                seconds=60,
            )
            record_property("worker_pids_after_respawn", respawned)
            before_the_kill: Final = _observed(gateway.upstream_url, handle.scenario_id)
            assert set(_sent_types(before_the_kill)) <= {"session.update", "input_audio_buffer.commit"}, before_the_kill
            assert set(map(json.dumps, _session_updates(before_the_kill))) == {json.dumps(GA_TRANSCRIPTION_UPDATE)}
            after_kill: Final = _transcribe(_ws_base(owned_url), f"model={model}&{TRANSCRIPTION_QUERY}", key)
            _assert_transcription_left_alone(
                after_kill,
                _observed(gateway.upstream_url, handle.scenario_id),
                CLEAN_TRANSCRIPT,
                update=GA_TRANSCRIPTION_UPDATE,
            )
            codes: Final = asyncio.run(
                _sessions_through_proxy_shutdown(
                    _ws_base(owned_url), model, key, lambda: stop_root_process(owned.process)
                )
            )
            record_property("close_codes_during_proxy_shutdown", sorted(codes))
            assert len(codes) == OPEN_SESSIONS, codes
        with owned_proxy_process(gateway, tmp_path, _overrides(ca_bundle), config=config, workers=WORKERS) as restarted:
            recovered: Final = _transcribe(_ws_base(_owned_url(restarted)), f"model={model}&{TRANSCRIPTION_QUERY}", key)
            _assert_transcription_left_alone(
                recovered,
                _observed(gateway.upstream_url, handle.scenario_id),
                CLEAN_TRANSCRIPT,
                update=GA_TRANSCRIPTION_UPDATE,
            )
