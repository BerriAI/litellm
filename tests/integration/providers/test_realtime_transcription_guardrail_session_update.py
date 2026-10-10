"""Transcript guardrails leave transcription-only realtime sessions alone.

With a ``realtime_input_transcription`` guardrail configured, the proxy disables the provider's VAD auto-response
by injecting ``turn_detection.create_response: false`` into ``session.update`` frames so the guardrail can gate
every assistant turn. A transcription session has no assistant turn and the vendors reject those updates, so the
injection is skipped when the route intent or a backend session event says the session is transcription-only,
while a client frame alone never flips a voice session into one. Every row runs against the scripted upstream,
which answers the injected and rewritten updates the way the vendors do.

A blocked transcript on a transcription session reaches the client as a ``guardrail_violation`` error and
nothing else: the ``response.cancel``, the voiced block prompt and the ``response.create`` a voice session gets
never go to a backend that cannot speak, while ``on_violation: end_session`` and ``end_session_after_n_fails``
still close the session. A guardrail that raises anything but a block closes the session the way it does on a
voice session.
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
SESSION_TYPE_MISMATCH: Final = ("invalid_request_error", "invalid_parameter")
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
VOICE_UPDATE_GATED: Final[dict[str, JsonValue]] = {
    **VOICE_UPDATE,
    "audio": {"input": {"turn_detection": {"type": "server_vad", "create_response": False}}},
}
# The client's first update races the proxy's own gate, so it is gated as a first update or forwarded as a later one
VOICE_UPDATE_FORWARDINGS: Final = (VOICE_UPDATE, VOICE_UPDATE_GATED)
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
    "audio": {"input": {"turn_detection": {"type": "server_vad", "create_response": False}}},
}
REALTIME_UPDATE_REFUSED: Final = "Passing a transcription session update to a realtime session is not allowed."
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
RESPONSE_CREATE: Final[dict[str, JsonValue]] = {"type": "response.create"}
TRANSCRIPTION_UPDATE_FRAME: Final[dict[str, JsonValue]] = {"type": "session.update", "session": GA_TRANSCRIPTION_UPDATE}
VOICE_UPDATE_FRAME: Final[dict[str, JsonValue]] = {"type": "session.update", "session": VOICE_UPDATE}
REALTIME_HOOK: Final = "realtime_input_transcription"
ENDER_GUARDRAIL: Final = "transcript-ender"
ENDER_WORD: Final = "anchovy"
ENDER_MESSAGE: Final = "The session was ended by the transcript policy."
TWO_STRIKES_GUARDRAIL: Final = "transcript-two-strikes"
TWO_STRIKES_WORD: Final = "olives"
ONE_STRIKE_GUARDRAIL: Final = "transcript-one-strike"
ONE_STRIKE_WORD: Final = "radish"
WARN_GUARDRAIL: Final = "transcript-warn"
WARN_WORD: Final = "capers"
VALUE_ERROR_GUARDRAIL: Final = "transcript-value-error"
VALUE_ERROR_WORD: Final = "durian"
RUNTIME_ERROR_GUARDRAIL: Final = "transcript-runtime-error"
RUNTIME_ERROR_WORD: Final = "lychee"
RAISING_MODULE: Final = "raising_guardrails"
CONFIGURED_GUARDRAILS: Final = frozenset(
    {
        TRANSCRIPT_GUARDRAIL,
        OPTIN_GUARDRAIL,
        PROMPT_GUARDRAIL,
        ENDER_GUARDRAIL,
        TWO_STRIKES_GUARDRAIL,
        ONE_STRIKE_GUARDRAIL,
        WARN_GUARDRAIL,
        VALUE_ERROR_GUARDRAIL,
        RUNTIME_ERROR_GUARDRAIL,
    }
)
RELAYED_CLOSE: Final = ("server_error", "")
GUARDRAIL_END_CLOSE: Final = 1000
PROXY_FAILURE_CLOSE: Final = 1011
PROXY_FAILURE_REASON: Final = "proxy failed while relaying the upstream websocket"
TOOL_OUTPUT_BLOCKED: Final = json.dumps({"error": "Tool output blocked by content policy"})
VOICE_BLOCK_FRAMES: Final = ("response.cancel", "conversation.item.create", "response.create")
CREATED_SECONDS: Final = 60.0
STEP_SECONDS: Final = 15.0
SDK_SECONDS: Final = 45.0
RAISING_GUARDRAILS_SOURCE: Final = f"""\
from litellm.integrations.custom_guardrail import CustomGuardrail


class WordRaiser(CustomGuardrail):
    word = ""
    error = Exception

    async def apply_guardrail(self, inputs, request_data, input_type, logging_obj=None):
        if any(self.word in str(text) for text in inputs["texts"]):
            raise self.error(f"{{self.word}} is not allowed")
        return inputs


class ValueErrorGuardrail(WordRaiser):
    word = "{VALUE_ERROR_WORD}"
    error = ValueError


class RuntimeErrorGuardrail(WordRaiser):
    word = "{RUNTIME_ERROR_WORD}"
    error = RuntimeError
"""


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


def _blocked(word: str) -> str:
    return f"please add {word} to the order"


def _optin(guardrail: str) -> str:
    return f"guardrails={guardrail}"


def _transcript_event(transcript: JsonValue) -> dict[str, JsonValue]:
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


def _transcript_event_without_the_field() -> dict[str, JsonValue]:
    return {key: value for key, value in _transcript_event("").items() if key != "transcript"}


def _transcription_events(events: tuple[dict[str, JsonValue], ...], *, repeats: int = 1) -> RealtimeResponse:
    return RealtimeResponse(
        content_type="application/x-realtime", events=events, session_type=TRANSCRIPTION, created_repeats=repeats
    )


def _transcription_scenario(*transcripts: JsonValue, repeats: int = 1) -> RealtimeResponse:
    return _transcription_events(tuple(_transcript_event(transcript) for transcript in transcripts), repeats=repeats)


def _older_transcription_scenario(transcript: str) -> RealtimeResponse:
    return RealtimeResponse(
        content_type="application/x-realtime",
        events=(_transcript_event(transcript),),
        session_type=TRANSCRIPTION,
        created_event="transcription_session.created",
    )


def _voice_turns(transcripts: tuple[JsonValue, ...]) -> Iterator[dict[str, JsonValue]]:
    for transcript in transcripts:
        yield _transcript_event(transcript)
        yield _done_event()


def _voice_scenario(*transcripts: JsonValue) -> RealtimeResponse:
    return RealtimeResponse(content_type="application/x-realtime", events=tuple(_voice_turns(transcripts)))


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


@dataclass(frozen=True, slots=True)
class Step:
    frames: tuple[dict[str, JsonValue], ...]
    until: str


def _step(*frames: dict[str, JsonValue], until: str) -> Step:
    return Step(frames, until)


def _update_frame(update: JsonValue) -> dict[str, JsonValue]:
    return {"type": "session.update", "session": update}


def _probe(update: JsonValue = GA_TRANSCRIPTION_UPDATE) -> Step:
    return Step((_update_frame(update),), "session.updated")


async def _until(socket: ClientConnection, until: str, seconds: float) -> AsyncIterator[dict[str, JsonValue]]:
    deadline: Final = asyncio.get_running_loop().time() + seconds
    while True:
        event: Final = await _next_event(socket, deadline)
        if event is None:
            yield {"type": "timeout"}
            return
        yield event
        if event.get("type") == until:
            return


async def _stepped(
    socket: ClientConnection, steps: tuple[Step, ...], seconds: float
) -> AsyncIterator[dict[str, JsonValue]]:
    try:
        first: Final = JSON_OBJECT.validate_json(await asyncio.wait_for(socket.recv(), CREATED_SECONDS))
        yield first
        if first.get("type") not in CREATED_TYPES:
            async for message in socket:
                yield JSON_OBJECT.validate_json(message)
            return
        for step in steps:
            for frame in step.frames:
                await socket.send(json.dumps(frame))
            async for event in _until(socket, step.until, seconds):
                yield event
                if event["type"] == "timeout":
                    return
    except ConnectionClosed as closed:
        yield {"type": "closed", "code": _close_code(closed)}


async def _driven(
    ws_base: str,
    path: str,
    query: str,
    key: str,
    steps: tuple[Step, ...],
    headers: Mapping[str, str] | None,
    seconds: float,
) -> Session:
    request_headers: Final = {"Authorization": f"Bearer {key}", **(headers or {})}
    async with websockets.connect(f"{ws_base}{path}?{query}", additional_headers=request_headers) as socket:
        return Session(tuple([event async for event in _stepped(socket, steps, seconds)]), None)


def _drive(
    ws_base: str,
    query: str,
    key: str,
    steps: tuple[Step, ...],
    *,
    path: str = "/v1/realtime",
    headers: Mapping[str, str] | None = None,
    seconds: float = STEP_SECONDS,
) -> Session:
    return asyncio.run(_driven(ws_base, path, query, key, steps, headers, seconds))


def _transcribe_blocked(
    ws_base: str,
    query: str,
    key: str,
    *,
    path: str = "/v1/realtime",
    update: JsonValue = GA_TRANSCRIPTION_UPDATE,
    headers: Mapping[str, str] | None = None,
) -> Session:
    steps: Final = (_step(_update_frame(update), COMMIT, until="error"), _probe(update))
    return _drive(ws_base, query, key, steps, path=path, headers=headers)


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


def _assert_voice_updates(observed: tuple[dict[str, JsonValue], ...], *later: dict[str, JsonValue]) -> None:
    updates: Final = _session_updates(observed)
    assert updates[0] == GA_INJECTED_UPDATE, updates
    assert updates[1] in VOICE_UPDATE_FORWARDINGS, updates
    assert updates[2:] == later, updates


def _query(upgrade: dict[str, JsonValue]) -> JsonValue:
    return upgrade["query"]


def _spend_rows(key: str, count: int) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows(SPEND_SQL, (sha256(key.encode()).hexdigest(),)),
        lambda rows: len(rows) == count,
        seconds=70,
    )


def _content_filter(
    name: str, mode: str, *, default_on: bool, word: str = BLOCKED_WORD, **settings: JsonValue
) -> dict[str, JsonValue]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "litellm_content_filter",
            "mode": mode,
            "default_on": default_on,
            "blocked_words": [{"keyword": word, "action": "BLOCK"}],
            **settings,
        },
    }


def _raising_guardrail(name: str, class_name: str) -> dict[str, JsonValue]:
    return {
        "guardrail_name": name,
        "litellm_params": {"guardrail": f"{RAISING_MODULE}.{class_name}", "mode": REALTIME_HOOK, "default_on": False},
    }


def _write_config(directory: Path, ca_bundle: Path) -> Path:
    config: Final = directory / f"realtime_guardrails_{uuid.uuid4().hex[:8]}.yaml"
    (directory / f"{RAISING_MODULE}.py").write_text(RAISING_GUARDRAILS_SOURCE)
    config.write_text(
        json.dumps(
            {
                "guardrails": [
                    _content_filter(TRANSCRIPT_GUARDRAIL, REALTIME_HOOK, default_on=True),
                    _content_filter(OPTIN_GUARDRAIL, REALTIME_HOOK, default_on=False),
                    _content_filter(PROMPT_GUARDRAIL, "pre_call", default_on=True),
                    _content_filter(
                        ENDER_GUARDRAIL,
                        REALTIME_HOOK,
                        default_on=False,
                        word=ENDER_WORD,
                        on_violation="end_session",
                        realtime_violation_message=ENDER_MESSAGE,
                    ),
                    _content_filter(
                        TWO_STRIKES_GUARDRAIL,
                        REALTIME_HOOK,
                        default_on=False,
                        word=TWO_STRIKES_WORD,
                        end_session_after_n_fails=2,
                    ),
                    _content_filter(
                        ONE_STRIKE_GUARDRAIL,
                        REALTIME_HOOK,
                        default_on=False,
                        word=ONE_STRIKE_WORD,
                        end_session_after_n_fails=1,
                    ),
                    _content_filter(
                        WARN_GUARDRAIL,
                        REALTIME_HOOK,
                        default_on=False,
                        word=WARN_WORD,
                        on_violation="warn",
                        end_session_after_n_fails=None,
                    ),
                    _raising_guardrail(VALUE_ERROR_GUARDRAIL, "ValueErrorGuardrail"),
                    _raising_guardrail(RUNTIME_ERROR_GUARDRAIL, "RuntimeErrorGuardrail"),
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


def _assert_transcription_blocked(
    session: Session, observed: tuple[dict[str, JsonValue], ...], transcript: str, *, update: JsonValue
) -> None:
    assert session.types == ("session.created", "session.updated", TRANSCRIPT_COMPLETED, "error", "session.updated"), (
        session
    )
    assert session.session_type == TRANSCRIPTION, session
    assert session.transcripts == (transcript,), session
    assert session.errors == (GUARDRAIL_VIOLATION,), session
    assert _sent_types(observed) == ("session.update", "input_audio_buffer.commit", "session.update"), _sent(observed)
    assert _session_updates(observed) == (update, update), _session_updates(observed)


def _relayed_close(session: Session) -> str:
    return f"upstream websocket closed with code {session.close_code}"


def _assert_ended_by_the_guardrail(session: Session, *, violations: int) -> None:
    assert session.errors == (GUARDRAIL_VIOLATION,) * violations, session
    assert session.types[-1] == "closed", session
    assert session.close_code == GUARDRAIL_END_CLOSE, session


def _assert_closed_by_a_proxy_failure(session: Session) -> None:
    assert session.errors[-1] == RELAYED_CLOSE, session
    assert session.close_code == PROXY_FAILURE_CLOSE, session
    assert session.error_messages[-1] == f"{_relayed_close(session)}: {PROXY_FAILURE_REASON}", session


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


async def _sdk_async_events(
    connection: AsyncRealtimeConnection, until: str = TRANSCRIPT_COMPLETED
) -> AsyncIterator[dict[str, JsonValue]]:
    async for event in connection:
        yield JSON_OBJECT.validate_python(event.model_dump())
        if event.type == until:
            return


def _sdk_sync_events(
    connection: RealtimeConnection, until: str = TRANSCRIPT_COMPLETED
) -> Iterator[dict[str, JsonValue]]:
    for event in connection:
        yield JSON_OBJECT.validate_python(event.model_dump())
        if event.type == until:
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
            "session.updated",
            TRANSCRIPT_COMPLETED,
            RESPONSE_DONE,
        ), session
        assert session.errors == (), session
        assert _sent_types(observed) == (
            "session.update",
            "session.update",
            "input_audio_buffer.commit",
            "response.create",
        ), _sent(observed)
        _assert_voice_updates(observed)
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
        assert session.errors == (GUARDRAIL_VIOLATION,), session
        assert session.types[-1] == RESPONSE_DONE, session
        assert _sent_types(observed) == (
            "session.update",
            "session.update",
            "input_audio_buffer.commit",
            "response.cancel",
            "conversation.item.create",
            "response.create",
        ), _sent(observed)
        _assert_voice_updates(observed)


def test_voice_session_cannot_re_enable_auto_response_with_a_later_update(guardrail_proxy: OwnedProxy) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _voice_scenario(CLEAN_TRANSCRIPT))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id, model=VOICE_MODEL)
        frames: Final = _voice_frames(VOICE_UPDATE, RE_ENABLE_UPDATE)
        session: Final = _talk(_ws_base(_owned_url(guardrail_proxy)), f"model={model}", key, frames)
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        assert session.types[-1] == RESPONSE_DONE, session
        assert session.errors == (), session
        _assert_voice_updates(observed, RE_ENABLE_FORWARDED)


def test_client_declared_transcription_type_does_not_bypass_the_voice_guardrail(guardrail_proxy: OwnedProxy) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _voice_scenario(BLOCKED_TRANSCRIPT))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id, model=VOICE_MODEL)
        frames: Final = _voice_frames(CLIENT_DECLARED_TRANSCRIPTION)
        session: Final = _talk(_ws_base(_owned_url(guardrail_proxy)), f"model={model}", key, frames, seconds=20)
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        assert session.transcripts == (BLOCKED_TRANSCRIPT,), session
        assert session.errors == (SESSION_TYPE_MISMATCH, GUARDRAIL_VIOLATION), session
        assert session.error_messages[0] == REALTIME_UPDATE_REFUSED, session
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


MUSE_PUSH_TO_TALK_FRAMES: Final[tuple[dict[str, JsonValue], ...]] = (
    {"binary_bytes": MUSE_PACKET_BYTES},
    {"binary_bytes": MUSE_PACKET_BYTES},
    {"binary_bytes": MUSE_REMAINDER_BYTES},
    END_STREAM,
)


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
    *,
    until: str | None = None,
) -> tuple[Session, tuple[dict[str, JsonValue], ...]]:
    handle: Final = _scripted(scenario, _muse_scenario(transcript), control_url=tls_upstream)
    key: Final = scenario.key()
    model: Final = _muse_deployment(scenario, handle.scenario_id, tls_upstream)
    verdict: Final = TRANSCRIPT_COMPLETED if BLOCKED_WORD not in transcript else "error"
    session: Final = _talk(
        _ws_base(_owned_url(guardrail_proxy)),
        f"model={model}{query}",
        key,
        _muse_frames(turn_detection),
        until=verdict if until is None else until,
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
        assert _muse_audio_frames(observed) == MUSE_PUSH_TO_TALK_FRAMES, _muse_audio_frames(observed)


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
        assert _muse_audio_frames(observed) == MUSE_PUSH_TO_TALK_FRAMES, _muse_audio_frames(observed)


def test_muse_transcription_session_on_violation_end_session_closes_the_session(
    guardrail_proxy: OwnedProxy, tls_upstream: str
) -> None:
    blocked: Final = _blocked(ENDER_WORD)
    with guardrail_proxy.gateway.scenario() as scenario:
        session, observed = _muse_session(
            guardrail_proxy,
            scenario,
            tls_upstream,
            blocked,
            f"&{TRANSCRIPTION_QUERY}&{_optin(ENDER_GUARDRAIL)}",
            None,
            until="closed",
        )
        assert session.transcripts == (blocked,), session
        _assert_ended_by_the_guardrail(session, violations=1)
        assert session.error_messages[0] == ENDER_MESSAGE, session
        assert _muse_audio_frames(observed) == MUSE_PUSH_TO_TALK_FRAMES, _muse_audio_frames(observed)


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


def test_without_a_guardrail_the_proxy_sends_no_session_update_or_response_create_of_its_own(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        transcription: Final = _scripted(scenario, _transcription_scenario(BLOCKED_TRANSCRIPT))
        voice: Final = _scripted(scenario, _voice_scenario(BLOCKED_TRANSCRIPT))
        key: Final = scenario.key()
        transcribe_model: Final = _openai_deployment(scenario, transcription.scenario_id)
        voice_model: Final = _openai_deployment(scenario, voice.scenario_id, model=VOICE_MODEL)
        transcribed: Final = _transcribe(_ws_base(_proxy_url()), f"model={transcribe_model}&{TRANSCRIPTION_QUERY}", key)
        talked: Final = _talk_without_a_reply(_ws_base(_proxy_url()), f"model={voice_model}", key)
        observed: Final = _observed_all(gateway.upstream_url)
        transcription_observed: Final = _belonging(observed, transcription.scenario_id)
        _assert_transcription_left_alone(
            transcribed, transcription_observed, BLOCKED_TRANSCRIPT, update=GA_TRANSCRIPTION_UPDATE
        )
        _assert_voice_session_ungated(talked, _belonging(observed, voice.scenario_id))


def _talk_without_a_reply(ws_base: str, query: str, key: str) -> Session:
    return _drive(
        ws_base,
        query,
        key,
        (_step(VOICE_UPDATE_FRAME, COMMIT, until=TRANSCRIPT_COMPLETED), _step(until=RESPONSE_DONE)),
    )


def _assert_voice_session_ungated(session: Session, observed: tuple[dict[str, JsonValue], ...]) -> None:
    assert session.types == ("session.created", "session.updated", TRANSCRIPT_COMPLETED, "timeout"), session
    assert session.transcripts == (BLOCKED_TRANSCRIPT,), session
    assert session.errors == (), session
    assert _sent_types(observed) == ("session.update", "input_audio_buffer.commit"), _sent(observed)
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
        session: Final = _talk_without_a_reply(_ws_base(_owned_url(guardrail_proxy)), f"model={model}", key)
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
        session: Final = _talk_without_a_reply(_ws_base(_owned_url(guardrail_proxy)), f"model={model}", key)
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
        assert session.errors == (GUARDRAIL_VIOLATION,), session
        _assert_voice_updates(observed)


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


def test_guardrails_list_names_every_configured_guardrail(guardrail_proxy: OwnedProxy) -> None:
    listed: Final = guardrail_proxy.gateway.get("/guardrails/list")
    names: Final = {string_value(object_value(entry)["guardrail_name"]) for entry in _list(listed["guardrails"])}
    assert names == CONFIGURED_GUARDRAILS, listed


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


@pytest.mark.parametrize("path", ["/v1/realtime", "/realtime", "/openai/v1/realtime"])
def test_blocked_transcript_on_a_transcription_session_reports_a_violation_and_sends_nothing_upstream(
    guardrail_proxy: OwnedProxy, path: str
) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _transcription_scenario(BLOCKED_TRANSCRIPT))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        session: Final = _transcribe_blocked(
            _ws_base(_owned_url(guardrail_proxy)), f"model={model}&{TRANSCRIPTION_QUERY}", key, path=path
        )
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        _assert_transcription_blocked(session, observed, BLOCKED_TRANSCRIPT, update=GA_TRANSCRIPTION_UPDATE)
        assert [_query(upgrade) for upgrade in _upgrades(observed)] == [
            [["model", TRANSCRIBE_MODEL], ["intent", TRANSCRIPTION]]
        ]
        rows: Final = _spend_rows(key, 1)
        assert rows[0]["call_type"] == "_arealtime", rows


async def _sdk_async_blocked_transcription(proxy_url: str, key: str, model: str) -> Session:
    client: Final = AsyncOpenAI(api_key=key, base_url=f"{proxy_url}/v1", websocket_base_url=f"{_ws_base(proxy_url)}/v1")
    async with client.realtime.connect(model=model, extra_query={"intent": TRANSCRIPTION}) as connection:
        await connection.session.update(session=SDK_TRANSCRIPTION_SESSION)
        await connection.input_audio_buffer.commit()
        verdict: Final = tuple([event async for event in _sdk_async_events(connection, until="error")])
        await connection.session.update(session=SDK_TRANSCRIPTION_SESSION)
        probe: Final = tuple([event async for event in _sdk_async_events(connection, until="session.updated")])
        return Session((*verdict, *probe), None)


def _sdk_sync_blocked_transcription(proxy_url: str, key: str, model: str) -> Session:
    client: Final = OpenAI(api_key=key, base_url=f"{proxy_url}/v1", websocket_base_url=f"{_ws_base(proxy_url)}/v1")
    with client.realtime.connect(model=model, extra_query={"intent": TRANSCRIPTION}) as connection:
        connection.session.update(session=SDK_TRANSCRIPTION_SESSION)
        connection.input_audio_buffer.commit()
        verdict: Final = tuple(_sdk_sync_events(connection, until="error"))
        connection.session.update(session=SDK_TRANSCRIPTION_SESSION)
        probe: Final = tuple(_sdk_sync_events(connection, until="session.updated"))
        return Session((*verdict, *probe), None)


@pytest.mark.parametrize("client", ["async", pytest.param("sync", marks=pytest.mark.timeout(90))])
def test_openai_sdk_transcription_session_gets_the_violation_and_stays_open(
    guardrail_proxy: OwnedProxy, client: str
) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _transcription_scenario(BLOCKED_TRANSCRIPT))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        proxy_url: Final = _owned_url(guardrail_proxy)
        session: Final = (
            asyncio.run(asyncio.wait_for(_sdk_async_blocked_transcription(proxy_url, key, model), SDK_SECONDS))
            if client == "async"
            else _sdk_sync_blocked_transcription(proxy_url, key, model)
        )
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        _assert_transcription_blocked(session, observed, BLOCKED_TRANSCRIPT, update=GA_TRANSCRIPTION_UPDATE)


def test_beta_protocol_transcription_session_blocked_transcript_reports_a_violation(
    guardrail_proxy: OwnedProxy,
) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _transcription_scenario(BLOCKED_TRANSCRIPT))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        session: Final = _transcribe_blocked(
            _ws_base(_owned_url(guardrail_proxy)),
            f"model={model}&{TRANSCRIPTION_QUERY}",
            key,
            update=BETA_TRANSCRIPTION_UPDATE,
            headers=BETA_HEADERS,
        )
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        _assert_transcription_blocked(session, observed, BLOCKED_TRANSCRIPT, update=BETA_TRANSCRIPTION_UPDATE)


def test_intent_without_model_blocked_transcript_reports_a_violation_on_the_whisper_default(
    guardrail_proxy: OwnedProxy,
) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _transcription_scenario(BLOCKED_TRANSCRIPT))
        key: Final = scenario.key()
        _named_deployment(
            guardrail_proxy.gateway, scenario, WHISPER_DEFAULT, f"openai/{WHISPER_DEFAULT}", handle.scenario_id
        )
        session: Final = _transcribe_blocked(_ws_base(_owned_url(guardrail_proxy)), TRANSCRIPTION_QUERY, key)
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        forwarded: Final = _with_transcription_model(GA_TRANSCRIPTION_UPDATE, WHISPER_DEFAULT)
        _assert_transcription_blocked(session, observed, BLOCKED_TRANSCRIPT, update=forwarded)
        assert [_query(upgrade) for upgrade in _upgrades(observed)] == [[["intent", TRANSCRIPTION]]]


def test_azure_transcription_session_blocked_transcript_reports_a_violation_and_sends_nothing_upstream(
    guardrail_proxy: OwnedProxy, tls_upstream: str
) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _transcription_scenario(BLOCKED_TRANSCRIPT), control_url=tls_upstream)
        key: Final = scenario.key()
        model: Final = _azure_deployment(scenario, handle.scenario_id, tls_upstream)
        session: Final = _transcribe_blocked(
            _ws_base(_owned_url(guardrail_proxy)), f"model={model}&{TRANSCRIPTION_QUERY}", key
        )
        observed: Final = _observed(tls_upstream, handle.scenario_id)
        _assert_transcription_blocked(session, observed, BLOCKED_TRANSCRIPT, update=GA_TRANSCRIPTION_UPDATE)
        upgrades: Final = tuple(request for request in observed if request["method"] == "WEBSOCKET")
        assert [upgrade["path"] for upgrade in upgrades] == ["/openai/v1/realtime"], upgrades
        assert [_query(object_value(upgrade["body"])) for upgrade in upgrades] == [[["intent", TRANSCRIPTION]]], (
            upgrades
        )


def test_second_violation_under_end_session_after_n_fails_closes_the_transcription_session(
    guardrail_proxy: OwnedProxy,
) -> None:
    blocked: Final = _blocked(TWO_STRIKES_WORD)
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _transcription_scenario(blocked, blocked))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        steps: Final = (_step(TRANSCRIPTION_UPDATE_FRAME, COMMIT, until="error"), _step(COMMIT, until="closed"))
        session: Final = _drive(
            _ws_base(_owned_url(guardrail_proxy)),
            f"model={model}&{TRANSCRIPTION_QUERY}&{_optin(TWO_STRIKES_GUARDRAIL)}",
            key,
            steps,
        )
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        assert session.types == (
            "session.created",
            "session.updated",
            TRANSCRIPT_COMPLETED,
            "error",
            TRANSCRIPT_COMPLETED,
            "error",
            "closed",
        ), session
        assert session.transcripts == (blocked, blocked), session
        _assert_ended_by_the_guardrail(session, violations=2)
        assert _sent_types(observed) == (
            "session.update",
            "input_audio_buffer.commit",
            "input_audio_buffer.commit",
        ), _sent(observed)
        rows: Final = _spend_rows(key, 1)
        assert rows[0]["call_type"] == "_arealtime", rows


def test_on_violation_end_session_closes_the_transcription_session_with_the_configured_message(
    guardrail_proxy: OwnedProxy,
) -> None:
    blocked: Final = _blocked(ENDER_WORD)
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _transcription_scenario(blocked))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        session: Final = _drive(
            _ws_base(_owned_url(guardrail_proxy)),
            f"model={model}&{TRANSCRIPTION_QUERY}&{_optin(ENDER_GUARDRAIL)}",
            key,
            (_step(TRANSCRIPTION_UPDATE_FRAME, COMMIT, until="closed"),),
        )
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        assert session.types == ("session.created", "session.updated", TRANSCRIPT_COMPLETED, "error", "closed"), session
        assert session.transcripts == (blocked,), session
        _assert_ended_by_the_guardrail(session, violations=1)
        assert session.error_messages[0] == ENDER_MESSAGE, session
        assert _sent_types(observed) == ("session.update", "input_audio_buffer.commit"), _sent(observed)


def test_end_session_after_one_fail_closes_the_transcription_session_on_the_first_violation(
    guardrail_proxy: OwnedProxy,
) -> None:
    blocked: Final = _blocked(ONE_STRIKE_WORD)
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _transcription_scenario(blocked))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        session: Final = _drive(
            _ws_base(_owned_url(guardrail_proxy)),
            f"model={model}&{TRANSCRIPTION_QUERY}&{_optin(ONE_STRIKE_GUARDRAIL)}",
            key,
            (_step(TRANSCRIPTION_UPDATE_FRAME, COMMIT, until="closed"),),
        )
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        assert session.types == ("session.created", "session.updated", TRANSCRIPT_COMPLETED, "error", "closed"), session
        _assert_ended_by_the_guardrail(session, violations=1)
        assert _sent_types(observed) == ("session.update", "input_audio_buffer.commit"), _sent(observed)


def _twice_blocked_and_open(guardrail_proxy: OwnedProxy, scenario: Scenario, blocked: str, query: str) -> None:
    handle: Final = _scripted(scenario, _transcription_scenario(blocked, blocked))
    key: Final = scenario.key()
    model: Final = _openai_deployment(scenario, handle.scenario_id)
    steps: Final = (
        _step(TRANSCRIPTION_UPDATE_FRAME, COMMIT, until="error"),
        _step(COMMIT, until="error"),
        _probe(),
    )
    session: Final = _drive(
        _ws_base(_owned_url(guardrail_proxy)), f"model={model}&{TRANSCRIPTION_QUERY}{query}", key, steps
    )
    observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
    assert session.types == (
        "session.created",
        "session.updated",
        TRANSCRIPT_COMPLETED,
        "error",
        TRANSCRIPT_COMPLETED,
        "error",
        "session.updated",
    ), session
    assert session.transcripts == (blocked, blocked), session
    assert session.errors == (GUARDRAIL_VIOLATION, GUARDRAIL_VIOLATION), session
    assert _sent_types(observed) == (
        "session.update",
        "input_audio_buffer.commit",
        "input_audio_buffer.commit",
        "session.update",
    ), _sent(observed)
    rows: Final = _spend_rows(key, 1)
    assert rows[0]["call_type"] == "_arealtime", rows


def test_same_blocked_transcript_twice_reports_two_violations_and_keeps_the_session_open(
    guardrail_proxy: OwnedProxy,
) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        _twice_blocked_and_open(guardrail_proxy, scenario, BLOCKED_TRANSCRIPT, "")


def test_on_violation_warn_with_a_null_end_rule_reports_each_violation_and_keeps_the_session_open(
    guardrail_proxy: OwnedProxy,
) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        _twice_blocked_and_open(guardrail_proxy, scenario, _blocked(WARN_WORD), f"&{_optin(WARN_GUARDRAIL)}")


def test_first_configured_guardrail_wins_when_two_match_one_transcript(guardrail_proxy: OwnedProxy) -> None:
    transcript: Final = f"please add {BLOCKED_WORD} and {ENDER_WORD} to the order"
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _transcription_scenario(transcript))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        session: Final = _transcribe_blocked(
            _ws_base(_owned_url(guardrail_proxy)), f"model={model}&{TRANSCRIPTION_QUERY}&{_optin(ENDER_GUARDRAIL)}", key
        )
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        _assert_transcription_blocked(session, observed, transcript, update=GA_TRANSCRIPTION_UPDATE)
        assert ENDER_MESSAGE not in session.error_messages, session


def test_guardrail_raising_value_error_reports_the_exception_text_and_keeps_the_transcription_session_open(
    guardrail_proxy: OwnedProxy,
) -> None:
    blocked: Final = _blocked(VALUE_ERROR_WORD)
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _transcription_scenario(blocked))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        session: Final = _transcribe_blocked(
            _ws_base(_owned_url(guardrail_proxy)),
            f"model={model}&{TRANSCRIPTION_QUERY}&{_optin(VALUE_ERROR_GUARDRAIL)}",
            key,
        )
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        _assert_transcription_blocked(session, observed, blocked, update=GA_TRANSCRIPTION_UPDATE)
        assert session.error_messages == (f"{VALUE_ERROR_WORD} is not allowed",), session


def test_guardrail_raising_runtime_error_closes_the_transcription_session_with_a_proxy_failure(
    guardrail_proxy: OwnedProxy,
) -> None:
    blocked: Final = _blocked(RUNTIME_ERROR_WORD)
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _transcription_scenario(blocked))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        session: Final = _drive(
            _ws_base(_owned_url(guardrail_proxy)),
            f"model={model}&{TRANSCRIPTION_QUERY}&{_optin(RUNTIME_ERROR_GUARDRAIL)}",
            key,
            (_step(TRANSCRIPTION_UPDATE_FRAME, COMMIT, until="closed"),),
        )
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        assert session.types == ("session.created", "session.updated", TRANSCRIPT_COMPLETED, "error", "closed"), session
        assert session.transcripts == (blocked,), session
        _assert_closed_by_a_proxy_failure(session)
        assert _sent_types(observed) == ("session.update", "input_audio_buffer.commit"), _sent(observed)
        rows: Final = _spend_rows(key, 1)
        assert rows[0]["call_type"] == "_arealtime", rows


@pytest.mark.parametrize("guardrail", [VALUE_ERROR_GUARDRAIL, RUNTIME_ERROR_GUARDRAIL])
def test_raising_guardrails_leave_a_clean_transcription_session_alone(
    guardrail_proxy: OwnedProxy, guardrail: str
) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _transcription_scenario(CLEAN_TRANSCRIPT))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        session: Final = _transcribe(
            _ws_base(_owned_url(guardrail_proxy)), f"model={model}&{TRANSCRIPTION_QUERY}&{_optin(guardrail)}", key
        )
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        _assert_transcription_left_alone(session, observed, CLEAN_TRANSCRIPT, update=GA_TRANSCRIPTION_UPDATE)


def _voice_session(
    guardrail_proxy: OwnedProxy, scenario: Scenario, response: RealtimeResponse, query: str, steps: tuple[Step, ...]
) -> tuple[Session, tuple[dict[str, JsonValue], ...]]:
    handle: Final = _scripted(scenario, response)
    key: Final = scenario.key()
    model: Final = _openai_deployment(scenario, handle.scenario_id, model=VOICE_MODEL)
    session: Final = _drive(_ws_base(_owned_url(guardrail_proxy)), f"model={model}{query}", key, steps)
    return session, _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)


def test_voice_session_guardrail_raising_runtime_error_closes_with_the_same_proxy_failure(
    guardrail_proxy: OwnedProxy,
) -> None:
    blocked: Final = _blocked(RUNTIME_ERROR_WORD)
    with guardrail_proxy.gateway.scenario() as scenario:
        session, observed = _voice_session(
            guardrail_proxy,
            scenario,
            _voice_scenario(blocked),
            f"&{_optin(RUNTIME_ERROR_GUARDRAIL)}",
            (_step(VOICE_UPDATE_FRAME, COMMIT, until="closed"),),
        )
        assert session.types == (
            "session.created",
            "session.updated",
            "session.updated",
            TRANSCRIPT_COMPLETED,
            "error",
            "closed",
        ), session
        _assert_closed_by_a_proxy_failure(session)
        assert _sent_types(observed) == ("session.update", "session.update", "input_audio_buffer.commit"), _sent(
            observed
        )


def test_voice_session_guardrail_raising_value_error_is_voiced_through_the_backend(guardrail_proxy: OwnedProxy) -> None:
    blocked: Final = _blocked(VALUE_ERROR_WORD)
    with guardrail_proxy.gateway.scenario() as scenario:
        session, observed = _voice_session(
            guardrail_proxy,
            scenario,
            _voice_scenario(blocked),
            f"&{_optin(VALUE_ERROR_GUARDRAIL)}",
            (_step(VOICE_UPDATE_FRAME, COMMIT, until=RESPONSE_DONE),),
        )
        assert session.transcripts == (blocked,), session
        assert session.errors == (GUARDRAIL_VIOLATION,), session
        assert session.error_messages == (f"{VALUE_ERROR_WORD} is not allowed",), session
        assert session.types[-1] == RESPONSE_DONE, session
        assert _sent_types(observed) == (
            "session.update",
            "session.update",
            "input_audio_buffer.commit",
            *VOICE_BLOCK_FRAMES,
        ), _sent(observed)


def test_voice_session_second_violation_under_end_session_after_n_fails_closes_after_voicing_both(
    guardrail_proxy: OwnedProxy,
) -> None:
    blocked: Final = _blocked(TWO_STRIKES_WORD)
    with guardrail_proxy.gateway.scenario() as scenario:
        session, observed = _voice_session(
            guardrail_proxy,
            scenario,
            _voice_scenario(blocked, blocked),
            f"&{_optin(TWO_STRIKES_GUARDRAIL)}",
            (_step(VOICE_UPDATE_FRAME, COMMIT, until=RESPONSE_DONE), _step(COMMIT, until="closed")),
        )
        assert session.transcripts == (blocked, blocked), session
        assert session.errors == (GUARDRAIL_VIOLATION, GUARDRAIL_VIOLATION), session
        assert session.types[-1] == "closed", session
        assert session.close_code == GUARDRAIL_END_CLOSE, session
        assert _sent_types(observed) == (
            "session.update",
            "session.update",
            "input_audio_buffer.commit",
            *VOICE_BLOCK_FRAMES,
            "input_audio_buffer.commit",
            *VOICE_BLOCK_FRAMES,
        ), _sent(observed)


def _user_text_item(text: str) -> dict[str, JsonValue]:
    return {
        "type": "conversation.item.create",
        "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]},
    }


def _tool_output_item(output: str) -> dict[str, JsonValue]:
    return {
        "type": "conversation.item.create",
        "item": {"type": "function_call_output", "call_id": "call_realtime_guard", "output": output},
    }


def test_blocked_user_text_on_a_transcription_session_is_dropped_without_a_voiced_block(
    guardrail_proxy: OwnedProxy,
) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _transcription_scenario(CLEAN_TRANSCRIPT))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        steps: Final = (
            _probe(),
            _step(_user_text_item(BLOCKED_TRANSCRIPT), RESPONSE_CREATE, until="error"),
            _probe(),
        )
        session: Final = _drive(
            _ws_base(_owned_url(guardrail_proxy)), f"model={model}&{TRANSCRIPTION_QUERY}", key, steps
        )
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        assert session.types == ("session.created", "session.updated", "error", "session.updated"), session
        assert session.errors == (GUARDRAIL_VIOLATION,), session
        assert _sent_types(observed) == ("session.update", "session.update"), _sent(observed)


def test_blocked_tool_output_on_a_transcription_session_is_sanitized_without_a_voiced_block(
    guardrail_proxy: OwnedProxy,
) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _transcription_scenario(CLEAN_TRANSCRIPT))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        steps: Final = (
            _probe(),
            _step(_tool_output_item(BLOCKED_TRANSCRIPT), until="error"),
            _probe(),
        )
        session: Final = _drive(
            _ws_base(_owned_url(guardrail_proxy)), f"model={model}&{TRANSCRIPTION_QUERY}", key, steps
        )
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        assert session.types == ("session.created", "session.updated", "error", "session.updated"), session
        assert session.errors == (GUARDRAIL_VIOLATION,), session
        assert _sent_types(observed) == ("session.update", "conversation.item.create", "session.update"), _sent(
            observed
        )
        assert object_value(_sent(observed)[1]["item"])["output"] == TOOL_OUTPUT_BLOCKED, _sent(observed)


def test_clean_user_text_on_a_transcription_session_is_forwarded_verbatim(guardrail_proxy: OwnedProxy) -> None:
    item: Final = _user_text_item(CLEAN_TRANSCRIPT)
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _transcription_scenario(CLEAN_TRANSCRIPT))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        steps: Final = (_step(TRANSCRIPTION_UPDATE_FRAME, item, RESPONSE_CREATE, until=TRANSCRIPT_COMPLETED),)
        session: Final = _drive(
            _ws_base(_owned_url(guardrail_proxy)), f"model={model}&{TRANSCRIPTION_QUERY}", key, steps
        )
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        assert session.types == ("session.created", "session.updated", TRANSCRIPT_COMPLETED), session
        assert session.errors == (), session
        assert _sent(observed) == (TRANSCRIPTION_UPDATE_FRAME, item, RESPONSE_CREATE), _sent(observed)


CLEAN_TRANSCRIPT_SHAPES: Final = (pytest.param("", id="empty"), pytest.param(FIVE_KB, id="five_kilobytes"))
NON_STRING_TRANSCRIPTS: Final = (
    pytest.param(None, id="null"),
    pytest.param(123, id="integer"),
    pytest.param(["a"], id="list"),
)


@pytest.mark.parametrize("transcript", CLEAN_TRANSCRIPT_SHAPES)
def test_clean_transcript_field_shapes_are_relayed_and_the_session_stays_open(
    guardrail_proxy: OwnedProxy, transcript: str
) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _transcription_scenario(transcript))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        steps: Final = (_step(TRANSCRIPTION_UPDATE_FRAME, COMMIT, until=TRANSCRIPT_COMPLETED), _probe())
        session: Final = _drive(
            _ws_base(_owned_url(guardrail_proxy)), f"model={model}&{TRANSCRIPTION_QUERY}", key, steps
        )
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        assert session.types == ("session.created", "session.updated", TRANSCRIPT_COMPLETED, "session.updated"), session
        assert session.transcripts == (transcript,), session
        assert session.errors == (), session
        assert _sent_types(observed) == ("session.update", "input_audio_buffer.commit", "session.update"), _sent(
            observed
        )


def test_transcript_event_without_the_field_is_relayed_and_the_session_stays_open(guardrail_proxy: OwnedProxy) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _transcription_events((_transcript_event_without_the_field(),)))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        steps: Final = (_step(TRANSCRIPTION_UPDATE_FRAME, COMMIT, until=TRANSCRIPT_COMPLETED), _probe())
        session: Final = _drive(
            _ws_base(_owned_url(guardrail_proxy)), f"model={model}&{TRANSCRIPTION_QUERY}", key, steps
        )
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        assert session.types == ("session.created", "session.updated", TRANSCRIPT_COMPLETED, "session.updated"), session
        assert "transcript" not in session.events[2], session
        assert session.errors == (), session
        assert _sent_types(observed) == ("session.update", "input_audio_buffer.commit", "session.update"), _sent(
            observed
        )


def test_five_kilobyte_transcript_ending_in_the_blocked_word_reports_a_violation(guardrail_proxy: OwnedProxy) -> None:
    transcript: Final = f"{FIVE_KB} {BLOCKED_TRANSCRIPT}"
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _transcription_scenario(transcript))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        session: Final = _transcribe_blocked(
            _ws_base(_owned_url(guardrail_proxy)), f"model={model}&{TRANSCRIPTION_QUERY}", key
        )
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        _assert_transcription_blocked(session, observed, transcript, update=GA_TRANSCRIPTION_UPDATE)


@pytest.mark.parametrize("transcript", NON_STRING_TRANSCRIPTS)
def test_non_string_transcript_field_closes_the_transcription_session_with_a_proxy_failure(
    guardrail_proxy: OwnedProxy, transcript: JsonValue
) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        handle: Final = _scripted(scenario, _transcription_scenario(transcript))
        key: Final = scenario.key()
        model: Final = _openai_deployment(scenario, handle.scenario_id)
        session: Final = _drive(
            _ws_base(_owned_url(guardrail_proxy)),
            f"model={model}&{TRANSCRIPTION_QUERY}",
            key,
            (_step(TRANSCRIPTION_UPDATE_FRAME, COMMIT, until="closed"),),
        )
        observed: Final = _observed(guardrail_proxy.gateway.upstream_url, handle.scenario_id)
        assert session.types == ("session.created", "session.updated", TRANSCRIPT_COMPLETED, "error", "closed"), session
        assert session.events[2]["transcript"] == transcript, session
        _assert_closed_by_a_proxy_failure(session)
        assert _sent_types(observed) == ("session.update", "input_audio_buffer.commit"), _sent(observed)


@pytest.mark.parametrize("transcript", NON_STRING_TRANSCRIPTS)
def test_non_string_transcript_field_closes_a_voice_session_with_the_same_proxy_failure(
    guardrail_proxy: OwnedProxy, transcript: JsonValue
) -> None:
    with guardrail_proxy.gateway.scenario() as scenario:
        session, observed = _voice_session(
            guardrail_proxy,
            scenario,
            _voice_scenario(transcript),
            "",
            (_step(VOICE_UPDATE_FRAME, COMMIT, until="closed"),),
        )
        assert session.types == (
            "session.created",
            "session.updated",
            "session.updated",
            TRANSCRIPT_COMPLETED,
            "error",
            "closed",
        ), session
        assert session.events[3]["transcript"] == transcript, session
        _assert_closed_by_a_proxy_failure(session)
        assert _sent_types(observed) == ("session.update", "session.update", "input_audio_buffer.commit"), _sent(
            observed
        )


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
    return _relayed_close(session) in session.error_messages[-1]


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
        blocked_after_kill: Final = _scripted(scenario, _transcription_scenario(BLOCKED_TRANSCRIPT))
        blocked_after_restart: Final = _scripted(scenario, _transcription_scenario(BLOCKED_TRANSCRIPT))
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
            blocked_model_after_kill: Final = _named_deployment(
                owned.gateway,
                scenario,
                f"realtime-guard-owned-blocked-{uuid.uuid4().hex[:8]}",
                f"openai/{TRANSCRIBE_MODEL}",
                blocked_after_kill.scenario_id,
            )
            blocked_model_after_restart: Final = _named_deployment(
                owned.gateway,
                scenario,
                f"realtime-guard-owned-blocked-{uuid.uuid4().hex[:8]}",
                f"openai/{TRANSCRIBE_MODEL}",
                blocked_after_restart.scenario_id,
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
            blocked_on_the_survivor: Final = _transcribe_blocked(
                _ws_base(owned_url), f"model={blocked_model_after_kill}&{TRANSCRIPTION_QUERY}", key
            )
            _assert_transcription_blocked(
                blocked_on_the_survivor,
                _observed(gateway.upstream_url, blocked_after_kill.scenario_id),
                BLOCKED_TRANSCRIPT,
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
            blocked_on_the_restart: Final = _transcribe_blocked(
                _ws_base(_owned_url(restarted)), f"model={blocked_model_after_restart}&{TRANSCRIPTION_QUERY}", key
            )
            _assert_transcription_blocked(
                blocked_on_the_restart,
                _observed(gateway.upstream_url, blocked_after_restart.scenario_id),
                BLOCKED_TRANSCRIPT,
                update=GA_TRANSCRIPTION_UPDATE,
            )


async def _frames_reporting_verdicts(
    socket: ClientConnection, session_id: str, verdicts: asyncio.Queue[str]
) -> AsyncIterator[dict[str, JsonValue]]:
    async for frame in _frames_until_closed(socket):
        if frame.get("type") == "error" and object_value(frame["error"]).get("type") == GUARDRAIL_VIOLATION[0]:
            await verdicts.put(session_id)
        yield frame


async def _hold_blocked_until_closed(
    ws_base: str, query: str, key: str, opened: asyncio.Queue[str], verdicts: asyncio.Queue[str]
) -> Session:
    async with websockets.connect(
        f"{ws_base}/v1/realtime?{query}", additional_headers={"Authorization": f"Bearer {key}"}
    ) as socket:
        created: Final = JSON_OBJECT.validate_json(await socket.recv())
        assert created.get("type") == "session.created", created
        session_id: Final = string_value(object_value(created["session"])["id"])
        await opened.put(session_id)
        await socket.send(json.dumps(TRANSCRIPTION_UPDATE_FRAME))
        await socket.send(json.dumps(COMMIT))
        return Session(tuple([frame async for frame in _frames_reporting_verdicts(socket, session_id, verdicts)]), None)


@dataclass(frozen=True, slots=True)
class BlockedBurst:
    sessions: tuple[Session, ...]
    before_the_outage: tuple[dict[str, JsonValue], ...]


async def _blocked_burst_through_outage(
    ws_base: str,
    proxy_url: str,
    upstream_url: str,
    scenario_id: str,
    model: str,
    key: str,
    stop_upstream: Callable[[], None],
) -> BlockedBurst:
    opened: Final[asyncio.Queue[str]] = asyncio.Queue()
    verdicts: Final[asyncio.Queue[str]] = asyncio.Queue()
    query: Final = f"model={model}&{TRANSCRIPTION_QUERY}"
    holders: Final = tuple(
        asyncio.ensure_future(_hold_blocked_until_closed(ws_base, query, key, opened, verdicts)) for _ in range(BURST)
    )
    opened_sessions: Final = await asyncio.wait_for(_drain(opened, BURST), 60)
    assert len(opened_sessions) == BURST, opened_sessions
    judged_sessions: Final = await asyncio.wait_for(_drain(verdicts, BURST), 60)
    assert sorted(judged_sessions) == sorted(opened_sessions), judged_sessions
    before_the_outage: Final = await asyncio.to_thread(_observed, upstream_url, scenario_id)
    await asyncio.to_thread(stop_upstream)
    async with httpx.AsyncClient(base_url=proxy_url, timeout=15, trust_env=False) as client:
        liveliness: Final = await client.get("/health/liveliness")
    assert liveliness.status_code == 200, liveliness.text
    return BlockedBurst(tuple(await asyncio.wait_for(asyncio.gather(*holders), 90)), before_the_outage)


@pytest.mark.timeout(240)
def test_upstream_outage_closes_every_blocked_transcription_session_and_the_verdict_survives_the_restart(
    guardrail_proxy: OwnedProxy, tmp_path: Path, record_property: RecordProperty
) -> None:
    with guardrail_proxy.gateway.scenario() as scenario, owned_upstream(tmp_path) as slot:
        scenario_id: Final = f"realtime-guard-blocked-outage-{uuid.uuid4().hex[:12]}"
        register_scenario(scenario_id, _transcription_scenario(BLOCKED_TRANSCRIPT), control_url=slot.url)
        key: Final = scenario.key()
        model: Final = scenario.model(model=f"openai/{TRANSCRIBE_MODEL}", api_key=scenario_id, api_base=slot.url)
        proxy_url: Final = _owned_url(guardrail_proxy)
        burst: Final = asyncio.run(
            _blocked_burst_through_outage(_ws_base(proxy_url), proxy_url, slot.url, scenario_id, model, key, slot.stop)
        )
        record_property(
            "close_codes_during_upstream_outage", sorted(session.close_code or 0 for session in burst.sessions)
        )
        assert [session.types for session in burst.sessions] == [
            ("session.updated", TRANSCRIPT_COMPLETED, "error", "error", "closed")
        ] * BURST, burst.sessions
        assert [session.errors for session in burst.sessions] == [(GUARDRAIL_VIOLATION, RELAYED_CLOSE)] * BURST, (
            burst.sessions
        )
        assert all(_relays_the_upstream_close(session) for session in burst.sessions), burst.sessions
        assert len({session.close_code for session in burst.sessions}) == 1, burst.sessions
        assert sorted(map(str, _sent_types(burst.before_the_outage))) == sorted(
            ("session.update", "input_audio_buffer.commit") * BURST
        ), _sent(burst.before_the_outage)
        slot.start()
        register_scenario(scenario_id, _transcription_scenario(BLOCKED_TRANSCRIPT), control_url=slot.url)
        recovered: Final = _transcribe_blocked(_ws_base(proxy_url), f"model={model}&{TRANSCRIPTION_QUERY}", key)
        _assert_transcription_blocked(
            recovered, _observed(slot.url, scenario_id), BLOCKED_TRANSCRIPT, update=GA_TRANSCRIPTION_UPDATE
        )
        rows: Final = _spend_rows(key, BURST + 1)
        assert {str(row["call_type"]) for row in rows} == {"_arealtime"}, rows
