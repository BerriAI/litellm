"""Upstream websocket URL for realtime sessions on a custom ``api_base``.

``model`` leaves the upstream URL only for ``intent=transcription`` on OpenAI's own hosts. Every row here
runs against the scripted upstream on 127.0.0.1, where that gate is off, so the rows pin the forwarding
that must not move: ``model`` and ``intent`` reach the upstream exactly as the proxy resolved them, on
every route alias, through the OpenAI SDK and raw websockets, under malformed query strings, and while
the upstream or a proxy worker dies.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import psutil
import pytest
import websockets
from openai import AsyncOpenAI, OpenAI
from pydantic import JsonValue
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed, InvalidStatus

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value, string_value
from tests.integration._support.database import read_rows
from tests.integration._support.process import owned_proxy_process, owned_upstream, stop_root_process
from tests.integration._support.upstream import ScenarioHandle, delete_scenario, register_scenario
from tests.integration.cost_calculation.cost_tracking_case import RealtimeResponse

RecordProperty = Callable[[str, object], None]

pytestmark: Final = pytest.mark.timeout(180)

TRANSCRIBE_MODEL: Final = "gpt-live-transcribe"
SECOND_TRANSCRIBE_MODEL: Final = "gpt-live-transcribe-second"
CONVERSATION_MODEL: Final = "gpt-realtime-2"
WHISPER_DEFAULT: Final = "gpt-realtime-whisper"
XAI_MODEL: Final = "grok-4-1-fast-non-reasoning"
TRANSCRIPTION: Final = "transcription"
FIVE_KB: Final = "x" * 5120
BURST: Final = 20
OPEN_SESSIONS: Final = 6
WORKERS: Final = 2
INPUT_TOKENS: Final = 7
OUTPUT_TOKENS: Final = 5
USAGE: Final[dict[str, JsonValue]] = {
    "total_tokens": INPUT_TOKENS + OUTPUT_TOKENS,
    "input_tokens": INPUT_TOKENS,
    "output_tokens": OUTPUT_TOKENS,
    "input_token_details": {"text_tokens": INPUT_TOKENS, "audio_tokens": 0, "cached_tokens": 0},
    "output_token_details": {"text_tokens": OUTPUT_TOKENS, "audio_tokens": 0},
}
TRANSCRIPTION_PAIRS: Final = [["model", TRANSCRIBE_MODEL], ["intent", TRANSCRIPTION]]
TRANSCRIPTION_QUERY: Final = f"intent={TRANSCRIPTION}"
UNAUTHENTICATED_STATUS: Final = 403
UPSTREAM_REFUSAL_CLOSE: Final = 1008
UNKNOWN_MODEL_CLOSE: Final = 1011
SPEND_SQL: Final = 'SELECT call_type, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE api_key = %s'


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
    def session_model(self) -> str:
        return string_value(object_value(self.events[0]["session"])["model"])

    @property
    def response_ids(self) -> tuple[str, ...]:
        done: Final = tuple(event for event in self.events if event["type"] == "response.done")
        return tuple(string_value(object_value(event["response"])["id"]) for event in done)

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


def _done_event() -> RealtimeResponse:
    return RealtimeResponse(
        content_type="application/x-realtime",
        events=(
            {
                "type": "response.done",
                "event_id": "evt_$UNIQUE_ID",
                "response": {
                    "id": "resp_$UNIQUE_ID",
                    "object": "realtime.response",
                    "status": "completed",
                    "output": [],
                    "usage": USAGE,
                },
            },
        ),
    )


def _scripted(scenario: Scenario) -> ScenarioHandle:
    handle: Final = register_scenario(f"realtime-url-{uuid.uuid4().hex[:12]}", _done_event())
    scenario.cleanups.callback(delete_scenario, handle)
    return handle


def _deployment(
    gateway: Gateway,
    scenario: Scenario,
    scenario_id: str,
    *,
    model: str = f"openai/{TRANSCRIBE_MODEL}",
    api_base: str | None = None,
) -> str:
    base: Final = gateway.upstream_url.rstrip("/") if api_base is None else api_base
    return scenario.model(model=model, api_key=scenario_id, api_base=base)


def _named_deployment(creator: Gateway, scenario: Scenario, name: str, model: str, scenario_id: str) -> str:
    created: Final = creator.post(
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {"model": model, "api_key": scenario_id, "api_base": creator.upstream_url.rstrip("/")},
            "model_info": {},
        },
    )
    scenario.cleanups.callback(scenario.delete_model, string_value(object_value(created["model_info"])["id"]))
    return name


def _close_code(closed: ConnectionClosed) -> int:
    return 1006 if closed.rcvd is None else closed.rcvd.code


async def _frames(socket: ClientConnection, turns: int) -> AsyncIterator[dict[str, JsonValue]]:
    try:
        first: Final = JSON_OBJECT.validate_json(await socket.recv())
        yield first
        if first.get("type") != "session.created":
            async for message in socket:
                yield JSON_OBJECT.validate_json(message)
            return
        for _ in range(turns):
            await socket.send(json.dumps({"type": "response.create"}))
            async for message in socket:
                event: Final = JSON_OBJECT.validate_json(message)
                yield event
                if event.get("type") == "response.done":
                    break
    except ConnectionClosed as closed:
        yield {"type": "closed", "code": _close_code(closed)}


async def _collect(socket: ClientConnection, turns: int) -> tuple[dict[str, JsonValue], ...]:
    return tuple([frame async for frame in _frames(socket, turns)])


async def _session(ws_base: str, path: str, query: str, key: str | None, *, turns: int = 0) -> Session:
    headers: Final = {} if key is None else {"Authorization": f"Bearer {key}"}
    try:
        async with websockets.connect(f"{ws_base}{path}?{query}", additional_headers=headers) as socket:
            return Session(await asyncio.wait_for(_collect(socket, turns), 60), None)
    except InvalidStatus as refusal:
        return Session((), refusal.response.status_code)


def _run(path: str, query: str, key: str | None, *, turns: int = 0, ws_base: str | None = None) -> Session:
    return asyncio.run(_session(_ws_base(_proxy_url()) if ws_base is None else ws_base, path, query, key, turns=turns))


def _upgrades(gateway: Gateway, scenario_id: str) -> tuple[dict[str, JsonValue], ...]:
    with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream:
        observed: Final = tuple(map(object_value, upstream.get("/__observations").json()["requests"]))
    return tuple(
        request
        for request in observed
        if request["method"] == "WEBSOCKET" and request["authorization"] == f"Bearer {scenario_id}"
    )


def _pairs(upgrade: dict[str, JsonValue]) -> JsonValue:
    return object_value(upgrade["body"])["query"]


def _spend_rows(key: str, count: int) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows(SPEND_SQL, (sha256(key.encode()).hexdigest(),)),
        lambda rows: len(rows) == count,
        seconds=70,
    )


def _assert_transcription_reached_upstream(gateway: Gateway, scenario_id: str, session: Session) -> None:
    assert session.types == ("session.created",), session
    assert session.session_model == TRANSCRIBE_MODEL, session
    upgrades: Final = _upgrades(gateway, scenario_id)
    assert [upgrade["path"] for upgrade in upgrades] == ["/v1/realtime"], upgrades
    assert [_pairs(upgrade) for upgrade in upgrades] == [TRANSCRIPTION_PAIRS], upgrades


@pytest.mark.parametrize("path", ["/v1/realtime", "/realtime", "/openai/v1/realtime"])
def test_transcription_session_forwards_model_and_intent_to_a_custom_api_base(gateway: Gateway, path: str) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _scripted(scenario)
        key: Final = scenario.key()
        model: Final = _deployment(gateway, scenario, handle.scenario_id)
        session: Final = _run(path, f"model={model}&{TRANSCRIPTION_QUERY}", key)
        _assert_transcription_reached_upstream(gateway, handle.scenario_id, session)
        rows: Final = _spend_rows(key, 1)
        assert rows[0]["call_type"] == "_arealtime", rows


def test_openai_sdk_async_transcription_session_reaches_a_custom_api_base(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _scripted(scenario)
        key: Final = scenario.key()
        model: Final = _deployment(gateway, scenario, handle.scenario_id)

        async def connect() -> dict[str, JsonValue]:
            client: Final = AsyncOpenAI(
                api_key=key, base_url=f"{_proxy_url()}/v1", websocket_base_url=f"{_ws_base(_proxy_url())}/v1"
            )
            async with client.realtime.connect(model=model, extra_query={"intent": TRANSCRIPTION}) as connection:
                return JSON_OBJECT.validate_python((await connection.recv()).model_dump())

        created: Final = asyncio.run(connect())
        _assert_transcription_reached_upstream(gateway, handle.scenario_id, Session((created,), None))


def test_openai_sdk_sync_transcription_session_reaches_a_custom_api_base(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _scripted(scenario)
        key: Final = scenario.key()
        model: Final = _deployment(gateway, scenario, handle.scenario_id)
        client: Final = OpenAI(
            api_key=key, base_url=f"{_proxy_url()}/v1", websocket_base_url=f"{_ws_base(_proxy_url())}/v1"
        )
        with client.realtime.connect(model=model, extra_query={"intent": TRANSCRIPTION}) as connection:
            created: Final = JSON_OBJECT.validate_python(connection.recv().model_dump())
        _assert_transcription_reached_upstream(gateway, handle.scenario_id, Session((created,), None))


def test_conversation_session_forwards_only_model_and_bills_the_scripted_usage(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _scripted(scenario)
        key: Final = scenario.key()
        model: Final = _deployment(gateway, scenario, handle.scenario_id, model=f"openai/{CONVERSATION_MODEL}")
        session: Final = _run("/v1/realtime", f"model={model}", key, turns=1)
        assert session.types == ("session.created", "response.done"), session
        assert session.session_model == CONVERSATION_MODEL, session
        upgrades: Final = _upgrades(gateway, handle.scenario_id)
        assert [_pairs(upgrade) for upgrade in upgrades] == [[["model", CONVERSATION_MODEL]]], upgrades
        rows: Final = _spend_rows(key, 1)
        assert rows[0]["call_type"] == "_arealtime", rows
        assert rows[0]["prompt_tokens"] == INPUT_TOKENS, rows
        assert rows[0]["completion_tokens"] == OUTPUT_TOKENS, rows


def test_intent_without_model_routes_to_the_whisper_default_and_forwards_intent_only(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _scripted(scenario)
        key: Final = scenario.key()
        _named_deployment(gateway, scenario, WHISPER_DEFAULT, f"openai/{WHISPER_DEFAULT}", handle.scenario_id)
        session: Final = _run("/v1/realtime", TRANSCRIPTION_QUERY, key)
        assert session.types == ("session.created",), session
        assert session.session_model == "", session
        upgrades: Final = _upgrades(gateway, handle.scenario_id)
        assert [_pairs(upgrade) for upgrade in upgrades] == [[["intent", TRANSCRIPTION]]], upgrades


def test_xai_transcription_session_keeps_model_on_a_custom_api_base(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _scripted(scenario)
        key: Final = scenario.key()
        model: Final = _deployment(gateway, scenario, handle.scenario_id, model=f"xai/{XAI_MODEL}")
        session: Final = _run("/v1/realtime", f"model={model}&{TRANSCRIPTION_QUERY}", key)
        assert session.types == ("session.created",), session
        assert session.session_model == XAI_MODEL, session
        upgrades: Final = _upgrades(gateway, handle.scenario_id)
        assert [_pairs(upgrade) for upgrade in upgrades] == [[["model", XAI_MODEL], ["intent", TRANSCRIPTION]]], (
            upgrades
        )


def test_api_base_with_a_version_path_still_upgrades_at_v1_realtime(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _scripted(scenario)
        key: Final = scenario.key()
        model: Final = _deployment(gateway, scenario, handle.scenario_id, api_base=f"{gateway.upstream_url}/v1")
        session: Final = _run("/v1/realtime", f"model={model}&{TRANSCRIPTION_QUERY}", key)
        _assert_transcription_reached_upstream(gateway, handle.scenario_id, session)


@pytest.mark.parametrize(
    ("intent_query", "forwarded_intent"),
    [
        (f"intent={TRANSCRIPTION}&intent={TRANSCRIPTION}", TRANSCRIPTION),
        ("intent=", ""),
        ("intent=1", "1"),
        (f"intent={FIVE_KB}", FIVE_KB),
        (f"intent={TRANSCRIPTION}&intent=other", "other"),
    ],
    ids=["twice_same_value", "empty", "integer", "five_kilobytes", "two_values"],
)
def test_malformed_intent_is_forwarded_as_the_proxy_resolved_it(
    gateway: Gateway, intent_query: str, forwarded_intent: str
) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _scripted(scenario)
        key: Final = scenario.key()
        model: Final = _deployment(gateway, scenario, handle.scenario_id)
        session: Final = _run("/v1/realtime", f"model={model}&{intent_query}", key)
        assert session.types == ("session.created",), session
        upgrades: Final = _upgrades(gateway, handle.scenario_id)
        assert [_pairs(upgrade) for upgrade in upgrades] == [
            [["model", TRANSCRIBE_MODEL], ["intent", forwarded_intent]]
        ]


def test_model_given_twice_resolves_to_the_last_deployment(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _scripted(scenario)
        key: Final = scenario.key()
        first: Final = _deployment(gateway, scenario, handle.scenario_id)
        second: Final = _deployment(gateway, scenario, handle.scenario_id, model=f"openai/{SECOND_TRANSCRIBE_MODEL}")
        session: Final = _run("/v1/realtime", f"model={first}&model={second}&{TRANSCRIPTION_QUERY}", key)
        assert session.types == ("session.created",), session
        assert session.session_model == SECOND_TRANSCRIBE_MODEL, session
        upgrades: Final = _upgrades(gateway, handle.scenario_id)
        expected: Final = [[["model", SECOND_TRANSCRIBE_MODEL], ["intent", TRANSCRIPTION]]]
        assert [_pairs(upgrade) for upgrade in upgrades] == expected, upgrades


def test_unauthenticated_upgrade_is_refused_and_the_next_key_still_connects(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _scripted(scenario)
        model: Final = _deployment(gateway, scenario, handle.scenario_id)
        refused: Final = _run("/v1/realtime", f"model={model}&{TRANSCRIPTION_QUERY}", None)
        assert refused.refused == UNAUTHENTICATED_STATUS, refused
        assert _upgrades(gateway, handle.scenario_id) == (), "the upstream saw an unauthenticated upgrade"
        session: Final = _run("/v1/realtime", f"model={model}&{TRANSCRIPTION_QUERY}", scenario.key())
        _assert_transcription_reached_upstream(gateway, handle.scenario_id, session)


def test_upstream_handshake_refusal_reaches_the_client_and_the_next_key_still_connects(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        unknown: Final = f"realtime-url-unknown-{uuid.uuid4().hex[:12]}"
        key: Final = scenario.key()
        refused_model: Final = _deployment(gateway, scenario, unknown)
        refused: Final = _run("/v1/realtime", f"model={refused_model}&{TRANSCRIPTION_QUERY}", key)
        assert refused.types == ("error", "closed"), refused
        assert refused.close_code == UPSTREAM_REFUSAL_CLOSE, refused
        assert [_pairs(upgrade) for upgrade in _upgrades(gateway, unknown)] == [TRANSCRIPTION_PAIRS]
        handle: Final = _scripted(scenario)
        model: Final = _deployment(gateway, scenario, handle.scenario_id)
        session: Final = _run("/v1/realtime", f"model={model}&{TRANSCRIPTION_QUERY}", scenario.key())
        _assert_transcription_reached_upstream(gateway, handle.scenario_id, session)


def test_unknown_model_is_rejected_before_any_upstream_upgrade(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _scripted(scenario)
        key: Final = scenario.key()
        model: Final = _deployment(gateway, scenario, handle.scenario_id)
        missing: Final = f"realtime-url-missing-{uuid.uuid4().hex[:12]}"
        rejected: Final = _run("/v1/realtime", f"model={missing}&{TRANSCRIPTION_QUERY}", key)
        assert rejected.types == ("error", "closed"), rejected
        assert rejected.close_code == UNKNOWN_MODEL_CLOSE, rejected
        assert "Invalid model" in string_value(object_value(rejected.events[0]["error"])["message"]), rejected
        session: Final = _run("/v1/realtime", f"model={model}&{TRANSCRIPTION_QUERY}", key)
        upgrades: Final = _upgrades(gateway, handle.scenario_id)
        assert [_pairs(upgrade) for upgrade in upgrades] == [TRANSCRIPTION_PAIRS], upgrades
        assert session.types == ("session.created",), session


def test_repeated_conversation_sessions_write_one_spend_row_each(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _scripted(scenario)
        key: Final = scenario.key()
        model: Final = _deployment(gateway, scenario, handle.scenario_id, model=f"openai/{CONVERSATION_MODEL}")
        sessions: Final = tuple(_run("/v1/realtime", f"model={model}", key, turns=1) for _ in range(3))
        assert [session.types for session in sessions] == [("session.created", "response.done")] * 3, sessions
        response_ids: Final = frozenset(session.response_ids[0] for session in sessions)
        assert len(response_ids) == 3, sessions
        rows: Final = _spend_rows(key, 3)
        assert {str(row["call_type"]) for row in rows} == {"_arealtime"}, rows


async def _burst(ws_base: str, model: str, key: str) -> tuple[Session, ...]:
    query: Final = f"model={model}&{TRANSCRIPTION_QUERY}"
    return tuple(await asyncio.gather(*(_session(ws_base, "/v1/realtime", query, key) for _ in range(BURST))))


def test_concurrent_transcription_sessions_each_reach_the_upstream_with_model_and_intent(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _scripted(scenario)
        key: Final = scenario.key()
        model: Final = _deployment(gateway, scenario, handle.scenario_id)
        sessions: Final = asyncio.run(_burst(_ws_base(_proxy_url()), model, key))
        assert [session.types for session in sessions] == [("session.created",)] * BURST, sessions
        upgrades: Final = _upgrades(gateway, handle.scenario_id)
        assert [_pairs(upgrade) for upgrade in upgrades] == [TRANSCRIPTION_PAIRS] * BURST, upgrades
        rows: Final = _spend_rows(key, BURST)
        assert {str(row["call_type"]) for row in rows} == {"_arealtime"}, rows


async def _frames_until_closed(socket: ClientConnection) -> AsyncIterator[dict[str, JsonValue]]:
    try:
        async for message in socket:
            yield JSON_OBJECT.validate_json(message)
    except ConnectionClosed as closed:
        yield {"type": "closed", "code": _close_code(closed)}


async def _hold_until_closed(ws_base: str, query: str, key: str, opened: asyncio.Queue[str]) -> Session:
    async with websockets.connect(
        f"{ws_base}/v1/realtime?{query}", additional_headers={"Authorization": f"Bearer {key}"}
    ) as socket:
        created: Final = JSON_OBJECT.validate_json(await socket.recv())
        assert created.get("type") == "session.created", created
        await opened.put(string_value(object_value(created["session"])["id"]))
        return Session(tuple([frame async for frame in _frames_until_closed(socket)]), None)


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
def test_upstream_outage_closes_every_open_transcription_session_and_recovers(
    gateway: Gateway, tmp_path: Path, record_property: RecordProperty
) -> None:
    with gateway.scenario() as scenario, owned_upstream(tmp_path) as slot:
        scenario_id: Final = f"realtime-url-outage-{uuid.uuid4().hex[:12]}"
        register_scenario(scenario_id, _done_event(), control_url=slot.url)
        key: Final = scenario.key()
        model: Final = _deployment(gateway, scenario, scenario_id, api_base=slot.url)
        held: Final = asyncio.run(_burst_through_outage(_ws_base(_proxy_url()), _proxy_url(), model, key, slot.stop))
        record_property("close_codes_during_upstream_outage", sorted(session.close_code or 0 for session in held))
        assert [session.types for session in held] == [("error", "closed")] * BURST, held
        assert all(_relays_the_upstream_close(session) for session in held), held
        assert len({session.close_code for session in held}) == 1, held
        slot.start()
        register_scenario(scenario_id, _done_event(), control_url=slot.url)
        recovered: Final = _run("/v1/realtime", f"model={model}&{TRANSCRIPTION_QUERY}", key)
        assert recovered.types == ("session.created",), recovered
        assert recovered.session_model == TRANSCRIBE_MODEL, recovered
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


async def _one_turn_or_close(socket: ClientConnection) -> bool:
    try:
        await socket.send(json.dumps({"type": "response.create"}))
        async for message in socket:
            if JSON_OBJECT.validate_json(message).get("type") == "response.done":
                return True
    except ConnectionClosed:
        return False
    raise AssertionError("session ended without response.done or a close frame")


async def _first_frames(sockets: tuple[ClientConnection, ...]) -> tuple[dict[str, JsonValue], ...]:
    return tuple([JSON_OBJECT.validate_json(await socket.recv()) for socket in sockets])


async def _sessions_through_worker_kill(ws_base: str, model: str, key: str, root: psutil.Process) -> KillOutcome:
    query: Final = f"model={model}&{TRANSCRIPTION_QUERY}"
    headers: Final = {"Authorization": f"Bearer {key}"}
    sockets: Final = tuple(
        [
            await websockets.connect(f"{ws_base}/v1/realtime?{query}", additional_headers=headers)
            for _ in range(OPEN_SESSIONS)
        ]
    )
    try:
        created: Final = await asyncio.wait_for(_first_frames(sockets), 60)
        assert [event.get("type") for event in created] == ["session.created"] * OPEN_SESSIONS, created
        workers: Final = _workers(root)
        assert len(workers) == WORKERS, [process.pid for process in workers]
        victim: Final = workers[0]
        victim.kill()
        await asyncio.to_thread(victim.wait, 10)
        served: Final = await asyncio.wait_for(asyncio.gather(*(_one_turn_or_close(socket) for socket in sockets)), 60)
        return KillOutcome(served.count(True), served.count(False), victim.pid)
    finally:
        await asyncio.gather(*(socket.close() for socket in sockets))


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
def test_worker_kill_then_proxy_restart_keep_transcription_sessions_serving(
    gateway: Gateway, tmp_path: Path, record_property: RecordProperty
) -> None:
    overrides: Final = {"DATABASE_URL": os.environ["DATABASE_URL"]}
    with gateway.scenario() as scenario:
        handle: Final = _scripted(scenario)
        key: Final = scenario.key()
        with owned_proxy_process(gateway, tmp_path, overrides, workers=WORKERS) as owned:
            owned_url: Final = str(owned.gateway.client.base_url).rstrip("/")
            model: Final = _named_deployment(
                owned.gateway,
                scenario,
                f"realtime-url-owned-{uuid.uuid4().hex[:8]}",
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
            after_kill: Final = _run(
                "/v1/realtime", f"model={model}&{TRANSCRIPTION_QUERY}", key, ws_base=_ws_base(owned_url)
            )
            assert after_kill.types == ("session.created",), after_kill
            codes: Final = asyncio.run(
                _sessions_through_proxy_shutdown(
                    _ws_base(owned_url), model, key, lambda: stop_root_process(owned.process)
                )
            )
            record_property("close_codes_during_proxy_shutdown", sorted(codes))
            assert len(codes) == OPEN_SESSIONS, codes
        with owned_proxy_process(gateway, tmp_path, overrides, workers=WORKERS) as restarted:
            restarted_url: Final = str(restarted.gateway.client.base_url).rstrip("/")
            recovered: Final = _run(
                "/v1/realtime", f"model={model}&{TRANSCRIPTION_QUERY}", key, ws_base=_ws_base(restarted_url)
            )
            assert recovered.types == ("session.created",), recovered
            assert recovered.session_model == TRANSCRIBE_MODEL, recovered
        upgrades: Final = _upgrades(gateway, handle.scenario_id)
        assert len(upgrades) == OPEN_SESSIONS * 2 + 2, upgrades
        assert {json.dumps(_pairs(upgrade)) for upgrade in upgrades} == {json.dumps(TRANSCRIPTION_PAIRS)}, upgrades
