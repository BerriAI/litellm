"""Vertex AI Live sessions resolve their host through the shared Vertex resolver.

``VertexAIRealtimeConfig.get_complete_url`` dials ``aiplatform.{us,eu}.rep.googleapis.com`` for the
multi-regions, ``{region}-aiplatform.googleapis.com`` for a region and ``aiplatform.googleapis.com`` for
``global``, and a malformed location fails the shared validator before any socket opens. Every row here
runs against the scripted upstream on 127.0.0.1: an ``api_base`` override pins the path and the host the
upstream sees, the setup frame it receives names the location the proxy resolved, and the malformed rows
pin the validator's answer, which the proxy gives without dialing anything.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from hashlib import sha256
from itertools import chain, repeat
from pathlib import Path
from typing import Final

import httpx
import pytest
import websockets
from pydantic import JsonValue
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value, string_value
from tests.integration._support.database import read_rows
from tests.integration._support.process import owned_upstream
from tests.integration._support.upstream import GEMINI_LIVE_PATH, ScenarioHandle, delete_scenario, register_scenario
from tests.integration.cost_calculation.cost_tracking_case import RealtimeResponse

RecordProperty = Callable[[str, object], None]

pytestmark: Final = pytest.mark.timeout(180)

LIVE_MODEL: Final = "gemini-3.8-live"
LOCATIONS: Final = ("us", "eu", "us-central1", "global")
MALFORMED_LOCATIONS: Final = ("US", "us/evil")
DEFAULT_LOCATION: Final = "us-central1"
INVALID_LOCATION: Final = "Invalid vertex_location format"
HANDSHAKE_REFUSED: Final = "Upstream realtime handshake rejected with HTTP 403"
INTERNAL_CLOSE: Final = 1011
REFUSAL_CLOSE: Final = 1008
SESSIONS_PER_LOCATION: Final = 3
OUTAGE_BURST: Final = 20
INPUT_TOKENS: Final = 7
OUTPUT_TOKENS: Final = 5
CONVERGENCE_SECONDS: Final = 30
SPEND_SQL: Final = 'SELECT call_type, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE api_key = %s'


@dataclass(frozen=True, slots=True)
class Session:
    events: tuple[dict[str, JsonValue], ...]

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

    @property
    def text(self) -> str:
        deltas: Final = tuple(event for event in self.events if event["type"] == "response.output_text.delta")
        return "".join(string_value(event["delta"]) for event in deltas)

    def completed_turn(self, scenario_id: str) -> bool:
        return (
            self.types[0] == "session.created"
            and self.types[-1] == "response.done"
            and self.session_model == LIVE_MODEL
            and self.text == f"scripted {scenario_id}"
            and len(self.response_ids) == 1
        )


def _integer(value: JsonValue) -> int:
    assert isinstance(value, int), value
    return value


def _ws_base(http_url: str) -> str:
    return http_url.replace("https://", "wss://").replace("http://", "ws://")


def _proxy_url(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


def _upstream_url(gateway: Gateway) -> str:
    return gateway.upstream_url.rstrip("/")


def _scripted_turn() -> RealtimeResponse:
    return RealtimeResponse(
        content_type="application/x-realtime",
        events=(
            {"serverContent": {"modelTurn": {"parts": [{"text": "scripted $REQUEST_ID"}]}}},
            {
                "serverContent": {"turnComplete": True},
                "usageMetadata": {
                    "promptTokenCount": INPUT_TOKENS,
                    "responseTokenCount": OUTPUT_TOKENS,
                    "totalTokenCount": INPUT_TOKENS + OUTPUT_TOKENS,
                },
            },
        ),
    )


def _scripted(scenario: Scenario) -> ScenarioHandle:
    handle: Final = register_scenario(f"vertex-live-{uuid.uuid4().hex[:12]}", _scripted_turn())
    scenario.cleanups.callback(delete_scenario, handle)
    return handle


def _credentials(upstream_url: str) -> str:
    return json.dumps(
        {
            "type": "external_account",
            "audience": "synthetic-vertex-audience",
            "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
            "token_url": f"{upstream_url}/_oauth/token",
            "credential_source": {"url": f"{upstream_url}/health"},
        }
    )


def _deployment(
    gateway: Gateway, scenario: Scenario, project: str, *, location: str | None, api_base: str | None
) -> str:
    name: Final = f"vertex-live-{uuid.uuid4().hex[:12]}"
    litellm_params: Final[dict[str, JsonValue]] = {
        "model": f"vertex_ai/{LIVE_MODEL}",
        "vertex_project": project,
        "vertex_credentials": _credentials(_upstream_url(gateway)),
        **({} if location is None else {"vertex_location": location}),
        **({} if api_base is None else {"api_base": api_base}),
    }
    created: Final = gateway.post(
        "/model/new", {"model_name": name, "litellm_params": litellm_params, "model_info": {"mode": "realtime"}}
    )
    scenario.cleanups.callback(scenario.delete_model, string_value(object_value(created["model_info"])["id"]))
    return name


def _model_path(project: str, location: str) -> str:
    return f"projects/{project}/locations/{location}/publishers/google/models/{LIVE_MODEL}"


def _user_turn() -> str:
    return json.dumps(
        {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "say the scripted line"}],
            },
        }
    )


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
            await socket.send(_user_turn())
            async for message in socket:
                event: Final = JSON_OBJECT.validate_json(message)
                yield event
                if event.get("type") == "response.done":
                    break
    except ConnectionClosed as closed:
        yield {"type": "closed", "code": _close_code(closed)}


async def _collect(socket: ClientConnection, turns: int) -> tuple[dict[str, JsonValue], ...]:
    return tuple([frame async for frame in _frames(socket, turns)])


async def _session(ws_base: str, model: str, key: str, *, turns: int = 1) -> Session:
    headers: Final = {"Authorization": f"Bearer {key}"}
    async with websockets.connect(f"{ws_base}/v1/realtime?model={model}", additional_headers=headers) as socket:
        return Session(await asyncio.wait_for(_collect(socket, turns), 60))


def _run(gateway: Gateway, model: str, key: str, *, turns: int = 1) -> Session:
    return asyncio.run(_session(_ws_base(_proxy_url(gateway)), model, key, turns=turns))


def _observations(gateway: Gateway) -> tuple[dict[str, JsonValue], ...]:
    with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream:
        return tuple(map(object_value, upstream.get("/__observations").json()["requests"]))


def _for_project(observed: tuple[dict[str, JsonValue], ...], project: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(request for request in observed if request["api_key"] == project)


def _upgrade_hosts(observed: tuple[dict[str, JsonValue], ...]) -> tuple[str, ...]:
    upgrades: Final = tuple(request for request in observed if request["method"] == "WEBSOCKET")
    assert all(request["path"] == GEMINI_LIVE_PATH for request in upgrades), upgrades
    return tuple(string_value(object_value(request["body"])["host"]) for request in upgrades)


def _setup_models(observed: tuple[dict[str, JsonValue], ...]) -> tuple[str, ...]:
    frames: Final = tuple(
        object_value(request["body"]) for request in observed if request["method"] == "WEBSOCKET_FRAME"
    )
    return tuple(string_value(object_value(frame["setup"])["model"]) for frame in frames if "setup" in frame)


def _authority(gateway: Gateway) -> str:
    return _upstream_url(gateway).removeprefix("http://")


def _spend_rows(key: str, count: int) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows(SPEND_SQL, (sha256(key.encode()).hexdigest(),)),
        lambda rows: len(rows) == count,
        seconds=70,
    )


def _billed(rows: list[dict[str, JsonValue]]) -> list[tuple[JsonValue, JsonValue, JsonValue]]:
    return [(row["call_type"], row["prompt_tokens"], row["completion_tokens"]) for row in rows]


def _health(gateway: Gateway, model: str) -> httpx.Response:
    return gateway.request("GET", "/health", params={"model": model})


def _probed_count(response: httpx.Response) -> int:
    body: Final = JSON_OBJECT.validate_json(response.content)
    return _integer(body.get("healthy_count", 0)) + _integer(body.get("unhealthy_count", 0))


def _converged_health(gateway: Gateway, model: str) -> httpx.Response:
    return eventually(
        lambda: _health(gateway, model), lambda response: _probed_count(response) == 1, seconds=CONVERGENCE_SECONDS
    )


def _assert_scripted_turn_reached_upstream(gateway: Gateway, project: str, location: str, session: Session) -> None:
    assert session.completed_turn(project), session
    observed: Final = _for_project(_observations(gateway), project)
    assert _upgrade_hosts(observed) == (_authority(gateway),), observed
    assert _setup_models(observed) == (_model_path(project, location),), observed


@pytest.mark.parametrize("location", LOCATIONS)
def test_api_base_override_completes_a_turn_for_every_location(gateway: Gateway, location: str) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _scripted(scenario)
        key: Final = scenario.key()
        model: Final = _deployment(
            gateway, scenario, handle.scenario_id, location=location, api_base=_upstream_url(gateway)
        )
        session: Final = _run(gateway, model, key)
        _assert_scripted_turn_reached_upstream(gateway, handle.scenario_id, location, session)
        assert _billed(_spend_rows(key, 1)) == [("_arealtime", INPUT_TOKENS, OUTPUT_TOKENS)]


@pytest.mark.parametrize("location", MALFORMED_LOCATIONS)
def test_malformed_location_is_refused_by_the_validator_before_any_dial(gateway: Gateway, location: str) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _scripted(scenario)
        key: Final = scenario.key()
        malformed: Final = _deployment(gateway, scenario, handle.scenario_id, location=location, api_base=None)
        refused: Final = _run(gateway, malformed, key)
        assert refused.types == ("error", "closed"), refused
        assert refused.error_messages == (INVALID_LOCATION,), refused
        assert refused.close_code == INTERNAL_CLOSE, refused
        model: Final = _deployment(
            gateway, scenario, handle.scenario_id, location="us", api_base=_upstream_url(gateway)
        )
        session: Final = _run(gateway, model, key)
        _assert_scripted_turn_reached_upstream(gateway, handle.scenario_id, "us", session)


@pytest.mark.parametrize("location", [None, ""], ids=["omitted", "empty"])
def test_missing_location_defaults_to_us_central1(gateway: Gateway, location: str | None) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _scripted(scenario)
        key: Final = scenario.key()
        model: Final = _deployment(
            gateway, scenario, handle.scenario_id, location=location, api_base=_upstream_url(gateway)
        )
        session: Final = _run(gateway, model, key)
        _assert_scripted_turn_reached_upstream(gateway, handle.scenario_id, DEFAULT_LOCATION, session)


@pytest.mark.parametrize("location", ["us", "eu"])
def test_health_check_handshakes_with_the_override_host(gateway: Gateway, location: str) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _scripted(scenario)
        model: Final = _deployment(
            gateway, scenario, handle.scenario_id, location=location, api_base=_upstream_url(gateway)
        )
        health: Final = _converged_health(gateway, model)
        assert health.status_code == 200, health.text
        body: Final = JSON_OBJECT.validate_json(health.content)
        assert (body["healthy_count"], body["unhealthy_count"]) == (1, 0), health.text
        observed: Final = _for_project(_observations(gateway), handle.scenario_id)
        assert _upgrade_hosts(observed) == (_authority(gateway),), observed
        assert _setup_models(observed) == (), observed


def test_health_check_reports_a_malformed_location_without_dialing(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        handle: Final = _scripted(scenario)
        model: Final = _deployment(gateway, scenario, handle.scenario_id, location="US", api_base=None)
        health: Final = _converged_health(gateway, model)
        assert health.status_code == 503, health.text
        body: Final = JSON_OBJECT.validate_json(health.content)
        assert (body["healthy_count"], body["unhealthy_count"]) == (0, 1), health.text
        unhealthy: Final = body["unhealthy_endpoints"]
        assert isinstance(unhealthy, list) and len(unhealthy) == 1, health.text
        assert INVALID_LOCATION in string_value(object_value(unhealthy[0])["error"]), health.text
        assert _for_project(_observations(gateway), handle.scenario_id) == (), "the upstream saw a dial"


def test_upstream_handshake_refusal_reaches_the_client_and_the_next_session_connects(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        unregistered: Final = f"vertex-live-unknown-{uuid.uuid4().hex[:12]}"
        refused_model: Final = _deployment(
            gateway, scenario, unregistered, location="us", api_base=_upstream_url(gateway)
        )
        refused: Final = _run(gateway, refused_model, key)
        assert refused.types == ("error", "closed"), refused
        assert refused.error_messages == (HANDSHAKE_REFUSED,), refused
        assert refused.close_code == REFUSAL_CLOSE, refused
        assert _upgrade_hosts(_for_project(_observations(gateway), unregistered)) == (_authority(gateway),)
        handle: Final = _scripted(scenario)
        model: Final = _deployment(
            gateway, scenario, handle.scenario_id, location="us", api_base=_upstream_url(gateway)
        )
        session: Final = _run(gateway, model, key)
        _assert_scripted_turn_reached_upstream(gateway, handle.scenario_id, "us", session)
        chat: Final = gateway.chat(scenario.model(), key=key)
        assert string_value(chat["object"]) == "chat.completion", chat


async def _burst(ws_base: str, models: tuple[str, ...], key: str) -> tuple[Session, ...]:
    return tuple(await asyncio.gather(*(_session(ws_base, model, key) for model in models)))


def test_concurrent_sessions_across_locations_each_reach_the_upstream_once(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        handles: Final = {location: _scripted(scenario) for location in LOCATIONS}
        models: Final = {
            location: _deployment(
                gateway, scenario, handles[location].scenario_id, location=location, api_base=_upstream_url(gateway)
            )
            for location in LOCATIONS
        }
        order: Final = tuple(chain.from_iterable(repeat(location, SESSIONS_PER_LOCATION) for location in LOCATIONS))
        sessions: Final = asyncio.run(
            _burst(_ws_base(_proxy_url(gateway)), tuple(models[location] for location in order), key)
        )
        assert all(
            session.completed_turn(handles[location].scenario_id) for location, session in zip(order, sessions)
        ), sessions
        assert len({session.response_ids[0] for session in sessions}) == len(order), sessions
        observed: Final = _observations(gateway)
        for location, handle in handles.items():
            mine: Final = _for_project(observed, handle.scenario_id)
            assert _upgrade_hosts(mine) == (_authority(gateway),) * SESSIONS_PER_LOCATION, mine
            assert _setup_models(mine) == (_model_path(handle.scenario_id, location),) * SESSIONS_PER_LOCATION, mine
        assert _billed(_spend_rows(key, len(order))) == [("_arealtime", INPUT_TOKENS, OUTPUT_TOKENS)] * len(order)


async def _frames_until_closed(socket: ClientConnection) -> AsyncIterator[dict[str, JsonValue]]:
    try:
        async for message in socket:
            yield JSON_OBJECT.validate_json(message)
    except ConnectionClosed as closed:
        yield {"type": "closed", "code": _close_code(closed)}


async def _hold_until_closed(ws_base: str, model: str, key: str, opened: asyncio.Queue[str]) -> Session:
    headers: Final = {"Authorization": f"Bearer {key}"}
    async with websockets.connect(f"{ws_base}/v1/realtime?model={model}", additional_headers=headers) as socket:
        created: Final = JSON_OBJECT.validate_json(await socket.recv())
        assert created.get("type") == "session.created", created
        await opened.put(string_value(object_value(created["session"])["id"]))
        return Session(tuple([frame async for frame in _frames_until_closed(socket)]))


def _relays_the_upstream_close(session: Session) -> bool:
    return f"upstream websocket closed with code {session.close_code}" in session.error_messages[0]


async def _drain(opened: asyncio.Queue[str], count: int) -> tuple[str, ...]:
    return tuple([await opened.get() for _ in range(count)])


async def _burst_through_outage(
    ws_base: str, proxy_url: str, model: str, key: str, stop_upstream: Callable[[], None]
) -> tuple[Session, ...]:
    opened: Final[asyncio.Queue[str]] = asyncio.Queue()
    holders: Final = tuple(
        asyncio.ensure_future(_hold_until_closed(ws_base, model, key, opened)) for _ in range(OUTAGE_BURST)
    )
    opened_sessions: Final = await asyncio.wait_for(_drain(opened, OUTAGE_BURST), 60)
    assert len(opened_sessions) == OUTAGE_BURST, opened_sessions
    await asyncio.to_thread(stop_upstream)
    async with httpx.AsyncClient(base_url=proxy_url, timeout=15, trust_env=False) as client:
        liveliness: Final = await client.get("/health/liveliness")
    assert liveliness.status_code == 200, liveliness.text
    return tuple(await asyncio.wait_for(asyncio.gather(*holders), 90))


@pytest.mark.timeout(240)
def test_upstream_outage_closes_every_open_session_and_recovers(
    gateway: Gateway, tmp_path: Path, record_property: RecordProperty
) -> None:
    with gateway.scenario() as scenario, owned_upstream(tmp_path) as slot:
        project: Final = f"vertex-live-outage-{uuid.uuid4().hex[:12]}"
        register_scenario(project, _scripted_turn(), control_url=slot.url)
        key: Final = scenario.key()
        model: Final = _deployment(gateway, scenario, project, location="us", api_base=slot.url)
        held: Final = asyncio.run(
            _burst_through_outage(_ws_base(_proxy_url(gateway)), _proxy_url(gateway), model, key, slot.stop)
        )
        record_property("close_codes_during_upstream_outage", sorted(session.close_code or 0 for session in held))
        assert [session.types for session in held] == [("error", "closed")] * OUTAGE_BURST, held
        assert all(_relays_the_upstream_close(session) for session in held), held
        assert len({session.close_code for session in held}) == 1, held
        slot.start()
        register_scenario(project, _scripted_turn(), control_url=slot.url)
        recovered: Final = _run(gateway, model, key)
        assert recovered.completed_turn(project), recovered
        rows: Final = _spend_rows(key, OUTAGE_BURST + 1)
        assert {str(row["call_type"]) for row in rows} == {"_arealtime"}, rows
