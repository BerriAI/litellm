"""Shared helpers for the ``test_dec1_realtime_*`` contract files.

Two upstreams are used. The shared scripted upstream (``_support/upstream.py``) answers ``/v1/realtime``,
the Gemini AI Studio Live path and the Vertex Live path, and records every handshake and frame in
``/__observations``; it is enough for relay, routing and billing. An owned peer started here
(``realtime_peer``) is a plain ``ws://`` websockets server whose every reaction is scripted per frame, so a
test can make the provider close with a chosen code, drop the TCP connection, refuse the handshake with an
HTTP status, or send an oversized frame. OpenAI realtime dials ``ws://`` for an ``http://`` ``api_base``
(``OpenAIRealtime._construct_url``), so the peer needs no TLS.

The client side is driven with the ``websockets`` asyncio client through a small step list (send a frame,
wait for an event type, wait for the close) and returns an ``Outcome`` with the handshake status, the
negotiated subprotocol, every frame received and the close code and reason.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from http import HTTPStatus
from queue import SimpleQueue
from typing import Final, TypeAlias
from urllib.parse import parse_qsl, urlsplit

import httpx
import websockets
from pydantic import JsonValue
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed, InvalidStatus

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value
from tests.integration._support.upstream import ScenarioHandle, delete_scenario, register_scenario
from tests.integration.cost_calculation.cost_tracking_case import RealtimeResponse

REALTIME_MODEL: Final = "gpt-4o-realtime-preview"
GEMINI_LIVE_MODEL: Final = "gemini-2.5-flash-native-audio-latest"
INPUT_RATE: Final = 0.001
OUTPUT_RATE: Final = 0.002
SPEND_SQL: Final = (
    "SELECT request_id, call_type, spend, prompt_tokens, completion_tokens, total_tokens, status, model, "
    'model_group, model_id, custom_llm_provider, api_base, metadata, "startTime"::text AS "startTime", "endTime"::text AS "endTime" '
    'FROM "LiteLLM_SpendLogs" WHERE api_key = %s ORDER BY "LiteLLM_SpendLogs"."startTime"'
)
POLICY_VIOLATION: Final = 1008
INTERNAL_ERROR: Final = 1011
TRY_AGAIN_LATER: Final = 1013
NORMAL_CLOSURE: Final = 1000


def key_hash(key: str) -> str:
    return sha256(key.encode()).hexdigest()


def proxy_ws_base(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/").replace("https://", "wss://").replace("http://", "ws://")


# --------------------------------------------------------------------------------------------------------
# client driver
# --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Send:
    """Send one frame to the proxy: a dict is JSON encoded, a str or bytes is sent as is."""

    payload: Mapping[str, JsonValue] | str | bytes


@dataclass(frozen=True, slots=True)
class Await:
    """Read frames until one of this ``type`` arrives (or the socket closes)."""

    event_type: str


@dataclass(frozen=True, slots=True)
class Idle:
    """Read for ``seconds`` expecting nothing; any frame that arrives is recorded."""

    seconds: float


@dataclass(frozen=True, slots=True)
class AwaitClose:
    """Read until the proxy closes the socket."""


@dataclass(frozen=True, slots=True)
class CloseWith:
    """Close the client side with ``code``."""

    code: int = NORMAL_CLOSURE


Step: TypeAlias = Send | Await | Idle | AwaitClose | CloseWith


@dataclass(frozen=True, slots=True)
class Outcome:
    handshake_status: int | None
    subprotocol: str | None
    frames: tuple[dict[str, JsonValue], ...]
    raw_texts: tuple[str, ...]
    close_code: int | None
    close_reason: str

    @property
    def types(self) -> tuple[str, ...]:
        return tuple(str(frame.get("type")) for frame in self.frames)

    def of_type(self, event_type: str) -> tuple[dict[str, JsonValue], ...]:
        return tuple(frame for frame in self.frames if frame.get("type") == event_type)

    def error_messages(self) -> tuple[str, ...]:
        return tuple(str(object_value(frame["error"]).get("message")) for frame in self.of_type("error"))

    def error_types(self) -> tuple[str, ...]:
        return tuple(str(object_value(frame["error"]).get("type")) for frame in self.of_type("error"))


def _decode(message: str | bytes) -> tuple[str, dict[str, JsonValue]]:
    text: Final = message if isinstance(message, str) else message.decode("utf-8", errors="replace")
    try:
        return text, JSON_OBJECT.validate_json(text)
    except ValueError:
        return text, {"type": "__non_json__", "text": text}


async def _drive(
    url: str,
    headers: Mapping[str, str],
    subprotocols: Sequence[str] | None,
    steps: Sequence[Step],
    timeout: float,
) -> Outcome:
    frames: Final[list[dict[str, JsonValue]]] = []
    texts: Final[list[str]] = []
    try:
        async with websockets.connect(
            url,
            additional_headers=dict(headers),
            subprotocols=[websockets.Subprotocol(value) for value in subprotocols] if subprotocols else None,
            max_size=None,
            open_timeout=15,
        ) as socket:
            negotiated: Final = socket.subprotocol
            try:
                for step in steps:
                    match step:
                        case Send(payload=payload):
                            await socket.send(json.dumps(payload) if isinstance(payload, Mapping) else payload)
                        case Await(event_type=event_type):
                            while True:
                                text, frame = _decode(await asyncio.wait_for(socket.recv(), timeout))
                                texts.append(text)
                                frames.append(frame)
                                if frame.get("type") == event_type:
                                    break
                        case Idle(seconds=seconds):
                            try:
                                while True:
                                    text, frame = _decode(await asyncio.wait_for(socket.recv(), seconds))
                                    texts.append(text)
                                    frames.append(frame)
                            except TimeoutError:
                                pass
                        case AwaitClose():
                            while True:
                                text, frame = _decode(await asyncio.wait_for(socket.recv(), timeout))
                                texts.append(text)
                                frames.append(frame)
                        case CloseWith(code=code):
                            await socket.close(code=code)
                            return Outcome(None, negotiated, tuple(frames), tuple(texts), code, "")
                await socket.close()
                return Outcome(None, negotiated, tuple(frames), tuple(texts), None, "")
            except ConnectionClosed as closed:
                received: Final = closed.rcvd
                return Outcome(
                    None,
                    negotiated,
                    tuple(frames),
                    tuple(texts),
                    1006 if received is None else received.code,
                    "" if received is None else received.reason,
                )
    except InvalidStatus as refused:
        return Outcome(refused.response.status_code, None, (), (), None, "")


def realtime(
    gateway: Gateway,
    query: str,
    *,
    key: str | None = None,
    headers: Mapping[str, str] | None = None,
    subprotocols: Sequence[str] | None = None,
    steps: Sequence[Step] = (Await("session.created"),),
    path: str = "/v1/realtime",
    timeout: float = 30,
) -> Outcome:
    """Open ``<path>?<query>`` on the proxy and run ``steps``. ``key`` is sent as ``Authorization: Bearer``."""
    all_headers: Final = {**({} if key is None else {"Authorization": f"Bearer {key}"}), **(headers or {})}
    suffix: Final = f"?{query}" if query else ""
    return asyncio.run(_drive(f"{proxy_ws_base(gateway)}{path}{suffix}", all_headers, subprotocols, steps, timeout))


def turn_steps(turns: int, *, first: Sequence[Step] = (Await("session.created"),)) -> tuple[Step, ...]:
    """``session.created`` then ``turns`` rounds of ``response.create`` answered by ``response.done``."""
    rounds: Final = tuple(
        step for _ in range(turns) for step in (Send({"type": "response.create"}), Await("response.done"))
    )
    return (*first, *rounds)


# --------------------------------------------------------------------------------------------------------
# scripted shared upstream
# --------------------------------------------------------------------------------------------------------


def done_event(
    *,
    text_in: int = 10,
    audio_in: int = 0,
    text_out: int = 5,
    audio_out: int = 0,
    cached_text: int = 0,
    cached_audio: int = 0,
    response_id: str = "resp_$UNIQUE_ID",
) -> dict[str, JsonValue]:
    """A GA ``response.done`` whose usage carries the given text, audio and cached token counts."""
    return {
        "type": "response.done",
        "event_id": "evt_$UNIQUE_ID",
        "response": {
            "id": response_id,
            "object": "realtime.response",
            "status": "completed",
            "output": [],
            "usage": {
                "total_tokens": text_in + audio_in + text_out + audio_out,
                "input_tokens": text_in + audio_in,
                "output_tokens": text_out + audio_out,
                "input_token_details": {
                    "text_tokens": text_in,
                    "audio_tokens": audio_in,
                    "cached_tokens": cached_text + cached_audio,
                    "cached_tokens_details": {"text_tokens": cached_text, "audio_tokens": cached_audio},
                },
                "output_token_details": {"text_tokens": text_out, "audio_tokens": audio_out},
            },
        },
    }


def scripted(
    scenario: Scenario,
    events: Sequence[Mapping[str, JsonValue]],
    *,
    session_model: str | None = None,
    session_type: str | None = None,
    control_url: str | None = None,
) -> ScenarioHandle:
    """Register a realtime scenario on the shared scripted upstream. Each ``response.create`` (or
    ``input_audio_buffer.commit``) the upstream receives is answered with the next event of ``events``."""
    response: Final = RealtimeResponse(
        content_type="application/x-realtime",
        events=tuple(dict(event) for event in events),
        session_model=session_model,
        session_type=session_type,
    )
    scenario_id: Final = f"dec1-rt-{uuid.uuid4().hex[:12]}"
    handle: Final = (
        register_scenario(scenario_id, response)
        if control_url is None
        else register_scenario(scenario_id, response, control_url=control_url)
    )
    scenario.cleanups.callback(delete_scenario, handle)
    return handle


def openai_deployment(
    scenario: Scenario, handle: ScenarioHandle, *, model: str = f"openai/{REALTIME_MODEL}", **params: JsonValue
) -> str:
    """An OpenAI realtime deployment on the shared scripted upstream; the scenario id is the provider key."""
    return scenario.model(
        model=model,
        api_key=handle.scenario_id,
        api_base=scenario.gateway.upstream_url.rstrip("/"),
        **params,
    )


def priced_deployment(scenario: Scenario, handle: ScenarioHandle, **params: JsonValue) -> str:
    return openai_deployment(
        scenario, handle, input_cost_per_token=INPUT_RATE, output_cost_per_token=OUTPUT_RATE, **params
    )


def observations(gateway: Gateway) -> tuple[dict[str, JsonValue], ...]:
    with httpx.Client(base_url=gateway.upstream_url, timeout=10, trust_env=False) as upstream:
        return tuple(map(object_value, upstream.get("/__observations").json()["requests"]))


def _belongs(entry: Mapping[str, JsonValue], scenario_id: str) -> bool:
    return entry.get("authorization") == f"Bearer {scenario_id}" or entry.get("api_key") == scenario_id


def upgrades(gateway: Gateway, scenario_id: str) -> tuple[dict[str, JsonValue], ...]:
    """The upstream handshakes made with the scenario's provider key, in arrival order."""
    return tuple(
        entry for entry in observations(gateway) if entry.get("method") == "WEBSOCKET" and _belongs(entry, scenario_id)
    )


def upstream_frames(gateway: Gateway, scenario_id: str) -> tuple[dict[str, JsonValue], ...]:
    """The frame bodies the upstream received on the scenario's sockets, in arrival order."""
    return tuple(
        object_value(entry["body"])
        for entry in observations(gateway)
        if entry.get("method") == "WEBSOCKET_FRAME" and _belongs(entry, scenario_id)
    )


def upgrade_query(upgrade: Mapping[str, JsonValue]) -> JsonValue:
    return object_value(upgrade["body"])["query"]


def spend_rows(gateway: Gateway, key: str, count: int, seconds: float = 70) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: gateway.rows(SPEND_SQL, (key_hash(key),)),
        lambda rows: len(rows) >= count,
        seconds=seconds,
    )


def key_spend(gateway: Gateway, key: str) -> float:
    rows: Final = gateway.rows('SELECT spend FROM "LiteLLM_VerificationToken" WHERE token = %s', (key_hash(key),))
    return float(str(rows[0]["spend"])) if rows else 0.0


# --------------------------------------------------------------------------------------------------------
# owned provider peer
# --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Reply:
    event: Mapping[str, JsonValue] | str | bytes


@dataclass(frozen=True, slots=True)
class Close:
    code: int
    reason: str = ""


@dataclass(frozen=True, slots=True)
class DropTcp:
    """Abort the TCP connection without a close frame."""


Action: TypeAlias = Reply | Close | DropTcp
FrameScript: TypeAlias = Callable[[dict[str, JsonValue]], tuple[Action, ...]]


def answer_response_create(event: Mapping[str, JsonValue]) -> FrameScript:
    def script(frame: dict[str, JsonValue]) -> tuple[Action, ...]:
        return (Reply(event),) if frame.get("type") == "response.create" else ()

    return script


@dataclass(frozen=True, slots=True)
class Handshake:
    path: str
    query: tuple[tuple[str, str], ...]
    headers: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class Peer:
    url: str
    handshakes: SimpleQueue[Handshake]
    frames: SimpleQueue[dict[str, JsonValue]]
    closes: SimpleQueue[tuple[int | None, str]]

    def drained_handshakes(self) -> tuple[Handshake, ...]:
        return tuple(self.handshakes.get_nowait() for _ in range(self.handshakes.qsize()))

    def drained_frames(self) -> tuple[dict[str, JsonValue], ...]:
        return tuple(self.frames.get_nowait() for _ in range(self.frames.qsize()))


@dataclass(frozen=True, slots=True)
class PeerScript:
    """What the owned peer does: refuse the handshake with ``refuse_status``, or accept, send ``opening``
    events, then answer each client frame with ``on_frame``."""

    refuse_status: int | None = None
    opening: tuple[Action, ...] = (
        Reply(
            {"type": "session.created", "event_id": "evt_open", "session": {"id": "sess_peer", "model": REALTIME_MODEL}}
        ),
    )
    on_frame: FrameScript = field(default=lambda _frame: ())


async def _act(connection: ServerConnection, actions: Sequence[Action]) -> bool:
    for action in actions:
        match action:
            case Reply(event=event):
                await connection.send(json.dumps(event) if isinstance(event, Mapping) else event)
            case Close(code=code, reason=reason):
                await connection.close(code=code, reason=reason)
                return False
            case DropTcp():
                connection.transport.abort()
                return False
    return True


async def _serve_peer(script: PeerScript, peer: Peer, ports: SimpleQueue[int], stop: asyncio.Event) -> None:
    def process_request(connection: ServerConnection, request: websockets.http11.Request) -> object:
        split: Final = urlsplit(request.path)
        peer.handshakes.put(
            Handshake(split.path, tuple(parse_qsl(split.query)), {k.lower(): v for k, v in request.headers.raw_items()})
        )
        if script.refuse_status is not None:
            return connection.respond(HTTPStatus(script.refuse_status), "scripted refusal\n")
        return None

    async def handler(connection: ServerConnection) -> None:
        try:
            if not await _act(connection, script.opening):
                return
            async for raw in connection:
                text = raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")
                try:
                    frame = JSON_OBJECT.validate_json(text)
                except ValueError:
                    frame = {"type": "__non_json__", "text": text, "binary": not isinstance(raw, str)}
                peer.frames.put(frame)
                if not await _act(connection, script.on_frame(frame)):
                    return
        except ConnectionClosed:
            pass
        finally:
            peer.closes.put((connection.close_code, connection.close_reason or ""))

    async with serve(handler, "127.0.0.1", 0, process_request=process_request, max_size=None) as server:
        ports.put(next(iter(server.sockets)).getsockname()[1])
        await stop.wait()


@contextmanager
def realtime_peer(script: PeerScript | None = None) -> Iterator[Peer]:
    """A scripted ``ws://`` OpenAI-style realtime provider on 127.0.0.1. ``Peer.url`` is the ``api_base``."""
    loop: Final = asyncio.new_event_loop()
    stop: Final = asyncio.Event()
    ports: Final = SimpleQueue[int]()
    peer_queues: Final = (
        SimpleQueue[Handshake](),
        SimpleQueue[dict[str, JsonValue]](),
        SimpleQueue[tuple[int | None, str]](),
    )
    holder: Final[list[Peer]] = []

    def run() -> None:
        loop.run_until_complete(_serve_peer(script or PeerScript(), holder[0], ports, stop))

    port_peer: Final = Peer("", *peer_queues)
    holder.append(port_peer)
    thread: Final = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        port: Final = ports.get(timeout=10)
        yield Peer(f"http://127.0.0.1:{port}", *peer_queues)
    finally:
        loop.call_soon_threadsafe(stop.set)
        thread.join(timeout=10)
        loop.close()


def peer_deployment(scenario: Scenario, peer: Peer, **params: JsonValue) -> str:
    return scenario.model(
        model=f"openai/{REALTIME_MODEL}",
        api_key="dec1-realtime-peer-key",
        api_base=peer.url,
        input_cost_per_token=INPUT_RATE,
        output_cost_per_token=OUTPUT_RATE,
        **params,
    )


def proxy_url() -> str:
    return os.environ["INTEGRATION_PROXY_URL"].rstrip("/")
