from __future__ import annotations

import asyncio
import itertools
import json
import re
import ssl
import threading
import time
import uuid
from collections.abc import Generator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, SimpleQueue
from types import MappingProxyType
from typing import Final, TypeVar
from urllib.parse import parse_qs, urlsplit

import httpx
import psutil
import pytest
import websockets
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.database import read_rows, scratch_database
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy, owned_proxy_process
from integration._support.tls import server_context, write_self_signed_cert
from integration._support.upstream import (
    JsonResponse,
    RoutedResponse,
    delete_scenario,
    register_scenario,
)
from pydantic import JsonValue, TypeAdapter, ValidationError
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.typing import Subprotocol

OWNED_PROXY_CELL_SECONDS: Final = 2 * graceful_stop_seconds() + 180
pytestmark: Final = pytest.mark.timeout(max(360.0, OWNED_PROXY_CELL_SECONDS))

PROVIDER_MODEL: Final = "ws-peer-model"
STALL_PROVIDER_MODEL: Final = "ws-stall-peer-model"
DEAF_PROVIDER_MODEL: Final = "ws-deaf-peer-model"
DEAF_RESPONSE_DELAY_SECONDS: Final = 10
DEAF_READ_PAUSE_SECONDS: Final = 20
PEER_TEXT: Final = "responses websocket peer"
TERMINAL: Final = frozenset({"response.completed", "response.failed", "error"})
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
SESSION_LIMIT_FIELD: Final = "responses_websocket_session_limit_seconds"
LIMIT_CLOSE_REASON: Final = "Session duration limit reached"
RELOAD_INTERVAL_SECONDS: Final = 3
WORKER_SYNC_SECONDS: Final = RELOAD_INTERVAL_SECONDS + 7
CAP_SECONDS: Final = 60
RESTORED_IDLE_SECONDS: Final = 75
CAPPED_PATHS: Final = ("/v1/responses", "/responses")
CAPPED_SOCKETS_MINIMUM: Final = 8
CAPPED_SOCKETS_MAXIMUM: Final = 40
KILLED_POOL_SIZE: Final = 8
SUBPROTOCOLS: Final = (Subprotocol("litellm-responses-first"), Subprotocol("litellm-responses-second"))
STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
WORKER_PIDS: Final = TypeAdapter(tuple[int, ...])
CLIENT_ADDRESS: Final = TypeAdapter(tuple[str, int])
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class PeerConnection:
    path: str
    frames: SimpleQueue[dict[str, JsonValue]]
    closed: SimpleQueue[float]


@dataclass(frozen=True, slots=True)
class ResponsesPeer:
    url: str
    connections: SimpleQueue[PeerConnection]
    connections_by_model: Mapping[str, SimpleQueue[PeerConnection]]


@dataclass(frozen=True, slots=True)
class PeerSnapshot:
    path: str
    frames: tuple[dict[str, JsonValue], ...]
    closed: bool


@dataclass(frozen=True, slots=True)
class ReceiveResult:
    timed_out: bool
    closed: bool
    frame: dict[str, JsonValue] | None
    close_code: int | None
    close_reason: str | None


@dataclass(frozen=True, slots=True)
class SessionResult:
    text: str
    idle: ReceiveResult
    events: tuple[dict[str, JsonValue], ...]
    error: str | None


@dataclass(frozen=True, slots=True)
class HandshakeResult:
    status_code: int | None
    error: str | None


@dataclass(frozen=True, slots=True)
class SubprotocolResult:
    text: str
    negotiated: str | None
    events: tuple[dict[str, JsonValue], ...]
    error: str | None


@dataclass(frozen=True, slots=True)
class DefaultResults:
    pool: tuple[SessionResult, ...]
    query: SessionResult
    rejected: HandshakeResult
    subprotocol: SubprotocolResult
    provider: tuple[PeerSnapshot, ...]


@dataclass(frozen=True, slots=True)
class AuthResult:
    idle: ReceiveResult
    frame: dict[str, JsonValue] | None
    close_code: int | None
    close_reason: str | None
    error: str | None


@dataclass(frozen=True, slots=True)
class AuthResults:
    immediate: AuthResult
    delayed: AuthResult
    provider_connections: int


@dataclass(frozen=True, slots=True)
class CloseResult:
    outcome: ReceiveResult
    elapsed: float


@dataclass(frozen=True, slots=True)
class ActiveResult:
    turn: tuple[dict[str, JsonValue], ...]
    turn_error: str | None
    close: CloseResult
    provider_closed: bool


@dataclass(frozen=True, slots=True)
class MidResult:
    created: bool
    turn_error: str | None
    close: CloseResult
    provider_closed: bool
    fresh_completed: bool
    fresh_error: str | None


@dataclass(frozen=True, slots=True)
class CapResults:
    idle: CloseResult
    active: ActiveResult
    mid: MidResult
    deaf: MidResult
    provider: tuple[PeerSnapshot, ...]


@dataclass(frozen=True, slots=True)
class BurstResult:
    path: str
    status_code: int | None
    body: dict[str, JsonValue] | None
    error: str | None


@dataclass(frozen=True, slots=True)
class InvalidResult:
    session: SessionResult
    warning_found: bool


@dataclass(frozen=True, slots=True)
class UpdateResult:
    status_code: int
    body: str


@dataclass(frozen=True, slots=True)
class CappedSocket:
    path: str
    worker_pid: int | None
    close: CloseResult


@dataclass(frozen=True, slots=True)
class OpenedSocket:
    path: str
    worker_pid: int | None
    connection: websockets.ClientConnection
    started: float


@dataclass(frozen=True, slots=True)
class HeldResult:
    held_seconds: float
    events: tuple[dict[str, JsonValue], ...]
    error: str | None


@dataclass(frozen=True, slots=True)
class OverrideResults:
    workers: frozenset[int]
    updates: tuple[UpdateResult, ...]
    stored_after_updates: JsonValue
    capped: tuple[CappedSocket, ...]
    earlier: HeldResult
    delete: UpdateResult
    stored_after_delete: dict[str, JsonValue] | None
    restored: SessionResult


@dataclass(frozen=True, slots=True)
class TurnOutcome:
    events: tuple[dict[str, JsonValue], ...]
    error: str | None


@dataclass(frozen=True, slots=True)
class KillResults:
    held_by: Mapping[int, int]
    victim: int
    victim_closes: tuple[ReceiveResult, ...]
    survivor_turns: tuple[TurnOutcome, ...]
    health_status: int
    fresh_completed: bool


@dataclass(frozen=True, slots=True)
class ChaosResults:
    burst: tuple[BurstResult, ...]
    health_status: int
    websocket_completed: bool
    kill: KillResults


def _object(value: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(value, dict), value
    return value


def _list(value: JsonValue) -> list[JsonValue]:
    assert isinstance(value, list), value
    return value


def _string(value: JsonValue) -> str:
    assert isinstance(value, str), value
    return value


def _drain(queue: SimpleQueue[T]) -> tuple[T, ...]:
    return tuple(queue.get_nowait() for _ in range(queue.qsize()))


def _events(response_id: str, model: str, text: str, *, stall: bool = False) -> tuple[dict[str, JsonValue], ...]:
    message: Final[dict[str, JsonValue]] = {
        "type": "message",
        "id": f"msg_{response_id}",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    response: Final[dict[str, JsonValue]] = {
        "id": response_id,
        "object": "response",
        "created_at": 1700000000,
        "model": model,
    }
    created: Final[dict[str, JsonValue]] = {
        "type": "response.created",
        "response": {**response, "status": "in_progress", "output": []},
    }
    if stall:
        return (created,)
    return (
        created,
        {
            "type": "response.output_text.delta",
            "item_id": f"msg_{response_id}",
            "output_index": 0,
            "content_index": 0,
            "delta": text,
        },
        {
            "type": "response.completed",
            "response": {**response, "status": "completed", "output": [message]},
        },
    )


def _path_model(path: str) -> str | None:
    values: Final = parse_qs(urlsplit(path).query).get("model")
    return values[0] if values else None


async def _peer_handler(connection: ServerConnection, peer: ResponsesPeer) -> None:
    path: Final = connection.request.path if connection.request is not None else ""
    record: Final = PeerConnection(path, SimpleQueue(), SimpleQueue())
    peer.connections.put(record)
    model: Final = _path_model(path)
    if model is not None and model in peer.connections_by_model:
        peer.connections_by_model[model].put(record)
    turns: Final = itertools.count(1)
    try:
        async for raw in connection:
            await _peer_frame(raw, connection, record, turns)
    finally:
        record.closed.put(time.monotonic())


async def _peer_frame(
    raw: str | bytes,
    connection: ServerConnection,
    record: PeerConnection,
    turns: itertools.count[int],
) -> None:
    frame: Final = JSON_OBJECT.validate_json(raw)
    record.frames.put(frame)
    if frame.get("type") != "response.create":
        return
    model: Final = _string(frame.get("model", ""))
    stall: Final = model in (STALL_PROVIDER_MODEL, DEAF_PROVIDER_MODEL)
    if model == DEAF_PROVIDER_MODEL:
        await asyncio.sleep(DEAF_RESPONSE_DELAY_SECONDS)
    for event in _events(f"resp_peer_{next(turns)}", model, PEER_TEXT, stall=stall):
        await connection.send(json.dumps(event))
    if model == DEAF_PROVIDER_MODEL:
        transport: Final = connection.transport
        assert transport is not None
        transport.pause_reading()
        try:
            await asyncio.sleep(DEAF_READ_PAUSE_SECONDS)
        finally:
            transport.resume_reading()


async def _serve_peer(
    tls: ssl.SSLContext,
    peer: ResponsesPeer,
    ports: SimpleQueue[int],
    stop: asyncio.Event,
) -> None:
    async with serve(lambda connection: _peer_handler(connection, peer), "127.0.0.1", 0, ssl=tls) as server:
        address: object = next(iter(server.sockets)).getsockname()  # pyright: ignore[reportAny]  # socket.getsockname is typed Any
        port: Final = TypeAdapter(tuple[str, int]).validate_python(address)[1]
        ports.put(port)
        await stop.wait()


@contextmanager
def responses_peer(cert: tuple[Path, Path]) -> Generator[ResponsesPeer, None, None]:
    loop: Final = asyncio.new_event_loop()
    stop: Final = asyncio.Event()
    ports: Final = SimpleQueue[int]()
    connections_by_model: Final[Mapping[str, SimpleQueue[PeerConnection]]] = MappingProxyType(
        {model: SimpleQueue[PeerConnection]() for model in (PROVIDER_MODEL, STALL_PROVIDER_MODEL, DEAF_PROVIDER_MODEL)}
    )
    peer: Final = ResponsesPeer("", SimpleQueue(), connections_by_model)
    thread: Final = threading.Thread(
        target=loop.run_until_complete,
        args=(_serve_peer(server_context(*cert), peer, ports, stop),),
        daemon=True,
    )
    thread.start()
    try:
        port: Final = ports.get(timeout=10)
        yield ResponsesPeer(f"https://127.0.0.1:{port}/v1", peer.connections, peer.connections_by_model)
    finally:
        loop.call_soon_threadsafe(stop.set)
        thread.join(timeout=10)
        loop.close()


def _proxy_ws_url(candidate: Gateway, path: str) -> str:
    base: Final = str(candidate.client.base_url).rstrip("/").replace("http://", "ws://", 1)
    return f"{base}{path}"


def _create(model: str, text: str, *, include_model: bool = True) -> str:
    body: Final[dict[str, JsonValue]] = {
        "type": "response.create",
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}],
        **({"model": model} if include_model else {}),
    }
    return json.dumps(body)


async def _receive(connection: websockets.ClientConnection, timeout: float) -> ReceiveResult:
    try:
        raw: Final = await asyncio.wait_for(connection.recv(), timeout=timeout)
    except asyncio.TimeoutError:
        return ReceiveResult(True, False, None, None, None)
    except ConnectionClosed as error:
        received: Final = error.rcvd
        return ReceiveResult(
            False,
            True,
            None,
            received.code if received is not None else None,
            received.reason if received is not None else None,
        )
    return ReceiveResult(False, False, JSON_OBJECT.validate_json(raw), None, None)


async def _wait_for_close(connection: websockets.ClientConnection, timeout: float) -> ReceiveResult:
    started: Final = time.monotonic()
    result: Final = await _receive(connection, timeout)
    if result.closed or result.timed_out:
        return result
    remaining: Final = timeout - (time.monotonic() - started)
    if remaining <= 0:
        return ReceiveResult(True, False, None, None, None)
    return await _wait_for_close(connection, remaining)


async def _until_terminal(
    connection: websockets.ClientConnection,
    received: tuple[dict[str, JsonValue], ...] = (),
) -> tuple[dict[str, JsonValue], ...]:
    event: Final = JSON_OBJECT.validate_json(await asyncio.wait_for(connection.recv(), timeout=20))
    collected: Final = (*received, event)
    if event.get("type") in TERMINAL or len(collected) >= 50:
        return collected
    return await _until_terminal(connection, collected)


async def _turn(connection: websockets.ClientConnection, frame: str) -> tuple[dict[str, JsonValue], ...]:
    await connection.send(frame)
    return await _until_terminal(connection)


async def _idle_turn(
    proxy: str,
    key: str,
    model: str,
    path: str,
    text: str,
    *,
    query_model: bool = False,
    idle_seconds: float = 35,
) -> SessionResult:
    query: Final = f"?model={model}" if query_model else ""
    try:
        async with websockets.connect(
            f"{proxy}{path}{query}",
            additional_headers={"Authorization": f"Bearer {key}"},
            open_timeout=10,
        ) as connection:
            idle: Final = await _receive(connection, idle_seconds)
            try:
                events: Final = await _turn(connection, _create(model, text, include_model=not query_model))
            except (ConnectionClosed, asyncio.TimeoutError) as error:
                return SessionResult(text, idle, (), f"{type(error).__name__}: {error}")
            return SessionResult(text, idle, events, None)
    except (ConnectionClosed, asyncio.TimeoutError) as error:
        return SessionResult(text, ReceiveResult(False, True, None, None, None), (), f"{type(error).__name__}: {error}")


async def _pool_workload(candidate: Gateway, key: str, model: str) -> tuple[SessionResult, ...]:
    proxy: Final = _proxy_ws_url(candidate, "")
    paths: Final = ("/v1/responses",) * 4 + ("/responses",) * 4
    return tuple(
        await asyncio.gather(*tuple(_idle_turn(proxy, key, model, path, f"pool-{uuid.uuid4().hex}") for path in paths))
    )


async def _rejected_handshake(proxy: str) -> HandshakeResult:
    try:
        async with websockets.connect(
            f"{proxy}/v1/responses",
            additional_headers={"Authorization": f"Bearer sk-rejected-{uuid.uuid4().hex}"},
            open_timeout=10,
        ):
            return HandshakeResult(None, "handshake accepted")
    except InvalidStatus as error:
        return HandshakeResult(error.response.status_code, None)
    except (ConnectionClosed, asyncio.TimeoutError, OSError) as error:
        return HandshakeResult(None, f"{type(error).__name__}: {error}")


async def _subprotocol_turn(proxy: str, key: str, model: str) -> SubprotocolResult:
    text: Final = f"subprotocol-{uuid.uuid4().hex}"
    try:
        async with websockets.connect(
            f"{proxy}/v1/responses",
            additional_headers={"Authorization": f"Bearer {key}"},
            subprotocols=SUBPROTOCOLS,
            open_timeout=10,
        ) as connection:
            events, error = await _turn_result(connection, _create(model, text))
            return SubprotocolResult(text, connection.subprotocol, events, error)
    except (ConnectionClosed, asyncio.TimeoutError, InvalidStatus) as error:
        return SubprotocolResult(text, None, (), f"{type(error).__name__}: {error}")


async def _default_workload(
    candidate: Gateway, key: str, model: str
) -> tuple[tuple[SessionResult, ...], SessionResult, HandshakeResult, SubprotocolResult]:
    proxy: Final = _proxy_ws_url(candidate, "")
    pool, query, rejected, negotiated = await asyncio.gather(
        _pool_workload(candidate, key, model),
        _idle_turn(proxy, key, model, "/v1/responses", f"query-{uuid.uuid4().hex}", query_model=True),
        _rejected_handshake(proxy),
        _subprotocol_turn(proxy, key, model),
    )
    return pool, query, rejected, negotiated


def _snapshot(record: PeerConnection) -> PeerSnapshot:
    return PeerSnapshot(record.path, _drain(record.frames), record.closed.qsize() > 0)


def _snapshots(peer: ResponsesPeer, count: int) -> tuple[PeerSnapshot, ...]:
    eventually(lambda: peer.connections.qsize(), lambda size: size >= count, seconds=15, return_last_on_timeout=True)
    records: Final = _drain(peer.connections)
    eventually(
        lambda: tuple(record.closed.qsize() for record in records),
        lambda counts: all(count > 0 for count in counts),
        seconds=10,
        return_last_on_timeout=True,
    )
    return tuple(_snapshot(record) for record in records)


def _available_snapshots(peer: ResponsesPeer) -> tuple[PeerSnapshot, ...]:
    eventually(lambda: peer.connections.qsize(), lambda count: count >= 1, seconds=10)
    records: Final = _drain(peer.connections)
    eventually(
        lambda: tuple(record.closed.qsize() for record in records),
        lambda counts: all(count > 0 for count in counts),
        seconds=10,
    )
    return tuple(_snapshot(record) for record in records)


def _frame_text(frame: dict[str, JsonValue]) -> str:
    input_value: Final = _list(frame["input"])
    message: Final = _object(input_value[0])
    content: Final = _list(message["content"])
    item: Final = _object(content[0])
    return _string(item["text"])


def _completed_text(events: tuple[dict[str, JsonValue], ...]) -> str:
    assert events, events
    completed: Final = events[-1]
    assert completed.get("type") == "response.completed", completed
    response: Final = _object(completed["response"])
    output: Final = _list(response["output"])
    message: Final = _object(output[0])
    content: Final = _list(message["content"])
    item: Final = _object(content[0])
    return _string(item["text"])


def _auth_close(error: ConnectionClosed) -> tuple[int | None, str | None]:
    received: Final = error.rcvd
    return (
        received.code if received is not None else None,
        received.reason if received is not None else None,
    )


async def _auth_attempt(proxy: str, key: str, model: str, *, delay: bool) -> AuthResult:
    try:
        async with websockets.connect(
            f"{proxy}/v1/responses",
            additional_headers={"Authorization": f"Bearer {key}"},
            open_timeout=10,
        ) as connection:
            idle: Final = await _receive(connection, 35) if delay else ReceiveResult(False, False, None, None, None)
            try:
                await connection.send(_create(model, f"auth-{uuid.uuid4().hex}"))
                frame: Final = JSON_OBJECT.validate_json(await asyncio.wait_for(connection.recv(), timeout=20))
                try:
                    await asyncio.wait_for(connection.recv(), timeout=20)
                except ConnectionClosed as error:
                    code, reason = _auth_close(error)
                    return AuthResult(idle, frame, code, reason, None)
                return AuthResult(idle, frame, None, None, "connection remained open after rejection")
            except (ConnectionClosed, asyncio.TimeoutError) as error:
                code, reason = _auth_close(error) if isinstance(error, ConnectionClosed) else (None, None)
                return AuthResult(idle, None, code, reason, f"{type(error).__name__}: {error}")
    except (ConnectionClosed, asyncio.TimeoutError) as error:
        return AuthResult(
            ReceiveResult(False, True, None, None, None),
            None,
            None,
            None,
            f"{type(error).__name__}: {error}",
        )


async def _auth_workload(candidate: Gateway, key: str, model: str, peer: ResponsesPeer) -> AuthResults:
    proxy: Final = _proxy_ws_url(candidate, "")
    immediate, delayed = await asyncio.gather(
        _auth_attempt(proxy, key, model, delay=False),
        _auth_attempt(proxy, key, model, delay=True),
    )
    return AuthResults(immediate, delayed, peer.connections.qsize())


async def _close_at_limit(
    candidate: Gateway,
    key: str,
    model: str,
    *,
    delay: float | None = None,
) -> CloseResult:
    started: Final = time.monotonic()
    outcome: Final = await _close_session(candidate, key, model, delay=delay)
    return CloseResult(outcome, time.monotonic() - started)


async def _close_session(
    candidate: Gateway,
    key: str,
    model: str,
    *,
    delay: float | None,
) -> ReceiveResult:
    started: Final = time.monotonic()
    try:
        async with websockets.connect(
            _proxy_ws_url(candidate, "/v1/responses"),
            additional_headers={"Authorization": f"Bearer {key}"},
            open_timeout=10,
        ) as connection:
            if delay is not None:
                await asyncio.sleep(max(0, started + delay - time.monotonic()))
            return await _wait_for_close(connection, 75)
    except (ConnectionClosed, asyncio.TimeoutError) as error:
        code, reason = _auth_close(error) if isinstance(error, ConnectionClosed) else (None, None)
        return ReceiveResult(False, True, None, code, reason)


async def _turn_result(
    connection: websockets.ClientConnection,
    frame: str,
) -> tuple[tuple[dict[str, JsonValue], ...], str | None]:
    try:
        return await _turn(connection, frame), None
    except (ConnectionClosed, asyncio.TimeoutError) as error:
        return (), f"{type(error).__name__}: {error}"


async def _active_session(candidate: Gateway, key: str, model: str, peer: ResponsesPeer) -> ActiveResult:
    started: Final = time.monotonic()
    try:
        async with websockets.connect(
            _proxy_ws_url(candidate, "/v1/responses"),
            additional_headers={"Authorization": f"Bearer {key}"},
            open_timeout=10,
        ) as connection:
            await asyncio.sleep(max(0, started + 5 - time.monotonic()))
            turn, turn_error = await _turn_result(connection, _create(model, f"active-{uuid.uuid4().hex}"))
            remaining: Final = max(0, 75 - (time.monotonic() - started))
            close: Final = CloseResult(await _wait_for_close(connection, remaining), time.monotonic() - started)
            provider_closed: Final = await asyncio.to_thread(_provider_closed_within, peer, PROVIDER_MODEL, 5)
            return ActiveResult(turn, turn_error, close, provider_closed)
    except (ConnectionClosed, asyncio.TimeoutError) as error:
        code, reason = _auth_close(error) if isinstance(error, ConnectionClosed) else (None, None)
        return ActiveResult(
            (),
            f"{type(error).__name__}: {error}",
            CloseResult(ReceiveResult(False, True, None, code, reason), time.monotonic() - started),
            False,
        )


def _provider_closed_within(peer: ResponsesPeer, model: str, seconds: float) -> bool:
    deadline: Final = time.monotonic() + seconds
    try:
        record: Final = peer.connections_by_model[model].get(timeout=seconds)
    except Empty:
        return False
    try:
        eventually(
            lambda: record.closed.qsize(),
            lambda count: count >= 1,
            seconds=max(0, deadline - time.monotonic()),
        )
    except AssertionError:
        return False
    return True


async def _stall_turn(connection: websockets.ClientConnection, model: str) -> tuple[bool, str | None]:
    try:
        await connection.send(_create(model, f"stall-{uuid.uuid4().hex}"))
        first: Final = JSON_OBJECT.validate_json(await asyncio.wait_for(connection.recv(), timeout=20))
        return first.get("type") == "response.created", None
    except (ConnectionClosed, asyncio.TimeoutError) as error:
        return False, f"{type(error).__name__}: {error}"


async def _mid_session(
    candidate: Gateway,
    key: str,
    stall_model: str,
) -> tuple[bool, str | None, CloseResult]:
    started: Final = time.monotonic()
    try:
        async with websockets.connect(
            _proxy_ws_url(candidate, "/v1/responses"),
            additional_headers={"Authorization": f"Bearer {key}"},
            open_timeout=10,
        ) as connection:
            await asyncio.sleep(max(0, started + 40 - time.monotonic()))
            created, turn_error = await _stall_turn(connection, stall_model)
            remaining: Final = max(0, 75 - (time.monotonic() - started))
            close: Final = CloseResult(await _wait_for_close(connection, remaining), time.monotonic() - started)
            return created, turn_error, close
    except (ConnectionClosed, asyncio.TimeoutError) as error:
        code, reason = _auth_close(error) if isinstance(error, ConnectionClosed) else (None, None)
        return (
            False,
            f"{type(error).__name__}: {error}",
            CloseResult(
                ReceiveResult(False, True, None, code, reason),
                time.monotonic() - started,
            ),
        )


async def _fresh_session(candidate: Gateway, key: str, normal_model: str) -> tuple[bool, str | None]:
    try:
        async with websockets.connect(
            _proxy_ws_url(candidate, "/v1/responses"),
            additional_headers={"Authorization": f"Bearer {key}"},
            open_timeout=10,
        ) as connection:
            events: Final = await _turn(connection, _create(normal_model, f"fresh-{uuid.uuid4().hex}"))
            return _completed_text(events) == PEER_TEXT, None
    except (ConnectionClosed, asyncio.TimeoutError) as error:
        return False, f"{type(error).__name__}: {error}"


async def _mid_response(
    candidate: Gateway,
    key: str,
    normal_model: str,
    stall_model: str,
    peer: ResponsesPeer,
) -> MidResult:
    created, turn_error, close = await _mid_session(candidate, key, stall_model)
    provider_closed: Final = await asyncio.to_thread(_provider_closed_within, peer, STALL_PROVIDER_MODEL, 5)
    fresh_completed, fresh_error = await _fresh_session(candidate, key, normal_model)
    return MidResult(created, turn_error, close, provider_closed, fresh_completed, fresh_error)


async def _deaf_response(
    candidate: Gateway,
    key: str,
    normal_model: str,
    deaf_model: str,
    peer: ResponsesPeer,
) -> MidResult:
    created, turn_error, close = await _mid_session(candidate, key, deaf_model)
    provider_closed: Final = await asyncio.to_thread(_provider_closed_within, peer, DEAF_PROVIDER_MODEL, 30)
    fresh_completed, fresh_error = await _fresh_session(candidate, key, normal_model)
    return MidResult(created, turn_error, close, provider_closed, fresh_completed, fresh_error)


async def _cap_workload(
    candidate: Gateway,
    key: str,
    normal_model: str,
    stall_model: str,
    deaf_model: str,
    peer: ResponsesPeer,
) -> tuple[CloseResult, ActiveResult, MidResult, MidResult]:
    idle, active, mid, deaf = await asyncio.gather(
        _close_at_limit(candidate, key, normal_model),
        _active_session(candidate, key, normal_model, peer),
        _mid_response(candidate, key, normal_model, stall_model, peer),
        _deaf_response(candidate, key, normal_model, deaf_model, peer),
    )
    return idle, active, mid, deaf


def _session_config(path: Path, seconds: int) -> Path:
    source: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"
    content: Final = source.read_text()
    updated: Final = content.replace(
        "general_settings:\n",
        f"general_settings:\n  responses_websocket_session_limit_seconds: {seconds}\n",
        1,
    )
    path.write_text(updated)
    return path


def _worker_pids(log: Path) -> frozenset[int]:
    return frozenset(
        eventually(
            lambda: WORKER_PIDS.validate_python(STARTED_WORKER.findall(log.read_text())),
            lambda pids: len(pids) == 2,
            seconds=30,
        )
    )


def _client_port(connection: websockets.ClientConnection) -> int:
    return CLIENT_ADDRESS.validate_python(connection.transport.get_extra_info("sockname"))[1]


def _holding_worker(workers: frozenset[int], client_port: int) -> int | None:
    def holds(pid: int) -> bool:
        return any(
            connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == client_port
            for connection in psutil.Process(pid).net_connections(kind="tcp")
        )

    return next((pid for pid in sorted(workers) if holds(pid)), None)


def _settled_on_every_worker(written_at: float) -> None:
    eventually(
        lambda: time.monotonic() - written_at,
        lambda elapsed: elapsed >= WORKER_SYNC_SECONDS,
        seconds=WORKER_SYNC_SECONDS + 5,
    )


def _config_field(candidate: Gateway, action: str, body: Mapping[str, JsonValue]) -> UpdateResult:
    response: Final = candidate.request("POST", f"/config/field/{action}", body)
    return UpdateResult(response.status_code, response.text)


def _update_limit(candidate: Gateway, value: JsonValue) -> UpdateResult:
    return _config_field(
        candidate,
        "update",
        {"field_name": SESSION_LIMIT_FIELD, "field_value": value, "config_type": "general_settings"},
    )


def _delete_limit(candidate: Gateway) -> UpdateResult:
    return _config_field(candidate, "delete", {"field_name": SESSION_LIMIT_FIELD, "config_type": "general_settings"})


def _general_settings_row(database_url: str | None) -> dict[str, JsonValue] | None:
    rows: Final = read_rows(
        'SELECT param_value FROM "LiteLLM_Config" WHERE param_name = %s',
        ("general_settings",),
        database_url=database_url,
    )
    return _object(rows[0]["param_value"]) if rows else None


def _stored_limit(database_url: str) -> JsonValue:
    row: Final = _general_settings_row(database_url)
    return None if row is None else row.get(SESSION_LIMIT_FIELD)


def _health_status(candidate: Gateway) -> int:
    try:
        return candidate.request("GET", "/health/liveliness").status_code
    except httpx.HTTPError:
        return 0


async def _update_sequence(candidate: Gateway, values: tuple[int, ...]) -> tuple[UpdateResult, ...]:
    if not values:
        return ()
    first: Final = await asyncio.to_thread(_update_limit, candidate, values[0])
    return (first, *await _update_sequence(candidate, values[1:]))


async def _open_capped_socket(proxy: str, key: str, path: str, workers: frozenset[int]) -> OpenedSocket:
    started: Final = time.monotonic()
    connection: Final = await websockets.connect(
        f"{proxy}{path}",
        additional_headers={"Authorization": f"Bearer {key}"},
        open_timeout=10,
    )
    holder: Final = await asyncio.to_thread(_holding_worker, workers, _client_port(connection))
    return OpenedSocket(path, holder, connection, started)


async def _sockets_on_every_worker(
    proxy: str, key: str, workers: frozenset[int], opened: tuple[OpenedSocket, ...] = ()
) -> tuple[OpenedSocket, ...]:
    holders: Final = frozenset(socket.worker_pid for socket in opened)
    enough: Final = len(opened) >= CAPPED_SOCKETS_MINIMUM
    if enough and (holders >= workers or len(opened) >= CAPPED_SOCKETS_MAXIMUM):
        return opened
    path: Final = CAPPED_PATHS[len(opened) % len(CAPPED_PATHS)]
    next_socket: Final = await _open_capped_socket(proxy, key, path, workers)
    return await _sockets_on_every_worker(proxy, key, workers, (*opened, next_socket))


async def _capped_close(socket: OpenedSocket) -> CappedSocket:
    try:
        outcome: Final = await _wait_for_close(socket.connection, CAP_SECONDS + 15)
        return CappedSocket(socket.path, socket.worker_pid, CloseResult(outcome, time.monotonic() - socket.started))
    finally:
        await socket.connection.close()


async def _held_turn(connection: websockets.ClientConnection, model: str, accepted_at: float) -> HeldResult:
    events, error = await _turn_result(connection, _create(model, f"earlier-{uuid.uuid4().hex}"))
    return HeldResult(time.monotonic() - accepted_at, events, error)


async def _override_workload(owned: OwnedProxy, key: str, model: str, database_url: str) -> OverrideResults:
    candidate: Final = owned.gateway
    proxy: Final = _proxy_ws_url(candidate, "")
    workers: Final = await asyncio.to_thread(_worker_pids, owned.log)
    async with websockets.connect(
        f"{proxy}/v1/responses",
        additional_headers={"Authorization": f"Bearer {key}"},
        open_timeout=10,
    ) as earlier:
        accepted_at: Final = time.monotonic()
        updates: Final = await _update_sequence(candidate, (7200, CAP_SECONDS, CAP_SECONDS))
        written_at: Final = time.monotonic()
        stored_after_updates: Final = await asyncio.to_thread(_stored_limit, database_url)
        await asyncio.to_thread(_settled_on_every_worker, written_at)
        opened: Final = await _sockets_on_every_worker(proxy, key, workers)
        capped: Final = await asyncio.gather(*tuple(_capped_close(socket) for socket in opened))
        earlier_result: Final = await _held_turn(earlier, model, accepted_at)
    delete: Final = await asyncio.to_thread(_delete_limit, candidate)
    deleted_at: Final = time.monotonic()
    stored_after_delete: Final = await asyncio.to_thread(_general_settings_row, database_url)
    await asyncio.to_thread(_settled_on_every_worker, deleted_at)
    restored: Final = await _idle_turn(
        proxy,
        key,
        model,
        "/v1/responses",
        f"restored-{uuid.uuid4().hex}",
        idle_seconds=RESTORED_IDLE_SECONDS,
    )
    return OverrideResults(
        workers,
        updates,
        stored_after_updates,
        tuple(capped),
        earlier_result,
        delete,
        stored_after_delete,
        restored,
    )


async def _kill_workload(owned: OwnedProxy, key: str, model: str) -> KillResults:
    candidate: Final = owned.gateway
    workers: Final = await asyncio.to_thread(_worker_pids, owned.log)
    connections: Final = await asyncio.gather(
        *tuple(
            websockets.connect(
                _proxy_ws_url(candidate, "/v1/responses"),
                additional_headers={"Authorization": f"Bearer {key}"},
                open_timeout=10,
            )
            for _ in range(KILLED_POOL_SIZE)
        )
    )
    holders: Final = await asyncio.gather(
        *tuple(asyncio.to_thread(_holding_worker, workers, _client_port(connection)) for connection in connections)
    )
    held_by: Final = MappingProxyType({pid: holders.count(pid) for pid in sorted(workers)})
    victim: Final = max(sorted(workers), key=held_by.__getitem__)
    psutil.Process(victim).kill()
    victims: Final = tuple(
        connection for connection, holder in zip(connections, holders, strict=True) if holder == victim
    )
    survivors: Final = tuple(
        connection for connection, holder in zip(connections, holders, strict=True) if holder != victim
    )
    victim_closes, survivor_turns = await asyncio.gather(
        asyncio.gather(*tuple(_wait_for_close(connection, 30) for connection in victims)),
        asyncio.gather(
            *tuple(_turn_result(connection, _create(model, f"survivor-{uuid.uuid4().hex}")) for connection in survivors)
        ),
    )
    health: Final = await asyncio.to_thread(
        eventually,
        lambda: _health_status(candidate),
        lambda status_code: status_code == 200,
        30,
    )
    fresh: Final = await _chaos_turn(candidate, key, model)
    await asyncio.gather(*tuple(connection.close() for connection in survivors))
    return KillResults(
        held_by,
        victim,
        tuple(victim_closes),
        tuple(TurnOutcome(events, error) for events, error in survivor_turns),
        health,
        fresh,
    )


def _responses_body(model: str) -> JsonResponse:
    message: Final[dict[str, JsonValue]] = {
        "type": "message",
        "id": "msg_$UNIQUE_ID",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "responses-$UNIQUE_ID", "annotations": []}],
    }
    return JsonResponse(
        content_type="application/json",
        body={
            "id": "$UNIQUE_ID",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": model,
            "output": [message],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        },
    )


def _chat_body(model: str) -> JsonResponse:
    return JsonResponse(
        content_type="application/json",
        body={
            "id": "$UNIQUE_ID",
            "object": "chat.completion",
            "created": 1700000000,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "chat-$UNIQUE_ID"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


def _responses_spec(model: str) -> tuple[str, dict[str, JsonValue]]:
    return "/v1/responses", {"model": model, "input": f"responses-{uuid.uuid4().hex}"}


def _chat_spec(model: str) -> tuple[str, dict[str, JsonValue]]:
    return "/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": f"chat-{uuid.uuid4().hex}"}],
    }


async def _burst_request(
    client: httpx.AsyncClient,
    key: str,
    path: str,
    body: dict[str, JsonValue],
) -> BurstResult:
    try:
        response: Final = await client.post(path, json=body, headers={"Authorization": f"Bearer {key}"})
        try:
            parsed: Final = JSON_OBJECT.validate_json(response.content)
        except ValidationError as error:
            return BurstResult(path, response.status_code, None, str(error))
        return BurstResult(path, response.status_code, parsed, None)
    except httpx.HTTPError as error:
        return BurstResult(path, None, None, str(error))


async def _chaos_workload(
    owned: OwnedProxy,
    key: str,
    scripted_model: str,
    peer_model: str,
) -> ChaosResults:
    candidate: Final = owned.gateway
    base: Final = str(candidate.client.base_url)
    specs: Final = tuple(
        _responses_spec(scripted_model) if index % 2 == 0 else _chat_spec(scripted_model) for index in range(20)
    )
    connections: Final = await asyncio.gather(
        *tuple(
            websockets.connect(
                _proxy_ws_url(candidate, "/v1/responses"),
                additional_headers={"Authorization": f"Bearer {key}"},
                open_timeout=10,
            )
            for _ in range(30)
        )
    )
    async with httpx.AsyncClient(base_url=base, timeout=30, trust_env=False) as client:
        results: Final = await asyncio.gather(*tuple(_burst_request(client, key, path, body) for path, body in specs))
    tuple(connection.transport.abort() for connection in connections)
    health: Final = candidate.request("GET", "/health/liveliness").status_code
    websocket_completed: Final = await _chaos_turn(candidate, key, peer_model)
    kill: Final = await _kill_workload(owned, key, peer_model)
    return ChaosResults(tuple(results), health, websocket_completed, kill)


async def _chaos_turn(candidate: Gateway, key: str, model: str) -> bool:
    async with websockets.connect(
        _proxy_ws_url(candidate, "/v1/responses"),
        additional_headers={"Authorization": f"Bearer {key}"},
        open_timeout=10,
    ) as connection:
        events: Final = await _turn(connection, _create(model, f"chaos-peer-{uuid.uuid4().hex}"))
        return _completed_text(events) == PEER_TEXT


@pytest.fixture(scope="module")
def cert(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return write_self_signed_cert(tmp_path_factory.mktemp("responses-ws-session-limit"))


@pytest.fixture(scope="module")
def default_results(
    cert: tuple[Path, Path],
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[DefaultResults]:
    with responses_peer(cert) as peer, gateway_from_environment() as base:
        with (
            owned_proxy(
                base,
                tmp_path_factory.mktemp("responses-ws-default"),
                {"SSL_CERT_FILE": str(cert[0])},
                workers=2,
            ) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(model=f"openai/{PROVIDER_MODEL}", api_base=peer.url)
            key: Final = scenario.key(models=[model])
            pool, query, rejected, negotiated = asyncio.run(_default_workload(candidate, key, model))
            provider: Final = _snapshots(peer, 10)
            yield DefaultResults(pool, query, rejected, negotiated, provider)


@pytest.fixture(scope="module")
def cap_results(
    cert: tuple[Path, Path],
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[CapResults]:
    with responses_peer(cert) as peer, gateway_from_environment() as base:
        config: Final = _session_config(tmp_path_factory.mktemp("responses-ws-cap") / "config.yaml", 60)
        with (
            owned_proxy(
                base,
                tmp_path_factory.mktemp("responses-ws-cap-proxy"),
                {"SSL_CERT_FILE": str(cert[0])},
                config=config,
                workers=2,
            ) as candidate,
            candidate.scenario() as scenario,
        ):
            normal: Final = scenario.model(model=f"openai/{PROVIDER_MODEL}", api_base=peer.url)
            stall: Final = scenario.model(model=f"openai/{STALL_PROVIDER_MODEL}", api_base=peer.url)
            deaf: Final = scenario.model(model=f"openai/{DEAF_PROVIDER_MODEL}", api_base=peer.url)
            key: Final = scenario.key(models=[normal, stall, deaf])
            idle, active, mid, deaf_result = asyncio.run(_cap_workload(candidate, key, normal, stall, deaf, peer))
            provider: Final = _available_snapshots(peer)
            yield CapResults(idle, active, mid, deaf_result, provider)


@pytest.fixture(scope="module")
def invalid_results(
    cert: tuple[Path, Path],
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[InvalidResult]:
    with responses_peer(cert) as peer, gateway_from_environment() as base:
        config: Final = _session_config(tmp_path_factory.mktemp("responses-ws-invalid") / "config.yaml", 30)
        with owned_proxy_process(
            base,
            tmp_path_factory.mktemp("responses-ws-invalid-proxy"),
            {"SSL_CERT_FILE": str(cert[0])},
            config=config,
            workers=2,
        ) as owned:
            with owned.gateway.scenario() as scenario:
                model: Final = scenario.model(model=f"openai/{PROVIDER_MODEL}", api_base=peer.url)
                key: Final = scenario.key(models=[model])
                session: Final = asyncio.run(
                    _idle_turn(
                        _proxy_ws_url(owned.gateway, ""),
                        key,
                        model,
                        "/v1/responses",
                        f"invalid-{uuid.uuid4().hex}",
                    )
                )
                warning: Final = "invalid general_settings.responses_websocket_session_limit_seconds=30"
                yield InvalidResult(session, warning in owned.log.read_text())


@pytest.fixture(scope="module")
def chaos_results(
    cert: tuple[Path, Path],
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[ChaosResults]:
    with responses_peer(cert) as peer, gateway_from_environment() as base:
        with (
            owned_proxy_process(
                base,
                tmp_path_factory.mktemp("responses-ws-chaos"),
                {"SSL_CERT_FILE": str(cert[0])},
                workers=2,
            ) as owned,
            owned.gateway.scenario() as scenario,
        ):
            scenario_id: Final = f"responses-chaos-{uuid.uuid4().hex}"
            scripted: Final = register_scenario(
                scenario_id,
                RoutedResponse(
                    content_type="application/x-routed",
                    routes={
                        "POST /responses": _responses_body("scripted-responses-model"),
                        "POST /chat/completions": _chat_body("scripted-chat-model"),
                    },
                ),
                control_url=owned.gateway.upstream_url,
            )
            scenario.cleanups.callback(delete_scenario, scripted)
            scripted_model: Final = scenario.model(
                model="openai/scripted-responses-model",
                api_base=scripted.api_base(),
            )
            peer_model: Final = scenario.model(model=f"openai/{PROVIDER_MODEL}", api_base=peer.url)
            key: Final = scenario.key(models=[scripted_model, peer_model])
            yield asyncio.run(_chaos_workload(owned, key, scripted_model, peer_model))


@pytest.fixture(scope="module")
def override_results(
    cert: tuple[Path, Path],
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[OverrideResults]:
    with responses_peer(cert) as peer, gateway_from_environment() as base, scratch_database() as database_url:
        with owned_proxy_process(
            base,
            tmp_path_factory.mktemp("responses-ws-override"),
            {
                "SSL_CERT_FILE": str(cert[0]),
                "DATABASE_URL": database_url,
                "PROXY_CONFIG_RELOAD_INTERVAL_SECONDS": str(RELOAD_INTERVAL_SECONDS),
            },
            remove_environment=("DATABASE_URL_READ_REPLICA",),
            workers=2,
        ) as owned:
            with owned.gateway.scenario() as scenario:
                model: Final = scenario.model(model=f"openai/{PROVIDER_MODEL}", api_base=peer.url)
                key: Final = scenario.key(models=[model])
                yield asyncio.run(_override_workload(owned, key, model, database_url))


def test_default_config_idle_pool_stays_open_and_completes(default_results: DefaultResults) -> None:
    assert all(result.idle.timed_out for result in default_results.pool), default_results.pool
    assert all(result.error is None for result in default_results.pool), default_results.pool
    assert all(_completed_text(result.events) == PEER_TEXT for result in default_results.pool)
    texts: Final = frozenset(result.text for result in default_results.pool)
    provider: Final = tuple(snapshot for snapshot in default_results.provider if snapshot.frames)
    assert len(provider) == 10, default_results.provider
    assert all(snapshot.path == f"/v1/responses?model={PROVIDER_MODEL}" for snapshot in provider)
    frames: Final = tuple(snapshot.frames[0] for snapshot in provider)
    expected_texts: Final = texts | {default_results.query.text, default_results.subprotocol.text}
    assert frozenset(_frame_text(frame) for frame in frames) == expected_texts
    assert all(frame.get("model") == PROVIDER_MODEL for frame in frames)


def test_default_config_query_model_socket_stays_open_and_completes(default_results: DefaultResults) -> None:
    assert default_results.query.idle.timed_out, default_results.query
    assert default_results.query.error is None, default_results.query
    assert _completed_text(default_results.query.events) == PEER_TEXT
    assert any(
        default_results.query.text == _frame_text(snapshot.frames[0])
        for snapshot in default_results.provider
        if snapshot.frames
    )


def test_delayed_first_frame_model_auth_matches_immediate_rejection(
    cert: tuple[Path, Path],
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    with responses_peer(cert) as peer, gateway_from_environment() as base:
        with (
            owned_proxy(
                base,
                tmp_path_factory.mktemp("responses-ws-auth"),
                {"SSL_CERT_FILE": str(cert[0])},
                workers=2,
            ) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(model=f"openai/{PROVIDER_MODEL}", api_base=peer.url)
            key: Final = scenario.key(models=["not-authorized-model"])
            results: Final = asyncio.run(_auth_workload(candidate, key, model, peer))
    assert results.immediate.error is None, results.immediate
    assert results.delayed.idle.timed_out, results.delayed
    assert results.delayed.error is None, results.delayed
    assert results.immediate.frame is not None, results.immediate
    assert results.immediate.frame.get("type") == "error", results.immediate
    rejection: Final = _object(results.immediate.frame["error"])
    assert rejection.get("type") == "invalid_request_error", results.immediate
    assert results.immediate.close_code == 1008, results.immediate
    assert results.immediate.close_reason == "Pre-call error", results.immediate
    assert results.immediate.frame == results.delayed.frame
    assert results.immediate.close_code == results.delayed.close_code
    assert results.immediate.close_reason == results.delayed.close_reason
    assert results.provider_connections == 0, results


def test_session_cap_closes_never_started_socket(cap_results: CapResults) -> None:
    assert cap_results.idle.outcome.closed, cap_results.idle
    assert cap_results.idle.outcome.close_code == 1000, cap_results.idle
    assert cap_results.idle.outcome.close_reason == "Session duration limit reached", cap_results.idle
    assert 59 <= cap_results.idle.elapsed <= 75, cap_results.idle


def test_session_cap_closes_active_socket_and_provider(cap_results: CapResults) -> None:
    assert cap_results.active.turn_error is None, cap_results.active
    assert _completed_text(cap_results.active.turn) == PEER_TEXT
    assert cap_results.active.close.outcome.closed, cap_results.active
    assert cap_results.active.close.outcome.close_code == 1000, cap_results.active
    assert cap_results.active.close.outcome.close_reason == "Session duration limit reached", cap_results.active
    assert 59 <= cap_results.active.close.elapsed <= 75, cap_results.active
    assert cap_results.active.provider_closed, cap_results.active
    assert all(snapshot.closed for snapshot in cap_results.provider), cap_results.provider


def test_session_cap_closes_mid_response_and_allows_new_session(cap_results: CapResults) -> None:
    assert cap_results.mid.created, cap_results.mid
    assert cap_results.mid.turn_error is None, cap_results.mid
    assert cap_results.mid.close.outcome.closed, cap_results.mid
    assert cap_results.mid.close.outcome.close_code == 1000, cap_results.mid
    assert cap_results.mid.close.outcome.close_reason == "Session duration limit reached", cap_results.mid
    assert 59 <= cap_results.mid.close.elapsed <= 75, cap_results.mid
    assert cap_results.mid.provider_closed, cap_results.mid
    assert cap_results.mid.fresh_error is None, cap_results.mid
    assert cap_results.mid.fresh_completed, cap_results.mid


def test_session_cap_closes_client_promptly_when_provider_ignores_close(cap_results: CapResults) -> None:
    assert cap_results.deaf.created, cap_results.deaf
    assert cap_results.deaf.turn_error is None, cap_results.deaf
    assert cap_results.deaf.close.outcome.closed, cap_results.deaf
    assert cap_results.deaf.close.outcome.close_code == 1000, cap_results.deaf
    assert cap_results.deaf.close.outcome.close_reason == "Session duration limit reached", cap_results.deaf
    assert 59 <= cap_results.deaf.close.elapsed <= 63, cap_results.deaf
    assert cap_results.deaf.provider_closed, cap_results.deaf
    assert cap_results.deaf.fresh_error is None, cap_results.deaf
    assert cap_results.deaf.fresh_completed, cap_results.deaf


def test_invalid_session_cap_falls_back_to_default(invalid_results: InvalidResult) -> None:
    assert invalid_results.session.idle.timed_out, invalid_results.session
    assert invalid_results.session.error is None, invalid_results.session
    assert _completed_text(invalid_results.session.events) == PEER_TEXT
    assert invalid_results.warning_found


def _matches_scripted_body(result: BurstResult) -> bool:
    if result.body is None:
        return False
    if result.path == "/v1/responses":
        output: Final = _list(result.body["output"])
        responses_message: Final = _object(output[0])
        message_id: Final = _string(responses_message["id"])
        if not message_id.startswith("msg_"):
            return False
        content: Final = _list(responses_message["content"])
        return _string(_object(content[0])["text"]) == f"responses-{message_id.removeprefix('msg_')}"
    identity: Final = _string(result.body["id"])
    choices: Final = _list(result.body["choices"])
    chat_message: Final = _object(_object(choices[0])["message"])
    return _string(chat_message["content"]) == f"chat-{identity}"


def test_aborted_idle_pool_preserves_http_health_and_new_websocket(chaos_results: ChaosResults) -> None:
    results: Final = chaos_results.burst
    assert len(results) == 20
    assert all(result.status_code == 200 and result.error is None for result in results), results
    identities: Final = tuple(_string(_object(result.body)["id"]) for result in results if result.body is not None)
    assert len(set(identities)) == 20, identities
    assert all(_matches_scripted_body(result) for result in results), results
    assert chaos_results.health_status == 200
    assert chaos_results.websocket_completed


def test_rejected_key_fails_the_handshake_with_403(default_results: DefaultResults) -> None:
    assert default_results.rejected == HandshakeResult(403, None), default_results.rejected


def test_requested_subprotocol_is_accepted_and_the_turn_completes(default_results: DefaultResults) -> None:
    negotiated: Final = default_results.subprotocol
    assert negotiated.error is None, negotiated
    assert negotiated.negotiated == SUBPROTOCOLS[0], negotiated
    assert _completed_text(negotiated.events) == PEER_TEXT


def test_db_override_caps_new_sessions_on_every_worker_without_restart(override_results: OverrideResults) -> None:
    assert len(override_results.workers) == 2, override_results.workers
    capped: Final = override_results.capped
    assert CAPPED_SOCKETS_MINIMUM <= len(capped) <= CAPPED_SOCKETS_MAXIMUM, capped
    assert tuple(socket.path for socket in capped) == tuple(
        CAPPED_PATHS[i % len(CAPPED_PATHS)] for i in range(len(capped))
    )
    assert all(socket.close.outcome.closed and socket.close.outcome.close_code == 1000 for socket in capped), capped
    assert all(socket.close.outcome.close_reason == LIMIT_CLOSE_REASON for socket in capped), capped
    assert all(CAP_SECONDS - 1 <= socket.close.elapsed <= CAP_SECONDS + 15 for socket in capped), capped
    assert frozenset(socket.worker_pid for socket in capped) == override_results.workers, capped


def test_db_override_leaves_sessions_accepted_before_it_alone(override_results: OverrideResults) -> None:
    earlier: Final = override_results.earlier
    assert earlier.error is None, earlier
    assert earlier.held_seconds > CAP_SECONDS, earlier
    assert _completed_text(earlier.events) == PEER_TEXT


def test_db_override_accepts_the_range_bounds_and_repeats(override_results: OverrideResults) -> None:
    updates: Final = override_results.updates
    assert tuple(update.status_code for update in updates) == (200, 200, 200), updates
    assert override_results.stored_after_updates == CAP_SECONDS


def test_deleting_the_db_override_restores_the_default(override_results: OverrideResults) -> None:
    assert override_results.delete.status_code == 200, override_results.delete
    assert override_results.stored_after_delete is not None
    assert SESSION_LIMIT_FIELD not in override_results.stored_after_delete
    restored: Final = override_results.restored
    assert restored.idle.timed_out, restored
    assert restored.error is None, restored
    assert _completed_text(restored.events) == PEER_TEXT


@pytest.mark.parametrize(
    ("value", "kind"),
    (
        pytest.param(30, "int", id="below-minimum"),
        pytest.param(7201, "int", id="above-maximum"),
        pytest.param("abc", "str", id="text"),
        pytest.param("", "str", id="empty-string"),
        pytest.param("x" * 5000, "str", id="five-kilobyte-string"),
        pytest.param([], "list", id="list"),
        pytest.param({}, "dict", id="object"),
        pytest.param(None, "NoneType", id="null"),
    ),
)
def test_out_of_range_or_wrong_type_update_is_refused(gateway: Gateway, value: JsonValue, kind: str) -> None:
    before: Final = _general_settings_row(None)
    refused: Final = _update_limit(gateway, value)
    assert refused.status_code == 400, refused
    assert json.loads(refused.body) == {"detail": {"error": f"Invalid type of field value=<class '{kind}'> passed in."}}
    assert _general_settings_row(None) == before


def test_killing_one_worker_drops_only_its_sockets_and_the_proxy_keeps_serving(chaos_results: ChaosResults) -> None:
    kill: Final = chaos_results.kill
    assert sum(kill.held_by.values()) == KILLED_POOL_SIZE, kill.held_by
    assert len(kill.victim_closes) == kill.held_by[kill.victim] >= 1, kill
    assert all(close.closed and close.close_code is None for close in kill.victim_closes), kill.victim_closes
    assert all(turn.error is None for turn in kill.survivor_turns), kill.survivor_turns
    assert all(_completed_text(turn.events) == PEER_TEXT for turn in kill.survivor_turns), kill.survivor_turns
    assert kill.health_status == 200
    assert kill.fresh_completed
