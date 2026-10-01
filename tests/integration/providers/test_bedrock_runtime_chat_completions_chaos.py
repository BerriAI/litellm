import asyncio
import base64
import binascii
import multiprocessing
import os
import re
import signal
import socket
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from multiprocessing.sharedctypes import Synchronized
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, Literal
from urllib.parse import urlsplit, urlunsplit

import httpx
import psutil
import pytest
import yaml
from integration._support.bedrock_runtime_peer import MARKER, marker_of, respond, serve_peer
from integration._support.client import Gateway, Scenario, eventually, object_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

BEDROCK_MODEL: Final = "us.openai.gpt-5.6-sol"
TOKEN: Final = "synthetic-bedrock-bearer"
_CONFIG_MODEL: Final = "bedrock-gpt-chat-completions-chaos"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_ENDPOINTS: Final[tuple["Endpoint", ...]] = ("chat", "messages", "responses")

Endpoint = Literal["chat", "messages", "responses"]


@dataclass(frozen=True, slots=True)
class _Call:
    endpoint: Endpoint
    stream: bool
    marker: str


@dataclass(frozen=True, slots=True)
class _Served:
    call: _Call
    status: int
    text: str
    call_id: str | None


@dataclass(frozen=True, slots=True)
class _ChildPeer:
    process: BaseProcess
    received: Synchronized[int]
    url: str


def _path(endpoint: Endpoint) -> str:
    match endpoint:
        case "chat":
            return "/v1/chat/completions"
        case "messages":
            return "/v1/messages"
        case "responses":
            return "/v1/responses"


def _terminal(endpoint: Endpoint) -> str:
    match endpoint:
        case "chat":
            return "data: [DONE]"
        case "messages":
            return "event: message_stop"
        case "responses":
            return '"type":"response.completed"'


def _body(model: str, call: _Call) -> dict[str, JsonValue]:
    question: Final = f"Question marker-{call.marker}"
    common: Final[dict[str, JsonValue]] = {"model": model, "stream": call.stream, "cache": {"no-cache": True}}
    match call.endpoint:
        case "chat":
            return {**common, "messages": [{"role": "user", "content": question}]}
        case "messages":
            return {**common, "max_tokens": 64, "messages": [{"role": "user", "content": question}]}
        case "responses":
            return {**common, "input": question}


def _deployment(scenario: Scenario, endpoint: str) -> str:
    return scenario.model(
        model=f"bedrock/{BEDROCK_MODEL}",
        api_key=TOKEN,
        api_base=None,
        aws_region_name="us-east-1",
        aws_bedrock_runtime_endpoint=endpoint,
    )


def _frames(text: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        _JSON_OBJECT.validate_json(line[6:])
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )


def _frame_id(frame: Mapping[str, JsonValue]) -> str | None:
    if frame.get("type") == "message_start":
        return str(object_value(frame["message"])["id"])
    response: Final = frame.get("response")
    if isinstance(response, dict) and "id" in response:
        return str(response["id"])
    identity: Final = frame.get("id")
    return identity if isinstance(identity, str) else None


def _response_id(served: _Served) -> str:
    if not served.call.stream:
        return str(_JSON_OBJECT.validate_json(served.text)["id"])
    ids: Final = tuple(identity for identity in map(_frame_id, _frames(served.text)) if identity is not None)
    assert ids, served.text
    return ids[0]


def _assert_answered_with_its_own_marker(served: _Served) -> None:
    assert served.status == 200, served.text
    assert set(MARKER.findall(served.text)) == {served.call.marker}, served.text
    if served.call.stream:
        assert _terminal(served.call.endpoint) in served.text, served.text


def _spend_rows(model: str, expected: int) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows('SELECT request_id, status FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
        lambda found: len(found) >= expected,
        seconds=60,
    )


def _rows_by_status(rows: list[dict[str, JsonValue]], status: str) -> list[str]:
    return sorted(str(row["request_id"]) for row in rows if row["status"] == status)


def _upstream_id_inside(row_id: str) -> str | None:
    try:
        payload: Final = base64.b64decode(row_id.removeprefix("resp_"), validate=True).decode()
    except (binascii.Error, UnicodeDecodeError):
        return None
    return payload.rsplit("response_id:", 1)[1] if "response_id:" in payload else None


# TODO: a Bedrock non-stream /v1/responses spend row can carry the pre-encryption resp_<base64> id instead of the
# ciphertext the caller received, because the spend row id is read from response_obj["id"] before the
# ResponsesIDSecurity hook rewrites it in place; such a row is matched by the upstream id inside that payload until
# that ordering is fixed on main
def _row_belongs_to(row_id: str, served: _Served) -> bool:
    if row_id == _response_id(served):
        return True
    return served.call.endpoint == "responses" and _upstream_id_inside(row_id) == f"resp_upstream_{served.call.marker}"


def _assert_each_success_landed_once(rows: list[dict[str, JsonValue]], served: tuple[_Served, ...]) -> None:
    success_ids: Final = _rows_by_status(rows, "success")
    assert len(success_ids) == len(served), rows
    for item in served:
        owned: Final = [row_id for row_id in success_ids if _row_belongs_to(row_id, item)]
        assert len(owned) == 1, (item.call, owned, success_ids)


async def _send(client: httpx.AsyncClient, key: str, model: str, call: _Call) -> _Served:
    async with client.stream(
        "POST",
        _path(call.endpoint),
        json=_body(model, call),
        headers={"Authorization": f"Bearer {key}", "anthropic-version": "2023-06-01"},
    ) as response:
        raw: Final = await response.aread()
    return _Served(
        call=call, status=response.status_code, text=raw.decode(), call_id=response.headers.get("x-litellm-call-id")
    )


async def _burst(
    base_url: str, key: str, model: str, calls: tuple[_Call, ...], *, tolerate_transport_errors: bool = False
) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(base_url=base_url, timeout=60, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(_send(client, key, model, call) for call in calls), return_exceptions=tolerate_transport_errors
        )
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, httpx.TransportError), repr(result)
    return tuple(result for result in results if isinstance(result, _Served))


def _calls(count: int, endpoints: tuple[Endpoint, ...], stream: Callable[[int], bool]) -> tuple[_Call, ...]:
    return tuple(
        _Call(endpoint=endpoints[index % len(endpoints)], stream=stream(index), marker=uuid.uuid4().hex)
        for index in range(count)
    )


def _free_port() -> int:
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        return reserve.getsockname()[1]


def _accepts_connections(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


@contextmanager
def _child_peer(port: int, answer_first: int) -> Iterator[_ChildPeer]:
    received: Final = multiprocessing.Value("i", 0)
    process: Final = multiprocessing.get_context("spawn").Process(
        target=serve_peer, args=(port, received, answer_first), daemon=True
    )
    process.start()
    try:
        eventually(lambda: _accepts_connections(port), bool, seconds=30)
        yield _ChildPeer(process=process, received=received, url=f"http://127.0.0.1:{port}")
    finally:
        process.kill()
        process.join(timeout=10)
        assert not process.is_alive(), "Owned peer survived cleanup"


async def test_burst_across_every_endpoint_lands_each_response_id_once(gateway: Gateway) -> None:
    calls: Final = _calls(36, _ENDPOINTS, lambda index: index % 2 == 0)
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url)
        served: Final = await _burst(str(gateway.client.base_url), gateway.key, model, calls)
        assert len(served) == 36
        for item in served:
            _assert_answered_with_its_own_marker(item)
        ids: Final = sorted(_response_id(item) for item in served)
        assert len(set(ids)) == 36, ids
        assert sorted(marker_of(request) for request in wire.drain()) == sorted(call.marker for call in calls)
        rows: Final = _spend_rows(model, 36)
        _assert_each_success_landed_once(rows, served)
        assert len(rows) == 36, rows


@pytest.mark.timeout(180)
async def test_peer_killed_mid_burst_fails_only_the_held_calls_and_a_restarted_peer_serves_again(
    gateway: Gateway,
) -> None:
    calls: Final = _calls(12, _ENDPOINTS, lambda index: index % 2 == 0)
    recovery: Final = _calls(6, _ENDPOINTS, lambda index: index % 2 == 1)
    port: Final = _free_port()
    with gateway.scenario() as scenario:
        model: Final = _deployment(scenario, f"http://127.0.0.1:{port}")
        with _child_peer(port, answer_first=6) as peer:
            burst: Final = asyncio.create_task(_burst(str(gateway.client.base_url), gateway.key, model, calls))
            await asyncio.to_thread(eventually, lambda: peer.received.value, lambda count: count == 12, 60)
            peer.process.kill()
            peer.process.join(timeout=10)
            served: Final = await burst
        succeeded: Final = tuple(item for item in served if item.status == 200)
        failed: Final = tuple(item for item in served if item.status != 200)
        assert (len(succeeded), len(failed)) == (6, 6), [(item.call.marker, item.status) for item in served]
        for item in succeeded:
            _assert_answered_with_its_own_marker(item)
        assert {item.status for item in failed} == {503}, [
            (item.call.endpoint, item.call.stream, item.status, item.text) for item in failed
        ]
        for item in failed:
            assert "ServiceUnavailableError: BedrockException - Server disconnected" in item.text, item.text
            assert "marker-" not in item.text and item.call_id is not None, item.text
        with _child_peer(port, answer_first=10**6) as revived:
            recovered: Final = await _burst(str(gateway.client.base_url), gateway.key, model, recovery)
            assert revived.received.value == 6, revived.received.value
        for item in recovered:
            _assert_answered_with_its_own_marker(item)
        rows: Final = _spend_rows(model, 18)
        _assert_each_success_landed_once(rows, (*succeeded, *recovered))
        assert _rows_by_status(rows, "failure") == sorted(str(item.call_id) for item in failed), rows
        assert len(rows) == 18, rows


async def test_slow_peer_streams_are_forwarded_once_and_terminated(gateway: Gateway) -> None:
    calls: Final = _calls(10, ("chat",), lambda _: True)
    with wire_server(lambda request: respond(request, pause=0.3)) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire.url)
        served: Final = await _burst(str(gateway.client.base_url), gateway.key, model, calls)
        assert len(served) == 10
        for item in served:
            _assert_answered_with_its_own_marker(item)
        assert sorted(marker_of(request) for request in wire.drain()) == sorted(call.marker for call in calls)
        ids: Final = sorted(_response_id(item) for item in served)
        rows: Final = _spend_rows(model, 10)
        assert _rows_by_status(rows, "success") == ids, rows
        assert len(rows) == 10, rows


def _chaos_config(wire: Wire, tmp_path: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["model_list"] = [
        {
            "model_name": _CONFIG_MODEL,
            "litellm_params": {
                "model": f"bedrock/{BEDROCK_MODEL}",
                "api_key": TOKEN,
                "aws_region_name": "us-east-1",
                "aws_bedrock_runtime_endpoint": wire.url,
            },
        }
    ]
    path: Final = tmp_path / "bedrock-gpt-chat-completions-chaos.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _pooled_database_url() -> str:
    parts: Final = urlsplit(os.environ["DATABASE_URL"])
    query: Final = "&".join(part for part in (parts.query, "connection_limit=5") if part)
    return urlunsplit(parts._replace(query=query))


def _open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


def _landed_once(ids: tuple[str, ...]) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows(
            'SELECT request_id, status FROM "LiteLLM_SpendLogs" WHERE request_id = ANY(%s)',
            (list(ids),),  # pyright: ignore[reportArgumentType]  # psycopg adapts the list to a text array
        ),
        lambda found: len(found) >= len(ids),
        seconds=60,
    )


@pytest.mark.timeout(180)
async def test_worker_sigkill_mid_burst_leaves_the_sibling_serving(gateway: Gateway, tmp_path: Path) -> None:
    calls: Final = _calls(20, ("chat",), lambda _: False)
    release: Final = threading.Event()
    held_markers: Final[SimpleQueue[str]] = SimpleQueue()

    def held(request: Request) -> Reply:
        held_markers.put(marker_of(request))
        assert release.wait(timeout=60), "The burst was never released"
        return respond(request)

    with wire_server(held) as wire:
        path: Final = _chaos_config(wire, tmp_path)
        overrides: Final = {"DATABASE_URL": _pooled_database_url()}
        with owned_proxy_process(gateway, tmp_path, overrides, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            workers: Final = eventually(
                lambda: tuple(int(pid) for pid in _STARTED_WORKER.findall(owned.log.read_text())),
                lambda pids: len(pids) == 2,
                seconds=30,
            )
            burst: Final = asyncio.create_task(
                _burst(
                    str(candidate.client.base_url), candidate.key, _CONFIG_MODEL, calls, tolerate_transport_errors=True
                )
            )
            await asyncio.to_thread(eventually, held_markers.qsize, lambda size: size == 20, 60)
            held_by: Final = MappingProxyType({pid: _open_upstream_connections(pid, wire.url) for pid in workers})
            assert sum(held_by.values()) == 20, held_by
            victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
            victim: Final = psutil.Process(victim_pid)
            victim.suspend()
            victim.send_signal(signal.SIGKILL)
            release.set()
            served: Final = await burst
            assert held_by[survivor_pid] >= 10, held_by
            assert len(served) == held_by[survivor_pid], (held_by, len(served))
            for item in served:
                _assert_answered_with_its_own_marker(item)
            follow_up: Final = _Call(endpoint="chat", stream=False, marker=uuid.uuid4().hex)
            (answered,) = await _burst(str(candidate.client.base_url), candidate.key, _CONFIG_MODEL, (follow_up,))
            _assert_answered_with_its_own_marker(answered)
            received: Final = wire.drain()
            assert {request.method for request in received} == {"POST"}, received
            assert sorted(marker_of(request) for request in received) == sorted(
                call.marker for call in (*calls, follow_up)
            )
            ids: Final = tuple(sorted(_response_id(item) for item in (*served, answered)))
            rows: Final = _landed_once(ids)
            assert _rows_by_status(rows, "success") == list(ids), rows
            assert len(rows) == len(ids), rows
