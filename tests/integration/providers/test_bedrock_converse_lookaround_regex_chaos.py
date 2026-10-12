import asyncio
import json
import re
import signal
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, Literal
from urllib.parse import unquote, urlsplit

import httpx
import psutil
import pytest
import yaml
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.upstream import _aws_event_frame
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_KIMI: Final = "global.moonshotai.kimi-k3"
_NOVA: Final = "us.amazon.nova-lite-v1:0"
_AWS: Final[dict[str, JsonValue]] = {
    "aws_access_key_id": "AKIASCRIPTEDPROVIDER",
    "aws_secret_access_key": "scripted-secret",
    "aws_region_name": "us-east-1",
}
_LOOKAHEAD: Final = r"^(?!\.\.?(?:\/|$))[A-Za-z0-9_\-.~:@+]{1,200}$"
_PLAIN: Final = r"^[a-z][a-z0-9_]*$"
_TOOL: Final = "ArtifactData"
_EVENT_STREAM: Final = "application/vnd.amazon.eventstream"
_JSON: Final = TypeAdapter(dict[str, JsonValue])
_LIST: Final = TypeAdapter(list[JsonValue])
_MARKER: Final = re.compile(r"marker-([0-9a-f]{32})")
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_USAGE: Final[dict[str, JsonValue]] = {"inputTokens": 21, "outputTokens": 7, "totalTokens": 28}
_WIRE_AS_SENT: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {
        "collection": {"type": "string", "pattern": _LOOKAHEAD},
        "doc_id": {"type": "string", "pattern": _PLAIN},
    },
    "required": ["collection"],
}
_WIRE_LOOKAROUND_FREE: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"collection": {"type": "string"}, "doc_id": {"type": "string", "pattern": _PLAIN}},
    "required": ["collection"],
}
_SCHEMA_AS_SENT: Final[dict[str, JsonValue]] = {**_WIRE_AS_SENT, "additionalProperties": False}

Endpoint = Literal["chat", "messages", "responses"]
_ENDPOINTS: Final[tuple[Endpoint, ...]] = ("chat", "messages", "responses")


@dataclass(frozen=True, slots=True)
class _Fleet:
    kimi_bare: str
    kimi_flagged_true: str
    nova_off: str
    nova_bare: str

    def names(self) -> tuple[str, ...]:
        return (self.kimi_bare, self.kimi_flagged_true, self.nova_off, self.nova_bare)

    def expected_schema(self, model: str) -> dict[str, JsonValue]:
        return _WIRE_LOOKAROUND_FREE if model in (self.kimi_bare, self.nova_off) else _WIRE_AS_SENT


@dataclass(frozen=True, slots=True)
class _Call:
    model: str
    endpoint: Endpoint
    stream: bool
    marker: str


@dataclass(frozen=True, slots=True)
class _Served:
    call: _Call
    status: int
    text: str


def _answer(marker: str) -> str:
    return f"answer marker-{marker}"


def _frame(event_type: str, payload: Mapping[str, JsonValue]) -> bytes:
    return _aws_event_frame(event_type, payload, "sc", "u")


def _stream_frames(marker: str) -> tuple[bytes, ...]:
    return (
        _frame("messageStart", {"role": "assistant"}),
        _frame("contentBlockDelta", {"delta": {"text": "answer "}, "contentBlockIndex": 0}),
        _frame("contentBlockDelta", {"delta": {"text": f"marker-{marker}"}, "contentBlockIndex": 0}),
        _frame("contentBlockStop", {"contentBlockIndex": 0}),
        _frame("messageStop", {"stopReason": "end_turn"}),
        _frame("metadata", {"usage": _USAGE}),
    )


def _text_reply(marker: str, stream: bool, abort_after: int | None = None) -> Reply:
    if stream:
        return Reply(content_type=_EVENT_STREAM, chunks=_stream_frames(marker), abort_after=abort_after)
    return Reply(
        body=json.dumps(
            {
                "output": {"message": {"role": "assistant", "content": [{"text": _answer(marker)}]}},
                "stopReason": "end_turn",
                "usage": _USAGE,
                "metrics": {"latencyMs": 1},
            }
        ).encode()
    )


def _marker_of(request: Request) -> str:
    found: Final = _MARKER.search(request.body.decode())
    assert found is not None, request.body
    return found.group(1)


def _is_stream(request: Request) -> bool:
    return unquote(request.target).endswith("/converse-stream")


def _echo(request: Request) -> Reply:
    return _text_reply(_marker_of(request), _is_stream(request))


def _path(endpoint: Endpoint) -> str:
    match endpoint:
        case "chat":
            return "/v1/chat/completions"
        case "messages":
            return "/v1/messages"
        case "responses":
            return "/v1/responses"


def _body(call: _Call) -> dict[str, JsonValue]:
    question: Final = f"Question marker-{call.marker}"
    common: Final[dict[str, JsonValue]] = {
        "model": call.model,
        "stream": call.stream,
        "num_retries": 0,
        "cache": {"no-cache": True},
    }
    tool: Final[dict[str, JsonValue]] = {"description": f"{_TOOL} tool"}
    match call.endpoint:
        case "chat":
            return {
                **common,
                "messages": [{"role": "user", "content": question}],
                "max_tokens": 64,
                "tools": [{"type": "function", "function": {"name": _TOOL, **tool, "parameters": _SCHEMA_AS_SENT}}],
            }
        case "messages":
            return {
                **common,
                "messages": [{"role": "user", "content": question}],
                "max_tokens": 64,
                "tools": [{"name": _TOOL, **tool, "input_schema": _SCHEMA_AS_SENT}],
            }
        case "responses":
            return {
                **common,
                "input": question,
                "max_output_tokens": 64,
                "tools": [{"type": "function", "name": _TOOL, **tool, "parameters": _SCHEMA_AS_SENT}],
            }


def _received_schema(request: Request) -> dict[str, JsonValue]:
    body: Final = _JSON.validate_json(request.body)
    (tool,) = _LIST.validate_python(object_value(body["toolConfig"])["tools"])
    spec: Final = object_value(object_value(tool)["toolSpec"])
    assert spec["name"] == _TOOL, spec
    return object_value(object_value(spec["inputSchema"])["json"])


def _assert_schemas_by_marker(received: tuple[Request, ...], calls: tuple[_Call, ...], fleet: _Fleet) -> None:
    by_marker: Final = MappingProxyType({call.marker: call for call in calls})
    assert sorted(_marker_of(request) for request in received) == sorted(by_marker), len(received)
    for request in received:
        call: Final = by_marker[_marker_of(request)]
        assert _is_stream(request) == call.stream, (call, request.target)
        assert _received_schema(request) == fleet.expected_schema(call.model), (call, request.body)


def _spend_statuses(model: str, expected: int) -> list[JsonValue]:
    rows: Final = eventually(
        lambda: read_rows('SELECT request_id, status FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
        lambda found: len(found) >= expected,
        seconds=60,
    )
    assert len({row["request_id"] for row in rows}) == len(rows), rows
    return [row["status"] for row in rows]


async def _send(client: httpx.AsyncClient, key: str, call: _Call) -> _Served:
    async with client.stream(
        "POST",
        _path(call.endpoint),
        json=_body(call),
        headers={"Authorization": f"Bearer {key}", "anthropic-version": "2023-06-01"},
    ) as response:
        raw: Final = await response.aread()
    return _Served(call=call, status=response.status_code, text=raw.decode())


async def _burst(
    base_url: str, key: str, calls: tuple[_Call, ...], *, tolerate_transport_errors: bool = False
) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(base_url=base_url, timeout=60, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(_send(client, key, call) for call in calls), return_exceptions=tolerate_transport_errors
        )
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, httpx.TransportError), repr(result)
    return tuple(result for result in results if isinstance(result, _Served))


def _mixed_calls(fleet: _Fleet, count: int) -> tuple[_Call, ...]:
    names: Final = fleet.names()
    return tuple(
        _Call(
            model=names[index % len(names)],
            endpoint=_ENDPOINTS[(index // len(names)) % len(_ENDPOINTS)],
            stream=(index // (len(names) * len(_ENDPOINTS))) % 2 == 0,
            marker=uuid.uuid4().hex,
        )
        for index in range(count)
    )


def _assert_answered_with_its_own_marker(served: _Served) -> None:
    assert served.status == 200, served.text
    assert set(_MARKER.findall(served.text)) == {served.call.marker}, served.text


def _fleet_config(wire: Wire, tmp_path: Path) -> tuple[Path, _Fleet]:
    run_id: Final = uuid.uuid4().hex[:8]
    fleet: Final = _Fleet(
        kimi_bare=f"kimi-bare-{run_id}",
        kimi_flagged_true=f"kimi-flagged-true-{run_id}",
        nova_off=f"nova-off-{run_id}",
        nova_bare=f"nova-bare-{run_id}",
    )
    params: Final[dict[str, JsonValue]] = {"api_base": wire.url, **_AWS}
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["model_list"] = [
        {"model_name": fleet.kimi_bare, "litellm_params": {"model": f"bedrock/{_KIMI}", **params}},
        {
            "model_name": fleet.kimi_flagged_true,
            "litellm_params": {"model": f"bedrock/{_KIMI}", **params},
            "model_info": {"supports_regex_lookaround": True},
        },
        {
            "model_name": fleet.nova_off,
            "litellm_params": {"model": f"bedrock/converse/{_NOVA}", **params},
            "model_info": {"supports_regex_lookaround": False},
        },
        {"model_name": fleet.nova_bare, "litellm_params": {"model": f"bedrock/converse/{_NOVA}", **params}},
    ]
    path: Final = tmp_path / "bedrock-lookaround-chaos.yaml"
    path.write_text(yaml.safe_dump(config))
    return path, fleet


def _open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


@pytest.mark.timeout(600)
async def test_a_mixed_burst_across_two_workers_cleans_only_the_flagged_deployments(
    gateway: Gateway, tmp_path: Path
) -> None:
    with wire_server(_echo) as wire:
        path, fleet = _fleet_config(wire, tmp_path)
        calls: Final = _mixed_calls(fleet, 36)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            served: Final = await _burst(str(candidate.client.base_url), candidate.key, calls)
            assert len(served) == 36
            for item in served:
                _assert_answered_with_its_own_marker(item)
            _assert_schemas_by_marker(wire.drain(), calls, fleet)
            for name in fleet.names():
                assert _spend_statuses(name, 9) == ["success"] * 9


@pytest.mark.timeout(600)
async def test_worker_sigkill_mid_burst_leaves_the_sibling_cleaning_schemas(gateway: Gateway, tmp_path: Path) -> None:
    release: Final = threading.Event()
    held_markers: Final[SimpleQueue[str]] = SimpleQueue()

    def held(request: Request) -> Reply:
        held_markers.put(_marker_of(request))
        assert release.wait(timeout=60), "The burst was never released"
        return _echo(request)

    with wire_server(held) as wire:
        path, fleet = _fleet_config(wire, tmp_path)
        calls: Final = tuple(
            _Call(model=fleet.kimi_bare, endpoint="chat", stream=False, marker=uuid.uuid4().hex) for _ in range(20)
        )
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            workers: Final = eventually(
                lambda: tuple(int(pid) for pid in _STARTED_WORKER.findall(owned.log.read_text())),
                lambda pids: len(pids) == 2,
                seconds=30,
            )
            burst: Final = asyncio.create_task(
                _burst(str(candidate.client.base_url), candidate.key, calls, tolerate_transport_errors=True)
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
            follow_up: Final = _Call(model=fleet.kimi_bare, endpoint="chat", stream=False, marker=uuid.uuid4().hex)
            (answered,) = await _burst(str(candidate.client.base_url), candidate.key, (follow_up,))
            _assert_answered_with_its_own_marker(answered)
            _assert_schemas_by_marker(wire.drain(), (*calls, follow_up), fleet)


@pytest.mark.timeout(600)
async def test_peer_stream_aborts_reach_callers_while_the_rest_of_the_burst_is_cleaned(
    gateway: Gateway, tmp_path: Path
) -> None:
    markers: Final = tuple(uuid.uuid4().hex for _ in range(12))
    aborted: Final = frozenset(marker for index, marker in enumerate(markers) if index % 3 == 0)

    def respond(request: Request) -> Reply:
        marker: Final = _marker_of(request)
        return _text_reply(marker, stream=True, abort_after=0 if marker in aborted else None)

    with wire_server(respond) as wire:
        path, fleet = _fleet_config(wire, tmp_path)
        calls: Final = tuple(
            _Call(model=fleet.kimi_bare, endpoint=_ENDPOINTS[index % 3], stream=True, marker=marker)
            for index, marker in enumerate(markers)
        )
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            served: Final = await _burst(str(candidate.client.base_url), candidate.key, calls)
            assert len(served) == 12
            for item in served:
                if item.call.marker in aborted:
                    assert "marker-" not in item.text, item.text
                    assert item.status >= 500 or "error" in item.text.lower(), (item.status, item.text)
                else:
                    _assert_answered_with_its_own_marker(item)
            recovery: Final = _Call(model=fleet.kimi_bare, endpoint="chat", stream=True, marker=uuid.uuid4().hex)
            (recovered,) = await _burst(str(candidate.client.base_url), candidate.key, (recovery,))
            _assert_answered_with_its_own_marker(recovered)
            _assert_schemas_by_marker(wire.drain(), (*calls, recovery), fleet)
