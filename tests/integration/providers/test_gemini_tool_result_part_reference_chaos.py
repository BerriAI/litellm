import asyncio
import json
import re
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, Literal, TypeAlias
from urllib.parse import urlsplit

import httpx
import psutil
import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.shares_worker_state

_BACKEND: Final = "gemini-2.5-flash"
_API_KEY: Final = "synthetic-gemini-key"
_CONFIG_MODEL: Final = "gemini-tool-result-part-reference-chaos"
_TOOL: Final = "inspect_schema"
_PARAMETERS: Final[Mapping[str, JsonValue]] = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
}
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MARKER: Final = re.compile(r"marker-([0-9a-f]{32})")
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_GENERATE: Final = f"/models/{_BACKEND}:generateContent"
_STREAM_GENERATE: Final = f"/models/{_BACKEND}:streamGenerateContent?alt=sse"
_USAGE: Final = {"promptTokenCount": 30, "candidatesTokenCount": 5, "totalTokenCount": 35}

Endpoint: TypeAlias = Literal["chat", "messages", "responses"]


@dataclass(frozen=True, slots=True)
class _Call:
    endpoint: Endpoint
    stream: bool
    marker: str
    reference: bool


@dataclass(frozen=True, slots=True)
class _Served:
    call: _Call
    status: int
    text: str


def _tool_text(call: _Call) -> str:
    value: Final = {"$ref": "#/$defs/humanScalar"} if call.reference else {"type": "string"}
    return json.dumps(
        {
            "title": f"marker-{call.marker}",
            "properties": {"value": value},
            "$defs": {"humanScalar": {"type": "string"}},
        },
        separators=(",", ":"),
    )


def _answer(marker: str) -> str:
    return f"answer marker-{marker}"


def _response_id(marker: str) -> str:
    return f"gemini-reply-{marker}"


def _path(endpoint: Endpoint) -> str:
    match endpoint:
        case "chat":
            return "/v1/chat/completions"
        case "messages":
            return "/v1/messages"
        case "responses":
            return "/v1/responses"


def _body(model: str, call: _Call) -> Mapping[str, JsonValue]:
    question: Final = f"Question marker-{call.marker}"
    arguments: Final = json.dumps({"name": "Config"})
    common: Final[Mapping[str, JsonValue]] = {
        "model": model,
        "stream": call.stream,
        "num_retries": 0,
        "cache": {"no-cache": True},
    }
    match call.endpoint:
        case "chat":
            return {
                **common,
                "tools": [{"type": "function", "function": {"name": _TOOL, "parameters": dict(_PARAMETERS)}}],
                "messages": [
                    {"role": "user", "content": question},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {"id": "call_0", "type": "function", "function": {"name": _TOOL, "arguments": arguments}}
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_0", "content": _tool_text(call)},
                ],
            }
        case "messages":
            return {
                **common,
                "max_tokens": 64,
                "tools": [{"name": _TOOL, "input_schema": dict(_PARAMETERS)}],
                "messages": [
                    {"role": "user", "content": question},
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": "toolu_0", "name": _TOOL, "input": {"name": "Config"}}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "toolu_0", "content": _tool_text(call)}],
                    },
                ],
            }
        case "responses":
            return {
                **common,
                "tools": [{"type": "function", "name": _TOOL, "parameters": dict(_PARAMETERS)}],
                "input": [
                    {"role": "user", "content": question},
                    {"type": "function_call", "call_id": "call_0", "name": _TOOL, "arguments": arguments},
                    {"type": "function_call_output", "call_id": "call_0", "output": _tool_text(call)},
                ],
            }


def _frame(marker: str, text: str, finished: bool) -> Mapping[str, JsonValue]:
    return {
        "responseId": _response_id(marker),
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": text}]},
                "index": 0,
                **({"finishReason": "STOP"} if finished else {}),
            }
        ],
        "usageMetadata": dict(_USAGE),
        "modelVersion": _BACKEND,
    }


def _gemini_reply(marker: str, stream: bool, abort_after: int | None = None, pause: float = 0) -> Reply:
    if not stream:
        return Reply(body=json.dumps(_frame(marker, _answer(marker), finished=True)).encode())
    frames: Final = (_frame(marker, "answer ", finished=False), _frame(marker, f"marker-{marker}", finished=True))
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"data: {json.dumps(frame)}\n\n".encode() for frame in frames),
        abort_after=abort_after,
        pause_between_chunks=pause,
    )


def _marker_of(request: Request) -> str:
    found: Final = _MARKER.search(request.body.decode())
    assert found is not None, request.body
    return found.group(1)


def _function_response(request: Request) -> Mapping[str, JsonValue]:
    contents: Final = _JSON_OBJECT.validate_json(request.body)["contents"]
    assert isinstance(contents, list), contents
    last: Final = contents[-1]
    assert isinstance(last, dict) and isinstance(last["parts"], list), last
    (part,) = last["parts"]
    assert isinstance(part, dict), part
    found: Final = part["function_response"]
    assert isinstance(found, dict), part
    return found


def _items_gemini_walks(items: list[JsonValue]) -> Iterator[JsonValue]:
    return (item for item in items if not isinstance(item, list))


def _references(node: JsonValue) -> Sequence[str]:
    if isinstance(node, dict):
        own: Final = (node["$ref"],) if isinstance(node.get("$ref"), str) and node["$ref"] else ()
        return (*own, *(ref for child in node.values() for ref in _references(child)))
    if isinstance(node, list):
        return tuple(ref for child in _items_gemini_walks(node) for ref in _references(child))
    return ()


def _vendor_rejection(request: Request) -> Reply | None:
    unresolved: Final = _references(_function_response(request)["response"])
    if not unresolved:
        return None
    message: Final = (
        f"The referenced name `{unresolved[0]}` in function_response.response does not match to a display_name in "
        "the function_response.parts."
    )
    return Reply(
        status=400,
        body=json.dumps({"error": {"code": 400, "message": message, "status": "INVALID_ARGUMENT"}}).encode(),
    )


def _echo(request: Request) -> Reply:
    assert request.target in (_GENERATE, _STREAM_GENERATE), request.target
    rejection: Final = _vendor_rejection(request)
    if rejection is not None:
        return rejection
    return _gemini_reply(_marker_of(request), stream=request.target == _STREAM_GENERATE)


def _assert_each_in_its_own_form(received: tuple[Request, ...], calls: Sequence[_Call]) -> None:
    by_marker: Final = MappingProxyType({call.marker: call for call in calls})
    forwarded: Final = tuple((_marker_of(request), _function_response(request)["response"]) for request in received)
    assert sorted(marker for marker, _ in forwarded) == sorted(by_marker), forwarded
    expected: Final = tuple(
        (marker, {"content": _tool_text(by_marker[marker])})
        if by_marker[marker].reference
        else (marker, json.loads(_tool_text(by_marker[marker])))
        for marker, _ in forwarded
    )
    assert forwarded == expected


def _spend_rows_by_model(model: str, expected: int) -> frozenset[str]:
    rows: Final = eventually(
        lambda: read_rows('SELECT request_id, status FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
        lambda found: len(found) >= expected,
        seconds=60,
    )
    assert [row["status"] for row in rows] == ["success"] * len(rows), rows
    assert len(rows) == expected, rows
    return frozenset(str(row["request_id"]) for row in rows)


def _spend_status_by_request(request_ids: frozenset[str]) -> Mapping[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id, status FROM "LiteLLM_SpendLogs" WHERE request_id = ANY(%s)',
            (list(request_ids),),  # pyright: ignore[reportArgumentType]  # psycopg adapts the list to a text array
        ),
        lambda found: len(found) >= len(request_ids),
        seconds=60,
    )
    assert len(rows) == len(request_ids), rows
    return {str(row["request_id"]): row["status"] for row in rows}


async def _send(client: httpx.AsyncClient, key: str, model: str, call: _Call) -> _Served:
    async with client.stream(
        "POST",
        _path(call.endpoint),
        json=_body(model, call),
        headers={"Authorization": f"Bearer {key}", "anthropic-version": "2023-06-01"},
    ) as response:
        raw: Final = await response.aread()
    return _Served(call=call, status=response.status_code, text=raw.decode())


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


def _calls(
    count: int, endpoints: tuple[Endpoint, ...], stream: Callable[[int], bool], reference: Callable[[int], bool]
) -> tuple[_Call, ...]:
    return tuple(
        _Call(
            endpoint=endpoints[index % len(endpoints)],
            stream=stream(index),
            marker=uuid.uuid4().hex,
            reference=reference(index),
        )
        for index in range(count)
    )


def _assert_answered_with_its_own_marker(served: _Served) -> None:
    assert served.status == 200, served.text
    assert set(_MARKER.findall(served.text)) == {served.call.marker}, served.text


async def test_a_concurrent_mixed_burst_reaches_gemini_with_each_tool_result_in_its_own_form(gateway: Gateway) -> None:
    calls: Final = _calls(
        30, ("chat", "messages", "responses"), lambda index: index % 2 == 0, lambda index: (index // 2) % 2 == 0
    )
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"gemini/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        served: Final = await _burst(str(gateway.client.base_url), gateway.key, model, calls)
        assert len(served) == 30
        for item in served:
            _assert_answered_with_its_own_marker(item)
        _assert_each_in_its_own_form(wire.drain(), calls)
        landed: Final = _spend_rows_by_model(model, 30)
        assert landed >= {
            _response_id(call.marker) for call in calls if not (call.endpoint == "messages" and call.stream)
        }


async def test_upstream_stream_drops_reach_callers_and_a_later_ref_tool_result_still_reaches_gemini(
    gateway: Gateway,
) -> None:
    calls: Final = _calls(12, ("chat", "messages", "responses"), lambda _: True, lambda _: True)
    dropped: Final = frozenset(call.marker for index, call in enumerate(calls) if index % 3 == 0)

    def respond(request: Request) -> Reply:
        rejection: Final = _vendor_rejection(request)
        if rejection is not None:
            return rejection
        marker: Final = _marker_of(request)
        return _gemini_reply(marker, stream=True, abort_after=0 if marker in dropped else None)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"gemini/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        served: Final = await _burst(str(gateway.client.base_url), gateway.key, model, calls)
        assert len(served) == 12
        for item in served:
            if item.call.marker in dropped:
                assert item.status == 500, item.text
                assert "marker-" not in item.text, item.text
            else:
                _assert_answered_with_its_own_marker(item)
        recovery: Final = _Call(endpoint="chat", stream=True, marker=uuid.uuid4().hex, reference=True)
        (recovered,) = await _burst(str(gateway.client.base_url), gateway.key, model, (recovery,))
        _assert_answered_with_its_own_marker(recovered)
        assert recovered.text.rstrip().endswith("data: [DONE]"), recovered.text
        _assert_each_in_its_own_form(wire.drain(), (*calls, recovery))


async def test_a_slow_upstream_completes_every_ref_tool_result_stream_exactly_once(gateway: Gateway) -> None:
    calls: Final = _calls(12, ("chat", "messages", "responses"), lambda _: True, lambda _: True)

    def respond(request: Request) -> Reply:
        rejection: Final = _vendor_rejection(request)
        if rejection is not None:
            return rejection
        return _gemini_reply(_marker_of(request), stream=True, pause=0.5)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"gemini/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        served: Final = await _burst(str(gateway.client.base_url), gateway.key, model, calls)
        assert len(served) == 12
        for item in served:
            _assert_answered_with_its_own_marker(item)
        _assert_each_in_its_own_form(wire.drain(), calls)
        _spend_rows_by_model(model, 12)


def _chaos_config(wire: Wire, tmp_path: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["model_list"] = [
        {
            "model_name": _CONFIG_MODEL,
            "litellm_params": {"model": f"gemini/{_BACKEND}", "api_base": wire.url, "api_key": _API_KEY},
        }
    ]
    path: Final = tmp_path / "gemini-tool-result-part-reference-chaos.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


def _non_stream_spend_ids(served: Sequence[_Served]) -> frozenset[str]:
    return frozenset(_response_id(item.call.marker) for item in served)


@dataclass(frozen=True, slots=True)
class _HeldUpstream:
    release: threading.Event
    held: SimpleQueue[str]

    def respond(self, request: Request) -> Reply:
        self.held.put(_marker_of(request))
        assert self.release.wait(timeout=120), "The burst was never released"
        return _echo(request)


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
async def test_worker_sigkill_mid_burst_leaves_the_sibling_sending_ref_tool_results_as_raw_text(
    gateway: Gateway, tmp_path: Path
) -> None:
    calls: Final = _calls(20, ("chat", "messages", "responses"), lambda _: False, lambda _: True)
    upstream: Final = _HeldUpstream(threading.Event(), SimpleQueue())
    with wire_server(upstream.respond) as wire:
        path: Final = _chaos_config(wire, tmp_path)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
            candidate: Final = owned.gateway
            workers: Final = eventually(
                lambda: tuple(int(pid) for pid in _STARTED_WORKER.findall(owned.log.read_text())),
                lambda pids: len(pids) == 2,
                seconds=graceful_stop_seconds() + 60,
            )
            burst: Final = asyncio.create_task(
                _burst(
                    str(candidate.client.base_url), candidate.key, _CONFIG_MODEL, calls, tolerate_transport_errors=True
                )
            )
            await asyncio.to_thread(eventually, upstream.held.qsize, lambda size: size == 20, 60)
            held_by: Final = MappingProxyType({pid: _open_upstream_connections(pid, wire.url) for pid in workers})
            assert sum(held_by.values()) == 20, held_by
            victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
            with owned.replacing_worker(victim_pid) as kill:
                kill.freeze()
                await asyncio.to_thread(kill)
                upstream.release.set()
                served: Final = await burst
                assert held_by[survivor_pid] >= 10, held_by
                assert len(served) == held_by[survivor_pid], (held_by, len(served))
                for item in served:
                    _assert_answered_with_its_own_marker(item)
            follow_up: Final = _Call(endpoint="messages", stream=False, marker=uuid.uuid4().hex, reference=True)
            (answered,) = await _burst(str(candidate.client.base_url), candidate.key, _CONFIG_MODEL, (follow_up,))
            _assert_answered_with_its_own_marker(answered)
            received: Final = wire.drain()
            assert {(request.method, request.target) for request in received} == {("POST", _GENERATE)}, received
            _assert_each_in_its_own_form(received, (*calls, follow_up))
            assert _spend_status_by_request(_non_stream_spend_ids((*served, answered))) == {
                spend_id: "success" for spend_id in _non_stream_spend_ids((*served, answered))
            }


@pytest.mark.timeout(600)
async def test_proxy_sigterm_mid_burst_finishes_in_flight_ref_tool_results_and_a_relaunch_serves_again(
    gateway: Gateway, tmp_path: Path
) -> None:
    calls: Final = _calls(9, ("chat", "messages", "responses"), lambda _: False, lambda _: True)
    upstream: Final = _HeldUpstream(threading.Event(), SimpleQueue())
    with wire_server(upstream.respond) as wire:
        first: Final = tmp_path / "first"
        first.mkdir()
        with owned_proxy_process(gateway, first, {}, config=_chaos_config(wire, first), workers=2) as owned:
            burst: Final = asyncio.create_task(
                _burst(
                    str(owned.gateway.client.base_url),
                    owned.gateway.key,
                    _CONFIG_MODEL,
                    calls,
                    tolerate_transport_errors=True,
                )
            )
            await asyncio.to_thread(eventually, upstream.held.qsize, lambda size: size == len(calls), 120)
            owned.begin_shutdown()
            upstream.release.set()
            served: Final = await burst
            exit_code: Final = await asyncio.to_thread(
                eventually, owned.process.poll, lambda code: code is not None, 120
            )
            assert exit_code is not None
        assert len(served) == len(calls), (len(served), len(calls))
        for item in served:
            _assert_answered_with_its_own_marker(item)
        _assert_each_in_its_own_form(wire.drain(), calls)
        second: Final = tmp_path / "second"
        second.mkdir()
        with owned_proxy_process(gateway, second, {}, config=_chaos_config(wire, second), workers=2) as relaunched:
            follow_up: Final = _Call(endpoint="chat", stream=False, marker=uuid.uuid4().hex, reference=True)
            (answered,) = await _burst(
                str(relaunched.gateway.client.base_url), relaunched.gateway.key, _CONFIG_MODEL, (follow_up,)
            )
            _assert_answered_with_its_own_marker(answered)
            _assert_each_in_its_own_form(wire.drain(), (follow_up,))
            assert _spend_status_by_request(_non_stream_spend_ids((*served, answered))) == {
                spend_id: "success" for spend_id in _non_stream_spend_ids((*served, answered))
            }
