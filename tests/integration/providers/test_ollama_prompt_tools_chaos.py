import asyncio
import json
import re
import signal
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, Literal
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

_BACKEND: Final = "llama3-prompt-tools-chaos"
_API_KEY: Final = "synthetic-ollama-key"
_CONFIG_MODEL: Final = "ollama-prompt-tools-chaos"
_INSTRUCTION: Final = (
    'To call a function, reply with JSON ONLY in this format {"name": "function_name", '
    '"arguments":{"argument_name": "argument_value"}}. Once a function result answers the request, '
    "reply to the user in plain text instead of calling a function again. "
    "The following functions are available to you:"
)
_CALL_ID: Final = "call_prompt_tools_chaos_1"
_ARGUMENTS: Final[dict[str, JsonValue]] = {"city": "Paris"}
_PARAMETERS: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}
_CHAT_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "function": {"name": "get_weather", "description": "Weather for a city", "parameters": _PARAMETERS},
}
_ANTHROPIC_TOOL: Final[dict[str, JsonValue]] = {
    "name": "get_weather",
    "description": "Weather for a city",
    "input_schema": _PARAMETERS,
}
_RESPONSES_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "name": "get_weather",
    "description": "Weather for a city",
    "parameters": _PARAMETERS,
}
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MARKER: Final = re.compile(r"marker-([0-9a-f]{32})")
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_OWNED_PROXY_CELL_SECONDS: Final = 2 * graceful_stop_seconds() + 120

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


def _result(marker: str) -> str:
    return f"Paris: 22 degrees Celsius marker-{marker}"


def _answer(marker: str) -> str:
    return f"answer marker-{marker}"


def _path(endpoint: Endpoint) -> str:
    match endpoint:
        case "chat":
            return "/v1/chat/completions"
        case "messages":
            return "/v1/messages"
        case "responses":
            return "/v1/responses"


def _body(model: str, call: _Call) -> dict[str, JsonValue]:
    question: Final = "What is the weather in Paris?"
    common: Final[dict[str, JsonValue]] = {
        "model": model,
        "stream": call.stream,
        "num_retries": 0,
        "cache": {"no-cache": True},
    }
    match call.endpoint:
        case "chat":
            return {
                **common,
                "tools": [_CHAT_TOOL],
                "messages": [
                    {"role": "user", "content": question},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": _CALL_ID,
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": json.dumps(_ARGUMENTS)},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": _CALL_ID, "content": _result(call.marker)},
                ],
            }
        case "messages":
            return {
                **common,
                "max_tokens": 64,
                "tools": [_ANTHROPIC_TOOL],
                "messages": [
                    {"role": "user", "content": question},
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "id": _CALL_ID, "name": "get_weather", "input": _ARGUMENTS}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": _CALL_ID, "content": _result(call.marker)}],
                    },
                ],
            }
        case "responses":
            return {
                **common,
                "store": False,
                "tools": [_RESPONSES_TOOL],
                "input": [
                    {"role": "user", "content": question},
                    {
                        "type": "function_call",
                        "call_id": _CALL_ID,
                        "name": "get_weather",
                        "arguments": json.dumps(_ARGUMENTS),
                    },
                    {"type": "function_call_output", "call_id": _CALL_ID, "output": _result(call.marker)},
                ],
            }


def _generate_reply(marker: str, stream: bool, drop_connection: bool = False) -> Reply:
    done: Final = {
        "model": _BACKEND,
        "created_at": "2026-10-07T00:00:00Z",
        "response": "",
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 30,
        "eval_count": 12,
    }
    if not stream:
        return Reply(body=json.dumps({**done, "response": _answer(marker)}).encode(), drop_connection=drop_connection)
    pieces: Final = ("answer ", f"marker-{marker}")
    frames: Final = (
        *(
            {"model": _BACKEND, "created_at": "2026-10-07T00:00:00Z", "response": piece, "done": False}
            for piece in pieces
        ),
        done,
    )
    return Reply(
        content_type="application/x-ndjson",
        chunks=tuple(json.dumps(frame).encode() + b"\n" for frame in frames),
        drop_connection=drop_connection,
    )


def _is_generate(request: Request) -> bool:
    return (request.method, request.target) == ("POST", "/api/generate")


def _marker_of(request: Request) -> str:
    found: Final = _MARKER.search(request.body.decode())
    assert found is not None, request.body
    return found.group(1)


def _echo(request: Request) -> Reply:
    if not _is_generate(request):
        return Reply(body=b"{}")
    stream: Final = _JSON_OBJECT.validate_json(request.body).get("stream") is True
    return _generate_reply(_marker_of(request), stream)


def _assert_each_prompt_is_instructed_once(received: tuple[Request, ...], markers: frozenset[str]) -> None:
    generates: Final = tuple(request for request in received if _is_generate(request))
    assert sorted(_marker_of(request) for request in generates) == sorted(markers)
    for request in generates:
        body: Final = _JSON_OBJECT.validate_json(request.body)
        assert body["format"] == "json", sorted(body)
        prompt: Final = body["prompt"]
        assert isinstance(prompt, str)
        assert prompt.count(_INSTRUCTION) == 1, prompt
        assert set(_MARKER.findall(prompt)) == {_marker_of(request)}, prompt


def _response_id(served: _Served) -> str | None:
    if served.call.endpoint == "responses":
        return None
    if not served.call.stream:
        identity: Final = _JSON_OBJECT.validate_json(served.text)["id"]
        assert isinstance(identity, str)
        return identity
    for line in served.text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        payload: Final = _JSON_OBJECT.validate_json(line.removeprefix("data: "))
        if served.call.endpoint == "chat":
            first: Final = payload["id"]
            assert isinstance(first, str)
            return first
        if payload.get("type") == "message_start":
            message: Final = payload["message"]
            assert isinstance(message, dict) and isinstance(message["id"], str)
            return message["id"]
    raise AssertionError(served.text)


def _spend_statuses(model: str, expected: int) -> MappingProxyType[str, str]:
    rows: Final = eventually(
        lambda: read_rows('SELECT request_id, status FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
        lambda found: len(found) >= expected,
        seconds=70,
    )
    statuses: Final = MappingProxyType({str(row["request_id"]): str(row["status"]) for row in rows})
    assert len(statuses) == len(rows) == expected, rows
    return statuses


def _successes(statuses: MappingProxyType[str, str]) -> frozenset[str]:
    return frozenset(identity for identity, status in statuses.items() if status == "success")


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


def _calls(count: int, endpoints: tuple[Endpoint, ...], stream: Callable[[int], bool]) -> tuple[_Call, ...]:
    return tuple(
        _Call(endpoint=endpoints[index % len(endpoints)], stream=stream(index), marker=uuid.uuid4().hex)
        for index in range(count)
    )


def _assert_answered_with_its_own_marker(served: _Served) -> None:
    assert served.status == 200, served.text
    assert set(_MARKER.findall(served.text)) == {served.call.marker}, served.text


async def test_concurrent_tool_result_turns_across_endpoints_each_get_their_own_instructed_prompt(
    gateway: Gateway,
) -> None:
    calls: Final = _calls(24, ("chat", "messages", "responses"), lambda index: index % 2 == 0)
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        served: Final = await _burst(str(gateway.client.base_url), gateway.key, model, calls)
        assert len(served) == 24
        for item in served:
            _assert_answered_with_its_own_marker(item)
        _assert_each_prompt_is_instructed_once(wire.drain(), frozenset(call.marker for call in calls))
        known: Final = frozenset(identity for identity in map(_response_id, served) if identity is not None)
        assert len(known) == 16, known
        statuses: Final = _spend_statuses(model, 24)
        assert _successes(statuses) == frozenset(statuses), statuses
        assert known <= _successes(statuses)


async def test_dropped_ollama_connections_fail_their_callers_and_the_rest_keep_their_prompts(gateway: Gateway) -> None:
    calls: Final = _calls(12, ("chat",), lambda index: index % 2 == 1)
    dropped: Final = frozenset(call.marker for index, call in enumerate(calls) if index % 3 == 0)

    def respond(request: Request) -> Reply:
        if not _is_generate(request):
            return Reply(body=b"{}")
        marker: Final = _marker_of(request)
        stream: Final = _JSON_OBJECT.validate_json(request.body).get("stream") is True
        return _generate_reply(marker, stream, drop_connection=marker in dropped)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"ollama/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        served: Final = await _burst(str(gateway.client.base_url), gateway.key, model, calls)
        assert len(served) == 12
        for item in served:
            if item.call.marker in dropped:
                assert item.status == 500, item.text
                assert "APIConnectionError" in item.text and "marker-" not in item.text, item.text
            else:
                _assert_answered_with_its_own_marker(item)
        recovery: Final = _Call(endpoint="chat", stream=False, marker=uuid.uuid4().hex)
        (recovered,) = await _burst(str(gateway.client.base_url), gateway.key, model, (recovery,))
        _assert_answered_with_its_own_marker(recovered)
        _assert_each_prompt_is_instructed_once(wire.drain(), frozenset(call.marker for call in (*calls, recovery)))
        answered: Final = tuple(item for item in served if item.call.marker not in dropped)
        survivors: Final = frozenset(
            identity for identity in map(_response_id, (*answered, recovered)) if identity is not None
        )
        assert len(survivors) == 9, survivors
        statuses: Final = _spend_statuses(model, 13)
        assert _successes(statuses) == survivors, statuses
        assert sum(status == "failure" for status in statuses.values()) == len(dropped), statuses


def _chaos_config(wire: Wire, tmp_path: Path) -> Path:
    base: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    deployment: Final = {
        "model_name": _CONFIG_MODEL,
        "litellm_params": {"model": f"ollama/{_BACKEND}", "api_base": wire.url, "api_key": _API_KEY},
    }
    path: Final = tmp_path / "ollama-prompt-tools-chaos.yaml"
    path.write_text(yaml.safe_dump({**base, "model_list": [deployment]}))
    return path


def _open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


@pytest.mark.timeout(_OWNED_PROXY_CELL_SECONDS)
async def test_worker_sigkill_mid_burst_leaves_the_sibling_instructing_ollama(gateway: Gateway, tmp_path: Path) -> None:
    calls: Final = _calls(20, ("chat",), lambda _: False)
    release: Final = threading.Event()
    held_markers: Final[SimpleQueue[str]] = SimpleQueue()

    def held(request: Request) -> Reply:
        if not _is_generate(request):
            return Reply(body=b"{}")
        held_markers.put(_marker_of(request))
        assert release.wait(timeout=60), "The burst was never released"
        return _echo(request)

    with wire_server(held) as wire:
        path: Final = _chaos_config(wire, tmp_path)
        with owned_proxy_process(gateway, tmp_path, {}, config=path, workers=2) as owned:
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
            _assert_each_prompt_is_instructed_once(wire.drain(), frozenset(call.marker for call in (*calls, follow_up)))
