import asyncio
import json
import re
import signal
import threading
import uuid
from collections.abc import Callable, Mapping
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
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "gemini-2.5-flash"
_API_KEY: Final = "synthetic-gemini-key"
_CONFIG_MODEL: Final = "gemini-thinking-replay-chaos"
_CLAUDE_SIGNATURE: Final = "CAQSyAsKEAgSGAI4AUIIdGhpbmtpbmcSDAlY-synthetic-claude-signature"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_CONTENTS: Final = TypeAdapter(list[dict[str, JsonValue]])
_PARTS: Final = TypeAdapter(list[dict[str, JsonValue]])
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


@dataclass(frozen=True, slots=True)
class _Served:
    call: _Call
    status: int
    text: str


def _thought(marker: str) -> str:
    return f"private thought for {marker}"


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


def _block(marker: str) -> Mapping[str, JsonValue]:
    return {"type": "thinking", "thinking": _thought(marker), "signature": _CLAUDE_SIGNATURE}


def _body(model: str, call: _Call) -> Mapping[str, JsonValue]:
    question: Final = f"Question marker-{call.marker}"
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
                "messages": [
                    {"role": "user", "content": question},
                    {
                        "role": "assistant",
                        "content": "Working on it.",
                        "reasoning_content": _thought(call.marker),
                        "thinking_blocks": [_block(call.marker)],
                    },
                    {"role": "user", "content": "Go on."},
                ],
            }
        case "messages":
            return {
                **common,
                "max_tokens": 64,
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": [_block(call.marker), {"type": "text", "text": "Working on it."}]},
                    {"role": "user", "content": "Go on."},
                ],
            }
        case "responses":
            return {
                **common,
                "input": [
                    {"role": "user", "content": question},
                    {
                        "id": f"rs_{call.marker}",
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": _thought(call.marker)}],
                        "encrypted_content": json.dumps([_block(call.marker)]),
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "id": f"msg_{call.marker}",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Working on it.", "annotations": []}],
                    },
                    {"role": "user", "content": "Go on."},
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


def _echo(request: Request) -> Reply:
    assert request.target in (_GENERATE, _STREAM_GENERATE), request.target
    return _gemini_reply(_marker_of(request), stream=request.target == _STREAM_GENERATE)


def _forwarded_thought(request: Request) -> tuple[str, JsonValue]:
    body: Final = _JSON_OBJECT.validate_json(request.body)
    assert "thoughtSignature" not in request.body.decode(), request.body
    model_turn: Final = _CONTENTS.validate_python(body["contents"])[1]
    assert model_turn["role"] == "model", model_turn
    parts: Final = _PARTS.validate_python(model_turn["parts"])
    assert [part.get("thought") for part in parts] == [True, None], parts
    return _marker_of(request), parts[0]["text"]


def _assert_no_bleed(received: tuple[Request, ...], markers: frozenset[str]) -> None:
    forwarded: Final = [_forwarded_thought(request) for request in received]
    assert sorted(marker for marker, _ in forwarded) == sorted(markers)
    assert all(thought == _thought(marker) for marker, thought in forwarded), forwarded


def _spend_request_ids(model: str, expected: int) -> frozenset[str]:
    rows: Final = eventually(
        lambda: read_rows('SELECT request_id, status FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
        lambda found: len(found) >= expected,
        seconds=60,
    )
    assert [row["status"] for row in rows] == ["success"] * len(rows), rows
    assert len(rows) == expected, rows
    return frozenset(str(row["request_id"]) for row in rows)


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


async def test_concurrent_thinking_replays_across_endpoints_reach_gemini_signature_free(gateway: Gateway) -> None:
    calls: Final = _calls(30, ("chat", "messages", "responses"), lambda index: index % 2 == 0)
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"gemini/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        served: Final = await _burst(str(gateway.client.base_url), gateway.key, model, calls)
        assert len(served) == 30
        for item in served:
            _assert_answered_with_its_own_marker(item)
        _assert_no_bleed(wire.drain(), frozenset(call.marker for call in calls))
        landed: Final = _spend_request_ids(model, 30)
        assert landed >= {
            _response_id(call.marker) for call in calls if not (call.endpoint == "messages" and call.stream)
        }


async def test_upstream_stream_drops_reach_callers_and_later_replays_still_reach_gemini(gateway: Gateway) -> None:
    calls: Final = _calls(12, ("chat", "messages", "responses"), lambda _: True)
    dropped: Final = frozenset(call.marker for index, call in enumerate(calls) if index % 3 == 0)

    def respond(request: Request) -> Reply:
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
        recovery: Final = _Call(endpoint="chat", stream=True, marker=uuid.uuid4().hex)
        (recovered,) = await _burst(str(gateway.client.base_url), gateway.key, model, (recovery,))
        _assert_answered_with_its_own_marker(recovered)
        assert recovered.text.rstrip().endswith("data: [DONE]"), recovered.text
        _assert_no_bleed(wire.drain(), frozenset(call.marker for call in (*calls, recovery)))


def _chaos_config(wire: Wire, tmp_path: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["model_list"] = [
        {
            "model_name": _CONFIG_MODEL,
            "litellm_params": {"model": f"gemini/{_BACKEND}", "api_base": wire.url, "api_key": _API_KEY},
        }
    ]
    path: Final = tmp_path / "gemini-thinking-replay-chaos.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


@pytest.mark.timeout(180)
async def test_worker_sigkill_mid_burst_leaves_the_sibling_replaying_thinking_signature_free(
    gateway: Gateway, tmp_path: Path
) -> None:
    calls: Final = _calls(20, ("chat", "messages", "responses"), lambda _: False)
    release: Final = threading.Event()
    held_markers: Final[SimpleQueue[str]] = SimpleQueue()

    def held(request: Request) -> Reply:
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
            follow_up: Final = _Call(endpoint="messages", stream=False, marker=uuid.uuid4().hex)
            (answered,) = await _burst(str(candidate.client.base_url), candidate.key, _CONFIG_MODEL, (follow_up,))
            _assert_answered_with_its_own_marker(answered)
            received: Final = wire.drain()
            assert {(request.method, request.target) for request in received} == {("POST", _GENERATE)}, received
            _assert_no_bleed(received, frozenset(call.marker for call in (*calls, follow_up)))
