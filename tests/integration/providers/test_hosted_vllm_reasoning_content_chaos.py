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
from typing import Final, Literal

import httpx
import psutil
import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_BACKEND: Final = "qwen3-reasoning-chaos"
_API_KEY: Final = "synthetic-hosted-vllm-key"
_CONFIG_MODEL: Final = "hosted-vllm-reasoning-chaos"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MESSAGES: Final = TypeAdapter(list[dict[str, JsonValue]])
_MARKER: Final = re.compile(r"marker-([0-9a-f]{32})")
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_MODEL_LIST: Final = json.dumps(
    {"object": "list", "data": [{"id": _BACKEND, "object": "model", "owned_by": "vllm"}]}
).encode()

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


def _thought(marker: str) -> str:
    return f"private thought for {marker}"


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
    question: Final = f"Question marker-{call.marker}"
    common: Final[dict[str, JsonValue]] = {"model": model, "stream": call.stream, "num_retries": 0}
    match call.endpoint:
        case "chat":
            return {
                **common,
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": "Working on it.", "reasoning_content": _thought(call.marker)},
                    {"role": "user", "content": "Go on."},
                ],
            }
        case "messages":
            return {
                **common,
                "max_tokens": 64,
                "messages": [
                    {"role": "user", "content": question},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": _thought(call.marker), "signature": "sig"},
                            {"type": "text", "text": "Working on it."},
                        ],
                    },
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
                    },
                    {"role": "user", "content": "Go on."},
                ],
            }


def _chat_reply(marker: str, stream: bool, abort_after: int | None = None, pause: float = 0) -> Reply:
    usage: Final = {"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35}
    if not stream:
        return Reply(
            body=json.dumps(
                {
                    "id": f"chatcmpl-{marker}",
                    "object": "chat.completion",
                    "created": 1,
                    "model": _BACKEND,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": _answer(marker)},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": usage,
                }
            ).encode()
        )
    chunk: Final = {"id": f"chatcmpl-{marker}", "object": "chat.completion.chunk", "created": 1, "model": _BACKEND}
    frames: Final = (
        {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "answer "}}]},
        {**chunk, "choices": [{"index": 0, "delta": {"content": f"marker-{marker}"}}]},
        {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=(*(b"data: " + json.dumps(frame).encode() + b"\n\n" for frame in frames), b"data: [DONE]\n\n"),
        abort_after=abort_after,
        pause_between_chunks=pause,
    )


def _responses_reply(marker: str, stream: bool) -> Reply:
    identity: Final = f"resp_upstream_{marker}"
    response: Final = {
        "id": identity,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": _BACKEND,
        "output": [
            {
                "id": f"msg_{marker}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": _answer(marker), "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 30, "output_tokens": 5, "total_tokens": 35},
    }
    if not stream:
        return Reply(body=json.dumps(response).encode())
    events: Final = (
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {**response, "status": "in_progress", "output": []},
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": f"msg_{marker}",
            "output_index": 0,
            "content_index": 0,
            "delta": _answer(marker),
        },
        {"type": "response.completed", "sequence_number": 2, "response": response},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events),
    )


def _marker_of(request: Request) -> str:
    found: Final = _MARKER.search(request.body.decode())
    assert found is not None, request.body
    return found.group(1)


def _echo(request: Request) -> Reply:
    marker: Final = _marker_of(request)
    stream: Final = _JSON_OBJECT.validate_json(request.body).get("stream") is True
    if request.target == "/v1/responses":
        return _responses_reply(marker, stream)
    return _chat_reply(marker, stream)


def _forwarded_reasoning(request: Request) -> tuple[str, JsonValue]:
    body: Final = _JSON_OBJECT.validate_json(request.body)
    if request.target == "/v1/responses":
        reasoning_item: Final = _MESSAGES.validate_python(body["input"])[1]
        return _marker_of(request), _MESSAGES.validate_python(reasoning_item["summary"])[0]["text"]
    assert request.target == "/v1/chat/completions", request.target
    return _marker_of(request), _MESSAGES.validate_python(body["messages"])[1].get("reasoning_content")


def _assert_no_bleed(received: tuple[Request, ...], markers: frozenset[str]) -> None:
    forwarded: Final = [_forwarded_reasoning(request) for request in received]
    assert sorted(marker for marker, _ in forwarded) == sorted(markers)
    assert all(reasoning == _thought(marker) for marker, reasoning in forwarded), forwarded


def _spend_statuses(model: str, expected: int) -> list[JsonValue]:
    rows: Final = eventually(
        lambda: read_rows('SELECT request_id, status FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
        lambda found: len(found) >= expected,
        seconds=60,
    )
    assert len({row["request_id"] for row in rows}) == len(rows), rows
    return [row["status"] for row in rows]


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


async def test_concurrent_replays_across_endpoints_keep_each_reasoning_with_its_request(gateway: Gateway) -> None:
    calls: Final = _calls(30, ("chat", "messages", "responses"), lambda index: index % 2 == 0)
    with wire_server(_echo) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"hosted_vllm/{_BACKEND}", api_base=wire.url + "/v1", api_key=_API_KEY)
        served: Final = await _burst(str(gateway.client.base_url), gateway.key, model, calls)
        assert len(served) == 30
        for item in served:
            _assert_answered_with_its_own_marker(item)
        _assert_no_bleed(wire.drain(), frozenset(call.marker for call in calls))
        assert _spend_statuses(model, 30) == ["success"] * 30


async def test_upstream_stream_aborts_reach_callers_and_later_replays_still_forward_reasoning(
    gateway: Gateway,
) -> None:
    calls: Final = _calls(12, ("chat",), lambda _: True)
    aborted: Final = frozenset(call.marker for index, call in enumerate(calls) if index % 3 == 0)

    def respond(request: Request) -> Reply:
        marker: Final = _marker_of(request)
        return _chat_reply(marker, stream=True, abort_after=0 if marker in aborted else None)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"hosted_vllm/{_BACKEND}", api_base=wire.url + "/v1", api_key=_API_KEY)
        served: Final = await _burst(str(gateway.client.base_url), gateway.key, model, calls)
        assert len(served) == 12
        for item in served:
            if item.call.marker in aborted:
                assert item.status == 500, item.text
                assert "APIConnectionError" in item.text and "marker-" not in item.text, item.text
            else:
                _assert_answered_with_its_own_marker(item)
                assert item.text.rstrip().endswith("data: [DONE]"), item.text
        recovery: Final = _Call(endpoint="chat", stream=True, marker=uuid.uuid4().hex)
        (recovered,) = await _burst(str(gateway.client.base_url), gateway.key, model, (recovery,))
        _assert_answered_with_its_own_marker(recovered)
        _assert_no_bleed(wire.drain(), frozenset(call.marker for call in (*calls, recovery)))


async def test_slow_upstream_streams_are_forwarded_once_with_their_own_reasoning(gateway: Gateway) -> None:
    calls: Final = _calls(10, ("chat",), lambda _: True)
    with (
        wire_server(lambda request: _chat_reply(_marker_of(request), stream=True, pause=0.3)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model=f"hosted_vllm/{_BACKEND}", api_base=wire.url + "/v1", api_key=_API_KEY)
        served: Final = await _burst(str(gateway.client.base_url), gateway.key, model, calls)
        assert len(served) == 10
        for item in served:
            _assert_answered_with_its_own_marker(item)
            assert item.text.rstrip().endswith("data: [DONE]"), item.text
        _assert_no_bleed(wire.drain(), frozenset(call.marker for call in calls))
        assert _spend_statuses(model, 10) == ["success"] * 10


def _chaos_config(wire: Wire, tmp_path: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["model_list"] = [
        {
            "model_name": _CONFIG_MODEL,
            "litellm_params": {"model": f"hosted_vllm/{_BACKEND}", "api_base": wire.url + "/v1", "api_key": _API_KEY},
        }
    ]
    path: Final = tmp_path / "hosted-vllm-reasoning-chaos.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.mark.timeout(180)
async def test_worker_sigkill_mid_burst_leaves_the_sibling_forwarding_reasoning(
    gateway: Gateway, tmp_path: Path
) -> None:
    calls: Final = _calls(20, ("chat",), lambda _: False)
    release: Final = threading.Event()
    held_markers: Final[SimpleQueue[str]] = SimpleQueue()

    def held(request: Request) -> Reply:
        if (request.method, request.target) == ("GET", "/v1/models"):
            return Reply(body=_MODEL_LIST)
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
            victim: Final = psutil.Process(workers[0])
            victim.suspend()
            victim.send_signal(signal.SIGKILL)
            release.set()
            served: Final = await burst
            for item in served:
                _assert_answered_with_its_own_marker(item)
            follow_up: Final = _Call(endpoint="chat", stream=False, marker=uuid.uuid4().hex)
            (answered,) = await _burst(str(candidate.client.base_url), candidate.key, _CONFIG_MODEL, (follow_up,))
            _assert_answered_with_its_own_marker(answered)
            received: Final = wire.drain()
            chats: Final = tuple(request for request in received if request.method == "POST")
            probes: Final = [(request.method, request.target) for request in received if request.method != "POST"]
            assert set(probes) <= {("GET", "/v1/models")}, probes
            _assert_no_bleed(chats, frozenset(call.marker for call in (*calls, follow_up)))
