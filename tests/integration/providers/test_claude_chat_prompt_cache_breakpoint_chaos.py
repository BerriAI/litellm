import asyncio
import socket
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from typing import Final, Literal

import httpx
import pytest
from integration._support import bedrock_runtime_peer
from integration._support.client import Gateway, Scenario, eventually
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.prompt_cache_breakpoint import open_upstream_connections, started_worker_pids
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._cache_control_marks_support import (
    ASK,
    ASK_LABEL,
    BEDROCK_MODEL,
    PROVIDER_KEY,
    SYSTEM,
    bedrock_labels,
    final_text,
    new_marker,
    owned_config,
)
from pydantic import JsonValue

Kind = Literal["chat", "chat_stream", "messages", "messages_stream"]

_KINDS: Final[tuple[Kind, ...]] = ("chat", "chat_stream", "messages", "messages_stream")
_EXPLICIT: Final[dict[str, JsonValue]] = {"mode": "explicit"}
_CONVERSE_SYSTEM: Final = f"system:text:{SYSTEM}"
_OWNED_MODEL: Final = "breakpoint-converse"
_HOLD_SECONDS: Final = 60


@dataclass(frozen=True, slots=True)
class _Call:
    kind: Kind
    marker: str


@dataclass(frozen=True, slots=True)
class _Answer:
    call: _Call
    status: int
    text: str


@dataclass(frozen=True, slots=True)
class _Held:
    release: threading.Event
    arrived: SimpleQueue[str]

    def respond(self, request: Request) -> Reply:
        self.arrived.put(bedrock_runtime_peer.marker_of(request))
        self.release.wait(timeout=_HOLD_SECONDS)
        return bedrock_runtime_peer.respond(request)


def _calls(count: int) -> tuple[_Call, ...]:
    return tuple(_Call(_KINDS[index % len(_KINDS)], new_marker()) for index in range(count))


def _converse_params(url: str) -> dict[str, JsonValue]:
    return {
        "model": f"bedrock/converse/{BEDROCK_MODEL}",
        "api_key": PROVIDER_KEY,
        "aws_region_name": "us-east-1",
        "aws_bedrock_runtime_endpoint": url,
    }


def _converse(scenario: Scenario, url: str) -> str:
    return scenario.model(**_converse_params(url))


def _request(call: _Call, model: str) -> tuple[str, dict[str, JsonValue]]:
    stream: Final = call.kind.endswith("_stream")
    if call.kind.startswith("chat"):
        return "/v1/chat/completions", {
            "model": model,
            "max_tokens": 64,
            "stream": stream,
            "num_retries": 0,
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM, "prompt_cache_breakpoint": _EXPLICIT}]},
                {"role": "user", "content": final_text(call.marker)},
            ],
        }
    return "/v1/messages", {
        "model": model,
        "max_tokens": 64,
        "stream": stream,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ASK, "prompt_cache_breakpoint": _EXPLICIT},
                    {"type": "text", "text": final_text(call.marker)},
                ],
            }
        ],
    }


def _expected_label(call: _Call) -> str:
    return _CONVERSE_SYSTEM if call.kind.startswith("chat") else ASK_LABEL


async def _send(client: httpx.AsyncClient, key: str, model: str, call: _Call) -> _Answer:
    path, body = _request(call, model)
    async with client.stream("POST", path, json=body, headers={"Authorization": f"Bearer {key}"}) as response:
        raw: Final = (await response.aread()).decode()
    return _Answer(call, response.status_code, raw)


async def _burst(
    base_url: str, key: str, model: str, calls: tuple[_Call, ...], *, tolerate_transport_errors: bool = False
) -> tuple[_Answer, ...]:
    async with httpx.AsyncClient(base_url=base_url, timeout=90, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(_send(client, key, model, call) for call in calls), return_exceptions=tolerate_transport_errors
        )
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, httpx.TransportError), repr(result)
    return tuple(result for result in results if isinstance(result, _Answer))


def _posts(wire: Wire) -> tuple[Request, ...]:
    return tuple(request for request in wire.drain() if request.method == "POST")


def _assert_answered(answers: tuple[_Answer, ...]) -> None:
    for answer in answers:
        assert answer.status == 200, answer
        assert f"marker-{answer.call.marker}" in answer.text, answer


def _assert_marked(received: tuple[Request, ...], calls: tuple[_Call, ...]) -> None:
    by_marker: Final = {bedrock_runtime_peer.marker_of(request): request for request in received}
    assert len(received) == len(by_marker) == len(calls), [request.target for request in received]
    assert set(by_marker) == {call.marker for call in calls}, by_marker.keys()
    for call in calls:
        assert bedrock_labels(by_marker[call.marker]) == [_expected_label(call)], call


def _reserved_port() -> int:
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        return reserve.getsockname()[1]


async def test_a_mixed_burst_marks_every_request_once(gateway: Gateway) -> None:
    calls: Final = _calls(32)
    with wire_server(bedrock_runtime_peer.respond) as wire, gateway.scenario() as scenario:
        model: Final = _converse(scenario, wire.url)
        answers: Final = await _burst(str(gateway.client.base_url), gateway.key, model, calls)
        received: Final = _posts(wire)
    assert len(answers) == len(calls)
    _assert_answered(answers)
    _assert_marked(received, calls)


async def test_an_upstream_outage_mid_burst_reaches_the_caller_and_recovered_requests_are_marked(
    gateway: Gateway,
) -> None:
    port: Final = _reserved_port()
    url: Final = str(gateway.client.base_url)
    before_calls, during_calls, after_calls = _calls(8), _calls(8), _calls(8)
    with gateway.scenario() as scenario:
        model: Final = _converse(scenario, f"http://127.0.0.1:{port}")
        with wire_server(bedrock_runtime_peer.respond, port=port) as first:
            before: Final = await _burst(url, gateway.key, model, before_calls)
            first_received: Final = _posts(first)
        during: Final = await _burst(url, gateway.key, model, during_calls)
        liveliness: Final = gateway.request("GET", "/health/liveliness")
        with wire_server(bedrock_runtime_peer.respond, port=port) as second:
            after: Final = await _burst(url, gateway.key, model, after_calls)
            second_received: Final = _posts(second)
    _assert_answered(before)
    _assert_marked(first_received, before_calls)
    assert len(during) == len(during_calls), during
    assert all(answer.status >= 500 and "error" in answer.text for answer in during), during
    assert liveliness.status_code == 200, liveliness.text
    _assert_answered(after)
    _assert_marked(second_received, after_calls)


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
async def test_a_worker_killed_mid_burst_leaves_the_survivor_answering_with_marks(
    gateway: Gateway, tmp_path: Path
) -> None:
    held: Final = _Held(threading.Event(), SimpleQueue())
    calls: Final = _calls(12)
    follow_up_calls: Final = _calls(8)
    with wire_server(held.respond) as wire:
        config: Final = owned_config(
            tmp_path, [{"model_name": _OWNED_MODEL, "litellm_params": _converse_params(wire.url)}]
        )
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            url: Final = str(owned.gateway.client.base_url)
            workers: Final = eventually(lambda: started_worker_pids(owned.log), lambda pids: len(pids) == 2, seconds=60)
            burst: Final = asyncio.create_task(
                _burst(url, owned.gateway.key, _OWNED_MODEL, calls, tolerate_transport_errors=True)
            )
            await asyncio.to_thread(eventually, held.arrived.qsize, lambda size: size == len(calls), 60)
            held_by: Final = {pid: open_upstream_connections(pid, wire.url) for pid in workers}
            assert sum(held_by.values()) == len(calls), held_by
            victim, survivor = sorted(workers, key=held_by.__getitem__)
            with owned.replacing_worker(victim) as kill:
                await asyncio.to_thread(kill)
                held.release.set()
                served: Final = await burst
            follow_up: Final = await _burst(url, owned.gateway.key, _OWNED_MODEL, follow_up_calls)
        received: Final = _posts(wire)
    assert len(served) == held_by[survivor], (held_by, served)
    _assert_answered(served)
    assert len(follow_up) == len(follow_up_calls), follow_up
    _assert_answered(follow_up)
    _assert_marked(received, (*calls, *follow_up_calls))
