import asyncio
import os
import re
import signal
import socket
import threading
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, wire_server
from integration.providers._cache_control_marks_support import (
    ASK_LABEL,
    CITIES,
    POINTS,
    SYSTEM_LABEL,
    anthropic_deployment,
    anthropic_labels,
    anthropic_peer,
    chat_body,
    client_marked,
    conversation,
    final_label,
    marked_calls,
    marker_of,
    messages_body,
    new_marker,
    owned_config,
    responses_body,
    tool_call,
    tool_use_label,
)
from pydantic import JsonValue

_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_SURFACES: Final = ("chat", "chat-stream", "chat-unmarked", "messages", "responses")
_MODEL: Final = "capped-claude"
_BURST: Final = 30
_OUTAGE_STATUS: Final = 500


@dataclass(frozen=True, slots=True)
class _Sent:
    surface: str
    marker: str
    status: int
    text: str
    call_id: str


def _free_port() -> int:
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        return int(reserve.getsockname()[1])


def _surface_request(surface: str, marker: str) -> tuple[str, dict[str, JsonValue]]:
    if surface == "messages":
        return "/v1/messages", messages_body(_MODEL, marker, stream=False)
    if surface == "responses":
        return "/v1/responses", responses_body(_MODEL, marker)
    if surface == "chat-unmarked":
        unmarked: Final = [tool_call(city) for city in CITIES]
        return "/v1/chat/completions", chat_body(_MODEL, conversation(marker, unmarked, ask_marked=False))
    return "/v1/chat/completions", chat_body(
        _MODEL, conversation(marker, marked_calls()), stream=surface == "chat-stream"
    )


def _expected_labels(item: _Sent) -> list[str]:
    if item.surface == "responses":
        return [SYSTEM_LABEL, ASK_LABEL]
    if item.surface == "chat-unmarked":
        return [SYSTEM_LABEL, final_label(item.marker)]
    return client_marked()


async def _fire(owned_url: str, key: str, *, tolerate_transport_errors: bool = False) -> tuple[_Sent, ...]:
    async def one(client: httpx.AsyncClient, index: int) -> _Sent:
        surface: Final = _SURFACES[index % len(_SURFACES)]
        marker: Final = new_marker()
        path, body = _surface_request(surface, marker)
        response: Final = await client.post(path, json=body, headers={"Authorization": f"Bearer {key}"})
        return _Sent(
            surface, marker, response.status_code, response.text, response.headers.get("x-litellm-call-id", "")
        )

    async with httpx.AsyncClient(base_url=owned_url, timeout=30, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(one(client, index) for index in range(_BURST)), return_exceptions=tolerate_transport_errors
        )
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, httpx.TransportError), repr(result)
    return tuple(result for result in results if isinstance(result, _Sent))


def _held_anthropic_peer(release: threading.Event) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert release.wait(timeout=120), "Held upstream was never released"
        return anthropic_peer(request)

    return respond


def _by_marker(received: Sequence[Request]) -> dict[str, tuple[Request, ...]]:
    counted: Final = Counter(marker_of(request) for request in received)
    return {marker: tuple(request for request in received if marker_of(request) == marker) for marker in counted}


def _assert_capped(served: Sequence[_Sent], received: Sequence[Request]) -> None:
    by_marker: Final = _by_marker(received)
    for item in served:
        assert item.status == 200, (item.surface, item.text)
        assert len(by_marker.get(item.marker, ())) == 1, (item.surface, item.marker)
        assert anthropic_labels(by_marker[item.marker][0]) == _expected_labels(item), item.surface


def _assert_outage_error(item: _Sent, received: Sequence[Request]) -> None:
    assert item.status == _OUTAGE_STATUS, (item.surface, item.status, item.text)
    assert '"error"' in item.text, (item.surface, item.text)
    assert item.marker not in {marker_of(request) for request in received}, item.surface


def _spend_rows(call_id: str) -> list[dict[str, JsonValue]]:
    return read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE litellm_call_id=%s', (call_id,))


def _single_spend_row(item: _Sent) -> None:
    assert item.call_id, (item.surface, item.status, item.text)
    rows: Final = eventually(lambda: _spend_rows(item.call_id), lambda values: len(values) == 1, seconds=70)
    assert len(rows) == 1, (item.surface, item.call_id)


@pytest.mark.timeout(240)
async def test_capped_burst_rides_out_a_provider_outage_and_logs_every_request_once(
    gateway: Gateway, tmp_path: Path
) -> None:
    port: Final = _free_port()
    config: Final = owned_config(
        tmp_path, [anthropic_deployment(_MODEL, f"http://127.0.0.1:{port}", cache_control_injection_points=POINTS)]
    )
    with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
        owned_url: Final = str(owned.gateway.client.base_url)
        with wire_server(anthropic_peer, port=port) as wire:
            healthy: Final = await _fire(owned_url, owned.gateway.key)
            _assert_capped(healthy, wire.drain())
            racing: Final = asyncio.create_task(_fire(owned_url, owned.gateway.key))
            await asyncio.to_thread(eventually, lambda: wire.received.qsize(), lambda size: size >= 5, 30)
        down: Final = await _fire(owned_url, owned.gateway.key)
        raced: Final = await racing
        raced_received: Final = wire.drain()
        with wire_server(anthropic_peer, port=port) as restarted:
            recovered: Final = await _fire(owned_url, owned.gateway.key)
            recovered_received: Final = restarted.drain()
        assert len(_STARTED_WORKER.findall(owned.log.read_text())) >= 2
    assert all(len(requests) == 1 for requests in _by_marker(raced_received).values())
    _assert_capped(tuple(item for item in raced if item.status == 200), raced_received)
    for item in raced:
        if item.status != 200:
            _assert_outage_error(item, raced_received)
    for item in down:
        _assert_outage_error(item, (*raced_received, *recovered_received))
    _assert_capped(recovered, recovered_received)
    assert {marker_of(request) for request in recovered_received} == {item.marker for item in recovered}
    for item in (*healthy, *raced, *down, *recovered):
        _single_spend_row(item)


@pytest.mark.timeout(240)
async def test_worker_sigkill_mid_burst_leaves_the_sibling_serving_capped_requests(
    gateway: Gateway, tmp_path: Path
) -> None:
    release: Final = threading.Event()
    with wire_server(_held_anthropic_peer(release)) as wire:
        config: Final = owned_config(
            tmp_path, [anthropic_deployment(_MODEL, wire.url, cache_control_injection_points=POINTS)]
        )
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            workers: Final = eventually(
                lambda: tuple(int(pid) for pid in _STARTED_WORKER.findall(owned.log.read_text())),
                lambda pids: len(pids) == 2,
                seconds=30,
            )
            owned_url: Final = str(owned.gateway.client.base_url)
            burst: Final = asyncio.create_task(_fire(owned_url, owned.gateway.key, tolerate_transport_errors=True))
            try:
                await asyncio.to_thread(eventually, lambda: wire.received.qsize(), lambda size: size >= _BURST, 20)
                os.kill(workers[0], signal.SIGKILL)
            finally:
                release.set()
            served: Final = await burst
            during: Final = wire.drain()
            after: Final = await _fire(owned_url, owned.gateway.key)
            after_received: Final = wire.drain()
    assert 0 < len(served) < _BURST, len(served)
    assert all(len(requests) == 1 for requests in _by_marker(during).values())
    _assert_capped(served, during)
    _assert_capped(after, after_received)
    for item in (*served, *after):
        _single_spend_row(item)


@pytest.mark.timeout(240)
def test_yaml_auto_caching_stands_down_for_tool_call_marks_and_outranks_a_key_opt_out(
    gateway: Gateway, tmp_path: Path
) -> None:
    markers: Final = (new_marker(), new_marker(), new_marker())
    unmarked: Final = [tool_call(city) for city in CITIES]
    with wire_server(anthropic_peer) as wire:
        config: Final = owned_config(
            tmp_path,
            [anthropic_deployment(_MODEL, wire.url)],
            litellm_settings={"enable_anthropic_prompt_caching": True},
        )
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            with owned.gateway.scenario() as scenario:
                opted_out: Final = scenario.key(metadata={"enable_prompt_caching": False})
                cells: Final = (
                    (conversation(markers[0], marked_calls(), ask_marked=False), owned.gateway.key),
                    (conversation(markers[1], unmarked, ask_marked=False), owned.gateway.key),
                    (conversation(markers[2], unmarked, ask_marked=False), opted_out),
                )
                responses: Final = tuple(
                    owned.gateway.request("POST", "/v1/chat/completions", chat_body(_MODEL, messages), key=key)
                    for messages, key in cells
                )
        received: Final = _by_marker(wire.drain())
    assert [response.status_code for response in responses] == [200, 200, 200], [
        response.text for response in responses
    ]
    assert [anthropic_labels(received[marker][0]) for marker in markers] == [
        [tool_use_label(city) for city in CITIES],
        [SYSTEM_LABEL, final_label(markers[1])],
        [SYSTEM_LABEL, final_label(markers[2])],
    ]
