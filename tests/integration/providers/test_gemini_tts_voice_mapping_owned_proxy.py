"""The speech route, the audio_speech health check and deferred Vertex Live setups map OpenAI voices.

The speech bridge calls the Gemini chat path without the deployment's ``api_base`` (tracked separately),
so these rows boot their own proxy with ``GEMINI_API_BASE`` pointed at the scripted peer, and with
``LITELLM_GEMINI_LIVE_DEFER_SETUP`` so a client ``session.update`` voice builds the Vertex Live setup.
"""

from __future__ import annotations

import asyncio
import os
import re
import signal
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final
from urllib.parse import urlsplit

import httpx
import psutil
import pytest
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._gemini_tts_voice_support import (
    BACKEND,
    GEMINI_API_KEY,
    HEALTH_DEFAULT,
    HEALTH_FABLE,
    HEALTH_NOVA,
    NO_VOICE,
    PCM,
    REJECTED_PREFIX,
    async_openai_client,
    frame_types,
    gemini_peer,
    health_config,
    live_deployment,
    live_scenario,
    live_setups,
    live_turn,
    marker,
    openai_client,
    received_text,
    received_voice,
    speech_deployment,
    spend_row_by_call,
    voice_in,
    wav_payload,
)
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
BURST: Final = 12
VOICES: Final = ("alloy", "fable", "Kore")
MAPPED: Final = MappingProxyType({"alloy": "Kore", "fable": "Umbriel", "Kore": "Kore"})


def _overrides(wire_url: str) -> dict[str, str]:
    return {"GEMINI_API_BASE": wire_url, "LITELLM_GEMINI_LIVE_DEFER_SETUP": "true"}


@pytest.fixture(scope="module")
def shared_wire() -> Iterator[Wire]:
    with wire_server(gemini_peer) as wire:
        yield wire


@pytest.fixture(scope="module")
def owned(shared_wire: Wire, tmp_path_factory: pytest.TempPathFactory) -> Iterator[OwnedProxy]:
    directory: Final = tmp_path_factory.mktemp("gemini-tts")
    with gateway_from_environment() as gateway:
        config: Final = health_config(directory)
        with owned_proxy_process(gateway, directory, _overrides(shared_wire.url), config=config, workers=2) as proxy:
            yield proxy


def _speech_body(model: str, text: str, voice: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "input": text, "voice": voice, **extra}


def _received(wire: Wire, text: str) -> Request:
    matching: Final = tuple(request for request in wire.drain() if received_text(request) == text)
    assert len(matching) == 1, [request.target for request in matching]
    return matching[0]


def _assert_speech_logged(call_id: str) -> None:
    row: Final = spend_row_by_call(call_id)
    assert row["call_type"] == "aspeech", row


def test_speech_route_maps_alloy_and_answers_wav(owned: OwnedProxy, shared_wire: Wire) -> None:
    with owned.gateway.scenario() as scenario:
        model: Final = speech_deployment(scenario, shared_wire.url)
        text: Final = marker()
        response: Final = owned.gateway.request("POST", "/v1/audio/speech", _speech_body(model, text, "alloy"))
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("audio/wav"), response.headers
        assert wav_payload(response.content) == PCM
        assert received_voice(_received(shared_wire, text)) == "Kore"
        _assert_speech_logged(response.headers.get("x-litellm-call-id", ""))


def test_speech_sdk_maps_alloy_and_answers_wav(owned: OwnedProxy, shared_wire: Wire) -> None:
    with owned.gateway.scenario() as scenario:
        model: Final = speech_deployment(scenario, shared_wire.url)
        text: Final = marker()
        raw: Final = openai_client(owned.gateway).audio.speech.with_raw_response.create(
            model=model, input=text, voice="alloy"
        )
        assert wav_payload(raw.content) == PCM
        assert received_voice(_received(shared_wire, text)) == "Kore"
        _assert_speech_logged(raw.headers.get("x-litellm-call-id", ""))


async def _async_speech(gateway: Gateway, model: str, text: str) -> tuple[bytes, str, str]:
    client: Final = async_openai_client(gateway)
    try:
        raw: Final = await client.audio.speech.with_raw_response.create(
            model=model, input=text, voice="Alloy", response_format="pcm"
        )
        return raw.content, raw.headers.get("content-type", ""), raw.headers.get("x-litellm-call-id", "")
    finally:
        await client.close()


def test_async_speech_sdk_maps_a_capitalised_alloy_and_answers_raw_pcm(owned: OwnedProxy, shared_wire: Wire) -> None:
    with owned.gateway.scenario() as scenario:
        model: Final = speech_deployment(scenario, shared_wire.url)
        text: Final = marker()
        content, content_type, call_id = asyncio.run(_async_speech(owned.gateway, model, text))
        assert content == PCM, content[:16]
        assert content_type.startswith("audio/pcm"), content_type
        assert received_voice(_received(shared_wire, text)) == "Kore"
        _assert_speech_logged(call_id)


@pytest.mark.parametrize("voice", ["nova", "Kore"])
def test_speech_route_forwards_gemini_voice_names_verbatim(owned: OwnedProxy, shared_wire: Wire, voice: str) -> None:
    with owned.gateway.scenario() as scenario:
        model: Final = speech_deployment(scenario, shared_wire.url)
        text: Final = marker()
        response: Final = owned.gateway.request("POST", "/v1/audio/speech", _speech_body(model, text, voice))
        assert response.status_code == 200, response.text
        assert wav_payload(response.content) == PCM
        assert received_voice(_received(shared_wire, text)) == voice
        _assert_speech_logged(response.headers.get("x-litellm-call-id", ""))


def _health(gateway: Gateway, model: str) -> dict[str, JsonValue]:
    response: Final = gateway.request("GET", "/health", params={"model": model})
    return JSON_OBJECT.validate_json(response.content)


def _counts(report: dict[str, JsonValue]) -> tuple[JsonValue, JsonValue]:
    return report["healthy_count"], report["unhealthy_count"]


def _only_voice(wire: Wire) -> JsonValue:
    requests: Final = wire.drain()
    assert len(requests) == 1, [request.target for request in requests]
    return received_voice(requests[0])


@pytest.mark.parametrize(
    ("model", "mapped"), [(HEALTH_DEFAULT, "Kore"), (HEALTH_NOVA, "nova"), (HEALTH_FABLE, "Umbriel")]
)
def test_audio_speech_health_check_reaches_the_peer_with_a_gemini_voice(
    owned: OwnedProxy, shared_wire: Wire, model: str, mapped: str
) -> None:
    shared_wire.drain()
    report: Final = _health(owned.gateway, model)
    assert _counts(report) == (1, 0), report
    assert _only_voice(shared_wire) == mapped


def test_unknown_health_check_voice_reports_the_vendor_rejection(owned: OwnedProxy, shared_wire: Wire) -> None:
    with owned.gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"gemini/{BACKEND}",
            api_base=shared_wire.url,
            api_key=GEMINI_API_KEY,
            model_info={"mode": "audio_speech", "health_check_voice": "Custom-Voice"},
        )
        shared_wire.drain()
        report: Final = _health(owned.gateway, model)
        assert _counts(report) == (0, 1), report
        unhealthy: Final = report["unhealthy_endpoints"]
        assert isinstance(unhealthy, list) and len(unhealthy) == 1, report
        assert f"{REJECTED_PREFIX}Custom-Voice" in string_value(object_value(unhealthy[0])["error"]), report
        assert _only_voice(shared_wire) == "Custom-Voice"


@pytest.mark.parametrize(
    ("voice", "expected"), [("fable", "Umbriel"), ("alloy", NO_VOICE), ("Kore", "Kore")], ids=["fable", "alloy", "Kore"]
)
def test_deferred_vertex_live_setup_carries_the_session_voice(owned: OwnedProxy, voice: str, expected: str) -> None:
    with owned.gateway.scenario() as scenario:
        handle: Final = live_scenario(scenario)
        key: Final = scenario.key()
        model: Final = live_deployment(scenario, handle.scenario_id, owned.gateway.upstream_url)
        frames: Final = live_turn(owned.gateway, model, key, voice)
        types: Final = frame_types(frames)
        assert types[0] == "session.created" and types[-1] == "response.done", frames
        setups: Final = live_setups(owned.gateway, handle.scenario_id)
        assert len(setups) == 1, setups
        assert voice_in(setups[0]) == expected, setups[0]


@dataclass(frozen=True, slots=True)
class Sent:
    voice: str
    text: str
    status: int
    body: bytes
    call_id: str


async def _fire(url: str, key: str, model: str, *, tolerate_transport_errors: bool = False) -> tuple[Sent, ...]:
    async def one(client: httpx.AsyncClient, index: int) -> Sent:
        voice: Final = VOICES[index % len(VOICES)]
        text: Final = marker()
        try:
            response: Final = await client.post(
                "/v1/audio/speech", json=_speech_body(model, text, voice), headers={"Authorization": f"Bearer {key}"}
            )
        except httpx.TransportError as error:
            if not tolerate_transport_errors:
                raise
            return Sent(voice, text, 0, repr(error).encode(), "")
        return Sent(voice, text, response.status_code, response.content, response.headers.get("x-litellm-call-id", ""))

    async with httpx.AsyncClient(base_url=url, timeout=60, trust_env=False) as client:
        return tuple(await asyncio.gather(*(one(client, index) for index in range(BURST))))


def _held_peer(release: threading.Event) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert release.wait(timeout=120), "Held peer was never released"
        return gemini_peer(request)

    return respond


def _held_connections(pid: int, wire_url: str) -> int:
    port: Final = urlsplit(wire_url).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


def _by_text(received: tuple[Request, ...]) -> dict[str, tuple[Request, ...]]:
    texts: Final = {received_text(request) for request in received}
    return {text: tuple(request for request in received if received_text(request) == text) for text in texts}


def _assert_served(items: tuple[Sent, ...], received: tuple[Request, ...]) -> None:
    by_text: Final = _by_text(received)
    for item in items:
        assert item.status == 200, (item.voice, item.body[:200])
        assert wav_payload(item.body) == PCM, item.text
        assert len(by_text.get(item.text, ())) == 1, item.text
        assert received_voice(by_text[item.text][0]) == MAPPED[item.voice], item.text
        _assert_speech_logged(item.call_id)


def _worker_pids(proxy: OwnedProxy) -> tuple[int, ...]:
    return eventually(
        lambda: tuple(int(pid) for pid in _STARTED_WORKER.findall(proxy.log.read_text())),
        lambda pids: len(pids) == 2,
        seconds=graceful_stop_seconds(),
    )


async def test_worker_sigkill_mid_speech_burst_leaves_the_sibling_serving(gateway: Gateway, tmp_path: Path) -> None:
    release: Final = threading.Event()
    with wire_server(_held_peer(release)) as wire:
        config: Final = health_config(tmp_path)
        with owned_proxy_process(gateway, tmp_path, _overrides(wire.url), config=config, workers=2) as proxy:
            workers: Final = _worker_pids(proxy)
            url: Final = str(proxy.gateway.client.base_url)
            with proxy.gateway.scenario() as scenario:
                model: Final = speech_deployment(scenario, wire.url)
                async with asyncio.TaskGroup() as tasks:
                    burst: Final = tasks.create_task(
                        _fire(url, proxy.gateway.key, model, tolerate_transport_errors=True)
                    )
                    try:
                        await asyncio.to_thread(
                            eventually, lambda: wire.received.qsize(), lambda size: size >= BURST, 90
                        )
                        held_by: Final = MappingProxyType({pid: _held_connections(pid, wire.url) for pid in workers})
                        victim: Final = max(workers, key=held_by.__getitem__)
                        os.kill(victim, signal.SIGKILL)
                    finally:
                        release.set()
                    served: Final = await burst
                during: Final = wire.drain()
                after: Final = await _fire(url, proxy.gateway.key, model)
                after_received: Final = wire.drain()
    assert sum(held_by.values()) == BURST, held_by
    assert held_by[victim] > 0, held_by
    completed: Final = tuple(item for item in served if item.status == 200)
    assert len(completed) == BURST - held_by[victim], (len(completed), held_by)
    assert all(len(requests) == 1 for requests in _by_text(during).values())
    _assert_served(completed, during)
    _assert_served(after, after_received)


async def test_proxy_restart_mid_speech_burst_serves_the_next_burst_after_the_reboot(
    gateway: Gateway, tmp_path: Path
) -> None:
    release: Final = threading.Event()
    with wire_server(_held_peer(release)) as wire, gateway.scenario() as scenario:
        config: Final = health_config(tmp_path)
        model: Final = speech_deployment(scenario, wire.url)
        with owned_proxy_process(gateway, tmp_path, _overrides(wire.url), config=config, workers=2) as first:
            url: Final = str(first.gateway.client.base_url)
            async with asyncio.TaskGroup() as tasks:
                burst: Final = tasks.create_task(_fire(url, first.gateway.key, model, tolerate_transport_errors=True))
                try:
                    await asyncio.to_thread(eventually, lambda: wire.received.qsize(), lambda size: size >= BURST, 90)
                    first.process.terminate()
                finally:
                    release.set()
                served: Final = await burst
            during: Final = wire.drain()
        with owned_proxy_process(gateway, tmp_path, _overrides(wire.url), config=config, workers=2) as second:
            after: Final = await _fire(str(second.gateway.client.base_url), second.gateway.key, model)
            after_received: Final = wire.drain()
    completed: Final = tuple(item for item in served if item.status == 200)
    assert all(len(requests) == 1 for requests in _by_text(during).values())
    _assert_served(completed, during)
    for item in served:
        if item.status != 200:
            assert item.status == 0 or b'"error"' in item.body or item.body == b"", (item.status, item.body[:200])
    _assert_served(after, after_received)
