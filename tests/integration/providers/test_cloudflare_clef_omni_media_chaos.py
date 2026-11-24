import asyncio
import json
import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final
from urllib.parse import unquote, urlsplit

import httpx
import psutil
import pytest
import yaml
from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from integration._support.ports import allocate_port
from pydantic import JsonValue, TypeAdapter

_API_KEY: Final = "synthetic-cloudflare-key"
_CANONICAL: Final = "cloudflare/@cf/cloudflare/clef-omni"
_CONFIG_MODEL: Final = "clef-omni-media-chaos"
_UPSTREAM_PATH: Final = "/ai/run/@cf/cloudflare/clef-omni"
_WAV: Final = "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA="
_MP4: Final = "data:video/mp4;base64,AAAAHGZ0eXBpc29tAAACAGlzb21pc28ybXA0MQ=="
_INPUT_TOKENS: Final = 537
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MARKER: Final = re.compile(r"marker-([0-9a-f]{32})")
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_QUESTIONS: Final[dict[str, JsonValue]] = {
    "pineapple": {"type": "noul", "instructions": "Does the audio say pineapple?"},
    "red": {"type": "noul", "instructions": "Is the video red?"},
}
_RED: Final = 0.97
_CHAOS_TIMEOUT: Final = int(2 * graceful_stop_seconds() + 120)


@dataclass(frozen=True, slots=True)
class _Served:
    marker: str
    call_id: str
    status: int
    text: str


def _marker_of(request: Request) -> str:
    found: Final = _MARKER.search(request.body.decode())
    assert found is not None, request.body[:200]
    return found.group(1)


def _state(marker: str) -> str:
    return f"Does the clip say pineapple? marker-{marker}"


def _probability(marker: str) -> float:
    return round(int(marker[:6], 16) / 0x1000000, 7)


def _echo(request: Request) -> Reply:
    marker: Final = _marker_of(request)
    reply: Final[dict[str, JsonValue]] = {
        "result": {
            "model": "clef-omni",
            "answers": {
                "pineapple": {"type": "noul", "noul": _probability(marker)},
                "red": {"type": "noul", "noul": _RED},
            },
            "usage": {"input_tokens": _INPUT_TOKENS, "output_tokens": 0},
        },
        "success": True,
        "errors": [],
        "messages": [],
    }
    return Reply(body=json.dumps(reply).encode())


def _body(model: str, marker: str) -> dict[str, JsonValue]:
    return {
        "model": model,
        "state": _state(marker),
        "questions": _QUESTIONS,
        "audio": [_WAV],
        "videos": [_MP4],
        "num_retries": 0,
    }


def _expected_upstream_body(marker: str) -> dict[str, JsonValue]:
    return {"model": "clef-omni", "state": _state(marker), "audio": [_WAV], "videos": [_MP4], "questions": _QUESTIONS}


def _upstream_body(request: Request) -> dict[str, JsonValue]:
    assert request.method == "POST", request.method
    assert unquote(request.target) == _UPSTREAM_PATH, request.target
    assert request.headers["authorization"] == f"Bearer {_API_KEY}", request.headers
    return _JSON_OBJECT.validate_json(request.body)


def _assert_each_landed_once_with_media(received: tuple[Request, ...], markers: frozenset[str]) -> None:
    bodies: Final = {_marker_of(request): _upstream_body(request) for request in received}
    assert len(bodies) == len(received) == len(markers), (len(bodies), len(received), len(markers))
    for marker in markers:
        assert bodies[marker] == _expected_upstream_body(marker), marker


def _assert_answered_with_its_own_probability(served: _Served) -> None:
    assert served.status == 200, served.text
    answers: Final = object_value(_JSON_OBJECT.validate_json(served.text)["answers"])
    assert object_value(answers["pineapple"])["noul"] == _probability(served.marker), served.text


async def _send(client: httpx.AsyncClient, key: str, model: str, marker: str) -> _Served:
    response: Final = await client.post(
        "/v1/systemone", json=_body(model, marker), headers={"Authorization": f"Bearer {key}"}
    )
    return _Served(
        marker=marker,
        call_id=response.headers.get("x-litellm-call-id", ""),
        status=response.status_code,
        text=response.text,
    )


async def _burst(
    base_url: str, key: str, model: str, markers: tuple[str, ...], *, tolerate_transport_errors: bool = False
) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(base_url=base_url, timeout=60, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(_send(client, key, model, marker) for marker in markers), return_exceptions=tolerate_transport_errors
        )
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, httpx.TransportError), repr(result)
    return tuple(result for result in results if isinstance(result, _Served))


def _markers(count: int) -> tuple[str, ...]:
    return tuple(uuid.uuid4().hex for _ in range(count))


def _spend_rows(model: str, expected: int) -> list[dict[str, JsonValue]]:
    rows: Final = eventually(
        lambda: read_rows('SELECT litellm_call_id, status FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
        lambda found: len(found) >= expected,
        seconds=70,
    )
    assert len({row["litellm_call_id"] for row in rows}) == len(rows) == expected, rows
    return rows


def _assert_each_billed_once(
    rows: list[dict[str, JsonValue]], failed: tuple[_Served, ...], served: tuple[_Served, ...]
) -> None:
    by_call: Final = {string_value(row["litellm_call_id"]): row for row in rows}
    assert len(by_call) == len(failed) + len(served), (len(by_call), len(failed), len(served))
    for item in failed:
        assert by_call[item.call_id]["status"] == "failure", (item.call_id, by_call[item.call_id])
    for item in served:
        assert by_call[item.call_id]["status"] == "success", (item.call_id, by_call[item.call_id])


def _assert_failed_cleanly(served: _Served) -> None:
    assert served.status >= 500, (served.status, served.text)
    assert "answers" not in served.text, served.text
    assert "error" in served.text.lower(), served.text
    assert served.call_id, served


@pytest.mark.timeout(_CHAOS_TIMEOUT)
async def test_an_upstream_dropping_connections_mid_burst_then_stopped_and_restarted_lands_each_success_once(
    gateway: Gateway,
) -> None:
    port: Final = allocate_port()
    first_wave: Final = _markers(12)
    dropped: Final = frozenset(marker for index, marker in enumerate(first_wave) if index % 2 == 1)
    while_down: Final = _markers(6)
    after_restart: Final = _markers(12)

    def flaky(request: Request) -> Reply:
        if _marker_of(request) in dropped:
            return Reply(drop_connection=True)
        return _echo(request)

    with gateway.scenario() as scenario:
        model: Final = scenario.model(model=_CANONICAL, api_base=f"http://127.0.0.1:{port}", api_key=_API_KEY)
        base_url: Final = str(gateway.client.base_url)
        with wire_server(flaky, port=port) as first:
            first_served: Final = await _burst(base_url, gateway.key, model, first_wave)
            assert len(first_served) == 12
            for item in first_served:
                if item.marker in dropped:
                    _assert_failed_cleanly(item)
                else:
                    _assert_answered_with_its_own_probability(item)
            _assert_each_landed_once_with_media(first.drain(), frozenset(first_wave))
        down: Final = await _burst(base_url, gateway.key, model, while_down)
        assert len(down) == 6
        for item in down:
            _assert_failed_cleanly(item)
        with wire_server(_echo, port=port) as second:
            recovered: Final = await _burst(base_url, gateway.key, model, after_restart)
            assert len(recovered) == 12
            for item in recovered:
                _assert_answered_with_its_own_probability(item)
            _assert_each_landed_once_with_media(second.drain(), frozenset(after_restart))
        failed: Final = (*(item for item in first_served if item.marker in dropped), *down)
        served: Final = (*(item for item in first_served if item.marker not in dropped), *recovered)
        _assert_each_billed_once(_spend_rows(model, 30), failed, served)


def _chaos_config(wire: Wire, tmp_path: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["model_list"] = [
        {
            "model_name": _CONFIG_MODEL,
            "litellm_params": {"model": _CANONICAL, "api_base": wire.url, "api_key": _API_KEY},
        }
    ]
    path: Final = tmp_path / "clef-omni-media-chaos.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


@pytest.mark.timeout(_CHAOS_TIMEOUT)
async def test_worker_sigkill_mid_burst_leaves_the_sibling_forwarding_media(gateway: Gateway, tmp_path: Path) -> None:
    markers: Final = _markers(20)
    release: Final = threading.Event()
    held_markers: Final[SimpleQueue[str]] = SimpleQueue()

    def held(request: Request) -> Reply:
        if request.method != "POST":
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
                    str(candidate.client.base_url),
                    candidate.key,
                    _CONFIG_MODEL,
                    markers,
                    tolerate_transport_errors=True,
                )
            )
            await asyncio.to_thread(eventually, held_markers.qsize, lambda size: size == 20, 60)
            held_by: Final = MappingProxyType({pid: _open_upstream_connections(pid, wire.url) for pid in workers})
            assert sum(held_by.values()) == 20, held_by
            victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
            with owned.replacing_worker(victim_pid) as kill:
                kill.freeze()
                await asyncio.to_thread(kill)
                release.set()
                served: Final = await burst
                assert held_by[survivor_pid] >= 10, held_by
                assert len(served) == held_by[survivor_pid], (held_by, len(served))
                for item in served:
                    _assert_answered_with_its_own_probability(item)
            follow_up: Final = uuid.uuid4().hex
            (answered,) = await _burst(str(candidate.client.base_url), candidate.key, _CONFIG_MODEL, (follow_up,))
            _assert_answered_with_its_own_probability(answered)
            received: Final = wire.drain()
            posts: Final = tuple(request for request in received if request.method == "POST")
            assert len(posts) == len(received), [(request.method, request.target) for request in received]
            _assert_each_landed_once_with_media(posts, frozenset((*markers, follow_up)))
