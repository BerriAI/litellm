import asyncio
import json
import re
import signal
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final
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

_BACKEND: Final = "pplx-decider-v1-27b"
_CONFIG_MODEL: Final = "decisions-chaos"
_API_KEY: Final = "synthetic-decisions-key"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_QUESTIONS: Final[list[JsonValue]] = [{"type": "predicate", "name": "fine", "instructions": "Is the state fine?"}]
_ROUTES: Final = ("/v1/decisions", "/decisions")


@dataclass(frozen=True, slots=True)
class _Call:
    route: str
    marker: str
    fail: bool


@dataclass(frozen=True, slots=True)
class _Served:
    call: _Call
    status: int
    call_id: str
    model_group: str
    text: str


def _calls(count: int, *, fail: bool) -> tuple[_Call, ...]:
    return tuple(
        _Call(
            route=_ROUTES[index % len(_ROUTES)],
            marker=f"{'fail' if fail and index % 2 else 'ok'}-{uuid.uuid4().hex}",
            fail=fail and index % 2 == 1,
        )
        for index in range(count)
    )


def _marker_of(request: Request) -> str:
    state: Final = _JSON_OBJECT.validate_json(request.body)["state"]
    assert isinstance(state, str), request.body
    return state


def _asked_questions(request: Request) -> tuple[str, ...]:
    questions: Final = _JSON_OBJECT.validate_json(request.body)["questions"]
    assert isinstance(questions, dict), request.body
    return tuple(questions)


def _reply(request: Request) -> Reply:
    marker: Final = _marker_of(request)
    if marker.startswith("fail-"):
        return Reply(status=500, body=json.dumps({"error": {"message": f"scripted outage {marker}"}}).encode())
    answer: Final = {
        "model": f"model-{marker}",
        "answers": {name: {"type": "noul", "noul": 0.5} for name in _asked_questions(request)},
        "usage": {"input_tokens": 12, "output_tokens": 1},
    }
    return Reply(body=json.dumps(answer).encode())


async def _send(client: httpx.AsyncClient, key: str, model: str | None, call: _Call) -> _Served:
    body: Final[dict[str, JsonValue]] = {
        **({"model": model} if model is not None else {}),
        "input": call.marker,
        "questions": _QUESTIONS,
        "num_retries": 0,
    }
    response: Final = await client.post(call.route, json=body, headers={"Authorization": f"Bearer {key}"})
    return _Served(
        call=call,
        status=response.status_code,
        call_id=response.headers.get("x-litellm-call-id", ""),
        model_group=response.headers.get("x-litellm-model-group", ""),
        text=response.text,
    )


async def _burst(
    base_url: str, key: str, model: str | None, calls: tuple[_Call, ...], *, tolerate_transport_errors: bool = False
) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(base_url=base_url, timeout=60, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(_send(client, key, model, call) for call in calls), return_exceptions=tolerate_transport_errors
        )
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, httpx.TransportError), repr(result)
    return tuple(result for result in results if isinstance(result, _Served))


def _assert_served_its_own(served: _Served) -> None:
    assert served.call_id, served.text
    if served.call.fail:
        assert served.status == 500, (served.status, served.text)
        assert served.call.marker in served.text, served.text
        return
    assert served.status == 200, (served.status, served.text)
    assert _JSON_OBJECT.validate_json(served.text)["model"] == f"model-{served.call.marker}", served.text


def _statuses_by_call_id(call_ids: tuple[str, ...]) -> dict[str, JsonValue]:
    placeholders: Final = ", ".join("%s" for _ in call_ids)
    rows: Final = eventually(
        lambda: read_rows(
            f'SELECT request_id, status FROM "LiteLLM_SpendLogs" WHERE request_id IN ({placeholders})', call_ids
        ),
        lambda found: len(found) >= len(call_ids),
        seconds=70,
    )
    assert len(rows) == len(call_ids), rows
    return {str(row["request_id"]): row["status"] for row in rows}


def _expected_statuses(served: tuple[_Served, ...]) -> dict[str, JsonValue]:
    return {item.call_id: "failure" if item.call.fail else "success" for item in served}


def _health(gateway: Gateway, model: str) -> tuple[int, int]:
    health: Final = gateway.request("GET", "/health", params={"model": model})
    assert health.status_code in (200, 503), health.text
    report: Final = health.json()
    return (report["healthy_count"], report["unhealthy_count"])


def _chaos_config(wire: Wire, tmp_path: Path) -> Path:
    config: Final = {
        **_JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())),
        "model_list": [
            {
                "model_name": _CONFIG_MODEL,
                "litellm_params": {"model": f"perplexity/{_BACKEND}", "api_base": wire.url, "api_key": _API_KEY},
            }
        ],
    }
    path: Final = tmp_path / "decisions-chaos.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _worker_startups(log: Path) -> tuple[tuple[int, ...], int]:
    text: Final = log.read_text()
    return tuple(int(pid) for pid in _STARTED_WORKER.findall(text)), text.count("Application startup complete.")


def _open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


@pytest.mark.timeout(180)
async def test_burst_over_both_routes_bills_each_call_once_with_its_own_status(gateway: Gateway) -> None:
    calls: Final = _calls(30, fail=True)
    with wire_server(_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"perplexity/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
        served: Final = await _burst(str(gateway.client.base_url), gateway.key, model, calls)
        assert len(served) == 30
        for item in served:
            _assert_served_its_own(item)
        assert len({item.call_id for item in served}) == 30
        assert _statuses_by_call_id(tuple(item.call_id for item in served)) == _expected_statuses(served)
        received: Final = wire.drain()
        assert sorted(_marker_of(request) for request in received) == sorted(call.marker for call in calls)
        assert {request.target for request in received} == {"/v1/decisions"}, received


@pytest.mark.timeout(420)
async def test_worker_sigkill_mid_burst_leaves_the_sibling_serving_the_default_model(
    gateway: Gateway, tmp_path: Path
) -> None:
    calls: Final = _calls(20, fail=False)
    release: Final = threading.Event()
    held_markers: Final[SimpleQueue[str]] = SimpleQueue()

    def held(request: Request) -> Reply:
        held_markers.put(_marker_of(request))
        assert release.wait(timeout=60), "The burst was never released"
        return _reply(request)

    with wire_server(held) as wire:
        path: Final = _chaos_config(wire, tmp_path)
        with owned_proxy_process(
            gateway, tmp_path, {}, config=path, workers=2, extra_arguments=("--model", _CONFIG_MODEL)
        ) as owned:
            candidate: Final = owned.gateway
            base_url: Final = str(candidate.client.base_url)
            workers, _ = eventually(
                lambda: _worker_startups(owned.log),
                lambda found: len(found[0]) == 2 and found[1] == 2,
                seconds=120,
            )
            burst: Final = asyncio.create_task(
                _burst(base_url, candidate.key, None, calls, tolerate_transport_errors=True)
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
            assert len(served) == held_by[survivor_pid], (held_by, len(served))
            follow_up: Final = _Call(route="/decisions", marker=f"ok-{uuid.uuid4().hex}", fail=False)
            (answered,) = await _burst(base_url, candidate.key, None, (follow_up,))
            await asyncio.to_thread(
                eventually,
                lambda: _worker_startups(owned.log),
                lambda found: len(found[0]) == 3 and found[1] == 3,
                180,
            )
            for item in (*served, answered):
                _assert_served_its_own(item)
                assert item.model_group == _CONFIG_MODEL, item.model_group
            call_ids: Final = tuple(item.call_id for item in (*served, answered))
            assert set(_statuses_by_call_id(call_ids).values()) == {"success"}
            assert len({_marker_of(request) for request in wire.drain()}) == 21


@pytest.mark.timeout(180)
async def test_upstream_outage_fails_its_calls_and_recovery_on_the_same_port_restores_them(gateway: Gateway) -> None:
    base_url: Final = str(gateway.client.base_url)
    with gateway.scenario() as scenario:
        with wire_server(_reply) as wire:
            port: Final = urlsplit(wire.url).port
            assert port is not None
            model: Final = scenario.model(model=f"perplexity/{_BACKEND}", api_base=wire.url, api_key=_API_KEY)
            before: Final = await _burst(base_url, gateway.key, model, _calls(5, fail=False))
            assert _health(gateway, model) == (1, 0)
        during: Final = await _burst(base_url, gateway.key, model, _calls(5, fail=False))
        assert [item.status for item in during] == [500] * 5, [item.text for item in during]
        assert _health(gateway, model) == (0, 1)
        with wire_server(_reply, port=port):
            after: Final = await _burst(base_url, gateway.key, model, _calls(5, fail=False))
            assert _health(gateway, model) == (1, 0)
        for item in (*before, *after):
            _assert_served_its_own(item)
        statuses: Final = _statuses_by_call_id(tuple(item.call_id for item in (*before, *during, *after)))
        assert statuses == {
            **{item.call_id: "success" for item in (*before, *after)},
            **{item.call_id: "failure" for item in during},
        }
