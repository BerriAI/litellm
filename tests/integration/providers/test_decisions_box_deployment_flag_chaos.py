import asyncio
import json
import threading
import uuid
from collections.abc import Mapping
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
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_API_KEY: Final = "synthetic-decisions-key"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_BOX: Final[dict[str, JsonValue]] = {"top": 0.1, "bottom": 0.4, "left": 0.2, "right": 0.6}
_BOX_QUESTION: Final[dict[str, JsonValue]] = {"type": "box", "name": "crack", "instructions": "Where is the crack?"}
_SYSTEM_ONE_BOX_QUESTION: Final[dict[str, JsonValue]] = {"type": "box", "instructions": "Where is the crack?"}
_SYSTEM_ONE_BOX_ANSWER: Final[dict[str, JsonValue]] = {"type": "box", "box": _BOX, "confidence": 0.7}
_BOX_ANSWER: Final[dict[str, JsonValue]] = {"type": "box", "name": "crack", "box": _BOX, "confidence": 0.7}
_REFUSED_BOX: Final = "does not support parameters: ['box questions']"
_OPENAI_FORMAT_ROUTES: Final = ("/v1/decisions", "/decisions")
_ROUTES: Final = (*_OPENAI_FORMAT_ROUTES, "/v1/systemone", "/systemone")
_FLAGS: Final = ("true", "false", "unset")


@dataclass(frozen=True, slots=True)
class _Call:
    route: str
    model: str
    marker: str
    answers: bool


@dataclass(frozen=True, slots=True)
class _Served:
    call: _Call
    status: int
    call_id: str
    text: str


def _marker_of(request: Request) -> str:
    state: Final = _JSON_OBJECT.validate_json(request.body)["state"]
    assert isinstance(state, str), request.body
    return state


def _reply(request: Request) -> Reply:
    marker: Final = _marker_of(request)
    answer: Final = {
        "model": f"model-{marker}",
        "answers": {"crack": _SYSTEM_ONE_BOX_ANSWER},
        "usage": {"input_tokens": 12, "output_tokens": 1},
    }
    return Reply(body=json.dumps(answer).encode())


def _body(call: _Call) -> dict[str, JsonValue]:
    if call.route in _OPENAI_FORMAT_ROUTES:
        return {"model": call.model, "input": call.marker, "questions": [_BOX_QUESTION], "num_retries": 0}
    return {
        "model": call.model,
        "state": call.marker,
        "questions": {"crack": _SYSTEM_ONE_BOX_QUESTION},
        "num_retries": 0,
    }


async def _send(client: httpx.AsyncClient, key: str, call: _Call) -> _Served:
    response: Final = await client.post(call.route, json=_body(call), headers={"Authorization": f"Bearer {key}"})
    return _Served(call, response.status_code, response.headers.get("x-litellm-call-id", ""), response.text)


async def _burst(
    base_url: str, key: str, calls: tuple[_Call, ...], *, tolerate_transport_errors: bool = False
) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(base_url=base_url, timeout=60, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(_send(client, key, call) for call in calls), return_exceptions=tolerate_transport_errors
        )
    assert all(not isinstance(result, BaseException) or isinstance(result, httpx.TransportError) for result in results)
    return tuple(result for result in results if isinstance(result, _Served))


def _assert_served_its_own(served: _Served) -> None:
    if not served.call.answers:
        assert served.status == 400, (served.status, served.text)
        assert _REFUSED_BOX in served.text, served.text
        return
    assert served.status == 200, (served.status, served.text)
    body: Final = _JSON_OBJECT.validate_json(served.text)
    assert body["model"] == f"model-{served.call.marker}", served.text
    expected: Final[JsonValue] = (
        [_BOX_ANSWER] if served.call.route in _OPENAI_FORMAT_ROUTES else {"crack": _SYSTEM_ONE_BOX_ANSWER}
    )
    assert body["answers"] == expected, served.text


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


def _answered(served: tuple[_Served, ...]) -> tuple[_Served, ...]:
    return tuple(item for item in served if item.call.answers)


def _assert_each_served_its_own_and_logged_once(served: tuple[_Served, ...]) -> None:
    for item in served:
        _assert_served_its_own(item)
    answered: Final = _answered(served)
    assert len({item.call_id for item in answered}) == len(answered), answered
    assert set(_statuses_by_call_id(tuple(item.call_id for item in answered)).values()) == {"success"}


def _backend() -> str:
    return f"perplexity/box-{uuid.uuid4().hex[:12]}"


def _calls(models: Mapping[str, str], count: int) -> tuple[_Call, ...]:
    return tuple(
        _Call(
            route=_ROUTES[index % len(_ROUTES)],
            model=models[_FLAGS[(index // len(_ROUTES)) % len(_FLAGS)]],
            marker=f"{_FLAGS[(index // len(_ROUTES)) % len(_FLAGS)]}-{uuid.uuid4().hex}",
            answers=_FLAGS[(index // len(_ROUTES)) % len(_FLAGS)] == "true",
        )
        for index in range(count)
    )


@pytest.mark.timeout(180)
async def test_a_burst_over_every_route_and_flag_sends_boxes_only_for_flagged_deployments_and_logs_each_once(
    gateway: Gateway,
) -> None:
    with wire_server(_reply) as wire, gateway.scenario() as scenario:
        flags: Final[Mapping[str, Mapping[str, JsonValue] | None]] = MappingProxyType(
            {"true": {"supports_box_questions": True}, "false": {"supports_box_questions": False}, "unset": None}
        )
        models: Final = MappingProxyType(
            {
                flag: scenario.model(model=_backend(), api_base=wire.url, api_key=_API_KEY, model_info=model_info)
                for flag, model_info in flags.items()
            }
        )
        calls: Final = _calls(models, 36)
        served: Final = await _burst(str(gateway.client.base_url), gateway.key, calls)
        assert len(served) == 36
        for item in served:
            _assert_served_its_own(item)
        answered: Final = _answered(served)
        assert len(answered) == 12
        assert len({item.call_id for item in answered}) == 12
        assert set(_statuses_by_call_id(tuple(item.call_id for item in answered)).values()) == {"success"}
        received: Final = wire.drain()
        assert sorted(_marker_of(request) for request in received) == sorted(item.call.marker for item in answered)
        assert {request.target for request in received} == {"/v1/decisions"}, received


@pytest.mark.timeout(180)
async def test_a_flagged_deployment_recovers_box_answers_after_an_upstream_outage_on_the_same_port(
    gateway: Gateway,
) -> None:
    base_url: Final = str(gateway.client.base_url)
    with gateway.scenario() as scenario:
        with wire_server(_reply) as wire:
            port: Final = urlsplit(wire.url).port
            assert port is not None
            model: Final = scenario.model(
                model=_backend(), api_base=wire.url, api_key=_API_KEY, model_info={"supports_box_questions": True}
            )
            before: Final = await _burst(base_url, gateway.key, _calls(MappingProxyType({"true": model}), 4))
        during: Final = await _burst(base_url, gateway.key, _calls(MappingProxyType({"true": model}), 4))
        assert [item.status for item in during] == [500] * 4, [item.text for item in during]
        with wire_server(_reply, port=port) as recovered:
            after: Final = await _burst(base_url, gateway.key, _calls(MappingProxyType({"true": model}), 4))
            assert sorted(map(_marker_of, recovered.drain())) == sorted(item.call.marker for item in after)
        for item in (*before, *after):
            _assert_served_its_own(item)
        statuses: Final = _statuses_by_call_id(tuple(item.call_id for item in (*before, *during, *after)))
        assert statuses == {
            **{item.call_id: "success" for item in (*before, *after)},
            **{item.call_id: "failure" for item in during},
        }


_YAML_FLAGS: Final[tuple[tuple[str, str, Mapping[str, JsonValue] | None, bool], ...]] = (
    ("box-yaml-true", "own", {"supports_box_questions": True}, True),
    ("box-yaml-false-on-true-row", "shared", {"supports_box_questions": False}, False),
    ("box-yaml-true-row", "shared", {"supports_box_questions": True, "mode": "evaluation"}, True),
    ("box-yaml-unset-on-true-row", "shared", None, True),
    ("box-yaml-string", "string", {"supports_box_questions": "true"}, False),
)


def _yaml_config(wire: Wire, tmp_path: Path, backends: Mapping[str, str]) -> Path:
    base: Final = _JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    model_list: Final = [
        {
            "model_name": name,
            "litellm_params": {"model": backends[backend], "api_base": wire.url, "api_key": _API_KEY},
            **({} if model_info is None else {"model_info": dict(model_info)}),
        }
        for name, backend, model_info, _ in _YAML_FLAGS
    ]
    path: Final = tmp_path / "decisions-box-flags.yaml"
    path.write_text(yaml.safe_dump({**base, "model_list": model_list}))
    return path


def _yaml_calls(per_model: int) -> tuple[_Call, ...]:
    return tuple(
        _Call(
            route=_ROUTES[index % len(_ROUTES)],
            model=_YAML_FLAGS[index % len(_YAML_FLAGS)][0],
            marker=f"{_YAML_FLAGS[index % len(_YAML_FLAGS)][0]}-{uuid.uuid4().hex}",
            answers=_YAML_FLAGS[index % len(_YAML_FLAGS)][3],
        )
        for index in range(per_model * len(_YAML_FLAGS))
    )


def _open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
async def test_yaml_box_flags_survive_a_worker_sigkill_mid_burst_and_hold_on_respawned_workers(
    gateway: Gateway, tmp_path: Path
) -> None:
    release: Final = threading.Event()
    held_markers: Final[SimpleQueue[str]] = SimpleQueue()
    backends: Final = MappingProxyType({"own": _backend(), "shared": _backend(), "string": _backend()})

    def held(request: Request) -> Reply:
        marker: Final = _marker_of(request)
        if marker.startswith("held-"):
            held_markers.put(marker)
            assert release.wait(timeout=60), "The burst was never released"
        return _reply(request)

    with wire_server(held) as wire:
        config: Final = _yaml_config(wire, tmp_path, backends)
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            base_url: Final = str(owned.gateway.client.base_url)
            key: Final = owned.gateway.key
            first: Final = await _burst(base_url, key, _yaml_calls(2))
            _assert_each_served_its_own_and_logged_once(first)
            workers: Final = owned.worker_pids()
            assert len(workers) == 2, workers
            calls: Final = tuple(
                _Call(
                    route=_ROUTES[index % len(_ROUTES)],
                    model="box-yaml-true",
                    marker=f"held-{uuid.uuid4().hex}",
                    answers=True,
                )
                for index in range(20)
            )
            burst: Final = asyncio.create_task(_burst(base_url, key, calls, tolerate_transport_errors=True))
            await asyncio.to_thread(eventually, held_markers.qsize, lambda size: size == 20, 60)
            held_by: Final = MappingProxyType({pid: _open_upstream_connections(pid, wire.url) for pid in workers})
            assert sum(held_by.values()) == 20, held_by
            victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
            with owned.replacing_worker(victim_pid) as kill:
                kill.freeze()
                await asyncio.to_thread(kill)
                release.set()
                survived: Final = await burst
                assert len(survived) == held_by[survivor_pid], (held_by, len(survived))
            await asyncio.to_thread(_assert_each_served_its_own_and_logged_once, survived)
            await asyncio.to_thread(owned.kill_worker, survivor_pid)
            assert not workers & owned.worker_pids(), (workers, owned.worker_pids())
            respawned: Final = await _burst(base_url, key, _yaml_calls(4))
            await asyncio.to_thread(_assert_each_served_its_own_and_logged_once, respawned)
        received: Final = tuple(map(_marker_of, wire.drain()))
        assert sorted(received) == sorted(
            (*(call.marker for call in calls), *(item.call.marker for item in _answered((*first, *respawned))))
        )
