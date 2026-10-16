import asyncio
import base64
import json
import os
import re
import threading
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, Literal
from urllib.parse import unquote, urlsplit, urlunsplit

import httpx
import psutil
import pytest
import yaml
from integration._support.bedrock_runtime_peer import PNG_1X1
from integration._support.client import Gateway, eventually, list_value, object_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

Route = Literal["/v1/systemone", "/systemone", "/v1/decisions", "/decisions"]

_ROUTES: Final[tuple[Route, ...]] = ("/v1/systemone", "/systemone", "/v1/decisions", "/decisions")
_API_KEY: Final = "synthetic-cloudflare-key"
_INPUT_TOKENS: Final = 412
_PNG_DATA_URL: Final = f"data:image/png;base64,{base64.b64encode(PNG_1X1).decode()}"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MARKER: Final = re.compile(r"marker-([0-9a-f]{32})")
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_SYSTEM_ONE_QUESTIONS: Final[dict[str, JsonValue]] = {"red": {"type": "noul", "instructions": "Is the screenshot red?"}}
_OPENAI_QUESTIONS: Final[list[JsonValue]] = [
    {"type": "predicate", "name": "red", "instructions": "Is the screenshot red?"}
]


@dataclass(frozen=True, slots=True)
class _Call:
    route: Route
    marker: str
    images: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Served:
    call: _Call
    status: int
    text: str


def _state(marker: str) -> str:
    return f"Is the checkout screenshot red? marker-{marker}"


def _probability(marker: str) -> float:
    return round(int(marker[:6], 16) / 0x1000000, 7)


def _marker_of(request: Request) -> str:
    found: Final = _MARKER.search(request.body.decode() or request.target)
    assert found is not None, (request.target, request.body[:200])
    return found.group(1)


def _clef_reply(request: Request) -> Reply:
    reply: Final[dict[str, JsonValue]] = {
        "result": {
            "model": "clef",
            "answers": {"red": {"type": "noul", "noul": _probability(_marker_of(request))}},
            "usage": {"input_tokens": _INPUT_TOKENS, "output_tokens": 0},
        },
        "success": True,
    }
    return Reply(body=json.dumps(reply).encode())


def _is_image_fetch(request: Request) -> bool:
    return request.method == "GET" and urlsplit(request.target).path == "/image.png"


def _body(call: _Call, model: str) -> dict[str, JsonValue]:
    if call.route.endswith("systemone"):
        return {
            "model": model,
            "state": _state(call.marker),
            "questions": _SYSTEM_ONE_QUESTIONS,
            "images": list(call.images),
            "num_retries": 0,
        }
    parts: Final[list[JsonValue]] = [
        {"type": "input_text", "text": _state(call.marker)},
        *({"type": "input_image", "image_url": image} for image in call.images),
    ]
    return {
        "model": model,
        "input": [{"role": "user", "content": parts}],
        "questions": _OPENAI_QUESTIONS,
        "num_retries": 0,
    }


def _upstream_by_marker(received: Sequence[Request]) -> dict[str, dict[str, JsonValue]]:
    posts: Final = tuple(request for request in received if request.method == "POST")
    for request in posts:
        assert unquote(request.target) == "/ai/run/@cf/cloudflare/clef", request.target
        assert request.headers["authorization"] == f"Bearer {_API_KEY}", request.headers
    by_marker: Final = {_marker_of(request): _JSON_OBJECT.validate_json(request.body) for request in posts}
    assert len(by_marker) == len(posts), "a request reached clef twice"
    return by_marker


def _answered_probability(route: Route, answers: JsonValue) -> JsonValue:
    if route.endswith("systemone"):
        return object_value(object_value(answers)["red"])["noul"]
    (answer,) = list_value(answers)
    return object_value(answer)["probability"]


def _assert_answered(served: _Served) -> None:
    assert served.status == 200, served.text
    answers: Final = _JSON_OBJECT.validate_json(served.text)["answers"]
    assert _answered_probability(served.call.route, answers) == _probability(served.call.marker), served.text


def _assert_clef_got_inlined_images(call: _Call, body: dict[str, JsonValue]) -> None:
    assert body == {
        "model": "clef",
        "state": _state(call.marker),
        "questions": _SYSTEM_ONE_QUESTIONS,
        "images": [_PNG_DATA_URL] * len(call.images),
    }, (call, body)


async def _send(client: httpx.AsyncClient, key: str, model: str, call: _Call) -> _Served:
    response: Final = await client.post(call.route, json=_body(call, model), headers={"Authorization": f"Bearer {key}"})
    return _Served(call=call, status=response.status_code, text=response.text)


async def _burst(
    gateway: Gateway, model: str, calls: Sequence[_Call], *, tolerate_transport_errors: bool = False
) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(base_url=str(gateway.client.base_url), timeout=90, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(_send(client, gateway.key, model, call) for call in calls), return_exceptions=tolerate_transport_errors
        )
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, httpx.TransportError), repr(result)
    return tuple(result for result in results if isinstance(result, _Served))


def _pooled_database_url(url: str) -> str:
    parts: Final = urlsplit(url)
    query: Final = "&".join(part for part in (parts.query, "connection_limit=5") if part)
    return urlunsplit(parts._replace(query=query))


def _config(wire: Wire, tmp_path: Path) -> tuple[Path, str]:
    model: Final = f"clef-images-chaos-{uuid.uuid4().hex[:8]}"
    config: Final = _JSON_OBJECT.validate_python(
        yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    )
    path: Final = tmp_path / "clef-images-chaos.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **config,
                "model_list": [
                    {
                        "model_name": model,
                        "litellm_params": {"model": "cloudflare/clef", "api_base": wire.url, "api_key": _API_KEY},
                    }
                ],
                "general_settings": {
                    **object_value(config["general_settings"]),
                    "user_url_allowed_hosts": ["127.0.0.1"],
                },
            }
        )
    )
    return path, model


def _successful_spend_rows(model: str, expected: int) -> int:
    rows: Final = eventually(
        lambda: read_rows('SELECT request_id, status FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
        lambda found: sum(row["status"] == "success" for row in found) >= expected,
        seconds=70,
    )
    assert len({row["request_id"] for row in rows}) == len(rows), rows
    return sum(row["status"] == "success" for row in rows)


def _open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


def _mixed_calls(wire: Wire, count: int) -> tuple[_Call, ...]:
    def images(index: int, marker: str) -> tuple[str, ...]:
        remote: Final = f"{wire.url}/image.png?id=marker-{marker}"
        return ((_PNG_DATA_URL,), (remote,), (remote, _PNG_DATA_URL, remote, _PNG_DATA_URL))[index % 3]

    markers: Final = tuple(uuid.uuid4().hex for _ in range(count))
    return tuple(
        _Call(route=_ROUTES[index % len(_ROUTES)], marker=marker, images=images(index, marker))
        for index, marker in enumerate(markers)
    )


def _overflow_calls(wire: Wire, count: int) -> tuple[_Call, ...]:
    markers: Final = tuple(uuid.uuid4().hex for _ in range(count))
    return tuple(
        _Call(
            route=_ROUTES[index % len(_ROUTES)], marker=marker, images=(f"{wire.url}/image.png?id=marker-{marker}",) * 5
        )
        for index, marker in enumerate(markers)
    )


def _image_host_and_clef(request: Request) -> Reply:
    if _is_image_fetch(request):
        return Reply(body=PNG_1X1, content_type="image/png")
    return _clef_reply(request)


@pytest.mark.timeout(600)
async def test_a_mixed_image_burst_across_two_workers_reaches_clef_once_per_request(
    gateway: Gateway, tmp_path: Path
) -> None:
    with wire_server(_image_host_and_clef) as wire:
        path, model = _config(wire, tmp_path)
        calls: Final = _mixed_calls(wire, 24)
        overflow: Final = _overflow_calls(wire, 8)
        overrides: Final = {"DATABASE_URL": _pooled_database_url(os.environ["DATABASE_URL"])}
        with owned_proxy_process(gateway, tmp_path, overrides, config=path, workers=2) as owned:
            served: Final = await _burst(owned.gateway, model, (*calls, *overflow))
            assert len(served) == 32
            overflow_markers: Final = frozenset(call.marker for call in overflow)
            for item in served:
                if item.call.marker in overflow_markers:
                    assert item.status == 400, item.text
                    assert "at most 4 images per request, got 5" in item.text, item.text
                else:
                    _assert_answered(item)
            received: Final = wire.drain()
            fetched: Final = sorted(_marker_of(request) for request in received if _is_image_fetch(request))
            assert overflow_markers.isdisjoint(fetched), fetched
            by_marker: Final = _upstream_by_marker(received)
            assert sorted(by_marker) == sorted(call.marker for call in calls)
            for call in calls:
                _assert_clef_got_inlined_images(call, by_marker[call.marker])
            assert _successful_spend_rows(model, 24) == 24


@pytest.mark.timeout(600)
async def test_worker_sigkill_while_images_are_fetched_leaves_the_sibling_serving(
    gateway: Gateway, tmp_path: Path
) -> None:
    release: Final = threading.Event()
    held_fetches: Final[SimpleQueue[str]] = SimpleQueue()

    def respond(request: Request) -> Reply:
        if not _is_image_fetch(request):
            return _clef_reply(request)
        held_fetches.put(_marker_of(request))
        assert release.wait(timeout=60), "The image fetches were never released"
        return Reply(body=PNG_1X1, content_type="image/png")

    with wire_server(respond) as wire:
        path, model = _config(wire, tmp_path)
        calls: Final = tuple(
            _Call(route="/v1/systemone", marker=marker, images=(f"{wire.url}/image.png?id=marker-{marker}",))
            for marker in (uuid.uuid4().hex for _ in range(16))
        )
        overrides: Final = {"DATABASE_URL": _pooled_database_url(os.environ["DATABASE_URL"])}
        with owned_proxy_process(gateway, tmp_path, overrides, config=path, workers=2) as owned:
            workers: Final = eventually(
                lambda: tuple(int(pid) for pid in _STARTED_WORKER.findall(owned.log.read_text())),
                lambda pids: len(pids) == 2,
                seconds=30,
            )
            burst: Final = asyncio.create_task(_burst(owned.gateway, model, calls, tolerate_transport_errors=True))
            await asyncio.to_thread(eventually, held_fetches.qsize, lambda size: size == 16, 60)
            held_by: Final = MappingProxyType({pid: _open_upstream_connections(pid, wire.url) for pid in workers})
            assert sum(held_by.values()) == 16, held_by
            victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
            with owned.replacing_worker(victim_pid) as kill:
                kill.freeze()
                await asyncio.to_thread(kill)
                release.set()
                served: Final = await burst
                assert held_by[survivor_pid] >= 8, held_by
                assert len(served) == held_by[survivor_pid], (held_by, len(served))
                for item in served:
                    _assert_answered(item)
            follow_up: Final = _Call(route="/v1/decisions", marker=uuid.uuid4().hex, images=(_PNG_DATA_URL,))
            (answered,) = await _burst(owned.gateway, model, (follow_up,))
            _assert_answered(answered)
            by_marker: Final = _upstream_by_marker(wire.drain())
            assert sorted(by_marker) == sorted((*(item.call.marker for item in served), follow_up.marker))
            for item in served:
                _assert_clef_got_inlined_images(item.call, by_marker[item.call.marker])
            _assert_clef_got_inlined_images(follow_up, by_marker[follow_up.marker])
