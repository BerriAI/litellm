import asyncio
import base64
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
from typing import Final, Literal
from urllib.parse import urlsplit

import httpx
import psutil
import pytest
import yaml
from integration._support.anthropic_sse import (
    Attempts,
    delta_text,
    dropping_reply,
    event_type,
    event_types,
    message_id,
    message_json,
    message_stream,
    parse_sse,
    status_reply,
    stream_reply,
    user_prompt,
)
from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.openai_wire import chat_reply, openai_error, responses_reply
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

_ANTHROPIC_BACKEND: Final = "claude-under-test"
_ANTHROPIC_KEY: Final = "synthetic-anthropic-key"
_OPENAI_BACKEND: Final = "gpt-4o-mini"
_OPENAI_KEY: Final = "integration-provider-key"
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_OUTAGE_STATUSES: Final = (529, 429, 503)
_ROUTING_ENCODED_ID: Final = re.compile(r"resp_([A-Za-z0-9+/]+=*)")

pytestmark = pytest.mark.timeout(240)

Endpoint = Literal["messages", "chat", "responses"]


@dataclass(frozen=True, slots=True)
class _Models:
    messages: str
    openai: str

    def for_endpoint(self, endpoint: Endpoint) -> str:
        return self.messages if endpoint == "messages" else self.openai


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


def _answer(marker: str) -> str:
    return f"answer-{marker}"


def _path(endpoint: Endpoint) -> str:
    match endpoint:
        case "messages":
            return "/v1/messages"
        case "chat":
            return "/v1/chat/completions"
        case "responses":
            return "/v1/responses"


def _body(models: _Models, call: _Call) -> dict[str, JsonValue]:
    prompt: Final = f"chaos:{call.marker}"
    model: Final = models.for_endpoint(call.endpoint)
    match call.endpoint:
        case "messages":
            return {
                "model": model,
                "max_tokens": 16,
                "stream": call.stream,
                "messages": [{"role": "user", "content": prompt}],
            }
        case "chat":
            return {"model": model, "stream": call.stream, "messages": [{"role": "user", "content": prompt}]}
        case "responses":
            return {"model": model, "stream": call.stream, "input": prompt}


def _marker_of(request: Request) -> str:
    body: Final = object_value(json.loads(request.body))
    prompt: Final = string_value(body["input"]) if "input" in body else user_prompt(body)
    return prompt.removeprefix("chaos:")


def _streaming(request: Request) -> bool:
    return object_value(json.loads(request.body)).get("stream") is True


def _served(request: Request, marker: str, attempt: int) -> Reply:
    text: Final = _answer(marker)
    match request.target:
        case "/v1/messages":
            served_id: Final = f"msg_{marker}_a{attempt}"
            if _streaming(request):
                return stream_reply(message_stream(served_id, _ANTHROPIC_BACKEND, text))
            return Reply(body=message_json(served_id, _ANTHROPIC_BACKEND, text))
        case "/v1/chat/completions":
            return chat_reply(f"chatcmpl-{marker}-a{attempt}", _OPENAI_BACKEND, text, stream=_streaming(request))
        case "/v1/responses":
            return responses_reply(f"resp_{marker}_a{attempt}", _OPENAI_BACKEND, text, stream=_streaming(request))
    raise AssertionError(request.target)


def _drop_or_500(request: Request, marker: str) -> Reply:
    if request.target == "/v1/messages":
        return dropping_reply(message_stream(f"msg_{marker}_a1", _ANTHROPIC_BACKEND, _answer(marker)), abort_after=1)
    return openai_error(500)


def _outage(statuses: Mapping[str, int]) -> Callable[[Request, str], Reply]:
    def first_attempt(request: Request, marker: str) -> Reply:
        if request.target == "/v1/messages":
            return status_reply(statuses[marker])
        return openai_error(statuses[marker])

    return first_attempt


@dataclass(frozen=True, slots=True)
class _Upstream:
    attempts: Attempts
    first_attempt: Callable[[Request, str], Reply]
    held: SimpleQueue[str] | None = None
    release: threading.Event | None = None

    def __call__(self, request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=json.dumps({"object": "list", "data": []}).encode())
        assert request.method == "POST", request
        marker: Final = _marker_of(request)
        attempt: Final = self.attempts.record(marker)
        if attempt > 1:
            return _served(request, marker, attempt)
        if self.held is not None and self.release is not None:
            self.held.put(marker)
            assert self.release.wait(timeout=120), "The burst was never released"
        return self.first_attempt(request, marker)


def _config(wire: Wire, directory: Path, models: _Models) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["model_list"] = [
        {
            "model_name": models.messages,
            "litellm_params": {
                "model": f"anthropic/{_ANTHROPIC_BACKEND}",
                "api_base": wire.url,
                "api_key": _ANTHROPIC_KEY,
                "num_retries": 1,
            },
        },
        {
            "model_name": models.openai,
            "litellm_params": {
                "model": f"openai/{_OPENAI_BACKEND}",
                "api_base": wire.url + "/v1",
                "api_key": _OPENAI_KEY,
                "num_retries": 1,
            },
        },
    ]
    config["router_settings"] = {"num_retries": 0, "disable_cooldowns": True}
    path: Final = directory / "pre-content-retry-chaos.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _models() -> _Models:
    suffix: Final = uuid.uuid4().hex
    return _Models(messages=f"audit-chaos-messages-{suffix}", openai=f"audit-chaos-openai-{suffix}")


async def _send(client: httpx.AsyncClient, key: str, models: _Models, call: _Call) -> _Served:
    async with client.stream(
        "POST",
        _path(call.endpoint),
        json=_body(models, call),
        headers={"Authorization": f"Bearer {key}", "anthropic-version": "2023-06-01"},
    ) as response:
        raw: Final = await response.aread()
    return _Served(call=call, status=response.status_code, text=raw.decode())


async def _burst(
    base_url: str, key: str, models: _Models, calls: tuple[_Call, ...], *, tolerate_transport_errors: bool = False
) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(base_url=base_url, timeout=120, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(_send(client, key, models, call) for call in calls), return_exceptions=tolerate_transport_errors
        )
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, httpx.TransportError), repr(result)
    return tuple(result for result in results if isinstance(result, _Served))


def _calls(plan: tuple[tuple[Endpoint, bool, int], ...]) -> tuple[_Call, ...]:
    return tuple(
        _Call(endpoint=endpoint, stream=stream, marker=uuid.uuid4().hex)
        for endpoint, stream, count in plan
        for _ in range(count)
    )  # comprehension-ok: a flat plan expansion, one marker per planned call


def _first_data_frame(text: str) -> dict[str, JsonValue]:
    line: Final = next(line for line in text.splitlines() if line.startswith("data: ") and "[DONE]" not in line)
    return object_value(json.loads(line.removeprefix("data: ")))


def _served_id(served: _Served) -> str:
    match served.call.endpoint, served.call.stream:
        case "messages", True:
            return message_id(parse_sse(served.text))
        case "responses", True:
            completed: Final = next(
                event for event in parse_sse(served.text) if event_type(event) == "response.completed"
            )
            return string_value(object_value(completed.data["response"])["id"])
        case "chat", True:
            return string_value(_first_data_frame(served.text)["id"])
        case _:
            return string_value(object_value(json.loads(served.text))["id"])


def _assert_completed_on_the_second_attempt(served: _Served) -> None:
    assert served.status == 200, (served.call, served.text)
    assert _answer(served.call.marker) in served.text, (served.call, served.text)
    assert "error" not in served.text.lower() or served.call.endpoint == "responses", (served.call, served.text)
    match served.call.endpoint, served.call.stream:
        case "messages", True:
            events: Final = parse_sse(served.text)
            assert event_types(events)[-1] == "message_stop", events
            assert delta_text(events) == _answer(served.call.marker), events
            assert message_id(events) == f"msg_{served.call.marker}_a2", events
        case "messages", False:
            assert _served_id(served) == f"msg_{served.call.marker}_a2", served.text
        case "chat", _:
            assert _served_id(served) == f"chatcmpl-{served.call.marker}-a2", served.text
        case "responses", _:
            assert f"msg_resp_{served.call.marker}_a2" in served.text, served.text


def _upstream_served_id(served: _Served) -> str:
    match served.call.endpoint:
        case "messages":
            return f"msg_{served.call.marker}_a2"
        case "chat":
            return f"chatcmpl-{served.call.marker}-a2"
        case "responses":
            return f"resp_{served.call.marker}_a2"


def _routing_decoded_upstream_id(request_id: str) -> str | None:
    encoded: Final = _ROUTING_ENCODED_ID.fullmatch(request_id)
    if encoded is None:
        return None
    decoded: Final = base64.b64decode(encoded.group(1)).decode(errors="replace")
    if not decoded.startswith("litellm:"):
        return None
    return decoded.rpartition("response_id:")[2]


def _names_of_row(request_id: str) -> frozenset[str]:
    return frozenset(name for name in (request_id, _routing_decoded_upstream_id(request_id)) if name is not None)


def _names_of_served(served: _Served) -> frozenset[str]:
    return frozenset({_served_id(served), _upstream_served_id(served)})


def _spend_ids(model: str, expected: int) -> tuple[str, ...]:
    rows: Final = eventually(
        lambda: read_rows('SELECT request_id, status FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
        lambda found: len(found) >= expected,
        seconds=90,
    )
    assert [row["status"] for row in rows] == ["success"] * len(rows), rows
    return tuple(string_value(row["request_id"]) for row in rows)


def _assert_rows_name_each_served_response_once(model: str, served: tuple[_Served, ...]) -> None:
    rows: Final = _spend_ids(model, len(served))
    named: Final = tuple(
        tuple(index for index, item in enumerate(served) if _names_of_served(item) & _names_of_row(request_id))
        for request_id in rows
    )
    assert sorted(named) == [(index,) for index in range(len(served))], (model, named, rows)


def _assert_each_served_id_landed_exactly_once(models: _Models, served: tuple[_Served, ...]) -> None:
    _assert_rows_name_each_served_response_once(
        models.messages, tuple(item for item in served if item.call.endpoint == "messages")
    )
    _assert_rows_name_each_served_response_once(
        models.openai, tuple(item for item in served if item.call.endpoint != "messages")
    )


def _open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


@pytest.mark.timeout(300)
async def test_worker_sigkill_mid_burst_leaves_the_sibling_retrying_pre_content_failures(
    gateway: Gateway, tmp_path: Path
) -> None:
    models: Final = _models()
    calls: Final = _calls(
        (("messages", True, 24), ("chat", False, 3), ("chat", True, 3), ("responses", False, 3), ("responses", True, 3))
    )
    release: Final = threading.Event()
    held: Final[SimpleQueue[str]] = SimpleQueue()
    upstream: Final = _Upstream(Attempts(), _drop_or_500, held, release)
    with wire_server(upstream) as wire:
        config: Final = _config(wire, tmp_path, models)
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            candidate: Final = owned.gateway
            base_url: Final = str(candidate.client.base_url)
            workers: Final = eventually(
                lambda: tuple(int(pid) for pid in _STARTED_WORKER.findall(owned.log.read_text())),
                lambda pids: len(pids) == 2,
                seconds=30,
            )
            burst: Final = asyncio.create_task(
                _burst(base_url, candidate.key, models, calls, tolerate_transport_errors=True)
            )
            await asyncio.to_thread(eventually, held.qsize, lambda size: size == len(calls), 90)
            async with httpx.AsyncClient(base_url=base_url, timeout=15, trust_env=False) as probe:
                alive: Final = await probe.get("/health/liveliness")
            assert alive.status_code == 200, alive.text
            held_by: Final = MappingProxyType({pid: _open_upstream_connections(pid, wire.url) for pid in workers})
            assert sum(held_by.values()) == len(calls), held_by
            victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
            victim: Final = psutil.Process(victim_pid)
            victim.suspend()
            victim.send_signal(signal.SIGKILL)
            release.set()
            served: Final = await burst
            assert held_by[survivor_pid] >= 10, held_by
            assert len(served) == held_by[survivor_pid], (held_by, len(served))
            for item in served:
                _assert_completed_on_the_second_attempt(item)
            follow_up: Final = _Call(endpoint="messages", stream=True, marker=uuid.uuid4().hex)
            (answered,) = await _burst(base_url, candidate.key, models, (follow_up,))
            _assert_completed_on_the_second_attempt(answered)
            _assert_each_served_id_landed_exactly_once(models, (*served, answered))
            assert all(upstream.attempts.count(item.call.marker) == 2 for item in (*served, answered))


@pytest.mark.timeout(300)
async def test_outage_on_every_first_attempt_is_absorbed_by_the_deployment_budget(
    gateway: Gateway, tmp_path: Path
) -> None:
    models: Final = _models()
    calls: Final = _calls(
        (
            ("messages", True, 6),
            ("messages", False, 6),
            ("chat", True, 6),
            ("chat", False, 6),
            ("responses", True, 6),
            ("responses", False, 6),
        )
    )
    statuses: Final = MappingProxyType(
        {call.marker: _OUTAGE_STATUSES[index % len(_OUTAGE_STATUSES)] for index, call in enumerate(calls)}
    )
    upstream: Final = _Upstream(Attempts(), _outage(statuses))
    with wire_server(upstream) as wire:
        config: Final = _config(wire, tmp_path, models)
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            candidate: Final = owned.gateway
            served: Final = await _burst(str(candidate.client.base_url), candidate.key, models, calls)
            assert len(served) == len(calls)
            for item in served:
                _assert_completed_on_the_second_attempt(item)
            _assert_each_served_id_landed_exactly_once(models, served)
            assert all(upstream.attempts.count(call.marker) == 2 for call in calls)
