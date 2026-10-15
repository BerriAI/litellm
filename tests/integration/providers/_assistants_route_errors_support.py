from __future__ import annotations

import asyncio
import json
import re
import socket
import threading
import uuid
from collections.abc import AsyncIterator, Callable, Generator, Mapping
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final

import httpx
import psutil
import yaml
from integration._support.client import JSON_OBJECT, Gateway, eventually, object_value
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

MARKER: Final = re.compile(r"m[0-9a-f]{32}")
TRACEBACK: Final = "Traceback (most recent call last)"
BUG_NOTICE: Final = "This looks like a bug in LiteLLM"
ASSISTANT_MODEL: Final = "scripted-assistant-model"
CONTROL_MODEL: Final = "assistants-route-errors-control"
SESSION_HEADER: Final = MappingProxyType({"x-litellm-session-id": "assistants-route-errors"})
STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")


@dataclass(frozen=True, slots=True)
class Route:
    name: str
    method: str
    path: str
    reads_body: bool
    carries_marker: bool


ROUTES: Final = (
    Route("get_assistants", "GET", "/v1/assistants", reads_body=False, carries_marker=False),
    Route("create_assistant", "POST", "/v1/assistants", reads_body=True, carries_marker=True),
    Route("delete_assistant", "DELETE", "/v1/assistants/asst_{marker}", reads_body=False, carries_marker=True),
    Route("create_thread", "POST", "/v1/threads", reads_body=False, carries_marker=False),
    Route("get_thread", "GET", "/v1/threads/thread_{marker}", reads_body=False, carries_marker=True),
    Route("add_message", "POST", "/v1/threads/thread_{marker}/messages", reads_body=True, carries_marker=True),
    Route("get_messages", "GET", "/v1/threads/thread_{marker}/messages", reads_body=False, carries_marker=True),
    Route("run_thread", "POST", "/v1/threads/thread_{marker}/runs", reads_body=True, carries_marker=True),
)
ROUTE_IDS: Final = tuple(route.name for route in ROUTES)
MARKED_ROUTES: Final = tuple(route for route in ROUTES if route.carries_marker)
BODY_ROUTES: Final = tuple(route for route in ROUTES if route.reads_body)
ROUTE_BY_NAME: Final = MappingProxyType({route.name: route for route in ROUTES})


@dataclass(frozen=True, slots=True)
class Mapped:
    status: int
    type: str
    param: str | None


STATUS_TABLE: Final = (
    Mapped(400, "scripted_type", "scripted_param"),
    Mapped(401, "authentication_error", None),
    Mapped(403, "permission_error", None),
    Mapped(404, "invalid_request_error", None),
    Mapped(408, "invalid_request_error", None),
    Mapped(409, "invalid_request_error", None),
    Mapped(422, "scripted_type", "scripted_param"),
    Mapped(429, "throttling_error", "scripted_param"),
    Mapped(500, "internal_server_error", "scripted_param"),
    Mapped(502, "internal_server_error", None),
    Mapped(503, "internal_server_error", None),
    Mapped(504, "internal_server_error", None),
)
MAPPED_BY_STATUS: Final = MappingProxyType({mapped.status: mapped for mapped in STATUS_TABLE})
UNMAPPED_500: Final = Mapped(500, "internal_server_error", None)


def new_marker() -> str:
    return f"m{uuid.uuid4().hex}"


def free_port() -> int:
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        return reserve.getsockname()[1]


def proxy_path(route: Route, marker: str) -> str:
    return route.path.format(marker=marker)


def request_body(route: Route, marker: str) -> dict[str, JsonValue] | None:
    match route.name:
        case "create_assistant":
            return {"model": ASSISTANT_MODEL, "name": marker}
        case "create_thread":
            return {"messages": [{"role": "user", "content": marker}]}
        case "add_message":
            return {"role": "user", "content": marker}
        case "run_thread":
            return {"assistant_id": f"asst_{marker}"}
        case _:
            return None


def marker_of(request: Request) -> str | None:
    found: Final = MARKER.search(request.target) or MARKER.search(request.body.decode(errors="replace"))
    return found.group(0) if found else None


def route_of(request: Request, marker: str) -> Route:
    target: Final = (request.method, request.target.split("?", 1)[0])
    matches: Final = tuple(route for route in ROUTES if (route.method, proxy_path(route, marker)) == target)
    assert len(matches) == 1, target
    return matches[0]


def error_reply(status: int, message: str) -> Reply:
    error: Final = {"message": message, "type": "scripted_type", "param": "scripted_param", "code": "scripted_code"}
    return Reply(status=status, body=json.dumps({"error": error}).encode())


def scripted_message(marker: str, status: int) -> str:
    return f"scripted {marker} status {status}"


def _assistant(marker: str) -> dict[str, JsonValue]:
    return {
        "id": f"asst_{marker}",
        "object": "assistant",
        "created_at": 1,
        "name": marker,
        "description": None,
        "model": ASSISTANT_MODEL,
        "instructions": None,
        "tools": [],
        "metadata": {},
    }


def _thread(marker: str) -> dict[str, JsonValue]:
    return {"id": f"thread_{marker}", "object": "thread", "created_at": 1, "metadata": {}, "tool_resources": None}


def _message(marker: str) -> dict[str, JsonValue]:
    return {
        "id": f"msg_{marker}",
        "object": "thread.message",
        "created_at": 1,
        "thread_id": f"thread_{marker}",
        "role": "user",
        "content": [{"type": "text", "text": {"value": marker, "annotations": []}}],
        "assistant_id": None,
        "run_id": None,
        "attachments": [],
        "metadata": {},
        "status": "completed",
    }


def _run(marker: str, status: str) -> dict[str, JsonValue]:
    return {
        "id": f"run_{marker}",
        "object": "thread.run",
        "created_at": 1,
        "thread_id": f"thread_{marker}",
        "assistant_id": f"asst_{marker}",
        "status": status,
        "model": ASSISTANT_MODEL,
        "instructions": "",
        "tools": [],
        "metadata": {},
        "parallel_tool_calls": True,
    }


def _page(item: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {"object": "list", "data": [item], "first_id": item["id"], "last_id": item["id"], "has_more": False}


def success_body(route: Route, marker: str) -> dict[str, JsonValue]:
    match route.name:
        case "get_assistants":
            return _page(_assistant(marker))
        case "create_assistant":
            return _assistant(marker)
        case "delete_assistant":
            return {"id": f"asst_{marker}", "object": "assistant.deleted", "deleted": True}
        case "create_thread" | "get_thread":
            return _thread(marker)
        case "add_message":
            return _message(marker)
        case "get_messages":
            return _page(_message(marker))
        case _:
            return _run(marker, "queued")


def run_poll_path(marker: str) -> str:
    return f"/v1/threads/thread_{marker}/runs/run_{marker}"


def success_trail(route: Route, marker: str) -> tuple[tuple[str, str], ...]:
    first: Final = (route.method, proxy_path(route, marker))
    return (first, ("GET", run_poll_path(marker))) if route.name == "run_thread" else (first,)


def chat_completion_body(marker: str) -> dict[str, JsonValue]:
    return {
        "id": f"chatcmpl-{marker}",
        "object": "chat.completion",
        "created": 1,
        "model": "scripted-chat-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": marker}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def success_peer(marker: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.target == "/v1/chat/completions":
            return Reply(body=json.dumps(chat_completion_body(marker)).encode())
        if request.method == "GET" and request.target == run_poll_path(marker):
            return Reply(body=json.dumps(_run(marker, "completed")).encode())
        return Reply(body=json.dumps(success_body(route_of(request, marker), marker)).encode())

    return respond


def error_object(response: httpx.Response) -> dict[str, JsonValue]:
    body: Final = JSON_OBJECT.validate_json(response.content)
    assert set(body) == {"error"}, response.text
    return object_value(body["error"])


def assert_clean_message(message: JsonValue) -> str:
    assert isinstance(message, str), message
    assert TRACEBACK not in message, message
    assert BUG_NOTICE not in message, message
    return message


def assert_openai_error(response: httpx.Response, mapped: Mapped) -> str:
    assert response.status_code == mapped.status, response.text
    error: Final = error_object(response)
    assert set(error) == {"message", "type", "param", "code"}, response.text
    assert (error["type"], error["param"], error["code"]) == (mapped.type, mapped.param, str(mapped.status)), (
        response.text
    )
    return assert_clean_message(error["message"])


def assert_mapped_upstream_error(response: httpx.Response, mapped: Mapped, marker: str) -> None:
    message: Final = assert_openai_error(response, mapped)
    assert marker in message, response.text


def is_model_listing(request: Request) -> bool:
    return request.method == "GET" and request.target.split("?", 1)[0].endswith("/models")


def provider_requests(received: tuple[Request, ...]) -> tuple[Request, ...]:
    return tuple(request for request in received if not is_model_listing(request))


def upstream_trail(received: tuple[Request, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((request.method, request.target.split("?", 1)[0]) for request in provider_requests(received))


@contextmanager
def assistants_wire(respond: Callable[[Request], Reply], port: int = 0) -> Generator[Wire, None, None]:
    def answer(request: Request) -> Reply:
        if is_model_listing(request):
            return Reply(body=json.dumps({"object": "list", "data": []}).encode())
        return respond(request)

    with wire_server(answer, port=port) as wire:
        yield wire


def assert_reached_upstream_once(received: tuple[Request, ...], route: Route, marker: str) -> None:
    assert upstream_trail(received) == ((route.method, proxy_path(route, marker)),), received


def owned_config(
    directory: Path,
    model_list: tuple[Mapping[str, JsonValue], ...],
    assistant_settings: Mapping[str, JsonValue],
    general_settings: Mapping[str, JsonValue] = MappingProxyType({}),
) -> Path:
    config: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    merged: Final = {
        **config,
        "model_list": [dict(model) for model in model_list],
        "general_settings": {**object_value(config["general_settings"]), **general_settings},
        "router_settings": {**object_value(config["router_settings"]), "num_retries": 0},
        "assistant_settings": dict(assistant_settings),
    }
    path: Final = directory / f"assistants-route-errors-{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump(merged))
    return path


def openai_assistants_config(directory: Path, upstream_port: int, timeout_seconds: int) -> Path:
    api_base: Final = f"http://127.0.0.1:{upstream_port}/v1"
    deployment: Final[dict[str, JsonValue]] = {
        "api_base": api_base,
        "api_key": "sk-scripted-assistants",
        "max_retries": 0,
        "timeout": timeout_seconds,
    }
    control: Final[dict[str, JsonValue]] = {
        "model_name": CONTROL_MODEL,
        "litellm_params": {"model": "openai/scripted-chat-model", **deployment},
    }
    return owned_config(directory, (control,), {"custom_llm_provider": "openai", "litellm_params": deployment})


def worker_pids(log: Path) -> tuple[int, ...]:
    return tuple(int(pid) for pid in STARTED_WORKER.findall(log.read_text()))


def live_worker_pids(log: Path) -> tuple[int, ...]:
    return tuple(pid for pid in worker_pids(log) if psutil.pid_exists(pid))


def open_upstream_connections(pid: int, upstream_port: int) -> int:
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == upstream_port
    )


def held_upstream_connections(workers: tuple[int, ...], upstream_port: int, expected: int) -> Mapping[int, int]:
    return eventually(
        lambda: MappingProxyType({pid: open_upstream_connections(pid, upstream_port) for pid in workers}),
        lambda held_by: sum(held_by.values()) == expected,
        seconds=10,
    )


def held_error_peer(
    status_by_marker: Mapping[str, int], arrived: SimpleQueue[str], release: threading.Event
) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        marker: Final = marker_of(request)
        assert marker is not None, request.target
        arrived.put(marker)
        assert release.wait(timeout=60), "The burst was never released"
        return error_reply(status_by_marker[marker], scripted_message(marker, status_by_marker[marker]))

    return respond


@dataclass(frozen=True, slots=True)
class Call:
    route: Route
    marker: str


@dataclass(frozen=True, slots=True)
class Answer:
    call: Call
    response: httpx.Response


async def _answer(client: httpx.AsyncClient, call: Call) -> Answer:
    response: Final = await client.request(
        call.route.method, proxy_path(call.route, call.marker), json=request_body(call.route, call.marker)
    )
    return Answer(call, response)


def _async_client(gateway: Gateway) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=str(gateway.client.base_url),
        headers={"Authorization": f"Bearer {gateway.key}"},
        timeout=60,
        trust_env=False,
    )


def _answers_of(results: tuple[Answer | BaseException, ...]) -> tuple[Answer, ...]:
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, httpx.TransportError), repr(result)
    return tuple(result for result in results if isinstance(result, Answer))


async def answer_all(
    gateway: Gateway, calls: tuple[Call, ...], *, tolerate_transport_errors: bool = False
) -> tuple[Answer, ...]:
    async with _async_client(gateway) as client:
        results: Final = await asyncio.gather(
            *(_answer(client, call) for call in calls), return_exceptions=tolerate_transport_errors
        )
    return _answers_of(tuple(results))


async def _send_one_by_one(
    client: httpx.AsyncClient,
    calls: tuple[Call, ...],
    arrived: SimpleQueue[str],
    enough: Callable[[], bool],
) -> tuple[asyncio.Task[Answer], ...]:
    if not calls:
        return ()
    sent: Final = asyncio.create_task(_answer(client, calls[0]))
    assert await asyncio.to_thread(arrived.get, True, 30) == calls[0].marker
    if enough():
        return (sent,)
    return (sent, *await _send_one_by_one(client, calls[1:], arrived, enough))


@dataclass(frozen=True, slots=True)
class Held:
    calls: tuple[Call, ...]
    pending: tuple[asyncio.Task[Answer], ...]

    async def answers(self) -> tuple[Answer, ...]:
        return _answers_of(tuple(await asyncio.gather(*self.pending, return_exceptions=True)))


@asynccontextmanager
async def held_one_by_one(
    gateway: Gateway, calls: tuple[Call, ...], arrived: SimpleQueue[str], enough: Callable[[], bool]
) -> AsyncIterator[Held]:
    async with _async_client(gateway) as client:
        pending: Final = await _send_one_by_one(client, calls, arrived, enough)
        yield Held(calls[: len(pending)], pending)


def assert_answered_with_its_own_marker(answer: Answer, mapped: Mapped) -> None:
    assert_mapped_upstream_error(answer.response, mapped, answer.call.marker)
    assert set(MARKER.findall(answer.response.text)) == {answer.call.marker}, answer.response.text
