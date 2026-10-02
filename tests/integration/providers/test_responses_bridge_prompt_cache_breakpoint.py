from __future__ import annotations

import asyncio
import json
import re
import signal
import threading
import uuid
from collections.abc import Iterable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from typing import Final, Literal, TypeAlias, cast
from urllib.parse import urlsplit

import httpx
import psutil
import pytest
import yaml
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolUnionParam
from pydantic import JsonValue, TypeAdapter

from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from litellm.responses.utils import ResponsesAPIRequestUtils as _RU

_MODEL: Final = "openai/gpt-5.6"
_UNSUPPORTED_MODEL: Final = "openai/gpt-5.4-mini"
_CONFIG_MODEL: Final = "responses-bridge-cache-breakpoint-chaos"
_API_KEY: Final = "synthetic-responses-bridge-key"
_MARKER: Final = re.compile(rb"marker-([0-9a-f]{32})")
_STARTED_WORKER: Final[re.Pattern[str]] = re.compile(r"Started server process \[(\d+)\]")
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_BREAKPOINT: Final[dict[str, JsonValue]] = {"mode": "explicit"}
_TOOLS: Final[list[JsonValue]] = [
    {
        "type": "function",
        "function": {
            "name": "synthetic_tool",
            "description": "Synthetic bridge test tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]
_IMAGE_URL: Final = "data:image/png;base64,aGVsbG8="
_ClientKind: TypeAlias = Literal["openai_sync", "openai_async", "httpx"]
_Surface: TypeAlias = Literal["chat", "responses"]


@dataclass(frozen=True, slots=True)
class _Call:
    surface: _Surface
    stream: bool
    marker: str


@dataclass(frozen=True, slots=True)
class _Served:
    call: _Call
    status: int
    response_id: str | None
    text: str


def _response_id(marker: str) -> str:
    return f"resp_{marker}"


def _request_marker(request: Request) -> str:
    match: Final = _MARKER.search(request.body)
    assert match is not None, request.body
    return match.group(1).decode()


def _contains_breakpoint(value: JsonValue) -> bool:
    if isinstance(value, dict):
        return "prompt_cache_breakpoint" in value or any(_contains_breakpoint(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_breakpoint(item) for item in value)
    return False


def _responses_body(marker: str) -> dict[str, JsonValue]:
    response_id: Final = _response_id(marker)
    return _JSON_OBJECT.validate_python(
        {
            "id": response_id,
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "gpt-5.6",
            "output": [
                {
                    "id": f"msg_{marker}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": f"answer marker-{marker}", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        }
    )


def _responses_reply(request: Request, *, reject_breakpoints: bool = False) -> Reply:
    if request.method == "GET" and request.target == "/v1/models":
        return Reply(body=b'{"object":"list","data":[{"id":"gpt-5.6","object":"model"}]}')
    body: Final = _JSON_OBJECT.validate_json(request.body)
    if reject_breakpoints and _contains_breakpoint(body):
        return Reply(
            status=400,
            body=json.dumps(
                {
                    "error": {
                        "message": "prompt_cache_breakpoint is not supported on this model",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": None,
                    }
                }
            ).encode(),
        )
    marker: Final = _request_marker(request)
    stream: Final = body.get("stream") is True
    response: Final = _responses_body(marker)
    if not stream:
        return Reply(body=json.dumps(response).encode())
    created: Final = {
        "type": "response.created",
        "sequence_number": 0,
        "response": {**response, "status": "in_progress", "output": []},
    }
    delta: Final = {
        "type": "response.output_text.delta",
        "sequence_number": 1,
        "item_id": f"msg_{marker}",
        "output_index": 0,
        "content_index": 0,
        "delta": f"answer marker-{marker}",
    }
    completed: Final = {"type": "response.completed", "sequence_number": 2, "response": response}
    events: Final = (created, delta, completed)
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events),
    )


def _prompt(marker: str, label: str) -> str:
    return f"{label} marker-{marker}"


def _simple_chat_body(
    model: str,
    marker: str,
    *,
    stream: bool = False,
    marked: bool = True,
    system_as_string: bool = False,
    prompt_cache_options: dict[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    marker_field: Final = {"prompt_cache_breakpoint": _BREAKPOINT} if marked else {}
    user: Final = [{"type": "text", "text": _prompt(marker, "user"), **marker_field}]
    messages: Final = (
        [{"role": "system", "content": _prompt(marker, "system")}, {"role": "user", "content": user}]
        if system_as_string
        else [{"role": "user", "content": user}]
    )
    return _JSON_OBJECT.validate_python(
        {
            "model": model,
            "messages": messages,
            "tools": _TOOLS,
            "reasoning_effort": "low",
            "stream": stream,
            "num_retries": 0,
            **({"prompt_cache_options": prompt_cache_options} if prompt_cache_options is not None else {}),
        }
    )


def _multimodal_chat_body(model: str, marker: str, stream: bool) -> dict[str, JsonValue]:
    return _JSON_OBJECT.validate_python(
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": _prompt(marker, "system"),
                            "prompt_cache_breakpoint": _BREAKPOINT,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _prompt(marker, "user"), "prompt_cache_breakpoint": _BREAKPOINT},
                        {
                            "type": "image_url",
                            "image_url": {"url": _IMAGE_URL},
                            "prompt_cache_breakpoint": _BREAKPOINT,
                        },
                        {
                            "type": "file",
                            "file": {"file_id": "file-abc"},
                            "prompt_cache_breakpoint": _BREAKPOINT,
                        },
                        {"type": "text", "text": "unmarked extra text"},
                    ],
                },
            ],
            "tools": _TOOLS,
            "reasoning_effort": "low",
            "stream": stream,
            "num_retries": 0,
            "prompt_cache_options": {"mode": "explicit"},
        }
    )


def _expected_multimodal_input(marker: str) -> list[JsonValue]:
    return [
        {
            "type": "message",
            "role": "system",
            "content": [
                {"type": "input_text", "text": _prompt(marker, "system"), "prompt_cache_breakpoint": _BREAKPOINT}
            ],
        },
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": _prompt(marker, "user"), "prompt_cache_breakpoint": _BREAKPOINT},
                {
                    "type": "input_image",
                    "image_url": _IMAGE_URL,
                    "detail": "auto",
                    "prompt_cache_breakpoint": _BREAKPOINT,
                },
                {"type": "input_file", "file_id": "file-abc", "prompt_cache_breakpoint": _BREAKPOINT},
                {"type": "input_text", "text": "unmarked extra text"},
            ],
        },
    ]


def _simple_expected_input(marker: str, *, marked: bool) -> list[JsonValue]:
    text_block: Final = {"type": "input_text", "text": _prompt(marker, "user")}
    return [
        {
            "type": "message",
            "role": "user",
            "content": [{**text_block, **({"prompt_cache_breakpoint": _BREAKPOINT} if marked else {})}],
        }
    ]


def _request_body(request: Request) -> dict[str, JsonValue]:
    assert request.method == "POST" and request.target == "/v1/responses", request.target
    return _JSON_OBJECT.validate_json(request.body)


def _decoded_response_id(response_id: str) -> str:
    decoded: Final = _RU._decode_responses_api_response_id(  # pyright: ignore[reportPrivateUsage]  # reuse ID decoder
        response_id
    )
    raw_response_id: Final = decoded.get("response_id")
    assert isinstance(raw_response_id, str), decoded
    return raw_response_id


def _spend_request_id_matches(
    row: Mapping[str, JsonValue],
    caller_response_id: str,
    peer_response_id: str,
    surface: _Surface,
) -> bool:
    request_id: Final = row.get("request_id")
    if not isinstance(request_id, str):
        return False
    match surface:
        case "responses":
            return request_id == caller_response_id
        case "chat":
            return _decoded_response_id(request_id) == peer_response_id


def _spend_rows(
    model: str,
    caller_response_id: str,
    peer_response_id: str,
    surface: _Surface,
) -> tuple[dict[str, JsonValue], ...]:
    def matching_rows(rows: list[dict[str, JsonValue]]) -> tuple[dict[str, JsonValue], ...]:
        return tuple(
            row for row in rows if _spend_request_id_matches(row, caller_response_id, peer_response_id, surface)
        )

    rows: Final = eventually(
        lambda: read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
        lambda candidates: len(matching_rows(candidates)) == 1,
        seconds=60,
    )
    matched: Final = matching_rows(rows)
    assert len(matched) == 1, matched
    return matched


def _response_id_from_chat_stream(text: str) -> str:
    payloads: Final = tuple(
        _JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: {")
    )
    assert payloads, text
    response_id: Final = payloads[0].get("id")
    assert isinstance(response_id, str), payloads[0]
    return response_id


def _extra_body(body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        key: value
        for key, value in body.items()
        if key not in {"model", "messages", "tools", "reasoning_effort", "stream", "num_retries"}
    }


def _sync_sdk_chat(gateway: Gateway, body: dict[str, JsonValue], stream: bool) -> _Served:
    base_url: Final = f"{str(gateway.client.base_url).rstrip('/')}/v1"
    model: Final = str(body["model"])
    messages: Final = cast(Iterable[ChatCompletionMessageParam], body["messages"])
    tools: Final = cast(Iterable[ChatCompletionToolUnionParam], body["tools"])
    extras: Final = _extra_body(body)
    with OpenAI(api_key=gateway.key, base_url=base_url, max_retries=0) as client:
        if stream:
            response_stream: Final = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                reasoning_effort="low",
                stream=True,
                extra_body=extras,
            )
            chunks: Final = tuple(response_stream)
            assert chunks
            return _Served(_Call("chat", True, _request_marker_from_body(body)), 200, chunks[0].id, "")
        response: Final = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            reasoning_effort="low",
            stream=False,
            extra_body=extras,
        )
        return _Served(_Call("chat", False, _request_marker_from_body(body)), 200, response.id, "")


async def _async_sdk_chat(gateway: Gateway, body: dict[str, JsonValue], stream: bool) -> _Served:
    base_url: Final = f"{str(gateway.client.base_url).rstrip('/')}/v1"
    model: Final = str(body["model"])
    messages: Final = cast(Iterable[ChatCompletionMessageParam], body["messages"])
    tools: Final = cast(Iterable[ChatCompletionToolUnionParam], body["tools"])
    extras: Final = _extra_body(body)
    async with AsyncOpenAI(api_key=gateway.key, base_url=base_url, max_retries=0) as client:
        if stream:
            response_stream: Final = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                reasoning_effort="low",
                stream=True,
                extra_body=extras,
            )
            chunks: Final = tuple([chunk async for chunk in response_stream])
            assert chunks
            return _Served(_Call("chat", True, _request_marker_from_body(body)), 200, chunks[0].id, "")
        response: Final = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            reasoning_effort="low",
            stream=False,
            extra_body=extras,
        )
        return _Served(_Call("chat", False, _request_marker_from_body(body)), 200, response.id, "")


async def _serve_chat(
    gateway: Gateway,
    body: dict[str, JsonValue],
    client_kind: _ClientKind,
    stream: bool,
) -> _Served:
    match client_kind:
        case "openai_sync":
            return _sync_sdk_chat(gateway, body, stream)
        case "openai_async":
            return await _async_sdk_chat(gateway, body, stream)
        case "httpx":
            async with httpx.AsyncClient(
                base_url=str(gateway.client.base_url),
                headers={"Authorization": f"Bearer {gateway.key}"},
                timeout=20,
                trust_env=False,
            ) as client:
                return await _raw_call(
                    client,
                    "/v1/chat/completions",
                    body,
                    _Call("chat", stream, _request_marker_from_body(body)),
                )


def _request_marker_from_body(body: Mapping[str, JsonValue]) -> str:
    match: Final = _MARKER.search(json.dumps(body).encode())
    assert match is not None, body
    return match.group(1).decode()


async def _raw_call(
    client: httpx.AsyncClient,
    path: str,
    body: Mapping[str, JsonValue],
    call: _Call,
) -> _Served:
    async with client.stream(
        "POST",
        path,
        json=body,
        headers={"Authorization": f"Bearer {client.headers['Authorization'].removeprefix('Bearer ')}"},
    ) as response:
        content: Final = await response.aread()
        status: Final = response.status_code
    text: Final = content.decode()
    response_id: Final = (
        _response_id_from_chat_stream(text)
        if status == 200 and call.surface == "chat" and call.stream
        else _JSON_OBJECT.validate_json(content).get("id")
        if status == 200
        else None
    )
    return _Served(call, status, response_id if isinstance(response_id, str) else None, text)


async def _send_call(
    client: httpx.AsyncClient,
    model: str,
    call: _Call,
) -> _Served:
    body: Final = (
        _simple_chat_body(model, call.marker, stream=call.stream)
        if call.surface == "chat"
        else {
            "model": model,
            "input": _simple_expected_input(call.marker, marked=True),
            "stream": call.stream,
            "num_retries": 0,
        }
    )
    path: Final = "/v1/chat/completions" if call.surface == "chat" else "/v1/responses"
    try:
        return await _raw_call(client, path, body, call)
    except httpx.TransportError as error:
        return _Served(call, 0, None, f"{type(error).__name__}: {error}")


async def _burst(
    base_url: str,
    key: str,
    model: str,
    calls: tuple[_Call, ...],
) -> tuple[_Served, ...]:
    async with httpx.AsyncClient(
        base_url=base_url,
        headers={"Authorization": f"Bearer {key}"},
        timeout=20,
        trust_env=False,
        limits=httpx.Limits(max_connections=100),
    ) as client:
        return tuple(await asyncio.gather(*(_send_call(client, model, call) for call in calls)))


def _calls(count: int) -> tuple[_Call, ...]:
    surfaces: Final[tuple[_Surface, ...]] = ("chat", "chat", "responses")
    return tuple(
        _Call(
            surface=surfaces[index % len(surfaces)],
            stream=index % 3 == 1,
            marker=uuid.uuid4().hex,
        )
        for index in range(count)
    )


def _requests_for_marker(requests: tuple[Request, ...], marker: str) -> tuple[Request, ...]:
    return tuple(
        request
        for request in requests
        if request.method == "POST" and request.target == "/v1/responses" and _request_marker(request) == marker
    )


def _peer_request_has_marker(request: Request, marker: str) -> bool:
    body: Final = _request_body(request)
    return _contains_breakpoint(body) and _request_marker(request) == marker


def _peer_marker_matches_response(served: _Served, requests: tuple[Request, ...]) -> bool:
    peer_requests: Final = _requests_for_marker(requests, served.call.marker)
    assert len(peer_requests) == 1, (served, peer_requests)
    (peer_request,) = peer_requests
    return _peer_request_has_marker(peer_request, served.call.marker)


def _assert_spend_for_result(served: _Served, model: str) -> None:
    assert served.status == 200 and served.response_id is not None, served
    peer_response_id: Final = _response_id(served.call.marker)
    match served.call.surface:
        case "responses":
            (row,) = _spend_rows(model, served.response_id, peer_response_id, served.call.surface)
        case "chat":
            assert _decoded_response_id(served.response_id) == peer_response_id, served
            (row,) = _spend_rows(model, served.response_id, peer_response_id, served.call.surface)
    request_id: Final = row.get("request_id")
    assert isinstance(request_id, str), row
    match served.call.surface:
        case "responses":
            assert request_id == served.response_id, row
        case "chat":
            assert _decoded_response_id(request_id) == served.response_id, row


@pytest.mark.parametrize("client_kind", ("openai_sync", "openai_async", "httpx"))
@pytest.mark.parametrize("stream", (False, True))
async def test_caller_prompt_cache_breakpoints_survive_chat_to_responses_bridge(
    gateway: Gateway,
    client_kind: _ClientKind,
    stream: bool,
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_responses_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url + "/v1")
        body: Final = _multimodal_chat_body(model, marker, stream)
        served: Final = await _serve_chat(gateway, body, client_kind, stream)
        assert served.status == 200, served.text
        _assert_spend_for_result(served, model)
        (peer_request,) = wire.drain()
        peer_body: Final = _request_body(peer_request)
        assert peer_body["input"] == _expected_multimodal_input(marker), peer_body
        assert peer_body["prompt_cache_options"] == {"mode": "explicit"}, peer_body


def _expected_uninjected_system_bridge_body(
    marker: str,
    prompt_cache_options: dict[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    return {
        "input": _simple_expected_input(marker, marked=False),
        "instructions": _prompt(marker, "system"),
        "model": "gpt-5.6",
        "reasoning": {"effort": "low"},
        "stream": False,
        "tools": [
            {
                "type": "function",
                "name": "synthetic_tool",
                "parameters": {"type": "object", "properties": {}},
                "strict": None,
                "description": "Synthetic bridge test tool",
            }
        ],
        **({"prompt_cache_options": prompt_cache_options} if prompt_cache_options is not None else {}),
    }


async def test_deployment_cache_control_injection_without_options_is_unchanged(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_responses_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=_MODEL,
            api_base=wire.url + "/v1",
            cache_control_injection_points=[{"location": "message", "role": "system"}],
        )
        body: Final = _simple_chat_body(model, marker, system_as_string=True, marked=False)
        async with httpx.AsyncClient(
            base_url=str(gateway.client.base_url),
            headers={"Authorization": f"Bearer {gateway.key}"},
            timeout=20,
            trust_env=False,
        ) as client:
            served: Final = await _raw_call(client, "/v1/chat/completions", body, _Call("chat", False, marker))
        assert served.status == 200, served.text
        _assert_spend_for_result(served, model)
        (peer_request,) = wire.drain()
        peer_body: Final = _request_body(peer_request)
        assert peer_body == _expected_uninjected_system_bridge_body(marker), peer_body
        assert not _contains_breakpoint(peer_body), peer_body
        assert "prompt_cache_options" not in peer_body, peer_body


async def test_deployment_prompt_cache_options_override_is_unchanged(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    options: Final[dict[str, JsonValue]] = {"mode": "implicit", "ttl": "30m"}
    with wire_server(_responses_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=_MODEL,
            api_base=wire.url + "/v1",
            prompt_cache_options=options,
        )
        body: Final = _simple_chat_body(model, marker, system_as_string=True, marked=False)
        async with httpx.AsyncClient(
            base_url=str(gateway.client.base_url),
            headers={"Authorization": f"Bearer {gateway.key}"},
            timeout=20,
            trust_env=False,
        ) as client:
            served: Final = await _raw_call(client, "/v1/chat/completions", body, _Call("chat", False, marker))
        assert served.status == 200, served.text
        _assert_spend_for_result(served, model)
        (peer_request,) = wire.drain()
        peer_body: Final = _request_body(peer_request)
        assert peer_body == _expected_uninjected_system_bridge_body(marker, options), peer_body
        assert not _contains_breakpoint(peer_body), peer_body


async def test_unmarked_bridge_and_direct_responses_marker_are_forwarded_unchanged(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_responses_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url + "/v1")
        unmarked_body: Final = _simple_chat_body(model, marker, marked=False)
        async with httpx.AsyncClient(
            base_url=str(gateway.client.base_url),
            headers={"Authorization": f"Bearer {gateway.key}"},
            timeout=20,
            trust_env=False,
        ) as client:
            unmarked: Final = await _raw_call(
                client,
                "/v1/chat/completions",
                unmarked_body,
                _Call("chat", False, marker),
            )
        assert unmarked.status == 200, unmarked.text
        _assert_spend_for_result(unmarked, model)
        (unmarked_peer,) = wire.drain()
        unmarked_body_at_peer: Final = _request_body(unmarked_peer)
        assert not _contains_breakpoint(unmarked_body_at_peer), unmarked_body_at_peer
        assert "prompt_cache_options" not in unmarked_body_at_peer, unmarked_body_at_peer

        direct_marker: Final = uuid.uuid4().hex
        direct_input: Final = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": _prompt(direct_marker, "direct"),
                        "prompt_cache_breakpoint": _BREAKPOINT,
                    }
                ],
            }
        ]
        direct_body: Final = _JSON_OBJECT.validate_python({"model": model, "input": direct_input, "store": False})
        async with httpx.AsyncClient(
            base_url=str(gateway.client.base_url),
            headers={"Authorization": f"Bearer {gateway.key}"},
            timeout=20,
            trust_env=False,
        ) as client:
            direct: Final = await _raw_call(
                client,
                "/v1/responses",
                direct_body,
                _Call("responses", False, direct_marker),
            )
        assert direct.status == 200, direct.text
        _assert_spend_for_result(direct, model)
        (direct_peer,) = wire.drain()
        assert _request_body(direct_peer)["input"] == direct_input, _request_body(direct_peer)


@pytest.mark.parametrize("stream", (False, True))
async def test_unsupported_model_drops_breakpoints_without_rejecting_the_request(
    gateway: Gateway,
    stream: bool,
) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(lambda request: _responses_reply(request, reject_breakpoints=True)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model=_UNSUPPORTED_MODEL, api_base=wire.url + "/v1")
        body: Final = _simple_chat_body(model, marker, stream=stream)
        async with httpx.AsyncClient(
            base_url=str(gateway.client.base_url),
            headers={"Authorization": f"Bearer {gateway.key}"},
            timeout=20,
            trust_env=False,
        ) as client:
            served: Final = await _raw_call(client, "/v1/chat/completions", body, _Call("chat", stream, marker))
        assert served.status == 200, served.text
        _assert_spend_for_result(served, model)
        (peer_request,) = wire.drain()
        peer_body: Final = _request_body(peer_request)
        assert peer_body["input"] == _simple_expected_input(marker, marked=False), peer_body
        assert not _contains_breakpoint(peer_body), peer_body


def _chaos_config(wire: Wire, tmp_path: Path) -> Path:
    base_config: Final = _JSON_OBJECT.validate_python(
        yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    )
    config: Final = {
        **base_config,
        "model_list": [
            {
                "model_name": _CONFIG_MODEL,
                "litellm_params": {
                    "model": _MODEL,
                    "api_base": wire.url + "/v1",
                    "api_key": _API_KEY,
                },
            },
        ],
    }
    path: Final = tmp_path / "responses-bridge-cache-breakpoint-chaos.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _open_upstream_connections(pid: int, port: int) -> int:
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


@pytest.mark.timeout(180)
async def test_worker_and_peer_outages_preserve_markers_and_recover(
    gateway: Gateway,
    tmp_path: Path,
) -> None:
    calls: Final = _calls(30)
    release: Final = threading.Event()
    early_release: Final = threading.Event()
    outage_release: Final = threading.Event()
    held_markers: Final[SimpleQueue[str]] = SimpleQueue()
    early_calls: Final = calls[:10]
    early_markers: Final = frozenset(call.marker for call in early_calls)

    def held(request: Request) -> Reply:
        if request.method == "GET" and request.target == "/v1/models":
            return _responses_reply(request)
        marker: Final = _request_marker(request)
        held_markers.put(marker)
        gate: Final = early_release if marker in early_markers else release
        assert gate.wait(timeout=60), "The worker-kill burst was never released"
        return _responses_reply(request)

    with ExitStack() as peer_stack:
        wire: Final = peer_stack.enter_context(wire_server(held))
        config: Final = _chaos_config(wire, tmp_path)
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            try:
                candidate: Final = owned.gateway
                workers: Final[tuple[int, ...]] = eventually(
                    lambda: tuple(int(match.group(1)) for match in _STARTED_WORKER.finditer(owned.log.read_text())),
                    lambda pids: len(pids) == 2,
                    seconds=30,
                )
                async with httpx.AsyncClient(
                    base_url=str(candidate.client.base_url),
                    headers={"Authorization": f"Bearer {candidate.key}"},
                    timeout=20,
                    trust_env=False,
                    limits=httpx.Limits(max_connections=100),
                ) as client:
                    burst_tasks: Final = tuple(
                        asyncio.create_task(_send_call(client, _CONFIG_MODEL, call)) for call in calls
                    )
                    await asyncio.to_thread(eventually, held_markers.qsize, lambda size: size == len(calls), 60)
                    early_release.set()
                    early_served: Final = await asyncio.gather(*burst_tasks[: len(early_calls)])
                    early_successful: Final = tuple(item for item in early_served if item.status == 200)
                    for item in early_successful:
                        _assert_spend_for_result(item, _CONFIG_MODEL)
                    upstream_port_value: Final = urlsplit(wire.url).port
                    assert upstream_port_value is not None
                    upstream_port: Final = upstream_port_value
                    active_by_worker: Final = eventually(
                        lambda: {pid: _open_upstream_connections(pid, upstream_port) for pid in workers},
                        lambda counts: sum(counts.values()) == len(calls) - len(early_calls),
                        seconds=30,
                    )
                    victim_pid: Final = max(workers, key=active_by_worker.__getitem__)
                    survivor_pids: Final = tuple(pid for pid in workers if pid != victim_pid)
                    assert active_by_worker[victim_pid] > 0 and len(survivor_pids) == 1, active_by_worker
                    (survivor_pid,) = survivor_pids
                    victim: Final = psutil.Process(victim_pid)
                    victim.suspend()
                    victim.send_signal(signal.SIGKILL)
                    release.set()
                    remaining_served: Final = await asyncio.gather(*burst_tasks[len(early_calls) :])
                    served: Final = (*early_served, *remaining_served)
                successful: Final = tuple(item for item in served if item.status == 200)
                connection_errors: Final = tuple(item for item in served if item.status == 0)
                print(f"worker-kill burst: {len(successful)} HTTP 200, {len(connection_errors)} connection errors")
                assert len(successful) + len(connection_errors) == len(calls), {
                    "successes": len(successful),
                    "connection_errors": len(connection_errors),
                    "responses": served,
                }
                assert successful and connection_errors, {
                    "successes": len(successful),
                    "connection_errors": len(connection_errors),
                }
                follow_ups: Final = (
                    _Call("chat", False, uuid.uuid4().hex),
                    _Call("responses", False, uuid.uuid4().hex),
                )
                recovered: Final = await _burst(
                    str(candidate.client.base_url),
                    candidate.key,
                    _CONFIG_MODEL,
                    follow_ups,
                )
                assert all(item.status == 200 for item in recovered), recovered
                assert psutil.pid_exists(survivor_pid), survivor_pid
                received_after_worker_kill: Final = wire.drain()
                worker_marker_failures: Final = tuple(
                    item.call.marker
                    for item in (*successful, *recovered)
                    if not _peer_marker_matches_response(item, received_after_worker_kill)
                )
                for item in (*successful, *recovered):
                    assert item.response_id is not None, item
                    _assert_spend_for_result(item, _CONFIG_MODEL)

                peer_stack.close()
                outage_seen: Final[SimpleQueue[str]] = SimpleQueue()

                def outage(request: Request) -> Reply:
                    if request.method == "GET" and request.target == "/v1/models":
                        return _responses_reply(request)
                    outage_seen.put(_request_marker(request))
                    assert outage_release.wait(timeout=60), "The peer-outage burst was never stopped"
                    return Reply(
                        status=503,
                        body=b'{"error":{"message":"synthetic peer outage","type":"server_error"}}',
                    )

                peer_stack.enter_context(wire_server(outage, port=upstream_port))
                outage_calls: Final = _calls(12)
                outage_burst: Final = asyncio.create_task(
                    _burst(str(candidate.client.base_url), candidate.key, _CONFIG_MODEL, outage_calls)
                )
                await asyncio.to_thread(eventually, outage_seen.qsize, lambda size: size == len(outage_calls), 30)
                outage_release.set()
                peer_stack.close()
                outage_served: Final = await outage_burst
                assert len(outage_served) == len(outage_calls), outage_served
                assert all(item.status >= 400 and "error" in item.text.lower() for item in outage_served), outage_served
                down_call: Final = _Call("chat", False, uuid.uuid4().hex)
                (down_response,) = await _burst(
                    str(candidate.client.base_url),
                    candidate.key,
                    _CONFIG_MODEL,
                    (down_call,),
                )
                assert down_response.status >= 400 and "error" in down_response.text.lower(), down_response

                restarted_wire: Final = peer_stack.enter_context(wire_server(_responses_reply, port=upstream_port))
                recovery_calls: Final = (
                    _Call("chat", False, uuid.uuid4().hex),
                    _Call("responses", False, uuid.uuid4().hex),
                )
                recovered_after_peer_restart: Final = await _burst(
                    str(candidate.client.base_url),
                    candidate.key,
                    _CONFIG_MODEL,
                    recovery_calls,
                )
                assert all(item.status == 200 for item in recovered_after_peer_restart), recovered_after_peer_restart
                restarted_requests: Final = restarted_wire.drain()
                recovery_marker_failures: Final = tuple(
                    item.call.marker
                    for item in recovered_after_peer_restart
                    if not _peer_marker_matches_response(item, restarted_requests)
                )
                for item in recovered_after_peer_restart:
                    assert item.response_id is not None, item
                    _assert_spend_for_result(item, _CONFIG_MODEL)
                assert not (*worker_marker_failures, *recovery_marker_failures), {
                    "worker_marker_failures": worker_marker_failures,
                    "recovery_marker_failures": recovery_marker_failures,
                }
            finally:
                release.set()
                outage_release.set()
