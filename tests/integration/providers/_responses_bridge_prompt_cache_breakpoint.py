from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias, cast

import httpx
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolUnionParam
from pydantic import JsonValue, TypeAdapter

from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request
from litellm.responses.utils import ResponsesAPIRequestUtils as _RU

_MODEL: Final = "openai/gpt-5.6"

_UNSUPPORTED_MODEL: Final = "openai/gpt-5.4-mini"

_MARKER: Final = re.compile(rb"marker-([0-9a-f]{32})")

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
