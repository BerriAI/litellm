import asyncio
import base64
import json
import re
import threading
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from types import MappingProxyType
from typing import Final, Literal

import anthropic
import httpx
import openai
import pytest
import websockets
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    gateway_from_environment,
    object_value,
    string_value,
)
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from integration._support.upstream import GEMINI_LIVE_PATH
from integration.providers._gemini_tts_voice_support import live_deployment, live_scenario
from integration.providers._vertex_batch_rows import encoded, gcs_peer, is_gcs_upload, jsonl, upload, vertex_model
from pydantic import JsonValue, TypeAdapter

Provider = Literal["vertex_ai", "gemini"]
Endpoint = Literal["chat", "messages", "responses"]

_BACKEND: Final = "gemini-3.8-flash"
_NO_JSON_SCHEMA_BACKEND: Final = "gemini-flash-latest"
_IMAGE_BACKEND: Final = "gemini-3.1-flash-image"
_PROJECT: Final = "scripted-project"
_LOCATION: Final = "us-central1"
_GEMINI_KEY: Final = "sk-scripted-gemini"
_COUNT: Final = 777
_FAIL_MARKER: Final = "scripted-invalid-argument"
_STRING_ANYOF: Final = re.compile(rb'"anyOf":\s*"')
_PNG: Final = base64.b64encode(b"\x89PNG\r\n\x1a\nscripted").decode()
_JSON_LIST: Final = TypeAdapter(list[JsonValue])

_NULL: Final[Mapping[str, JsonValue]] = {"type": "null"}
_STRING: Final[Mapping[str, JsonValue]] = {"type": "string"}
_INTEGER: Final[Mapping[str, JsonValue]] = {"type": "integer"}
_NULLABLE_STRING: Final[JsonValue] = {"anyOf": [{"type": "string", "nullable": True}]}
_NULLABLE_NULL: Final[JsonValue] = {"anyOf": [{"type": "null", "nullable": True}]}

_UNIONS: Final[tuple[tuple[str, JsonValue, JsonValue], ...]] = (
    ("null-null-string", {"anyOf": [_NULL, _NULL, _STRING]}, _NULLABLE_STRING),
    ("null-only", {"anyOf": [_NULL]}, _NULLABLE_NULL),
    ("two-nulls-only", {"anyOf": [_NULL, _NULL]}, _NULLABLE_NULL),
    ("string-null", {"anyOf": [_STRING, _NULL]}, _NULLABLE_STRING),
    (
        "integer-null-null-string",
        {"anyOf": [_INTEGER, _NULL, _NULL, _STRING]},
        {"anyOf": [{"type": "integer", "nullable": True}, {"type": "string", "nullable": True}]},
    ),
    ("null-empty", {"anyOf": [_NULL, {}]}, {"anyOf": [{"type": "object", "nullable": True}]}),
    ("empty-only", {"anyOf": [{}]}, {"anyOf": [{"type": "object"}]}),
    ("empty-list", {"anyOf": []}, {"anyOf": []}),
    ("object-anyof", {"anyOf": _STRING}, {"anyOf": _STRING}),
    ("null-anyof", {"anyOf": None}, {"anyOf": None}),
    ("five-hundred-nulls-then-string", {"anyOf": [*([_NULL] * 500), _STRING]}, _NULLABLE_STRING),
)
_FANOUT_UNIONS: Final = tuple(case for case in _UNIONS if case[0] in {"null-null-string", "null-only", "object-anyof"})


def _text_reply() -> Mapping[str, JsonValue]:
    return {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "OK"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 1, "totalTokenCount": 10},
        "modelVersion": _BACKEND,
    }


def _image_reply() -> Mapping[str, JsonValue]:
    return {
        "candidates": [
            {"content": {"role": "model", "parts": [{"inlineData": {"mimeType": "image/png", "data": _PNG}}]}}
        ],
        "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 1, "totalTokenCount": 10},
    }


def _rejection(message: str) -> Reply:
    error: Final = {"code": 400, "message": message, "status": "INVALID_ARGUMENT"}
    return Reply(status=400, body=json.dumps({"error": error}).encode())


def _google(request: Request) -> Reply:
    if is_gcs_upload(request):
        return gcs_peer(request)
    if _FAIL_MARKER.encode() in request.body:
        return _rejection(f"{_FAIL_MARKER}: the scripted tool schema was refused")
    if _STRING_ANYOF.search(request.body):
        return _rejection("Invalid value at 'tools[0].function_declarations[0].parameters' (anyOf), not a list")
    path: Final = request.target.split("?")[0]
    if path.endswith(":countTokens"):
        return Reply(body=json.dumps({"totalTokens": _COUNT}).encode())
    if path.endswith(":streamGenerateContent"):
        return Reply(content_type="text/event-stream", chunks=(f"data: {json.dumps(_text_reply())}\n\n".encode(),))
    if path.endswith(f"{_IMAGE_BACKEND}:generateContent"):
        return Reply(body=json.dumps(_image_reply()).encode())
    if path.endswith(":generateContent"):
        return Reply(body=json.dumps(_text_reply()).encode())
    return Reply(status=404, body=json.dumps({"error": {"code": 404, "message": request.target}}).encode())


def _deployment(gateway: Gateway, scenario: Scenario, wire: Wire, provider: Provider, backend: str = _BACKEND) -> str:
    if provider == "gemini":
        return scenario.model(model=f"gemini/{backend}", api_base=wire.url, api_key=_GEMINI_KEY)
    return scenario.model(
        model=f"vertex_ai/{backend}",
        api_base=wire.url,
        api_key=None,
        vertex_project=_PROJECT,
        vertex_location=_LOCATION,
        vertex_credentials=service_account_json(_PROJECT, gateway.upstream_url.rstrip("/")),
    )


def _parameters(value: JsonValue) -> dict[str, JsonValue]:
    return {"type": "object", "properties": {"value": value}, "required": ["value"]}


def _declaration(description: str, value: JsonValue) -> dict[str, JsonValue]:
    return {"name": "record", "description": description, "parameters": _parameters(value)}


def _tool(endpoint: Endpoint, description: str, parameters: JsonValue) -> dict[str, JsonValue]:
    if endpoint == "messages":
        return {"name": "record", "description": description, "input_schema": parameters}
    if endpoint == "responses":
        return {"type": "function", "name": "record", "description": description, "parameters": parameters}
    return {"type": "function", "function": {"name": "record", "description": description, "parameters": parameters}}


def _request_body(
    endpoint: Endpoint, model: str, description: str, parameters: JsonValue, stream: bool
) -> tuple[str, dict[str, JsonValue]]:
    prompt: Final = f"Record the value hello by calling the record tool. {description}"
    tools: Final[list[JsonValue]] = [_tool(endpoint, description, parameters)]
    streaming: Final[dict[str, JsonValue]] = {"stream": True} if stream else {}
    if endpoint == "messages":
        messages_body: Final[dict[str, JsonValue]] = {
            "model": model,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": prompt}],
            "tools": tools,
        }
        return "/v1/messages", {**messages_body, **streaming}
    if endpoint == "responses":
        return "/v1/responses", {"model": model, "input": prompt, "tools": tools, **streaming}
    chat_body: Final[dict[str, JsonValue]] = {
        "model": model,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": prompt}],
        "tools": tools,
    }
    return "/v1/chat/completions", {**chat_body, **streaming}


def _body(request: Request) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(request.body)


def _json(response: httpx.Response) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(response.content)


def _generate_requests(wire: Wire) -> tuple[Request, ...]:
    return tuple(
        request
        for request in wire.drain()
        if request.target.split("?")[0].endswith((":generateContent", ":streamGenerateContent"))
    )


def _sent_declarations(body: Mapping[str, JsonValue]) -> JsonValue:
    tools: Final = _JSON_LIST.validate_python(body["tools"])
    assert len(tools) == 1, tools
    tool: Final = object_value(tools[0])
    assert len(tool) == 1, tool
    return next(iter(tool.values()))


def _description() -> str:
    return f"Record a value {uuid.uuid4().hex[:12]}"


def _assert_caller_saw_ok(endpoint: Endpoint, response: httpx.Response, stream: bool) -> None:
    assert response.status_code == 200, response.text
    if stream:
        assert "OK" in response.text, response.text
        return
    payload: Final = _json(response)
    assert string_value(payload["id"]), payload
    if endpoint == "chat":
        choice: Final = object_value(_JSON_LIST.validate_python(payload["choices"])[0])
        assert object_value(choice["message"])["content"] == "OK", payload
        return
    assert "OK" in response.text, payload


@pytest.mark.parametrize("provider", ("vertex_ai", "gemini"))
@pytest.mark.parametrize(("union", "expected"), [case[1:] for case in _UNIONS], ids=[case[0] for case in _UNIONS])
def test_chat_tool_union_reaches_google_with_null_branches_folded_into_nullable(
    gateway: Gateway, provider: Provider, union: JsonValue, expected: JsonValue
) -> None:
    with wire_server(_google) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, provider)
        description: Final = _description()
        path, body = _request_body("chat", model, description, _parameters(union), False)
        response: Final = gateway.request("POST", path, body)
        _assert_caller_saw_ok("chat", response, False)
        requests: Final = _generate_requests(wire)
        assert len(requests) == 1, [request.target for request in requests]
        assert requests[0].target.split("?")[0].endswith(f"{_BACKEND}:generateContent"), requests[0].target
        assert _sent_declarations(_body(requests[0])) == [_declaration(description, expected)]


@pytest.mark.parametrize("stream", (False, True), ids=("whole", "stream"))
@pytest.mark.parametrize("endpoint", ("chat", "messages", "responses"))
@pytest.mark.parametrize("provider", ("vertex_ai", "gemini"))
@pytest.mark.parametrize(
    ("union", "expected"), [case[1:] for case in _FANOUT_UNIONS], ids=[case[0] for case in _FANOUT_UNIONS]
)
def test_every_unified_endpoint_sends_the_same_folded_union(
    gateway: Gateway, provider: Provider, endpoint: Endpoint, stream: bool, union: JsonValue, expected: JsonValue
) -> None:
    with wire_server(_google) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, provider)
        description: Final = _description()
        path, body = _request_body(endpoint, model, description, _parameters(union), stream)
        response: Final = gateway.request("POST", path, body)
        _assert_caller_saw_ok(endpoint, response, stream)
        requests: Final = _generate_requests(wire)
        assert len(requests) == 1, [request.target for request in requests]
        method: Final = "streamGenerateContent" if stream else "generateContent"
        assert requests[0].target.split("?")[0].endswith(f"{_BACKEND}:{method}"), requests[0].target
        assert _sent_declarations(_body(requests[0])) == [_declaration(description, expected)]


def test_nested_item_and_property_unions_are_folded_at_every_depth(gateway: Gateway) -> None:
    parameters: Final[dict[str, JsonValue]] = {
        "type": "object",
        "properties": {
            "values": {"type": "array", "items": {"anyOf": [_NULL, _NULL, _INTEGER]}},
            "nested": {"type": "object", "properties": {"inner": {"anyOf": [_NULL, _NULL, _STRING]}}},
        },
        "required": ["values", "nested"],
    }
    with wire_server(_google) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, "vertex_ai")
        description: Final = _description()
        path, body = _request_body("chat", model, description, parameters, False)
        response: Final = gateway.request("POST", path, body)
        _assert_caller_saw_ok("chat", response, False)
        requests: Final = _generate_requests(wire)
        assert len(requests) == 1, [request.target for request in requests]
        assert _sent_declarations(_body(requests[0])) == [
            {
                "name": "record",
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "values": {"type": "array", "items": {"anyOf": [{"type": "integer", "nullable": True}]}},
                        "nested": {"type": "object", "properties": {"inner": _NULLABLE_STRING}},
                    },
                    "required": ["values", "nested"],
                },
            }
        ]


def test_a_string_anyof_reaches_google_and_its_400_reaches_the_caller(gateway: Gateway) -> None:
    with wire_server(_google) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, "gemini")
        description: Final = _description()
        path, body = _request_body("chat", model, description, _parameters({"anyOf": "string"}), False)
        response: Final = gateway.request("POST", path, body)
        assert response.status_code == 400, response.text
        assert "not a list" in response.text, response.text
        requests: Final = _generate_requests(wire)
        assert len(requests) == 1, [request.target for request in requests]
        assert _sent_declarations(_body(requests[0])) == [_declaration(description, {"anyOf": "string"})]


@pytest.mark.parametrize("provider", ("vertex_ai", "gemini"))
def test_a_google_rejection_of_a_folded_union_reaches_the_caller(gateway: Gateway, provider: Provider) -> None:
    with wire_server(_google) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, provider)
        description: Final = f"{_FAIL_MARKER} {uuid.uuid4().hex[:12]}"
        path, body = _request_body("chat", model, description, _parameters({"anyOf": [_NULL, _NULL, _STRING]}), False)
        response: Final = gateway.request("POST", path, body)
        assert response.status_code == 400, response.text
        assert _FAIL_MARKER in response.text, response.text
        requests: Final = _generate_requests(wire)
        assert len(requests) == 1, [request.target for request in requests]
        assert _sent_declarations(_body(requests[0])) == [_declaration(description, _NULLABLE_STRING)]


@pytest.mark.parametrize(
    "union",
    ({"anyOf": [True, _NULL]}, {"anyOf": [_NULL, 5]}),
    ids=("true-null", "null-int"),
)
def test_a_union_branch_that_is_not_an_object_fails_locally_and_the_next_request_still_works(
    gateway: Gateway, union: JsonValue
) -> None:
    with wire_server(_google) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, "vertex_ai")
        path, broken = _request_body("chat", model, _description(), _parameters(union), False)
        failed: Final = gateway.request("POST", path, broken)
        assert failed.status_code >= 400, failed.text
        assert "error" in _json(failed), failed.text
        assert _generate_requests(wire) == ()
        description: Final = _description()
        _, valid = _request_body("chat", model, description, _parameters({"anyOf": [_NULL, _NULL, _STRING]}), False)
        recovered: Final = gateway.request("POST", path, valid)
        _assert_caller_saw_ok("chat", recovered, False)
        requests: Final = _generate_requests(wire)
        assert len(requests) == 1, [request.target for request in requests]
        assert _sent_declarations(_body(requests[0])) == [_declaration(description, _NULLABLE_STRING)]


def test_a_repeated_identical_request_sends_the_same_folded_union_each_time(gateway: Gateway) -> None:
    with wire_server(_google) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, "vertex_ai")
        description: Final = _description()
        path, body = _request_body("chat", model, description, _parameters({"anyOf": [_NULL, _NULL, _STRING]}), False)
        first: Final = gateway.request("POST", path, body)
        second: Final = gateway.request("POST", path, body)
        _assert_caller_saw_ok("chat", first, False)
        _assert_caller_saw_ok("chat", second, False)
        assert _json(first)["id"] != _json(second)["id"]
        requests: Final = _generate_requests(wire)
        assert len(requests) == 2, [request.target for request in requests]
        assert [_sent_declarations(_body(request)) for request in requests] == [
            [_declaration(description, _NULLABLE_STRING)]
        ] * 2


def test_openai_sdk_chat_sends_the_folded_union(gateway: Gateway) -> None:
    with wire_server(_google) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, "gemini")
        description: Final = _description()
        client: Final = openai.OpenAI(
            base_url=f"{str(gateway.client.base_url).rstrip('/')}/v1", api_key=gateway.key, max_retries=0
        )
        completion: Final = client.chat.completions.create(
            model=model,
            max_tokens=16,
            messages=[{"role": "user", "content": "Record the value hello by calling the record tool."}],
            tools=[_tool("chat", description, _parameters({"anyOf": [_NULL, _NULL, _STRING]}))],  # pyright: ignore[reportArgumentType]  # SDK TypedDicts reject a JsonValue mapping
        )
        assert completion.choices[0].message.content == "OK"
        requests: Final = _generate_requests(wire)
        assert len(requests) == 1, [request.target for request in requests]
        assert _sent_declarations(_body(requests[0])) == [_declaration(description, _NULLABLE_STRING)]


async def _async_openai_stream_text(gateway: Gateway, model: str, description: str) -> str:
    client: Final = openai.AsyncOpenAI(
        base_url=f"{str(gateway.client.base_url).rstrip('/')}/v1", api_key=gateway.key, max_retries=0
    )
    stream: Final = await client.chat.completions.create(
        model=model,
        max_tokens=16,
        stream=True,
        messages=[{"role": "user", "content": "Record the value hello by calling the record tool."}],
        tools=[_tool("chat", description, _parameters({"anyOf": [_NULL]}))],  # pyright: ignore[reportArgumentType]  # SDK TypedDicts reject a JsonValue mapping
    )
    return "".join([chunk.choices[0].delta.content or "" async for chunk in stream if chunk.choices])


def test_async_openai_sdk_stream_sends_the_null_only_union_as_one_nullable_null(gateway: Gateway) -> None:
    with wire_server(_google) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, "vertex_ai")
        description: Final = _description()
        assert asyncio.run(_async_openai_stream_text(gateway, model, description)) == "OK"
        requests: Final = _generate_requests(wire)
        assert len(requests) == 1, [request.target for request in requests]
        assert requests[0].target.split("?")[0].endswith(":streamGenerateContent"), requests[0].target
        assert _sent_declarations(_body(requests[0])) == [_declaration(description, _NULLABLE_NULL)]


def test_anthropic_sdk_messages_sends_the_folded_union(gateway: Gateway) -> None:
    with wire_server(_google) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, "vertex_ai")
        description: Final = _description()
        client: Final = anthropic.Anthropic(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0)
        message: Final = client.messages.create(
            model=model,
            max_tokens=16,
            messages=[{"role": "user", "content": "Record the value hello by calling the record tool."}],
            tools=[_tool("messages", description, _parameters({"anyOf": [_NULL, _NULL]}))],  # pyright: ignore[reportArgumentType]  # SDK TypedDicts reject a JsonValue mapping
        )
        assert [block.type for block in message.content] == ["text"]
        requests: Final = _generate_requests(wire)
        assert len(requests) == 1, [request.target for request in requests]
        assert _sent_declarations(_body(requests[0])) == [_declaration(description, _NULLABLE_NULL)]


@pytest.mark.parametrize(
    ("union", "expected"), [case[1:] for case in _FANOUT_UNIONS], ids=[case[0] for case in _FANOUT_UNIONS]
)
def test_vertex_count_tokens_sends_the_folded_union(gateway: Gateway, union: JsonValue, expected: JsonValue) -> None:
    with wire_server(_google) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, "vertex_ai")
        description: Final = _description()
        response: Final = gateway.request(
            "POST",
            "/v1/messages/count_tokens",
            {
                "model": model,
                "messages": [{"role": "user", "content": "Record the value hello by calling the record tool."}],
                "tools": [_tool("messages", description, _parameters(union))],
            },
        )
        assert response.status_code == 200, response.text
        assert _json(response)["input_tokens"] == _COUNT, response.text
        requests: Final = tuple(request for request in wire.drain() if request.target.endswith(":countTokens"))
        assert len(requests) == 1, [request.target for request in requests]
        assert _sent_declarations(_body(requests[0])) == [_declaration(description, expected)]


@pytest.mark.parametrize(
    ("union", "expected"), [case[1:] for case in _FANOUT_UNIONS], ids=[case[0] for case in _FANOUT_UNIONS]
)
def test_gemini_token_counter_sends_the_folded_union(gateway: Gateway, union: JsonValue, expected: JsonValue) -> None:
    with wire_server(_google) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, "gemini")
        description: Final = _description()
        response: Final = gateway.request(
            "POST",
            "/utils/token_counter",
            {
                "model": model,
                "contents": [
                    {"role": "user", "parts": [{"text": "Record the value hello by calling the record tool."}]}
                ],
                "tools": [_tool("chat", description, _parameters(union))],
            },
            params={"call_endpoint": "true"},
        )
        assert response.status_code == 200, response.text
        assert _json(response)["total_tokens"] == _COUNT, response.text
        requests: Final = tuple(request for request in wire.drain() if ":countTokens" in request.target)
        assert len(requests) == 1, [request.target for request in requests]
        sent: Final = object_value(_body(requests[0])["generateContentRequest"])
        assert _sent_declarations(sent) == [_declaration(description, expected)]


def test_response_format_on_a_model_without_json_schema_support_folds_the_union(gateway: Gateway) -> None:
    with wire_server(_google) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, "gemini", _NO_JSON_SCHEMA_BACKEND)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "Return the value as JSON."}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "record", "schema": _parameters({"anyOf": [_NULL, _NULL, _STRING]})},
                },
            },
        )
        _assert_caller_saw_ok("chat", response, False)
        requests: Final = _generate_requests(wire)
        assert len(requests) == 1, [request.target for request in requests]
        generation_config: Final = object_value(_body(requests[0])["generationConfig"])
        assert generation_config["response_mime_type"] == "application/json", generation_config
        assert generation_config["response_schema"] == {**_parameters(_NULLABLE_STRING), "propertyOrdering": ["value"]}


def test_native_generate_content_response_schema_folds_the_union(gateway: Gateway) -> None:
    with wire_server(_google) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, "gemini", _NO_JSON_SCHEMA_BACKEND)
        response: Final = gateway.request(
            "POST",
            f"/v1beta/models/{model}:generateContent",
            {
                "contents": [{"role": "user", "parts": [{"text": "Return the value as JSON."}]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": _parameters({"anyOf": [_NULL, _NULL, _STRING]}),
                },
            },
        )
        assert response.status_code == 200, response.text
        requests: Final = _generate_requests(wire)
        assert len(requests) == 1, [request.target for request in requests]
        generation_config: Final = object_value(_body(requests[0])["generationConfig"])
        assert generation_config["responseSchema"] == {**_parameters(_NULLABLE_STRING), "propertyOrdering": ["value"]}


@pytest.mark.parametrize(
    ("union", "expected"), [case[1:] for case in _FANOUT_UNIONS], ids=[case[0] for case in _FANOUT_UNIONS]
)
def test_image_generation_tools_send_the_folded_union(gateway: Gateway, union: JsonValue, expected: JsonValue) -> None:
    with wire_server(_google) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(gateway, scenario, wire, "gemini", _IMAGE_BACKEND)
        description: Final = _description()
        response: Final = gateway.request(
            "POST",
            "/v1/images/generations",
            {
                "model": model,
                "prompt": "A small red square on a white background",
                "tools": [_tool("chat", description, _parameters(union))],
            },
        )
        assert response.status_code == 200, response.text
        assert len(_JSON_LIST.validate_python(_json(response)["data"])) == 1, response.text
        requests: Final = _generate_requests(wire)
        assert len(requests) == 1, [request.target for request in requests]
        assert _sent_declarations(_body(requests[0])) == [_declaration(description, expected)]


def test_vertex_batch_row_is_stored_with_the_folded_union(gateway: Gateway) -> None:
    with wire_server(_google) as wire, gateway.scenario() as scenario:
        model: Final = vertex_model(gateway, scenario, wire.url, _BACKEND)
        description: Final = _description()
        row: Final[dict[str, JsonValue]] = {
            "custom_id": "anyof-row",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [{"role": "user", "content": "Record the value hello by calling the record tool."}],
                "tools": [_tool("chat", description, _parameters({"anyOf": [_NULL, _NULL, _STRING]}))],
            },
        }
        response: Final = upload(gateway, "model", model, jsonl(encoded(row)))
        assert response.status_code == 200, response.text
        uploads: Final = tuple(request for request in wire.drain() if is_gcs_upload(request))
        assert len(uploads) == 1, [request.target for request in uploads]
        stored: Final = tuple(
            object_value(JSON_OBJECT.validate_json(line)["request"]) for line in uploads[0].body.splitlines() if line
        )
        assert len(stored) == 1, stored
        assert _sent_declarations(stored[0]) == [_declaration(description, _NULLABLE_STRING)]


async def _event_types_until_done(socket: websockets.ClientConnection) -> AsyncIterator[str]:
    async for message in socket:
        kind: Final = string_value(JSON_OBJECT.validate_json(message)["type"])
        yield kind
        if kind in {"response.done", "error"}:
            return


async def _realtime_session_with_tools(gateway: Gateway, model: str, tools: list[JsonValue]) -> tuple[str, ...]:
    ws_base: Final = str(gateway.client.base_url).rstrip("/").replace("http://", "ws://")
    headers: Final = {"Authorization": f"Bearer {gateway.key}"}
    async with websockets.connect(f"{ws_base}/v1/realtime?model={model}", additional_headers=headers) as socket:
        first: Final = JSON_OBJECT.validate_json(await socket.recv())
        await socket.send(json.dumps({"type": "session.update", "session": {"tools": tools}}))
        item: Final = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "say the line"}]}
        await socket.send(json.dumps({"type": "conversation.item.create", "item": item}))
        events: Final = [kind async for kind in _event_types_until_done(socket)]
        return (string_value(first["type"]), *events)


def _realtime_types(gateway: Gateway, model: str, tools: list[JsonValue]) -> tuple[str, ...]:
    async def bounded() -> tuple[str, ...]:
        try:
            return await asyncio.wait_for(_realtime_session_with_tools(gateway, model, tools), 30)
        except TimeoutError:
            return ("timeout",)

    return asyncio.run(bounded())


def _live_frames(gateway: Gateway, scenario_id: str) -> tuple[dict[str, JsonValue], ...]:
    with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream:
        observed: Final = tuple(
            map(object_value, _JSON_LIST.validate_python(_json(upstream.get("/__observations"))["requests"]))
        )
    return tuple(
        object_value(request["body"])
        for request in observed
        if request["api_key"] == scenario_id
        and request["method"] == "WEBSOCKET_FRAME"
        and request["path"] == GEMINI_LIVE_PATH
    )


@pytest.fixture(scope="module")
def deferred_live(tmp_path_factory: pytest.TempPathFactory) -> Iterator[OwnedProxy]:
    directory: Final = tmp_path_factory.mktemp("anyof-live")
    with gateway_from_environment() as gateway:
        with owned_proxy_process(gateway, directory, {"LITELLM_GEMINI_LIVE_DEFER_SETUP": "true"}, workers=2) as proxy:
            yield proxy


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
def test_deferred_realtime_setup_carries_session_tools_with_the_folded_union(deferred_live: OwnedProxy) -> None:
    gateway: Final = deferred_live.gateway
    with gateway.scenario() as scenario:
        handle: Final = live_scenario(scenario)
        model: Final = live_deployment(scenario, handle.scenario_id, gateway.upstream_url)
        description: Final = _description()
        tools: Final[list[JsonValue]] = [_tool("chat", description, _parameters({"anyOf": [_NULL, _NULL, _STRING]}))]
        types: Final = _realtime_types(gateway, model, tools)
        assert types[0] == "session.created", types
        assert types[-1] == "response.done", types
        frames: Final = _live_frames(gateway, handle.scenario_id)
        setups: Final = tuple(object_value(frame["setup"]) for frame in frames if "setup" in frame)
        assert len(setups) == 1, [sorted(frame) for frame in frames]
        assert _sent_declarations(setups[0]) == [_declaration(description, _NULLABLE_STRING)]


_PROVIDERS: Final[tuple[Provider, ...]] = ("vertex_ai", "gemini")
_ENDPOINTS: Final[tuple[Endpoint, ...]] = ("chat", "messages", "responses")
_BURST: Final[tuple[tuple[Provider, Endpoint, bool, bool], ...]] = tuple(
    (provider, endpoint, stream, index % 4 == 0)
    for index, (provider, endpoint, stream, _) in enumerate(product(_PROVIDERS, _ENDPOINTS, (False, True), range(2)))
)


def test_a_concurrent_mixed_burst_folds_every_union_once_and_keeps_each_status(gateway: Gateway) -> None:
    with wire_server(_google) as wire, gateway.scenario() as scenario:
        models: Final = MappingProxyType(
            {provider: _deployment(gateway, scenario, wire, provider) for provider in _PROVIDERS}
        )
        shots: Final = tuple(
            (provider, endpoint, stream, f"{_FAIL_MARKER if refused else 'burst'} {uuid.uuid4().hex}")
            for provider, endpoint, stream, refused in _BURST
        )

        def send(shot: tuple[Provider, Endpoint, bool, str]) -> httpx.Response:
            provider, endpoint, stream, description = shot
            path, body = _request_body(
                endpoint, models[provider], description, _parameters({"anyOf": [_NULL, _NULL, _STRING]}), stream
            )
            return gateway.request("POST", path, body)

        with ThreadPoolExecutor(max_workers=len(shots)) as pool:
            responses: Final = tuple(pool.map(send, shots))
        expected_statuses: Final = [400 if _FAIL_MARKER in shot[3] else 200 for shot in shots]
        assert [response.status_code for response in responses] == expected_statuses, [
            response.text[:120] for response in responses if response.status_code not in {200, 400}
        ]
        requests: Final = _generate_requests(wire)
        assert len(requests) == len(shots), len(requests)
        sent: Final = sorted((json.dumps(_sent_declarations(_body(request)), sort_keys=True) for request in requests))
        assert sent == sorted(json.dumps([_declaration(shot[3], _NULLABLE_STRING)], sort_keys=True) for shot in shots)
