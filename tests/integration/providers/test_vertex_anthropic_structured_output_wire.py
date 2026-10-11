import asyncio
import functools
import json
import re
import signal
import threading
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from typing import Final

import anthropic
import httpx
import openai
import psutil
import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually, list_value, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

_JSON_TOOL: Final = "json_tool_call"
_USER_TOOL: Final = "lookup"
_STRUCTURED_OUTPUTS_BETA: Final = "structured-outputs-2025-11-13"
_TOOL_MODEL: Final = "claude-sonnet-5"
_EFFORT_GATED_MODEL: Final = "claude-haiku-4-5"
_NATIVE_MODEL: Final = "claude-opus-5-5"
_UNMAPPED_MODEL: Final = "claude-scripted-9"
_BEDROCK_INVOKE_MODEL: Final = "anthropic.claude-sonnet-5"
_PROJECT: Final = "scripted-project"
_LOCATION: Final = "us-east5"
_VERTEX_MODELS: Final = f"/v1/projects/{_PROJECT}/locations/{_LOCATION}/publishers/anthropic/models/"
_VERTEX_STREAM_SUFFIX: Final = ":streamRawPredict?alt=sse"
_ANTHROPIC_KEY: Final = "synthetic-anthropic-key"
_AZURE_KEY: Final = "synthetic-azure-ai-key"
_BEDROCK_TOKEN: Final = "synthetic-bedrock-bearer"
_PLATFORM_KEY: Final = "synthetic-platform-key"
_OWNED_MODEL: Final = "partner-claude"
_DEPRECATED_OUTPUT_FORMAT: Final = "output_format: this field is deprecated, use output_config.format"
_BAD_OUTPUT_CONFIG: Final = "output_config: Input should be a valid dictionary"
_BAD_FORMAT: Final = "output_config.format: Input should be a json_schema object"
_SCHEMA: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"tier": {"type": "integer"}, "summary": {"type": "string"}},
    "required": ["tier", "summary"],
    "additionalProperties": False,
}
_RESPONSE_FORMAT: Final[dict[str, JsonValue]] = {
    "type": "json_schema",
    "json_schema": {"name": "risk", "schema": _SCHEMA},
}
_TEXT_FORMAT: Final[dict[str, JsonValue]] = {
    "format": {"type": "json_schema", "name": "risk", "schema": _SCHEMA},
}
_OUTPUT_FORMAT: Final[dict[str, JsonValue]] = {"type": "json_schema", "schema": _SCHEMA}
_USER_TOOL_SPEC: Final[dict[str, JsonValue]] = {
    "type": "function",
    "function": {
        "name": _USER_TOOL,
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
    },
}
_MARKER: Final = re.compile(r"\b([a-z_]+) ([0-9a-f]{32})\b")
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_OWNED_PROXY_CELL_SECONDS: Final = 2 * graceful_stop_seconds() + 120
_UNWRAPPED_SHAPES: Final = frozenset({"plain", "wrapped", "named", "values"})
_BURST_SHAPES: Final = ("wrapped", "plain", "named", "values")


@dataclass(frozen=True, slots=True)
class _Delivery:
    target: str
    beta: str | None
    body: dict[str, JsonValue]


def _payload(marker_hex: str) -> dict[str, JsonValue]:
    return {"tier": 2, "summary": marker_hex}


def _nest(payload: JsonValue, depth: int) -> JsonValue:
    return functools.reduce(lambda inner, _: {"parameters": inner}, range(depth), payload)


def _tool_input(shape: str, marker_hex: str) -> JsonValue:
    payload: Final = _payload(marker_hex)
    match shape:
        case "plain":
            return payload
        case "wrapped":
            return {"tool_name": _JSON_TOOL, "parameters": payload}
        case "named":
            return {"name": _JSON_TOOL, "parameters": payload}
        case "values":
            return {"values": payload}
        case "large":
            return {"tool_name": _JSON_TOOL, "parameters": {**payload, "summary": marker_hex + "x" * 5000}}
        case "two_candidates":
            return {"parameters": payload, "alternate": {**payload, "tier": 3}}
        case "inner_incomplete":
            return {"parameters": {"tier": 2}}
        case "self_shaped":
            return {**payload, "nested": payload}
        case "scalar":
            return marker_hex
        case "empty":
            return {}
        case "list_input":
            return [payload]
        case "deep":
            return _nest(payload, 60)
    raise AssertionError(shape)


def _expected(shape: str, marker_hex: str) -> JsonValue:
    match shape:
        case "plain" | "wrapped" | "named" | "values":
            return _payload(marker_hex)
        case "large":
            return {**_payload(marker_hex), "summary": marker_hex + "x" * 5000}
    return _tool_input(shape, marker_hex)


def _message(message_id: str, content: list[JsonValue], stop_reason: str) -> dict[str, JsonValue]:
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": _TOOL_MODEL,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }


def _sse(message: dict[str, JsonValue], block: dict[str, JsonValue], delta: dict[str, JsonValue]) -> tuple[bytes, ...]:
    events: Final[tuple[tuple[str, dict[str, JsonValue]], ...]] = (
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    **message,
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 5, "output_tokens": 1},
                },
            },
        ),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": block}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": delta}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
                "usage": {"output_tokens": 3},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    )
    return tuple(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode() for name, data in events)


def _error(status: int, message: str) -> Reply:
    body: Final[dict[str, JsonValue]] = {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": message},
    }
    return Reply(status=status, body=json.dumps(body).encode())


def _format_fault(body: Mapping[str, JsonValue], vertex: bool) -> str | None:
    if vertex and "output_format" in body:
        return _DEPRECATED_OUTPUT_FORMAT
    output_config: Final = body.get("output_config")
    if output_config is None:
        return None
    if not isinstance(output_config, dict):
        return _BAD_OUTPUT_CONFIG
    output_format: Final = output_config.get("format")
    if output_format is None:
        return None
    if not isinstance(output_format, dict) or output_format.get("type") != "json_schema":
        return _BAD_FORMAT
    return None if isinstance(output_format.get("schema"), dict) else _BAD_FORMAT


def _marker_of(body: Mapping[str, JsonValue]) -> tuple[str, str]:
    found: Final = _MARKER.search(json.dumps(body.get("messages")))
    assert found is not None, body
    return found.group(1), found.group(2)


def _tool_names(body: Mapping[str, JsonValue]) -> frozenset[str]:
    return frozenset(string_value(object_value(tool)["name"]) for tool in list_value(body.get("tools", [])))


def _tool_use(block_id: str, name: str, tool_input: JsonValue) -> dict[str, JsonValue]:
    return {"type": "tool_use", "id": block_id, "name": name, "input": tool_input}


def _tool_reply(message_id: str, shape: str, marker_hex: str, user_tool: bool, stream: bool) -> Reply:
    tool_input: Final = _tool_input(shape, marker_hex)
    json_block: Final = _tool_use(f"toolu_{marker_hex}", _JSON_TOOL, tool_input)
    user_blocks: Final[list[JsonValue]] = (
        [_tool_use(f"toolu_user_{marker_hex}", _USER_TOOL, {"q": marker_hex})] if user_tool else []
    )
    message: Final = _message(message_id, [*user_blocks, json_block], "tool_use")
    if not stream:
        return Reply(body=json.dumps(message).encode())
    return Reply(
        content_type="text/event-stream",
        chunks=_sse(
            message, {**json_block, "input": {}}, {"type": "input_json_delta", "partial_json": json.dumps(tool_input)}
        ),
    )


def _text_reply(message_id: str, text: str, stream: bool) -> Reply:
    message: Final = _message(message_id, [{"type": "text", "text": text}], "end_turn")
    if not stream:
        return Reply(body=json.dumps(message).encode())
    return Reply(
        content_type="text/event-stream",
        chunks=_sse(message, {"type": "text", "text": ""}, {"type": "text_delta", "text": text}),
    )


def _scripted_target(request: Request) -> bool:
    return (
        request.target.startswith(_VERTEX_MODELS)
        or request.target.endswith("/v1/messages")
        or request.target.endswith("/invoke")
    )


def _structured(body: Mapping[str, JsonValue]) -> bool:
    output_config: Final = body.get("output_config")
    return "output_format" in body or (isinstance(output_config, dict) and "format" in output_config)


def _peer(request: Request) -> Reply:
    vertex: Final = request.target.startswith(_VERTEX_MODELS)
    if vertex:
        assert request.headers["authorization"] == "Bearer scripted-token", request.headers
    elif not _scripted_target(request):
        return _error(404, f"unscripted target {request.target}")
    body: Final = _JSON_OBJECT.validate_json(request.body)
    fault: Final = _format_fault(body, vertex)
    if fault is not None:
        return _error(400, fault)
    shape, marker_hex = _marker_of(body)
    stream: Final = request.target.endswith(_VERTEX_STREAM_SUFFIX) or body.get("stream") is True
    tools: Final = _tool_names(body)
    message_id: Final = f"msg_{marker_hex}"
    if _JSON_TOOL in tools:
        return _tool_reply(message_id, shape, marker_hex, _USER_TOOL in tools, stream)
    if _structured(body):
        return _text_reply(message_id, json.dumps(_payload(marker_hex)), stream)
    return _text_reply(message_id, f"plain text {marker_hex}", stream)


def _failing(status: int) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        _, marker_hex = _marker_of(_JSON_OBJECT.validate_json(request.body))
        return _error(status, f"scripted {status} {marker_hex}")

    return respond


def _holding(held: SimpleQueue[str], release: threading.Event) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        held.put(request.target)
        assert release.wait(timeout=60), "Held request was never released"
        return _peer(request)

    return respond


def _delivered(wire: Wire) -> tuple[_Delivery, ...]:
    return tuple(
        _Delivery(request.target, request.headers.get("anthropic-beta"), _JSON_OBJECT.validate_json(request.body))
        for request in wire.drain()
    )


def _one_delivery(wire: Wire, marker_hex: str) -> _Delivery:
    (sent,) = _delivered(wire)
    assert marker_hex in json.dumps(sent.body), sent.body
    return sent


def _json_tool_schema(delivery: _Delivery) -> dict[str, JsonValue]:
    tools: Final = tuple(object_value(tool) for tool in list_value(delivery.body["tools"]))
    (json_tool,) = tuple(tool for tool in tools if tool["name"] == _JSON_TOOL)
    return object_value(json_tool["input_schema"])


def _folded_format(delivery: _Delivery) -> dict[str, JsonValue]:
    assert "output_format" not in delivery.body, delivery.body
    return object_value(object_value(delivery.body["output_config"])["format"])


def _marker(shape: str) -> tuple[str, str]:
    marker_hex: Final = uuid.uuid4().hex
    return f"{shape} {marker_hex}", marker_hex


def _chat_body(
    model: str,
    marker: str,
    *,
    stream: bool = False,
    response_format: JsonValue = _RESPONSE_FORMAT,
    tools: JsonValue = None,
) -> dict[str, JsonValue]:
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "messages": [{"role": "user", "content": marker}],
        "stream": stream,
        "response_format": response_format,
    }
    return body if tools is None else {**body, "tools": tools}


def _messages_body(model: str, marker: str, *, stream: bool = False, **extra: JsonValue) -> dict[str, JsonValue]:
    return {
        "model": model,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": marker}],
        "stream": stream,
        **extra,
    }


def _responses_body(model: str, marker: str) -> dict[str, JsonValue]:
    return {"model": model, "input": marker, "text": _TEXT_FORMAT}


def _chat_message(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 200, response.text
    choice: Final = object_value(list_value(_JSON_OBJECT.validate_json(response.content)["choices"])[0])
    assert choice["finish_reason"] == "stop", response.text
    return object_value(choice["message"])


def _chat_content(response: httpx.Response) -> JsonValue:
    return json.loads(string_value(_chat_message(response)["content"]))


def _response_id(response: httpx.Response) -> str:
    return string_value(_JSON_OBJECT.validate_json(response.content)["id"])


def _delta_content(frame: str) -> str:
    choices: Final = list_value(_JSON_OBJECT.validate_json(frame).get("choices", []))
    if not choices:
        return ""
    content: Final = object_value(object_value(choices[0])["delta"]).get("content")
    return "" if content is None else string_value(content)


def _stream_content(response: httpx.Response) -> JsonValue:
    assert response.status_code == 200, response.text
    assert response.text.rstrip().endswith("data: [DONE]"), response.text
    frames: Final = tuple(
        line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: {")
    )
    return json.loads("".join(_delta_content(frame) for frame in frames))


def _responses_text(response: httpx.Response) -> JsonValue:
    assert response.status_code == 200, response.text
    items: Final = tuple(
        object_value(item) for item in list_value(_JSON_OBJECT.validate_json(response.content)["output"])
    )
    (message,) = tuple(item for item in items if item["type"] == "message")
    return json.loads(string_value(object_value(list_value(message["content"])[0])["text"]))


def _text_block_json(message: Mapping[str, JsonValue]) -> JsonValue:
    block: Final = object_value(list_value(message["content"])[0])
    return json.loads(string_value(block["text"]))


def _messages_text(response: httpx.Response) -> JsonValue:
    assert response.status_code == 200, response.text
    return _text_block_json(_JSON_OBJECT.validate_json(response.content))


def _messages_stream_text(response: httpx.Response) -> JsonValue:
    assert response.status_code == 200, response.text
    assert "event: message_stop" in response.text, response.text
    frames: Final = tuple(
        _JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: {")
    )
    deltas: Final = tuple(object_value(frame["delta"]) for frame in frames if frame["type"] == "content_block_delta")
    return json.loads("".join(string_value(delta["text"]) for delta in deltas if delta["type"] == "text_delta"))


def _vertex_deployment(gateway: Gateway, scenario: Scenario, api_base: str, backend: str) -> str:
    return scenario.model(
        model=f"vertex_ai/{backend}",
        api_base=api_base,
        api_key=None,
        vertex_project=_PROJECT,
        vertex_location=_LOCATION,
        vertex_credentials=service_account_json(_PROJECT, gateway.upstream_url.rstrip("/")),
    )


def _family_deployment(gateway: Gateway, scenario: Scenario, api_base: str, member: str) -> str:
    match member:
        case "anthropic_unmapped":
            return scenario.model(model=f"anthropic/{_UNMAPPED_MODEL}", api_base=api_base, api_key=_ANTHROPIC_KEY)
        case "azure_ai_unmapped":
            return scenario.model(model=f"azure_ai/{_UNMAPPED_MODEL}", api_base=api_base, api_key=_AZURE_KEY)
        case "bedrock_invoke":
            return scenario.model(
                model=f"bedrock/invoke/{_BEDROCK_INVOKE_MODEL}",
                api_base=api_base,
                api_key=_BEDROCK_TOKEN,
                aws_region_name="us-east-1",
            )
        case "claude_platform_unmapped":
            return scenario.model(
                model=f"bedrock/claude_platform/{_UNMAPPED_MODEL}",
                api_base=api_base,
                api_key=_PLATFORM_KEY,
                aws_region_name="us-east-1",
                workspace_id="scripted-workspace",
            )
        case "vertex_haiku":
            return _vertex_deployment(gateway, scenario, api_base, _EFFORT_GATED_MODEL)
    raise AssertionError(member)


def _owned_config(path: Path, gateway: Gateway, api_base: str) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    path.write_text(
        yaml.safe_dump(
            {
                **config,
                "model_list": [
                    {
                        "model_name": _OWNED_MODEL,
                        "litellm_params": {
                            "model": f"vertex_ai/{_TOOL_MODEL}",
                            "api_base": api_base,
                            "vertex_project": _PROJECT,
                            "vertex_location": _LOCATION,
                            "vertex_credentials": service_account_json(_PROJECT, gateway.upstream_url.rstrip("/")),
                        },
                    }
                ],
            }
        )
    )
    return path


def _clients(stack: ExitStack, base_url: str, count: int) -> tuple[httpx.Client, ...]:
    return tuple(
        stack.enter_context(httpx.Client(base_url=base_url, timeout=60, trust_env=False)) for _ in range(count)
    )


def _post(client: httpx.Client, key: str, path: str, body: Mapping[str, JsonValue]) -> httpx.Response:
    return client.post(path, json=dict(body), headers={"Authorization": f"Bearer {key}"})


def _post_or_dropped(client: httpx.Client, key: str, path: str, body: Mapping[str, JsonValue]) -> httpx.Response | None:
    try:
        return _post(client, key, path, body)
    except httpx.TransportError:
        return None


def _local_port(client: httpx.Client) -> int:
    with client.stream("GET", "/health/liveliness") as response:
        port: Final = int(response.extensions["network_stream"].get_extra_info("client_addr")[1])
        response.read()
    assert response.status_code == 200, response.text
    return port


def _accepted_client_ports(pid: int, proxy_port: int) -> frozenset[int]:
    return frozenset(
        connection.raddr.port
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.raddr and connection.laddr.port == proxy_port
    )


def _seen_once(delivered: tuple[_Delivery, ...], marker_hexes: tuple[str, ...]) -> None:
    bodies: Final = tuple(json.dumps(sent.body) for sent in delivered)
    seen: Final = tuple(sum(marker_hex in body for body in bodies) for marker_hex in marker_hexes)
    assert seen == (1,) * len(marker_hexes), seen


def _spend_rows(request_id: str) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows('SELECT request_id, model FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (request_id,)),
        lambda rows: len(rows) == 1,
        seconds=70,
    )


def _openai_client(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(
        base_url=f"{gateway.client.base_url}/v1",
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False, timeout=60),
    )


def _anthropic_client(gateway: Gateway) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        base_url=str(gateway.client.base_url),
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False, timeout=60),
    )


@pytest.mark.parametrize(
    "shape",
    [
        "plain",
        "wrapped",
        "named",
        "values",
        "large",
        "two_candidates",
        "inner_incomplete",
        "self_shaped",
        "scalar",
        "empty",
        "list_input",
        "deep",
    ],
)
def test_vertex_chat_json_schema_content_is_the_schema_object(gateway: Gateway, shape: str) -> None:
    marker, marker_hex = _marker(shape)
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker))
        assert _chat_content(response) == _expected(shape, marker_hex), response.text
        assert "tool_calls" not in _chat_message(response) or _chat_message(response)["tool_calls"] is None
        sent: Final = _one_delivery(wire, marker_hex)
        assert sent.target == f"{_VERTEX_MODELS}{_TOOL_MODEL}:rawPredict", sent.target
        assert _json_tool_schema(sent)["properties"] == _SCHEMA["properties"], sent.body
        assert "output_config" not in sent.body and "output_format" not in sent.body, sent.body


def test_vertex_chat_through_the_openai_sdk_unwraps_the_envelope(gateway: Gateway) -> None:
    marker, marker_hex = _marker("wrapped")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
        with _openai_client(gateway) as client:
            reply: Final = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": marker}],
                response_format={"type": "json_schema", "json_schema": {"name": "risk", "schema": _SCHEMA}},
            )
        assert json.loads(reply.choices[0].message.content or "") == _payload(marker_hex), reply
        _one_delivery(wire, marker_hex)


def test_vertex_chat_through_the_async_openai_sdk_unwraps_the_envelope(gateway: Gateway) -> None:
    marker, marker_hex = _marker("wrapped")

    async def generate(model: str) -> str | None:
        async with openai.AsyncOpenAI(
            base_url=f"{gateway.client.base_url}/v1",
            api_key=gateway.key,
            max_retries=0,
            http_client=httpx.AsyncClient(trust_env=False, timeout=60),
        ) as client:
            reply: Final = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": marker}],
                response_format={"type": "json_schema", "json_schema": {"name": "risk", "schema": _SCHEMA}},
            )
        return reply.choices[0].message.content

    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
        content: Final = asyncio.run(generate(model))
        assert json.loads(content or "") == _payload(marker_hex), content
        _one_delivery(wire, marker_hex)


def test_vertex_responses_json_schema_output_text_is_the_schema_object(gateway: Gateway) -> None:
    marker, marker_hex = _marker("wrapped")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
        response: Final = gateway.request("POST", "/v1/responses", _responses_body(model, marker))
        assert _responses_text(response) == _payload(marker_hex), response.text
        sent: Final = _one_delivery(wire, marker_hex)
        assert _json_tool_schema(sent)["properties"] == _SCHEMA["properties"], sent.body


def test_vertex_chat_stream_json_schema_unwraps_the_envelope_when_the_block_ends(gateway: Gateway) -> None:
    marker, marker_hex = _marker("wrapped")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker, stream=True))
        assert _stream_content(response) == _payload(marker_hex), response.text
        sent: Final = _one_delivery(wire, marker_hex)
        assert sent.target == f"{_VERTEX_MODELS}{_TOOL_MODEL}{_VERTEX_STREAM_SUFFIX}", sent.target


def test_vertex_chat_json_schema_next_to_a_user_tool_unwraps_the_envelope_and_keeps_the_tool_call(
    gateway: Gateway,
) -> None:
    marker, marker_hex = _marker("wrapped")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", _chat_body(model, marker, tools=[_USER_TOOL_SPEC])
        )
        assert response.status_code == 200, response.text
        message: Final = object_value(
            object_value(list_value(_JSON_OBJECT.validate_json(response.content)["choices"])[0])["message"]
        )
        assert json.loads(string_value(message["content"])) == _payload(marker_hex), response.text
        (tool_call,) = tuple(object_value(call) for call in list_value(message["tool_calls"]))
        function: Final = object_value(tool_call["function"])
        assert function["name"] == _USER_TOOL, response.text
        assert json.loads(string_value(function["arguments"])) == {"q": marker_hex}, response.text
        sent: Final = _one_delivery(wire, marker_hex)
        assert _tool_names(sent.body) == frozenset({_USER_TOOL, _JSON_TOOL}), sent.body


@pytest.mark.parametrize(
    "member",
    ["anthropic_unmapped", "azure_ai_unmapped", "bedrock_invoke", "claude_platform_unmapped", "vertex_haiku"],
)
def test_tool_path_family_unwraps_the_envelope(gateway: Gateway, member: str) -> None:
    marker, marker_hex = _marker("wrapped")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _family_deployment(gateway, scenario, wire.url, member)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker))
        assert _chat_content(response) == _payload(marker_hex), response.text
        sent: Final = _one_delivery(wire, marker_hex)
        assert _json_tool_schema(sent)["properties"] == _SCHEMA["properties"], sent.body


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_vertex_chat_json_schema_native_folds_output_format_into_output_config(gateway: Gateway, stream: bool) -> None:
    marker, marker_hex = _marker("native")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _NATIVE_MODEL)
        response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker, stream=stream))
        content: Final = _stream_content(response) if stream else _chat_content(response)
        assert content == _payload(marker_hex), response.text
        sent: Final = _one_delivery(wire, marker_hex)
        assert _folded_format(sent) == _OUTPUT_FORMAT, sent.body
        assert "tools" not in sent.body, sent.body


def test_vertex_chat_native_through_the_openai_sdk_folds_output_format(gateway: Gateway) -> None:
    marker, marker_hex = _marker("native")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _NATIVE_MODEL)
        with _openai_client(gateway) as client:
            reply: Final = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": marker}],
                response_format={"type": "json_schema", "json_schema": {"name": "risk", "schema": _SCHEMA}},
            )
        assert json.loads(reply.choices[0].message.content or "") == _payload(marker_hex), reply
        assert _folded_format(_one_delivery(wire, marker_hex)) == _OUTPUT_FORMAT


def test_vertex_responses_native_folds_output_format(gateway: Gateway) -> None:
    marker, marker_hex = _marker("native")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _NATIVE_MODEL)
        response: Final = gateway.request("POST", "/v1/responses", _responses_body(model, marker))
        assert _responses_text(response) == _payload(marker_hex), response.text
        assert _folded_format(_one_delivery(wire, marker_hex)) == _OUTPUT_FORMAT


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_vertex_messages_folds_output_format_into_output_config(gateway: Gateway, stream: bool) -> None:
    marker, marker_hex = _marker("native")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _NATIVE_MODEL)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            _messages_body(model, marker, stream=stream, output_format=_OUTPUT_FORMAT),
            headers={"anthropic-beta": _STRUCTURED_OUTPUTS_BETA},
        )
        text: Final = _messages_stream_text(response) if stream else _messages_text(response)
        assert text == _payload(marker_hex), response.text
        assert f"msg_{marker_hex}" in response.text, response.text
        sent: Final = _one_delivery(wire, marker_hex)
        assert _folded_format(sent) == _OUTPUT_FORMAT, sent.body
        assert sent.target.endswith(_VERTEX_STREAM_SUFFIX if stream else ":rawPredict"), sent.target


def test_vertex_messages_through_the_anthropic_sdk_folds_output_format(gateway: Gateway) -> None:
    marker, marker_hex = _marker("native")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _NATIVE_MODEL)
        with _anthropic_client(gateway) as client:
            reply: Final = client.beta.messages.create(
                model=model,
                max_tokens=64,
                messages=[{"role": "user", "content": marker}],
                output_format={"type": "json_schema", "schema": _SCHEMA},
                betas=[_STRUCTURED_OUTPUTS_BETA],
            )
        assert reply.id == f"msg_{marker_hex}", reply
        assert _text_block_json(_JSON_OBJECT.validate_json(reply.model_dump_json())) == _payload(marker_hex), reply
        assert _folded_format(_one_delivery(wire, marker_hex)) == _OUTPUT_FORMAT


def test_vertex_messages_through_the_async_anthropic_sdk_folds_output_format(gateway: Gateway) -> None:
    marker, marker_hex = _marker("native")

    async def generate(model: str) -> tuple[str, str]:
        async with anthropic.AsyncAnthropic(
            base_url=str(gateway.client.base_url),
            api_key=gateway.key,
            max_retries=0,
            http_client=httpx.AsyncClient(trust_env=False, timeout=60),
        ) as client:
            reply: Final = await client.beta.messages.create(
                model=model,
                max_tokens=64,
                messages=[{"role": "user", "content": marker}],
                output_format={"type": "json_schema", "schema": _SCHEMA},
                betas=[_STRUCTURED_OUTPUTS_BETA],
            )
        return reply.id, reply.model_dump_json()

    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _NATIVE_MODEL)
        reply_id, dumped = asyncio.run(generate(model))
        assert reply_id == f"msg_{marker_hex}", dumped
        assert _text_block_json(_JSON_OBJECT.validate_json(dumped)) == _payload(marker_hex), dumped
        assert _folded_format(_one_delivery(wire, marker_hex)) == _OUTPUT_FORMAT


def test_vertex_messages_keeps_the_callers_output_config_format_over_output_format(gateway: Gateway) -> None:
    marker, marker_hex = _marker("native")
    callers_format: Final[dict[str, JsonValue]] = {
        "type": "json_schema",
        "schema": {**_SCHEMA, "title": "callers"},
    }
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _NATIVE_MODEL)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            _messages_body(model, marker, output_format=_OUTPUT_FORMAT, output_config={"format": callers_format}),
        )
        assert _messages_text(response) == _payload(marker_hex), response.text
        assert _folded_format(_one_delivery(wire, marker_hex)) == callers_format


def test_vertex_messages_replaces_a_non_dict_output_config_with_the_folded_format(gateway: Gateway) -> None:
    marker, marker_hex = _marker("native")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _NATIVE_MODEL)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            _messages_body(model, marker, output_format=_OUTPUT_FORMAT, output_config="bogus"),
        )
        assert _messages_text(response) == _payload(marker_hex), response.text
        sent: Final = _one_delivery(wire, marker_hex)
        assert sent.body["output_config"] == {"format": _OUTPUT_FORMAT}, sent.body


def test_vertex_messages_drops_effort_but_keeps_the_folded_format_on_haiku(gateway: Gateway) -> None:
    marker, marker_hex = _marker("native")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _EFFORT_GATED_MODEL)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            _messages_body(model, marker, output_format=_OUTPUT_FORMAT, output_config={"effort": "high"}),
        )
        assert _messages_text(response) == _payload(marker_hex), response.text
        sent: Final = _one_delivery(wire, marker_hex)
        assert sent.body["output_config"] == {"format": _OUTPUT_FORMAT}, sent.body


def test_vertex_messages_forwards_an_explicit_output_config_format_unchanged(gateway: Gateway) -> None:
    marker, marker_hex = _marker("native")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _NATIVE_MODEL)
        response: Final = gateway.request(
            "POST", "/v1/messages", _messages_body(model, marker, output_config={"format": _OUTPUT_FORMAT})
        )
        assert _messages_text(response) == _payload(marker_hex), response.text
        sent: Final = _one_delivery(wire, marker_hex)
        assert sent.body["output_config"] == {"format": _OUTPUT_FORMAT}, sent.body


def test_anthropic_messages_forwards_top_level_output_format_verbatim(gateway: Gateway) -> None:
    marker, marker_hex = _marker("native")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_NATIVE_MODEL}", api_base=wire.url, api_key=_ANTHROPIC_KEY)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            _messages_body(model, marker, output_format=_OUTPUT_FORMAT),
            headers={"anthropic-beta": _STRUCTURED_OUTPUTS_BETA},
        )
        assert _messages_text(response) == _payload(marker_hex), response.text
        sent: Final = _one_delivery(wire, marker_hex)
        assert sent.target == "/v1/messages", sent.target
        assert sent.body["output_format"] == _OUTPUT_FORMAT, sent.body
        assert "output_config" not in sent.body, sent.body


def test_vertex_chat_without_response_format_sends_no_tool_and_no_output_config(gateway: Gateway) -> None:
    marker, marker_hex = _marker("plain_text")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": marker}]}
        )
        assert response.status_code == 200, response.text
        assert f"plain text {marker_hex}" in response.text, response.text
        sent: Final = _one_delivery(wire, marker_hex)
        assert not {"tools", "output_config", "output_format"} & sent.body.keys(), sent.body


@pytest.mark.parametrize(
    "output_format",
    [5, [], {}, "x" * 5000],
    ids=["int", "list", "empty", "5kb"],
)
def test_vertex_messages_malformed_output_format_reaches_the_caller_as_the_peer_400(
    gateway: Gateway, output_format: JsonValue
) -> None:
    marker, marker_hex = _marker("native")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _NATIVE_MODEL)
        response: Final = gateway.request(
            "POST", "/v1/messages", _messages_body(model, marker, output_format=output_format)
        )
        assert response.status_code == 400, response.text
        assert _BAD_FORMAT in response.text, response.text
        sent: Final = _one_delivery(wire, marker_hex)
        assert "output_format" not in sent.body, sent.body
        assert object_value(sent.body["output_config"])["format"] == output_format, sent.body
        assert gateway.request("GET", "/health/liveliness").status_code == 200


def test_vertex_messages_duplicate_output_format_keys_fold_the_last_value(gateway: Gateway) -> None:
    marker, marker_hex = _marker("native")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _NATIVE_MODEL)
        raw: Final = (
            json.dumps(_messages_body(model, marker, output_format=5))[:-1]
            + f', "output_format": {json.dumps(_OUTPUT_FORMAT)}}}'
        )
        response: Final = gateway.client.post(
            "/v1/messages",
            content=raw,
            headers={"Authorization": f"Bearer {gateway.key}", "Content-Type": "application/json"},
        )
        assert _messages_text(response) == _payload(marker_hex), response.text
        assert _folded_format(_one_delivery(wire, marker_hex)) == _OUTPUT_FORMAT


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "properties": _SCHEMA["properties"], "required": "tier"},
        {"type": "object", "properties": ["tier", "summary"]},
        {},
    ],
    ids=["required_string", "properties_list", "empty_schema"],
)
def test_vertex_chat_malformed_response_format_schema_keeps_the_tool_input(gateway: Gateway, schema: JsonValue) -> None:
    marker, marker_hex = _marker("wrapped")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            _chat_body(
                model,
                marker,
                response_format={"type": "json_schema", "json_schema": {"name": "risk", "schema": schema}},
            ),
        )
        assert _chat_content(response) == _tool_input("wrapped", marker_hex), response.text
        _one_delivery(wire, marker_hex)
        assert gateway.request("GET", "/health/liveliness").status_code == 200


@pytest.mark.parametrize("status", [400, 403, 404])
def test_vertex_chat_peer_error_reaches_the_caller_and_the_deployment_keeps_serving(
    gateway: Gateway, status: int
) -> None:
    marker, marker_hex = _marker("wrapped")
    after, after_hex = _marker("wrapped")
    with gateway.scenario() as scenario:
        with wire_server(_failing(status)) as failing:
            port: Final = int(failing.url.rsplit(":", 1)[1])
            model: Final = _vertex_deployment(gateway, scenario, failing.url, _TOOL_MODEL)
            response: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker))
            assert response.status_code == status, response.text
            assert f"scripted {status} {marker_hex}" in response.text, response.text
            _one_delivery(failing, marker_hex)
        with wire_server(_peer, port=port) as wire:
            recovered: Final = gateway.request("POST", "/v1/chat/completions", _chat_body(model, after))
            assert _chat_content(recovered) == _payload(after_hex), recovered.text
            _one_delivery(wire, after_hex)


def test_vertex_chat_unauthenticated_request_never_reaches_the_peer(gateway: Gateway) -> None:
    marker, _ = _marker("wrapped")
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", _chat_body(model, marker), key="sk-not-a-key-this-proxy-issued"
        )
        assert response.status_code == 401, response.text
        assert wire.drain() == ()


def test_vertex_chat_repeated_requests_each_reach_the_peer_and_log_spend_once(gateway: Gateway) -> None:
    markers: Final = tuple(_marker("wrapped") for _ in range(2))
    with wire_server(_peer) as wire, gateway.scenario() as scenario:
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
        responses: Final = tuple(
            gateway.request("POST", "/v1/chat/completions", _chat_body(model, marker)) for marker, _ in markers
        )
        for response, (_, marker_hex) in zip(responses, markers, strict=True):
            assert _chat_content(response) == _payload(marker_hex), response.text
        delivered: Final = _delivered(wire)
        assert len(delivered) == 2, delivered
        _seen_once(delivered, tuple(marker_hex for _, marker_hex in markers))
        request_ids: Final = tuple(_response_id(response) for response in responses)
        assert len(set(request_ids)) == 2, request_ids
        for request_id in request_ids:
            assert _spend_rows(request_id)[0]["request_id"] == request_id


def test_concurrent_burst_across_both_workers_unwraps_every_envelope(gateway: Gateway) -> None:
    with ExitStack() as stack:
        wire: Final = stack.enter_context(wire_server(_peer))
        scenario: Final = stack.enter_context(gateway.scenario())
        model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
        shapes: Final = tuple(_BURST_SHAPES[index % len(_BURST_SHAPES)] for index in range(12))
        markers: Final = tuple(_marker(shape) for shape in shapes)
        streams: Final = tuple(index % 6 == 5 for index in range(len(markers)))
        bodies: Final = tuple(
            _chat_body(model, marker, stream=stream) for (marker, _), stream in zip(markers, streams, strict=True)
        )
        clients: Final = _clients(stack, str(gateway.client.base_url), len(bodies))
        pool: Final = stack.enter_context(ThreadPoolExecutor(max_workers=len(bodies)))
        responses: Final = tuple(
            pool.map(
                lambda pair: _post(pair[0], gateway.key, "/v1/chat/completions", pair[1]),
                zip(clients, bodies, strict=True),
            )
        )
        for response, shape, (_, marker_hex), stream in zip(responses, shapes, markers, streams, strict=True):
            observed: Final = _stream_content(response) if stream else _chat_content(response)
            expected: Final = _expected(shape, marker_hex)
            assert observed == expected, (shape, stream, response.text)
        _seen_once(_delivered(wire), tuple(marker_hex for _, marker_hex in markers))


def test_upstream_outage_mid_run_recovers_with_envelopes_still_unwrapped(gateway: Gateway) -> None:
    with ExitStack() as stack:
        clients: Final = _clients(stack, str(gateway.client.base_url), 8)
        pool: Final = stack.enter_context(ThreadPoolExecutor(max_workers=len(clients)))
        scenario: Final = stack.enter_context(gateway.scenario())

        def wave() -> tuple[tuple[httpx.Response | None, str], ...]:
            markers: Final = tuple(_marker("wrapped") for _ in clients)
            responses: Final = tuple(
                pool.map(
                    lambda pair: _post_or_dropped(
                        pair[0], gateway.key, "/v1/chat/completions", _chat_body(model, pair[1][0])
                    ),
                    zip(clients, markers, strict=True),
                )
            )
            return tuple(zip(responses, tuple(marker_hex for _, marker_hex in markers), strict=True))

        def unwrapped(wire: Wire, results: tuple[tuple[httpx.Response | None, str], ...]) -> None:
            for response, marker_hex in results:
                assert response is not None, results
                assert _chat_content(response) == _payload(marker_hex), response.text
            _seen_once(_delivered(wire), tuple(marker_hex for _, marker_hex in results))

        with wire_server(_peer) as wire:
            port: Final = int(wire.url.rsplit(":", 1)[1])
            model: Final = _vertex_deployment(gateway, scenario, wire.url, _TOOL_MODEL)
            unwrapped(wire, wave())
        outage: Final = wave()
        assert all(response is not None and response.status_code >= 500 for response, _ in outage), outage
        assert gateway.request("GET", "/health/liveliness").status_code == 200
        with wire_server(_peer, port=port) as revived:
            unwrapped(revived, wave())


@pytest.mark.timeout(_OWNED_PROXY_CELL_SECONDS)
def test_worker_sigkill_mid_burst_leaves_the_replacement_unwrapping_envelopes(gateway: Gateway, tmp_path: Path) -> None:
    held: Final[SimpleQueue[str]] = SimpleQueue()
    release: Final = threading.Event()
    with ExitStack() as stack:
        wire: Final = stack.enter_context(wire_server(_holding(held, release)))
        stack.callback(release.set)
        config: Final = _owned_config(tmp_path / "structured-output-worker-kill.yaml", gateway, wire.url)
        owned: Final = stack.enter_context(owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2))
        proxy_url: Final = owned.gateway.client.base_url
        workers: Final = eventually(
            lambda: tuple(int(pid) for pid in _STARTED_WORKER.findall(owned.log.read_text())),
            lambda pids: len(pids) == 2,
            seconds=30,
        )
        clients: Final = _clients(stack, str(proxy_url), 12)
        pool: Final = stack.enter_context(ThreadPoolExecutor(max_workers=len(clients)))
        ports: Final = tuple(_local_port(client) for client in clients)
        markers: Final = tuple(_marker("wrapped") for _ in clients)
        futures: Final = tuple(
            pool.submit(_post_or_dropped, client, gateway.key, "/v1/chat/completions", _chat_body(_OWNED_MODEL, marker))
            for client, (marker, _) in zip(clients, markers, strict=True)
        )
        eventually(held.qsize, lambda size: size == len(clients), seconds=30)
        shares: Final = {pid: _accepted_client_ports(pid, proxy_url.port or 0) & frozenset(ports) for pid in workers}
        assert sum(map(len, shares.values())) == len(clients), shares
        victim: Final = min((pid for pid in workers if shares[pid]), key=lambda pid: len(shares[pid]))
        psutil.Process(victim).send_signal(signal.SIGKILL)
        release.set()
        results: Final = tuple(future.result(timeout=60) for future in futures)
        for port, result, (_, marker_hex) in zip(ports, results, markers, strict=True):
            if port in shares[victim]:
                assert result is None, (port, result, shares)
            else:
                assert result is not None and _chat_content(result) == _payload(marker_hex), (port, result, shares)
        eventually(lambda: len(_STARTED_WORKER.findall(owned.log.read_text())), lambda started: started >= 3, 120)
        second_wave: Final = tuple(_marker("wrapped") for _ in range(6))
        for marker, marker_hex in second_wave:
            after: Final = _post(
                owned.gateway.client, gateway.key, "/v1/chat/completions", _chat_body(_OWNED_MODEL, marker)
            )
            assert _chat_content(after) == _payload(marker_hex), after.text
        _seen_once(_delivered(wire), tuple(marker_hex for _, marker_hex in markers + second_wave))
        assert owned.process.poll() is None
