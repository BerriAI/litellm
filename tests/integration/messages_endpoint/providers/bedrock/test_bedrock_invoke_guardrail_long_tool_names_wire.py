import base64
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, gateway_from_environment, object_value
from integration._support.ports import allocate_port
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.upstream import _aws_event_frame
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.timeout(4 * graceful_stop_seconds() + 240)

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_TOKEN: Final = "synthetic-bedrock-bearer"
_GUARDRAIL: Final = "headroom-compression"
_SONNET: Final = "claude-sonnet-5-5"
_OPUS: Final = "claude-opus-5"
_SEARCH: Final = "ToolSearch"
_NAME_SOURCE: Final = "mcp__plugin_support-kit_remote-workspace__getAccessibleWorkspaceResourcesForTheCurrentUser"
_REPLY_TEXT: Final = "done"
_FORCED_ID: Final = "toolu_forced"


def _tool_name(length: int) -> str:
    name: Final = _NAME_SOURCE[:length]
    assert len(name) == length, name
    return name


def _tool(name: str, **fields: JsonValue) -> dict[str, JsonValue]:
    return {"name": name, "description": "Returns OK", "input_schema": {"type": "object", "properties": {}}, **fields}


def _blocks(message: JsonValue) -> tuple[Mapping[str, JsonValue], ...]:
    content: Final = object_value(message).get("content")
    return tuple(object_value(block) for block in content) if isinstance(content, list) else ()


def _parts(block: Mapping[str, JsonValue]) -> tuple[Mapping[str, JsonValue], ...]:
    content: Final = block.get("content")
    return tuple(object_value(part) for part in content) if isinstance(content, list) else ()


def _tool_results(blocks: tuple[Mapping[str, JsonValue], ...]) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(block for block in blocks if block.get("type") == "tool_result")


def _message_blocks(body: Mapping[str, JsonValue]) -> tuple[Mapping[str, JsonValue], ...]:
    messages: Final = body.get("messages")
    assert isinstance(messages, list), body
    return tuple(block for message in messages for block in _blocks(message))


def _tool_references(body: Mapping[str, JsonValue]) -> tuple[str, ...]:
    parts: Final = tuple(part for block in _tool_results(_message_blocks(body)) for part in _parts(block))
    return tuple(str(part["tool_name"]) for part in parts if part.get("type") == "tool_reference")


def _referenced_names(body: Mapping[str, JsonValue]) -> tuple[str, ...]:
    used: Final = tuple(str(block["name"]) for block in _message_blocks(body) if block.get("type") == "tool_use")
    return (*used, *_tool_references(body))


def _tool_names(body: Mapping[str, JsonValue]) -> tuple[str, ...]:
    tools: Final = body.get("tools")
    assert isinstance(tools, list), body
    return tuple(str(object_value(tool)["name"]) for tool in tools)


def _forced_name(body: Mapping[str, JsonValue]) -> str | None:
    choice: Final = body.get("tool_choice")
    if not isinstance(choice, dict) or choice.get("type") != "tool":
        return None
    return str(choice["name"])


def _bedrock_error(message: str) -> Reply:
    return Reply(status=400, body=json.dumps({"message": message}).encode())


def _stop_reason(forced: str | None) -> str:
    return "end_turn" if forced is None else "tool_use"


def _content(forced: str | None) -> list[JsonValue]:
    if forced is None:
        return [{"type": "text", "text": _REPLY_TEXT}]
    return [{"type": "tool_use", "id": _FORCED_ID, "name": forced, "input": {}}]


def _message(model: str, forced: str | None) -> dict[str, JsonValue]:
    return {
        "id": "msg_long_tool_names",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": _content(forced),
        "stop_reason": _stop_reason(forced),
        "stop_sequence": None,
        "usage": {"input_tokens": 11, "output_tokens": 3},
    }


def _invoke_chunk(event: Mapping[str, JsonValue]) -> bytes:
    payload: Final = {"bytes": base64.b64encode(json.dumps(event).encode()).decode()}
    return _aws_event_frame("chunk", payload, "sc", "u")


def _block_events(forced: str | None) -> tuple[Mapping[str, JsonValue], ...]:
    if forced is None:
        return (
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": _REPLY_TEXT}},
        )
    tool_use: Final = {"type": "tool_use", "id": _FORCED_ID, "name": forced, "input": {}}
    return (
        {"type": "content_block_start", "index": 0, "content_block": tool_use},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
    )


def _stream(model: str, forced: str | None) -> Reply:
    start: Final = {**_message(model, forced), "content": [], "stop_reason": None}
    events: Final = (
        {"type": "message_start", "message": start},
        *_block_events(forced),
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": _stop_reason(forced)}, "usage": {"output_tokens": 3}},
        {"type": "message_stop"},
    )
    frames: Final = tuple(_invoke_chunk(event) for event in events)
    return Reply(chunks=frames, content_type="application/vnd.amazon.eventstream")


def _bedrock(request: Request) -> Reply:
    body: Final = _JSON_OBJECT.validate_json(request.body)
    available: Final = frozenset(_tool_names(body))
    missing: Final = [name for name in _tool_references(body) if name not in available]
    if missing:
        return _bedrock_error(f"Tool reference '{missing[0]}' not found in available tools")
    forced: Final = _forced_name(body)
    if forced is not None and forced not in available:
        return _bedrock_error(f"Tool '{forced}' not found in provided tools")
    model: Final = request.target.split("/")[2]
    if request.target.endswith("/invoke-with-response-stream"):
        return _stream(model, forced)
    return Reply(body=json.dumps(_message(model, forced)).encode())


def _compress(request: Request) -> Reply:
    assert request.target == "/v1/compress", request.target
    messages: Final = _JSON_OBJECT.validate_json(request.body)["messages"]
    reply: Final = {"messages": messages, "tokens_before": 40, "tokens_after": 32, "compression_ratio": 0.8}
    return Reply(body=json.dumps(reply).encode())


def _deployments(provider_url: str) -> list[JsonValue]:
    shared: Final = {"aws_region_name": "eu-central-1", "api_base": provider_url, "api_key": _TOKEN, "guardrails": []}
    return [
        {
            "model_name": _SONNET,
            "litellm_params": {
                "model": "global.anthropic.claude-sonnet-5-5",
                "custom_llm_provider": "bedrock",
                **shared,
            },
        },
        {"model_name": _OPUS, "litellm_params": {"model": "bedrock/eu.anthropic.claude-opus-5", **shared}},
    ]


def _config(directory: Path, provider_url: str, headroom_url: str) -> Path:
    stock: Final = _JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    guardrail: Final = {
        "guardrail": "headroom",
        "mode": "pre_call",
        "default_on": True,
        "api_base": headroom_url,
        "unreachable_fallback": "fail_open",
    }
    config: Final[dict[str, JsonValue]] = {
        **stock,
        "model_list": _deployments(provider_url),
        "router_settings": {**object_value(stock["router_settings"]), "num_retries": 0},
        "guardrails": [{"guardrail_name": _GUARDRAIL, "litellm_params": guardrail}],
    }
    path: Final = directory / "long-tool-names.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@dataclass(frozen=True, slots=True)
class _Rig:
    gateway: Gateway
    provider: Wire
    compressor: Wire | None


@pytest.fixture(scope="module", params=("fail_open", "compressed"))
def rig(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp(f"long-tool-names-{request.param}")
    with wire_server(_bedrock) as provider, wire_server(_compress) as compressor, gateway_from_environment() as gateway:
        reachable: Final = request.param == "compressed"
        headroom_url: Final = compressor.url if reachable else f"http://127.0.0.1:{allocate_port()}"
        config: Final = _config(directory, provider.url, headroom_url)
        with owned_proxy(gateway, directory, {}, config=config, workers=2) as owned:
            yield _Rig(owned, provider, compressor if reachable else None)


def _search_turns(name: str) -> list[JsonValue]:
    return [
        {"role": "user", "content": "find the tool"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_01", "name": _SEARCH, "input": {"query": "x"}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01",
                    "content": [{"type": "tool_reference", "tool_name": name}],
                }
            ],
        },
        {"role": "assistant", "content": "Found it."},
        {"role": "user", "content": "say done"},
    ]


def _history_turns(name: str) -> list[JsonValue]:
    return [
        {"role": "user", "content": "use the tool"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_02", "name": name, "input": {}}]},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_02", "content": "OK"},
                {"type": "text", "text": "say done"},
            ],
        },
    ]


def _forced_turns() -> list[JsonValue]:
    return [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "call the tool"},
    ]


def _body(shape: str, model: str, name: str, stream: bool) -> dict[str, JsonValue]:
    tools: Final = [_tool(_SEARCH), _tool(name, defer_loading=True)] if shape == "tool_reference" else [_tool(name)]
    body: Final[dict[str, JsonValue]] = {"model": model, "max_tokens": 20, "stream": stream, "tools": tools}
    match shape:
        case "tool_reference":
            return {**body, "messages": _search_turns(name)}
        case "tool_use_history":
            return {**body, "messages": _history_turns(name)}
        case _:
            return {**body, "messages": _forced_turns(), "tool_choice": {"type": "tool", "name": name}}


def _sse_tool_names(text: str) -> tuple[str, ...]:
    events: Final = tuple(
        _JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: ")
    )
    starts: Final = tuple(
        object_value(event["content_block"]) for event in events if event.get("type") == "content_block_start"
    )
    return tuple(str(block["name"]) for block in starts if block.get("type") == "tool_use")


def _answered_tool_names(text: str, stream: bool) -> tuple[str, ...]:
    if stream:
        return _sse_tool_names(text)
    content: Final = _JSON_OBJECT.validate_json(text)["content"]
    assert isinstance(content, list), text
    return tuple(str(block["name"]) for block in map(object_value, content) if block.get("type") == "tool_use")


@pytest.mark.parametrize("stream", (False, True), ids=("json", "sse"))
@pytest.mark.parametrize("length", (64, 65, 90))
@pytest.mark.parametrize("shape", ("tool_reference", "tool_choice", "tool_use_history"))
def test_guarded_messages_keep_long_tool_names_the_caller_sent(
    rig: _Rig, shape: str, length: int, stream: bool
) -> None:
    name: Final = _tool_name(length)
    body: Final = _body(shape, _OPUS if shape == "tool_choice" else _SONNET, name, stream)
    response: Final = rig.gateway.request("POST", "/v1/messages", body, headers={"anthropic-version": "2023-06-01"})
    sent: Final = rig.provider.drain()
    compressed: Final = None if rig.compressor is None else rig.compressor.drain()

    assert _GUARDRAIL in response.headers.get("x-litellm-applied-guardrails", ""), dict(response.headers)
    assert compressed is None or len(compressed) == 1, compressed
    assert response.status_code == 200, response.text
    assert len(sent) == 1, sent
    outbound: Final = _JSON_OBJECT.validate_json(sent[0].body)
    assert _tool_names(outbound) == ((_SEARCH, name) if shape == "tool_reference" else (name,)), outbound
    assert _referenced_names(outbound) == _referenced_names(body), outbound
    expected_calls: Final = (name,) if shape == "tool_choice" else ()
    assert _answered_tool_names(response.text, stream) == expected_calls, response.text
