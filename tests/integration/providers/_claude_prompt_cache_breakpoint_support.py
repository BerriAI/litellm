import json
from collections.abc import Iterator
from typing import Final, Literal

from integration._support.client import Gateway, Scenario
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire
from integration.providers._cache_control_marks_support import (
    ANTHROPIC_MODEL,
    BEDROCK_MODEL,
    PROVIDER_KEY,
    Mark,
    anthropic_labels,
    anthropic_peer,
    bedrock_labels,
    bedrock_peer,
)
from pydantic import JsonValue, TypeAdapter
from typing_extensions import assert_never

Provider = Literal[
    "anthropic",
    "bedrock_converse",
    "bedrock_invoke",
    "bedrock_claude_platform",
    "bedrock_mantle",
    "vertex_ai",
    "azure_ai",
]

PROVIDERS: Final[tuple[Provider, ...]] = (
    "anthropic",
    "bedrock_converse",
    "bedrock_invoke",
    "bedrock_claude_platform",
    "bedrock_mantle",
    "vertex_ai",
    "azure_ai",
)
ANTHROPIC_BODY_PROVIDERS: Final = tuple(provider for provider in PROVIDERS if provider != "bedrock_converse")
_BEDROCK_TOKEN: Final = "synthetic-bedrock-bearer"
_MANTLE_TOKEN: Final = "synthetic-mantle-bearer"
_VERTEX_PROJECT: Final = "scripted-project"
_VERTEX_LOCATION: Final = "us-east5"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
# Claude's 400s on api.anthropic.com, Bedrock Invoke, Bedrock Mantle, Claude Platform and Bedrock Converse, observed
# live 2026-10-15 for PR 46391 (two marks on one tool result, and a 1h cachePoint read after a 5m one on Converse)
_TOOL_RESULT_MARKED_TWICE: Final = "cache_control may not be specified within `tool_result.content`"
_TTL_ORDER: Final = "a ttl='1h' cache_control block must not come after a ttl='5m' cache_control block"


def _ttl_out_of_order(marks: tuple[Mark, ...]) -> bool:
    first_short: Final = next((index for index, mark in enumerate(marks) if mark.ttl != "1h"), len(marks))
    return any(mark.ttl == "1h" for mark in marks[first_short:])


def _marked(block: JsonValue) -> bool:
    return isinstance(block, dict) and block.get("cache_control") is not None


def _marked_more_than_once(result: dict[str, JsonValue]) -> bool:
    content: Final = result.get("content")
    return _marked(result) + sum(map(_marked, content if isinstance(content, list) else [])) > 1


def _tool_results(request: Request) -> Iterator[dict[str, JsonValue]]:
    for message in _messages(_JSON_OBJECT.validate_json(request.body)):
        content = message.get("content") if isinstance(message, dict) else None
        for block in content if isinstance(content, list) else []:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                yield block


def respond(provider: Provider, request: Request) -> Reply:
    """The scripted peers, holding a request to the tool result and ttl order rules Claude enforces"""
    if provider == "bedrock_converse":
        return (
            Reply(
                status=400,
                headers={"x-amzn-errortype": "ValidationException"},
                body=json.dumps({"message": _TTL_ORDER}).encode(),
            )
            if _ttl_out_of_order(sent_marks(provider, request))
            else bedrock_peer(request)
        )
    if any(map(_marked_more_than_once, _tool_results(request))):
        return Reply(
            status=400,
            body=json.dumps(
                {"type": "error", "error": {"type": "invalid_request_error", "message": _TOOL_RESULT_MARKED_TWICE}}
            ).encode(),
        )
    return anthropic_peer(request)


def labels(provider: Provider, request: Request) -> list[str]:
    return bedrock_labels(request) if provider == "bedrock_converse" else anthropic_labels(request)


def litellm_params(gateway: Gateway, wire: Wire, provider: Provider) -> dict[str, JsonValue]:
    match provider:
        case "anthropic":
            return {"model": f"anthropic/{ANTHROPIC_MODEL}", "api_base": wire.url, "api_key": PROVIDER_KEY}
        case "bedrock_converse":
            return {
                "model": f"bedrock/converse/{BEDROCK_MODEL}",
                "api_key": PROVIDER_KEY,
                "aws_region_name": "us-east-1",
                "aws_bedrock_runtime_endpoint": wire.url,
            }
        case "bedrock_invoke":
            return {
                "model": f"bedrock/invoke/{BEDROCK_MODEL}",
                "api_key": _BEDROCK_TOKEN,
                "aws_region_name": "us-east-1",
                "api_base": wire.url,
                "aws_bedrock_runtime_endpoint": wire.url,
            }
        case "bedrock_claude_platform":
            return {
                "model": f"bedrock/claude_platform/{ANTHROPIC_MODEL}",
                "api_base": wire.url,
                "api_key": PROVIDER_KEY,
                "workspace_id": "wrkspc_scripted",
                "aws_region_name": "us-east-1",
            }
        case "bedrock_mantle":
            return {"model": f"bedrock_mantle/{BEDROCK_MODEL}", "api_base": wire.url, "api_key": _MANTLE_TOKEN}
        case "vertex_ai":
            return {
                "model": f"vertex_ai/{ANTHROPIC_MODEL}",
                "api_base": wire.url,
                "api_key": None,
                "vertex_project": _VERTEX_PROJECT,
                "vertex_location": _VERTEX_LOCATION,
                "vertex_credentials": service_account_json(_VERTEX_PROJECT, gateway.upstream_url.rstrip("/")),
            }
        case "azure_ai":
            return {"model": f"azure_ai/{ANTHROPIC_MODEL}", "api_base": wire.url, "api_key": PROVIDER_KEY}
        case _:
            assert_never(provider)


def deployment(gateway: Gateway, scenario: Scenario, wire: Wire, provider: Provider, **fields: JsonValue) -> str:
    return scenario.model(model_info=None, **litellm_params(gateway, wire, provider), **fields)


def only_request(wire: Wire) -> Request:
    received: Final = wire.drain()
    assert len(received) == 1, [request.target for request in received]
    return received[0]


def _ttl(mark: JsonValue) -> str | None:
    ttl: Final = mark.get("ttl") if isinstance(mark, dict) else None
    return ttl if isinstance(ttl, str) else None


def _anthropic_label(block: dict[str, JsonValue]) -> str:
    match block.get("type"):
        case "text":
            return f"text:{block.get('text')}"
        case "tool_use":
            return f"call:{block.get('id')}"
        case "tool_result":
            return f"result:{block.get('tool_use_id')}"
        case _:
            return f"tool:{block.get('name')}"


def _anthropic_marks(value: JsonValue) -> Iterator[Mark]:
    if isinstance(value, list):
        for item in value:
            yield from _anthropic_marks(item)
        return
    if not isinstance(value, dict):
        return
    if "cache_control" in value:
        yield Mark(_anthropic_label(value), _ttl(value["cache_control"]))
    yield from _anthropic_marks(value.get("content"))


def _converse_label(block: JsonValue) -> str:
    if not isinstance(block, dict):
        return "start"
    if isinstance(block.get("text"), str):
        return f"text:{block['text']}"
    for kind, label in (("toolUse", "call"), ("toolResult", "result")):
        inner = block.get(kind)
        if isinstance(inner, dict):
            return f"{label}:{inner.get('toolUseId')}"
    spec: Final = block.get("toolSpec")
    return f"tool:{spec.get('name')}" if isinstance(spec, dict) else "other"


def _converse_marks(blocks: JsonValue) -> Iterator[Mark]:
    listed: Final = blocks if isinstance(blocks, list) else []
    for previous, block in zip([None, *listed], listed):
        if not isinstance(block, dict):
            continue
        result = block.get("toolResult")
        if isinstance(result, dict):
            yield from _converse_marks(result.get("content"))
        if "cachePoint" in block:
            yield Mark(_converse_label(previous), _ttl(block["cachePoint"]))


def _messages(body: dict[str, JsonValue]) -> list[JsonValue]:
    messages: Final = body.get("messages")
    return messages if isinstance(messages, list) else []


def sent_marks(provider: Provider, request: Request) -> tuple[Mark, ...]:
    """Every cache mark the provider received in wire order, nested tool result content included"""
    body: Final = _JSON_OBJECT.validate_json(request.body)
    contents: Final = [message.get("content") for message in _messages(body) if isinstance(message, dict)]
    if provider == "bedrock_converse":
        tool_config: Final = body.get("toolConfig")
        return (
            *_converse_marks(tool_config.get("tools") if isinstance(tool_config, dict) else None),
            *_converse_marks(body.get("system")),
            *(mark for content in contents for mark in _converse_marks(content)),
        )
    return (*_anthropic_marks(body.get("tools")), *_anthropic_marks(body.get("system")), *_anthropic_marks(contents))
