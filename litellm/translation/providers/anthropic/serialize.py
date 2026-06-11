"""Serialize the IR into an Anthropic ``/v1/messages`` request body.

Pure and total: a valid ``ChatRequest`` always produces a body, and the body
is assembled by collecting the present ``(key, value)`` pairs and handing them
to ``dict`` rather than mutating one in place. Only fields the IR actually
carries are emitted, matching the v1 ``AnthropicConfig`` output field-for-field.
"""

from __future__ import annotations

from typing import List, Tuple

from expression import Nothing, Option, Some
from expression.collections import Block

from ...boundary import thaw
from ...ir import (
    Body,
    ChatRequest,
    ContentBlock,
    Message,
    PlainJson,
    SystemText,
    ToolChoice,
    ToolDef,
    has_tool_blocks,
)

# Anthropic requires max_tokens, so v1 merges this class default via its "Load
# Config" step when the caller omits it (AnthropicConfig.get_config()).
_DEFAULT_MAX_TOKENS = 4096

def _dummy_tool() -> PlainJson:
    """Built fresh per request: the returned body is a plain mutable dict that
    downstream v1-era code may mutate, so sharing one module-level dict would
    bleed state across requests."""
    return {
        "name": "dummy_tool",
        "input_schema": {"type": "object", "properties": {}},
        "type": "custom",
        "description": "This is a dummy tool call",
    }


def serialize_request(request: ChatRequest) -> Body:
    base: List[Tuple[str, PlainJson]] = [
        ("model", request.model),
        ("max_tokens", request.params.max_tokens.default_value(_DEFAULT_MAX_TOKENS)),
        ("messages", _messages(request.messages)),
    ]
    optional: Block[Option[Tuple[str, PlainJson]]] = Block.of(
        request.params.temperature.map(lambda value: ("temperature", value)),
        request.params.top_p.map(lambda value: ("top_p", value)),
        _stop(request.params.stop),
        _stream(request.stream),
        _system(request.system),
        _tools(request),
        request.tool_choice.map(lambda choice: ("tool_choice", _tool_choice(choice))),
    )
    present: Block[Tuple[str, PlainJson]] = optional.choose(lambda option: option)
    return dict(base + list(present))


def _messages(messages: Block[Message]) -> PlainJson:
    return [
        {"role": message.role, "content": _content(message.content)}
        for message in messages
    ]


def _content(blocks: Block[ContentBlock]) -> PlainJson:
    return [_block(block) for block in blocks]


def _block(block: ContentBlock) -> PlainJson:
    match block:
        case ContentBlock(tag="text", text=text):
            return {"type": "text", "text": text.text}
        case ContentBlock(tag="image", image=image):
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image.media_type,
                    "data": image.data,
                },
            }
        case ContentBlock(tag="tool_use", tool_use=tool_use):
            return {
                "type": "tool_use",
                "id": tool_use.id,
                "name": tool_use.name,
                "input": thaw(tool_use.arguments),
            }
        case ContentBlock(tag="tool_result", tool_result=tool_result):
            return {
                "type": "tool_result",
                "tool_use_id": tool_result.tool_use_id,
                "content": tool_result.content,
            }
    return {}


def _stop(stop: Block[str]) -> Option[Tuple[str, PlainJson]]:
    if len(stop) == 0:
        return Nothing
    return Some(("stop_sequences", list(stop)))


def _stream(stream: bool) -> Option[Tuple[str, PlainJson]]:
    return Some(("stream", True)) if stream else Nothing


def _system(system: Block[SystemText]) -> Option[Tuple[str, PlainJson]]:
    if len(system) == 0:
        return Nothing
    blocks: PlainJson = [{"type": "text", "text": text.text} for text in system]
    return Some(("system", blocks))


def _tools(request: ChatRequest) -> Option[Tuple[str, PlainJson]]:
    if len(request.tools) > 0:
        return Some(("tools", [_tool(tool) for tool in request.tools]))
    if has_tool_blocks(request.messages):
        return Some(("tools", [_dummy_tool()]))
    return Nothing


def _tool(tool: ToolDef) -> PlainJson:
    base = {"name": tool.name, "input_schema": thaw(tool.parameters), "type": "custom"}
    match tool.description:
        case Option(tag="some", some=description):
            return {**base, "description": description}
        case _:
            return base


def _tool_choice(choice: ToolChoice) -> PlainJson:
    match choice:
        case ToolChoice(tag="auto"):
            return {"type": "auto"}
        case ToolChoice(tag="required"):
            return {"type": "any"}
        case ToolChoice(tag="none"):
            return {"type": "none"}
        case ToolChoice(tag="specific", specific=name):
            return {"type": "tool", "name": name}
    return {}
