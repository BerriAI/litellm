"""Parse an OpenAI chat-completions request body into the IR.

The body dict is untyped, so every field is checked and field-level problems
accumulate into a ``BoundaryError`` rather than raising. Values are extracted
best-effort; they are only handed to ``ChatRequest`` when no failure was
recorded, so a partially-parsed value is never observed by a caller.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Tuple, TypeVar

from expression import Error, Nothing, Ok, Option, Result, Some
from expression.collections import Block, Map

from ...boundary import as_mapping, as_sequence, as_str, freeze
from ...errors import BoundaryError, ParseResult
from ...ir import (
    ChatRequest,
    ContentBlock,
    Image,
    InferenceParams,
    Json,
    Message,
    Role,
    SystemText,
    Text,
    ToolChoice,
    ToolDef,
    ToolResult,
    ToolUse,
)

_T = TypeVar("_T")

_EMPTY_STR: Block[str] = Block.empty()


@dataclass(frozen=True)
class _ParsedMessage:
    systems: Block[SystemText]
    entries: Block[Message]
    failures: Block[str]


def parse_request(raw: Dict[str, object]) -> ParseResult:
    model, model_fail = _parse_model(raw.get("model"))
    systems, messages, message_fail = _parse_messages(raw.get("messages"))
    tools, tools_fail = _parse_tools(raw.get("tools"))
    tool_choice, tool_choice_fail = _parse_tool_choice(raw.get("tool_choice"))
    params = _parse_params(raw)
    stream = raw.get("stream") is True

    failures = _concat(Block.of(model_fail, message_fail, tools_fail, tool_choice_fail))
    if len(failures) > 0:
        return Error(BoundaryError.of(failures))

    return Ok(
        ChatRequest(
            model=model,
            system=systems,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            params=params,
            stream=stream,
        )
    )


def _parse_model(value: object) -> Tuple[str, Block[str]]:
    match as_str(value):
        case Option(tag="some", some=model):
            return model, _EMPTY_STR
        case _:
            return "", Block.of("field 'model' is required and must be a string")


def _parse_messages(
    value: object,
) -> Tuple[Block[SystemText], Block[Message], Block[str]]:
    match as_sequence(value):
        case Option(tag="some", some=items):
            parsed = Block.of_seq(_parse_one_message(item) for item in items)
            systems = _concat(parsed.map(lambda p: p.systems))
            entries = _merge_adjacent(_concat(parsed.map(lambda p: p.entries)))
            failures = _concat(parsed.map(lambda p: p.failures))
            return systems, entries, failures
        case _:
            return (
                Block.empty(),
                Block.empty(),
                Block.of("field 'messages' is required and must be an array"),
            )


def _parse_one_message(value: object) -> _ParsedMessage:
    match as_mapping(value):
        case Option(tag="some", some=message):
            return _route_message(message)
        case _:
            return _ParsedMessage(
                Block.empty(), Block.empty(), Block.of("each message must be an object")
            )


def _route_message(message: Dict[str, object]) -> _ParsedMessage:
    role = message.get("role")
    if role == "system":
        return _ParsedMessage(
            _parse_system_content(message.get("content")), Block.empty(), _EMPTY_STR
        )
    if role == "user":
        content, failures = _parse_message_content(message.get("content"))
        return _entry("user", content, failures)
    if role == "assistant":
        content, failures = _parse_assistant_content(message)
        return _entry("assistant", content, failures)
    if role == "tool":
        content, failures = _parse_tool_message(message)
        return _entry("user", content, failures)
    return _ParsedMessage(
        Block.empty(), Block.empty(), Block.of(f"unsupported message role: {role!r}")
    )


def _entry(
    role: Role, content: Block[ContentBlock], failures: Block[str]
) -> _ParsedMessage:
    return _ParsedMessage(
        Block.empty(), Block.of(Message(role=role, content=content)), failures
    )


def _parse_system_content(value: object) -> Block[SystemText]:
    match as_str(value):
        case Option(tag="some", some=text):
            return Block.of(SystemText(text=text)) if text else Block.empty()
        case _:
            pass
    match as_sequence(value):
        case Option(tag="some", some=parts):
            return Block.of_seq(parts).choose(_system_text_of)
        case _:
            return Block.empty()


def _system_text_of(part: object) -> Option[SystemText]:
    text = as_mapping(part).bind(lambda m: as_str(m.get("text")))
    match text:
        case Option(tag="some", some=value) if value:
            return Some(SystemText(text=value))
        case _:
            return Nothing


def _parse_message_content(value: object) -> Tuple[Block[ContentBlock], Block[str]]:
    if value is None:
        return Block.empty(), _EMPTY_STR
    match as_str(value):
        case Option(tag="some", some=text):
            blocks = (
                Block.of(ContentBlock.of_text(Text(text=text)))
                if text
                else Block.empty()
            )
            return blocks, _EMPTY_STR
        case _:
            pass
    match as_sequence(value):
        case Option(tag="some", some=parts):
            return _split(Block.of_seq(_parse_content_part(part) for part in parts))
        case _:
            return Block.empty(), Block.of(
                "message content must be a string or an array"
            )


def _parse_content_part(part: object) -> Result[ContentBlock, str]:
    match as_mapping(part):
        case Option(tag="some", some=mapping):
            return _content_block_of(mapping)
        case _:
            return Error("each content part must be an object")


def _content_block_of(part: Dict[str, object]) -> Result[ContentBlock, str]:
    kind = part.get("type")
    if kind == "text":
        return (
            as_str(part.get("text"))
            .to_result("text content part is missing 'text'")
            .map(lambda text: ContentBlock.of_text(Text(text=text)))
        )
    if kind == "image_url":
        return _parse_image(part.get("image_url")).map(ContentBlock.of_image)
    return Error(f"unsupported content part type: {kind!r}")


def _parse_image(value: object) -> Result[Image, str]:
    url = as_mapping(value).bind(lambda m: as_str(m.get("url")))
    match url:
        case Option(tag="some", some=data_url):
            return _parse_data_uri(data_url)
        case _:
            return Error("image_url is missing a 'url' string")


def _parse_data_uri(url: str) -> Result[Image, str]:
    if not url.startswith("data:"):
        return Error("only base64 data: image URLs are supported")
    header, _, data = url.partition(",")
    media_type = header[len("data:") :].split(";")[0]
    return Ok(Image(media_type=media_type, data=data))


def _parse_assistant_content(
    message: Dict[str, object],
) -> Tuple[Block[ContentBlock], Block[str]]:
    content, content_fail = _parse_message_content(message.get("content"))
    tool_uses, tool_fail = _parse_tool_calls(message.get("tool_calls"))
    return content + tool_uses, content_fail + tool_fail


def _parse_tool_calls(value: object) -> Tuple[Block[ContentBlock], Block[str]]:
    if value is None:
        return Block.empty(), _EMPTY_STR
    match as_sequence(value):
        case Option(tag="some", some=calls):
            return _split(Block.of_seq(_parse_tool_call(call) for call in calls))
        case _:
            return Block.empty(), Block.of("'tool_calls' must be an array")


def _parse_tool_call(value: object) -> Result[ContentBlock, str]:
    match as_mapping(value):
        case Option(tag="some", some=call):
            return _tool_use_of(call)
        case _:
            return Error("each tool_call must be an object")


def _tool_use_of(call: Dict[str, object]) -> Result[ContentBlock, str]:
    function = as_mapping(call.get("function")).default_value({})
    call_id = as_str(call.get("id"))
    name = as_str(function.get("name"))
    match (call_id, name):
        case (Option(tag="some", some=cid), Option(tag="some", some=fname)):
            return _parse_arguments(function.get("arguments")).map(
                lambda args: ContentBlock.of_tool_use(
                    ToolUse(id=cid, name=fname, arguments=args)
                )
            )
        case _:
            return Error("tool_call requires 'id' and 'function.name'")


def _parse_arguments(value: object) -> Result[Map[str, Json], str]:
    if not isinstance(value, str):
        return Error("tool_call.function.arguments must be a JSON string")
    try:
        loaded = json.loads(value) if value else {}
    except json.JSONDecodeError as exc:
        return Error(f"tool_call.function.arguments is not valid JSON: {exc.msg}")
    frozen = freeze(loaded)
    if not isinstance(frozen, Map):
        return Error("tool_call.function.arguments must be a JSON object")
    return Ok(frozen)


def _parse_tool_message(
    message: Dict[str, object],
) -> Tuple[Block[ContentBlock], Block[str]]:
    tool_use_id = as_str(message.get("tool_call_id"))
    content = as_str(message.get("content"))
    match (tool_use_id, content):
        case (Option(tag="some", some=tid), Option(tag="some", some=text)):
            return (
                Block.of(
                    ContentBlock.of_tool_result(
                        ToolResult(tool_use_id=tid, content=text)
                    )
                ),
                _EMPTY_STR,
            )
        case _:
            return Block.empty(), Block.of(
                "tool message requires 'tool_call_id' and string 'content'"
            )


def _parse_tools(value: object) -> Tuple[Block[ToolDef], Block[str]]:
    if value is None:
        return Block.empty(), _EMPTY_STR
    match as_sequence(value):
        case Option(tag="some", some=tools):
            return _split(Block.of_seq(_parse_tool(tool) for tool in tools))
        case _:
            return Block.empty(), Block.of("'tools' must be an array")


def _parse_tool(value: object) -> Result[ToolDef, str]:
    function = as_mapping(value).bind(lambda m: as_mapping(m.get("function")))
    match function:
        case Option(tag="some", some=fn):
            return _tool_def_of(fn)
        case _:
            return Error("each tool must have a 'function' object")


def _tool_def_of(function: Dict[str, object]) -> Result[ToolDef, str]:
    match as_str(function.get("name")):
        case Option(tag="some", some=name):
            parameters = freeze(
                function.get("parameters") or {"type": "object", "properties": {}}
            )
            schema = parameters if isinstance(parameters, Map) else Map.empty()
            return Ok(
                ToolDef(
                    name=name,
                    description=as_str(function.get("description")),
                    parameters=schema,
                )
            )
        case _:
            return Error("tool function requires a 'name'")


def _parse_tool_choice(value: object) -> Tuple[Option[ToolChoice], Block[str]]:
    if value is None:
        return Nothing, _EMPTY_STR
    if value == "auto":
        return Some(ToolChoice.of_auto()), _EMPTY_STR
    if value == "required":
        return Some(ToolChoice.of_required()), _EMPTY_STR
    if value == "none":
        return Some(ToolChoice.of_none()), _EMPTY_STR
    match as_mapping(value):
        case Option(tag="some", some=mapping):
            return _named_tool_choice(mapping)
        case _:
            return Nothing, Block.of(f"unsupported tool_choice: {value!r}")


def _named_tool_choice(
    mapping: Dict[str, object],
) -> Tuple[Option[ToolChoice], Block[str]]:
    name = as_mapping(mapping.get("function")).bind(lambda m: as_str(m.get("name")))
    match name:
        case Option(tag="some", some=tool_name):
            return Some(ToolChoice.of_specific(tool_name)), _EMPTY_STR
        case _:
            return Nothing, Block.of("tool_choice object requires 'function.name'")


def _parse_params(raw: Dict[str, object]) -> InferenceParams:
    return InferenceParams(
        max_tokens=_as_int(raw.get("max_tokens")).or_else(
            _as_int(raw.get("max_completion_tokens"))
        ),
        temperature=_as_float(raw.get("temperature")),
        top_p=_as_float(raw.get("top_p")),
        stop=_parse_stop(raw.get("stop")),
    )


def _parse_stop(value: object) -> Block[str]:
    match as_str(value):
        case Option(tag="some", some=text):
            return Block.of(text)
        case _:
            pass
    match as_sequence(value):
        case Option(tag="some", some=items):
            return Block.of_seq(item for item in items if isinstance(item, str))
        case _:
            return Block.empty()


def _as_int(value: object) -> Option[int]:
    return (
        Some(value)
        if isinstance(value, int) and not isinstance(value, bool)
        else Nothing
    )


def _as_float(value: object) -> Option[float]:
    if isinstance(value, bool):
        return Nothing
    return Some(float(value)) if isinstance(value, (int, float)) else Nothing


def _split(results: Block[Result[_T, str]]) -> Tuple[Block[_T], Block[str]]:
    oks = results.choose(lambda r: Some(r.ok) if r.is_ok() else Nothing)
    errors = results.choose(lambda r: Some(r.error) if r.is_error() else Nothing)
    return oks, errors


def _concat(blocks: Block[Block[_T]]) -> Block[_T]:
    return blocks.fold(lambda acc, block: acc + block, Block.empty())


def _merge_adjacent(entries: Block[Message]) -> Block[Message]:
    def folder(acc: Block[Message], message: Message) -> Block[Message]:
        if len(acc) > 0 and acc[-1].role == message.role:
            last = acc[-1]
            merged = Message(role=last.role, content=last.content + message.content)
            return acc.take(len(acc) - 1) + Block.of(merged)
        return acc + Block.of(message)

    return entries.fold(folder, Block.empty())
