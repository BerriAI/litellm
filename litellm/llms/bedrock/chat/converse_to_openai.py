import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate, chain, groupby, repeat, takewhile
from typing import Final, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionAssistantToolCall,
    ChatCompletionSystemMessage,
    ChatCompletionTextObject,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionToolMessage,
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
    ChatCompletionUserMessage,
)

JsonPath: TypeAlias = tuple[str | int, ...]
_Node: Final = TypeVar("_Node")
_OBJECT: Final = TypeAdapter(Mapping[str, object])
_ITEMS: Final = TypeAdapter(Sequence[object])


class _ConverseModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class _GuardText(_ConverseModel):
    text: str = ""


class _GuardContent(_ConverseModel):
    text: _GuardText | None = None


class _ToolUse(_ConverseModel):
    tool_use_id: str = Field(default="", alias="toolUseId")
    name: str = ""
    tool_input: object = Field(default=None, alias="input")


class _ToolResultContent(_ConverseModel):
    text: str = ""
    json_value: object = Field(default=None, alias="json")


class _ToolResult(_ConverseModel):
    tool_use_id: str = Field(default="", alias="toolUseId")
    content: tuple[_ToolResultContent, ...] = ()


class _ContentBlock(_ConverseModel):
    text: str = ""
    guard_content: _GuardContent | None = Field(default=None, alias="guardContent")
    tool_use: _ToolUse | None = Field(default=None, alias="toolUse")
    tool_result: _ToolResult | None = Field(default=None, alias="toolResult")


class _Message(_ConverseModel):
    role: str = ""
    content: tuple[_ContentBlock, ...] = ()


class _ToolInputSchema(_ConverseModel):
    json_schema: Mapping[str, object] | None = Field(default=None, alias="json")


class _ToolSpec(_ConverseModel):
    name: str = ""
    description: str = ""
    input_schema: _ToolInputSchema | None = Field(default=None, alias="inputSchema")


class _Tool(_ConverseModel):
    tool_spec: _ToolSpec | None = Field(default=None, alias="toolSpec")


class _ToolConfig(_ConverseModel):
    tools: tuple[_Tool, ...] = ()


class ConverseRequest(_ConverseModel):
    system: tuple[_ContentBlock, ...] = ()
    messages: tuple[_Message, ...] = ()
    tool_config: _ToolConfig | None = Field(default=None, alias="toolConfig")
    additional_model_request_fields: object = Field(default=None, alias="additionalModelRequestFields")


@dataclass(frozen=True, slots=True)
class TextSlot:
    path: JsonPath
    text: str


@dataclass(frozen=True, slots=True)
class JsonSlot:
    path: JsonPath
    value: object


Slot: TypeAlias = TextSlot | JsonSlot


@dataclass(frozen=True, slots=True)
class ConverseRow:
    message: AllMessageValues
    content: tuple[Slot, ...]
    calls: tuple[JsonSlot, ...] = ()


def parse_converse_request(body: Mapping[str, object]) -> ConverseRequest | None:
    try:
        return ConverseRequest.model_validate(body)
    except ValidationError:
        verbose_proxy_logger.debug("Converse body not convertible to OpenAI messages")
        return None


def as_json_object(value: object) -> Mapping[str, object] | None:
    try:
        return _OBJECT.validate_python(value)
    except ValidationError:
        return None


def as_json_array(value: object) -> Sequence[object] | None:
    try:
        return _ITEMS.validate_python(value)
    except ValidationError:
        return None


def breadth_first(root: _Node, children: Callable[[_Node], tuple[_Node, ...]]) -> Iterator[_Node]:
    def next_level(level: tuple[_Node, ...], _: None) -> tuple[_Node, ...]:
        return tuple(chain.from_iterable(map(children, level)))

    return chain.from_iterable(takewhile(bool, accumulate(repeat(None), next_level, initial=(root,))))


def _json_children(node: tuple[JsonPath, object]) -> tuple[tuple[JsonPath, object], ...]:
    path, value = node
    fields: Final = as_json_object(value)
    if fields is not None:
        return tuple(((*path, key), child) for key, child in fields.items())
    items: Final = as_json_array(value)
    return () if items is None else tuple(((*path, index), child) for index, child in enumerate(items))


def _text_leaf(node: tuple[JsonPath, object]) -> TextSlot | None:
    path, value = node
    return TextSlot(path, value) if isinstance(value, str) and value else None


def _string_leaves(path: JsonPath, value: object) -> tuple[TextSlot, ...]:
    leaves: Final = map(_text_leaf, breadth_first((path, value), _json_children))
    return tuple(leaf for leaf in leaves if leaf is not None)


def _block_slot(path: JsonPath, block: _ContentBlock) -> TextSlot | None:
    if block.text:
        return TextSlot((*path, "text"), block.text)
    guard_text: Final = block.guard_content.text if block.guard_content else None
    return (
        TextSlot((*path, "guardContent", "text", "text"), guard_text.text) if guard_text and guard_text.text else None
    )


def _text_slots(path: JsonPath, blocks: Iterable[tuple[int, _ContentBlock]]) -> tuple[TextSlot, ...]:
    slots: Final = (_block_slot((*path, index), block) for index, block in blocks)
    return tuple(slot for slot in slots if slot is not None)


def _text_parts(slots: Sequence[TextSlot]) -> tuple[ChatCompletionTextObject, ...]:
    return tuple(ChatCompletionTextObject(type="text", text=slot.text) for slot in slots)


def _slot_text(slot: Slot) -> str:
    match slot:
        case TextSlot():
            return slot.text
        case JsonSlot():
            return json.dumps(slot.value)


def _tool_part_slot(path: JsonPath, part: _ToolResultContent) -> Slot | None:
    if part.text:
        return TextSlot((*path, "text"), part.text)
    return None if part.json_value is None else JsonSlot((*path, "json"), part.json_value)


def _tool_result_row(path: JsonPath, result: _ToolResult) -> ConverseRow:
    slots: Final = tuple(
        slot
        for index, part in enumerate(result.content)
        if (slot := _tool_part_slot((*path, "content", index), part)) is not None
    )
    message: Final = ChatCompletionToolMessage(
        role="tool", tool_call_id=result.tool_use_id, content="\n".join(map(_slot_text, slots))
    )
    return ConverseRow(message, slots)


def _user_group(
    path: JsonPath, is_tool_result: bool, group: Sequence[tuple[int, _ContentBlock]]
) -> tuple[ConverseRow, ...]:
    if is_tool_result:
        return tuple(
            _tool_result_row((*path, index, "toolResult"), result)
            for index, block in group
            if (result := block.tool_result) is not None
        )
    slots: Final = _text_slots(path, group)
    return (
        (ConverseRow(ChatCompletionUserMessage(role="user", content=list(_text_parts(slots))), slots),) if slots else ()
    )


def _user_rows(path: JsonPath, blocks: Sequence[_ContentBlock]) -> tuple[ConverseRow, ...]:
    groups: Final = groupby(enumerate(blocks), key=lambda item: item[1].tool_result is not None)
    return tuple(
        chain.from_iterable(_user_group(path, is_tool_result, tuple(group)) for is_tool_result, group in groups)
    )


def _tool_call(tool_use: _ToolUse) -> ChatCompletionAssistantToolCall:
    return ChatCompletionAssistantToolCall(
        id=tool_use.tool_use_id,
        type="function",
        function=ChatCompletionToolCallFunctionChunk(
            name=tool_use.name,
            arguments="{}" if tool_use.tool_input is None else json.dumps(tool_use.tool_input),
        ),
    )


def _assistant_rows(
    path: JsonPath, blocks: Sequence[_ContentBlock], include_tool_calls: bool
) -> tuple[ConverseRow, ...]:
    slots: Final = _text_slots(path, enumerate(blocks))
    uses: Final = (
        tuple((index, tool_use) for index, block in enumerate(blocks) if (tool_use := block.tool_use) is not None)
        if include_tool_calls
        else ()
    )
    if uses:
        message: Final = ChatCompletionAssistantMessage(
            role="assistant",
            content=list(_text_parts(slots)) or None,
            tool_calls=[_tool_call(tool_use) for _, tool_use in uses],
        )
        calls: Final = tuple(
            JsonSlot((*path, index, "toolUse", "input"), tool_use.tool_input) for index, tool_use in uses
        )
        return (ConverseRow(message, slots, calls),)
    if not slots:
        return ()
    return (ConverseRow(ChatCompletionAssistantMessage(role="assistant", content=list(_text_parts(slots))), slots),)


def _message_rows(index: int, message: _Message, include_tool_calls: bool) -> tuple[ConverseRow, ...]:
    path: Final = ("messages", index, "content")
    if message.role == "assistant":
        return _assistant_rows(path, message.content, include_tool_calls)
    return _user_rows(path, message.content)


def _system_rows(blocks: Sequence[_ContentBlock]) -> tuple[ConverseRow, ...]:
    slots: Final = _text_slots(("system",), enumerate(blocks))
    return (
        (ConverseRow(ChatCompletionSystemMessage(role="system", content=list(_text_parts(slots))), slots),)
        if slots
        else ()
    )


def _request_field_rows(value: object) -> tuple[ConverseRow, ...]:
    slots: Final = _string_leaves(("additionalModelRequestFields",), value)
    return (
        (ConverseRow(ChatCompletionUserMessage(role="user", content=list(_text_parts(slots))), slots),) if slots else ()
    )


def converse_rows(request: ConverseRequest, include_tool_calls: bool) -> tuple[ConverseRow, ...]:
    messages: Final = (
        _message_rows(index, message, include_tool_calls) for index, message in enumerate(request.messages)
    )
    return (
        *_system_rows(request.system),
        *_request_field_rows(request.additional_model_request_fields),
        *chain.from_iterable(messages),
    )


def converse_tools(request: ConverseRequest) -> tuple[ChatCompletionToolParam, ...]:
    tools: Final = request.tool_config.tools if request.tool_config else ()
    return tuple(
        ChatCompletionToolParam(
            type="function",
            function=ChatCompletionToolParamFunctionChunk(
                name=spec.name,
                description=spec.description,
                parameters=dict(spec.input_schema.json_schema),
            ),
        )
        for tool in tools
        if (spec := tool.tool_spec) and spec.name and spec.input_schema and spec.input_schema.json_schema is not None
    )
