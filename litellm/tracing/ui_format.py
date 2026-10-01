"""The LiteLLM UI content format: span input / output reduced to messages, key/value fields or plain text."""

import json
from collections.abc import Mapping, Sequence
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.tracing.normalizers.messages import MESSAGE_ROLES, ChatRole, content_text


class UIToolCall(TypedDict):
    name: ReadOnly[str]
    arguments: ReadOnly[str]


class UIMessage(TypedDict):
    role: ReadOnly[ChatRole]
    content: ReadOnly[str]
    name: ReadOnly[NotRequired[str]]
    tool_calls: ReadOnly[NotRequired[tuple[UIToolCall, ...]]]


class UIField(TypedDict):
    key: ReadOnly[str]
    value: ReadOnly[str]


class UIMessages(TypedDict):
    kind: ReadOnly[Literal["messages"]]
    messages: ReadOnly[tuple[UIMessage, ...]]


class UIFields(TypedDict):
    kind: ReadOnly[Literal["fields"]]
    fields: ReadOnly[tuple[UIField, ...]]


class UIText(TypedDict):
    kind: ReadOnly[Literal["text"]]
    text: ReadOnly[str]


UIContent: TypeAlias = UIMessages | UIFields | UIText


class _ToolFunction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    name: str = ""
    arguments: JsonValue = None


class _RawToolCall(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    name: str = ""
    args: JsonValue = None
    arguments: JsonValue = None
    function: _ToolFunction | None = None


class _RawMessage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    role: str | None = None
    type: str | None = None
    content: JsonValue = None
    name: str | None = None
    tool_calls: tuple[_RawToolCall, ...] | None = None
    kwargs: "_RawMessage | None" = None


_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_MESSAGE: Final = TypeAdapter(_RawMessage)
_MESSAGES: Final = TypeAdapter(tuple[_RawMessage, ...])


def _unwrapped(message: _RawMessage) -> _RawMessage:
    return message.kwargs if message.kwargs is not None else message


def _is_message(message: _RawMessage) -> bool:
    has_role: Final = message.role is not None or message.type in MESSAGE_ROLES
    return has_role and ("content" in message.model_fields_set or bool(message.tool_calls))


def _arguments_text(arguments: JsonValue) -> str:
    match arguments:
        case str():
            return arguments
        case None:
            return "{}"
        case _:
            return json.dumps(arguments)


def _tool_call(call: _RawToolCall) -> UIToolCall:
    if call.function is not None:
        return UIToolCall(name=call.function.name or call.name, arguments=_arguments_text(call.function.arguments))
    return UIToolCall(name=call.name, arguments=_arguments_text(call.arguments if call.args is None else call.args))


def _role(message: _RawMessage, has_tool_calls: bool) -> ChatRole:
    """Known roles and LangChain types map directly; any other role is the assistant when it calls tools, else the user."""
    known: Final = MESSAGE_ROLES.get(message.role or message.type or "")
    if known is not None:
        return known
    return "assistant" if has_tool_calls else "user"


def _ui_message(message: _RawMessage) -> UIMessage:
    calls: Final = tuple(_tool_call(call) for call in message.tool_calls or ())
    role: Final = _role(message, bool(calls))
    content: Final = content_text(message.content)
    match (message.name or None, calls):
        case (None, ()):
            return UIMessage(role=role, content=content)
        case (None, _):
            return UIMessage(role=role, content=content, tool_calls=calls)
        case (str() as name, ()):
            return UIMessage(role=role, content=content, name=name)
        case (str() as name, _):
            return UIMessage(role=role, content=content, name=name, tool_calls=calls)


def _messages(parsed: Sequence[JsonValue] | Mapping[str, JsonValue]) -> tuple[_RawMessage, ...] | None:
    try:
        raw: Final = (
            (_MESSAGE.validate_python(parsed),) if isinstance(parsed, Mapping) else _MESSAGES.validate_python(parsed)
        )
    except ValidationError:
        return None
    unwrapped: Final = tuple(_unwrapped(message) for message in raw)
    return unwrapped if unwrapped and all(_is_message(message) for message in unwrapped) else None


def _field_value(value: JsonValue) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def _parsed(raw: str) -> JsonValue:
    try:
        return _JSON.validate_json(raw)
    except ValidationError:
        return raw


def to_ui_content(raw: str) -> UIContent:
    if not raw:
        return UIText(kind="text", text="")
    parsed: Final = _parsed(raw)
    if not isinstance(parsed, list | dict):
        return UIText(kind="text", text=parsed if isinstance(parsed, str) else raw)
    messages: Final = _messages(parsed)
    if messages is not None:
        return UIMessages(kind="messages", messages=tuple(_ui_message(message) for message in messages))
    if isinstance(parsed, dict):
        return UIFields(kind="fields", fields=tuple(UIField(key=k, value=_field_value(v)) for k, v in parsed.items()))
    return UIText(kind="text", text=raw)
