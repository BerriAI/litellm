import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate, chain, pairwise
from types import MappingProxyType
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

REPLAY_FIELD: Final = "openai_responses_replay"
PHASE_FIELD: Final = "phase"

Phase: TypeAlias = Literal["commentary", "final_answer"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class _OutputText(_Frozen):
    type: Literal["output_text"]
    text: str


class _OutputMessage(_Frozen):
    type: Literal["message"]
    role: Literal["assistant"]
    content: tuple[_OutputText, ...]
    phase: Phase | None = None


class _OutputReasoning(_Frozen):
    type: Literal["reasoning"]
    id: str


class _OutputToolCall(_Frozen):
    type: Literal["function_call", "custom_tool_call"]
    call_id: str
    id: str | None = None


class _OutputToolSearch(_Frozen):
    type: Literal["tool_search_call", "tool_search_output"]


_OutputItem: TypeAlias = Annotated[
    _OutputMessage | _OutputReasoning | _OutputToolCall | _OutputToolSearch, Field(discriminator="type")
]
_OUTPUT_ITEM: Final[TypeAdapter[_OutputItem]] = TypeAdapter(_OutputItem)


class _AnyMessage(_Frozen):
    type: Literal["message"]


class _MessageEntry(_Frozen):
    type: Literal["message"]
    parts: tuple[Annotated[int, Field(ge=0)], ...]
    phase: Phase | None = None


class _IndexedEntry(_Frozen):
    type: Literal["reasoning", "function_call", "custom_tool_call", "tool_search_call", "tool_search_output"]
    index: Annotated[int, Field(ge=0)]


_Entry: TypeAlias = Annotated[_MessageEntry | _IndexedEntry, Field(discriminator="type")]


class _Replay(_Frozen):
    version: Literal[1]
    items: tuple[_Entry, ...]
    checksum: str


class _Namespaced(_Frozen):
    namespace: str | None = None


class _ToolFunction(_Frozen):
    name: str = ""
    arguments: str = ""
    provider_specific_fields: _Namespaced | None = None


class _ToolCall(_Frozen):
    id: str
    function: _ToolFunction
    provider_specific_fields: _Namespaced | None = None


class _SummaryText(_Frozen):
    type: str = "summary_text"
    text: str = ""


class _Reasoning(_Frozen):
    id: str | None = None
    encrypted_content: str | None = None
    summary: tuple[_SummaryText, ...] | None = None


class _ProviderFields(_Frozen):
    phase: Phase | None = None
    tool_search_items: tuple[Mapping[str, object], ...] | None = None
    openai_responses_replay: object = None


class _ChatMessage(_Frozen):
    role: str | None = None
    content: str | None = None
    tool_calls: tuple[_ToolCall, ...] | None = None
    reasoning_items: tuple[_Reasoning, ...] | None = None
    phase: Phase | None = None
    provider_specific_fields: _ProviderFields | None = None


class _PhaseFields(_Frozen):
    phase: Phase | None = None


class _RequestedPhaseFields(_Frozen):
    phase: object = None


class _RequestedPhase(_Frozen):
    phase: object = None
    provider_specific_fields: _RequestedPhaseFields | None = None


@dataclass(frozen=True, slots=True)
class UnknownPhase:
    value: object


_PHASE: Final[TypeAdapter[Phase]] = TypeAdapter(Phase)


class _RetainedFields(_PhaseFields):
    openai_responses_replay: object = None


@dataclass(frozen=True, slots=True)
class ReplayedTurn:
    items: tuple[Mapping[str, object], ...]
    custom_call_ids: frozenset[str]


def _as_json(value: object) -> object:
    return value.model_dump() if isinstance(value, BaseModel) else value


def _parsed_output_item(item: object) -> _OutputItem | None:
    try:
        return _OUTPUT_ITEM.validate_python(_as_json(item))
    except ValidationError:
        return None


def _is_message(item: object) -> bool:
    try:
        _AnyMessage.model_validate(_as_json(item))
    except ValidationError:
        return False
    return True


def _chat_message(message: object) -> _ChatMessage | None:
    try:
        return _ChatMessage.model_validate(_as_json(message))
    except ValidationError:
        return None


def requested_phase(message: object) -> Phase | UnknownPhase | None:
    try:
        requested: Final = _RequestedPhase.model_validate(_as_json(message))
    except ValidationError:
        return None
    fields: Final = requested.provider_specific_fields
    value: Final = requested.phase or (fields.phase if fields else None)
    if value is None:
        return None
    try:
        return _PHASE.validate_python(value)
    except ValidationError:
        return UnknownPhase(value)


def reasoning_input_item(
    item_id: str | None, summary: Sequence[object] | None, encrypted_content: str | None
) -> Mapping[str, object]:
    return {
        "type": "reasoning",
        **({"id": item_id} if item_id else {}),
        "summary": list(summary or ()),
        **({"encrypted_content": encrypted_content} if encrypted_content else {}),
    }


def retained_replay_fields(provider_specific_fields: object) -> Mapping[str, object] | None:
    try:
        retained: Final = _RetainedFields.model_validate(provider_specific_fields)
    except ValidationError:
        return None
    return retained.model_dump(exclude_none=True) or None


def output_message_text(output_items: Sequence[object], *, first_message_only: bool) -> str:
    parsed: Final = map(_parsed_output_item, output_items)
    messages: Final = tuple(item for item in parsed if isinstance(item, _OutputMessage))
    return "".join(_texts(messages[:1] if first_message_only else messages))


def _tool_search_entries(message: _ChatMessage) -> tuple[Mapping[str, object], ...]:
    fields: Final = message.provider_specific_fields
    items: Final = (fields.tool_search_items if fields else None) or ()
    return tuple(item for item in items if item.get("type") in ("tool_search_call", "tool_search_output"))


def _checksum(message: _ChatMessage, entries: Sequence[_MessageEntry | _IndexedEntry]) -> str:
    fields: Final = message.provider_specific_fields
    bound: Final = {
        "content": message.content or "",
        "tool_calls": [[call.id, call.function.name, call.function.arguments] for call in message.tool_calls or ()],
        "reasoning": [
            [item.id, item.encrypted_content, [part.text for part in item.summary or ()]]
            for item in message.reasoning_items or ()
        ],
        "tool_search_items": list((fields.tool_search_items if fields else None) or ()),
        "phase": message.phase or (fields.phase if fields else None),
        "items": [entry.model_dump(mode="json", exclude_none=True) for entry in entries],
    }
    encoded: Final = json.dumps(bound, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8", "surrogatepass")).hexdigest()


def _kind(item: _OutputItem) -> str:
    match item:
        case _OutputToolCall():
            return "tool_call"
        case _OutputToolSearch():
            return "tool_search"
        case _OutputMessage() | _OutputReasoning():
            return item.type


def _entry(items: Sequence[_OutputItem], position: int) -> _MessageEntry | _IndexedEntry:
    item: Final = items[position]
    if isinstance(item, _OutputMessage):
        return _MessageEntry(type="message", parts=tuple(len(part.text) for part in item.content), phase=item.phase)
    ordinal: Final = sum(1 for earlier in items[:position] if _kind(earlier) == _kind(item))
    return _IndexedEntry(type=item.type, index=ordinal)


def _texts(items: Sequence[_OutputItem]) -> Iterator[str]:
    for item in items:
        if isinstance(item, _OutputMessage):
            yield from (part.text for part in item.content)


def _matches(items: Sequence[_OutputItem], message: _ChatMessage) -> bool:
    reasoning_ids: Final = tuple(item.id for item in items if isinstance(item, _OutputReasoning))
    calls: Final = tuple(item for item in items if isinstance(item, _OutputToolCall))
    tool_calls: Final = message.tool_calls or ()
    searches: Final = sum(1 for item in items if isinstance(item, _OutputToolSearch))
    return (
        "".join(_texts(items)) == (message.content or "")
        and reasoning_ids == tuple(item.id for item in message.reasoning_items or ())
        and len(calls) == len(tool_calls)
        and all(chat.id in (call.call_id, call.id) for call, chat in zip(calls, tool_calls))
        and searches == len(_tool_search_entries(message))
    )


def _content_phase(
    output_items: Sequence[object], parsed: Sequence[_OutputItem | None], *, first_message_only: bool
) -> Phase | None:
    messages: Final = tuple(item for raw, item in zip(output_items, parsed) if _is_message(raw))
    if not messages:
        return None
    content_message: Final = messages[0] if first_message_only else messages[-1]
    return content_message.phase if isinstance(content_message, _OutputMessage) else None


def _with_phase(message: _ChatMessage, phase: Phase | None) -> _ChatMessage:
    if phase is None:
        return message
    fields: Final = (message.provider_specific_fields or _ProviderFields()).model_copy(update={PHASE_FIELD: phase})
    return message.model_copy(update={"provider_specific_fields": fields})


def replay_fields(
    output_items: Sequence[object], message: object, *, completed: bool, first_message_only: bool
) -> Mapping[str, object]:
    parsed: Final = tuple(map(_parsed_output_item, output_items))
    phase: Final = _content_phase(output_items, parsed, first_message_only=first_message_only)
    phase_fields: Final[Mapping[str, object]] = MappingProxyType({PHASE_FIELD: phase} if phase else {})
    items: Final = tuple(item for item in parsed if item is not None)
    parsed_message: Final = _chat_message(message)
    chat: Final = _with_phase(parsed_message, phase) if parsed_message is not None else None
    if not completed or not items or len(items) != len(parsed) or chat is None or not _matches(items, chat):
        return phase_fields
    entries: Final = tuple(_entry(items, position) for position in range(len(items)))
    replay: Final = _Replay(version=1, items=entries, checksum=_checksum(chat, entries))
    return MappingProxyType({**phase_fields, REPLAY_FIELD: replay.model_dump(mode="json", exclude_none=True)})


def _stored_replay(message: _ChatMessage) -> _Replay | None:
    fields: Final = message.provider_specific_fields
    if fields is None or fields.openai_responses_replay is None:
        return None
    try:
        return _Replay.model_validate(fields.openai_responses_replay)
    except ValidationError:
        return None


def _message_texts(entries: Sequence[_MessageEntry], content: str) -> tuple[tuple[str, ...], ...]:
    lengths: Final = tuple(chain.from_iterable(entry.parts for entry in entries))
    bounds: Final = tuple(accumulate(lengths, initial=0))
    parts: Final = tuple(content[start:end] for start, end in pairwise(bounds))
    groups: Final = tuple(accumulate((len(entry.parts) for entry in entries), initial=0))
    return tuple(parts[start:end] for start, end in pairwise(groups))


def _message_item(texts: Sequence[str], phase: Phase | None) -> Mapping[str, object]:
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text} for text in texts],
        **({PHASE_FIELD: phase} if phase else {}),
    }


def _function_call_item(call: _ToolCall) -> Mapping[str, object]:
    namespaced: Final = call.provider_specific_fields or call.function.provider_specific_fields
    namespace: Final = namespaced.namespace if namespaced else None
    return {
        "type": "function_call",
        "call_id": call.id,
        "name": call.function.name,
        "arguments": call.function.arguments,
        **({"namespace": namespace} if namespace else {}),
    }


def _custom_tool_call_item(call: _ToolCall) -> Mapping[str, object]:
    return {
        "type": "custom_tool_call",
        "call_id": call.id,
        "name": call.function.name,
        "input": call.function.arguments,
    }


def _reasoning_item(item: _Reasoning) -> Mapping[str, object]:
    summary: Final = tuple(part.model_dump() for part in item.summary or ())
    return reasoning_input_item(item.id, summary, item.encrypted_content)


@dataclass(frozen=True, slots=True)
class _Resolved:
    message: _ChatMessage
    texts: Mapping[int, tuple[str, ...]]
    tool_searches: tuple[Mapping[str, object], ...]


def _input_item(resolved: _Resolved, position: int, entry: _MessageEntry | _IndexedEntry) -> Mapping[str, object]:
    if isinstance(entry, _MessageEntry):
        return _message_item(resolved.texts[position], entry.phase)
    tool_calls: Final = resolved.message.tool_calls or ()
    match entry.type:
        case "reasoning":
            return _reasoning_item((resolved.message.reasoning_items or ())[entry.index])
        case "function_call":
            return _function_call_item(tool_calls[entry.index])
        case "custom_tool_call":
            return _custom_tool_call_item(tool_calls[entry.index])
        case "tool_search_call" | "tool_search_output":
            return dict(resolved.tool_searches[entry.index])


def _fits(message: _ChatMessage, entries: Sequence[_MessageEntry | _IndexedEntry]) -> bool:
    tool_calls: Final = len(message.tool_calls or ())
    tool_searches: Final = len(_tool_search_entries(message))
    sizes: Final = MappingProxyType(
        {
            "reasoning": len(message.reasoning_items or ()),
            "function_call": tool_calls,
            "custom_tool_call": tool_calls,
            "tool_search_call": tool_searches,
            "tool_search_output": tool_searches,
        }
    )
    parts: Final = chain.from_iterable(entry.parts for entry in entries if isinstance(entry, _MessageEntry))
    indexed: Final = (entry for entry in entries if isinstance(entry, _IndexedEntry))
    return sum(parts) == len(message.content or "") and all(entry.index < sizes[entry.type] for entry in indexed)


def _resolve(
    message: _ChatMessage, entries: Sequence[_MessageEntry | _IndexedEntry], replay_tool_search: bool
) -> ReplayedTurn | None:
    indexed: Final = tuple(entry for entry in entries if isinstance(entry, _IndexedEntry))
    if not replay_tool_search and any(entry.type in ("tool_search_call", "tool_search_output") for entry in indexed):
        return None
    messages: Final = tuple(entry for entry in entries if isinstance(entry, _MessageEntry))
    calls: Final = tuple(entry for entry in indexed if entry.type in ("function_call", "custom_tool_call"))
    texts: Final = _message_texts(messages, message.content or "")
    positions: Final = tuple(position for position, entry in enumerate(entries) if isinstance(entry, _MessageEntry))
    resolved: Final = _Resolved(
        message=message,
        texts=MappingProxyType(dict(zip(positions, texts, strict=True))),
        tool_searches=_tool_search_entries(message),
    )
    tool_calls: Final = message.tool_calls or ()
    return ReplayedTurn(
        items=tuple(_input_item(resolved, position, entry) for position, entry in enumerate(entries)),
        custom_call_ids=frozenset(tool_calls[entry.index].id for entry in calls if entry.type == "custom_tool_call"),
    )


def replay_items(message: object, *, replay_tool_search: bool) -> ReplayedTurn | None:
    chat: Final = _chat_message(message)
    stored: Final = _stored_replay(chat) if chat is not None and chat.role == "assistant" else None
    if chat is None or stored is None or stored.checksum != _checksum(chat, stored.items):
        return None
    if not _fits(chat, stored.items):
        return None
    return _resolve(chat, stored.items, replay_tool_search)
