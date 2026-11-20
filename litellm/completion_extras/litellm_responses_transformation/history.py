import copy
import functools
import hashlib
import json
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate, chain, pairwise
from types import CoroutineType, MappingProxyType
from typing import Annotated, Final, Literal, ParamSpec, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping

REPLAY_FIELD: Final = "openai_responses_replay"
PHASE_FIELD: Final = "phase"
COMMENTARY_FIELD: Final = "openai_responses_commentary"
SHOWS_MOVED_COMMENTARY_MARKER: Final = "_litellm_shows_moved_commentary"
_P: Final = ParamSpec("_P")
_R: Final = TypeVar("_R")

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


class _AnyItem(_Frozen):
    type: str


_TOOL_CALL_TYPES: Final = frozenset({"function_call", "custom_tool_call", "apply_patch_call"})


class _MessageEntry(_Frozen):
    type: Literal["message"]
    parts: tuple[Annotated[int, Field(ge=0)], ...]
    phase: Phase | None = None
    stored: Literal[True] | None = None


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
    openai_responses_commentary: tuple[tuple[str, ...], ...] | None = None


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
    openai_responses_commentary: object = None


class _CommentaryFields(_Frozen):
    openai_responses_commentary: tuple[tuple[str, ...], ...] | None = None


class _CommentaryMessage(_Frozen):
    role: str | None = None
    provider_specific_fields: _CommentaryFields | None = None


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


def message_item_phase(item: object) -> Phase | None:
    parsed: Final = _parsed_output_item(item)
    return parsed.phase if isinstance(parsed, _OutputMessage) else None


def _is_message(item: object) -> bool:
    try:
        _AnyMessage.model_validate(_as_json(item))
    except ValidationError:
        return False
    return True


def fills_turn(item: object) -> bool:
    try:
        item_type: Final = _AnyItem.model_validate(_as_json(item)).type
    except ValidationError:
        return False
    if item_type == "message":
        return message_item_phase(item) != "commentary"
    return item_type in _TOOL_CALL_TYPES


def turn_moves_commentary(output_items: Sequence[object]) -> bool:
    """Commentary leaves content only when an answer or a tool call is left to carry the turn."""
    return any(message_item_phase(item) == "commentary" for item in output_items) and any(map(fills_turn, output_items))


def _is_moved(item: _OutputItem, moved_commentary: bool) -> bool:
    return moved_commentary and isinstance(item, _OutputMessage) and item.phase == "commentary"


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


def output_message_text(
    output_items: Sequence[object], *, first_message_only: bool, moved_commentary: bool = False
) -> str:
    parsed: Final = map(_parsed_output_item, output_items)
    messages: Final = tuple(item for item in parsed if isinstance(item, _OutputMessage))
    answers: Final = tuple(item for item in messages if not _is_moved(item, moved_commentary))
    return "".join(_texts(messages[:1] if first_message_only else answers))


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
        **({"commentary": [list(parts) for parts in _carrier(message)]} if _carrier(message) else {}),
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


def _entry(items: Sequence[_OutputItem], position: int, moved_commentary: bool) -> _MessageEntry | _IndexedEntry:
    item: Final = items[position]
    if isinstance(item, _OutputMessage):
        return _MessageEntry(
            type="message",
            parts=tuple(len(part.text) for part in item.content),
            phase=item.phase,
            stored=True if _is_moved(item, moved_commentary) else None,
        )
    ordinal: Final = sum(1 for earlier in items[:position] if _kind(earlier) == _kind(item))
    return _IndexedEntry(type=item.type, index=ordinal)


def _texts(items: Sequence[_OutputItem]) -> Iterator[str]:
    for item in items:
        if isinstance(item, _OutputMessage):
            yield from (part.text for part in item.content)


def _matches(items: Sequence[_OutputItem], message: _ChatMessage, moved_commentary: bool) -> bool:
    answers: Final = tuple(item for item in items if not _is_moved(item, moved_commentary))
    reasoning_ids: Final = tuple(item.id for item in items if isinstance(item, _OutputReasoning))
    calls: Final = tuple(item for item in items if isinstance(item, _OutputToolCall))
    tool_calls: Final = message.tool_calls or ()
    searches: Final = sum(1 for item in items if isinstance(item, _OutputToolSearch))
    return (
        "".join(_texts(answers)) == (message.content or "")
        and reasoning_ids == tuple(item.id for item in message.reasoning_items or ())
        and len(calls) == len(tool_calls)
        and all(chat.id in (call.call_id, call.id) for call, chat in zip(calls, tool_calls))
        and searches == len(_tool_search_entries(message))
    )


def _content_phase(
    output_items: Sequence[object],
    parsed: Sequence[_OutputItem | None],
    *,
    first_message_only: bool,
    moved_commentary: bool,
) -> Phase | None:
    messages: Final = tuple(item for raw, item in zip(output_items, parsed) if _is_message(raw))
    answers: Final = tuple(item for item in messages if item is None or not _is_moved(item, moved_commentary))
    if not answers:
        return None
    content_message: Final = answers[0] if first_message_only else answers[-1]
    return content_message.phase if isinstance(content_message, _OutputMessage) else None


def _carrier(message: _ChatMessage) -> tuple[tuple[str, ...], ...]:
    fields: Final = message.provider_specific_fields
    return (fields.openai_responses_commentary if fields else None) or ()


def _moved_texts(parsed: Sequence[_OutputItem | None], moved_commentary: bool) -> tuple[tuple[str, ...], ...]:
    messages: Final = tuple(item for item in parsed if isinstance(item, _OutputMessage))
    return tuple(tuple(part.text for part in item.content) for item in messages if _is_moved(item, moved_commentary))


def _carrier_fields(texts: tuple[tuple[str, ...], ...]) -> Mapping[str, object]:
    return {COMMENTARY_FIELD: [list(parts) for parts in texts]} if texts else {}


def commentary_carrier(output_items: Sequence[object], *, moved_commentary: bool) -> Mapping[str, object]:
    """The moved commentary of a turn, kept so the next request can send it back on any route."""
    return _carrier_fields(_moved_texts(tuple(map(_parsed_output_item, output_items)), moved_commentary))


def _with_carrier(message: _ChatMessage, texts: tuple[tuple[str, ...], ...]) -> _ChatMessage:
    if not texts:
        return message
    fields: Final = (message.provider_specific_fields or _ProviderFields()).model_copy(update={COMMENTARY_FIELD: texts})
    return message.model_copy(update={"provider_specific_fields": fields})


def _with_phase(message: _ChatMessage, phase: Phase | None) -> _ChatMessage:
    if phase is None:
        return message
    fields: Final = (message.provider_specific_fields or _ProviderFields()).model_copy(update={PHASE_FIELD: phase})
    return message.model_copy(update={"provider_specific_fields": fields})


def replay_fields(
    output_items: Sequence[object],
    message: object,
    *,
    completed: bool,
    first_message_only: bool,
    moved_commentary: bool = False,
) -> Mapping[str, object]:
    parsed: Final = tuple(map(_parsed_output_item, output_items))
    phase: Final = _content_phase(
        output_items, parsed, first_message_only=first_message_only, moved_commentary=moved_commentary
    )
    moved_texts: Final = _moved_texts(parsed, moved_commentary)
    turn_fields: Final[Mapping[str, object]] = MappingProxyType(
        {**({PHASE_FIELD: phase} if phase else {}), **_carrier_fields(moved_texts)}
    )
    items: Final = tuple(item for item in parsed if item is not None)
    parsed_message: Final = _chat_message(message)
    chat: Final = _with_carrier(_with_phase(parsed_message, phase), moved_texts) if parsed_message is not None else None
    if (
        not completed
        or not items
        or len(items) != len(parsed)
        or chat is None
        or not _matches(items, chat, moved_commentary)
    ):
        return turn_fields
    entries: Final = tuple(_entry(items, position, moved_commentary) for position in range(len(items)))
    replay: Final = _Replay(version=1, items=entries, checksum=_checksum(chat, entries))
    return MappingProxyType({**turn_fields, REPLAY_FIELD: replay.model_dump(mode="json", exclude_none=True)})


def _commentary_message(message: object) -> _CommentaryMessage | None:
    try:
        parsed: Final = _CommentaryMessage.model_validate(_as_json(message))
    except ValidationError:
        return None
    return parsed if parsed.role == "assistant" else None


def commentary_parts(message: object) -> tuple[tuple[str, ...], ...]:
    """The moved commentary an assistant message carries, which is sent upstream on its next turn."""
    parsed: Final = _commentary_message(message)
    fields: Final = parsed.provider_specific_fields if parsed is not None else None
    return (fields.openai_responses_commentary if fields is not None else None) or ()


def stored_commentary_items(message: object, *, phased: bool) -> tuple[Mapping[str, object], ...]:
    """Assistant items for carried commentary, labeled with its phase only on routes that read phase."""
    phase: Final = "commentary" if phased else None
    return tuple(_message_item(texts, phase) for texts in commentary_parts(message))


@dataclass(frozen=True, slots=True)
class _ShownRow:
    index: int
    original: object
    shown: Mapping[str, object]
    snapshot: Mapping[str, object]


def _combined_content(content: object, parts: tuple[tuple[str, ...], ...]) -> object:
    commentary: Final = "".join(chain.from_iterable(parts))
    content_parts: Final = as_object_list(content)
    if content_parts is not None:
        return [{"type": "text", "text": commentary}, *content_parts]
    return commentary + (content if isinstance(content, str) else "")


def _shown_row(index: int, message: object, parts: tuple[tuple[str, ...], ...]) -> _ShownRow:
    fields: Final = as_str_mapping(message) or {}
    carried: Final = as_str_mapping(fields.get("provider_specific_fields")) or {}
    shown: Final = {
        **fields,
        "content": copy.deepcopy(_combined_content(fields.get("content"), parts)),
        "provider_specific_fields": dict(without_commentary(carried)),
    }
    return _ShownRow(index=index, original=message, shown=shown, snapshot=copy.deepcopy(shown))


def _may_carry_commentary(message: object) -> bool:
    fields: Final = as_str_mapping((as_str_mapping(message) or {}).get("provider_specific_fields")) or {}
    return fields.get(COMMENTARY_FIELD) is not None


def _shown_rows(messages: Sequence[object]) -> tuple[_ShownRow, ...]:
    candidates: Final = tuple(
        (index, message) for index, message in enumerate(messages) if _may_carry_commentary(message)
    )
    carried: Final = ((index, message, commentary_parts(message)) for index, message in candidates)
    return tuple(_shown_row(index, message, parts) for index, message, parts in carried if parts)


def _settled_message(message: object, index: int, rows: Sequence[_ShownRow], same_shape: bool) -> object:
    """A shown turn the guardrail left exactly as shown goes back to the original, carrier and replay included.
    Any change keeps the guardrail's version, whose carrier is already gone."""
    by_identity: Final = next((row for row in rows if row.shown is message), None)
    by_position: Final = next((row for row in rows if row.index == index), None) if same_shape else None
    row: Final = by_identity or by_position
    return row.original if row is not None and as_str_mapping(message) == row.snapshot else message


def _settled_messages(
    messages: Sequence[object], shown: Sequence[object], rows: Sequence[_ShownRow]
) -> list[object]:  # mutable-ok: the request's messages list
    return [
        _settled_message(message, index, rows, len(messages) == len(shown)) for index, message in enumerate(messages)
    ]


def _with_data(
    args: tuple[object, ...], kwargs: Mapping[str, object], data_position: int, data: object
) -> tuple[tuple[object, ...], Mapping[str, object]]:
    if "data" in kwargs:
        return args, {**kwargs, "data": data}
    return (*args[:data_position], data, *args[data_position + 1 :]), kwargs


async def call_with_shown_commentary(
    hook: Callable[..., Awaitable[_R]],
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    *,
    data_position: int,
    applies_rewrites: bool,
) -> _R:
    """Runs a guardrail hook on a copy of its request in which each bridged assistant turn reads its moved
    commentary as content, so a hook that only reads ``content`` still sees and can rewrite it.

    The live request keeps its own messages until the hook returns. Then, when the hook may rewrite the request,
    each turn left exactly as shown gets its original message back and a changed one keeps the guardrail's text,
    and the hook's other top-level changes are copied over. A hook that raises or is cancelled leaves the live
    messages as they were. Requests without moved commentary reach the hook untouched."""
    live: Final = kwargs.get("data") if "data" in kwargs else args[data_position] if len(args) > data_position else None
    request: Final = as_str_mapping(live) or {}
    messages: Final = as_object_list(request.get("messages")) or ()
    rows: Final = _shown_rows(messages)
    if not rows or not isinstance(live, dict):
        return await hook(*args, **kwargs)
    shown_messages: Final = [
        next((row.shown for row in rows if row.index == index), message) for index, message in enumerate(messages)
    ]
    shown_request: Final = {**request, "messages": shown_messages}
    shown_args, shown_kwargs = _with_data(args, kwargs, data_position, shown_request)
    try:
        result: Final = await hook(*shown_args, **shown_kwargs)
    finally:
        for key in tuple(key for key in request if key not in shown_request):
            del live[key]
        for key, value in shown_request.items():
            if key != "messages":
                live[key] = value
    rewritten: Final[object] = result
    settled: Final = _settled_messages(
        as_object_list((as_str_mapping(rewritten) or shown_request).get("messages")) or (), shown_messages, rows
    )
    unchanged: Final = len(settled) == len(messages) and all(a is b for a, b in zip(settled, messages))
    if isinstance(rewritten, dict):
        rewritten["messages"] = messages if unchanged else settled
    if applies_rewrites and not unchanged:
        live["messages"] = settled
    return result


def shows_moved_commentary(
    *, data_position: int, applies_rewrites: bool
) -> Callable[[Callable[_P, "CoroutineType[object, object, _R]"]], Callable[_P, "CoroutineType[object, object, _R]"]]:
    """Decorates a guardrail request hook so it runs through ``call_with_shown_commentary``."""

    def decorate(
        hook: Callable[_P, "CoroutineType[object, object, _R]"],
    ) -> Callable[_P, "CoroutineType[object, object, _R]"]:
        @functools.wraps(hook)
        async def shown(*args: _P.args, **kwargs: _P.kwargs) -> _R:  # kwargs-ok: forwards the hook's own arguments
            return await call_with_shown_commentary(
                hook, args, kwargs, data_position=data_position, applies_rewrites=applies_rewrites
            )

        vars(shown)[SHOWS_MOVED_COMMENTARY_MARKER] = True
        return shown

    return decorate


def without_commentary(provider_specific_fields: Mapping[str, object]) -> Mapping[str, object]:
    """Logging copy without the carrier, and without a record that cannot validate once the carrier is gone."""
    if provider_specific_fields.get(COMMENTARY_FIELD) is None:
        return provider_specific_fields
    return {
        **provider_specific_fields,
        COMMENTARY_FIELD: None,
        **({REPLAY_FIELD: None} if REPLAY_FIELD in provider_specific_fields else {}),
    }


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


def _is_sliced(entry: _MessageEntry | _IndexedEntry) -> bool:
    """A message entry whose text is a slice of the chat content rather than of the commentary carrier."""
    return isinstance(entry, _MessageEntry) and entry.stored is None


def _is_stored(entry: _MessageEntry | _IndexedEntry) -> bool:
    return isinstance(entry, _MessageEntry) and entry.stored is True


def _sliced_entries(entries: Sequence[_MessageEntry | _IndexedEntry]) -> tuple[_MessageEntry, ...]:
    messages: Final = tuple(entry for entry in entries if isinstance(entry, _MessageEntry))
    return tuple(entry for entry in messages if entry.stored is None)


class _StoredReplay(_Frozen):
    openai_responses_replay: _Replay


def phased_message_texts(provider_specific_fields: object, content: str) -> tuple[tuple[str, Phase | None], ...]:
    try:
        replay: Final = _StoredReplay.model_validate(provider_specific_fields).openai_responses_replay
    except ValidationError:
        return ()
    entries: Final = _sliced_entries(replay.items)
    if sum(chain.from_iterable(entry.parts for entry in entries)) != len(content):
        return ()
    if all(entry.phase is None for entry in entries):
        return ()
    return tuple(
        ("".join(texts), entry.phase) for texts, entry in zip(_message_texts(entries, content), entries, strict=True)
    )


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
    parts: Final = chain.from_iterable(entry.parts for entry in _sliced_entries(entries))
    indexed: Final = (entry for entry in entries if isinstance(entry, _IndexedEntry))
    stored: Final = sum(1 for entry in entries if _is_stored(entry))
    return (
        sum(parts) == len(message.content or "")
        and stored == len(_carrier(message))
        and all(entry.index < sizes[entry.type] for entry in indexed)
    )


def _resolve(
    message: _ChatMessage, entries: Sequence[_MessageEntry | _IndexedEntry], replay_tool_search: bool
) -> ReplayedTurn | None:
    indexed: Final = tuple(entry for entry in entries if isinstance(entry, _IndexedEntry))
    if not replay_tool_search and any(entry.type in ("tool_search_call", "tool_search_output") for entry in indexed):
        return None
    calls: Final = tuple(entry for entry in indexed if entry.type in ("function_call", "custom_tool_call"))
    texts: Final = _message_texts(_sliced_entries(entries), message.content or "")
    sliced_positions: Final = tuple(position for position, entry in enumerate(entries) if _is_sliced(entry))
    stored_positions: Final = tuple(position for position, entry in enumerate(entries) if _is_stored(entry))
    resolved: Final = _Resolved(
        message=message,
        texts=MappingProxyType(
            {
                **dict(zip(sliced_positions, texts, strict=True)),
                **dict(zip(stored_positions, _carrier(message), strict=True)),
            }
        ),
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
