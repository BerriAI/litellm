from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from pydantic import JsonValue, TypeAdapter, ValidationError

HistorySurface = Literal["chat", "responses", "messages"]
_HISTORY_ADAPTER: Final = TypeAdapter(Sequence[Mapping[str, JsonValue]])
_INVARIANT_ROLES: Final = frozenset({"system", "developer"})
_MESSAGE_FIELDS: Final = frozenset({"role", "content", "type"})
_CHAT_FIELDS: Final = _MESSAGE_FIELDS.union(("name", "prefix", "tool_calls", "tool_call_id"))
_RESPONSE_FIELDS: Final = _MESSAGE_FIELDS.union(("id", "status"))
_TEXT_FIELDS: Final = frozenset({"type", "text", "cache_control", "annotations"})


@dataclass(frozen=True, slots=True)
class HistoryPartition:
    invariants: tuple[object, ...]
    past: tuple[object, ...]
    active: tuple[object, ...]
    requires_active: bool
    terminal_role: str | None


@dataclass(frozen=True, slots=True)
class _Event:
    role: str | None = None
    calls: tuple[str, ...] = ()
    results: tuple[str, ...] = ()
    user_text: bool = False


@dataclass(frozen=True, slots=True)
class _Exchange:
    pending: frozenset[str] = frozenset()
    start: int = -1
    results_started: bool = False


def _fields(item: Mapping[str, JsonValue], allowed: frozenset[str]) -> bool:
    return all(key in allowed or value is None for key, value in item.items())


def _identifier(value: JsonValue | None) -> bool:
    return isinstance(value, str) and bool(value)


def _text_block(block: JsonValue, surface: HistorySurface) -> bool:
    types: Final = ("input_text", "output_text") if surface == "responses" else ("text",)
    if not isinstance(block, dict):
        return False
    annotations: Final = block.get("annotations")
    return (
        block.get("type") in types
        and isinstance(block.get("text"), str)
        and (annotations is None or isinstance(annotations, list) and not annotations)
        and _fields(block, _TEXT_FIELDS)
    )


def _text(content: JsonValue | None, surface: HistorySurface, *, empty: bool = False) -> bool:
    if content is None:
        return empty
    if isinstance(content, str):
        return True
    return isinstance(content, list) and (empty or bool(content)) and all(_text_block(b, surface) for b in content)


def _chat_call_id(call: JsonValue) -> str | None:
    if not isinstance(call, dict) or not _fields(call, frozenset({"id", "type", "function"})):
        return None
    function: Final = call.get("function")
    identifier: Final = call.get("id")
    if not isinstance(identifier, str) or not identifier or call.get("type") != "function":
        return None
    if not isinstance(function, dict) or not _fields(function, frozenset({"name", "arguments"})):
        return None
    return identifier if _identifier(function.get("name")) and isinstance(function.get("arguments"), str) else None


def _chat_event(item: Mapping[str, JsonValue], role: str) -> _Event | None:
    calls: Final = item.get("tool_calls")
    result_id: Final = item.get("tool_call_id")
    if role == "tool":
        if (
            not isinstance(result_id, str)
            or not result_id
            or calls is not None
            or not _text(item.get("content"), "chat")
        ):
            return None
        return _Event(role, results=(result_id,))
    if result_id is not None:
        return None
    if calls is not None:
        if role != "assistant" or not isinstance(calls, list) or not calls:
            return None
        identifiers: Final = tuple(_chat_call_id(call) for call in calls)
        if any(identifier is None for identifier in identifiers) or not _text(item.get("content"), "chat", empty=True):
            return None
        return _Event(role, calls=tuple(identifier for identifier in identifiers if identifier is not None))
    return _Event(role) if _text(item.get("content"), "chat") else None


def _anthropic_block(block: JsonValue, role: str) -> _Event | None:
    if not isinstance(block, dict):
        return None
    if _text_block(block, "messages"):
        return _Event(role, user_text=role == "user" and bool(block.get("text")))
    if role == "assistant" and block.get("type") == "tool_use":
        identifier: Final = block.get("id")
        valid: Final = (
            isinstance(identifier, str)
            and bool(identifier)
            and _identifier(block.get("name"))
            and isinstance(block.get("input"), dict)
            and _fields(block, frozenset({"type", "id", "name", "input", "cache_control"}))
        )
        return _Event(role, calls=(identifier,)) if valid and isinstance(identifier, str) else None
    if role == "user" and block.get("type") == "tool_result":
        result_id: Final = block.get("tool_use_id")
        valid_result: Final = (
            isinstance(result_id, str)
            and bool(result_id)
            and _text(block.get("content"), "messages", empty=True)
            and (block.get("is_error") is None or isinstance(block.get("is_error"), bool))
            and _fields(block, frozenset({"type", "tool_use_id", "content", "is_error", "cache_control"}))
        )
        return _Event(role, results=(result_id,)) if valid_result and isinstance(result_id, str) else None
    return None


def _response_tool_event(item: Mapping[str, JsonValue]) -> _Event | None:
    identifier: Final = item.get("call_id")
    if not isinstance(identifier, str) or not identifier or item.get("status") not in (None, "completed"):
        return None
    if item.get("type") == "function_call":
        valid: Final = (
            _identifier(item.get("name"))
            and isinstance(item.get("arguments"), str)
            and _fields(item, frozenset({"type", "id", "call_id", "name", "arguments", "status"}))
        )
        return _Event(calls=(identifier,)) if valid else None
    valid_output: Final = isinstance(item.get("output"), str) and _fields(
        item, frozenset({"type", "id", "call_id", "output", "status"})
    )
    return _Event(results=(identifier,)) if valid_output else None


def _event(item: Mapping[str, JsonValue], surface: HistorySurface) -> _Event | None:
    if surface == "responses" and item.get("type") in ("function_call", "function_call_output"):
        return _response_tool_event(item)
    allowed: Final = (
        _CHAT_FIELDS if surface == "chat" else _RESPONSE_FIELDS if surface == "responses" else _MESSAGE_FIELDS
    )
    role: Final = item.get("role")
    if not _fields(item, allowed) or item.get("type") not in (None, "message") or not isinstance(role, str):
        return None
    roles: Final = ("user", "assistant") if surface == "messages" else ("system", "developer", "user", "assistant")
    if surface == "chat":
        return _chat_event(item, role) if role in (*roles, "tool") else None
    if role not in roles:
        return None
    content: Final = item.get("content")
    if surface == "responses" or isinstance(content, str):
        return _Event(role) if _text(content, surface) else None
    if not isinstance(content, list) or not content:
        return None
    blocks: Final = tuple(_anthropic_block(block, role) for block in content)
    if any(block is None for block in blocks):
        return None
    return _Event(
        role,
        calls=tuple(identifier for block in blocks if block is not None for identifier in block.calls),
        results=tuple(identifier for block in blocks if block is not None for identifier in block.results),
        user_text=any(block.user_text for block in blocks if block is not None),
    )


def _advance(state: _Exchange, event: _Event, index: int, surface: HistorySurface) -> _Exchange | None:
    if event.calls:
        if state.pending and (surface != "responses" or state.results_started):
            return None
        return _Exchange(state.pending | frozenset(event.calls), state.start if state.pending else index)
    if event.results:
        results: Final = frozenset(event.results)
        if not results <= state.pending:
            return None
        return _Exchange(state.pending - results, state.start, True)
    return None if state.pending else state


def partition_history(items: object, surface: HistorySurface) -> HistoryPartition | None:
    try:
        parsed: Final = _HISTORY_ADAPTER.validate_python(items, strict=True)
    except ValidationError:
        return None
    candidates: Final = tuple(_event(item, surface) for item in parsed)
    if any(event is None for event in candidates):
        return None
    events: Final = tuple(event for event in candidates if event is not None)
    prefix: Final = next(
        (index for index, event in enumerate(events) if event.role not in _INVARIANT_ROLES), len(events)
    )
    if any(event.role in _INVARIANT_ROLES for event in events[prefix:]):
        return None
    calls: Final = tuple(identifier for event in events for identifier in event.calls)
    results: Final = tuple(identifier for event in events for identifier in event.results)
    if (
        len(frozenset(calls)) != len(calls)
        or len(frozenset(results)) != len(results)
        or frozenset(calls) != frozenset(results)
    ):
        return None
    state: _Exchange = _Exchange()  # rebind-ok: bounded protocol-state scan
    mixed: range = range(0)  # rebind-ok: latest complete exchange carrying user text
    for index, event in enumerate(events):
        advanced: _Exchange | None = _advance(state, event, index, surface)  # rebind-ok: per-event transition
        if advanced is None:
            return None
        if event.results and (event.user_text or (mixed and mixed.start == advanced.start)):
            mixed = range(advanced.start, index + 1)
        state = advanced
    if state.pending:
        return None
    if prefix == len(parsed):
        return HistoryPartition(tuple(parsed), (), (), False, None)
    terminal: Final = events[-1]
    start: Final = state.start if terminal.calls or terminal.results else len(parsed) - 1
    latest_user: Final = max(
        (index for index, event in enumerate(events) if event.role == "user" and not event.results), default=-1
    )
    protected: Final = (
        frozenset(range(start, len(parsed)))
        | (frozenset({latest_user}) if calls and latest_user >= 0 else frozenset[int]())
        | (frozenset(mixed) if mixed.stop > latest_user else frozenset[int]())
    )
    return HistoryPartition(
        invariants=tuple(parsed[:prefix]),
        past=tuple(item for index, item in enumerate(parsed) if index >= prefix and index not in protected),
        active=tuple(item for index, item in enumerate(parsed) if index in protected),
        requires_active=bool(calls) or terminal.role == "assistant",
        terminal_role=terminal.role,
    )
