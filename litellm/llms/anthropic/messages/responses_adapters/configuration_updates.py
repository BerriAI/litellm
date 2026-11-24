"""GPT-6+ mid-conversation settings changes sent as inline Responses items, so the cached prefix never moves.

Top-level `reasoning`, `text` and `tools` stay at the values the conversation started with. A later change rides a
`configuration_update` or `additional_tools` item placed where it happened, replayed there on every later turn. Each
response's thinking signatures carry a stamp of what its request sent, which is how a later turn knows where to replay.
"""

import base64
import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain, groupby
from typing import Final, Generic, Literal, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping
from litellm.litellm_core_utils.prompt_templates.reasoning_utils import (
    anthropic_content_lists,
    encrypted_content_of_block,
    short_digest,
    stamp_of_block,
)
from litellm.responses.additional_tools import tool_identity

JsonObject: TypeAlias = dict[str, JsonValue]
EffortCheck: TypeAlias = Callable[[str], bool]
_SegmentKind: TypeAlias = Literal["user", "assistant"]
_T = TypeVar("_T")

CALLER_OWNED_SETTINGS: Final = ("input", "reasoning", "text", "tools")
_PLAIN_TEXT: Final[JsonObject] = {"format": {"type": "text"}}
_FORMAT_ONLY: Final = frozenset(("format",))
_ADDABLE_TOOL_TYPE: Final = "function"
_USER_SIDE_ROLES: Final = frozenset(("user", "system", "developer"))
_ASSISTANT_ITEM_TYPES: Final = frozenset(("reasoning", "function_call"))
_BLOCKS: Final = TypeAdapter(tuple[object, ...])


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Baseline(_Frozen):
    reasoning: JsonObject | None = None
    text: JsonObject | None = None
    tools: tuple[str, ...] = ()
    tools_sha: str | None = None


class Update(_Frozen):
    effort: str | None = None
    text: JsonObject | None = None
    tools: tuple[str, ...] = ()
    tools_sha: str | None = None


class Start(_Frozen):
    v: Literal[1] = 1
    base: Baseline


class Next(_Frozen):
    v: Literal[1] = 1
    n: int = Field(ge=1)
    update: Update


TurnStamp: TypeAlias = Start | Next
_STAMP: Final[TypeAdapter[TurnStamp]] = TypeAdapter(Start | Next)


@dataclass(frozen=True, slots=True)
class _Same:
    pass


@dataclass(frozen=True, slots=True)
class _Changed(Generic[_T]):
    value: _T


@dataclass(frozen=True, slots=True)
class _Unexpressible:
    pass


_Outcome: TypeAlias = _Same | _Changed[_T] | _Unexpressible
_SAME: Final = _Same()
_UNEXPRESSIBLE: Final = _Unexpressible()


class _Settings(_Frozen):
    reasoning: JsonObject | None
    text: JsonObject | None
    tools: tuple[JsonObject, ...]


@dataclass(frozen=True, slots=True)
class _Segment:
    kind: _SegmentKind
    items: tuple[Mapping[str, object], ...]


def encode_stamp(stamp: TurnStamp) -> str:
    return base64.urlsafe_b64encode(_STAMP.dump_json(stamp, exclude_none=True)).rstrip(b"=").decode()


def decode_stamp(encoded: str) -> TurnStamp | None:
    try:
        return _STAMP.validate_json(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
    except ValueError:
        return None


def _digest(tools: Iterable[JsonObject]) -> str:
    return short_digest(json.dumps(list(tools), sort_keys=True, separators=(",", ":")))


def _group_sha(tools: Sequence[JsonObject]) -> str | None:
    return _digest(tools) if tools else None


def _settings(kwargs: Mapping[str, object]) -> _Settings | None:
    try:
        return _Settings.model_validate(
            {"reasoning": kwargs.get("reasoning"), "text": kwargs.get("text"), "tools": kwargs.get("tools") or ()}
        )
    except ValidationError:
        return None


def _fresh_stamp(current: _Settings) -> Start:
    return Start(
        base=Baseline(
            reasoning=current.reasoning,
            text=current.text,
            tools=tuple(map(tool_identity, current.tools)),
            tools_sha=_group_sha(current.tools),
        )
    )


def _blocks(messages: Sequence[Mapping[str, object]]) -> Iterator[Mapping[str, object]]:
    for content in anthropic_content_lists(messages):
        yield from (block for block in map(as_str_mapping, _BLOCKS.validate_python(content)) if block is not None)


def _stamps_by_blob(messages: Sequence[Mapping[str, object]]) -> Mapping[str, TurnStamp | None]:
    encoded: Final = {
        blob: stamp_of_block(block)
        for block in _blocks(messages)
        if (blob := encrypted_content_of_block(block)) is not None
    }
    return {blob: None if stamp is None else decode_stamp(stamp) for blob, stamp in encoded.items()}


def _item_kind(item: Mapping[str, object]) -> _SegmentKind | None:
    match item.get("type"), item.get("role"):
        case "message", "assistant":
            return "assistant"
        case "message", role if role in _USER_SIDE_ROLES:
            return "user"
        case "function_call_output", _:
            return "user"
        case item_type, _ if item_type in _ASSISTANT_ITEM_TYPES:
            return "assistant"
        case _:
            return None


def _kinded(item: object) -> tuple[_SegmentKind, Mapping[str, object]] | None:
    mapping: Final = as_str_mapping(item)
    kind: Final = None if mapping is None else _item_kind(mapping)
    return None if mapping is None or kind is None else (kind, mapping)


def _kind_of(pair: tuple[_SegmentKind, Mapping[str, object]]) -> _SegmentKind:
    return pair[0]


def _segments(input_items: Sequence[object]) -> tuple[_Segment, ...] | None:
    kinded: Final = tuple(map(_kinded, input_items))
    known: Final = tuple(pair for pair in kinded if pair is not None)
    if len(known) != len(kinded):
        return None
    return tuple(
        _Segment(kind=kind, items=tuple(item for _, item in group)) for kind, group in groupby(known, key=_kind_of)
    )


def _run_stamp(run: _Segment, stamps_by_blob: Mapping[str, TurnStamp | None]) -> TurnStamp | None:
    blobs: Final = tuple(
        blob
        for item in run.items
        if item.get("type") == "reasoning" and isinstance(blob := item.get("encrypted_content"), str) and blob
    )
    stamps: Final = tuple(stamps_by_blob.get(blob) for blob in blobs)
    first: Final = stamps[0] if stamps else None
    return first if first is not None and all(stamp == first for stamp in stamps) else None


def _in_order_update(index: int, stamp: TurnStamp | None) -> Update | None:
    match stamp:
        case Next(n=n, update=update) if n == index:
            return update
        case _:
            return None


def _regime(run_stamps: Sequence[TurnStamp | None]) -> tuple[Baseline, tuple[Update, ...]] | None:
    match run_stamps[-1] if run_stamps else None:
        case Start(base=base):
            return base, ()
        case Next(n=n):
            first, *rest = run_stamps[-n - 1 :]
            updates: Final = tuple(
                update
                for index, stamp in enumerate(rest, start=1)
                if (update := _in_order_update(index, stamp)) is not None
            )
            return (first.base, updates) if isinstance(first, Start) and len(updates) == n else None
        case _:
            return None


def _verified_group(keys: Sequence[str], sha: str | None, defs: Mapping[str, JsonObject]) -> bool:
    if not keys:
        return sha is None
    return all(key in defs for key in keys) and _digest(defs[key] for key in keys) == sha


def _without_effort(reasoning: JsonObject) -> JsonObject:
    return {key: value for key, value in reasoning.items() if key != "effort"}


def _effort_outcome(
    base: Baseline, updates: Sequence[Update], current: JsonObject | None, accepts_effort: EffortCheck
) -> _Outcome[str]:
    efforts: Final = tuple(update.effort for update in updates if update.effort is not None)
    if base.reasoning is None or current is None:
        return _SAME if base.reasoning is None and current is None and not efforts else _UNEXPRESSIBLE
    base_effort: Final = base.reasoning.get("effort")
    stamped: Final = (*((base_effort,) if base_effort is not None else ()), *efforts)
    if _without_effort(base.reasoning) != _without_effort(current) or not all(
        isinstance(effort, str) and accepts_effort(effort) for effort in stamped
    ):
        return _UNEXPRESSIBLE
    current_effort: Final = current.get("effort")
    if current_effort == (efforts[-1] if efforts else base_effort):
        return _SAME
    return _Changed(current_effort) if isinstance(current_effort, str) else _UNEXPRESSIBLE


def _format_only(text: JsonObject | None) -> bool:
    return text is None or frozenset(text) == _FORMAT_ONLY


def _text_outcome(base: Baseline, updates: Sequence[Update], current: JsonObject | None) -> _Outcome[JsonObject]:
    texts: Final = tuple(update.text for update in updates if update.text is not None)
    if texts and not all(map(_format_only, (base.text, *texts))):
        return _UNEXPRESSIBLE
    effective: Final = (texts[-1] if texts else base.text) or _PLAIN_TEXT
    wanted: Final = current or _PLAIN_TEXT
    if effective == wanted:
        return _SAME
    return _Changed(wanted) if _format_only(effective) and _format_only(wanted) else _UNEXPRESSIBLE


def _delta(
    base: Baseline,
    updates: Sequence[Update],
    current: _Settings,
    defs: Mapping[str, JsonObject],
    accepts_effort: EffortCheck,
) -> Update | None:
    if not _verified_group(base.tools, base.tools_sha, defs):
        return None
    if not all(_verified_group(update.tools, update.tools_sha, defs) for update in updates):
        return None
    effective_tools: Final = (*base.tools, *chain.from_iterable(update.tools for update in updates))
    added: Final = tuple(tool for tool in current.tools if tool_identity(tool) not in effective_tools)
    if any(tool.get("type") != _ADDABLE_TOOL_TYPE for tool in added):
        return None
    match _effort_outcome(base, updates, current.reasoning, accepts_effort), _text_outcome(base, updates, current.text):
        case _Unexpressible(), _:
            return None
        case _, _Unexpressible():
            return None
        case effort, text:
            return Update(
                effort=effort.value if isinstance(effort, _Changed) else None,
                text=text.value if isinstance(text, _Changed) else None,
                tools=tuple(map(tool_identity, added)),
                tools_sha=_group_sha(added),
            )


def _update_items(update: Update, defs: Mapping[str, JsonObject]) -> tuple[JsonObject, ...]:
    reasoning_item: Final[tuple[JsonObject, ...]] = (
        ({"type": "configuration_update", "reasoning": {"effort": update.effort}},) if update.effort is not None else ()
    )
    text_item: Final[tuple[JsonObject, ...]] = (
        ({"type": "configuration_update", "text": update.text},) if update.text is not None else ()
    )
    tools: Final[list[JsonValue]] = [defs[key] for key in update.tools]  # mutable-ok: JSON payload list
    tools_item: Final[tuple[JsonObject, ...]] = (
        ({"type": "additional_tools", "role": "developer", "tools": tools},) if tools else ()
    )
    return (*reasoning_item, *text_item, *tools_item)


def _is_user_message(item: Mapping[str, object]) -> bool:
    return item.get("type") == "message" and item.get("role") == "user"


def _with_items_placed(segment: _Segment, placed: Sequence[JsonObject]) -> tuple[Mapping[str, object], ...]:
    position: Final = next(
        (index for index, item in enumerate(segment.items) if _is_user_message(item)), len(segment.items)
    )
    return (*segment.items[:position], *placed, *segment.items[position:])


def _top_level(name: str, value: object) -> Mapping[str, object]:
    return {} if value is None else {name: value}


def _replayed(
    kwargs: Mapping[str, object],
    messages: Sequence[Mapping[str, object]],
    current: _Settings,
    accepts_effort: EffortCheck,
) -> tuple[Mapping[str, object], Next] | None:
    segments: Final = _segments(as_object_list(kwargs.get("input")) or ())
    if not segments or segments[-1].kind != "user":
        return None
    stamps_by_blob: Final = _stamps_by_blob(messages)
    run_positions: Final = tuple(index for index, segment in enumerate(segments) if segment.kind == "assistant")
    regime: Final = _regime(tuple(_run_stamp(segments[index], stamps_by_blob) for index in run_positions))
    if regime is None:
        return None
    base, updates = regime
    defs: Final = {tool_identity(tool): tool for tool in current.tools}
    delta: Final = _delta(base, updates, current, defs, accepts_effort)
    if delta is None:
        return None
    first_update_run: Final = len(run_positions) - len(updates)
    placed: Final = {
        **{
            run_positions[first_update_run + offset] - 1: _update_items(update, defs)
            for offset, update in enumerate(updates)
        },
        len(segments) - 1: _update_items(delta, defs),
    }
    input_items: Final = [  # mutable-ok: the Responses request's input list
        *chain.from_iterable(
            _with_items_placed(segment, placed.get(index, ())) for index, segment in enumerate(segments)
        )
    ]
    restored: Final = {
        **{key: value for key, value in kwargs.items() if key not in CALLER_OWNED_SETTINGS},
        **_top_level("reasoning", base.reasoning),
        **_top_level("text", base.text),
        **_top_level("tools", [defs[key] for key in base.tools] if base.tools else None),
        "input": input_items,
    }
    return restored, Next(n=len(updates) + 1, update=delta)


def with_configuration_updates(
    kwargs: Mapping[str, object], messages: Sequence[Mapping[str, object]], accepts_effort: EffortCheck
) -> tuple[Mapping[str, object], str | None]:
    """The request with settings changed since the regime's first turn moved inline, and the stamp its response carries.

    A history that cannot be replayed exactly, a stamped value the deployment would not accept, or a change the inline
    items cannot express sends the request as is and starts a new regime with it.
    """
    current: Final = _settings(kwargs)
    if current is None:
        return kwargs, None
    replayed: Final = _replayed(kwargs, messages, current, accepts_effort)
    if replayed is None:
        return kwargs, encode_stamp(_fresh_stamp(current))
    request, stamp = replayed
    return request, encode_stamp(stamp)
