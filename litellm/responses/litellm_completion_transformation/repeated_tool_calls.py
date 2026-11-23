import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from itertools import accumulate, chain, islice, takewhile
from types import MappingProxyType
from typing import Final, TypeVar, cast  # noqa: TID251  # a dict of a generic type has no cast-free typed copy

from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError

from litellm.constants import LITELLM_TRUNCATED_PAYLOAD_FIELD, REDACTED_TOOL_CALL_ARGUMENTS_PLACEHOLDER
from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping
from litellm.litellm_core_utils.db_truncation import is_truncation_of, was_truncated_for_db
from litellm.litellm_core_utils.prompt_templates.factory import is_same_tool_call, tool_call_id_prefix
from litellm.litellm_core_utils.prompt_templates.mid_conversation_system import message_field

_FieldsT: Final = TypeVar("_FieldsT")
_MessageT: Final = TypeVar("_MessageT")
_FIELDS_OF_LONE_CALLS: Final = frozenset({"role", "tool_calls"})
_FIELDS_OF_A_CALL: Final = frozenset({"id", "type", "function", "index"})
_FIELDS_OF_A_FUNCTION: Final = frozenset({"name", "arguments"})
_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


def _canonical(arguments: object) -> object:
    if not isinstance(arguments, str) or not arguments:
        return arguments or REDACTED_TOOL_CALL_ARGUMENTS_PLACEHOLDER
    try:
        return json.dumps(_JSON.validate_json(arguments), sort_keys=True)
    except ValueError:
        return arguments


@dataclass(frozen=True)
class _MadeCall:
    made_by: int
    call_id: object
    name: object
    arguments: object

    @cached_property
    def canonical_arguments(self) -> object:
        return _canonical(self.arguments)


_NO_CALLS: Final[Mapping[str, _MadeCall]] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class _Step:
    unanswered: Mapping[str, _MadeCall]
    repeated: Mapping[str, _MadeCall]


def _call_key(call_id: object) -> str:
    return tool_call_id_prefix(call_id) or ""


def _calls(message: object) -> Sequence[object]:
    return as_object_list(message_field(message, "tool_calls")) or ()


def _bare(value: object, expected: frozenset[str]) -> Mapping[str, object] | None:
    fields: Final = as_str_mapping(value)
    if fields is None or any(field for name, field in fields.items() if name not in expected):
        return None
    return fields


def _is_from_memory(message_fields: Mapping[str, object]) -> bool:
    return frozenset(message_fields) == _FIELDS_OF_LONE_CALLS


def _is_same_tool(name: object, other_name: object) -> bool:
    if not isinstance(name, str) or not isinstance(other_name, str):
        return name == other_name
    bare, namespaced = sorted((name, other_name), key=len)
    return name == other_name or namespaced.endswith(f"__{bare}")


def _makes(call: _MadeCall, arguments: object) -> bool:
    return call.arguments == arguments or call.canonical_arguments == _canonical(arguments)


def _custom_tool_input(arguments: str) -> object:
    try:
        wrapped: Final = as_str_mapping(_JSON.validate_json(arguments))
    except ValidationError:
        return None
    return wrapped.get("content") if wrapped is not None and len(wrapped) == 1 else None


def _was_cut(arguments: object) -> bool:
    if not isinstance(arguments, str) or LITELLM_TRUNCATED_PAYLOAD_FIELD not in arguments:
        return False
    if was_truncated_for_db(arguments):
        return True
    custom_tool_input: Final = _custom_tool_input(arguments)
    return isinstance(custom_tool_input, str) and was_truncated_for_db(custom_tool_input)


def _with_whole_arguments(made: _MadeCall, repeated: object) -> _MadeCall:
    stored: Final = made.arguments
    if isinstance(stored, str) and isinstance(repeated, str) and is_truncation_of(stored, repeated):
        return replace(made, arguments=repeated)
    return made


def _repeated_call(call: object, unanswered: Mapping[str, _MadeCall], *, from_memory: bool) -> _MadeCall | None:
    fields: Final = _bare(call, _FIELDS_OF_A_CALL)
    function: Final = _bare(message_field(call, "function"), _FIELDS_OF_A_FUNCTION)
    if fields is None or function is None:
        return None
    call_id: Final = fields.get("id")
    made: Final = unanswered.get(_call_key(call_id))
    if made is None or not is_same_tool_call(call_id, made.call_id):
        return None
    if not from_memory and not _is_same_tool(made.name, function.get("name")):
        return None
    arguments: Final = function.get("arguments")
    whole: Final = _with_whole_arguments(made, arguments)
    if from_memory or _was_cut(arguments) or _makes(whole, arguments):
        return whole
    if isinstance(arguments, str) and whole.canonical_arguments == REDACTED_TOOL_CALL_ARGUMENTS_PLACEHOLDER:
        return replace(made, arguments=arguments)
    return None


def _repeated_calls(message: object, unanswered: Mapping[str, _MadeCall]) -> Mapping[str, _MadeCall]:
    message_fields: Final = _bare(message, _FIELDS_OF_LONE_CALLS)
    if message_fields is None:
        return _NO_CALLS
    calls: Final = _calls(message)
    from_memory: Final = _is_from_memory(message_fields)
    found: Final = takewhile(bool, (_repeated_call(call, unanswered, from_memory=from_memory) for call in calls))
    repeated: Final = MappingProxyType({_call_key(call.call_id): call for call in found if call is not None})
    return repeated if len(repeated) == len(calls) else _NO_CALLS


def _next_step(step: _Step, indexed_message: tuple[int, object]) -> _Step:
    index, message = indexed_message
    if message_field(message, "role") == "tool":
        answered: Final = _call_key(message_field(message, "tool_call_id"))
        unanswered: Final = MappingProxyType({key: made for key, made in step.unanswered.items() if key != answered})
        return _Step(unanswered, _NO_CALLS)
    repeated: Final = _repeated_calls(message, step.unanswered)
    if repeated:
        return _Step(MappingProxyType({**step.unanswered, **repeated}), repeated)
    made: Final = tuple(
        _MadeCall(
            made_by=index,
            call_id=message_field(call, "id"),
            name=message_field(message_field(call, "function"), "name"),
            arguments=message_field(message_field(call, "function"), "arguments"),
        )
        for call in _calls(message)
    )
    times_made: Final = Counter(_call_key(call.call_id) for call in made)
    made_once: Final = MappingProxyType({key: call for call in made if times_made[key := _call_key(call.call_id)] == 1})
    return _Step(made_once, _NO_CALLS)


def _with_field(fields: _FieldsT, name: str, value: object) -> _FieldsT:
    if isinstance(fields, BaseModel):
        return fields.model_copy(update={name: value})
    mapping: Final = as_str_mapping(fields)
    if mapping is None:
        return fields
    return cast(_FieldsT, {**mapping, name: value})  # cast-ok: a copy of the same dict with one field set


def _with_arguments(call: object, repeated: _MadeCall | None) -> object:
    function: Final = message_field(call, "function")
    if repeated is None or message_field(function, "arguments") == repeated.arguments:
        return call
    return _with_field(call, "function", _with_field(function, "arguments", repeated.arguments))


def _with_calls_whole(message: _MessageT, index: int, dropped: Mapping[tuple[int, str], _MadeCall]) -> _MessageT:
    calls: Final = _calls(message)
    made: Final = tuple((index, _call_key(message_field(call, "id"))) for call in calls)
    if dropped.keys().isdisjoint(made):
        return message
    return _with_field(
        message,
        "tool_calls",
        [_with_arguments(call, dropped.get(key)) for call, key in zip(calls, made)],
    )


def without_repeated_tool_calls(
    messages: Sequence[_MessageT], *, keep_stored_arguments: Callable[[], bool]
) -> tuple[_MessageT, ...]:
    steps: Final = accumulate(enumerate(messages), _next_step, initial=_Step(_NO_CALLS, _NO_CALLS))
    repeated: Final = tuple(step.repeated for step in islice(steps, 1, None))
    dropped: Final = MappingProxyType(
        {(call.made_by, key): call for key, call in chain.from_iterable(calls.items() for calls in repeated)}
    )
    kept: Final = tuple((index, message) for index, (message, calls) in enumerate(zip(messages, repeated)) if not calls)
    as_stored: Final = tuple(message for _, message in kept)
    whole: Final = tuple(_with_calls_whole(message, index, dropped) for index, message in kept)
    return as_stored if whole == as_stored or keep_stored_arguments() else whole
