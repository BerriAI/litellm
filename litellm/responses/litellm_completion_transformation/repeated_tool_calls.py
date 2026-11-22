from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate
from types import MappingProxyType
from typing import Final, TypeVar, cast  # noqa: TID251  # a dict of a generic type has no cast-free typed copy

from pydantic import BaseModel

from litellm.constants import REDACTED_TOOL_CALL_ARGUMENTS_PLACEHOLDER
from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping
from litellm.litellm_core_utils.db_truncation import is_truncation_of
from litellm.litellm_core_utils.prompt_templates.factory import tool_call_id_prefix
from litellm.litellm_core_utils.prompt_templates.mid_conversation_system import message_field

_FieldsT: Final = TypeVar("_FieldsT")
_MessageT: Final = TypeVar("_MessageT")
_FIELDS_OF_A_LONE_CALL: Final = frozenset({"role", "tool_calls"})
_FIELDS_OF_A_CALL: Final = frozenset({"id", "type", "function", "index"})
_FIELDS_OF_A_FUNCTION: Final = frozenset({"name", "arguments"})
_NO_CALLS: Final[Mapping[str, int]] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class _RepeatedCall:
    made_by: int
    key: str
    arguments: object


def _call_key(call_id: object) -> str | None:
    return tool_call_id_prefix(call_id) or None


def _calls(message: object) -> Sequence[object]:
    return as_object_list(message_field(message, "tool_calls")) or ()


def _says_more(fields: Mapping[str, object], expected: frozenset[str]) -> bool:
    return any(value for key, value in fields.items() if key not in expected)


def _repeated_call(message: object, unanswered: Mapping[str, int]) -> _RepeatedCall | None:
    calls: Final = _calls(message)
    if len(calls) != 1:
        return None
    fields: Final = as_str_mapping(message)
    call: Final = as_str_mapping(calls[0])
    function: Final = as_str_mapping(message_field(calls[0], "function"))
    if fields is None or call is None or function is None:
        return None
    key: Final = _call_key(call.get("id"))
    if key is None or key not in unanswered:
        return None
    if (
        _says_more(fields, _FIELDS_OF_A_LONE_CALL)
        or _says_more(call, _FIELDS_OF_A_CALL)
        or _says_more(function, _FIELDS_OF_A_FUNCTION)
    ):
        return None
    return _RepeatedCall(made_by=unanswered[key], key=key, arguments=function.get("arguments"))


def _unanswered_after(unanswered: Mapping[str, int], indexed_message: tuple[int, object]) -> Mapping[str, int]:
    index, message = indexed_message
    if message_field(message, "role") == "tool":
        answered: Final = _call_key(message_field(message, "tool_call_id"))
        return MappingProxyType({key: made_by for key, made_by in unanswered.items() if key != answered})
    if _repeated_call(message, unanswered) is not None:
        return unanswered
    keys: Final = tuple(_call_key(message_field(call, "id")) for call in _calls(message))
    return MappingProxyType({key: index for key in keys if key is not None})


def _with_field(fields: _FieldsT, name: str, value: object) -> _FieldsT:
    if isinstance(fields, BaseModel):
        return fields.model_copy(update={name: value})
    mapping: Final = as_str_mapping(fields)
    if mapping is None:
        return fields
    return cast(_FieldsT, {**mapping, name: value})  # cast-ok: a copy of the same dict with one field set


def _with_arguments_storage_lost(call: object, remembered: object) -> object:
    function: Final = message_field(call, "function")
    stored: Final = message_field(function, "arguments")
    if not isinstance(stored, str) or not isinstance(remembered, str):
        return call
    if stored != REDACTED_TOOL_CALL_ARGUMENTS_PLACEHOLDER and not is_truncation_of(stored, remembered):
        return call
    return _with_field(call, "function", _with_field(function, "arguments", remembered))


def _with_calls_whole(message: _MessageT, index: int, remembered: Mapping[tuple[int, str], object]) -> _MessageT:
    calls: Final = _calls(message)
    made: Final = tuple((index, _call_key(message_field(call, "id")) or "") for call in calls)
    if remembered.keys().isdisjoint(made):
        return message
    return _with_field(
        message,
        "tool_calls",
        [_with_arguments_storage_lost(call, remembered.get(key)) for call, key in zip(calls, made)],
    )


def without_repeated_tool_calls(messages: Sequence[_MessageT], *, keep_stored_arguments: bool) -> tuple[_MessageT, ...]:
    unanswered_before: Final = accumulate(enumerate(messages), _unanswered_after, initial=_NO_CALLS)
    repeated: Final = tuple(map(_repeated_call, messages, unanswered_before))
    remembered: Final = MappingProxyType(
        {
            (call.made_by, call.key): call.arguments
            for call in repeated
            if call is not None and not keep_stored_arguments
        }
    )
    return tuple(
        _with_calls_whole(message, index, remembered)
        for index, (message, repeat) in enumerate(zip(messages, repeated))
        if repeat is None
    )
