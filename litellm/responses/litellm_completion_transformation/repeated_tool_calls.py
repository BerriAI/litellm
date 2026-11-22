from collections.abc import Sequence
from itertools import accumulate
from typing import Final, TypeVar

from litellm.litellm_core_utils.core_helpers import as_object_list
from litellm.litellm_core_utils.prompt_templates.mid_conversation_system import message_field

_MessageT: Final = TypeVar("_MessageT")
_FIELDS_SAYING_MORE_THAN_A_CALL: Final = ("content", "reasoning_content", "thinking_blocks")


def _called_ids(message: object) -> tuple[object, ...]:
    return tuple(message_field(call, "id") for call in as_object_list(message_field(message, "tool_calls")) or ())


def _only_repeats_a_call(message: object, unanswered: frozenset[str]) -> bool:
    match _called_ids(message):
        case (str() as called_id,):
            return called_id in unanswered and not any(
                message_field(message, field) for field in _FIELDS_SAYING_MORE_THAN_A_CALL
            )
        case _:
            return False


def _unanswered_after(unanswered: frozenset[str], message: object) -> frozenset[str]:
    if message_field(message, "role") == "tool":
        answered: Final = message_field(message, "tool_call_id")
        return frozenset(call_id for call_id in unanswered if call_id != answered)
    if _only_repeats_a_call(message, unanswered):
        return unanswered
    return frozenset(call_id for call_id in _called_ids(message) if isinstance(call_id, str))


def without_repeated_tool_calls(messages: Sequence[_MessageT]) -> tuple[_MessageT, ...]:
    nothing_unanswered: Final[frozenset[str]] = frozenset()
    unanswered_before: Final = accumulate(messages, _unanswered_after, initial=nothing_unanswered)
    return tuple(
        message
        for message, unanswered in zip(messages, unanswered_before)
        if not _only_repeats_a_call(message, unanswered)
    )
