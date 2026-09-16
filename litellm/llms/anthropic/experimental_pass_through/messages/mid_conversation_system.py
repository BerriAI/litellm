from collections.abc import Mapping, Sequence
from typing import Final

CONVERTED_SYSTEM_NOTE: Final = (
    "Operator note (not from the user): the following was originally a mid-conversation system-role reminder."
)


def as_system_content_blocks(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    return [value]


def is_system_role_message(message: object) -> bool:
    return isinstance(message, dict) and message.get("role") == "system"


def system_role_message_as_user(message: Mapping[str, object]) -> Mapping[str, object]:
    return {
        "role": "user",
        "content": as_system_content_blocks(CONVERTED_SYSTEM_NOTE) + as_system_content_blocks(message.get("content")),
    }


def opens_with_tool_results(message: object) -> bool:
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    content: Final = message.get("content")
    return (
        isinstance(content, list)
        and len(content) > 0
        and isinstance(content[0], dict)
        and content[0].get("type") == "tool_result"
    )


def system_run_before(messages: Sequence[Mapping[str, object]], index: int) -> Sequence[Mapping[str, object]]:
    start: Final = next(
        (j + 1 for j in range(index - 1, -1, -1) if not is_system_role_message(messages[j])),
        0,
    )
    return messages[start:index]


def system_run_end(messages: Sequence[Mapping[str, object]], index: int) -> int:
    return next(
        (j for j in range(index, len(messages)) if not is_system_role_message(messages[j])),
        len(messages),
    )


def reordered_around_tool_results(
    messages: Sequence[Mapping[str, object]], index: int
) -> tuple[Mapping[str, object], ...]:
    message: Final = messages[index]
    if opens_with_tool_results(message):
        return (message, *system_run_before(messages, index))
    if not is_system_role_message(message):
        return (message,)
    run_end: Final = system_run_end(messages, index)
    follower: Final = messages[run_end] if run_end < len(messages) else None
    return () if opens_with_tool_results(follower) else (message,)


def system_turns_after_tool_results(
    messages: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        message for index in range(len(messages)) for message in reordered_around_tool_results(messages, index)
    )


def convert_mid_conversation_system_turns(
    messages: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        system_role_message_as_user(m) if is_system_role_message(m) else m
        for m in system_turns_after_tool_results(messages)
    )
