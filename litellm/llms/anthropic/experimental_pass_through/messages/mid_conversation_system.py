from collections.abc import Mapping, Sequence
from itertools import groupby
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


def system_run_placed_after_tool_results(
    system_run: Sequence[Mapping[str, object]], follower_run: Sequence[Mapping[str, object]]
) -> tuple[Mapping[str, object], ...]:
    if follower_run and opens_with_tool_results(follower_run[0]):
        return (follower_run[0], *system_run, *follower_run[1:])
    return (*system_run, *follower_run)


def system_turns_after_tool_results(
    messages: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    runs: Final = tuple(tuple(run) for _, run in groupby(messages, key=is_system_role_message))
    if not runs:
        return ()
    first_system_run: Final = 0 if is_system_role_message(runs[0][0]) else 1
    paired_runs: Final = tuple(
        (runs[i], runs[i + 1] if i + 1 < len(runs) else ()) for i in range(first_system_run, len(runs), 2)
    )
    return (
        *(runs[0] if first_system_run else ()),
        *(
            m
            for system_run, follower_run in paired_runs
            for m in system_run_placed_after_tool_results(system_run, follower_run)
        ),
    )


def convert_mid_conversation_system_turns(
    messages: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        system_role_message_as_user(m) if is_system_role_message(m) else m
        for m in system_turns_after_tool_results(messages)
    )
