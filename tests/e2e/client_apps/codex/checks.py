from __future__ import annotations

from collections.abc import Callable
from typing import Final

from client_apps.codex.events import CommandExecution, Transcript

type Check = Callable[[Transcript], str | None]


def final_message_has(expected: str) -> Check:
    def check(transcript: Transcript) -> str | None:
        message: Final = transcript.final_message
        if message is not None and expected in message.lower():
            return None
        return f"final message {message!r} lacks {expected!r}"

    return check


def all_checks(*checks: Check) -> Check:
    def check(transcript: Transcript) -> str | None:
        return next((error for error in (each(transcript) for each in checks) if error is not None), None)

    return check


def _ran_no_commands(transcript: Transcript) -> str | None:
    commands: Final = tuple(item for item in transcript.completed_items if isinstance(item, CommandExecution))
    return f"read the image through the shell instead of looking at it: {commands!r}" if commands else None


def named_by_eye(color: str) -> Check:
    return all_checks(_ran_no_commands, final_message_has(color))
