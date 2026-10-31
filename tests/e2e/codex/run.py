"""What one `codex exec --json` run did, read from the JSONL events it printed.

The event shapes are Codex's own (`codex-rs/exec/src/exec_events.rs`). Codex is unpinned in CI, so an event or item
type this module does not model is skipped, while one it does model is validated strictly and fails loudly when
Codex changes it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Annotated, Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Discriminator, Tag, TypeAdapter

_HTTP_STATUS: Final = re.compile(r"\bstatus:? (\d{3})\b")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class TokenUsage(_Frozen):
    input_tokens: int
    cached_input_tokens: int
    cache_write_input_tokens: int = 0
    output_tokens: int
    reasoning_output_tokens: int


class _AgentMessage(_Frozen):
    type: Literal["agent_message"]
    text: str


class _CommandExecution(_Frozen):
    type: Literal["command_execution"]
    command: str
    aggregated_output: str
    exit_code: int | None


class _FileUpdate(_Frozen):
    path: str
    kind: Literal["add", "delete", "update"]


class _FileChange(_Frozen):
    type: Literal["file_change"]
    changes: tuple[_FileUpdate, ...]
    status: Literal["in_progress", "completed", "failed"]


class _ErrorItem(_Frozen):
    type: Literal["error"]
    message: str


class _Unmodelled(_Frozen):
    type: str


class _Message(_Frozen):
    message: str


def _known_or_unmodelled(known: frozenset[str]) -> Callable[[object], str]:
    def tag(raw: object) -> str:
        kind: Final = cast("Mapping[str, object]", raw).get("type") if isinstance(raw, Mapping) else None
        return kind if isinstance(kind, str) and kind in known else "unmodelled"

    return tag


_Item = Annotated[
    Annotated[_AgentMessage, Tag("agent_message")]
    | Annotated[_CommandExecution, Tag("command_execution")]
    | Annotated[_FileChange, Tag("file_change")]
    | Annotated[_ErrorItem, Tag("error")]
    | Annotated[_Unmodelled, Tag("unmodelled")],
    Discriminator(_known_or_unmodelled(frozenset({"agent_message", "command_execution", "file_change", "error"}))),
]


class _ThreadStarted(_Frozen):
    type: Literal["thread.started"]
    thread_id: str


class _ItemCompleted(_Frozen):
    type: Literal["item.completed"]
    item: _Item


class _TurnCompleted(_Frozen):
    type: Literal["turn.completed"]
    usage: TokenUsage


class _TurnFailed(_Frozen):
    type: Literal["turn.failed"]
    error: _Message


class _StreamError(_Frozen):
    type: Literal["error"]
    message: str


_Event = Annotated[
    Annotated[_ThreadStarted, Tag("thread.started")]
    | Annotated[_ItemCompleted, Tag("item.completed")]
    | Annotated[_TurnCompleted, Tag("turn.completed")]
    | Annotated[_TurnFailed, Tag("turn.failed")]
    | Annotated[_StreamError, Tag("error")]
    | Annotated[_Unmodelled, Tag("unmodelled")],
    Discriminator(
        _known_or_unmodelled(frozenset({"thread.started", "item.completed", "turn.completed", "turn.failed", "error"}))
    ),
]
_EVENT: Final = TypeAdapter[_Event](_Event)


@dataclass(frozen=True, slots=True)
class Command:
    command: str
    output: str
    exit_code: int | None


@dataclass(frozen=True, slots=True)
class CodexError:
    message: str
    status_code: int | None


@dataclass(frozen=True, slots=True)
class CodexRun:
    exit_code: int
    thread_id: str | None
    commands: tuple[Command, ...]
    patched_files: tuple[Path, ...]
    answer: str
    usage: TokenUsage | None
    errors: tuple[CodexError, ...]
    stderr: str


def codex_run(*, exit_code: int, stdout: str, stderr: str) -> CodexRun:
    events: Final = tuple(_EVENT.validate_json(line) for line in stdout.splitlines() if line.strip())
    items: Final = tuple(event.item for event in events if isinstance(event, _ItemCompleted))
    patches: Final = tuple(item for item in items if isinstance(item, _FileChange) and item.status == "completed")
    answers: Final = tuple(item.text for item in items if isinstance(item, _AgentMessage))
    usages: Final = tuple(event.usage for event in events if isinstance(event, _TurnCompleted))
    return CodexRun(
        exit_code=exit_code,
        thread_id=next((event.thread_id for event in events if isinstance(event, _ThreadStarted)), None),
        commands=tuple(
            Command(command=item.command, output=item.aggregated_output, exit_code=item.exit_code)
            for item in items
            if isinstance(item, _CommandExecution)
        ),
        patched_files=tuple(Path(change.path) for change in chain.from_iterable(patch.changes for patch in patches)),
        answer=answers[-1] if answers else "",
        usage=usages[-1] if usages else None,
        errors=tuple(_errors(events)),
        stderr=stderr,
    )


def _errors(events: tuple[_Event, ...]) -> Iterator[CodexError]:
    for event in events:
        match event:
            case _TurnFailed(error=_Message(message=message)) | _StreamError(message=message):
                yield _codex_error(message)
            case _ItemCompleted(item=_ErrorItem(message=message)):
                yield _codex_error(message)
            case _:
                pass


def _codex_error(message: str) -> CodexError:
    status: Final = _HTTP_STATUS.search(message)
    return CodexError(message=message, status_code=int(status.group(1)) if status else None)
