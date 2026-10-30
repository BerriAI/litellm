from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError


class _Event(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class AgentMessage(_Event):
    type: Literal["agent_message"]
    text: str


class Reasoning(_Event):
    type: Literal["reasoning"]
    text: str = ""


class CommandExecution(_Event):
    type: Literal["command_execution"]
    command: str
    aggregated_output: str = ""
    exit_code: int | None = None
    status: str


class FileUpdate(_Event):
    path: str
    kind: str


class FileChange(_Event):
    type: Literal["file_change"]
    changes: tuple[FileUpdate, ...]
    status: str


class WebSearch(_Event):
    type: Literal["web_search"]
    query: str = ""


class McpToolCall(_Event):
    type: Literal["mcp_tool_call"]
    server: str = ""
    tool: str = ""
    status: str = ""


class TodoList(_Event):
    type: Literal["todo_list"]


class ItemError(_Event):
    type: Literal["error"]
    message: str


class UnknownItem(_Event):
    type: str


type Item = Annotated[
    AgentMessage
    | Reasoning
    | CommandExecution
    | FileChange
    | WebSearch
    | McpToolCall
    | TodoList
    | ItemError
    | UnknownItem,
    Field(union_mode="left_to_right"),
]


class Usage(_Event):
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0


class ErrorDetail(_Event):
    message: str


class ThreadStarted(_Event):
    type: Literal["thread.started"]
    thread_id: str


class TurnStarted(_Event):
    type: Literal["turn.started"]


class TurnCompleted(_Event):
    type: Literal["turn.completed"]
    usage: Usage


class TurnFailed(_Event):
    type: Literal["turn.failed"]
    error: ErrorDetail


class ItemEvent(_Event):
    type: Literal["item.started", "item.updated", "item.completed"]
    item: Item


class StreamError(_Event):
    type: Literal["error"]
    message: str


class UnknownEvent(_Event):
    type: str


type CodexEvent = Annotated[
    ThreadStarted | TurnStarted | TurnCompleted | TurnFailed | ItemEvent | StreamError | UnknownEvent,
    Field(union_mode="left_to_right"),
]

EVENT: Final = TypeAdapter[CodexEvent](CodexEvent)


@dataclass(frozen=True, slots=True)
class UnparsedLine:
    text: str
    error: str


def parse_line(line: str) -> CodexEvent | UnparsedLine:
    try:
        return EVENT.validate_json(line)
    except ValidationError as error:
        return UnparsedLine(text=line, error=str(error))


@dataclass(frozen=True, slots=True)
class Transcript:
    events: tuple[CodexEvent, ...]
    unparsed: tuple[UnparsedLine, ...]

    @property
    def thread_id(self) -> str | None:
        return next((event.thread_id for event in self.events if isinstance(event, ThreadStarted)), None)

    @property
    def completed_items(self) -> tuple[Item, ...]:
        return tuple(
            event.item for event in self.events if isinstance(event, ItemEvent) and event.type == "item.completed"
        )

    @property
    def final_message(self) -> str | None:
        messages: Final = tuple(item.text for item in self.completed_items if isinstance(item, AgentMessage))
        return messages[-1] if messages else None

    @property
    def usage(self) -> Usage | None:
        turns: Final = tuple(event.usage for event in self.events if isinstance(event, TurnCompleted))
        return turns[-1] if turns else None

    @property
    def failure(self) -> str | None:
        turn_failures: Final = tuple(event.error.message for event in self.events if isinstance(event, TurnFailed))
        turn_failure: Final = turn_failures[0] if turn_failures else None
        if turn_failure is not None or self.usage is not None:
            return turn_failure
        stream_errors: Final = tuple(event.message for event in self.events if isinstance(event, StreamError))
        return stream_errors[-1] if stream_errors else None


def parse_transcript(stdout: str) -> Transcript:
    parsed: Final = tuple(parse_line(line) for line in stdout.splitlines() if line.strip())
    return Transcript(
        events=tuple(event for event in parsed if not isinstance(event, UnparsedLine)),
        unparsed=tuple(event for event in parsed if isinstance(event, UnparsedLine)),
    )
