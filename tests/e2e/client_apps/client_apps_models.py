"""What each client app prints, typed down to the fields the tests read: Claude
Code's stream-json events, Codex's `exec --json` events, the summary
`vercel_ai_turn.mjs` writes, and the request body the proxy stores on a spend row."""

from __future__ import annotations

from typing import Annotated, Final, Literal

from pydantic import BaseModel, Field, JsonValue, TypeAdapter


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"]
    name: str
    input: JsonValue = None


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"]
    content: JsonValue = None
    is_error: bool = False


class OtherBlock(BaseModel):
    type: str


type ContentBlock = Annotated[ToolUseBlock | ToolResultBlock | OtherBlock, Field(union_mode="left_to_right")]


class ClaudeMessage(BaseModel):
    content: list[ContentBlock] | str


class ClaudeAssistantEvent(BaseModel):
    type: Literal["assistant"]
    message: ClaudeMessage


class ClaudeUserEvent(BaseModel):
    type: Literal["user"]
    message: ClaudeMessage


class StreamDelta(BaseModel):
    type: str


class ContentBlockDelta(BaseModel):
    type: Literal["content_block_delta"]
    delta: StreamDelta


class OtherStreamPayload(BaseModel):
    type: str


class ClaudeStreamEvent(BaseModel):
    type: Literal["stream_event"]
    event: Annotated[ContentBlockDelta | OtherStreamPayload, Field(union_mode="left_to_right")]


class ClaudeResultEvent(BaseModel):
    type: Literal["result"]
    subtype: str
    is_error: bool
    result: str | None = None
    num_turns: int | None = None
    total_cost_usd: float | None = None


class ClaudeOtherEvent(BaseModel):
    type: str


type ClaudeEvent = Annotated[
    ClaudeAssistantEvent | ClaudeUserEvent | ClaudeStreamEvent | ClaudeResultEvent | ClaudeOtherEvent,
    Field(union_mode="left_to_right"),
]
CLAUDE_EVENTS: Final = TypeAdapter(list[ClaudeEvent])


class CodexCommandExecution(BaseModel):
    type: Literal["command_execution"]
    command: str
    aggregated_output: str
    exit_code: int | None = None
    status: str


class CodexAgentMessage(BaseModel):
    type: Literal["agent_message"]
    text: str


class CodexOtherItem(BaseModel):
    type: str


type CodexItem = Annotated[
    CodexCommandExecution | CodexAgentMessage | CodexOtherItem, Field(union_mode="left_to_right")
]


class CodexItemCompleted(BaseModel):
    type: Literal["item.completed"]
    item: CodexItem


class CodexUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class CodexTurnCompleted(BaseModel):
    type: Literal["turn.completed"]
    usage: CodexUsage


class CodexTurnFailed(BaseModel):
    type: Literal["turn.failed"]
    error: JsonValue = None


class CodexOtherEvent(BaseModel):
    type: str


type CodexEvent = Annotated[
    CodexItemCompleted | CodexTurnCompleted | CodexTurnFailed | CodexOtherEvent,
    Field(union_mode="left_to_right"),
]
CODEX_EVENTS: Final = TypeAdapter(list[CodexEvent])


type VercelApi = Literal["responses", "chat"]


class VercelToolCall(BaseModel):
    tool_name: str
    input: JsonValue = None


class SecretWord(BaseModel):
    word: str


class VercelToolResult(BaseModel):
    tool_name: str
    output: SecretWord


class VercelTurn(BaseModel):
    api: VercelApi
    model: str
    text_deltas: int
    tool_calls: list[VercelToolCall]
    tool_results: list[VercelToolResult]
    errors: list[str]
    text: str
    steps: int
    finish_reason: str


class RecordedRequest(BaseModel):
    """The client's request body as the proxy stored it on the spend row
    (`store_prompts_in_spend_logs: true`): `input` for a Responses call,
    `messages` for a chat or Anthropic Messages call."""

    stream: bool = False
    input: JsonValue = None
    messages: JsonValue = None
