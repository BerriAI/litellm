"""
Type definitions for the Command Code API.

Command Code (https://api.commandcode.ai) exposes a custom streaming
generation endpoint at POST /alpha/generate. The wire protocol is not
OpenAI-compatible: OpenAI-style params are nested under ``params`` and the
response is a newline-delimited stream of typed JSON events.
"""

from __future__ import annotations

from typing_extensions import Literal, TypedDict


# Request Types
class CommandCodeConfigBlock(TypedDict):
    """CLI-workspace metadata block sent with every request.

    A gateway has no meaningful values for these fields, so LiteLLM sends
    neutral defaults.
    """

    workingDir: str
    date: str
    environment: str
    structure: list[str]
    isGitRepo: bool
    currentBranch: str
    mainBranch: str
    gitStatus: str
    recentCommits: list[str]


class CommandCodeTool(TypedDict, total=False):
    """Tool definition format for Command Code.

    Note: uses ``input_schema`` (a JSON schema), not OpenAI's nested
    ``function.parameters``.
    """

    type: Literal["function"]
    name: str
    description: str | None
    input_schema: dict[str, object]


class CommandCodeParamsBlock(TypedDict, total=False):
    """OpenAI-style generation params, nested under ``params``."""

    model: str
    messages: list[dict[str, object]]
    tools: list[CommandCodeTool]
    system: str
    max_tokens: int
    temperature: float
    stream: bool


class CommandCodeRequestBody(TypedDict, total=False):
    """Request body for POST /alpha/generate."""

    config: CommandCodeConfigBlock
    memory: str | None
    taste: str | None
    skills: str | None
    threadId: str
    params: CommandCodeParamsBlock


# Response Types - Streaming
class CommandCodeTextDeltaEvent(TypedDict, total=False):
    """Incremental assistant text."""

    type: Literal["text-delta"]
    text: str


class CommandCodeReasoningStartEvent(TypedDict, total=False):
    """Marks the start of a reasoning block."""

    type: Literal["reasoning-start"]


class CommandCodeReasoningDeltaEvent(TypedDict, total=False):
    """Incremental reasoning text."""

    type: Literal["reasoning-delta"]
    text: str


class CommandCodeReasoningEndEvent(TypedDict, total=False):
    """Marks the end of a reasoning block."""

    type: Literal["reasoning-end"]


class CommandCodeToolCallEvent(TypedDict, total=False):
    """A complete tool call. Arrives whole, not incrementally.

    ``input`` may arrive as a dict or as a JSON-encoded string; some
    responses use ``args`` or ``arguments`` instead of ``input``.
    """

    type: Literal["tool-call"]
    toolCallId: str
    toolName: str
    input: dict[str, object] | str
    args: dict[str, object] | str
    arguments: dict[str, object] | str


class CommandCodeToolResultEvent(TypedDict, total=False):
    """Server-side tool result echo. Ignored by LiteLLM."""

    type: Literal["tool-result"]
    toolCallId: str


class CommandCodeInputTokenDetails(TypedDict, total=False):
    """Cache token breakdown nested inside the usage object."""

    cacheReadTokens: int
    cacheWriteTokens: int


class CommandCodeUsage(TypedDict, total=False):
    """Usage object carried on the finish event (``totalUsage``)."""

    inputTokens: int
    outputTokens: int
    inputTokenDetails: CommandCodeInputTokenDetails


class CommandCodeFinishEvent(TypedDict, total=False):
    """Terminal event carrying the finish reason and usage."""

    type: Literal["finish"]
    finishReason: str
    totalUsage: CommandCodeUsage


class CommandCodeErrorEvent(TypedDict, total=False):
    """Stream-level error event. ``error`` may be an object or a string."""

    type: Literal["error"]
    error: dict[str, object] | str
