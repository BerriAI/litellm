"""
Type definitions for LangGraph API.

LangGraph provides a streaming and non-streaming API for running agents.
"""

from typing import Any, Literal

from typing_extensions import TypedDict


# Request Types
class LangGraphMessage(TypedDict, total=False):
    """Message format for LangGraph input."""

    role: Literal["human", "assistant", "system"]
    content: str


class LangGraphInput(TypedDict, total=False):
    """Input structure for LangGraph request."""

    messages: list[LangGraphMessage]


class LangGraphRequest(TypedDict, total=False):
    """Request structure for LangGraph API."""

    assistant_id: str
    input: LangGraphInput
    stream_mode: str | None
    config: dict[str, Any] | None
    metadata: dict[str, Any] | None


# Response Types - Streaming
class LangGraphStreamEvent(TypedDict, total=False):
    """Single event in a LangGraph stream response."""

    event: str
    data: Any


# Response Types - Non-streaming
class LangGraphResponseMessage(TypedDict, total=False):
    """Message in LangGraph response."""

    type: str
    content: str
    id: str | None
    name: str | None


class LangGraphResponse(TypedDict, total=False):
    """Non-streaming response structure from LangGraph."""

    messages: list[LangGraphResponseMessage]
    values: dict[str, Any]


# Parsed response for internal use
class LangGraphParsedResponse(TypedDict):
    """Parsed response from LangGraph."""

    content: str
    role: str
    usage: dict[str, int] | None
