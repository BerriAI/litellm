"""
Type definitions for LangGraph API.

LangGraph provides a streaming and non-streaming API for running agents.
"""

from typing import Any, Dict, List, Optional

from typing_extensions import Literal, TypedDict


# Request Types
class LangGraphMessage(TypedDict, total=False):
    """Message format for LangGraph input."""

    role: Literal["human", "assistant", "system"]
    content: str


class LangGraphInput(TypedDict, total=False):
    """Input structure for LangGraph request."""

    messages: List[LangGraphMessage]


class LangGraphRequest(TypedDict, total=False):
    """Request structure for LangGraph API."""

    assistant_id: str
    input: LangGraphInput
    stream_mode: Optional[str]
    config: Optional[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]]


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
    id: Optional[str]
    name: Optional[str]


class LangGraphResponse(TypedDict, total=False):
    """Non-streaming response structure from LangGraph."""

    messages: List[LangGraphResponseMessage]
    values: Dict[str, Any]


# Parsed response for internal use
class LangGraphParsedResponse(TypedDict):
    """Parsed response from LangGraph."""

    content: str
    role: str
    usage: Optional[Dict[str, int]]

