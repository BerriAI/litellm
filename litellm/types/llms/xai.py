from dataclasses import dataclass
from typing import Literal, TypedDict


@dataclass(frozen=True, slots=True)
class RealtimeMeteredUsage:
    """Per-session xAI realtime usage counted from raw wire frames."""

    push_to_talk: bool
    audio_seconds_sent: float
    audio_seconds_received: float
    text_messages: int


class XAIWebSearchFilters(TypedDict, total=False):
    """Filters for XAI web search tool"""

    allowed_domains: list[str] | None  # Max 5 domains
    excluded_domains: list[str] | None  # Max 5 domains


class XAIWebSearchTool(TypedDict, total=False):
    """XAI web search tool configuration"""

    type: Literal["web_search"]
    filters: XAIWebSearchFilters | None
    enable_image_understanding: bool | None


class XAIXSearchTool(TypedDict, total=False):
    """XAI X (Twitter) search tool configuration"""

    type: Literal["x_search"]
    allowed_x_handles: list[str] | None  # Max 10 handles
    excluded_x_handles: list[str] | None  # Max 10 handles
    from_date: str | None  # ISO8601 format: YYYY-MM-DD
    to_date: str | None  # ISO8601 format: YYYY-MM-DD
    enable_image_understanding: bool | None
    enable_video_understanding: bool | None
