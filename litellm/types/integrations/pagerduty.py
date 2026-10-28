from datetime import datetime
from typing import Literal

from typing_extensions import TypedDict

from litellm.types.utils import StandardLoggingUserAPIKeyMetadata


class LinkDict(TypedDict, total=False):
    href: str
    text: str | None


class ImageDict(TypedDict, total=False):
    src: str
    href: str | None
    alt: str | None


class PagerDutyPayload(TypedDict, total=False):
    summary: str
    timestamp: str | None  # ISO 8601 date-time format
    severity: Literal["critical", "warning", "error", "info"]
    source: str
    component: str | None
    group: str | None
    class_: str | None  # Using class_ since 'class' is a reserved keyword
    custom_details: dict | None


class PagerDutyRequestBody(TypedDict, total=False):
    payload: PagerDutyPayload
    routing_key: str
    event_action: Literal["trigger", "acknowledge", "resolve"]
    dedup_key: str | None
    client: str | None
    client_url: str | None
    links: list[LinkDict] | None
    images: list[ImageDict] | None


class AlertingConfig(TypedDict, total=False):
    """
    Config for alerting thresholds
    """

    # Requests failing threshold
    failure_threshold: int  # Number of requests failing in a window
    failure_threshold_window_seconds: int  # Window in seconds

    # Requests hanging threshold
    hanging_threshold_seconds: (
        float  # Number of seconds of waiting for a response before a request is considered hanging
    )
    hanging_threshold_fails: int  # Number of requests hanging in a window
    hanging_threshold_window_seconds: int  # Window in seconds


class PagerDutyInternalEvent(StandardLoggingUserAPIKeyMetadata, total=False):
    """Simple structure to hold timestamp and error info."""

    failure_event_type: Literal["failed_response", "hanging_response"]
    timestamp: datetime
    error_class: str | None
    error_code: str | None
    error_llm_provider: str | None
