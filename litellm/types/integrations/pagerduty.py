from datetime import datetime
from typing import List, Literal, Optional, TypedDict, Union

from litellm.types.utils import StandardLoggingUserAPIKeyMetadata


class LinkDict(TypedDict, total=False):
    href: str
    text: Optional[str]


class ImageDict(TypedDict, total=False):
    src: str
    href: Optional[str]
    alt: Optional[str]


class PagerDutyPayload(TypedDict, total=False):
    summary: str
    timestamp: Optional[str]  # ISO 8601 date-time format
    severity: Literal["critical", "warning", "error", "info"]
    source: str
    component: Optional[str]
    group: Optional[str]
    class_: Optional[str]  # Using class_ since 'class' is a reserved keyword
    custom_details: Optional[dict]


class PagerDutyRequestBody(TypedDict, total=False):
    payload: PagerDutyPayload
    routing_key: str
    event_action: Literal["trigger", "acknowledge", "resolve"]
    dedup_key: Optional[str]
    client: Optional[str]
    client_url: Optional[str]
    links: Optional[List[LinkDict]]
    images: Optional[List[ImageDict]]


class AlertingConfig(TypedDict, total=False):
    """
    Config for alerting thresholds
    """

    # Requests failing threshold
    failure_threshold: int  # Number of requests failing in a window
    failure_threshold_window_seconds: int  # Window in seconds

    # Requests hanging threshold
    hanging_threshold_seconds: float  # Number of seconds of waiting for a response before a request is considered hanging
    hanging_threshold_fails: int  # Number of requests hanging in a window
    hanging_threshold_window_seconds: int  # Window in seconds


class PagerDutyInternalEvent(StandardLoggingUserAPIKeyMetadata, total=False):
    """Simple structure to hold timestamp and error info."""

    failure_event_type: Literal["failed_response", "hanging_response"]
    timestamp: datetime
    error_class: Optional[str]
    error_code: Optional[str]
    error_llm_provider: Optional[str]
