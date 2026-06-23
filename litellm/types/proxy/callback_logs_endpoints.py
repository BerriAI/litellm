"""
Types for the callback-logs ingest endpoint (POST /v1/callbacks/logs).

External producers (e.g. the litellm-rust gateway) POST finished logging
payloads here; the proxy replays them through the standard callback fan-out.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel


class CallbackLogRecord(BaseModel):
    """A single finished logging event to replay through the callbacks."""

    status: Literal["success", "failure"]
    standard_logging_payload: dict[str, Any]
    error: Optional[str] = None


class CallbackLogsRequest(BaseModel):
    """A batch of logging events posted by an external producer."""

    records: list[CallbackLogRecord]


class CallbackLogsResponse(BaseModel):
    """Per-batch result: how many records replayed and how many failed."""

    processed: int
    failed: int
