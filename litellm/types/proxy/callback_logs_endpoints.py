"""
Types for the callback-logs ingest endpoint (POST /v1/callbacks/logs).

External producers (e.g. the litellm-rust gateway) POST finished logging
payloads here; the proxy replays them through the standard callback fan-out.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from litellm.constants import MAX_CALLBACK_LOG_RECORDS


class CallbackLogRecord(BaseModel):
    """A single finished logging event to replay through the callbacks."""

    status: Literal["success", "failure"]
    standard_logging_payload: dict[str, Any]
    error: Optional[str] = None


class CallbackLogsRequest(BaseModel):
    """A batch of logging events posted by an external producer."""

    # Bounded so one POST can't trigger an unbounded callback/DB fan-out (each
    # record fires every registered integration). Over the cap → 422.
    records: list[CallbackLogRecord] = Field(..., max_length=MAX_CALLBACK_LOG_RECORDS)


class CallbackLogFailure(BaseModel):
    """A record that failed to replay, identified by its index in the batch."""

    index: int
    error: str


class CallbackLogsResponse(BaseModel):
    """Per-batch result: counts plus per-record failure detail so the caller can
    distinguish a transient callback error from a structurally bad payload."""

    processed: int
    failed: int
    failures: list[CallbackLogFailure] = Field(default_factory=list)
