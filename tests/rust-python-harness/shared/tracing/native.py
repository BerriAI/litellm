from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict

from .profiler import FunctionTraceEvent


class _TraceEventPayload(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    id: int
    parent_id: int | None
    function: str


class TraceResponsePayload(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    response: object
    trace: tuple[_TraceEventPayload, ...] | list[_TraceEventPayload]


def native_trace_events(payload: object) -> tuple[FunctionTraceEvent, ...]:
    response: Final = TraceResponsePayload.model_validate(payload)
    return tuple(FunctionTraceEvent(event.id, event.parent_id, event.function) for event in response.trace)
