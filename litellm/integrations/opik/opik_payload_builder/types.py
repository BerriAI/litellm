"""Type definitions for Opik payload building."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal


@dataclass
class TracePayload:
    """Opik trace payload structure"""

    project_name: str
    id: str
    name: str
    start_time: str
    end_time: str
    input: object
    output: object
    metadata: Mapping[str, object]
    tags: list[str]
    thread_id: str | None = None


@dataclass
class SpanPayload:
    """Opik span payload structure"""

    id: str
    project_name: str
    trace_id: str
    name: str
    type: Literal["llm"]
    model: str
    start_time: str
    end_time: str
    input: object
    output: object
    metadata: Mapping[str, object]
    tags: list[str]
    usage: dict[str, int]
    parent_span_id: str | None = None
    provider: str | None = None
    total_cost: float | None = None


PayloadItem = TracePayload | SpanPayload
TraceSpanPayloadTuple: Final = tuple[TracePayload | None, SpanPayload]
