"""Type definitions for Opik payload building."""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union


@dataclass
class TracePayload:
    """Opik trace payload structure"""

    project_name: str
    id: str
    name: str
    start_time: str
    end_time: str
    input: Any
    output: Any
    metadata: Dict[str, Any]
    tags: List[str]
    thread_id: Optional[str] = None


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
    input: Any
    output: Any
    metadata: Dict[str, Any]
    tags: List[str]
    usage: Dict[str, int]
    parent_span_id: Optional[str] = None
    provider: Optional[str] = None
    total_cost: Optional[float] = None


PayloadItem = Union[TracePayload, SpanPayload]
TraceSpanPayloadTuple = Tuple[Optional[TracePayload], SpanPayload]
