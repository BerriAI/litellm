"""Type definitions for Opik payload building."""

from typing import Any, Dict, List, Literal, Optional, TypedDict, Union


class TracePayload(TypedDict, total=False):
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
    thread_id: Optional[str]


class SpanPayload(TypedDict, total=False):
    """Opik span payload structure"""

    id: str
    project_name: str
    trace_id: str
    parent_span_id: Optional[str]
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


PayloadItem = Union[TracePayload, SpanPayload]
OpikPayloadList = List[PayloadItem]
