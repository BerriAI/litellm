from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .profiler import FunctionTraceEvent


@dataclass(frozen=True, slots=True)
class PipelineStep:
    id: int
    parent_id: int | None
    span: str
    raw: str


def pipeline_projection(events: Sequence[FunctionTraceEvent]) -> tuple[PipelineStep, ...]:
    raw_parents: dict[int, int | None] = {}
    projected_ids: set[int] = set()
    shown: list[PipelineStep] = []
    for event in events:
        if event.id in raw_parents:
            raise ValueError(f"duplicate trace event id {event.id}")
        if event.parent_id is not None and event.parent_id not in raw_parents:
            raise ValueError(f"trace event {event.id} references unknown or later parent {event.parent_id}")
        raw_parents[event.id] = event.parent_id
        parent_id: int | None = event.parent_id
        while parent_id is not None and parent_id not in projected_ids:
            parent_id = raw_parents[parent_id]
        shown.append(PipelineStep(event.id, parent_id, event.function, event.raw))
        projected_ids.add(event.id)
    return tuple(shown)


def trace_depths(steps: Sequence[PipelineStep]) -> dict[int, int]:
    depths: dict[int, int] = {}
    for step in steps:
        depths[step.id] = 0 if step.parent_id is None else depths[step.parent_id] + 1
    return depths
