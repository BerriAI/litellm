from __future__ import annotations

from typing import Final

import pytest

from .profiler import FunctionTraceEvent
from .steps import pipeline_projection, trace_depths


def event(event_id: int, function: str, parent_id: int | None = None) -> FunctionTraceEvent:
    return FunctionTraceEvent(event_id, parent_id, function)


def test_projection_keeps_every_call_and_parent() -> None:
    events: Final = (
        event(0, "module.py:1 entry"),
        event(1, "module.py:2 internal_helper", 0),
        event(2, "module.py:3 nested", 1),
        event(3, "module.py:2 internal_helper", 0),
    )

    steps: Final = pipeline_projection(events)

    assert tuple((step.id, step.parent_id, step.span, step.raw) for step in steps) == tuple(
        (item.id, item.parent_id, item.function, item.raw) for item in events
    )


def test_projection_preserves_repeated_occurrences() -> None:
    steps: Final = pipeline_projection((event(0, "route"), event(1, "http", 0), event(2, "http", 0)))
    assert [step.span for step in steps] == ["route", "http", "http"]


def test_projection_preserves_multiple_roots() -> None:
    steps: Final = pipeline_projection((event(0, "route"), event(1, "request")))
    assert trace_depths(steps) == {0: 0, 1: 0}


def test_projection_rejects_duplicate_and_unknown_parent_ids() -> None:
    with pytest.raises(ValueError, match="duplicate trace event id"):
        pipeline_projection((event(0, "route"), event(0, "request")))
    with pytest.raises(ValueError, match="unknown or later parent"):
        pipeline_projection((event(1, "request", 0),))
