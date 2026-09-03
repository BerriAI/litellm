from __future__ import annotations

from typing import Final

import pytest

from .profiler import FunctionTraceEvent
from .steps import pipeline_issues, pipeline_steps, step, trace_diff

STEPS: Final = (
    step("route", r"entry$", "route"),
    step("provider", r"provider$", "provider"),
    step("request", r"request$", "request"),
    step("http", r"post$", "http"),
    step("response", r"response$", "response"),
)
EDGES: Final = (("route", "provider"), ("provider", "request"), ("request", "http"), ("http", "response"))


def test_python_projection_keeps_pipeline_and_drops_noise() -> None:
    events: Final = (
        FunctionTraceEvent("noise", 0),
        FunctionTraceEvent("module.py:1 entry", 1),
        FunctionTraceEvent("module.py:2 provider", 2),
        FunctionTraceEvent("module.py:3 request", 2),
        FunctionTraceEvent("client.py:4 post", 3),
        FunctionTraceEvent("module.py:5 response", 2),
    )
    assert pipeline_steps("python", events, STEPS) == (
        FunctionTraceEvent("route", 0),
        FunctionTraceEvent("provider", 1),
        FunctionTraceEvent("request", 1),
        FunctionTraceEvent("http", 2),
        FunctionTraceEvent("response", 1),
    )


def test_rust_projection_keeps_unknown_spans() -> None:
    events: Final = (FunctionTraceEvent("route", 0), FunctionTraceEvent("new_span", 1))
    assert pipeline_steps("rust", events, STEPS) == events


def test_projection_keeps_only_first_occurrence() -> None:
    events: Final = (FunctionTraceEvent("route", 0), FunctionTraceEvent("route", 0))
    assert pipeline_steps("rust", events, STEPS) == (FunctionTraceEvent("route", 0),)


def test_projection_resets_depth_on_thread_root() -> None:
    events: Final = (FunctionTraceEvent("route", 1), FunctionTraceEvent("request", 0))
    assert pipeline_steps("rust", events, STEPS) == (
        FunctionTraceEvent("route", 0),
        FunctionTraceEvent("request", 0),
    )


def test_projection_uses_actual_ancestors_after_coroutine_resumption() -> None:
    events: Final = (
        FunctionTraceEvent("module.py:1 entry", 0, ()),
        FunctionTraceEvent("module.py:2 provider", 1, ("module.py:1 entry",)),
        FunctionTraceEvent("module.py:3 response", 1, ("module.py:2 provider",)),
    )
    assert pipeline_steps("python", events, STEPS)[-1] == FunctionTraceEvent("response", 2)


def test_projection_does_not_nest_siblings_under_returned_call() -> None:
    events: Final = (
        FunctionTraceEvent("module.py:1 entry", 0, ()),
        FunctionTraceEvent("module.py:2 provider", 1, ("module.py:1 entry",)),
        FunctionTraceEvent("module.py:3 request", 1, ("module.py:1 entry",)),
    )
    assert pipeline_steps("python", events, STEPS)[-1] == FunctionTraceEvent("request", 1)


@pytest.mark.parametrize("missing", ("route", "provider", "request", "http", "response"))
def test_pipeline_check_rejects_missing_stages(missing: str) -> None:
    events: Final = tuple(FunctionTraceEvent(item.name, 0) for item in STEPS if item.name != missing)
    assert f"missing {missing}" in pipeline_issues("rust", events, STEPS, EDGES)


def test_pipeline_check_rejects_reordered_stages() -> None:
    events: Final = tuple(FunctionTraceEvent(name, 0) for name in ("route", "provider", "http", "request", "response"))
    assert "request must precede http" in pipeline_issues("rust", events, STEPS, EDGES)


def test_trace_diff_matches_identical_steps() -> None:
    events: Final = (FunctionTraceEvent("route", 0), FunctionTraceEvent("request", 1))
    assert trace_diff(events, events).matches


def test_trace_diff_reports_exclusive_steps() -> None:
    diff: Final = trace_diff((FunctionTraceEvent("python", 0),), (FunctionTraceEvent("rust", 0),))
    assert diff.python_only == ("python",)
    assert diff.rust_only == ("rust",)
    assert not diff.matches


def test_trace_diff_reports_reordered_shared_steps() -> None:
    first: Final = FunctionTraceEvent("first", 0)
    second: Final = FunctionTraceEvent("second", 0)
    assert not trace_diff((first, second), (second, first)).shared_order_matches


def test_trace_diff_does_not_claim_empty_traces_match() -> None:
    assert not trace_diff((), ()).matches


def test_pipeline_edges_only_apply_when_both_steps_ran() -> None:
    events: Final = (FunctionTraceEvent("route", 0),)
    assert "route must precede provider" not in pipeline_issues("python", events, STEPS, EDGES)
