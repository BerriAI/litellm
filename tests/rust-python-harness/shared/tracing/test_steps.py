from __future__ import annotations

from typing import Final

import pytest

from .profiler import FunctionTraceEvent
from .steps import Engine, TraceContract, mapping, pipeline_projection, trace_depths, trace_diff

MAPPINGS: Final = (
    mapping(rust_span="route", python_frame=r"entry$"),
    mapping(rust_span="provider", python_frame=r"provider$"),
    mapping(rust_span="request", python_frame=r"request$"),
    mapping(rust_span="http", python_frame=r"post$"),
    mapping(rust_span="response", python_frame=r"response$"),
)


def event(event_id: int, function: str, parent_id: int | None = None) -> FunctionTraceEvent:
    return FunctionTraceEvent(event_id, parent_id, function)


def test_python_projection_collapses_unmapped_parents_and_counts_noise() -> None:
    events: Final = (
        event(0, "module.py:1 entry"),
        event(1, "noise", 0),
        event(2, "module.py:2 provider", 1),
        event(3, "module.py:3 request", 0),
        event(4, "client.py:4 post", 3),
        event(5, "module.py:5 response", 0),
    )
    projection: Final = pipeline_projection("python", events, MAPPINGS)
    assert projection.unmatched == 1
    assert [(step.id, step.parent_id, step.span, step.raw) for step in projection.steps] == [
        (0, None, "route", "module.py:1 entry"),
        (2, 0, "provider", "module.py:2 provider"),
        (3, 0, "request", "module.py:3 request"),
        (4, 3, "http", "client.py:4 post"),
        (5, 0, "response", "module.py:5 response"),
    ]


def test_rust_projection_keeps_unknown_spans() -> None:
    projection: Final = pipeline_projection("rust", (event(0, "route"), event(1, "new_span", 0)), MAPPINGS)
    assert [(step.span, step.parent_id) for step in projection.steps] == [("route", None), ("new_span", 0)]


def test_projection_preserves_repeated_occurrences() -> None:
    projection: Final = pipeline_projection(
        "rust",
        (event(0, "route"), event(1, "http", 0), event(2, "http", 0)),
        MAPPINGS,
    )
    assert [step.span for step in projection.steps] == ["route", "http", "http"]


def test_projection_preserves_multiple_roots() -> None:
    projection: Final = pipeline_projection("rust", (event(0, "route"), event(1, "request")), MAPPINGS)
    assert trace_depths(projection.steps) == {0: 0, 1: 0}


def test_projection_rejects_duplicate_and_unknown_parent_ids() -> None:
    with pytest.raises(ValueError, match="duplicate trace event id"):
        pipeline_projection("rust", (event(0, "route"), event(0, "request")), MAPPINGS)
    with pytest.raises(ValueError, match="unknown or later parent"):
        pipeline_projection("rust", (event(1, "request", 0),), MAPPINGS)


@pytest.mark.parametrize("engine", ("python", "rust"))
def test_rust_only_mappings_do_not_swallow_python_frames(engine: Engine) -> None:
    projection: Final = pipeline_projection(
        engine,
        (event(0, "anything"),),
        (mapping(rust_span="rust_only_span"),),
    )
    if engine == "python":
        assert projection.unmatched == 1
        assert projection.steps == ()
    else:
        assert projection.unmatched == 0
        assert projection.steps[0].span == "anything"


def test_mapping_builder_rejects_empty_and_ambiguous_declarations() -> None:
    with pytest.raises(ValueError, match="mapping needs"):
        mapping()
    with pytest.raises(ValueError, match="python-only mapping needs"):
        mapping(python_frame=r"frame$")
    with pytest.raises(ValueError, match="disagrees with"):
        mapping(rust_span="span_a", python_frame=r"frame$", span="span_b")


def test_projection_rejects_ambiguous_python_mapping() -> None:
    mappings: Final = (
        mapping(rust_span="first", python_frame=r"same$"),
        mapping(rust_span="second", python_frame=r"same$"),
    )
    with pytest.raises(ValueError, match="multiple trace mappings"):
        pipeline_projection("python", (event(0, "module.py:1 same"),), mappings)


def test_trace_diff_matches_identical_occurrence_trees() -> None:
    mappings: Final = (MAPPINGS[0], MAPPINGS[2])
    steps: Final = pipeline_projection(
        "rust", (event(0, "route"), event(1, "request", 0), event(2, "request", 0)), mappings
    ).steps
    assert trace_diff(steps, steps, mappings).matches


def test_trace_diff_rejects_missing_occurrence_and_parent_drift() -> None:
    python: Final = pipeline_projection(
        "rust", (event(0, "route"), event(1, "request", 0), event(2, "request", 0)), MAPPINGS
    ).steps
    missing: Final = pipeline_projection("rust", (event(0, "route"), event(1, "request", 0)), MAPPINGS).steps
    reparented: Final = pipeline_projection(
        "rust", (event(0, "route"), event(1, "request", 0), event(2, "request", 1)), MAPPINGS
    ).steps
    assert trace_diff(python, missing, MAPPINGS).python_only == ("request",)
    assert not trace_diff(python, reparented, MAPPINGS).matches


def test_trace_diff_rejects_sequential_reorder() -> None:
    first: Final = pipeline_projection(
        "rust", (event(0, "route"), event(1, "request", 0), event(2, "response", 0)), MAPPINGS
    ).steps
    second: Final = pipeline_projection(
        "rust", (event(0, "route"), event(1, "response", 0), event(2, "request", 0)), MAPPINGS
    ).steps
    diff: Final = trace_diff(first, second, MAPPINGS)
    assert not diff.matches
    assert diff.first_difference == "root/child[1]/route/child[1]: Python='request', Rust='response'"


def test_trace_diff_allows_reordered_concurrent_children() -> None:
    mappings: Final = (MAPPINGS[0], MAPPINGS[2], MAPPINGS[4])
    first: Final = pipeline_projection(
        "rust", (event(0, "route"), event(1, "request", 0), event(2, "response", 0)), mappings
    ).steps
    second: Final = pipeline_projection(
        "rust", (event(0, "route"), event(1, "response", 0), event(2, "request", 0)), mappings
    ).steps
    contract: Final = TraceContract(frozenset({"route"}))
    assert trace_diff(first, second, mappings, contract).matches


def test_trace_diff_prunes_declared_engine_only_nodes_but_requires_them() -> None:
    mappings: Final = (MAPPINGS[0], mapping(rust_span="rust_prepare"))
    python: Final = pipeline_projection("python", (event(0, "module.py:1 entry"),), mappings).steps
    rust: Final = pipeline_projection(
        "rust", (event(0, "route"), event(1, "rust_prepare", 0)), mappings
    ).steps
    assert trace_diff(python, rust, mappings).matches
    assert trace_diff(python, rust[:1], mappings).missing_mappings == ("rust_prepare",)


def test_trace_diff_does_not_claim_empty_traces_match() -> None:
    assert not trace_diff((), ()).matches
