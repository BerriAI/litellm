from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

from .profiler import FunctionTraceEvent

Engine = Literal["python", "rust"]


@dataclass(frozen=True, slots=True)
class TraceMapping:
    span: str
    python: re.Pattern[str] | None
    rust: str | None


def mapping(
    *,
    python_frame: str | None = None,
    rust_span: str | None = None,
    span: str | None = None,
) -> TraceMapping:
    if rust_span is None:
        if python_frame is None:
            raise ValueError("mapping needs a python_frame pattern, a rust_span name, or both")
        if span is None:
            raise ValueError("a python-only mapping needs an explicit span to compare under")
        return TraceMapping(span, re.compile(python_frame), None)
    if python_frame is None:
        return TraceMapping(rust_span, None, rust_span)
    if span is not None and span != rust_span:
        raise ValueError(f"span {span!r} disagrees with rust_span {rust_span!r}")
    return TraceMapping(rust_span, re.compile(python_frame), rust_span)


@dataclass(frozen=True, slots=True)
class TraceContract:
    unordered_children_of: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class PipelineStep:
    id: int
    parent_id: int | None
    span: str
    raw: str


@dataclass(frozen=True, slots=True)
class PipelineProjection:
    steps: tuple[PipelineStep, ...] = ()
    unmatched: int = 0


def _span_for(engine: Engine, function: str, mappings: Sequence[TraceMapping]) -> str | None:
    matches: Final = tuple(
        item.span
        for item in mappings
        if (
            engine == "python"
            and item.python is not None
            and item.python.search(function)
            or engine == "rust"
            and item.rust == function
        )
    )
    if len(matches) > 1:
        raise ValueError(f"{engine} event {function!r} matches multiple trace mappings: {matches}")
    if matches:
        return matches[0]
    return function if engine == "rust" else None


def pipeline_projection(
    engine: Engine, events: Sequence[FunctionTraceEvent], mappings: Sequence[TraceMapping]
) -> PipelineProjection:
    raw_parents: dict[int, int | None] = {}
    projected_ids: set[int] = set()
    shown: list[PipelineStep] = []
    unmatched: int = 0
    for event in events:
        if event.id in raw_parents:
            raise ValueError(f"duplicate trace event id {event.id}")
        if event.parent_id is not None and event.parent_id not in raw_parents:
            raise ValueError(f"trace event {event.id} references unknown or later parent {event.parent_id}")
        raw_parents[event.id] = event.parent_id
        span = _span_for(engine, event.function, mappings)
        if span is None:
            unmatched += 1
            continue
        parent_id: int | None = event.parent_id
        while parent_id is not None and parent_id not in projected_ids:
            parent_id = raw_parents[parent_id]
        shown.append(PipelineStep(event.id, parent_id, span, event.raw))
        projected_ids.add(event.id)
    return PipelineProjection(tuple(shown), unmatched)


@dataclass(frozen=True, slots=True)
class TraceNode:
    id: int
    span: str
    children: tuple[TraceNode, ...]


def trace_depths(steps: Sequence[PipelineStep]) -> dict[int, int]:
    depths: dict[int, int] = {}
    for step in steps:
        depths[step.id] = 0 if step.parent_id is None else depths[step.parent_id] + 1
    return depths


def _forest(steps: Sequence[PipelineStep]) -> tuple[TraceNode, ...]:
    children: dict[int | None, list[PipelineStep]] = {}
    known: set[int] = set()
    for step in steps:
        if step.id in known:
            raise ValueError(f"duplicate projected event id {step.id}")
        if step.parent_id is not None and step.parent_id not in known:
            raise ValueError(f"projected event {step.id} references unknown or later parent {step.parent_id}")
        known.add(step.id)
        children.setdefault(step.parent_id, []).append(step)

    def node(step: PipelineStep) -> TraceNode:
        return TraceNode(step.id, step.span, tuple(node(child) for child in children.get(step.id, ())))

    return tuple(node(step) for step in children.get(None, ()))


def _exclusive_spans(engine: Engine, mappings: Sequence[TraceMapping]) -> frozenset[str]:
    return frozenset(
        item.span
        for item in mappings
        if (engine == "python" and item.rust is None) or (engine == "rust" and item.python is None)
    )


def _comparable_steps(
    engine: Engine, steps: Sequence[PipelineStep], mappings: Sequence[TraceMapping]
) -> tuple[PipelineStep, ...]:
    exclusive: Final = _exclusive_spans(engine, mappings)
    raw_parents: Final = {step.id: step.parent_id for step in steps}
    included: Final = {step.id for step in steps if step.span not in exclusive}
    comparable: list[PipelineStep] = []
    for step in steps:
        if step.id not in included:
            continue
        parent_id: int | None = step.parent_id
        while parent_id is not None and parent_id not in included:
            parent_id = raw_parents[parent_id]
        comparable.append(PipelineStep(step.id, parent_id, step.span, step.raw))
    return tuple(comparable)


def _signature(node: TraceNode, contract: TraceContract) -> tuple[object, ...]:
    children: tuple[tuple[object, ...], ...] = tuple(_signature(child, contract) for child in node.children)
    normalized: Final = tuple(sorted(children, key=repr)) if node.span in contract.unordered_children_of else children
    return (node.span, normalized)


def trace_signature(
    engine: Engine,
    steps: Sequence[PipelineStep],
    mappings: Sequence[TraceMapping],
    contract: TraceContract,
) -> tuple[tuple[object, ...], ...]:
    return tuple(_signature(root, contract) for root in _forest(_comparable_steps(engine, steps, mappings)))


@dataclass(frozen=True, slots=True)
class TraceDiff:
    python_only: tuple[str, ...]
    rust_only: tuple[str, ...]
    shared_order_matches: bool
    missing_mappings: tuple[str, ...] = ()
    first_difference: str | None = None

    @property
    def matches(self) -> bool:
        return (
            not self.python_only
            and not self.rust_only
            and not self.missing_mappings
            and self.shared_order_matches
        )


def _missing_mappings(
    python: Sequence[PipelineStep], rust: Sequence[PipelineStep], mappings: Sequence[TraceMapping]
) -> tuple[str, ...]:
    python_seen: Final = frozenset(step.span for step in python)
    rust_seen: Final = frozenset(step.span for step in rust)
    return tuple(
        item.span
        for item in mappings
        if (item.python is not None and item.span not in python_seen)
        or (item.rust is not None and item.span not in rust_seen)
    )


def _first_difference(
    python: Sequence[PipelineStep],
    rust: Sequence[PipelineStep],
    mappings: Sequence[TraceMapping],
    contract: TraceContract,
) -> str | None:
    python_forest: Final = _forest(_comparable_steps("python", python, mappings))
    rust_forest: Final = _forest(_comparable_steps("rust", rust, mappings))

    def compare_children(
        python_nodes: Sequence[TraceNode], rust_nodes: Sequence[TraceNode], path: str, *, unordered: bool
    ) -> str | None:
        if unordered:
            python_signatures: Final = Counter(_signature(node, contract) for node in python_nodes)
            rust_signatures: Final = Counter(_signature(node, contract) for node in rust_nodes)
            if python_signatures != rust_signatures:
                return f"{path}: unordered child subtree multiset differs"
            return None
        for index in range(max(len(python_nodes), len(rust_nodes))):
            child_path = f"{path}/child[{index + 1}]"
            if index >= len(python_nodes):
                return f"{child_path}: Rust has extra {rust_nodes[index].span!r}"
            if index >= len(rust_nodes):
                return f"{child_path}: Python has extra {python_nodes[index].span!r}"
            python_node = python_nodes[index]
            rust_node = rust_nodes[index]
            if python_node.span != rust_node.span:
                return f"{child_path}: Python={python_node.span!r}, Rust={rust_node.span!r}"
            difference = compare_children(
                python_node.children,
                rust_node.children,
                f"{child_path}/{python_node.span}",
                unordered=python_node.span in contract.unordered_children_of,
            )
            if difference is not None:
                return difference
        return None

    return compare_children(python_forest, rust_forest, "root", unordered=False)


def trace_diff(
    python: Sequence[PipelineStep],
    rust: Sequence[PipelineStep],
    mappings: Sequence[TraceMapping] = (),
    contract: TraceContract = TraceContract(),
) -> TraceDiff:
    python_comparable: Final = _comparable_steps("python", python, mappings)
    rust_comparable: Final = _comparable_steps("rust", rust, mappings)
    python_spans: Final = tuple(step.span for step in python_comparable)
    rust_spans: Final = tuple(step.span for step in rust_comparable)
    python_counts: Final = Counter(python_spans)
    rust_counts: Final = Counter(rust_spans)
    python_only_counts: Final = python_counts - rust_counts
    rust_only_counts: Final = rust_counts - python_counts
    python_only: Final = tuple(
        span for span, count in python_only_counts.items() for _ in range(count)
    )
    rust_only: Final = tuple(span for span, count in rust_only_counts.items() for _ in range(count))
    first_difference: Final = _first_difference(python, rust, mappings, contract)
    return TraceDiff(
        python_only=python_only,
        rust_only=rust_only,
        shared_order_matches=bool(python_comparable or rust_comparable) and first_difference is None,
        missing_mappings=_missing_mappings(python, rust, mappings),
        first_difference=first_difference,
    )
