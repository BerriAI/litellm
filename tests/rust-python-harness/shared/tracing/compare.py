from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class Operation:
    name: str
    started: int
    finished: int


def compare_traces(
    python: Sequence[Operation],
    rust: Sequence[Operation],
    mapping: Mapping[str, str],
    required_order: Sequence[tuple[str, str]] = (),
) -> tuple[str, ...]:
    python_names: Final = {operation.name for operation in python}
    rust_names: Final = {operation.name for operation in rust}
    problems: Final = (
        *(f"unmapped Python operation: {name}" for name in sorted(python_names - mapping.keys())),
        *(f"unmapped Rust operation: {name}" for name in sorted(rust_names - set(mapping.values()))),
        *(f"ambiguous Rust operation: {name}" for name, count in Counter(mapping.values()).items() if count > 1),
        *(
            f"invalid interval: {operation.name}"
            for operation in (*python, *rust)
            if operation.started > operation.finished
        ),
    )
    if problems:
        return problems
    python_counts: Final = Counter(operation.name for operation in python)
    rust_counts: Final = Counter(operation.name for operation in rust)
    counts: Final = tuple(
        f"call count differs for {name}: Python={python_counts[name]}, Rust={rust_counts[target]}"
        for name, target in mapping.items()
        if python_counts[name] != rust_counts[target]
    )
    ordering: Final = tuple(
        f"{label}: required order {before} before {after} was not observed"
        for before, after in required_order
        for label, operations, first, second in (
            ("Python", python, before, after),
            ("Rust", rust, mapping.get(before), mapping.get(after)),
        )
        if not first
        or not second
        or not any(operation.name == first for operation in operations)
        or not any(operation.name == second for operation in operations)
        or max(operation.finished for operation in operations if operation.name == first)
        > min(operation.started for operation in operations if operation.name == second)
    )
    return (*counts, *ordering)
