from __future__ import annotations

from collections.abc import Iterator
from difflib import SequenceMatcher
from typing import Final

from tests.sdk_function_trace.profiler import FunctionTraceEvent


def _aligned_rows(
    python: tuple[FunctionTraceEvent, ...], rust: tuple[FunctionTraceEvent, ...]
) -> Iterator[tuple[FunctionTraceEvent | None, FunctionTraceEvent | None]]:
    matcher: Final = SequenceMatcher(
        a=tuple(event.function for event in python),
        b=tuple(event.function for event in rust),
        autojunk=False,
    )
    for tag, python_start, python_end, rust_start, rust_end in matcher.get_opcodes():
        if tag == "equal":
            yield from zip(python[python_start:python_end], rust[rust_start:rust_end])
        else:
            yield from ((event, None) for event in python[python_start:python_end])
            yield from ((None, event) for event in rust[rust_start:rust_end])


def _label(event: FunctionTraceEvent | None) -> str:
    return f"{'  ' * event.depth}{event.function}" if event is not None else ""


def _status(
    python: FunctionTraceEvent | None,
    rust: FunctionTraceEvent | None,
    python_names: frozenset[str],
    rust_names: frozenset[str],
) -> tuple[str, str]:
    if python is not None and rust is not None:
        return "match", "\033[32m"
    if python is not None:
        return ("reordered", "\033[31m") if python.function in rust_names else ("python only", "\033[34m")
    if rust is not None:
        return ("reordered", "\033[31m") if rust.function in python_names else ("rust only", "\033[33m")
    return "", ""


def format_trace_table(
    python: tuple[FunctionTraceEvent, ...],
    rust: tuple[FunctionTraceEvent, ...],
    *,
    colorize: bool,
) -> str:
    python_header: Final = f"python ({len(python)} steps)"
    rust_header: Final = f"rust ({len(rust)} steps)"
    python_width: Final = max(len(python_header), *(len(_label(event)) for event in python), 0)
    rust_width: Final = max(len(rust_header), *(len(_label(event)) for event in rust), 0)
    python_names: Final = frozenset(event.function for event in python)
    rust_names: Final = frozenset(event.function for event in rust)
    border: Final = f"+-{'-' * python_width}-+-{'-' * rust_width}-+-------------+"
    rows: Final = tuple(
        f"{color}{line}\033[0m" if colorize else line
        for left, right in _aligned_rows(python, rust)
        for status, color in (_status(left, right, python_names, rust_names),)
        for line in (f"| {_label(left):<{python_width}} | {_label(right):<{rust_width}} | {status:<11} |",)
    )
    return "\n".join(
        (
            border,
            f"| {python_header:<{python_width}} | {rust_header:<{rust_width}} | {'comparison':<11} |",
            border,
            *rows,
            border,
        )
    )
