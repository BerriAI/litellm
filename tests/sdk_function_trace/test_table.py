from __future__ import annotations

import re
from typing import Final

from tests.sdk_function_trace.profiler import FunctionTraceEvent
from tests.sdk_function_trace.table import format_trace_table


def test_table_aligns_matches_after_missing_steps_and_preserves_indentation() -> None:
    python: Final = (
        FunctionTraceEvent("ocr", 0),
        FunctionTraceEvent("python_helper", 1),
        FunctionTraceEvent("http_request", 2),
    )
    rust: Final = (
        FunctionTraceEvent("ocr", 0),
        FunctionTraceEvent("rust_helper", 1),
        FunctionTraceEvent("http_request", 1),
    )
    output: Final = format_trace_table(python, rust, colorize=False)
    rows: Final = tuple(line.split("|")[1:-1] for line in output.splitlines() if line.startswith("|"))

    assert tuple(tuple(cell.strip() for cell in row) for row in rows) == (
        ("python (3 steps)", "rust (3 steps)", "comparison"),
        ("ocr", "ocr", "match"),
        ("python_helper", "", "python only"),
        ("", "rust_helper", "rust only"),
        ("http_request", "http_request", "match"),
    )
    assert rows[-1][0].startswith("     http_request")
    assert rows[-1][1].startswith("   http_request")
    assert len({len(line) for line in output.splitlines()}) == 1
    assert "\033[" not in output


def test_table_marks_reordered_calls_and_keeps_both_execution_orders() -> None:
    python: Final = tuple(FunctionTraceEvent(name, 0) for name in ("ocr", "map", "validate", "http"))
    rust: Final = tuple(FunctionTraceEvent(name, 0) for name in ("ocr", "validate", "map", "http"))
    output: Final = format_trace_table(python, rust, colorize=True)
    plain: Final = re.sub(r"\033\[[0-9;]*m", "", output)
    rows: Final = tuple(line.split("|")[1:-1] for line in plain.splitlines() if line.startswith("|"))[1:]

    assert tuple(row[0].strip() for row in rows if row[0].strip()) == tuple(event.function for event in python)
    assert tuple(row[1].strip() for row in rows if row[1].strip()) == tuple(event.function for event in rust)
    assert plain.count("reordered") == 2
    assert output.count("\033[31m") == 2
    assert "only" not in output


def test_table_colors_match_and_exclusive_rows_without_changing_alignment() -> None:
    python: Final = (FunctionTraceEvent("ocr", 0), FunctionTraceEvent("python_helper", 1))
    rust: Final = (FunctionTraceEvent("ocr", 0), FunctionTraceEvent("rust_helper", 1))
    colored: Final = format_trace_table(python, rust, colorize=True)

    assert re.sub(r"\033\[[0-9;]*m", "", colored) == format_trace_table(python, rust, colorize=False)
    assert next(line for line in colored.splitlines() if "match" in line).startswith("\033[32m")
    assert next(line for line in colored.splitlines() if "python only" in line).startswith("\033[34m")
    assert next(line for line in colored.splitlines() if "rust only" in line).startswith("\033[33m")


def test_table_handles_empty_traces() -> None:
    output: Final = format_trace_table((), (), colorize=False)

    assert "python (0 steps)" in output
    assert "rust (0 steps)" in output
    assert "match" not in output
