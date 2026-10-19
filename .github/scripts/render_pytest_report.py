from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import accumulate, groupby
from pathlib import Path
from typing import Final, cast

TOTAL_ORDER: Final = ("failed", "passed", "skipped", "xfailed", "xpassed", "error")
COLUMNS: Final = ("passed", "failed", "error", "skipped", "xfailed", "xpassed")


@dataclass(frozen=True, slots=True)
class Outcome:
    nodeid: str
    category: str
    reason: str


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("expected an object")
    return cast(dict[str, object], value)


def _array(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list):
        raise ValueError("expected an array")
    return tuple(_object(row) for row in cast(list[object], value))


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("expected a string")
    return value


def _outcome(row: dict[str, object]) -> Outcome:
    return Outcome(_string(row["nodeid"]), _string(row["category"]), _string(row["reason"]))


def _cell(value: str | float) -> str:
    raw: Final = str(value)
    clipped: Final = raw[:240] + ("..." if len(raw) > 240 else "")
    return (
        html.escape(" ".join(clipped.split()), quote=True)
        .replace("|", "&#124;")
        .replace("`", "&#96;")
        .replace("[", "&#91;")
        .replace("]", "&#93;")
        .replace("*", "&#42;")
        .replace("_", "&#95;")
        .replace("\\", "&#92;")
    )


def _limited(rows: tuple[str, ...]) -> str:
    candidates: Final = rows[:500]
    sizes: Final = accumulate(len(row.encode("utf-8")) + 1 for row in candidates)
    visible: Final = tuple(row for row, size in zip(candidates, sizes) if size <= 200_000)
    remainder: Final = len(rows) - len(visible)
    return "\n".join(visible + ((f"\nand {remainder} more",) if remainder else ()))


def _file_row(path: str, outcomes: tuple[Outcome, ...]) -> str:
    counts: Final = Counter(outcome.category for outcome in outcomes)
    return "| " + _cell(path) + " | " + " | ".join(str(counts[column]) for column in COLUMNS) + " |"


def _metric_row(row: dict[str, object]) -> str:
    value: Final = row["value"]
    if not isinstance(value, (str, int, float)):
        raise ValueError("invalid metric value")
    return (
        "| "
        + " | ".join(
            _cell(value)
            for value in (
                _string(row["name"]),
                value,
                _string(row["unit"]),
                _string(row["nodeid"]),
            )
        )
        + " |"
    )


def render(data: dict[str, object], title: str) -> str:
    outcomes: Final = tuple(_outcome(row) for row in _array(data["outcomes"]))
    metrics: Final = _array(data["metrics"])
    counts: Final = Counter(outcome.category for outcome in outcomes)
    extras: Final = tuple(sorted(set(counts) - set(TOTAL_ORDER)))
    totals: Final = (
        ", ".join(
            f"{counts[category]} {'errors' if category == 'error' and counts[category] != 1 else category}"
            for category in TOTAL_ORDER + extras
            if counts[category]
        )
        or "No test outcomes recorded"
    )
    files: Final = frozenset(outcome.nodeid.split("::")[0] for outcome in outcomes if outcome.category != "passed")
    grouped: Final = groupby(
        sorted(outcomes, key=lambda outcome: outcome.nodeid), key=lambda outcome: outcome.nodeid.split("::")[0]
    )
    file_rows: Final = tuple(_file_row(path, tuple(group)) for path, group in grouped if path in files)
    table: Final = "\n".join(
        (
            "| File | passed | failed | error | skipped | xfailed | xpassed |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            _limited(file_rows),
        )
    )
    metric_table: Final = (
        "\n".join(
            (
                "",
                "| Metric | Value | Unit | Test |",
                "| --- | --- | --- | --- |",
                _limited(tuple(_metric_row(row) for row in metrics)),
            )
        )
        if metrics
        else ""
    )
    expected: Final = tuple(
        sorted(
            (outcome for outcome in outcomes if outcome.category in ("xfailed", "xpassed")),
            key=lambda outcome: (outcome.category != "xpassed", outcome.nodeid),
        )
    )
    details: Final = (
        "\n".join(
            (
                "",
                "<details>",
                "<summary>Xfail and xpass reasons</summary>",
                "",
                _limited(
                    tuple(
                        f"- {outcome.category}: {_cell(outcome.nodeid)}: {_cell(outcome.reason)}"
                        for outcome in expected
                    )
                ),
                "",
                "</details>",
            )
        )
        if expected
        else ""
    )
    return "\n".join((f"## {_cell(title)}", "", totals, "", table, metric_table, details, ""))


def main() -> None:
    parser: Final = argparse.ArgumentParser()
    _ = parser.add_argument("--title", required=True)
    _ = parser.add_argument("report", type=Path)
    arguments: Final = parser.parse_args()
    path: Final = cast(Path, arguments.report)
    title: Final = cast(str, arguments.title)
    try:
        data: Final = _object(cast(object, json.loads(path.read_text(encoding="utf-8"))))
        if data.get("version") != 1:
            sys.stdout.write("Unsupported report version; no report was produced\n")
            return
        sys.stdout.write(render(data, title))
    except (OSError, ValueError, KeyError, TypeError, RecursionError):
        sys.stdout.write("No report was produced (missing or unreadable report)\n")


if __name__ == "__main__":
    main()
