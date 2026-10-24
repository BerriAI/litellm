from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import accumulate, groupby
from os.path import commonpath
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


def _cell(value: str | float, limit: int = 240) -> str:
    raw: Final = str(value)
    clipped: Final = raw[:limit] + ("..." if len(raw) > limit else "")
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
    unit: Final = _string(row["unit"])
    measured: Final = _cell(f"{value} {unit}".strip())
    formatted: Final = f"<strong>{measured}</strong>" if isinstance(value, (int, float)) else measured
    return f"| {_cell(_string(row['name']))} | {formatted} |"


def _metric_group(nodeid: str, rows: tuple[dict[str, object], ...]) -> str:
    return "\n".join(
        (
            "| Metric | Value |",
            "| --- | --- |",
            _limited(tuple(_metric_row(row) for row in rows)),
            "",
            "<details>",
            "<summary>Metric source</summary>",
            "",
            f"<code>{_cell(nodeid)}</code>",
            "",
            "</details>",
            "",
        )
    )


def _expected_detail(outcome: Outcome) -> str:
    path, _, test = outcome.nodeid.partition("::")
    return (
        f"- **{outcome.category}**: {_cell(test or path)}\n\n"
        f"  {_cell(outcome.reason, 1000)}\n\n"
        f"  <sub>{_cell(path)}</sub>\n"
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
    prefix: Final = (commonpath(tuple(str(Path(path).parent) for path in files)) or ".") if files else "."
    grouped: Final = groupby(
        sorted(outcomes, key=lambda outcome: outcome.nodeid), key=lambda outcome: outcome.nodeid.split("::")[0]
    )
    file_rows: Final = tuple(
        _file_row(str(Path(path).relative_to(prefix)), tuple(group)) for path, group in grouped if path in files
    )
    table: Final = (
        "\n".join(
            (
                "### Files with non-pass outcomes",
                "",
                f"Paths relative to <code>{_cell(prefix)}/</code>\n" if prefix != "." else "",
                "| File | Passed | Failed | Errors | Skipped | Xfailed | Xpassed |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                _limited(file_rows),
            )
        )
        if file_rows
        else ""
    )
    metric_groups: Final = groupby(
        sorted(metrics, key=lambda row: _string(row["nodeid"])), key=lambda row: _string(row["nodeid"])
    )
    metric_table: Final = (
        "\n".join(
            (
                "### Metrics",
                "",
                _limited(tuple(_metric_group(nodeid, tuple(rows)) for nodeid, rows in metric_groups)),
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
                f"<summary>Xfail and xpass reasons ({len(expected)})</summary>",
                "",
                _limited(tuple(_expected_detail(outcome) for outcome in expected)),
                "",
                "</details>",
            )
        )
        if expected
        else ""
    )
    return (
        "\n\n".join(part for part in (f"## {_cell(title)}", f"**{totals}**", metric_table, table, details) if part)
        + "\n"
    )


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
