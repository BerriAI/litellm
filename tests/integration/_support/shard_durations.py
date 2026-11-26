from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Final
from xml.etree import ElementTree

from pydantic import TypeAdapter

ROOT: Final = Path(__file__).resolve().parents[3]
DURATIONS_FILE: Final = ROOT / "tests/integration/shard_durations.json"
DURATIONS: Final = TypeAdapter(dict[str, float])


def file_seconds(report: Path) -> Mapping[str, float]:
    pairs: Final = sorted(
        (case.get("file") or "", float(case.get("time") or 0)) for case in ElementTree.parse(report).iter("testcase")
    )
    return {file: sum(seconds for _, seconds in cases) for file, cases in groupby(pairs, key=itemgetter(0)) if file}


def recorded_durations(reports: tuple[Path, ...]) -> Mapping[str, float]:
    per_report: Final = tuple(file_seconds(report) for report in reports)
    files: Final = frozenset(file for seconds in per_report for file in seconds)
    return {file: max(seconds.get(file, 0.0) for seconds in per_report) for file in files}


def refreshed(existing: Mapping[str, float], recorded: Mapping[str, float], root: Path) -> Mapping[str, int]:
    merged: Final = {**existing, **recorded}
    return {file: max(1, round(merged[file])) for file in sorted(merged) if (root / file).is_file()}


def main() -> int:
    parser: Final = argparse.ArgumentParser(description="Refresh shard_durations.json from integration JUnit reports")
    parser.add_argument("reports", nargs="+", type=Path)
    reports: Final = tuple(parser.parse_args().reports)
    existing: Final = DURATIONS.validate_json(DURATIONS_FILE.read_bytes()) if DURATIONS_FILE.exists() else {}
    durations: Final = refreshed(existing, recorded_durations(reports), ROOT)
    DURATIONS_FILE.write_text(json.dumps(durations, indent=1) + "\n")
    sys.stdout.write(f"Recorded {len(durations)} integration file durations in {DURATIONS_FILE.relative_to(ROOT)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
