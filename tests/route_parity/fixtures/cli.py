from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast


@dataclass(frozen=True, slots=True)
class RecordingArgs:
    concurrency: int
    examples: int
    fixture_dir: Path | None


def _positive_int(value: str) -> int:
    parsed: Final = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_recording_args(argv: Sequence[str] | None = None) -> RecordingArgs:
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=_positive_int, default=1)
    parser.add_argument("--examples", type=_positive_int, default=4)
    parser.add_argument("--fixture-dir", type=Path)
    namespace: Final = parser.parse_args(argv)
    return RecordingArgs(
        concurrency=cast(int, namespace.concurrency),
        examples=cast(int, namespace.examples),
        fixture_dir=cast(Path | None, namespace.fixture_dir),
    )
