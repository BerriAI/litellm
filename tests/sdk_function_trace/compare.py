from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Final

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.sdk_function_trace.fixtures import ROUTES
from tests.sdk_function_trace.report import compare, render


def _run(route: str, asynchronous: bool, *, full: bool, colorize: bool) -> bool:
    comparison: Final = compare(route, asynchronous=asynchronous)
    sys.stdout.write(render(comparison, full=full, colorize=colorize))
    return comparison.passed


def main() -> None:
    parser: Final = argparse.ArgumentParser(description="Compare Python and Rust SDK pipeline steps per route")
    parser.add_argument("--route", choices=("all", *ROUTES), default="all")
    mode: Final = parser.add_mutually_exclusive_group()
    mode.add_argument("--async", dest="asynchronous", action="store_true", default=True)
    mode.add_argument("--sync", dest="asynchronous", action="store_false")
    mode.add_argument("--both", action="store_true", help="run async and sync for every selected route")
    parser.add_argument(
        "--check", action="store_true", help="exit nonzero for missing, extra, or reordered comparable steps"
    )
    parser.add_argument(
        "--full", action="store_true", help="print every captured runtime event instead of pipeline steps"
    )
    args: Final = parser.parse_args()
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    colorize: Final = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    results: Final = tuple(
        _run(selected, selected_mode, full=args.full, colorize=colorize)
        for selected in ROUTES
        if args.route in ("all", selected)
        for selected_mode in ((True, False) if args.both else (args.asynchronous,))
    )
    if args.check and not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
