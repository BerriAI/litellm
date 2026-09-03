from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Final

from ...shared.reporting.strategy import SuiteCaseSpec, StrategyDefinition
from ...shared.unit_runners.suite_runner import run_suites
from .runner import RustSuite, run_suite


def _load_suite(path: Path) -> RustSuite:
    return RustSuite.model_validate_json(path.read_text(encoding="utf-8"))


STRATEGY: Final = StrategyDefinition(
    directory=Path(__file__).parent,
    case_spec=SuiteCaseSpec,
    run=partial(run_suites, load=_load_suite, execute=run_suite),
)
