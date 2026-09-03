from pathlib import Path
from typing import Final

from ...shared.reporting.strategy import ModuleCaseSpec, StrategyDefinition
from .reporting import render_trace_results
from .runner import run_trace_cases

STRATEGY: Final = StrategyDefinition(
    directory=Path(__file__).parent,
    case_spec=ModuleCaseSpec,
    run=run_trace_cases,
    render=render_trace_results,
)
