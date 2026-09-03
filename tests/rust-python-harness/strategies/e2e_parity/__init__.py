from pathlib import Path
from typing import Final

from ...shared.reporting.rendering import render_outcomes
from ...shared.reporting.strategy import ModuleCaseSpec, StrategyDefinition
from .runner import run_e2e_cases

STRATEGY: Final = StrategyDefinition(
    directory=Path(__file__).parent,
    case_spec=ModuleCaseSpec,
    run=run_e2e_cases,
    render=render_outcomes,
)
