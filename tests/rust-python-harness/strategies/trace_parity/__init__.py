from pathlib import Path
from typing import Final

from ...shared.reporting.pytest_runner import run_pytest
from ...shared.reporting.strategy import SelectorCaseSpec, StrategyDefinition

STRATEGY: Final = StrategyDefinition(
    directory=Path(__file__).parent,
    case_spec=SelectorCaseSpec,
    run=run_pytest,
)
