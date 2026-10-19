from pathlib import Path
from typing import Final

from ...shared.reporting.strategy import RepositoryStrategyDefinition
from .command import drift_command

STRATEGY: Final = RepositoryStrategyDefinition(
    id="drift",
    order=5,
    label="Drift",
    description="Inventory committed Python and Rust changes from an explicit checkpoint.",
    directory=Path(__file__).parent,
    command=drift_command,
)
