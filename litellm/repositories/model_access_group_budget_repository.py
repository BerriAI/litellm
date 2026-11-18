from collections.abc import Sequence
from typing import TYPE_CHECKING, Final, NamedTuple

from litellm.repositories import rust_database
from litellm.repositories.table_repositories import PrismaTableRepository
from litellm.rust_bridge._native import RustModelAccessGroupRepository

if TYPE_CHECKING:
    from prisma import models as prisma_models  # noqa: F401  # used by quoted base-class subscripts


class ModelAccessGroupBudgetForAuth(NamedTuple):
    """A model access group's budget as auth reads it; ``has_budget_windows`` marks a row auth must refuse."""

    access_group_name: str
    spend: float
    max_budget: float | None
    has_budget_windows: bool


class ModelAccessGroupBudgetRepository(PrismaTableRepository["prisma_models.LiteLLM_ModelAccessGroupBudgetTable"]):
    table_name = "litellm_modelaccessgroupbudgettable"

    async def names(self, limit: int) -> tuple[str, ...]:
        """Up to ``limit`` names of the groups that have a budget row, read by the Rust repository."""
        return tuple(await rust_database.required_repository(RustModelAccessGroupRepository).names(limit))

    async def budgets_for_auth(self, names: Sequence[str]) -> tuple[ModelAccessGroupBudgetForAuth, ...]:
        """The budget rows of ``names`` that exist, read by the Rust repository."""
        rows: Final = await rust_database.required_repository(RustModelAccessGroupRepository).budgets(list(names))
        return tuple(ModelAccessGroupBudgetForAuth(*row) for row in rows)
