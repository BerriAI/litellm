from typing import TYPE_CHECKING

from litellm.models.budget_window_spend import BudgetWindowSpend
from litellm.repositories import rust_database
from litellm.repositories.table_repositories import PrismaTableRepository
from litellm.rust_bridge._native import RustBudgetWindowSpendRepository

if TYPE_CHECKING:
    from prisma import models as prisma_models  # noqa: F401  # used by quoted base-class subscripts


class BudgetWindowSpendRepository(PrismaTableRepository["prisma_models.LiteLLM_BudgetWindowSpend"]):
    table_name = "litellm_budgetwindowspend"

    async def find(
        self, entity_type: str, entity_id: str, window_duration: str, *, use_writer: bool = False
    ) -> BudgetWindowSpend | None:
        """The window row at this primary key, from the writer when ``use_writer`` is set."""
        return rust_database.decode(
            BudgetWindowSpend,
            await rust_database.required_repository(RustBudgetWindowSpendRepository).find(
                entity_type, entity_id, window_duration, use_writer=use_writer
            ),
        )
