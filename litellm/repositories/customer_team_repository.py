from typing import TYPE_CHECKING

from litellm.models.customer_hierarchy import LiteLLM_CustomerTeamTable
from litellm.repositories import rust_database
from litellm.repositories.table_repositories import PrismaTableRepository
from litellm.rust_bridge._native import RustCustomerTeamRepository

if TYPE_CHECKING:
    from prisma import models as prisma_models  # noqa: F401  # used by quoted base-class subscripts


class CustomerTeamRepository(PrismaTableRepository["prisma_models.LiteLLM_CustomerTeamTable"]):
    table_name = "litellm_customerteamtable"

    async def find_with_budgets(self, customer_team_id: str) -> LiteLLM_CustomerTeamTable | None:
        """The customer team with its budget and budget override rows."""
        return rust_database.decode(
            LiteLLM_CustomerTeamTable,
            await rust_database.required_repository(RustCustomerTeamRepository).find_with_budgets(customer_team_id),
        )
