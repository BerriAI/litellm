from typing import TYPE_CHECKING

from litellm.models.customer_hierarchy import LiteLLM_CustomerMemberTable
from litellm.repositories import rust_database
from litellm.repositories.table_repositories import PrismaTableRepository
from litellm.rust_bridge._native import RustCustomerMemberRepository

if TYPE_CHECKING:
    from prisma import models as prisma_models  # noqa: F401  # used by quoted base-class subscripts


class CustomerMemberRepository(PrismaTableRepository["prisma_models.LiteLLM_CustomerMemberTable"]):
    table_name = "litellm_customermembertable"

    async def find_with_budgets(self, customer_member_id: str) -> LiteLLM_CustomerMemberTable | None:
        """The customer member with its budget and budget override rows."""
        return rust_database.decode(
            LiteLLM_CustomerMemberTable,
            await rust_database.required_repository(RustCustomerMemberRepository).find_with_budgets(customer_member_id),
        )
