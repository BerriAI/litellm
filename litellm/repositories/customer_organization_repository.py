from typing import TYPE_CHECKING

from litellm.models.customer_hierarchy import LiteLLM_CustomerOrganizationTable
from litellm.repositories import rust_database
from litellm.repositories.table_repositories import PrismaTableRepository
from litellm.rust_bridge._native import RustCustomerOrganizationRepository

if TYPE_CHECKING:
    from prisma import models as prisma_models  # noqa: F401  # used by quoted base-class subscripts


class CustomerOrganizationRepository(PrismaTableRepository["prisma_models.LiteLLM_CustomerOrganizationTable"]):
    table_name = "litellm_customerorganizationtable"

    async def find_with_budget(self, customer_org_id: str) -> LiteLLM_CustomerOrganizationTable | None:
        """The customer organization with its budget row."""
        return rust_database.decode(
            LiteLLM_CustomerOrganizationTable,
            await rust_database.required_repository(RustCustomerOrganizationRepository).find_with_budget(
                customer_org_id
            ),
        )
