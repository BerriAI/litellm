from litellm.models.app import LiteLLM_AppCustomerOrgTable
from litellm.repositories import rust_database
from litellm.repositories.prisma_protocols import ModelDumpRecord
from litellm.repositories.table_repositories import PrismaTableRepository
from litellm.rust_bridge._native import RustAppCustomerOrganizationRepository


class AppCustomerOrganizationRepository(PrismaTableRepository[ModelDumpRecord]):
    table_name = "litellm_appcustomerorgtable"

    async def find(self, app_id: str, customer_org_id: str) -> LiteLLM_AppCustomerOrgTable | None:
        """Customer organization ``customer_org_id``'s row within app ``app_id``."""
        return rust_database.decode(
            LiteLLM_AppCustomerOrgTable,
            await rust_database.required_repository(RustAppCustomerOrganizationRepository).find(
                app_id, customer_org_id
            ),
        )
