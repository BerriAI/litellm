from litellm.models.app import LiteLLM_AppCustomerMemberTable
from litellm.repositories import rust_database
from litellm.repositories.prisma_protocols import ModelDumpRecord
from litellm.repositories.table_repositories import PrismaTableRepository
from litellm.rust_bridge._native import RustAppCustomerMemberRepository


class AppCustomerMemberRepository(PrismaTableRepository[ModelDumpRecord]):
    table_name = "litellm_appcustomermembertable"

    async def find(self, app_id: str, customer_member_id: str) -> LiteLLM_AppCustomerMemberTable | None:
        """Customer member ``customer_member_id``'s row within app ``app_id``."""
        return rust_database.decode(
            LiteLLM_AppCustomerMemberTable,
            await rust_database.required_repository(RustAppCustomerMemberRepository).find(app_id, customer_member_id),
        )
