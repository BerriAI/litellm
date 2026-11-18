from litellm.models.app import LiteLLM_AppTable
from litellm.repositories import rust_database
from litellm.repositories.prisma_protocols import ModelDumpRecord
from litellm.repositories.table_repositories import PrismaTableRepository
from litellm.rust_bridge._native import RustAppRepository


class AppRepository(PrismaTableRepository[ModelDumpRecord]):
    table_name = "litellm_apptable"

    async def find(self, app_id: str) -> LiteLLM_AppTable | None:
        """The app with id ``app_id``."""
        return rust_database.decode(
            LiteLLM_AppTable, await rust_database.required_repository(RustAppRepository).find(app_id)
        )
