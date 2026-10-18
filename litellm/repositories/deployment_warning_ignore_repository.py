from typing import TYPE_CHECKING, Final

from litellm.repositories.table_repositories import PrismaTableRepository

if TYPE_CHECKING:
    from prisma import models as prisma_models  # noqa: F401  # resolved only from the quoted base-class subscript below

DEPLOYMENT_WARNING_IGNORES_ROW_ID: Final = "deployment_warning_ignores"


class DeploymentWarningIgnoreRepository(PrismaTableRepository["prisma_models.LiteLLM_UISettings"]):
    table_name = "litellm_uisettings"

    async def get_raw_ignores(self) -> object:
        db_record: Final = await self.table.find_unique(where={"id": DEPLOYMENT_WARNING_IGNORES_ROW_ID})
        return db_record.ui_settings if db_record is not None else None

    async def upsert_ignores(self, payload: str) -> None:
        row: Final = {"id": DEPLOYMENT_WARNING_IGNORES_ROW_ID, "ui_settings": payload}
        await self.table.upsert(
            where={"id": DEPLOYMENT_WARNING_IGNORES_ROW_ID},
            data={"create": row, "update": {"ui_settings": payload}},
        )
