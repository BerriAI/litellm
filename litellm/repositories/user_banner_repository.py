from typing import Final

from litellm.repositories.table_repositories import PrismaTableRepository

USER_BANNER_ROW_ID: Final = "user_banner"


class UserBannerRepository(PrismaTableRepository):
    table_name = "litellm_uisettings"

    async def get_raw_settings(self) -> object:
        db_record: Final = await self.table.find_unique(
            where={"id": USER_BANNER_ROW_ID}  # mutable-ok: prisma filters are plain dicts
        )
        return db_record.ui_settings if db_record is not None else None

    async def upsert_settings(self, payload: str) -> None:
        row: Final = {"id": USER_BANNER_ROW_ID, "ui_settings": payload}  # mutable-ok: prisma rows are plain dicts
        await self.table.upsert(
            where={"id": USER_BANNER_ROW_ID},  # mutable-ok: prisma filters are plain dicts
            data={"create": row, "update": {"ui_settings": payload}},  # mutable-ok: prisma payloads are plain dicts
        )
