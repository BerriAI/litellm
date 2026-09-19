from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

from litellm.llms.base_llm.files.storage_backend import BaseFileStorageBackend
from litellm.repositories.prisma_protocols import TableActions
from litellm.repositories.table_repositories import PrismaTableRepository

if TYPE_CHECKING:
    from prisma import models as prisma_models

    from litellm.proxy.utils import PrismaClient

LITELLM_DB_STORAGE_BACKEND_NAME: Final = "litellm_db"
LITELLM_DB_STORAGE_URL_PREFIX: Final = f"{LITELLM_DB_STORAGE_BACKEND_NAME}://"


def storage_url_to_row_id(storage_url: str) -> str:
    if not storage_url.startswith(LITELLM_DB_STORAGE_URL_PREFIX):
        raise ValueError(f"Not a {LITELLM_DB_STORAGE_BACKEND_NAME} storage url: {storage_url}")
    return storage_url.removeprefix(LITELLM_DB_STORAGE_URL_PREFIX)


def _where_id(storage_url: str) -> Mapping[str, str]:
    return {"id": storage_url_to_row_id(storage_url)}  # mutable-ok: Prisma filter


class ManagedFileContentRepository(PrismaTableRepository["prisma_models.LiteLLM_ManagedFileContentTable"]):
    table_name = "litellm_managedfilecontenttable"


class LiteLLMDbStorageBackend(BaseFileStorageBackend):
    def __init__(self, prisma_client: "PrismaClient") -> None:
        self._prisma_client = prisma_client

    @property
    def _table(self) -> "TableActions[prisma_models.LiteLLM_ManagedFileContentTable]":
        return ManagedFileContentRepository(self._prisma_client).table

    async def upload_file(
        self,
        file_content: bytes,
        filename: str,
        content_type: str,
        path_prefix: str | None = None,
        file_naming_strategy: str = "uuid",
    ) -> str:
        from prisma import Base64

        data: Final = {"content": Base64.encode(file_content)}  # mutable-ok: Prisma payload
        row: Final = await self._table.create(data=data)
        return f"{LITELLM_DB_STORAGE_URL_PREFIX}{row.id}"

    async def download_file(self, storage_url: str) -> bytes:
        row: Final = await self._table.find_unique(where=_where_id(storage_url))
        if row is None:
            raise ValueError(f"No stored file content for {storage_url}")
        return row.content.decode()

    async def delete_file(self, storage_url: str) -> None:
        from prisma.errors import RecordNotFoundError

        try:
            await self._table.delete(where=_where_id(storage_url))
        except RecordNotFoundError:
            return
