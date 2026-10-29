from typing import TYPE_CHECKING, Final

from litellm.llms.base_llm.files.storage_backend import BaseFileStorageBackend
from litellm.repositories.managed_file_content_repository import ManagedFileContentRepository

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

LITELLM_DB_STORAGE_BACKEND_NAME: Final = "litellm_db"
LITELLM_DB_STORAGE_URL_PREFIX: Final = f"{LITELLM_DB_STORAGE_BACKEND_NAME}://"


def storage_url_to_row_id(storage_url: str) -> str:
    if not storage_url.startswith(LITELLM_DB_STORAGE_URL_PREFIX):
        raise ValueError(f"Not a {LITELLM_DB_STORAGE_BACKEND_NAME} storage url: {storage_url}")
    return storage_url.removeprefix(LITELLM_DB_STORAGE_URL_PREFIX)


class LiteLLMDbStorageBackend(BaseFileStorageBackend):
    def __init__(self, prisma_client: "PrismaClient") -> None:
        self._contents = ManagedFileContentRepository(prisma_client)

    async def upload_file(
        self,
        file_content: bytes,
        filename: str,
        content_type: str,
        path_prefix: str | None = None,
        file_naming_strategy: str = "uuid",
    ) -> str:
        return f"{LITELLM_DB_STORAGE_URL_PREFIX}{await self._contents.store(file_content)}"

    async def download_file(self, storage_url: str) -> bytes:
        content: Final = await self._contents.load(storage_url_to_row_id(storage_url))
        if content is None:
            raise ValueError(f"No stored file content for {storage_url}")
        return content

    async def delete_file(self, storage_url: str) -> None:
        await self._contents.delete(storage_url_to_row_id(storage_url))
