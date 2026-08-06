"""
Credentials repository for database operations on LiteLLM_CredentialsTable.

This is the only place that talks to ``litellm_credentialstable``. Encryption of
credential values is the caller's responsibility (see ``CredentialHelperUtils``),
so reads return the stored values verbatim.
"""

from typing import Any, Final

from litellm.models.credentials import CredentialItem
from litellm.proxy.common_utils.config_sync_pubsub import wrap_table_actions_for_config_sync


class CredentialsRepository:
    """Repository for credentials database operations, keyed by credential name."""

    def __init__(self, prisma_client: Any):
        self._prisma_client = prisma_client

    @property
    def prisma_client(self) -> Any:
        if self._prisma_client is None:
            raise RuntimeError("No DB Connected. See - https://docs.litellm.ai/docs/proxy/virtual_keys")
        return self._prisma_client

    @property
    def table(self) -> Any:
        return wrap_table_actions_for_config_sync(
            actions=self.prisma_client.db.litellm_credentialstable,
            table_name="litellm_credentialstable",
        )

    @staticmethod
    def _to_model(record: Any) -> CredentialItem | None:
        if record is None:
            return None
        data: Final = record.dict() if hasattr(record, "dict") else dict(record)
        return CredentialItem(
            credential_name=data["credential_name"],
            credential_values=data.get("credential_values") or {},
            credential_info=data.get("credential_info") or {},
        )

    async def find_all(self) -> Any:
        return await self.table.find_many()

    async def create(self, data: dict[str, Any]) -> Any:
        return await self.table.create(data=data)

    async def find_by_name(self, credential_name: str) -> CredentialItem | None:
        record: Final = await self.table.find_unique(where={"credential_name": credential_name})
        return self._to_model(record)

    async def update_by_name(self, credential_name: str, data: dict[str, Any]) -> Any:
        return await self.table.update(where={"credential_name": credential_name}, data=data)

    async def delete_by_name(self, credential_name: str) -> Any:
        return await self.table.delete(where={"credential_name": credential_name})
