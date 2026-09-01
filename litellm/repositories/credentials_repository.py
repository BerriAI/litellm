"""
Credentials repository for database operations on LiteLLM_CredentialsTable.

This is the only place that talks to ``litellm_credentialstable``. Encryption of
credential values is the caller's responsibility (see ``CredentialHelperUtils``),
so reads return the stored values verbatim.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, Protocol, TypeAlias

from litellm.models.credentials import CredentialItem
from litellm.proxy.common_utils.config_sync_pubsub import wrap_table_actions_for_config_sync
from litellm.repositories.base_repository import DbRecord, record_to_dict
from litellm.repositories.prisma_protocols import TableActions

if TYPE_CHECKING:
    from prisma import models as prisma_models

    _CredentialsTable: TypeAlias = TableActions[prisma_models.LiteLLM_CredentialsTable]


class _PrismaCredentialsDb(Protocol):
    @property
    def litellm_credentialstable(self) -> "_CredentialsTable": ...


class _PrismaClientView(Protocol):
    @property
    def db(self) -> _PrismaCredentialsDb: ...


class CredentialsRepository:
    """Repository for credentials database operations, keyed by credential name."""

    def __init__(self, prisma_client: Any):  # any-ok: PrismaClient is an untyped runtime wrapper
        self._prisma_client = prisma_client

    @property
    def prisma_client(self) -> _PrismaClientView:
        if self._prisma_client is None:
            raise RuntimeError("No DB Connected. See - https://docs.litellm.ai/docs/proxy/virtual_keys")
        client: Final[_PrismaClientView] = self._prisma_client
        return client

    @property
    def table(self) -> "_CredentialsTable":
        return wrap_table_actions_for_config_sync(
            actions=self.prisma_client.db.litellm_credentialstable,
            table_name="litellm_credentialstable",
        )

    @staticmethod
    def _to_model(record: DbRecord | None) -> CredentialItem | None:
        if record is None:
            return None
        data: Final = record_to_dict(record)
        return CredentialItem.model_validate(
            {
                "credential_name": data["credential_name"],
                "credential_values": data.get("credential_values") or {},
                "credential_info": data.get("credential_info") or {},
            }
        )

    async def find_all(self) -> Sequence["prisma_models.LiteLLM_CredentialsTable"]:
        return await self.table.find_many()

    async def create(self, data: Mapping[str, object]) -> "prisma_models.LiteLLM_CredentialsTable":
        return await self.table.create(data=data)

    async def find_by_name(self, credential_name: str) -> CredentialItem | None:
        record: Final = await self.table.find_unique(where={"credential_name": credential_name})
        return self._to_model(record)

    async def update_by_name(
        self, credential_name: str, data: Mapping[str, object]
    ) -> "prisma_models.LiteLLM_CredentialsTable | None":
        return await self.table.update(where={"credential_name": credential_name}, data=data)

    async def delete_by_name(self, credential_name: str) -> "prisma_models.LiteLLM_CredentialsTable | None":
        return await self.table.delete(where={"credential_name": credential_name})
