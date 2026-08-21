"""
Credentials repository for database operations on LiteLLM_CredentialsTable.

This is the only place that talks to ``litellm_credentialstable``. Encryption of
credential values is the caller's responsibility (see ``CredentialHelperUtils``),
so reads return the stored values verbatim.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, Protocol

from litellm.models.credentials import CredentialItem
from litellm.proxy.common_utils.config_sync_pubsub import wrap_table_actions_for_config_sync

if TYPE_CHECKING:
    from prisma.models import LiteLLM_CredentialsTable


class _CredentialsDb(Protocol):
    @property
    def litellm_credentialstable(self) -> object: ...


class _PrismaClientView(Protocol):
    @property
    def db(self) -> _CredentialsDb: ...


class _CredentialsActions(Protocol):
    """Prisma table actions used by :class:`CredentialsRepository`."""

    async def find_many(self) -> "Sequence[LiteLLM_CredentialsTable]": ...

    async def create(self, *, data: Mapping[str, object]) -> "LiteLLM_CredentialsTable": ...

    async def find_unique(self, *, where: Mapping[str, object]) -> "LiteLLM_CredentialsTable | None": ...

    async def update(
        self, *, where: Mapping[str, object], data: Mapping[str, object]
    ) -> "LiteLLM_CredentialsTable | None": ...

    async def delete(self, *, where: Mapping[str, object]) -> "LiteLLM_CredentialsTable | None": ...


class CredentialsRepository:
    """Repository for credentials database operations, keyed by credential name."""

    def __init__(self, prisma_client: Any):
        self._prisma_client = prisma_client

    @property
    def prisma_client(self) -> _PrismaClientView:
        if self._prisma_client is None:
            raise RuntimeError("No DB Connected. See - https://docs.litellm.ai/docs/proxy/virtual_keys")
        return self._prisma_client

    @property
    def table(self) -> Any:
        return wrap_table_actions_for_config_sync(
            actions=self.prisma_client.db.litellm_credentialstable,
            table_name="litellm_credentialstable",
        )

    @property
    def _credentials_table(self) -> _CredentialsActions:
        return self.table

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

    async def find_all(self) -> "Sequence[LiteLLM_CredentialsTable]":
        return await self._credentials_table.find_many()

    async def create(self, data: Mapping[str, object]) -> "LiteLLM_CredentialsTable":
        return await self._credentials_table.create(data=data)

    async def find_by_name(self, credential_name: str) -> CredentialItem | None:
        record: Final = await self._credentials_table.find_unique(where={"credential_name": credential_name})
        return self._to_model(record)

    async def update_by_name(
        self, credential_name: str, data: Mapping[str, object]
    ) -> "LiteLLM_CredentialsTable | None":
        return await self._credentials_table.update(where={"credential_name": credential_name}, data=data)

    async def delete_by_name(self, credential_name: str) -> "LiteLLM_CredentialsTable | None":
        return await self._credentials_table.delete(where={"credential_name": credential_name})
