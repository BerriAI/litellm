"""
Credentials repository for database operations on LiteLLM_CredentialsTable.

This is the only place that talks to ``litellm_credentialstable``. Encryption of
credential values is the caller's responsibility (see ``CredentialHelperUtils``),
so reads return the stored values verbatim.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final, Protocol, TypeAlias

from litellm.models.credentials import CredentialItem
from litellm.repositories.base_repository import DbRecord, record_to_dict
from litellm.repositories.prisma_protocols import TableActions
from litellm.repositories.table_repositories import PrismaTableRepository

if TYPE_CHECKING:
    from prisma import models as prisma_models

    _CredentialsTable: TypeAlias = TableActions[prisma_models.LiteLLM_CredentialsTable]


class _PrismaCredentialsDb(Protocol):
    @property
    def litellm_credentialstable(self) -> "_CredentialsTable": ...


class _PrismaClientView(Protocol):
    @property
    def db(self) -> _PrismaCredentialsDb: ...


class _CredentialsTableRepository(PrismaTableRepository["prisma_models.LiteLLM_CredentialsTable"]):
    table_name = "litellm_credentialstable"


class CredentialsRepository:
    """Repository for credentials database operations, keyed by credential name."""

    def __init__(self, prisma_client: object):
        self._prisma_client = prisma_client

    @property
    def prisma_client(self) -> _PrismaClientView:
        client: Final[_PrismaClientView] = _CredentialsTableRepository(self._prisma_client).prisma_client
        return client

    @property
    def table(self) -> "_CredentialsTable":
        return _CredentialsTableRepository(self._prisma_client).table

    @staticmethod
    def _to_model(record: DbRecord | None) -> CredentialItem | None:
        if record is None:
            return None
        data: Final = record_to_dict(record)
        return CredentialItem.model_validate(
            {
                "credential_name": data["credential_name"],
                "display_name": data.get("display_name"),
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
