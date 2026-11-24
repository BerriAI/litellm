"""
Credentials repository for database operations on LiteLLM_CredentialsTable.

This is the only place that talks to ``litellm_credentialstable``. Encryption of
credential values is the caller's responsibility (see ``CredentialHelperUtils``),
so reads return the stored values verbatim.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final, Protocol, TypeAlias

from litellm.litellm_core_utils.core_helpers import require_attributes
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

    async def execute_raw(self, query: str, *args: object) -> int: ...


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
        if self._prisma_client is None:
            raise RuntimeError("No DB Connected. See - https://docs.litellm.ai/docs/proxy/virtual_keys")
        return require_attributes(self._prisma_client, _PrismaClientView, "db")

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

    async def update_if_value_unchanged(
        self,
        credential_name: str,
        value_match: tuple[str, str],
        info_match: tuple[str, str] | None,
        data: Mapping[str, object],
    ) -> bool:
        from prisma import Json

        value_key, stored_value = value_match
        info_condition: Final = (
            {} if info_match is None else {"credential_info": {"path": [info_match[0]], "equals": Json(info_match[1])}}
        )
        updated_rows: Final = await self.table.update_many(
            where={
                "credential_name": credential_name,
                "credential_values": {"path": [value_key], "equals": Json(stored_value)},
                **info_condition,
            },
            data=data,
        )
        return updated_rows > 0

    async def update_if_unchanged(
        self,
        credential_name: str,
        expected_values: Mapping[str, object],
        expected_info: Mapping[str, object],
        data: Mapping[str, object],
    ) -> bool:
        from prisma import Json

        info_condition: Final = {"credential_info": {"equals": Json(dict(expected_info))}} if expected_info else {}
        updated_rows: Final = await self.table.update_many(
            where={
                "credential_name": credential_name,
                "credential_values": {"equals": Json(dict(expected_values))},
                **info_condition,
            },
            data=data,
        )
        return updated_rows > 0

    async def has_info_value(self, info_key: str, value: str) -> bool:
        from prisma import Json

        return await self.table.count(where={"credential_info": {"path": [info_key], "equals": Json(value)}}) > 0

    async def repoint_info_value(self, info_key: str, previous_value: str, new_value: str) -> int:
        return await self.prisma_client.db.execute_raw(
            'UPDATE "LiteLLM_CredentialsTable" '
            'SET "credential_info" = jsonb_set("credential_info", ARRAY[$1::text], to_jsonb($3::text)), '
            '"updated_at" = NOW() '
            'WHERE "credential_info" ->> $1::text = $2::text',
            info_key,
            previous_value,
            new_value,
        )

    async def delete_by_name(self, credential_name: str) -> "prisma_models.LiteLLM_CredentialsTable | None":
        return await self.table.delete(where={"credential_name": credential_name})
