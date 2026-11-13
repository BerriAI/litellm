from collections.abc import Awaitable, Mapping, Sequence
from enum import Enum
from typing import Final, LiteralString, Protocol

from pydantic import BaseModel, ConfigDict, TypeAdapter
from typing_extensions import assert_never


class QueryRows(Protocol):
    def __call__(self, query: LiteralString, *args: object) -> Awaitable[Sequence[Mapping[str, object]]]: ...


class CustomerEntity(str, Enum):
    ORGANIZATION = "customer_org_id"
    TEAM = "customer_team_id"
    MEMBER = "customer_member_id"

    @property
    def alias_query(self) -> LiteralString:
        match self:
            case CustomerEntity.ORGANIZATION:
                return (
                    'SELECT customer_org_id AS entity_id, alias FROM "LiteLLM_CustomerOrganizationTable" '
                    "WHERE customer_org_id = ANY($1::text[])"
                )
            case CustomerEntity.TEAM:
                return (
                    'SELECT customer_team_id AS entity_id, alias FROM "LiteLLM_CustomerTeamTable" '
                    "WHERE customer_team_id = ANY($1::text[])"
                )
            case CustomerEntity.MEMBER:
                return (
                    'SELECT customer_member_id AS entity_id, alias FROM "LiteLLM_CustomerMemberTable" '
                    "WHERE customer_member_id = ANY($1::text[])"
                )
            case _:
                assert_never(self)


class _EntityAlias(BaseModel):
    model_config = ConfigDict(frozen=True)

    entity_id: str
    alias: str | None


class _MemberCount(BaseModel):
    model_config = ConfigDict(frozen=True)

    customer_org_id: str
    member_count: int


_ALIASES: Final = TypeAdapter(tuple[_EntityAlias, ...])
_COUNTS: Final = TypeAdapter(tuple[_MemberCount, ...])


class CustomerActivityRepository:
    def __init__(self, query_rows: QueryRows) -> None:
        self._query_rows = query_rows

    async def aliases(self, entity: CustomerEntity, ids: tuple[str, ...]) -> Mapping[str, Mapping[str, object]]:
        if not ids:
            return {}
        rows: Final = _ALIASES.validate_python(await self._query_rows(entity.alias_query, list(ids)))
        return {row.entity_id: {"alias": row.alias or row.entity_id} for row in rows}

    async def member_counts(self, organization_ids: tuple[str, ...]) -> Mapping[str, int]:
        if not organization_ids:
            return {}
        rows: Final = _COUNTS.validate_python(
            await self._query_rows(
                "SELECT t.parent_org_id AS customer_org_id, COUNT(m.customer_member_id) AS member_count "
                'FROM "LiteLLM_CustomerTeamTable" t '
                'LEFT JOIN "LiteLLM_CustomerMemberTable" m ON m.parent_team_id = t.customer_team_id '
                "WHERE t.parent_org_id = ANY($1::text[]) GROUP BY t.parent_org_id",
                list(organization_ids),
            )
        )
        return {row.customer_org_id: row.member_count for row in rows}
