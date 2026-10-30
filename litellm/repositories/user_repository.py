"""
User repository for database operations on LiteLLM_UserTable.
"""

import asyncio
import json
from collections.abc import Generator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, contextmanager
from contextvars import ContextVar
from itertools import chain
from typing import TYPE_CHECKING, Final, Protocol

from pydantic import TypeAdapter

from litellm.models.user import LiteLLM_UserTable, SCIMPlaceholder
from litellm.repositories.base_repository import BaseRepository, DbRecord, record_to_dict
from litellm.repositories.chunked_in import IN_LIST_CHUNK_SIZE
from litellm.repositories.prisma_protocols import TableActions

if TYPE_CHECKING:
    from prisma import Prisma
    from prisma import models as prisma_models

_JSON_ENCODED_COLUMNS: Final = frozenset({"metadata", "model_spend", "model_max_budget"})

_SHADOWING_PLACEHOLDERS_SQL: Final = """
SELECT p.user_id AS placeholder_user_id,
       array_agg(r.user_id ORDER BY r.user_id) AS resolved_user_ids,
       p.teams AS team_ids
FROM "LiteLLM_UserTable" p
JOIN "LiteLLM_UserTable" r
  ON r.user_id <> p.user_id
 AND (r.sso_user_id = p.user_id OR LOWER(r.user_email) = LOWER(p.user_id))
WHERE p.sso_user_id IS NULL
  AND NOT EXISTS (SELECT 1 FROM "LiteLLM_VerificationToken" k WHERE k.user_id = p.user_id)
GROUP BY p.user_id, p.teams
ORDER BY p.user_id
"""

_PLACEHOLDER_ROWS_ADAPTER: Final = TypeAdapter(tuple[SCIMPlaceholder, ...])
_FREE_SSO_CREATION: Final[ContextVar[tuple[str | None, object] | None]] = ContextVar("free_sso_creation", default=None)


@contextmanager
def free_sso_user_creation(user_id: str | None, *, enabled: bool = True) -> Generator[None]:
    token: Final = _FREE_SSO_CREATION.set((user_id, asyncio.current_task()) if enabled else None)
    try:
        yield
    finally:
        _FREE_SSO_CREATION.reset(token)


class _UserDb(Protocol):
    @property
    def litellm_usertable(self) -> TableActions["prisma_models.LiteLLM_UserTable"]: ...


class _UserTransaction(_UserDb, Protocol):
    async def execute_raw(self, query: str) -> int: ...

    async def query_raw(self, query: str) -> object: ...


class _PrismaClientView(Protocol):
    @property
    def db(self) -> _UserDb: ...

    @property
    def writer_db(self) -> _UserDb: ...

    def tx(self) -> AbstractAsyncContextManager[_UserTransaction]: ...


async def _count_billable_users(table: TableActions["prisma_models.LiteLLM_UserTable"]) -> int:
    from prisma import Json

    total: Final = await table.count()
    deactivated: Final = await table.count(where={"metadata": {"path": ["scim_active"], "equals": Json(False)}})
    return max(0, total - deactivated)


class UserRepository(BaseRepository[LiteLLM_UserTable]):
    """Repository for user database operations."""

    def __init__(self, prisma_client: object, *, use_writer: bool = False) -> None:
        super().__init__(prisma_client)
        self._use_writer = use_writer

    @property
    def table(self) -> TableActions["prisma_models.LiteLLM_UserTable"]:
        client: Final[_PrismaClientView] = self.prisma_client
        database: Final = client.writer_db if self._use_writer else client.db
        return database.litellm_usertable

    @property
    def model_class(self) -> type[LiteLLM_UserTable]:
        return LiteLLM_UserTable

    def _to_model(self, record: DbRecord | None) -> LiteLLM_UserTable | None:
        """Convert a database record to a User model."""
        if record is None:
            return None

        return LiteLLM_UserTable.model_validate(
            {
                column: json.loads(value) if column in _JSON_ENCODED_COLUMNS and isinstance(value, str) else value
                for column, value in record_to_dict(record).items()
            }
        )

    async def find_by_id(self, id_value: str, id_field: str = "user_id") -> LiteLLM_UserTable | None:
        return await super().find_by_id(id_value, id_field)

    async def find_by_email(self, user_email: str) -> LiteLLM_UserTable | None:
        """Find a user by email."""
        records: Final = await self.find_many(where={"user_email": user_email})
        return records[0] if records else None

    async def find_by_emails(self, user_emails: Sequence[str]) -> Sequence[LiteLLM_UserTable]:
        """Every user whose email matches one of ``user_emails``, ignoring case.

        A roster entry stored by email can differ in case from its user row (member_add
        resolves emails case-insensitively), so an exact match would miss it. The list goes
        out in slices of ``IN_LIST_CHUNK_SIZE`` so one statement stays under Postgres's
        bind-parameter cap; ``chunked_in.find_many_in`` cannot carry the insensitive mode.
        """
        unique: Final = sorted(frozenset(user_emails))
        pages: Final = tuple(
            [
                await self.find_many(
                    where={
                        "user_email": {
                            # bounded-ok: sliced to IN_LIST_CHUNK_SIZE values per statement
                            "in": unique[start : start + IN_LIST_CHUNK_SIZE],
                            "mode": "insensitive",
                        }
                    }
                )
                for start in range(0, len(unique), IN_LIST_CHUNK_SIZE)
            ]
        )
        return tuple(chain.from_iterable(pages))

    async def find_by_sso_id(self, sso_user_id: str) -> LiteLLM_UserTable | None:
        """Find a user by SSO ID."""
        return await self.find_by_id(sso_user_id, id_field="sso_user_id")

    async def find_by_organization_id(self, organization_id: str) -> list[LiteLLM_UserTable]:
        """Find all users in an organization."""
        return await self.find_many(where={"organization_id": organization_id})

    async def find_by_team_id(self, team_id: str) -> list[LiteLLM_UserTable]:
        """Find all users in a team."""
        return await self.find_many(where={"teams": {"has": team_id}})

    async def find_shadowing_placeholders(self, tx: "Prisma") -> tuple[SCIMPlaceholder, ...]:
        """Users with no SSO id and no virtual keys whose id is another user's SSO id or email."""
        rows: Final = await tx.query_raw(_SHADOWING_PLACEHOLDERS_SQL)
        return _PLACEHOLDER_ROWS_ADAPTER.validate_python(rows)

    async def count_billable_users(self) -> int:
        """Number of users that count toward the license seat limit.

        Every user is billable except those SCIM-deactivated
        (metadata.scim_active == false). Rows where scim_active is absent,
        null, or true all count, so seats are counted as total users minus
        the deactivated ones.
        """
        return await _count_billable_users(self.table)

    async def upsert_user(self, data: Mapping[str, object]) -> "prisma_models.LiteLLM_UserTable | None":
        """Return the user row, or None when a new free SSO user would exceed capacity."""
        scope: Final = _FREE_SSO_CREATION.get()
        if (
            scope is None
            or (scope[0] is not None and scope[0] != data["user_id"])
            or scope[1] is not asyncio.current_task()
        ):
            return await self.table.upsert(
                where={"user_id": data["user_id"]}, data={"create": dict(data), "update": {}}
            )

        client: Final[_PrismaClientView] = self.prisma_client
        async with client.tx() as transaction:
            await transaction.execute_raw("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
            await transaction.query_raw("SELECT 1 AS locked FROM pg_advisory_xact_lock(468530001)")
            existing: Final = await transaction.litellm_usertable.find_unique(where={"user_id": data["user_id"]})
            if existing is not None:
                return existing
            if await _count_billable_users(transaction.litellm_usertable) >= 5:
                return None
            return await transaction.litellm_usertable.upsert(
                where={"user_id": data["user_id"]}, data={"create": dict(data), "update": {}}
            )

    async def create_user(
        self,
        user_id: str,
        user_alias: str | None = None,
        team_id: str | None = None,
        sso_user_id: str | None = None,
        organization_id: str | None = None,
        password: str | None = None,
        teams: list[str] | None = None,
        user_role: str | None = None,
        max_budget: float | None = None,
        user_email: str | None = None,
        models: list[str] | None = None,
        metadata: Mapping[str, object] | None = None,
        max_parallel_requests: int | None = None,
        tpm_limit: int | None = None,
        rpm_limit: int | None = None,
        budget_duration: str | None = None,
        allowed_cache_controls: list[str] | None = None,
        policies: list[str] | None = None,
        object_permission_id: str | None = None,
    ) -> LiteLLM_UserTable:
        """Create a new user."""
        data: Final[dict[str, object]] = {"user_id": user_id}
        if user_alias is not None:
            data["user_alias"] = user_alias
        if team_id is not None:
            data["team_id"] = team_id
        if sso_user_id is not None:
            data["sso_user_id"] = sso_user_id
        if organization_id is not None:
            data["organization_id"] = organization_id
        if password is not None:
            data["password"] = password
        if teams is not None:
            data["teams"] = teams
        if user_role is not None:
            data["user_role"] = user_role
        if max_budget is not None:
            data["max_budget"] = max_budget
        if user_email is not None:
            data["user_email"] = user_email
        if models is not None:
            data["models"] = models
        if metadata is not None:
            data["metadata"] = json.dumps(metadata)
        if max_parallel_requests is not None:
            data["max_parallel_requests"] = max_parallel_requests
        if tpm_limit is not None:
            data["tpm_limit"] = tpm_limit
        if rpm_limit is not None:
            data["rpm_limit"] = rpm_limit
        if budget_duration is not None:
            data["budget_duration"] = budget_duration
        if allowed_cache_controls is not None:
            data["allowed_cache_controls"] = allowed_cache_controls
        if policies is not None:
            data["policies"] = policies
        if object_permission_id is not None:
            data["object_permission_id"] = object_permission_id

        return await self.create(data)

    async def update_user(
        self,
        user_id: str,
        user_alias: str | None = None,
        team_id: str | None = None,
        sso_user_id: str | None = None,
        organization_id: str | None = None,
        password: str | None = None,
        teams: list[str] | None = None,
        user_role: str | None = None,
        max_budget: float | None = None,
        user_email: str | None = None,
        models: list[str] | None = None,
        metadata: Mapping[str, object] | None = None,
        max_parallel_requests: int | None = None,
        tpm_limit: int | None = None,
        rpm_limit: int | None = None,
        budget_duration: str | None = None,
        allowed_cache_controls: list[str] | None = None,
        policies: list[str] | None = None,
        object_permission_id: str | None = None,
    ) -> LiteLLM_UserTable | None:
        """Update a user."""
        data: Final[dict[str, object]] = {}
        if user_alias is not None:
            data["user_alias"] = user_alias
        if team_id is not None:
            data["team_id"] = team_id
        if sso_user_id is not None:
            data["sso_user_id"] = sso_user_id
        if organization_id is not None:
            data["organization_id"] = organization_id
        if password is not None:
            data["password"] = password
        if teams is not None:
            data["teams"] = teams
        if user_role is not None:
            data["user_role"] = user_role
        if max_budget is not None:
            data["max_budget"] = max_budget
        if user_email is not None:
            data["user_email"] = user_email
        if models is not None:
            data["models"] = models
        if metadata is not None:
            data["metadata"] = json.dumps(metadata)
        if max_parallel_requests is not None:
            data["max_parallel_requests"] = max_parallel_requests
        if tpm_limit is not None:
            data["tpm_limit"] = tpm_limit
        if rpm_limit is not None:
            data["rpm_limit"] = rpm_limit
        if budget_duration is not None:
            data["budget_duration"] = budget_duration
        if allowed_cache_controls is not None:
            data["allowed_cache_controls"] = allowed_cache_controls
        if policies is not None:
            data["policies"] = policies
        if object_permission_id is not None:
            data["object_permission_id"] = object_permission_id

        return await self.update(user_id, data, id_field="user_id")

    async def backfill_null_user_email(self, user_id: str, user_email: str) -> int:
        """Set user_email only when the stored value is null, atomically at the database.

        Returns the number of rows updated: 0 means another writer already set an email.
        """
        updated_count: Final[int] = await self.table.update_many(
            where={"user_id": user_id, "user_email": None},
            data={"user_email": user_email},
        )
        return updated_count

    async def delete_user(self, user_id: str) -> LiteLLM_UserTable | None:
        """Delete a user."""
        return await self.delete(user_id, id_field="user_id")

    async def update_spend(self, user_id: str, spend: float) -> LiteLLM_UserTable | None:
        """Update user spend."""
        return await self.update(user_id, {"spend": spend}, id_field="user_id")

    async def add_to_team(self, user_id: str, team_id: str) -> LiteLLM_UserTable | None:
        """Add a user to a team using atomic array push operation."""
        if not await self.exists(user_id, id_field="user_id"):
            return None

        return await self.update(user_id, {"teams": {"push": team_id}}, id_field="user_id")

    async def remove_from_team(self, user_id: str, team_id: str) -> LiteLLM_UserTable | None:
        """Remove a user from a team.

        Note: Prisma doesn't support atomic array removal, so we use a
        read-modify-write pattern here. For high-concurrency scenarios,
        consider using raw SQL with array_remove().
        """
        user: Final = await self.find_by_id(user_id)
        if user is None:
            return None

        teams: Final = [t for t in user.teams if t != team_id]
        return await self.update(user_id, {"teams": teams}, id_field="user_id")
