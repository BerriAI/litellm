"""In-memory stand-ins for the Rust repositories (the auth objects and the spend log window sums), patched in
through ``rust_database.repository``.

Each fake keeps separate reader and writer rows (a writer read sees ``writer`` when it is set, else ``rows``),
logs every lookup, and raises a row that is an exception. Threads porting more auth objects add their fake as
another field of ``RustAuthStores``.
"""

import json
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Final, Literal

import pytest
from pydantic import BaseModel
from pydantic_core import to_json

from litellm.proxy._types import (
    LiteLLM_OrganizationTable,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LiteLLM_VerificationTokenView,
)
from litellm.repositories import rust_database
from litellm.rust_bridge._native import (
    RustAccessGroupRepository,
    RustAgentIdentityRepository,
    RustAuditLogRepository,
    RustBudgetRepository,
    RustEndUserRepository,
    RustJwtKeyMappingRepository,
    RustKeyRepository,
    RustModelAccessGroupRepository,
    RustObjectPermissionRepository,
    RustOrganizationRepository,
    RustProjectRepository,
    RustSpendLogRepository,
    RustTagRepository,
    RustTeamMembershipRepository,
    RustTeamRepository,
    RustUserRepository,
)


@dataclass
class FakeKeyStore:
    """``combined_view`` rows by hash, written the way the Prisma row was, shaped through the view model."""

    rows: dict[str, Mapping[str, object] | Exception] = field(default_factory=dict)
    writer: dict[str, Mapping[str, object] | Exception] | None = None
    lookups: list[tuple[str, bool]] = field(default_factory=list)

    async def find(self, token: str, *, use_writer: bool = False) -> str | None:
        self.lookups.append((token, use_writer))
        source = self.writer if use_writer and self.writer is not None else self.rows
        row = source.get(token)
        if isinstance(row, Exception):
            raise row
        return None if row is None else LiteLLM_VerificationTokenView(**row).model_dump_json()


@dataclass(frozen=True)
class JwtMapping:
    jwt_claim_name: str
    jwt_claim_value: str
    token: str
    jwt_issuer: str = ""
    is_active: bool = True
    written_by: str | None = None


@dataclass
class FakeJwtKeyMappingStore:
    mappings: list[JwtMapping] = field(default_factory=list)
    error: Exception | None = None
    lookups: list[tuple[str, str, str | None]] = field(default_factory=list)

    async def mapped_token(
        self, jwt_claim_name: str, jwt_claim_value: str, jwt_issuer: str | None = None
    ) -> str | None:
        self.lookups.append((jwt_claim_name, jwt_claim_value, jwt_issuer))
        if self.error is not None:
            raise self.error
        return next(
            (
                m.token
                for m in self.mappings
                if m.is_active
                and (m.jwt_claim_name, m.jwt_claim_value, m.jwt_issuer)
                == (jwt_claim_name, jwt_claim_value, jwt_issuer or "")
            ),
            None,
        )

    async def mapped_claims(self, tokens: Sequence[str]) -> list[tuple[str, str, str]]:
        if self.error is not None:
            raise self.error
        return [(m.jwt_issuer, m.jwt_claim_name, m.jwt_claim_value) for m in self.mappings if m.token in tokens]

    async def insert(
        self, jwt_claim_name: str, jwt_claim_value: str, token: str, *, jwt_issuer: str | None = None, written_by: str
    ) -> tuple[bool, str | None]:
        if self.error is not None:
            raise self.error
        scope: Final = (jwt_claim_name, jwt_claim_value, jwt_issuer or "")
        holder: Final = next(
            (m for m in self.mappings if (m.jwt_claim_name, m.jwt_claim_value, m.jwt_issuer) == scope), None
        )
        if holder is not None:
            return False, holder.token if holder.is_active else None
        self.mappings.append(JwtMapping(*scope[:2], token=token, jwt_issuer=scope[2], written_by=written_by))
        return True, token


AuditLogRow = tuple[str, int, str | None, str | None, str, str, str, str | None, str | None]
_AUDIT_LOG_COLUMNS: Final = (
    "id",
    "updated_at",
    "changed_by",
    "changed_by_api_key",
    "action",
    "table_name",
    "object_id",
    "before_value",
    "updated_values",
)
_EPOCH: Final = datetime(1970, 1, 1, tzinfo=timezone.utc)


def audit_log_data(row: AuditLogRow) -> Mapping[str, object]:
    """One row the audit log repository was handed, as the mapping Prisma's ``create`` took: columns that are None
    left out, the time as an aware UTC datetime, and the two values as JSON text."""
    values: Final = (row[0], _EPOCH + timedelta(microseconds=row[1]), *row[2:])
    return {column: value for column, value in zip(_AUDIT_LOG_COLUMNS, values) if value is not None}


@dataclass
class FakeAuditLogStore:
    """Audit log rows in the order they were inserted, each as ``audit_log_data`` shapes it.

    A test that hands the proxy a Prisma double with an audit log table also sees each insert there, as a
    ``create`` or a ``create_many`` with the same mapping, so that double's side effects (an error, a stall) still
    decide how the write ends."""

    rows: list[Mapping[str, object]] = field(default_factory=list)
    error: Exception | None = None

    @staticmethod
    def _prisma_table() -> object | None:
        from litellm.proxy import proxy_server

        client: Final = proxy_server.prisma_client
        return None if client is None else client.db.litellm_auditlog

    async def insert(self, row: AuditLogRow) -> None:
        if self.error is not None:
            raise self.error
        data: Final = audit_log_data(row)
        self.rows.append(data)
        table: Final = self._prisma_table()
        if table is not None:
            await table.create(data=data)

    async def insert_many(self, rows: Sequence[AuditLogRow]) -> None:
        if self.error is not None:
            raise self.error
        data: Final = tuple(audit_log_data(row) for row in rows)
        self.rows.extend(data)
        table: Final = self._prisma_table()
        if table is not None:
            await table.create_many(data=data)


@dataclass
class FakeAgentIdentityStore:
    """``(agent_id, revision)`` of each identity that is active on an enabled, identity managed agent."""

    current: set[tuple[str, str]] = field(default_factory=set)
    error: Exception | None = None
    recorded: list[tuple[str, str]] = field(default_factory=list)

    async def record_authentication(self, agent_id: str, revision: str) -> bool:
        if self.error is not None:
            raise self.error
        if (agent_id, revision) not in self.current:
            return False
        self.recorded.append((agent_id, revision))
        return True


@dataclass(frozen=True)
class GroupBudget:
    spend: float = 0.0
    max_budget: float | None = None
    has_budget_windows: bool = False


@dataclass
class FakeModelAccessGroupStore:
    groups: dict[str, GroupBudget] = field(default_factory=dict)
    error: Exception | None = None
    budget_lookups: list[tuple[str, ...]] = field(default_factory=list)

    async def names(self, limit: int) -> list[str]:
        if self.error is not None:
            raise self.error
        return list(self.groups)[:limit]

    async def budgets(self, names: Sequence[str]) -> list[tuple[str, float, float | None, bool]]:
        self.budget_lookups.append(tuple(names))
        if self.error is not None:
            raise self.error
        return [
            (name, group.spend, group.max_budget, group.has_budget_windows)
            for name, group in self.groups.items()
            if name in names
        ]


def _json(row: Mapping[str, object] | Exception | None) -> str | None:
    if isinstance(row, Exception):
        raise row
    return None if row is None else to_json(row).decode()


@dataclass
class FakeBudgetStore:
    """``LiteLLM_BudgetTable`` rows by ``budget_id``."""

    rows: dict[str, Mapping[str, object] | Exception] = field(default_factory=dict)
    writer: dict[str, Mapping[str, object] | Exception] | None = None
    lookups: list[str] = field(default_factory=list)

    async def find_end_user_default_budget(self, budget_id: str) -> str | None:
        self.lookups.append(budget_id)
        return _json(self.rows.get(budget_id))

    async def find_team_member_default_budget(self, budget_id: str, *, use_writer: bool = False) -> str | None:
        if use_writer and self.writer is not None:
            self.lookups.append(budget_id)
            return _json(self.writer.get(budget_id))
        return await self.find_end_user_default_budget(budget_id)


_END_USER_RESTRICTIONS: Sequence[str] = (
    "budget_id",
    "allowed_model_region",
    "default_model",
    "object_permission_id",
)


@dataclass
class FakeEndUserStore:
    """``LiteLLM_EndUserTable`` rows by ``user_id``, with ``litellm_budget_table`` and ``object_permission`` nested."""

    rows: dict[str, Mapping[str, object] | Exception] = field(default_factory=dict)
    error: Exception | None = None
    lookups: list[str] = field(default_factory=list)

    async def find_with_budget_and_object_permission(self, end_user_id: str) -> str | None:
        self.lookups.append(end_user_id)
        if self.error is not None:
            raise self.error
        return _json(self.rows.get(end_user_id))

    async def find_restricted_ids(self, limit: int) -> list[str]:
        if self.error is not None:
            raise self.error
        return [
            end_user_id
            for end_user_id, row in self.rows.items()
            if not isinstance(row, Exception)
            and (
                row.get("blocked")
                or any(row.get(column) is not None for column in _END_USER_RESTRICTIONS)
                or bool(row.get("models"))
            )
        ][:limit]


@dataclass
class FakeUserStore:
    """``LiteLLM_UserTable`` rows by ``user_id``, each answered as stored with its ``organization_memberships``.

    A sign-in resolves to the row of its own id only: the SSO id and email fallbacks and their linking are the Rust
    repository's, tested against Postgres there. A writer read sees ``writer`` when it is set, else ``rows``; a row
    that is an exception is raised. ``sign_ins`` logs each resolve as its arguments and whether it read the writer,
    and ``inserts`` each insert as its arguments.
    """

    rows: MutableMapping[str, Mapping[str, object] | Exception] = field(default_factory=dict)
    writer: MutableMapping[str, Mapping[str, object] | Exception] | None = None
    error: Exception | None = None
    sign_ins: list[tuple[Mapping[str, object], bool]] = field(default_factory=list)
    email_checks: list[tuple[str, str]] = field(default_factory=list)
    inserts: list[Mapping[str, object]] = field(default_factory=list)

    def _source(self, writer: bool) -> MutableMapping[str, Mapping[str, object] | Exception]:
        return self.writer if writer and self.writer is not None else self.rows

    async def resolve(
        self,
        user_id: str,
        *,
        sso_user_id: str | None = None,
        user_email: str | None = None,
        email_verified: bool | None = None,
        links_any_user: bool,
        use_writer: bool = False,
    ) -> str | None:
        self.sign_ins.append(
            (
                {
                    "user_id": user_id,
                    "sso_user_id": sso_user_id,
                    "user_email": user_email,
                    "email_verified": email_verified,
                    "links_any_user": links_any_user,
                },
                use_writer,
            )
        )
        if self.error is not None:
            raise self.error
        row: Final = self._source(use_writer).get(user_id)
        if isinstance(row, Exception):
            raise row
        return None if row is None else to_json({"organization_memberships": [], **row}).decode()

    async def email_belongs_to_another_user(self, user_email: str, user_id: str) -> bool:
        self.email_checks.append((user_email, user_id))
        if self.error is not None:
            raise self.error
        return any(
            not isinstance(row, Exception)
            and row.get("user_id") != user_id
            and str(row.get("user_email") or "").lower() == user_email.lower()
            for row in self._source(True).values()
        )

    async def backfill_email(self, user_id: str, user_email: str, *, links_any_user: bool) -> str | None:
        if self.error is not None:
            raise self.error
        if not links_any_user and await self.email_belongs_to_another_user(user_email, user_id):
            return None
        source: Final = self._source(True)
        row: Final = source.get(user_id)
        if isinstance(row, Exception):
            raise row
        if row is None:
            return None
        stored: Final = row.get("user_email") or user_email
        source[user_id] = {**row, "user_email": stored}
        return str(stored)

    async def set_role(self, user_id: str, user_role: str) -> str | None:
        if self.error is not None:
            raise self.error
        source: Final = self._source(True)
        row: Final = source.get(user_id)
        if isinstance(row, Exception):
            raise row
        if row is None:
            return None
        source[user_id] = {**row, "user_role": user_role}
        return to_json({"organization_memberships": [], **row, "user_role": user_role}).decode()

    async def insert(
        self,
        user_id: str,
        *,
        sso_user_id: str | None = None,
        user_email: str | None = None,
        links_any_user: bool,
        user_role: str | None = None,
        max_budget: float | None = None,
        budget_duration: str | None = None,
        budget_reset_at: float | None = None,
        models: Sequence[str] | None = None,
        budget_limits: str | None = None,
        tpm_limit: int | None = None,
        rpm_limit: int | None = None,
        max_parallel_requests: int | None = None,
        user_alias: str | None = None,
        team_id: str | None = None,
        organization_id: str | None = None,
        object_permission_id: str | None = None,
        password_reset_required: bool | None = None,
        metadata: str | None = None,
        allowed_cache_controls: Sequence[str] | None = None,
        policies: Sequence[str] | None = None,
        model_max_budget: str | None = None,
    ) -> tuple[bool, str | None]:
        self.inserts.append(
            {
                "user_id": user_id,
                "sso_user_id": sso_user_id,
                "user_email": user_email,
                "links_any_user": links_any_user,
                "user_role": user_role,
                "max_budget": max_budget,
                "budget_duration": budget_duration,
                "budget_reset_at": budget_reset_at,
                "models": models,
                "budget_limits": budget_limits,
                "tpm_limit": tpm_limit,
                "rpm_limit": rpm_limit,
                "max_parallel_requests": max_parallel_requests,
                "user_alias": user_alias,
                "team_id": team_id,
                "organization_id": organization_id,
                "object_permission_id": object_permission_id,
                "password_reset_required": password_reset_required,
                "metadata": metadata,
                "allowed_cache_controls": allowed_cache_controls,
                "policies": policies,
                "model_max_budget": model_max_budget,
            }
        )
        if self.error is not None:
            raise self.error
        source: Final = self._source(True)
        existing: Final = source.get(user_id)
        if isinstance(existing, Exception):
            raise existing
        if existing is not None:
            return False, to_json({"organization_memberships": [], **existing}).decode()
        email_is_free: Final = user_email is not None and (
            links_any_user or not await self.email_belongs_to_another_user(user_email, user_id)
        )
        row: Final[Mapping[str, object]] = {
            "user_id": user_id,
            "sso_user_id": None if links_any_user else sso_user_id,
            "user_email": user_email if email_is_free else None,
            "user_role": user_role,
            "max_budget": max_budget,
            "budget_duration": budget_duration,
            "budget_reset_at": None
            if budget_reset_at is None
            else datetime.fromtimestamp(budget_reset_at, timezone.utc),
            "models": list(models or ()),
            "budget_limits": None if budget_limits is None else json.loads(budget_limits),
            "tpm_limit": tpm_limit,
            "rpm_limit": rpm_limit,
            "max_parallel_requests": max_parallel_requests,
            "user_alias": user_alias,
            "team_id": team_id,
            "organization_id": organization_id,
            "object_permission_id": object_permission_id,
            "password_reset_required": password_reset_required,
            "metadata": {} if metadata is None else json.loads(metadata),
            "allowed_cache_controls": list(allowed_cache_controls or ()),
            "policies": list(policies or ()),
            "model_max_budget": {} if model_max_budget is None else json.loads(model_max_budget),
        }
        source[user_id] = row
        return True, to_json({"organization_memberships": [], **row}).decode()


def _answer(row: Mapping[str, object] | BaseModel | Exception | None, model: type) -> str | None:
    if isinstance(row, Exception):
        raise row
    if isinstance(row, BaseModel):
        row = row.model_dump()
    return None if row is None else model.model_validate(row).model_dump_json()


def _answers_by_alias(
    rows: Mapping[str, Mapping[str, object] | BaseModel | Exception], column: str, alias: str, model: type
) -> list[str]:
    """Up to two rows whose ``column`` is ``alias``, ordered by id, as the alias stores answer."""
    matching: Final[list[str | None]] = []
    for _, row in sorted(rows.items()):
        if isinstance(row, Exception):
            raise row
        fields = row.model_dump() if isinstance(row, BaseModel) else row
        if fields.get(column) == alias:
            matching.append(_answer(row, model))
    return [answer for answer in matching[:2] if answer is not None]


@dataclass
class FakeTeamStore:
    """Team rows by id, shaped through ``LiteLLM_TeamTable``."""

    rows: dict[str, Mapping[str, object] | BaseModel | Exception] = field(default_factory=dict)
    writer: dict[str, Mapping[str, object] | BaseModel | Exception] | None = None
    lookups: list[tuple[str, bool]] = field(default_factory=list)

    async def find_with_model_aliases(self, team_id: str, *, use_writer: bool = False) -> str | None:
        self.lookups.append((team_id, use_writer))
        source = self.writer if use_writer and self.writer is not None else self.rows
        return _answer(source.get(team_id), LiteLLM_TeamTable)

    async def find_up_to_two_by_alias(self, alias: str) -> list[str]:
        return _answers_by_alias(self.rows, "team_alias", alias, LiteLLM_TeamTable)


@dataclass
class FakeTeamMembershipStore:
    """Membership rows by ``(user_id, team_id)``, shaped through ``LiteLLM_TeamMembership``.

    A member add is logged in ``adds`` as ``(user_id, team_id, budget_id)`` and ends as ``add_outcome``. An added
    member gets a row here and is appended to ``roster_before``, the roster the team holds when the add takes its
    lock, with ``roster_email`` as its email. The budget of its own it came with is kept in ``own_budgets`` as
    ``(allowed_models, budget_duration, budget_reset_at, created_by, updated_by)``."""

    rows: dict[tuple[str, str], Mapping[str, object] | BaseModel | Exception] = field(default_factory=dict)
    writer: dict[tuple[str, str], Mapping[str, object] | BaseModel | Exception] | None = None
    lookups: list[tuple[str, str, bool]] = field(default_factory=list)
    adds: list[tuple[str, str, str | None]] = field(default_factory=list)
    add_outcome: Literal["added", "already_member", "team_missing", "user_missing"] = "added"
    add_error: Exception | None = None
    own_budgets: dict[str, tuple[Sequence[str], str | None, float | None, str, str]] = field(default_factory=dict)
    roster_email: str | None = None
    roster_before: Sequence[Mapping[str, object]] = ()

    async def find_with_budget(self, user_id: str, team_id: str, *, use_writer: bool = False) -> str | None:
        self.lookups.append((user_id, team_id, use_writer))
        source = self.writer if use_writer and self.writer is not None else self.rows
        return _answer(source.get((user_id, team_id)), LiteLLM_TeamMembership)

    async def add_member(
        self,
        user_id: str,
        team_id: str,
        *,
        budget_id: str | None = None,
        own_budget: tuple[Sequence[str], str | None, float | None, str, str] | None = None,
    ) -> tuple[Literal["added", "already_member", "team_missing", "user_missing"], str | None, str | None]:
        self.adds.append((user_id, team_id, budget_id))
        if self.add_error is not None:
            raise self.add_error
        if self.add_outcome != "added":
            return self.add_outcome, None, None
        if own_budget is not None and budget_id is not None:
            self.own_budgets[budget_id] = own_budget
        source: Final = self.writer if self.writer is not None else self.rows
        row: Final = source.setdefault(
            (user_id, team_id), {"user_id": user_id, "team_id": team_id, "budget_id": budget_id}
        )
        stored_roster: Final = (
            *self.roster_before,
            {"user_id": user_id, "user_email": self.roster_email, "role": "user"},
        )
        return "added", _answer(row, LiteLLM_TeamMembership), json.dumps(stored_roster)


@dataclass
class FakeOrganizationStore:
    """Organization rows by id, shaped through ``LiteLLM_OrganizationTable``. The budget is dropped unless
    the lookup asks for it, as the store only joins it on request."""

    rows: dict[str, Mapping[str, object] | BaseModel | Exception] = field(default_factory=dict)
    lookups: list[tuple[str, bool]] = field(default_factory=list)

    async def find_with_optional_budget(
        self, organization_id: str, *, include_budget_table: bool = False
    ) -> str | None:
        self.lookups.append((organization_id, include_budget_table))
        row = self.rows.get(organization_id)
        if isinstance(row, Mapping) and not include_budget_table:
            row = {**row, "litellm_budget_table": None}
        return _answer(row, LiteLLM_OrganizationTable)

    async def find_up_to_two_by_alias(self, alias: str) -> list[str]:
        rows: Final = {
            organization_id: {**row, "litellm_budget_table": None} if isinstance(row, Mapping) else row
            for organization_id, row in self.rows.items()
        }
        return _answers_by_alias(rows, "organization_alias", alias, LiteLLM_OrganizationTable)


Row = BaseModel | BaseException


@dataclass
class RustStoreRows:
    """Rows one fake store serves, with the reader and writer sides apart like the Prisma ``db`` and
    ``writer_db`` clients."""

    reader: MutableMapping[str, Row] = field(default_factory=dict)
    writer: MutableMapping[str, Row] = field(default_factory=dict)
    lookups: list[tuple[str, bool]] = field(default_factory=list)

    def seed(self, key: str, row: Row) -> None:
        """Serve ``row`` for ``key`` on both sides."""
        self.reader[key] = row
        self.writer[key] = row

    def side(self, use_writer: bool) -> MutableMapping[str, Row]:
        return self.writer if use_writer else self.reader


@dataclass
class TagRows(RustStoreRows):
    """Tag rows, which are read from the reader only. ``batches`` logs each ``find_many_with_budgets`` call and
    ``name_listings`` each registry listing's limit; ``names_error`` makes the listing fail."""

    batches: list[tuple[str, ...]] = field(default_factory=list)
    name_listings: list[int] = field(default_factory=list)
    names_error: BaseException | None = None


def _encoded(row: Row | None) -> str | None:
    if isinstance(row, BaseException):
        raise row
    return None if row is None else row.model_dump_json()


class FakeFindStore:
    def __init__(self, rows: RustStoreRows) -> None:
        self._rows: Final = rows

    async def _find(self, key: str, *, use_writer: bool = False) -> str | None:
        self._rows.lookups.append((key, use_writer))
        return _encoded(self._rows.side(use_writer).get(key))


class FakeAccessGroupStore(FakeFindStore):
    async def find_by_access_group_id(self, access_group_id: str, *, use_writer: bool = False) -> str | None:
        return await self._find(access_group_id, use_writer=use_writer)


class FakeObjectPermissionStore(FakeFindStore):
    async def find_by_object_permission_id(self, object_permission_id: str, *, use_writer: bool = False) -> str | None:
        return await self._find(object_permission_id, use_writer=use_writer)


class FakeProjectStore:
    def __init__(self, rows: RustStoreRows) -> None:
        self._rows: Final = rows

    async def find_with_budget(self, project_id: str) -> str | None:
        self._rows.lookups.append((project_id, False))
        return _encoded(self._rows.reader.get(project_id))


class FakeTagStore:
    def __init__(self, rows: TagRows) -> None:
        self._rows: Final = rows

    async def find_many_with_budgets(self, tag_names: Iterable[str]) -> list[str]:
        names: Final = tuple(tag_names)
        self._rows.batches.append(names)
        return [encoded for name in names if (encoded := _encoded(self._rows.reader.get(name))) is not None]

    async def names(self, limit: int) -> list[str]:
        self._rows.name_listings.append(limit)
        if self._rows.names_error is not None:
            raise self._rows.names_error
        return list(self._rows.reader)[:limit]


@dataclass
class FakeSpendLogStore:
    """Window sums by (owner, owner_id); every lookup is logged as (owner, owner_id, since)."""

    totals: dict[tuple[str, str], float] = field(default_factory=dict)
    error: Exception | None = None
    lookups: list[tuple[str, str, datetime]] = field(default_factory=list)

    async def _window_spend(self, owner: str, owner_id: str, since: datetime) -> float:
        self.lookups.append((owner, owner_id, since))
        if self.error is not None:
            raise self.error
        return self.totals.get((owner, owner_id), 0.0)

    async def key_window_spend(self, api_key: str, since: datetime) -> float:
        return await self._window_spend("Key", api_key, since)

    async def team_window_spend(self, team_id: str, since: datetime) -> float:
        return await self._window_spend("Team", team_id, since)

    async def user_window_spend(self, user_id: str, since: datetime) -> float:
        return await self._window_spend("User", user_id, since)


@dataclass
class RustAuthStores:
    key: FakeKeyStore = field(default_factory=FakeKeyStore)
    jwt_key_mapping: FakeJwtKeyMappingStore = field(default_factory=FakeJwtKeyMappingStore)
    agent_identity: FakeAgentIdentityStore = field(default_factory=FakeAgentIdentityStore)
    audit_log: FakeAuditLogStore = field(default_factory=FakeAuditLogStore)
    model_access_group: FakeModelAccessGroupStore = field(default_factory=FakeModelAccessGroupStore)
    budget: FakeBudgetStore = field(default_factory=FakeBudgetStore)
    end_user: FakeEndUserStore = field(default_factory=FakeEndUserStore)
    user: FakeUserStore = field(default_factory=FakeUserStore)
    team: FakeTeamStore = field(default_factory=FakeTeamStore)
    team_membership: FakeTeamMembershipStore = field(default_factory=FakeTeamMembershipStore)
    organization: FakeOrganizationStore = field(default_factory=FakeOrganizationStore)
    project: RustStoreRows = field(default_factory=RustStoreRows)
    tag: TagRows = field(default_factory=TagRows)
    access_group: RustStoreRows = field(default_factory=RustStoreRows)
    object_permission: RustStoreRows = field(default_factory=RustStoreRows)
    spend_log: FakeSpendLogStore = field(default_factory=FakeSpendLogStore)

    def by_kind(self) -> Mapping[Callable[..., object], object]:
        return {
            RustKeyRepository: self.key,
            RustJwtKeyMappingRepository: self.jwt_key_mapping,
            RustAgentIdentityRepository: self.agent_identity,
            RustAuditLogRepository: self.audit_log,
            RustModelAccessGroupRepository: self.model_access_group,
            RustBudgetRepository: self.budget,
            RustEndUserRepository: self.end_user,
            RustUserRepository: self.user,
            RustTeamRepository: self.team,
            RustTeamMembershipRepository: self.team_membership,
            RustOrganizationRepository: self.organization,
            RustProjectRepository: FakeProjectStore(self.project),
            RustTagRepository: FakeTagStore(self.tag),
            RustAccessGroupRepository: FakeAccessGroupStore(self.access_group),
            RustObjectPermissionRepository: FakeObjectPermissionStore(self.object_permission),
            RustSpendLogRepository: self.spend_log,
        }


def install_rust_auth_stores(monkeypatch: pytest.MonkeyPatch, stores: RustAuthStores | None = None) -> RustAuthStores:
    """Answers `rust_database.repository` with these fakes; a kind with no fake answers None, so "No db connected" still fires."""
    installed = stores or RustAuthStores()
    monkeypatch.setattr(rust_database, "repository", lambda kind: installed.by_kind().get(kind))
    return installed


@pytest.fixture
def rust_auth_stores(monkeypatch: pytest.MonkeyPatch) -> RustAuthStores:
    return install_rust_auth_stores(monkeypatch)


@pytest.fixture(autouse=True)
def audit_log_store(monkeypatch: pytest.MonkeyPatch) -> FakeAuditLogStore:
    """Audit log rows are inserted through the Rust repository, which unit tests have no database for. This store
    takes them in every test, on top of whatever else answers ``rust_database.repository``."""
    store: Final = FakeAuditLogStore()
    answer: Final = rust_database.repository
    monkeypatch.setattr(
        rust_database, "repository", lambda kind: store if kind is RustAuditLogRepository else answer(kind)
    )
    return store
