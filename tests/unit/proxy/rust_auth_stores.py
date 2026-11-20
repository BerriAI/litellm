"""In-memory stand-ins for the Rust repositories (the auth objects and the spend log window sums), patched in
through ``rust_database.repository``.

Each fake keeps separate reader and writer rows (a writer read sees ``writer`` when it is set, else ``rows``),
logs every lookup, and raises a row that is an exception. Threads porting more auth objects add their fake as
another field of ``RustAuthStores``.
"""

from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final

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
    lookups: list[str] = field(default_factory=list)

    async def find_end_user_default_budget(self, budget_id: str) -> str | None:
        self.lookups.append(budget_id)
        return _json(self.rows.get(budget_id))

    async def find_team_member_default_budget(self, budget_id: str) -> str | None:
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
    that is an exception is raised. ``sign_ins`` logs each resolve as its arguments and whether it read the writer.
    """

    rows: MutableMapping[str, Mapping[str, object] | Exception] = field(default_factory=dict)
    writer: MutableMapping[str, Mapping[str, object] | Exception] | None = None
    error: Exception | None = None
    sign_ins: list[tuple[Mapping[str, object], bool]] = field(default_factory=list)
    email_checks: list[tuple[str, str]] = field(default_factory=list)

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
    """Membership rows by ``(user_id, team_id)``, shaped through ``LiteLLM_TeamMembership``."""

    rows: dict[tuple[str, str], Mapping[str, object] | BaseModel | Exception] = field(default_factory=dict)
    writer: dict[tuple[str, str], Mapping[str, object] | BaseModel | Exception] | None = None
    lookups: list[tuple[str, str, bool]] = field(default_factory=list)

    async def find_with_budget(self, user_id: str, team_id: str, *, use_writer: bool = False) -> str | None:
        self.lookups.append((user_id, team_id, use_writer))
        source = self.writer if use_writer and self.writer is not None else self.rows
        return _answer(source.get((user_id, team_id)), LiteLLM_TeamMembership)


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
