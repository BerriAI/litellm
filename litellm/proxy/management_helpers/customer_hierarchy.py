from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from itertools import chain
from typing import TYPE_CHECKING, Final, Generic, Protocol, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from litellm.models.budget import BudgetLimit, BudgetLimits, LiteLLM_BudgetTable, initialize_budget_limits
from litellm.proxy.common_utils.timezone_utils import (
    BudgetResetSettings,
    get_budget_reset_settings,
    get_budget_reset_time,
)
from litellm.proxy.db.db_span import db_span
from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler
from litellm.proxy.list_api.common import escape_like
from litellm.proxy.management_helpers.customer_budget_usage import customer_budget_usage
from litellm.repositories.budget_repository import create_budget_row, update_budget_row
from litellm.repositories.chunked_in import count_in, delete_many_in, find_many_in
from litellm.repositories.prisma_protocols import TableActions
from litellm.types.proxy.management_endpoints.customer_hierarchy import (
    Budget,
    BudgetGroupChange,
    BudgetGroupsUpdate,
    BudgetTemplate,
    CustomerMember,
    MemberCreate,
    MemberPatch,
    Organization,
    OrganizationBudget,
    OrganizationCreate,
    OrganizationPatch,
    Team,
    TeamCreate,
    TeamPatch,
    TemplatePatch,
)

if TYPE_CHECKING:
    from prisma import Json, Prisma
    from prisma.models import LiteLLM_BudgetWindowSpend


class ErrorCode(str, Enum):
    FREE_TIER_LIMIT = "free_tier_limit"
    CAPACITY_BUSY = "capacity_busy"
    NOT_FOUND = "not_found"
    PARENT_NOT_FOUND = "parent_not_found"
    ALREADY_EXISTS = "already_exists"
    INVALID_DEFAULT_BUDGET = "invalid_default_budget"
    TEMPLATE_NOT_IN_ORG = "template_not_in_org"
    HAS_CHILDREN = "has_children"
    FORBIDDEN = "forbidden"
    DATABASE_UNAVAILABLE = "database_unavailable"
    INVALID_TEMPLATE_NAMES = "invalid_template_names"
    TEMPLATE_IN_USE = "template_in_use"
    BUDGET_REQUIRED = "budget_required"
    ALIAS_ALREADY_EXISTS = "alias_already_exists"
    CUSTOMER_ORG_MISMATCH = "customer_org_mismatch"
    INCREASE_NOT_ALLOWED = "increase_not_allowed"
    REQUEST_NOT_PENDING = "request_not_pending"
    REQUEST_ALREADY_PENDING = "request_already_pending"
    EXPIRY_IN_PAST = "expiry_in_past"
    KEY_NOT_IN_ORG = "key_not_in_org"
    CONFIGURATION_CONFLICT = "configuration_conflict"


@dataclass(frozen=True, slots=True)
class Failure:
    code: ErrorCode


FREE_ORGANIZATION_LIMIT: Final = 5
FREE_TEAMS_PER_ORGANIZATION_LIMIT: Final = 5
FREE_MEMBERS_PER_TEAM_LIMIT: Final = 5
FREE_CAPACITY_ADVISORY_LOCK: Final = 714023581
FREE_CAPACITY_LOCK_TIMEOUT_MS: Final = 1000


class CustomerPortalLimits(BaseModel):
    model_config = ConfigDict(frozen=True)

    limits_apply: bool
    organization_limit: int = FREE_ORGANIZATION_LIMIT
    teams_per_organization_limit: int = FREE_TEAMS_PER_ORGANIZATION_LIMIT
    members_per_team_limit: int = FREE_MEMBERS_PER_TEAM_LIMIT
    organization_count: int
    any_limit_reached: bool


class _CapacityCounts(BaseModel):
    model_config = ConfigDict(frozen=True)

    full_organizations: int
    full_teams: int


_LOCK_ORGANIZATION_SQL: Final = (
    'SELECT 1 AS locked FROM "LiteLLM_CustomerOrganizationTable" WHERE customer_org_id = $1 FOR UPDATE'
)
_LOCK_TEMPLATES_SQL: Final = (
    'SELECT budget_id FROM "LiteLLM_BudgetTable" WHERE budget_id IN '
    '(SELECT unnest(budget_template_ids) FROM "LiteLLM_CustomerOrganizationTable" WHERE customer_org_id = $1) '
    "ORDER BY budget_id FOR UPDATE"
)

_LOCK_CUSTOMER_IDENTITIES_SQL: Final = (
    "SELECT 1 AS locked FROM pg_advisory_xact_lock(hashtextextended('litellm:customer-identities', 0))"
)


class TransactionOperation(str, Enum):
    READ_WRITE = "read_write"
    DELETE = "delete"


class PrismaRecord(Protocol):
    def model_dump(self) -> Mapping[str, object]: ...


class TransactionSource(Protocol):
    def tx(self) -> AbstractAsyncContextManager["Prisma"]: ...


T: Final = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Page(Generic[T]):
    data: tuple[T, ...]
    total_count: int


class BudgetRecord(Budget):
    model_config = ConfigDict(frozen=True, extra="ignore")

    temp_budget_increase: float | None = Field(default=None, exclude=True)
    temp_budget_expiry: datetime | None = Field(default=None, exclude=True)


class BudgetWindow(BaseModel):
    """The recurring window a budget row carries; customer rows read theirs from the budget they point at."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    budget_duration: str | None = None
    budget_reset_at: datetime | None = None


class OrganizationMemberCount(BaseModel):
    model_config = ConfigDict(frozen=True)

    customer_org_id: str
    member_count: int


_VALUES: Final = TypeAdapter(Mapping[str, object])
_MEMBER_COUNTS: Final = TypeAdapter(tuple[OrganizationMemberCount, ...])
_BUDGET: Final[TypeAdapter[BudgetRecord | None]] = TypeAdapter(BudgetRecord | None)
_EXPIRY: Final[TypeAdapter[datetime | None]] = TypeAdapter(datetime | None)
_WINDOW: Final[TypeAdapter[BudgetWindow | None]] = TypeAdapter(BudgetWindow | None)


def window_reset_at(value: object) -> datetime | None:
    window: Final = _WINDOW.validate_python(value)
    return None if window is None else window.budget_reset_at


class _StoredWindow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    budget_duration: str | None = None
    budget_reset_at: datetime | None = None


def budget_window_values(
    budget: Budget, patch: Collection[str], current: _StoredWindow | None = None
) -> Mapping[str, object]:
    """``budget_reset_at`` follows ``budget_duration``: stamped when a duration is written, cleared when it is removed.
    Resubmitting the duration a row already runs keeps its clock, or an edit landing after the window came due
    would push the reset out another window without the job ever clearing the spend."""
    if "budget_duration" not in patch:
        return {}
    duration: Final = budget.budget_duration
    if current is not None and current.budget_duration == duration and current.budget_reset_at is not None:
        return {}
    return {"budget_reset_at": get_budget_reset_time(duration) if duration is not None else None}


def budget_values(budget: Budget) -> Mapping[str, object]:
    return _VALUES.validate_python(budget.model_dump(include=set(Budget.model_fields)))


def group_budget_patch(group: BudgetGroupChange, current: OrganizationBudget | None) -> Budget:
    patch: Final = Budget.model_validate(group.model_dump(exclude_unset=True, exclude={"budget_id", "name"}))
    if patch.budget_limits is None:
        return patch
    resets: Final = (
        {limit.budget_duration: limit.reset_at for limit in (current.budget_limits or ())} if current else {}
    )
    return patch.model_copy(
        update={
            "budget_limits": tuple(
                limit.model_copy(update={"reset_at": resets.get(limit.budget_duration, limit.reset_at)})
                for limit in patch.budget_limits
            )
        }
    )


_JSON_BUDGET_COLUMNS: Final = frozenset({"model_tpm_limit", "model_rpm_limit"})


_MODEL_LIMIT_MAP: Final = TypeAdapter(dict[str, int])
_BUDGET_LIMITS: Final[TypeAdapter[tuple[BudgetLimit, ...]]] = TypeAdapter(BudgetLimits)


def budget_limit_json(value: object) -> "Json":
    from prisma import Json

    limits: Final = _BUDGET_LIMITS.validate_python(value)
    return Json(
        [
            {
                "budget_duration": limit.budget_duration,
                "max_budget": limit.max_budget,
                "reset_at": limit.reset_at.isoformat() if limit.reset_at is not None else None,
            }
            for limit in limits
        ]
    )


def _budget_json_column(key: str, value: object) -> object:
    from prisma import Json

    if key == "budget_limits":
        return budget_limit_json(value) if value is not None else Json(None)
    if key in _JSON_BUDGET_COLUMNS:
        return Json(_MODEL_LIMIT_MAP.validate_python(value))
    return value


def budget_row_values(values: Mapping[str, object]) -> Mapping[str, object]:
    """Prisma writes Json columns only through ``Json``: a budget without windows carries ``Json(None)``, the same
    JSON null the key path writes, since a bare ``None`` on a ``Json?`` column is rejected by the query engine."""
    return {
        key: _budget_json_column(key, value)
        for key, value in values.items()
        if not (key in _JSON_BUDGET_COLUMNS and value is None)
    }


def related_budget(value: object, now: datetime | None = None, inherited_maximum: float | None = None) -> Budget | None:
    record: Final = _BUDGET.validate_python(value)
    if record is None:
        return None
    values: Final = budget_values(record)
    if now is None:
        return Budget.model_validate(values)
    budget: Final = LiteLLM_BudgetTable(
        max_budget=record.max_budget,
        temp_budget_increase=record.temp_budget_increase,
        temp_budget_expiry=record.temp_budget_expiry,
    )
    maximum: Final = budget.effective_max_budget(utc_datetime(now))
    if maximum is not None:
        return Budget.model_validate({**values, "max_budget": maximum})
    if inherited_maximum is None:
        return Budget.model_validate(values)
    return Budget.model_validate(
        {
            **values,
            "max_budget": inherited_maximum + budget.active_temp_budget_increase(utc_datetime(now)),
        }
    )


class OrganizationRecord(Organization):
    budget_template_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def valid_ownership(self) -> "OrganizationRecord":
        if (
            self.default_budget_template_id is not None
            and self.default_budget_template_id not in self.budget_template_ids
        ):
            raise ValueError("Default budget template must belong to the organization")
        return self


class StoredBudget(BudgetRecord):
    budget_id: str
    name: str | None = None


class _MemberDeletionRow(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    customer_member_id: str
    budget_override_id: str | None = None


class _TeamDeletionRow(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    customer_team_id: str
    budget_id: str | None = None
    budget_override_id: str | None = None


_DELETE_BATCH_SIZE: Final = 100
_BUDGET_REFERENCE_FIELDS: Final = (
    "customer_organization_defaults",
    "customer_organizations",
    "customer_member_overrides",
    "customer_teams",
    "customer_team_overrides",
    "customer_members",
    "organization",
    "projects",
    "keys",
    "end_users",
    "tags",
    "model_access_groups",
    "team_membership",
    "organization_membership",
)


def organization_response(
    row: OrganizationRecord, budgets: Mapping[str, StoredBudget], member_count: int
) -> Organization:
    return Organization.model_validate(
        {
            **row.model_dump(exclude={"budget_template_ids"}),
            "member_count": member_count,
            "default_budgets": tuple(
                OrganizationBudget.model_validate(
                    {
                        **budgets[budget_id].model_dump(),
                        "name": budgets[budget_id].name or budget_id,
                    }
                )
                for budget_id in row.budget_template_ids
            ),
        }
    )


def team_response(row: PrismaRecord, now: datetime) -> Team:
    values: Final = row.model_dump()
    budget: Final = related_budget(values.get("budget"), now)
    override: Final = related_budget(
        values.get("budget_override"), now, budget.max_budget if budget is not None else None
    )
    expiry: Final = _EXPIRY.validate_python(values.get("budget_override_expires_at"))
    return Team.model_validate(
        {
            **values,
            "budget": related_budget(values.get("budget")),
            "budget_override": related_budget(values.get("budget_override")),
            "budget_reset_at": member_window_reset_at(values.get("budget"), values.get("budget_override"), expiry, now),
            "effective_budget": effective_member_budget(budget, override, expiry, now),
        }
    )


def override_expiry(
    row: Team | CustomerMember | None, payload: TeamPatch | MemberPatch, override_id: str | None
) -> datetime | None:
    if override_id is None:
        return None
    if "budget_override_expires_at" in payload.model_fields_set:
        return payload.budget_override_expires_at
    return row.budget_override_expires_at if row is not None and row.budget_override_id is not None else None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_datetime(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def effective_member_budget(
    template: Budget | None, override: Budget | None, expires_at: datetime | None, now: datetime
) -> Budget:
    inherited: Final = template if template is not None else Budget(allowed_models=())
    if override is None:
        return inherited
    if expires_at is not None and utc_datetime(expires_at) <= utc_datetime(now):
        return inherited
    values: Final = _VALUES.validate_python(
        override.model_dump(exclude_none=True, exclude={"allowed_models", "budget_limits"})
    )
    models: Final[Mapping[str, object]] = {"allowed_models": override.allowed_models} if override.allowed_models else {}
    inherited_windows: Final = () if override.max_budget is not None else inherited.budget_limits or ()
    windows: Final = {
        **{limit.budget_duration: limit for limit in inherited_windows},
        **{limit.budget_duration: limit for limit in override.budget_limits or ()},
    }
    return Budget.model_validate(
        {**budget_values(inherited), **values, **models, "budget_limits": tuple(windows.values()) or None}
    )


def member_window_reset_at(
    template: object, override: object, expires_at: datetime | None, now: datetime
) -> datetime | None:
    """An unexpired override with its own duration runs the member's window; otherwise the template does."""
    active: Final = _WINDOW.validate_python(override)
    unexpired: Final = expires_at is None or utc_datetime(expires_at) > utc_datetime(now)
    if active is not None and active.budget_duration is not None and unexpired:
        return active.budget_reset_at
    return window_reset_at(template)


class CustomerRowWindow(BaseModel):
    """The budget relations a customer row carries, read off a Prisma row or a dumped mapping."""

    model_config = ConfigDict(frozen=True, extra="ignore", from_attributes=True)

    budget: BudgetWindow | None = None
    budget_override: BudgetWindow | None = None
    budget_override_expires_at: datetime | None = None


def customer_row_reset_at(row: object, now: datetime) -> datetime | None:
    """When the row's current spend window resets: its budget row's clock, or an unexpired override's own."""
    window: Final = CustomerRowWindow.model_validate(row, from_attributes=True)
    return member_window_reset_at(window.budget, window.budget_override, window.budget_override_expires_at, now)


def member_response(row: PrismaRecord, now: datetime) -> CustomerMember:
    values: Final = row.model_dump()
    override: Final = related_budget(values.get("budget_override"))
    template: Final = related_budget(values.get("budget"), now)
    effective_override: Final = related_budget(
        values.get("budget_override"), now, template.max_budget if template is not None else None
    )
    expiry: Final = _EXPIRY.validate_python(values.get("budget_override_expires_at"))
    return CustomerMember.model_validate(
        {
            **values,
            "budget_override": override,
            "budget_reset_at": member_window_reset_at(values.get("budget"), values.get("budget_override"), expiry, now),
            "effective_budget": effective_member_budget(template, effective_override, expiry, now),
        }
    )


def default_template_index(templates: tuple[BudgetTemplate, ...], default_name: str | None) -> int | Failure | None:
    if default_name is not None:
        named: Final = tuple(index for index, template in enumerate(templates) if template.name == default_name)
        return named[0] if named else Failure(ErrorCode.INVALID_DEFAULT_BUDGET)
    if not templates:
        return None
    if len(templates) == 1:
        return 0
    return Failure(ErrorCode.INVALID_DEFAULT_BUDGET)


def alias_patch(payload: OrganizationPatch | TeamPatch | MemberPatch) -> Mapping[str, object]:
    return {"alias": payload.alias} if "alias" in payload.model_fields_set else {}


def default_template_patch(
    organization: Organization, payload: OrganizationPatch
) -> Mapping[str, str | None] | Failure:
    if "default_budget_template_id" not in payload.model_fields_set:
        return {}
    selected: Final = payload.default_budget_template_id
    if selected is not None and all(template.budget_id != selected for template in organization.default_budgets):
        return Failure(ErrorCode.TEMPLATE_NOT_IN_ORG)
    return {"default_budget_template_id": selected}


_BUDGET_INCLUDE: Final = {"budget": True}
_MEMBER_INCLUDE: Final = {"budget": True, "budget_override": True}
_ID: Final = TypeAdapter(str)


def _member_ids(rows: Sequence[PrismaRecord]) -> tuple[str, ...]:
    return tuple(_ID.validate_python(row.model_dump()["customer_member_id"]) for row in rows)


class CustomerHierarchyStore:
    def __init__(
        self, db: "Prisma", now: datetime, settings: BudgetResetSettings | None = None, *, premium_user: bool
    ) -> None:
        self.db = db
        self.premium_user = premium_user
        self.now = now
        self.settings = settings if settings is not None else get_budget_reset_settings()
        self.organizations: TableActions[PrismaRecord] = db.litellm_customerorganizationtable  # pyright: ignore[reportAttributeAccessIssue]  # protocol accepts immutable query mappings
        self.teams: TableActions[PrismaRecord] = db.litellm_customerteamtable  # pyright: ignore[reportAttributeAccessIssue]  # protocol accepts immutable query mappings
        self.members: TableActions[PrismaRecord] = db.litellm_customermembertable  # pyright: ignore[reportAttributeAccessIssue]  # protocol accepts immutable query mappings
        self.budgets: TableActions[PrismaRecord] = db.litellm_budgettable  # pyright: ignore[reportAttributeAccessIssue]  # protocol accepts immutable query mappings
        self.window_spend: TableActions[LiteLLM_BudgetWindowSpend] = db.litellm_budgetwindowspend  # pyright: ignore[reportAttributeAccessIssue]  # protocol accepts immutable query mappings

    async def lock_free_capacity(self, scope: str | None = None) -> None:
        await self.db.query_raw("SELECT set_config('lock_timeout', $1, true)", f"{FREE_CAPACITY_LOCK_TIMEOUT_MS}ms")
        if scope is None:
            await self.db.query_raw("SELECT 1 AS locked FROM pg_advisory_xact_lock($1)", FREE_CAPACITY_ADVISORY_LOCK)
            return
        await self.db.query_raw(
            "SELECT 1 AS locked FROM pg_advisory_xact_lock(hashtextextended($1, $2))",
            scope,
            FREE_CAPACITY_ADVISORY_LOCK,
        )

    async def member_capacity_reached(self, team_ids: Sequence[str]) -> bool:
        if self.premium_user or not team_ids:
            return False
        for team_id in sorted(frozenset(team_ids)):
            await self.lock_free_capacity(f"members:{team_id}")
            if await self.members.count(where={"parent_team_ids": {"has": team_id}}) >= FREE_MEMBERS_PER_TEAM_LIMIT:
                return True
        return False

    async def portal_limits(self) -> CustomerPortalLimits:
        organization_count: Final = await self.organizations.count()
        counts: Final = TypeAdapter(tuple[_CapacityCounts, ...]).validate_python(
            await self.db.query_raw(
                'SELECT (SELECT COUNT(*) FROM (SELECT parent_org_id FROM "LiteLLM_CustomerTeamTable" '
                "GROUP BY parent_org_id HAVING COUNT(*) >= $1) AS full_orgs) AS full_organizations, "
                '(SELECT COUNT(*) FROM (SELECT team_id FROM "LiteLLM_CustomerMemberTable", '
                "unnest(parent_team_ids) AS team_id GROUP BY team_id HAVING COUNT(*) >= $2) AS full_members) AS full_teams",
                FREE_TEAMS_PER_ORGANIZATION_LIMIT,
                FREE_MEMBERS_PER_TEAM_LIMIT,
            )
        )
        return CustomerPortalLimits(
            limits_apply=not self.premium_user,
            organization_count=organization_count,
            any_limit_reached=not self.premium_user
            and (
                organization_count >= FREE_ORGANIZATION_LIMIT
                or counts[0].full_organizations > 0
                or counts[0].full_teams > 0
            ),
        )

    async def organization_responses(self, rows: Sequence[PrismaRecord]) -> tuple[Organization, ...]:
        records: Final = tuple(
            OrganizationRecord.model_validate(
                {
                    **row.model_dump(),
                    "budget": related_budget(row.model_dump().get("budget"), self.now),
                    "budget_reset_at": window_reset_at(row.model_dump().get("budget")),
                }
            )
            for row in rows
        )
        budget_ids: Final = tuple(chain.from_iterable(record.budget_template_ids for record in records))
        budget_rows: Final = await find_many_in(self.budgets, "budget_id", budget_ids)
        resolved: Final = tuple(StoredBudget.model_validate(row.model_dump()) for row in budget_rows)
        budgets: Final = {budget.budget_id: budget for budget in resolved}

        counts: Final = await self.organization_member_counts(tuple(record.customer_org_id for record in records))
        responses: Final = tuple(
            organization_response(record, budgets, counts.get(record.customer_org_id, 0)) for record in records
        )
        return await customer_budget_usage(responses, self.window_spend, self.now, self.settings)

    async def organization_member_counts(self, organization_ids: tuple[str, ...]) -> Mapping[str, int]:
        if not organization_ids:
            return {}
        async with db_span("find_many", "LiteLLM_CustomerMemberTable"):
            rows: Final = _MEMBER_COUNTS.validate_python(
                await self.db.query_raw(
                    "SELECT parent_org_id AS customer_org_id, COUNT(customer_member_id) AS member_count "
                    'FROM "LiteLLM_CustomerMemberTable" '
                    "WHERE parent_org_id = ANY($1::text[]) GROUP BY parent_org_id",
                    list(organization_ids),
                )
            )
        return {row.customer_org_id: row.member_count for row in rows}

    async def get_organization(self, organization_id: str) -> Organization | Failure:
        row: Final = await self.organizations.find_unique(
            where={"customer_org_id": organization_id}, include=_BUDGET_INCLUDE
        )
        if row is None:
            return Failure(ErrorCode.NOT_FOUND)
        organizations: Final = await self.organization_responses((row,))
        return organizations[0]

    async def get_team(self, team_id: str) -> Team | Failure:
        row: Final = await self.teams.find_unique(where={"customer_team_id": team_id}, include=_MEMBER_INCLUDE)
        if row is None:
            return Failure(ErrorCode.NOT_FOUND)
        responses: Final = await customer_budget_usage(
            (team_response(row, self.now),), self.window_spend, self.now, self.settings
        )
        return responses[0]

    async def get_member(self, member_id: str) -> CustomerMember | Failure:
        row: Final = await self.members.find_unique(where={"customer_member_id": member_id}, include=_MEMBER_INCLUDE)
        if row is None:
            return Failure(ErrorCode.NOT_FOUND)
        responses: Final = await customer_budget_usage(
            (member_response(row, self.now),), self.window_spend, self.now, self.settings
        )
        return responses[0]

    async def create_budget(self, budget: Budget, actor: str, name: str | None = None) -> str:
        budget_id: Final = str(uuid4())
        await create_budget_row(
            self.budgets,
            budget_row_values(
                {
                    **budget_values(budget),
                    **budget_window_values(budget, budget.model_fields_set),
                    **(
                        {"budget_limits": initialize_budget_limits(budget.budget_limits, self.settings, self.now)}
                        if budget.budget_limits is not None
                        else {}
                    ),
                    "allowed_models": budget.allowed_models or (),
                    "budget_id": budget_id,
                    "name": name,
                }
            ),
            actor,
        )
        return budget_id

    async def update_budget(self, budget_id: str | None, budget: Budget, actor: str) -> str:
        if budget_id is None:
            return await self.create_budget(budget, actor)
        if budget.budget_limits:
            from litellm.proxy.management_helpers.utils import validate_budget_windows_update

            await validate_budget_windows_update(self.db, budget_id)
        patch: Final = _VALUES.validate_python(budget.model_dump(exclude_unset=True))
        stored: Final = await self.budgets.find_unique(where={"budget_id": budget_id})
        current: Final = _StoredWindow.model_validate(stored, from_attributes=True) if stored is not None else None
        await update_budget_row(
            self.budgets,
            budget_id,
            budget_row_values(
                {
                    **patch,
                    **budget_window_values(budget, patch, current),
                    **(
                        {"budget_limits": initialize_budget_limits(budget.budget_limits, self.settings, self.now)}
                        if budget.budget_limits is not None
                        else {}
                    ),
                    **({"allowed_models": budget.allowed_models or ()} if "allowed_models" in patch else {}),
                }
            ),
            actor,
        )
        return budget_id

    async def write_template_metadata(
        self, organization_id: str, template_ids: Sequence[str], default_id: str | None, actor: str
    ) -> None:
        await self.organizations.update(
            where={"customer_org_id": organization_id},
            data={
                "budget_template_ids": tuple(template_ids),
                "default_budget_template_id": default_id,
                "updated_by": actor,
            },
        )

    async def lock_organization(self, organization_id: str) -> None:
        await self.db.query_raw(_LOCK_ORGANIZATION_SQL, organization_id)

    async def lock_customer_identities(self) -> None:
        await self.db.query_raw(_LOCK_CUSTOMER_IDENTITIES_SQL)

    async def claim_organization_alias(self, alias: str | None, organization_id: str | None) -> Failure | None:
        await self.lock_customer_identities()
        if alias is None:
            return None
        duplicates: Final = await self.organizations.find_many(
            where={"alias": alias, "customer_org_id": {"not": organization_id}}
            if organization_id
            else {"alias": alias},
            take=1,
        )
        return Failure(ErrorCode.ALIAS_ALREADY_EXISTS) if duplicates else None

    async def create_organization(self, payload: OrganizationCreate, actor: str) -> Organization | Failure:
        alias_failure: Final = await self.claim_organization_alias(payload.alias, None)
        if alias_failure is not None:
            return alias_failure
        if not self.premium_user:
            await self.lock_free_capacity()
            if await self.organizations.count() >= FREE_ORGANIZATION_LIMIT and (
                payload.customer_org_id is None
                or await self.organizations.find_unique(where={"customer_org_id": payload.customer_org_id}) is None
            ):
                return Failure(ErrorCode.FREE_TIER_LIMIT)
        default_index: Final = default_template_index(payload.default_budgets, payload.default_budget_template)
        if isinstance(default_index, Failure):
            return default_index
        names: Final = tuple(template.name for template in payload.default_budgets)
        if len(set(names)) != len(names):
            return Failure(ErrorCode.INVALID_TEMPLATE_NAMES)
        organization_id: Final = payload.customer_org_id or str(uuid4())
        overall_id: Final = await self.create_budget(payload.budget, actor) if payload.budget is not None else None
        await self.organizations.create(
            data={
                "customer_org_id": organization_id,
                "alias": payload.alias,
                "budget_id": overall_id,
                "created_by": actor,
                "updated_by": actor,
            }
        )
        created: Final = tuple(
            [
                await self.create_budget(
                    Budget.model_validate(template.model_dump(exclude={"name"})), actor, template.name
                )
                for template in payload.default_budgets
            ]
        )
        await self.write_template_metadata(
            organization_id,
            created,
            created[default_index] if default_index is not None else None,
            actor,
        )
        return await self.get_organization(organization_id)

    async def overall_budget_patch(
        self, current_id: str | None, payload: OrganizationPatch | TeamPatch, actor: str
    ) -> str | None:
        if "budget" not in payload.model_fields_set:
            return current_id
        if payload.budget is None:
            return None
        return await self.update_budget(current_id, payload.budget, actor)

    async def delete_released_budget(self, previous_id: str | None, current_id: str | None) -> None:
        if previous_id is not None and current_id is None:
            await self._delete_unused_budgets((previous_id,))

    async def update_organization(
        self, organization_id: str, payload: OrganizationPatch, actor: str
    ) -> Organization | Failure:
        alias_failure: Final = await self.claim_organization_alias(payload.alias, organization_id)
        if alias_failure is not None:
            return alias_failure
        await self.lock_organization(organization_id)
        organization: Final = await self.get_organization(organization_id)
        if isinstance(organization, Failure):
            return organization
        default_patch: Final = default_template_patch(organization, payload)
        if isinstance(default_patch, Failure):
            return default_patch
        budget_id: Final = await self.overall_budget_patch(organization.budget_id, payload, actor)
        await self.organizations.update(
            where={"customer_org_id": organization_id},
            data={**alias_patch(payload), **default_patch, "budget_id": budget_id, "updated_by": actor},
        )
        await self.delete_released_budget(organization.budget_id, budget_id)
        return await self.get_organization(organization_id)

    async def add_template(self, organization_id: str, payload: BudgetTemplate, actor: str) -> Organization | Failure:
        await self.lock_organization(organization_id)
        organization: Final = await self.get_organization(organization_id)
        if isinstance(organization, Failure):
            return organization
        names: Final = {template.budget_id: template.name for template in organization.default_budgets}
        if payload.name in names.values():
            return Failure(ErrorCode.INVALID_TEMPLATE_NAMES)
        budget_id: Final = await self.create_budget(
            Budget.model_validate(payload.model_dump(exclude={"name"})), actor, payload.name
        )
        default_id: Final = organization.default_budget_template_id if names else budget_id
        await self.write_template_metadata(organization_id, (*names, budget_id), default_id, actor)
        return await self.get_organization(organization_id)

    async def update_template(
        self, organization_id: str, budget_id: str, payload: TemplatePatch, actor: str
    ) -> Organization | Failure:
        await self.lock_organization(organization_id)
        organization: Final = await self.get_organization(organization_id)
        if isinstance(organization, Failure):
            return organization
        names: Final = {template.budget_id: template.name for template in organization.default_budgets}
        if budget_id not in names:
            return Failure(ErrorCode.TEMPLATE_NOT_IN_ORG)
        name: Final = payload.name if payload.name is not None else names[budget_id]
        if any(existing_name == name for existing_id, existing_name in names.items() if existing_id != budget_id):
            return Failure(ErrorCode.INVALID_TEMPLATE_NAMES)
        patch: Final = Budget.model_validate(payload.model_dump(exclude_unset=True, exclude={"name"}))
        await self.update_budget(budget_id, patch, actor)
        if name != names[budget_id]:
            await update_budget_row(self.budgets, budget_id, {"name": name}, actor)
        await self.write_template_metadata(
            organization_id, tuple(names), organization.default_budget_template_id, actor
        )
        return await self.get_organization(organization_id)

    async def delete_template(self, organization_id: str, budget_id: str, actor: str) -> None | Failure:
        await self.lock_organization(organization_id)
        organization: Final = await self.get_organization(organization_id)
        if isinstance(organization, Failure):
            return organization
        names: Final = {template.budget_id: template.name for template in organization.default_budgets}
        if budget_id not in names:
            return Failure(ErrorCode.TEMPLATE_NOT_IN_ORG)
        if organization.default_budget_template_id == budget_id:
            return Failure(ErrorCode.TEMPLATE_IN_USE)
        if await self.members.count(where={"budget_id": budget_id}):
            return Failure(ErrorCode.TEMPLATE_IN_USE)
        remaining: Final = tuple(key for key in names if key != budget_id)
        await self.write_template_metadata(organization_id, remaining, organization.default_budget_template_id, actor)
        await self._delete_unused_budgets((budget_id,))
        return None

    async def resolve_member_template(self, organization_id: str, template_id: str | None) -> str | Failure | None:
        await self.lock_organization(organization_id)
        organization: Final = await self.get_organization(organization_id)
        if isinstance(organization, Failure):
            return Failure(ErrorCode.PARENT_NOT_FOUND)
        selected: Final = template_id if template_id is not None else organization.default_budget_template_id
        if selected is None:
            return None
        if any(template.budget_id == selected for template in organization.default_budgets):
            return selected
        return Failure(ErrorCode.TEMPLATE_NOT_IN_ORG)

    async def configure_templates(
        self, organization_id: str, payload: BudgetGroupsUpdate, actor: str
    ) -> Organization | Failure:
        await self.lock_organization(organization_id)
        await self.db.query_raw(_LOCK_TEMPLATES_SQL, organization_id)
        organization: Final = await self.get_organization(organization_id)
        if isinstance(organization, Failure):
            return organization
        if payload.expected_revision != organization.budget_groups_revision:
            return Failure(ErrorCode.CONFIGURATION_CONFLICT)
        originals: Final = {group.budget_id: group for group in organization.default_budgets}
        retained: Final = tuple(group.budget_id for group in payload.default_budgets if group.budget_id is not None)
        if len(set(retained)) != len(retained) or not set(retained).issubset(originals):
            return Failure(ErrorCode.TEMPLATE_NOT_IN_ORG)
        names: Final = tuple(group.name for group in payload.default_budgets)
        if len(set(names)) != len(names):
            return Failure(ErrorCode.INVALID_TEMPLATE_NAMES)
        if payload.default_budget_template is not None and payload.default_budget_template not in names:
            return Failure(ErrorCode.INVALID_DEFAULT_BUDGET)
        removed: Final = tuple(budget_id for budget_id in originals if budget_id not in retained)
        if await count_in(self.members, "budget_id", removed):
            return Failure(ErrorCode.TEMPLATE_IN_USE)
        ids: Final = tuple(
            [
                await self.write_configured_template(group, originals.get(group.budget_id or ""), actor)
                for group in payload.default_budgets
            ]
        )
        default_id: Final = (
            ids[names.index(payload.default_budget_template)] if payload.default_budget_template else None
        )
        await self.write_template_metadata(organization_id, ids, default_id, actor)
        await self._delete_unused_budgets(removed)
        return await self.get_organization(organization_id)

    async def write_configured_template(
        self, group: BudgetGroupChange, current: OrganizationBudget | None, actor: str
    ) -> str:
        patch: Final = group_budget_patch(group, current)
        if current is None:
            return await self.create_budget(patch, actor, group.name)
        if patch.model_fields_set:
            await self.update_budget(current.budget_id, patch, actor)
        if group.name != current.name:
            await update_budget_row(self.budgets, current.budget_id, {"name": group.name}, actor)
        return current.budget_id

    async def create_team(self, payload: TeamCreate, actor: str) -> Team | Failure:
        parent: Final = await self.organizations.find_unique(where={"customer_org_id": payload.parent_org_id})
        if parent is None:
            return Failure(ErrorCode.PARENT_NOT_FOUND)
        if not self.premium_user:
            await self.lock_free_capacity(f"teams:{payload.parent_org_id}")
            if await self.teams.count(
                where={"parent_org_id": payload.parent_org_id}
            ) >= FREE_TEAMS_PER_ORGANIZATION_LIMIT and (
                payload.customer_team_id is None
                or await self.teams.find_unique(where={"customer_team_id": payload.customer_team_id}) is None
            ):
                return Failure(ErrorCode.FREE_TIER_LIMIT)
        budget_id: Final = await self.create_budget(payload.budget, actor) if payload.budget is not None else None
        override_id: Final = (
            await self.create_budget(payload.budget_override, actor) if payload.budget_override is not None else None
        )
        team_id: Final = payload.customer_team_id or str(uuid4())
        await self.teams.create(
            data={
                "customer_team_id": team_id,
                "alias": payload.alias,
                "parent_org_id": payload.parent_org_id,
                "budget_id": budget_id,
                "budget_override_id": override_id,
                "budget_override_expires_at": override_expiry(None, payload, override_id),
                "created_by": actor,
                "updated_by": actor,
            }
        )
        return await self.get_team(team_id)

    async def update_team(self, team_id: str, payload: TeamPatch, actor: str) -> Team | Failure:
        team: Final = await self.get_team(team_id)
        if isinstance(team, Failure):
            return team
        budget_id: Final = await self.overall_budget_patch(team.budget_id, payload, actor)
        override_id: Final = await self.override_patch(team.budget_override_id, payload, actor)
        await self.teams.update(
            where={"customer_team_id": team_id},
            data={
                **alias_patch(payload),
                "budget_id": budget_id,
                "budget_override_id": override_id,
                "budget_override_expires_at": override_expiry(team, payload, override_id),
                "updated_by": actor,
            },
        )
        await self.delete_released_budget(team.budget_id, budget_id)
        await self.delete_released_budget(team.budget_override_id, override_id)
        return await self.get_team(team_id)

    async def member_organization(self, payload: MemberCreate) -> str | Failure:
        """The one organization every listed team belongs to, which the payload's ``parent_org_id`` must match."""
        teams: Final = await find_many_in(self.teams, "customer_team_id", payload.parent_team_ids)
        organizations: Final = frozenset(_ID.validate_python(team.model_dump()["parent_org_id"]) for team in teams)
        if len(teams) != len(set(payload.parent_team_ids)) or len(organizations) != 1:
            return Failure(ErrorCode.PARENT_NOT_FOUND)
        organization_id: Final = next(iter(organizations))
        if payload.parent_org_id is not None and payload.parent_org_id != organization_id:
            return Failure(ErrorCode.PARENT_NOT_FOUND)
        return organization_id

    async def create_member(
        self, payload: MemberCreate, actor: str, *, budget_required: bool = True
    ) -> CustomerMember | Failure:
        organization_id: Final = await self.member_organization(payload)
        if isinstance(organization_id, Failure):
            return organization_id
        if await self.member_capacity_reached(payload.parent_team_ids) and (
            payload.customer_member_id is None
            or await self.members.find_unique(where={"customer_member_id": payload.customer_member_id}) is None
        ):
            return Failure(ErrorCode.FREE_TIER_LIMIT)
        template_id: Final = await self.resolve_member_template(organization_id, payload.budget_id)
        if isinstance(template_id, Failure):
            return template_id
        if budget_required and template_id is None and payload.budget_override is None:
            return Failure(ErrorCode.BUDGET_REQUIRED)
        override_id: Final = (
            await self.create_budget(payload.budget_override, actor) if payload.budget_override is not None else None
        )
        member_id: Final = payload.customer_member_id or str(uuid4())
        await self.members.create(
            data={
                "customer_member_id": member_id,
                "alias": payload.alias,
                "parent_org_id": organization_id,
                "parent_team_ids": tuple(dict.fromkeys(payload.parent_team_ids)),
                "budget_id": template_id,
                "budget_override_id": override_id,
                "budget_override_expires_at": override_expiry(None, payload, override_id),
                "created_by": actor,
                "updated_by": actor,
            }
        )
        return await self.get_member(member_id)

    async def member_template_patch(self, member: CustomerMember, payload: MemberPatch) -> str | Failure | None:
        if "budget_id" not in payload.model_fields_set:
            return member.budget_id
        if payload.budget_id is None:
            return None
        return await self.resolve_member_template(member.parent_org_id, payload.budget_id)

    async def override_patch(self, current_id: str | None, payload: TeamPatch | MemberPatch, actor: str) -> str | None:
        if "budget_override" not in payload.model_fields_set:
            return current_id
        if payload.budget_override is None:
            return None
        return await self.update_budget(current_id, payload.budget_override, actor)

    async def update_member(self, member_id: str, payload: MemberPatch, actor: str) -> CustomerMember | Failure:
        member: Final = await self.get_member(member_id)
        if isinstance(member, Failure):
            return member
        selected: Final = await self.member_template_patch(member, payload)
        if isinstance(selected, Failure):
            return selected
        override: Final = (
            payload.budget_override if "budget_override" in payload.model_fields_set else member.budget_override
        )
        had_budget: Final = member.budget_id is not None or member.budget_override is not None
        if had_budget and selected is None and override is None:
            return Failure(ErrorCode.BUDGET_REQUIRED)
        override_id: Final = await self.override_patch(member.budget_override_id, payload, actor)
        await self.members.update(
            where={"customer_member_id": member_id},
            data={
                **alias_patch(payload),
                **({"budget_id": selected} if "budget_id" in payload.model_fields_set else {}),
                **(
                    {
                        "budget_override_id": override_id,
                        "budget_override_expires_at": override_expiry(member, payload, override_id),
                    }
                    if payload.model_fields_set & {"budget_override", "budget_override_expires_at"}
                    else {}
                ),
                "updated_by": actor,
            },
        )
        await self.delete_released_budget(member.budget_override_id, override_id)
        return await self.get_member(member_id)

    async def list_organizations(self, page: int, page_size: int, search: str = "") -> Page[Organization]:
        where: Final = (
            {
                "OR": tuple(
                    {field: {"contains": escape_like(search), "mode": "insensitive"}}
                    for field in ("alias", "customer_org_id")
                )
            }
            if search
            else {}
        )
        count: Final = await self.organizations.count(where=where)
        rows: Final = await self.organizations.find_many(
            where=where,
            skip=(page - 1) * page_size,
            take=page_size,
            order=[{"created_at": "asc"}, {"customer_org_id": "asc"}],
            include=_BUDGET_INCLUDE,
        )
        return Page(await self.organization_responses(rows), count)

    async def list_teams(self, parent_org_id: str | None, page: int, page_size: int) -> Page[Team]:
        where: Final = {"parent_org_id": parent_org_id} if parent_org_id is not None else {}
        count: Final = await self.teams.count(where=where)
        rows: Final = await self.teams.find_many(
            where=where,
            skip=(page - 1) * page_size,
            take=page_size,
            order=[{"created_at": "asc"}, {"customer_team_id": "asc"}],
            include=_MEMBER_INCLUDE,
        )
        return Page(
            await customer_budget_usage(
                tuple(team_response(row, self.now) for row in rows), self.window_spend, self.now, self.settings
            ),
            count,
        )

    async def list_members(
        self, parent_team_id: str | None, page: int, page_size: int, *, parent_org_id: str | None = None
    ) -> Page[CustomerMember]:
        where: Final[Mapping[str, object]] = {
            **({"parent_team_ids": {"has": parent_team_id}} if parent_team_id is not None else {}),
            **({"parent_org_id": parent_org_id} if parent_org_id is not None else {}),
        }
        count: Final = await self.members.count(where=where)
        rows: Final = await self.members.find_many(
            where=where,
            skip=(page - 1) * page_size,
            take=page_size,
            order=[{"created_at": "asc"}, {"customer_member_id": "asc"}],
            include=_MEMBER_INCLUDE,
        )
        return Page(
            await customer_budget_usage(
                tuple(member_response(row, self.now) for row in rows), self.window_spend, self.now, self.settings
            ),
            count,
        )

    async def _delete_unused_budgets(self, budget_ids: Sequence[str]) -> None:
        where: Final[Mapping[str, object]] = {field: {"none": {}} for field in _BUDGET_REFERENCE_FIELDS}
        await delete_many_in(
            self.budgets,
            "budget_id",
            budget_ids,
            atomicity="caller_transaction",
            where=where,
            chunk_size=_DELETE_BATCH_SIZE,
        )

    async def _delete_team_member_batch(self, team_id: str) -> bool:
        rows: Final = await self.members.find_many(where={"parent_team_ids": {"has": team_id}}, take=_DELETE_BATCH_SIZE)
        members: Final = tuple(_MemberDeletionRow.model_validate(row) for row in rows)
        if not members:
            return False
        member_ids: Final = tuple(member.customer_member_id for member in members)
        budget_ids: Final = tuple(
            member.budget_override_id for member in members if member.budget_override_id is not None
        )
        await delete_many_in(self.members, "customer_member_id", member_ids, atomicity="caller_transaction")
        await self._delete_unused_budgets(budget_ids)
        return True

    async def _delete_team_members(self, team_id: str) -> None:
        while True:
            if not await self._delete_team_member_batch(team_id):
                return

    async def _delete_organization_team_batch(self, organization_id: str) -> bool:
        rows: Final = await self.teams.find_many(where={"parent_org_id": organization_id}, take=_DELETE_BATCH_SIZE)
        teams: Final = tuple(_TeamDeletionRow.model_validate(row) for row in rows)
        if not teams:
            return False
        for team in teams:
            await self._delete_team_members(team.customer_team_id)
        team_ids: Final = tuple(team.customer_team_id for team in teams)
        team_budget_ids: Final = chain.from_iterable((team.budget_id, team.budget_override_id) for team in teams)
        budget_ids: Final = tuple(budget_id for budget_id in team_budget_ids if budget_id is not None)
        await delete_many_in(self.teams, "customer_team_id", team_ids, atomicity="caller_transaction")
        await self._delete_unused_budgets(budget_ids)
        return True

    async def _delete_organization_teams(self, organization_id: str) -> None:
        while True:
            if not await self._delete_organization_team_batch(organization_id):
                return

    async def member_ids_in_team(self, team_id: str) -> tuple[str, ...]:
        return _member_ids(await self.members.find_many(where={"parent_team_ids": {"has": team_id}}))

    async def member_ids_in_organization(self, organization_id: str) -> tuple[str, ...]:
        return _member_ids(await self.members.find_many(where={"parent_org_id": organization_id}))

    async def member_ids_using_template(self, budget_id: str) -> tuple[str, ...]:
        return _member_ids(await self.members.find_many(where={"budget_id": budget_id}))

    async def delete_member(self, member_id: str) -> None | Failure:
        member: Final = await self.get_member(member_id)
        if isinstance(member, Failure):
            return member
        await self.members.delete(where={"customer_member_id": member_id})
        if member.budget_override_id is not None:
            await self._delete_unused_budgets((member.budget_override_id,))
        return None

    async def delete_team(self, team_id: str, *, cascade: bool = False) -> None | Failure:
        team: Final = await self.get_team(team_id)
        if isinstance(team, Failure):
            return team
        if cascade:
            await self._delete_team_members(team_id)
        elif await self.members.count(where={"parent_team_ids": {"has": team_id}}):
            return Failure(ErrorCode.HAS_CHILDREN)
        await self.teams.delete(where={"customer_team_id": team_id})
        await self._delete_unused_budgets(tuple(b for b in (team.budget_id, team.budget_override_id) if b is not None))
        return None

    async def delete_organization(self, organization_id: str, *, cascade: bool = False) -> None | Failure:
        organization: Final = await self.get_organization(organization_id)
        if isinstance(organization, Failure):
            return organization
        if cascade:
            await self._delete_organization_teams(organization_id)
        elif await self.teams.count(where={"parent_org_id": organization_id}):
            return Failure(ErrorCode.HAS_CHILDREN)
        await self.organizations.delete(where={"customer_org_id": organization_id})
        budget_ids: Final = tuple(template.budget_id for template in organization.default_budgets) + (
            (organization.budget_id,) if organization.budget_id is not None else ()
        )
        await self._delete_unused_budgets(budget_ids)
        return None


class CustomerHierarchy:
    def __init__(self, db: TransactionSource, *, premium_user: bool, clock: Callable[[], datetime] = utc_now) -> None:
        self.db = db
        self.clock = clock
        self.premium_user = premium_user

    async def portal_limits(self) -> CustomerPortalLimits | Failure:
        return await self.transaction(lambda store: store.portal_limits())

    async def transaction(
        self,
        operation: Callable[[CustomerHierarchyStore], Awaitable[T]],
        *,
        operation_type: TransactionOperation = TransactionOperation.READ_WRITE,
    ) -> T | Failure:
        from prisma.errors import DataError, ForeignKeyViolationError, RecordNotFoundError, UniqueViolationError

        try:
            async with self.db.tx() as tx:
                return await operation(CustomerHierarchyStore(tx, self.clock(), premium_user=self.premium_user))
        except UniqueViolationError:
            return Failure(ErrorCode.ALREADY_EXISTS)
        except RecordNotFoundError:
            return Failure(ErrorCode.NOT_FOUND)
        except ForeignKeyViolationError:
            return Failure(
                ErrorCode.HAS_CHILDREN if operation_type is TransactionOperation.DELETE else ErrorCode.PARENT_NOT_FOUND
            )
        except DataError as error:
            if PrismaDBExceptionHandler.is_lock_timeout_error(error):
                return Failure(ErrorCode.CAPACITY_BUSY)
            raise

    async def create_organization(self, payload: OrganizationCreate, actor: str) -> Organization | Failure:
        return await self.transaction(lambda store: store.create_organization(payload, actor))

    async def get_organization(self, entity_id: str) -> Organization | Failure:
        return await self.transaction(lambda store: store.get_organization(entity_id))

    async def update_organization(
        self, entity_id: str, payload: OrganizationPatch, actor: str
    ) -> Organization | Failure:
        return await self.transaction(lambda store: store.update_organization(entity_id, payload, actor))

    async def delete_organization(self, entity_id: str, *, cascade: bool = False) -> None | Failure:
        return await self.transaction(
            lambda store: store.delete_organization(entity_id, cascade=cascade),
            operation_type=TransactionOperation.DELETE,
        )

    async def create_team(self, payload: TeamCreate, actor: str) -> Team | Failure:
        return await self.transaction(lambda store: store.create_team(payload, actor))

    async def get_team(self, entity_id: str) -> Team | Failure:
        return await self.transaction(lambda store: store.get_team(entity_id))

    async def update_team(self, entity_id: str, payload: TeamPatch, actor: str) -> Team | Failure:
        return await self.transaction(lambda store: store.update_team(entity_id, payload, actor))

    async def delete_team(self, entity_id: str, *, cascade: bool = False) -> None | Failure:
        return await self.transaction(
            lambda store: store.delete_team(entity_id, cascade=cascade), operation_type=TransactionOperation.DELETE
        )

    async def create_member(
        self, payload: MemberCreate, actor: str, *, budget_required: bool = True
    ) -> CustomerMember | Failure:
        return await self.transaction(
            lambda store: store.create_member(payload, actor, budget_required=budget_required)
        )

    async def get_member(self, entity_id: str) -> CustomerMember | Failure:
        return await self.transaction(lambda store: store.get_member(entity_id))

    async def update_member(self, entity_id: str, payload: MemberPatch, actor: str) -> CustomerMember | Failure:
        return await self.transaction(lambda store: store.update_member(entity_id, payload, actor))

    async def delete_member(self, entity_id: str) -> None | Failure:
        return await self.transaction(
            lambda store: store.delete_member(entity_id), operation_type=TransactionOperation.DELETE
        )

    async def member_ids_in_team(self, entity_id: str) -> tuple[str, ...] | Failure:
        return await self.transaction(lambda store: store.member_ids_in_team(entity_id))

    async def member_ids_in_organization(self, entity_id: str) -> tuple[str, ...] | Failure:
        return await self.transaction(lambda store: store.member_ids_in_organization(entity_id))

    async def member_ids_using_template(self, budget_id: str) -> tuple[str, ...] | Failure:
        return await self.transaction(lambda store: store.member_ids_using_template(budget_id))

    async def list_organizations(
        self, page: int = 1, page_size: int = 50, search: str = ""
    ) -> Page[Organization] | Failure:
        return await self.transaction(lambda store: store.list_organizations(page, page_size, search))

    async def list_teams(
        self, parent_org_id: str | None = None, page: int = 1, page_size: int = 50
    ) -> Page[Team] | Failure:
        return await self.transaction(lambda store: store.list_teams(parent_org_id, page, page_size))

    async def list_members(
        self, parent_team_id: str | None = None, page: int = 1, page_size: int = 50, *, parent_org_id: str | None = None
    ) -> Page[CustomerMember] | Failure:
        return await self.transaction(
            lambda store: store.list_members(parent_team_id, page, page_size, parent_org_id=parent_org_id)
        )

    async def add_template(self, organization_id: str, payload: BudgetTemplate, actor: str) -> Organization | Failure:
        return await self.transaction(lambda store: store.add_template(organization_id, payload, actor))

    async def configure_templates(
        self, organization_id: str, payload: BudgetGroupsUpdate, actor: str
    ) -> Organization | Failure:
        return await self.transaction(lambda store: store.configure_templates(organization_id, payload, actor))

    async def update_template(
        self, organization_id: str, budget_id: str, payload: TemplatePatch, actor: str
    ) -> Organization | Failure:
        return await self.transaction(lambda store: store.update_template(organization_id, budget_id, payload, actor))

    async def delete_template(self, organization_id: str, budget_id: str, actor: str) -> None | Failure:
        return await self.transaction(
            lambda store: store.delete_template(organization_id, budget_id, actor),
            operation_type=TransactionOperation.DELETE,
        )
