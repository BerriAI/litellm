from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from itertools import chain
from typing import TYPE_CHECKING, Final, Generic, Protocol, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from litellm.models.budget import LiteLLM_BudgetTable
from litellm.proxy.db.db_span import db_span
from litellm.proxy.list_api.common import escape_like
from litellm.repositories.budget_repository import create_budget_row, update_budget_row
from litellm.repositories.chunked_in import find_many_in
from litellm.repositories.prisma_protocols import TableActions
from litellm.types.proxy.management_endpoints.customer_hierarchy import (
    Budget,
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
    from prisma import Prisma


class ErrorCode(str, Enum):
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


@dataclass(frozen=True, slots=True)
class Failure:
    code: ErrorCode


_LOCK_ORGANIZATION_SQL: Final = (
    'SELECT 1 AS locked FROM "LiteLLM_CustomerOrganizationTable" WHERE customer_org_id = $1 FOR UPDATE'
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


class OrganizationMemberCount(BaseModel):
    model_config = ConfigDict(frozen=True)

    customer_org_id: str
    member_count: int


_VALUES: Final = TypeAdapter(Mapping[str, object])
_MEMBER_COUNTS: Final = TypeAdapter(tuple[OrganizationMemberCount, ...])
_BUDGET: Final[TypeAdapter[BudgetRecord | None]] = TypeAdapter(BudgetRecord | None)
_EXPIRY: Final[TypeAdapter[datetime | None]] = TypeAdapter(datetime | None)


def budget_values(budget: Budget) -> Mapping[str, object]:
    return _VALUES.validate_python(budget.model_dump())


_JSON_BUDGET_COLUMNS: Final = frozenset({"model_tpm_limit", "model_rpm_limit"})


_MODEL_LIMIT_MAP: Final = TypeAdapter(dict[str, int])


def budget_row_values(values: Mapping[str, object]) -> Mapping[str, object]:
    """Prisma writes Json columns only through ``Json``; the per-model limit maps are the budget's Json columns."""
    from prisma import Json

    return {
        key: Json(_MODEL_LIMIT_MAP.validate_python(value)) if key in _JSON_BUDGET_COLUMNS else value
        for key, value in values.items()
        if not (key in _JSON_BUDGET_COLUMNS and value is None)
    }


def related_budget(value: object, now: datetime | None = None, inherited_maximum: float | None = None) -> Budget | None:
    record: Final = _BUDGET.validate_python(value)
    if record is None:
        return None
    if now is None:
        return Budget.model_validate(record.model_dump())
    budget: Final = LiteLLM_BudgetTable.model_validate(value)
    maximum: Final = budget.effective_max_budget(utc_datetime(now))
    if maximum is not None:
        return Budget.model_validate({**record.model_dump(), "max_budget": maximum})
    if inherited_maximum is None:
        return Budget.model_validate(record.model_dump())
    return Budget.model_validate(
        {
            **record.model_dump(),
            "max_budget": inherited_maximum + budget.active_temp_budget_increase(utc_datetime(now)),
        }
    )


class OrganizationRecord(Organization):
    budget_ids: tuple[str, ...] = ()
    budget_names: Mapping[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_ownership(self) -> "OrganizationRecord":
        if set(self.budget_names) != set(self.budget_ids):
            raise ValueError("Budget names must match the organization's budget IDs")
        if len(set(self.budget_names.values())) != len(self.budget_ids):
            raise ValueError("Template names must be unique")
        if self.default_budget_id is not None and self.default_budget_id not in self.budget_ids:
            raise ValueError("Default budget must belong to the organization")
        return self


class StoredBudget(BudgetRecord):
    budget_id: str


def organization_response(
    row: OrganizationRecord, budgets: Mapping[str, StoredBudget], member_count: int
) -> Organization:
    return Organization.model_validate(
        {
            **row.model_dump(exclude={"budget_ids", "budget_names"}),
            "member_count": member_count,
            "default_budgets": tuple(
                OrganizationBudget.model_validate(
                    {
                        **budgets[budget_id].model_dump(),
                        "name": row.budget_names[budget_id],
                        "is_default": budget_id == row.default_budget_id,
                    }
                )
                for budget_id in row.budget_ids
            ),
        }
    )


def team_response(row: PrismaRecord) -> Team:
    values: Final = row.model_dump()
    return Team.model_validate({**values, "budget": related_budget(values.get("budget"))})


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
    values: Final = _VALUES.validate_python(override.model_dump(exclude_none=True, exclude={"allowed_models"}))
    models: Final[Mapping[str, object]] = {"allowed_models": override.allowed_models} if override.allowed_models else {}
    return Budget.model_validate({**budget_values(inherited), **values, **models})


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
            "effective_budget": effective_member_budget(template, effective_override, expiry, now),
        }
    )


def default_template_index(templates: tuple[BudgetTemplate, ...]) -> int | Failure | None:
    if not templates:
        return None
    flagged: Final = tuple(index for index, template in enumerate(templates) if template.is_default)
    if len(flagged) > 1:
        return Failure(ErrorCode.INVALID_DEFAULT_BUDGET)
    if flagged:
        return flagged[0]
    if len(templates) == 1:
        return 0
    return Failure(ErrorCode.INVALID_DEFAULT_BUDGET)


def alias_patch(payload: OrganizationPatch | TeamPatch | MemberPatch) -> Mapping[str, object]:
    return {"alias": payload.alias} if "alias" in payload.model_fields_set else {}


_BUDGET_INCLUDE: Final = {"budget": True}
_MEMBER_INCLUDE: Final = {"budget": True, "budget_override": True}
_ID: Final = TypeAdapter(str)


def _member_ids(rows: Sequence[PrismaRecord]) -> tuple[str, ...]:
    return tuple(_ID.validate_python(row.model_dump()["customer_member_id"]) for row in rows)


class CustomerHierarchyStore:
    def __init__(self, db: "Prisma", now: datetime) -> None:
        self.db = db
        self.now = now
        self.organizations: TableActions[PrismaRecord] = db.litellm_customerorganizationtable  # pyright: ignore[reportAttributeAccessIssue]  # protocol accepts immutable query mappings
        self.teams: TableActions[PrismaRecord] = db.litellm_customerteamtable  # pyright: ignore[reportAttributeAccessIssue]  # protocol accepts immutable query mappings
        self.members: TableActions[PrismaRecord] = db.litellm_customermembertable  # pyright: ignore[reportAttributeAccessIssue]  # protocol accepts immutable query mappings
        self.budgets: TableActions[PrismaRecord] = db.litellm_budgettable  # pyright: ignore[reportAttributeAccessIssue]  # protocol accepts immutable query mappings

    async def organization_responses(self, rows: Sequence[PrismaRecord]) -> tuple[Organization, ...]:
        records: Final = tuple(
            OrganizationRecord.model_validate(
                {**row.model_dump(), "budget": related_budget(row.model_dump().get("budget"), self.now)}
            )
            for row in rows
        )
        budget_ids: Final = tuple(chain.from_iterable(record.budget_ids for record in records))
        budget_rows: Final = await find_many_in(self.budgets, "budget_id", budget_ids)
        resolved: Final = tuple(StoredBudget.model_validate(row.model_dump()) for row in budget_rows)
        budgets: Final = {budget.budget_id: budget for budget in resolved}

        counts: Final = await self.organization_member_counts(tuple(record.customer_org_id for record in records))
        return tuple(
            organization_response(record, budgets, counts.get(record.customer_org_id, 0)) for record in records
        )

    async def organization_member_counts(self, organization_ids: tuple[str, ...]) -> Mapping[str, int]:
        if not organization_ids:
            return {}
        async with db_span("find_many", "LiteLLM_CustomerTeamTable"):
            rows: Final = _MEMBER_COUNTS.validate_python(
                await self.db.query_raw(
                    "SELECT t.parent_org_id AS customer_org_id, COUNT(m.customer_member_id) AS member_count "
                    'FROM "LiteLLM_CustomerTeamTable" t '
                    'LEFT JOIN "LiteLLM_CustomerMemberTable" m ON m.parent_team_id = t.customer_team_id '
                    "WHERE t.parent_org_id = ANY($1::text[]) GROUP BY t.parent_org_id",
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
        row: Final = await self.teams.find_unique(where={"customer_team_id": team_id}, include=_BUDGET_INCLUDE)
        return team_response(row) if row is not None else Failure(ErrorCode.NOT_FOUND)

    async def get_member(self, member_id: str) -> CustomerMember | Failure:
        row: Final = await self.members.find_unique(where={"customer_member_id": member_id}, include=_MEMBER_INCLUDE)
        return member_response(row, self.now) if row is not None else Failure(ErrorCode.NOT_FOUND)

    async def create_budget(self, budget: Budget, actor: str) -> str:
        budget_id: Final = str(uuid4())
        await create_budget_row(
            self.budgets,
            budget_row_values(
                {
                    **budget_values(budget),
                    "allowed_models": budget.allowed_models or (),
                    "budget_id": budget_id,
                }
            ),
            actor,
        )
        return budget_id

    async def update_budget(self, budget_id: str | None, budget: Budget, actor: str) -> str:
        if budget_id is None:
            return await self.create_budget(budget, actor)
        patch: Final = _VALUES.validate_python(budget.model_dump(exclude_unset=True))
        await update_budget_row(
            self.budgets,
            budget_id,
            budget_row_values(
                {
                    **patch,
                    **({"allowed_models": budget.allowed_models or ()} if "allowed_models" in patch else {}),
                }
            ),
            actor,
        )
        return budget_id

    async def write_template_metadata(
        self, organization_id: str, names: Mapping[str, str], default_id: str | None, actor: str
    ) -> None:
        from prisma import Json

        await self.organizations.update(
            where={"customer_org_id": organization_id},
            data={
                "budget_ids": tuple(names),
                "budget_names": Json(dict(names)),
                "default_budget_id": default_id,
                "updated_by": actor,
            },
        )

    async def lock_organization(self, organization_id: str) -> None:
        await self.db.query_raw(_LOCK_ORGANIZATION_SQL, organization_id)

    async def create_organization(self, payload: OrganizationCreate, actor: str) -> Organization | Failure:
        default_index: Final = default_template_index(payload.default_budgets)
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
                    Budget.model_validate(template.model_dump(exclude={"name", "is_default"})), actor
                )
                for template in payload.default_budgets
            ]
        )
        await self.write_template_metadata(
            organization_id,
            dict(zip(created, names)),
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
            await self.budgets.delete(where={"budget_id": previous_id})

    async def update_organization(
        self, organization_id: str, payload: OrganizationPatch, actor: str
    ) -> Organization | Failure:
        organization: Final = await self.get_organization(organization_id)
        if isinstance(organization, Failure):
            return organization
        budget_id: Final = await self.overall_budget_patch(organization.budget_id, payload, actor)
        await self.organizations.update(
            where={"customer_org_id": organization_id},
            data={**alias_patch(payload), "budget_id": budget_id, "updated_by": actor},
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
            Budget.model_validate(payload.model_dump(exclude={"name", "is_default"})), actor
        )
        default_id: Final = budget_id if payload.is_default or not names else organization.default_budget_id
        await self.write_template_metadata(organization_id, {**names, budget_id: payload.name}, default_id, actor)
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
        if payload.is_default is False and organization.default_budget_id == budget_id:
            return Failure(ErrorCode.INVALID_DEFAULT_BUDGET)
        patch: Final = Budget.model_validate(payload.model_dump(exclude_unset=True, exclude={"name", "is_default"}))
        await self.update_budget(budget_id, patch, actor)
        default_id: Final = budget_id if payload.is_default is True else organization.default_budget_id
        await self.write_template_metadata(organization_id, {**names, budget_id: name}, default_id, actor)
        return await self.get_organization(organization_id)

    async def delete_template(self, organization_id: str, budget_id: str, actor: str) -> None | Failure:
        await self.lock_organization(organization_id)
        organization: Final = await self.get_organization(organization_id)
        if isinstance(organization, Failure):
            return organization
        names: Final = {template.budget_id: template.name for template in organization.default_budgets}
        if budget_id not in names:
            return Failure(ErrorCode.TEMPLATE_NOT_IN_ORG)
        if organization.default_budget_id == budget_id:
            return Failure(ErrorCode.TEMPLATE_IN_USE)
        if await self.members.count(where={"budget_id": budget_id}):
            return Failure(ErrorCode.TEMPLATE_IN_USE)
        remaining: Final = {key: name for key, name in names.items() if key != budget_id}
        await self.write_template_metadata(organization_id, remaining, organization.default_budget_id, actor)
        await self.budgets.delete(where={"budget_id": budget_id})
        return None

    async def resolve_member_template(self, team_id: str, template_id: str | None) -> str | Failure | None:
        team: Final = await self.get_team(team_id)
        if isinstance(team, Failure):
            return Failure(ErrorCode.PARENT_NOT_FOUND)
        organization: Final = await self.get_organization(team.parent_org_id)
        if isinstance(organization, Failure):
            return Failure(ErrorCode.PARENT_NOT_FOUND)
        selected: Final = template_id if template_id is not None else organization.default_budget_id
        if selected is None:
            return None
        if not any(template.budget_id == selected for template in organization.default_budgets):
            return Failure(ErrorCode.TEMPLATE_NOT_IN_ORG)
        return selected

    async def create_team(self, payload: TeamCreate, actor: str) -> Team | Failure:
        parent: Final = await self.organizations.find_unique(where={"customer_org_id": payload.parent_org_id})
        if parent is None:
            return Failure(ErrorCode.PARENT_NOT_FOUND)
        budget_id: Final = await self.create_budget(payload.budget, actor) if payload.budget is not None else None
        team_id: Final = payload.customer_team_id or str(uuid4())
        await self.teams.create(
            data={
                "customer_team_id": team_id,
                "alias": payload.alias,
                "parent_org_id": payload.parent_org_id,
                "budget_id": budget_id,
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
        await self.teams.update(
            where={"customer_team_id": team_id},
            data={**alias_patch(payload), "budget_id": budget_id, "updated_by": actor},
        )
        await self.delete_released_budget(team.budget_id, budget_id)
        return await self.get_team(team_id)

    async def create_member(self, payload: MemberCreate, actor: str) -> CustomerMember | Failure:
        template_id: Final = await self.resolve_member_template(payload.parent_team_id, payload.budget_id)
        if isinstance(template_id, Failure):
            return template_id
        if template_id is None and payload.budget_override is None:
            return Failure(ErrorCode.BUDGET_REQUIRED)
        override_id: Final = (
            await self.create_budget(payload.budget_override, actor) if payload.budget_override is not None else None
        )
        member_id: Final = payload.customer_member_id or str(uuid4())
        await self.members.create(
            data={
                "customer_member_id": member_id,
                "alias": payload.alias,
                "parent_team_id": payload.parent_team_id,
                "budget_id": template_id,
                "budget_override_id": override_id,
                "budget_override_expires_at": payload.budget_override_expires_at,
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
        return await self.resolve_member_template(member.parent_team_id, payload.budget_id)

    async def member_override_patch(self, member: CustomerMember, payload: MemberPatch, actor: str) -> str | None:
        if "budget_override" not in payload.model_fields_set:
            return member.budget_override_id
        if payload.budget_override is None:
            return None
        return await self.update_budget(member.budget_override_id, payload.budget_override, actor)

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
        if selected is None and override is None:
            return Failure(ErrorCode.BUDGET_REQUIRED)
        override_id: Final = await self.member_override_patch(member, payload, actor)
        expiry: Final = (
            payload.budget_override_expires_at
            if "budget_override_expires_at" in payload.model_fields_set
            else member.budget_override_expires_at
            if override_id is not None
            else None
        )
        await self.members.update(
            where={"customer_member_id": member_id},
            data={
                **alias_patch(payload),
                "budget_id": selected,
                "budget_override_id": override_id,
                "budget_override_expires_at": expiry,
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
            include=_BUDGET_INCLUDE,
        )
        return Page(tuple(team_response(row) for row in rows), count)

    async def list_members(self, parent_team_id: str | None, page: int, page_size: int) -> Page[CustomerMember]:
        where: Final = {"parent_team_id": parent_team_id} if parent_team_id is not None else {}
        count: Final = await self.members.count(where=where)
        rows: Final = await self.members.find_many(
            where=where,
            skip=(page - 1) * page_size,
            take=page_size,
            order=[{"created_at": "asc"}, {"customer_member_id": "asc"}],
            include=_MEMBER_INCLUDE,
        )
        return Page(tuple(member_response(row, self.now) for row in rows), count)

    async def member_ids_in_team(self, team_id: str) -> tuple[str, ...]:
        return _member_ids(await self.members.find_many(where={"parent_team_id": team_id}))

    async def member_ids_in_organization(self, organization_id: str) -> tuple[str, ...]:
        teams: Final = await self.teams.find_many(where={"parent_org_id": organization_id})
        team_ids: Final = tuple(_ID.validate_python(team.model_dump()["customer_team_id"]) for team in teams)
        return _member_ids(await find_many_in(self.members, "parent_team_id", team_ids))

    async def member_ids_using_template(self, budget_id: str) -> tuple[str, ...]:
        return _member_ids(await self.members.find_many(where={"budget_id": budget_id}))

    async def delete_member(self, member_id: str) -> None | Failure:
        member: Final = await self.get_member(member_id)
        if isinstance(member, Failure):
            return member
        await self.members.delete(where={"customer_member_id": member_id})
        if member.budget_override_id is not None:
            await self.budgets.delete(where={"budget_id": member.budget_override_id})
        return None

    async def delete_team(self, team_id: str) -> None | Failure:
        team: Final = await self.get_team(team_id)
        if isinstance(team, Failure):
            return team
        if await self.members.count(where={"parent_team_id": team_id}):
            return Failure(ErrorCode.HAS_CHILDREN)
        await self.teams.delete(where={"customer_team_id": team_id})
        if team.budget_id is not None:
            await self.budgets.delete(where={"budget_id": team.budget_id})
        return None

    async def delete_organization(self, organization_id: str) -> None | Failure:
        organization: Final = await self.get_organization(organization_id)
        if isinstance(organization, Failure):
            return organization
        if await self.teams.count(where={"parent_org_id": organization_id}):
            return Failure(ErrorCode.HAS_CHILDREN)
        await self.organizations.delete(where={"customer_org_id": organization_id})
        for template in organization.default_budgets:
            await self.budgets.delete(where={"budget_id": template.budget_id})
        if organization.budget_id is not None:
            await self.budgets.delete(where={"budget_id": organization.budget_id})
        return None


class CustomerHierarchy:
    def __init__(self, db: TransactionSource, clock: Callable[[], datetime] = utc_now) -> None:
        self.db = db
        self.clock = clock

    async def transaction(
        self,
        operation: Callable[[CustomerHierarchyStore], Awaitable[T]],
        *,
        operation_type: TransactionOperation = TransactionOperation.READ_WRITE,
    ) -> T | Failure:
        from prisma.errors import ForeignKeyViolationError, RecordNotFoundError, UniqueViolationError

        try:
            async with self.db.tx() as tx:
                return await operation(CustomerHierarchyStore(tx, self.clock()))
        except UniqueViolationError:
            return Failure(ErrorCode.ALREADY_EXISTS)
        except RecordNotFoundError:
            return Failure(ErrorCode.NOT_FOUND)
        except ForeignKeyViolationError:
            return Failure(
                ErrorCode.HAS_CHILDREN if operation_type is TransactionOperation.DELETE else ErrorCode.PARENT_NOT_FOUND
            )

    async def create_organization(self, payload: OrganizationCreate, actor: str) -> Organization | Failure:
        return await self.transaction(lambda store: store.create_organization(payload, actor))

    async def get_organization(self, entity_id: str) -> Organization | Failure:
        return await self.transaction(lambda store: store.get_organization(entity_id))

    async def update_organization(
        self, entity_id: str, payload: OrganizationPatch, actor: str
    ) -> Organization | Failure:
        return await self.transaction(lambda store: store.update_organization(entity_id, payload, actor))

    async def delete_organization(self, entity_id: str) -> None | Failure:
        return await self.transaction(
            lambda store: store.delete_organization(entity_id), operation_type=TransactionOperation.DELETE
        )

    async def create_team(self, payload: TeamCreate, actor: str) -> Team | Failure:
        return await self.transaction(lambda store: store.create_team(payload, actor))

    async def get_team(self, entity_id: str) -> Team | Failure:
        return await self.transaction(lambda store: store.get_team(entity_id))

    async def update_team(self, entity_id: str, payload: TeamPatch, actor: str) -> Team | Failure:
        return await self.transaction(lambda store: store.update_team(entity_id, payload, actor))

    async def delete_team(self, entity_id: str) -> None | Failure:
        return await self.transaction(
            lambda store: store.delete_team(entity_id), operation_type=TransactionOperation.DELETE
        )

    async def create_member(self, payload: MemberCreate, actor: str) -> CustomerMember | Failure:
        return await self.transaction(lambda store: store.create_member(payload, actor))

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
        self, parent_team_id: str | None = None, page: int = 1, page_size: int = 50
    ) -> Page[CustomerMember] | Failure:
        return await self.transaction(lambda store: store.list_members(parent_team_id, page, page_size))

    async def add_template(self, organization_id: str, payload: BudgetTemplate, actor: str) -> Organization | Failure:
        return await self.transaction(lambda store: store.add_template(organization_id, payload, actor))

    async def update_template(
        self, organization_id: str, budget_id: str, payload: TemplatePatch, actor: str
    ) -> Organization | Failure:
        return await self.transaction(lambda store: store.update_template(organization_id, budget_id, payload, actor))

    async def delete_template(self, organization_id: str, budget_id: str, actor: str) -> None | Failure:
        return await self.transaction(
            lambda store: store.delete_template(organization_id, budget_id, actor),
            operation_type=TransactionOperation.DELETE,
        )
