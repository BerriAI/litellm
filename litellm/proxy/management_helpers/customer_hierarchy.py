from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Final, Generic, Protocol, TypeVar
from uuid import uuid4

from pydantic import ConfigDict, TypeAdapter

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


@dataclass(frozen=True, slots=True)
class Failure:
    code: ErrorCode


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


_VALUES: Final = TypeAdapter(Mapping[str, object])
_BUDGET: Final[TypeAdapter[BudgetRecord | None]] = TypeAdapter(BudgetRecord | None)


def budget_values(budget: Budget) -> Mapping[str, object]:
    return _VALUES.validate_python(budget.model_dump())


def related_budget(value: object) -> Budget | None:
    record: Final = _BUDGET.validate_python(value)
    return Budget.model_validate(record.model_dump()) if record is not None else None


def organization_response(row: PrismaRecord) -> Organization:
    values: Final = row.model_dump()
    templates: Final = TypeAdapter(tuple[Mapping[str, object], ...]).validate_python(values.get("default_budgets", ()))
    return Organization.model_validate(
        {
            **values,
            "default_budgets": tuple(
                OrganizationBudget.model_validate(
                    {
                        **budget_values(BudgetRecord.model_validate(template["budget"])),
                        "budget_id": template["budget_id"],
                        "is_default": template["budget_id"] == values.get("default_budget_id"),
                    }
                )
                for template in templates
            ),
        }
    )


def team_response(row: PrismaRecord) -> Team:
    values: Final = row.model_dump()
    return Team.model_validate({**values, "budget": related_budget(values.get("budget"))})


def member_response(row: PrismaRecord) -> CustomerMember:
    values: Final = row.model_dump()
    return CustomerMember.model_validate({**values, "budget": related_budget(values.get("budget"))})


def valid_templates(templates: tuple[BudgetTemplate, ...] | None) -> bool:
    flagged: Final = sum(template.is_default for template in templates or ())
    return not templates or flagged == 1 or (len(templates) == 1 and flagged == 0)


def alias_patch(payload: OrganizationPatch | TeamPatch | MemberPatch) -> Mapping[str, object]:
    return {"alias": payload.alias} if "alias" in payload.model_fields_set else {}


_ORGANIZATION_INCLUDE: Final = {
    "default_budgets": {"include": {"budget": True}, "order_by": [{"created_at": "asc"}, {"budget_id": "asc"}]}
}
_BUDGET_INCLUDE: Final = {"budget": True}


class CustomerHierarchyStore:
    def __init__(self, db: "Prisma") -> None:
        self.organizations: TableActions[PrismaRecord] = db.litellm_customerorganizationtable  # pyright: ignore[reportAttributeAccessIssue]  # protocol accepts immutable query mappings
        self.teams: TableActions[PrismaRecord] = db.litellm_customerteamtable  # pyright: ignore[reportAttributeAccessIssue]  # protocol accepts immutable query mappings
        self.members: TableActions[PrismaRecord] = db.litellm_customermembertable  # pyright: ignore[reportAttributeAccessIssue]  # protocol accepts immutable query mappings
        self.budgets: TableActions[PrismaRecord] = db.litellm_budgettable  # pyright: ignore[reportAttributeAccessIssue]  # protocol accepts immutable query mappings
        self.templates: TableActions[PrismaRecord] = db.litellm_customerorganizationbudgettable  # pyright: ignore[reportAttributeAccessIssue]  # protocol accepts immutable query mappings

    async def get_organization(self, organization_id: str) -> Organization | Failure:
        row: Final = await self.organizations.find_unique(
            where={"customer_org_id": organization_id}, include=_ORGANIZATION_INCLUDE
        )
        return organization_response(row) if row is not None else Failure(ErrorCode.NOT_FOUND)

    async def get_team(self, team_id: str) -> Team | Failure:
        row: Final = await self.teams.find_unique(where={"customer_team_id": team_id}, include=_BUDGET_INCLUDE)
        return team_response(row) if row is not None else Failure(ErrorCode.NOT_FOUND)

    async def get_member(self, member_id: str) -> CustomerMember | Failure:
        row: Final = await self.members.find_unique(where={"customer_member_id": member_id}, include=_BUDGET_INCLUDE)
        return member_response(row) if row is not None else Failure(ErrorCode.NOT_FOUND)

    async def create_budget(self, budget: Budget, actor: str) -> str:
        budget_id: Final = str(uuid4())
        await self.budgets.create(
            data={
                **budget_values(budget),
                "allowed_models": budget.allowed_models or (),
                "budget_id": budget_id,
                "created_by": actor,
                "updated_by": actor,
            }
        )
        return budget_id

    async def update_budget(self, budget_id: str | None, budget: Budget, actor: str) -> str:
        if budget_id is None:
            return await self.create_budget(budget, actor)
        patch: Final = _VALUES.validate_python(budget.model_dump(exclude_unset=True))
        await self.budgets.update(
            where={"budget_id": budget_id},
            data={
                **patch,
                **({"allowed_models": budget.allowed_models or ()} if "allowed_models" in patch else {}),
                "updated_by": actor,
            },
        )
        return budget_id

    async def copy_template_budget(self, template: Budget | None, overrides: Budget | None, actor: str) -> str:
        copied: Final[Mapping[str, object]] = budget_values(template) if template is not None else {}
        patch: Final[Mapping[str, object]] = (
            _VALUES.validate_python(overrides.model_dump(exclude_unset=True)) if overrides else {}
        )
        return await self.create_budget(Budget.model_validate({**copied, **patch}), actor)

    async def replace_organization_templates(
        self, organization_id: str, templates: tuple[BudgetTemplate, ...], actor: str
    ) -> None:
        old: Final = await self.templates.find_many(where={"customer_org_id": organization_id})
        await self.organizations.update(
            where={"customer_org_id": organization_id}, data={"default_budget_id": None, "updated_by": actor}
        )
        await self.templates.delete_many(where={"customer_org_id": organization_id})
        for row in old:
            await self.budgets.delete(where={"budget_id": row.model_dump()["budget_id"]})
        created: Final = tuple(
            [await self.add_organization_template(organization_id, template, actor) for template in templates]
        )
        selected: Final = next(
            (
                budget_id
                for budget_id, template in zip(created, templates)
                if template.is_default or len(templates) == 1
            ),
            None,
        )
        await self.organizations.update(
            where={"customer_org_id": organization_id}, data={"default_budget_id": selected, "updated_by": actor}
        )

    async def add_organization_template(self, organization_id: str, template: BudgetTemplate, actor: str) -> str:
        budget: Final = Budget.model_validate(template.model_dump(exclude={"is_default"}))
        budget_id: Final = await self.create_budget(budget, actor)
        await self.templates.create(data={"customer_org_id": organization_id, "budget_id": budget_id})
        return budget_id

    async def resolve_member_template(self, team_id: str, template_id: str | None) -> Budget | Failure | None:
        team: Final = await self.get_team(team_id)
        if isinstance(team, Failure):
            return Failure(ErrorCode.PARENT_NOT_FOUND)
        organization: Final = await self.get_organization(team.parent_org_id)
        if isinstance(organization, Failure):
            return Failure(ErrorCode.PARENT_NOT_FOUND)
        selected_id: Final = template_id if template_id is not None else organization.default_budget_id
        if selected_id is None:
            return None
        selected: Final = next(
            (template for template in organization.default_budgets if template.budget_id == selected_id), None
        )
        if selected is None:
            return Failure(ErrorCode.TEMPLATE_NOT_IN_ORG)
        return Budget.model_validate(selected.model_dump(exclude={"is_default", "budget_id"}))

    async def create_organization(self, payload: OrganizationCreate, actor: str) -> Organization | Failure:
        if not valid_templates(payload.default_budgets):
            return Failure(ErrorCode.INVALID_DEFAULT_BUDGET)
        organization_id: Final = payload.customer_org_id or str(uuid4())
        await self.organizations.create(
            data={
                "customer_org_id": organization_id,
                "alias": payload.alias,
                "created_by": actor,
                "updated_by": actor,
            }
        )
        await self.replace_organization_templates(organization_id, payload.default_budgets or (), actor)
        return await self.get_organization(organization_id)

    async def update_organization(
        self, organization_id: str, payload: OrganizationPatch, actor: str
    ) -> Organization | Failure:
        if not valid_templates(payload.default_budgets):
            return Failure(ErrorCode.INVALID_DEFAULT_BUDGET)
        row: Final = await self.organizations.update(
            where={"customer_org_id": organization_id}, data={**alias_patch(payload), "updated_by": actor}
        )
        if row is None:
            return Failure(ErrorCode.NOT_FOUND)
        if payload.default_budgets is not None:
            await self.replace_organization_templates(organization_id, payload.default_budgets, actor)
        return await self.get_organization(organization_id)

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
        budget_id: Final = (
            await self.update_budget(team.budget_id, payload.budget, actor)
            if payload.budget is not None
            else team.budget_id
        )
        await self.teams.update(
            where={"customer_team_id": team_id},
            data={**alias_patch(payload), "budget_id": budget_id, "updated_by": actor},
        )
        return await self.get_team(team_id)

    async def create_member(self, payload: MemberCreate, actor: str) -> CustomerMember | Failure:
        template: Final = await self.resolve_member_template(payload.parent_team_id, payload.budget_template_id)
        if isinstance(template, Failure):
            return template
        budget_id: Final = await self.copy_template_budget(template, payload.budget, actor)
        member_id: Final = payload.customer_member_id or str(uuid4())
        await self.members.create(
            data={
                "customer_member_id": member_id,
                "alias": payload.alias,
                "parent_team_id": payload.parent_team_id,
                "budget_id": budget_id,
                "created_by": actor,
                "updated_by": actor,
            }
        )
        return await self.get_member(member_id)

    async def update_member(self, member_id: str, payload: MemberPatch, actor: str) -> CustomerMember | Failure:
        member: Final = await self.get_member(member_id)
        if isinstance(member, Failure):
            return member
        if payload.budget is not None:
            await self.update_budget(member.budget_id, payload.budget, actor)
        await self.members.update(
            where={"customer_member_id": member_id}, data={**alias_patch(payload), "updated_by": actor}
        )
        return await self.get_member(member_id)

    async def list_organizations(self, page: int, page_size: int) -> Page[Organization]:
        count: Final = await self.organizations.count()
        rows: Final = await self.organizations.find_many(
            skip=(page - 1) * page_size,
            take=page_size,
            order=[{"created_at": "asc"}, {"customer_org_id": "asc"}],
            include=_ORGANIZATION_INCLUDE,
        )
        return Page(tuple(organization_response(row) for row in rows), count)

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
            include=_BUDGET_INCLUDE,
        )
        return Page(tuple(member_response(row) for row in rows), count)

    async def delete_member(self, member_id: str) -> None | Failure:
        member: Final = await self.get_member(member_id)
        if isinstance(member, Failure):
            return member
        await self.members.delete(where={"customer_member_id": member_id})
        await self.budgets.delete(where={"budget_id": member.budget_id})
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
        await self.replace_organization_templates(organization_id, (), organization.updated_by)
        await self.organizations.delete(where={"customer_org_id": organization_id})
        return None


class CustomerHierarchy:
    def __init__(self, db: TransactionSource) -> None:
        self.db = db

    async def transaction(
        self, operation: Callable[[CustomerHierarchyStore], Awaitable[T]], *, deleting: bool = False
    ) -> T | Failure:
        from prisma.errors import ForeignKeyViolationError, RecordNotFoundError, UniqueViolationError

        try:
            async with self.db.tx() as tx:
                return await operation(CustomerHierarchyStore(tx))
        except UniqueViolationError:
            return Failure(ErrorCode.ALREADY_EXISTS)
        except RecordNotFoundError:
            return Failure(ErrorCode.NOT_FOUND)
        except ForeignKeyViolationError:
            return Failure(ErrorCode.HAS_CHILDREN if deleting else ErrorCode.PARENT_NOT_FOUND)

    async def create_organization(self, payload: OrganizationCreate, actor: str) -> Organization | Failure:
        return await self.transaction(lambda store: store.create_organization(payload, actor))

    async def get_organization(self, entity_id: str) -> Organization | Failure:
        return await self.transaction(lambda store: store.get_organization(entity_id))

    async def update_organization(
        self, entity_id: str, payload: OrganizationPatch, actor: str
    ) -> Organization | Failure:
        return await self.transaction(lambda store: store.update_organization(entity_id, payload, actor))

    async def delete_organization(self, entity_id: str) -> None | Failure:
        return await self.transaction(lambda store: store.delete_organization(entity_id), deleting=True)

    async def create_team(self, payload: TeamCreate, actor: str) -> Team | Failure:
        return await self.transaction(lambda store: store.create_team(payload, actor))

    async def get_team(self, entity_id: str) -> Team | Failure:
        return await self.transaction(lambda store: store.get_team(entity_id))

    async def update_team(self, entity_id: str, payload: TeamPatch, actor: str) -> Team | Failure:
        return await self.transaction(lambda store: store.update_team(entity_id, payload, actor))

    async def delete_team(self, entity_id: str) -> None | Failure:
        return await self.transaction(lambda store: store.delete_team(entity_id), deleting=True)

    async def create_member(self, payload: MemberCreate, actor: str) -> CustomerMember | Failure:
        return await self.transaction(lambda store: store.create_member(payload, actor))

    async def get_member(self, entity_id: str) -> CustomerMember | Failure:
        return await self.transaction(lambda store: store.get_member(entity_id))

    async def update_member(self, entity_id: str, payload: MemberPatch, actor: str) -> CustomerMember | Failure:
        return await self.transaction(lambda store: store.update_member(entity_id, payload, actor))

    async def delete_member(self, entity_id: str) -> None | Failure:
        return await self.transaction(lambda store: store.delete_member(entity_id), deleting=True)

    async def list_organizations(self, page: int = 1, page_size: int = 50) -> Page[Organization] | Failure:
        return await self.transaction(lambda store: store.list_organizations(page, page_size))

    async def list_teams(
        self, parent_org_id: str | None = None, page: int = 1, page_size: int = 50
    ) -> Page[Team] | Failure:
        return await self.transaction(lambda store: store.list_teams(parent_org_id, page, page_size))

    async def list_members(
        self, parent_team_id: str | None = None, page: int = 1, page_size: int = 50
    ) -> Page[CustomerMember] | Failure:
        return await self.transaction(lambda store: store.list_members(parent_team_id, page, page_size))
