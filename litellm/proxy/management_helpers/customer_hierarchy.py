from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Final, Protocol, TypeAlias
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm.repositories.prisma_protocols import TableActions

if TYPE_CHECKING:
    from prisma import Prisma


class Budget(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_budget: float | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    allowed_models: tuple[str, ...] | None = None
    budget_duration: str | None = None


class BudgetTemplate(Budget):
    is_default: bool = False


class OrganizationBudget(BudgetTemplate):
    budget_id: str


class OrganizationPatch(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    alias: str | None = None
    default_budgets: tuple[BudgetTemplate, ...] | None = None


class OrganizationCreate(OrganizationPatch):
    customer_org_id: str | None = None


class TeamPatch(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    alias: str | None = None
    budget: Budget | None = None


class TeamCreate(TeamPatch):
    customer_team_id: str | None = None
    parent_org_id: str


class MemberCreate(TeamPatch):
    customer_member_id: str | None = None
    parent_team_id: str
    budget_template_id: str | None = None


class AuditFields(BaseModel):
    model_config = ConfigDict(frozen=True)

    alias: str | None = None
    created_at: datetime
    created_by: str
    updated_at: datetime
    updated_by: str


class Organization(AuditFields):
    customer_org_id: str
    default_budget_id: str | None = None
    default_budgets: tuple[OrganizationBudget, ...] = ()


class Team(AuditFields):
    customer_team_id: str
    parent_org_id: str
    budget_id: str | None = None
    budget: Budget | None = None


class CustomerMember(AuditFields):
    customer_member_id: str
    parent_team_id: str
    budget_id: str | None = None
    budget: Budget | None = None


class Kind(str, Enum):
    ORGANIZATION = "customer_org"
    TEAM = "customer_team"
    MEMBER = "customer_member"

    @property
    def id_field(self) -> str:
        return f"{self.value}_id"

    @property
    def budget_field(self) -> str:
        return "default_budget_id" if self is Kind.ORGANIZATION else "budget_id"

    @property
    def budget_payload(self) -> str:
        return "default_budgets" if self is Kind.ORGANIZATION else "budget"


@dataclass(frozen=True, slots=True)
class Failure:
    status: int
    detail: str


Entity: TypeAlias = Organization | Team | CustomerMember
Payload: TypeAlias = OrganizationCreate | OrganizationPatch | TeamCreate | TeamPatch | MemberCreate


class PrismaRecord(Protocol):
    def model_dump(self) -> Mapping[str, object]: ...


class TransactionSource(Protocol):
    def tx(self) -> AbstractAsyncContextManager["Prisma"]: ...


_VALUES: Final = TypeAdapter(Mapping[str, object])


class _CustomerHierarchyStore:
    def __init__(self, db: "Prisma") -> None:
        self.db = db

    def table(self, kind: Kind) -> TableActions[PrismaRecord]:
        match kind:
            case Kind.ORGANIZATION:
                return self.db.litellm_customerorganizationtable  # pyright: ignore[reportReturnType]  # protocol widens generated inputs to Mapping
            case Kind.TEAM:
                return self.db.litellm_customerteamtable  # pyright: ignore[reportReturnType]  # protocol widens generated inputs to Mapping
            case Kind.MEMBER:
                return self.db.litellm_customermembertable  # pyright: ignore[reportReturnType]  # protocol widens generated inputs to Mapping

    def budget_table(self) -> TableActions[PrismaRecord]:
        return self.db.litellm_budgettable  # pyright: ignore[reportReturnType]  # protocol widens generated inputs to Mapping

    def template_table(self) -> TableActions[PrismaRecord]:
        return self.db.litellm_customerorganizationbudgettable  # pyright: ignore[reportReturnType]  # protocol widens generated inputs to Mapping

    async def _read_budget(self, budget_id: str | None) -> Budget | None:
        row: Final = await self.budget_table().find_unique(where={"budget_id": budget_id}) if budget_id else None
        return (
            Budget.model_validate({key: value for key, value in row.model_dump().items() if key in Budget.model_fields})
            if row
            else None
        )

    async def _templates(self, org_id: str, default_id: str | None) -> tuple[OrganizationBudget, ...]:
        rows: Final = await self.template_table().find_many(
            where={"customer_org_id": org_id}, order={"budget_id": "asc"}
        )
        return tuple(
            [
                OrganizationBudget.model_validate(
                    {
                        **(
                            budget.model_dump()
                            if (budget := await self._read_budget(str(row.model_dump()["budget_id"])))
                            else {}
                        ),
                        "budget_id": row.model_dump()["budget_id"],
                        "is_default": row.model_dump()["budget_id"] == default_id,
                    }
                )
                for row in rows
            ]
        )

    async def get(self, kind: Kind, entity_id: str) -> Entity | Failure:
        row: Final = await self.table(kind).find_unique(where={kind.id_field: entity_id})
        if row is None:
            return Failure(404, "Customer entity not found.")
        values: Final = row.model_dump()
        budget_id: Final = values.get(kind.budget_field)
        budget: Final = await self._read_budget(budget_id if isinstance(budget_id, str) else None)
        resolved: Final = {**values, kind.budget_payload: budget}
        match kind:
            case Kind.ORGANIZATION:
                return Organization.model_validate(
                    {
                        **values,
                        "default_budgets": await self._templates(
                            entity_id, budget_id if isinstance(budget_id, str) else None
                        ),
                    }
                )
            case Kind.TEAM:
                return Team.model_validate(resolved)
            case Kind.MEMBER:
                return CustomerMember.model_validate(resolved)

    async def _parent_budget(self, kind: Kind, values: Mapping[str, object]) -> Budget | Failure | None:
        if kind is Kind.ORGANIZATION:
            return None
        parent_kind: Final = Kind.ORGANIZATION if kind is Kind.TEAM else Kind.TEAM
        parent_field: Final = "parent_org_id" if kind is Kind.TEAM else "parent_team_id"
        parent_id: Final = values.get(parent_field)
        parent: Final = await self.get(parent_kind, str(parent_id))
        if isinstance(parent, Failure):
            return parent
        if isinstance(parent, Team):
            org: Final = await self.get(Kind.ORGANIZATION, parent.parent_org_id)
            if isinstance(org, Failure):
                return org
            if not isinstance(org, Organization):
                return Failure(404, "Customer organization not found.")
            supplied_id: Final = values.get("budget_template_id")
            template_id: Final = org.default_budget_id if supplied_id is None else supplied_id
            selected: Final = next((budget for budget in org.default_budgets if budget.budget_id == template_id), None)
            if template_id is not None and selected is None:
                return Failure(422, "Budget template must belong to the member's organization.")
            return (
                Budget.model_validate(
                    {
                        key: value
                        for key, value in _VALUES.validate_python(selected.model_dump()).items()
                        if key in Budget.model_fields
                    }
                )
                if selected
                else None
            )
        return None

    async def _budget_id(
        self, existing_id: str | None, inherited: Budget | None, patch: Budget | None, actor: str
    ) -> str:
        inherited_values: Final = inherited.model_dump() if inherited else {}
        overrides: Final = patch.model_dump(exclude_unset=True) if patch else {}
        merged: Final = {**inherited_values, **overrides}
        models: Final = (
            {"allowed_models": merged.get("allowed_models") or ()}
            if existing_id is None or "allowed_models" in merged
            else {}
        )
        values: Final = {**merged, **models, "updated_by": actor}
        table: Final = self.budget_table()
        if existing_id is not None:
            await table.update(where={"budget_id": existing_id}, data=values)
            return existing_id
        budget_id: Final = str(uuid4())
        await table.create(data={**values, "budget_id": budget_id, "created_by": actor})
        return budget_id

    async def write(self, kind: Kind, payload: Payload, actor: str, entity_id: str | None) -> Entity | Failure:
        if isinstance(payload, OrganizationPatch):
            return await self._write_org(payload, actor, entity_id)
        values: Final = _VALUES.validate_python(payload.model_dump(exclude_unset=True))
        existing: Final = await self.get(kind, entity_id) if entity_id is not None else None
        if isinstance(existing, Failure):
            return existing
        inherited: Final = await self._parent_budget(kind, values) if entity_id is None else None
        if isinstance(inherited, Failure):
            return inherited
        patch: Final = payload.budget
        budget_patch: Final = Budget.model_validate(patch) if patch is not None else None
        existing_values: Final = existing.model_dump() if existing is not None else {}
        current_budget_id: Final = existing_values.get(kind.budget_field)
        budget_id: Final = (
            await self._budget_id(
                current_budget_id if isinstance(current_budget_id, str) else None, inherited, budget_patch, actor
            )
            if budget_patch is not None or (kind is Kind.MEMBER and entity_id is None)
            else current_budget_id
        )
        record_id: Final = entity_id or str(values.get(kind.id_field) or uuid4())
        data: Final = {
            **{
                key: value
                for key, value in values.items()
                if key not in (kind.budget_payload, kind.id_field, "budget_template_id")
            },
            kind.budget_field: budget_id,
            "updated_by": actor,
        }
        if entity_id is None:
            await self.table(kind).create(data={**data, kind.id_field: record_id, "created_by": actor})
        else:
            await self.table(kind).update(where={kind.id_field: record_id}, data=data)
        return await self.get(kind, record_id)

    async def _write_org(self, payload: OrganizationPatch, actor: str, entity_id: str | None) -> Entity | Failure:
        templates: Final = payload.default_budgets
        flagged: Final = tuple(budget for budget in templates or () if budget.is_default)
        if templates and (len(flagged) > 1 or (len(templates) > 1 and not flagged)):
            return Failure(422, "Exactly one organization budget must be marked default.")
        values: Final = payload.model_dump(exclude_unset=True)
        record_id: Final = entity_id if entity_id is not None else str(values.get("customer_org_id") or uuid4())
        data: Final = {
            **({"alias": payload.alias} if "alias" in payload.model_fields_set else {}),
            "updated_by": actor,
        }
        if entity_id is None:
            await self.table(Kind.ORGANIZATION).create(data={**data, "customer_org_id": record_id, "created_by": actor})
        elif await self.table(Kind.ORGANIZATION).update(where={"customer_org_id": record_id}, data=data) is None:
            return Failure(404, "Customer entity not found.")
        if templates is not None:
            await self.template_table().delete_many(where={"customer_org_id": record_id})
            created: Final = tuple(
                [await self._create_template(record_id, budget, actor, len(templates) == 1) for budget in templates]
            )
            default_id: Final = next((budget.budget_id for budget in created if budget.is_default), None)
            await self.table(Kind.ORGANIZATION).update(
                where={"customer_org_id": record_id}, data={"default_budget_id": default_id, "updated_by": actor}
            )
        return await self.get(Kind.ORGANIZATION, record_id)

    async def _create_template(
        self, org_id: str, template: BudgetTemplate, actor: str, single: bool
    ) -> OrganizationBudget:
        budget: Final = Budget.model_validate(template.model_dump(exclude={"is_default"}))
        budget_id: Final = await self._budget_id(None, None, budget, actor)
        await self.template_table().create(data={"customer_org_id": org_id, "budget_id": budget_id})
        return OrganizationBudget.model_validate(
            {**budget.model_dump(), "budget_id": budget_id, "is_default": single or template.is_default}
        )


class CustomerHierarchy:
    def __init__(self, db: TransactionSource) -> None:
        self.db = db

    async def get(self, kind: Kind, entity_id: str) -> Entity | Failure:
        async with self.db.tx() as tx:
            return await _CustomerHierarchyStore(tx).get(kind, entity_id)

    async def write(self, kind: Kind, payload: Payload, actor: str, entity_id: str | None = None) -> Entity | Failure:
        from prisma.errors import ForeignKeyViolationError, RecordNotFoundError, UniqueViolationError

        try:
            async with self.db.tx() as tx:
                return await _CustomerHierarchyStore(tx).write(kind, payload, actor, entity_id)
        except UniqueViolationError:
            return Failure(409, "Customer entity ID already exists.")
        except ForeignKeyViolationError:
            return Failure(404, "Customer parent or budget not found.")
        except RecordNotFoundError:
            return Failure(404, "Customer entity not found.")
