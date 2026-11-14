from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel

from litellm.proxy.auth.customer_member_auto_create import DEFAULT_TEAM_ALIAS, default_team_id
from litellm.proxy.management_helpers.customer_hierarchy import (
    FREE_TEAMS_PER_ORGANIZATION_LIMIT,
    CustomerHierarchy,
    CustomerHierarchyStore,
    ErrorCode,
    Failure,
    default_template_index,
)
from litellm.types.proxy.management_endpoints.customer_hierarchy import (
    Budget,
    BudgetTemplate,
    CustomerMember,
    MemberCreate,
    MemberPatch,
    MemberUpsert,
    MemberUpsertResponse,
    Organization,
    OrganizationPatch,
    OrganizationUpsert,
    OrganizationUpsertResponse,
    Team,
)


@dataclass(frozen=True, slots=True)
class OrganizationOutcome:
    data: OrganizationUpsertResponse
    previous_alias: str | None


def matches_patch(current: BaseModel | None, patch: BaseModel | None) -> bool:
    if current is None or patch is None:
        return current is patch
    values: Final = current.model_dump()
    return all(values.get(key) == value for key, value in patch.model_dump(exclude_unset=True).items())


async def organization_by_identity(
    store: CustomerHierarchyStore, customer_org_id: str | None, alias: str | None
) -> Organization | Failure:
    if customer_org_id is not None:
        await store.lock_organization(customer_org_id)
        return await store.get_organization(customer_org_id)
    rows: Final = await store.organizations.find_many(where={"alias": alias}, take=2)
    if len(rows) > 1:
        return Failure(ErrorCode.ALIAS_ALREADY_EXISTS)
    if not rows:
        return Failure(ErrorCode.NOT_FOUND)
    organization: Final = Organization.model_validate(rows[0].model_dump())
    await store.lock_organization(organization.customer_org_id)
    return await store.get_organization(organization.customer_org_id)


def template_budget(template: BudgetTemplate) -> Budget:
    return Budget.model_validate(template.model_dump(exclude_unset=True, exclude={"name"}))


def templates_match(organization: Organization, payload: OrganizationUpsert, default_index: int | None) -> bool:
    if not payload.default_budgets:
        return True
    existing: Final = {template.name: template for template in organization.default_budgets}
    return all(
        template.name in existing and matches_patch(existing[template.name], template_budget(template))
        for template in payload.default_budgets
    ) and (
        default_index is not None
        and payload.default_budgets[default_index].name in existing
        and existing[payload.default_budgets[default_index].name].budget_id == organization.default_budget_template_id
    )


async def write_templates(
    store: CustomerHierarchyStore,
    organization: Organization,
    payload: OrganizationUpsert,
    default_index: int,
    actor: str,
) -> None:
    existing: Final = {template.name: template.budget_id for template in organization.default_budgets}
    written: Final = tuple(
        [
            await store.update_budget(existing[template.name], template_budget(template), actor)
            if template.name in existing
            else await store.create_budget(template_budget(template), actor, template.name)
            for template in payload.default_budgets
        ]
    )
    current_ids: Final = tuple(template.budget_id for template in organization.default_budgets)
    await store.write_template_metadata(
        organization.customer_org_id,
        (*current_ids, *(budget_id for budget_id in written if budget_id not in current_ids)),
        written[default_index],
        actor,
    )


async def upsert_organization(
    store: CustomerHierarchyStore, payload: OrganizationUpsert, actor: str
) -> OrganizationOutcome | Failure:
    await store.lock_customer_identities()
    default_index: Final = default_template_index(payload.default_budgets, payload.default_budget_template)
    if isinstance(default_index, Failure):
        return default_index
    names: Final = tuple(template.name for template in payload.default_budgets)
    if len(set(names)) != len(names):
        return Failure(ErrorCode.INVALID_TEMPLATE_NAMES)
    existing: Final = await organization_by_identity(store, payload.customer_org_id, payload.alias)
    if isinstance(existing, Failure):
        if existing.code is not ErrorCode.NOT_FOUND:
            return existing
        created: Final = await store.create_organization(payload, actor)
        if isinstance(created, Failure):
            return created
        return OrganizationOutcome(
            OrganizationUpsertResponse.model_validate({**created.model_dump(), "created": True}), None
        )
    patch: Final = OrganizationPatch.model_validate(
        payload.model_dump(
            exclude_unset=True, exclude={"customer_org_id", "default_budgets", "default_budget_template"}
        )
    )
    alias_failure: Final = await store.claim_organization_alias(payload.alias, existing.customer_org_id)
    if alias_failure is not None:
        return alias_failure
    alias_unchanged: Final = "alias" not in patch.model_fields_set or existing.alias == patch.alias
    budget_unchanged: Final = "budget" not in patch.model_fields_set or matches_patch(existing.budget, patch.budget)
    if alias_unchanged and budget_unchanged and templates_match(existing, payload, default_index):
        return OrganizationOutcome(
            OrganizationUpsertResponse.model_validate({**existing.model_dump(), "created": False}), existing.alias
        )
    updated: Final = await store.update_organization(existing.customer_org_id, patch, actor)
    if isinstance(updated, Failure):
        return updated
    if default_index is not None:
        await write_templates(store, updated, payload, default_index, actor)
    result: Final = await store.get_organization(existing.customer_org_id)
    if isinstance(result, Failure):
        return result
    return OrganizationOutcome(
        OrganizationUpsertResponse.model_validate({**result.model_dump(), "created": False}), existing.alias
    )


async def customer_team(
    store: CustomerHierarchyStore, organization_id: str, team_id: str | None, actor: str
) -> Team | Failure:
    if team_id is not None:
        explicit_team: Final = await store.get_team(team_id)
        if isinstance(explicit_team, Failure):
            return Failure(ErrorCode.PARENT_NOT_FOUND)
        return (
            explicit_team
            if explicit_team.parent_org_id == organization_id
            else Failure(ErrorCode.CUSTOMER_ORG_MISMATCH)
        )
    if not store.premium_user:
        await store.lock_free_capacity(f"teams:{organization_id}")
        if (
            await store.teams.find_unique(where={"customer_team_id": default_team_id(organization_id)}) is None
            and await store.teams.count(where={"parent_org_id": organization_id}) >= FREE_TEAMS_PER_ORGANIZATION_LIMIT
        ):
            return Failure(ErrorCode.FREE_TIER_LIMIT)
    await store.teams.upsert(
        where={"customer_team_id": default_team_id(organization_id)},
        data={
            "create": {
                "customer_team_id": default_team_id(organization_id),
                "alias": DEFAULT_TEAM_ALIAS,
                "parent_org_id": organization_id,
                "created_by": actor,
                "updated_by": actor,
            },
            "update": {},
        },
    )
    team: Final = await store.get_team(default_team_id(organization_id))
    if isinstance(team, Failure):
        return team
    return team if team.parent_org_id == organization_id else Failure(ErrorCode.CUSTOMER_ORG_MISMATCH)


async def member_team_ids(
    store: CustomerHierarchyStore,
    organization_id: str,
    existing: CustomerMember | Failure,
    team_id: str | None,
    actor: str,
) -> tuple[str, ...] | Failure:
    if not isinstance(existing, Failure) and team_id is None:
        return existing.parent_team_ids
    team: Final = await customer_team(store, organization_id, team_id, actor)
    if isinstance(team, Failure):
        return team
    if isinstance(existing, Failure):
        return (team.customer_team_id,)
    return (team.customer_team_id, *(tid for tid in existing.parent_team_ids if tid != team.customer_team_id))


async def upsert_member(
    store: CustomerHierarchyStore, payload: MemberUpsert, actor: str
) -> MemberUpsertResponse | Failure:
    await store.lock_customer_identities()
    organization: Final = await organization_by_identity(store, payload.customer_org_id, payload.customer_org_alias)
    if isinstance(organization, Failure):
        return Failure(ErrorCode.PARENT_NOT_FOUND) if organization.code is ErrorCode.NOT_FOUND else organization
    if payload.customer_org_alias is not None and organization.alias != payload.customer_org_alias:
        return Failure(ErrorCode.CUSTOMER_ORG_MISMATCH)
    existing: Final = await store.get_member(payload.customer_member_id)
    if not isinstance(existing, Failure):
        if existing.parent_org_id != organization.customer_org_id:
            return Failure(ErrorCode.CUSTOMER_ORG_MISMATCH)
    elif existing.code is not ErrorCode.NOT_FOUND:
        return existing
    patch: Final = MemberPatch.model_validate(
        payload.model_dump(
            exclude_unset=True,
            exclude={"customer_member_id", "customer_org_id", "customer_org_alias", "parent_team_id"},
        )
    )
    requested_budget: Final = (
        patch.budget_id
        if "budget_id" in patch.model_fields_set
        else organization.default_budget_template_id
        if isinstance(existing, Failure)
        else existing.budget_id
    )
    selected: Final = (
        requested_budget or organization.default_budget_template_id
        if isinstance(existing, Failure)
        else requested_budget
    )
    if selected is not None and not any(template.budget_id == selected for template in organization.default_budgets):
        return Failure(ErrorCode.TEMPLATE_NOT_IN_ORG)
    override: Final = (
        patch.budget_override
        if "budget_override" in patch.model_fields_set or isinstance(existing, Failure)
        else existing.budget_override
    )
    if selected is None and override is None:
        return Failure(ErrorCode.BUDGET_REQUIRED)
    team_ids: Final = await member_team_ids(
        store, organization.customer_org_id, existing, payload.parent_team_id, actor
    )
    if isinstance(team_ids, Failure):
        return team_ids
    if isinstance(existing, Failure):
        member: Final = await store.create_member(
            MemberCreate.model_validate(
                {
                    **patch.model_dump(exclude_unset=True),
                    "customer_member_id": payload.customer_member_id,
                    "parent_team_ids": team_ids,
                }
            ),
            actor,
        )
        if isinstance(member, Failure):
            return member
        return MemberUpsertResponse.model_validate({**member.model_dump(), "created": True})
    alias_matches: Final = "alias" not in patch.model_fields_set or existing.alias == patch.alias
    override_matches: Final = "budget_override" not in patch.model_fields_set or matches_patch(
        existing.budget_override, patch.budget_override
    )
    expiry_matches: Final = "budget_override_expires_at" not in patch.model_fields_set or (
        existing.budget_override_expires_at == patch.budget_override_expires_at
    )
    if (
        existing.parent_team_ids == team_ids
        and selected == existing.budget_id
        and alias_matches
        and override_matches
        and expiry_matches
    ):
        return MemberUpsertResponse.model_validate({**existing.model_dump(), "created": False})
    if await store.member_capacity_reached(tuple(tid for tid in team_ids if tid not in existing.parent_team_ids)):
        return Failure(ErrorCode.FREE_TIER_LIMIT)
    await store.members.update(
        where={"customer_member_id": payload.customer_member_id}, data={"parent_team_ids": team_ids}
    )
    updated: Final = await store.update_member(payload.customer_member_id, patch, actor)
    if isinstance(updated, Failure):
        return updated
    return MemberUpsertResponse.model_validate({**updated.model_dump(), "created": False})


class CustomerUpsert:
    def __init__(self, hierarchy: CustomerHierarchy) -> None:
        self.hierarchy = hierarchy

    async def organization(self, payload: OrganizationUpsert, actor: str) -> OrganizationOutcome | Failure:
        return await self.hierarchy.transaction(lambda store: upsert_organization(store, payload, actor))

    async def member(self, payload: MemberUpsert, actor: str) -> MemberUpsertResponse | Failure:
        return await self.hierarchy.transaction(lambda store: upsert_member(store, payload, actor))
