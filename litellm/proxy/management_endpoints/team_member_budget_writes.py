from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence
from uuid import uuid4

from litellm.proxy.common_utils.timezone_utils import get_budget_reset_time
from litellm.proxy.management_endpoints.common_utils import (
    _TEAM_MEMBER_BUDGET_LIMIT_FIELDS,
    _budget_patch_to_write_data,
    _has_meaningful_budget_limit,
    _is_set_budget_value,
)


@dataclass(frozen=True, slots=True)
class MembershipBudgetSnapshot:
    user_id: str
    budget_id: str | None


@dataclass(frozen=True, slots=True)
class BudgetFieldSnapshot:
    budget_id: str
    fields: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class DisconnectBudget:
    user_id: str


@dataclass(frozen=True, slots=True)
class UpdateBudget:
    budget_id: str
    write_data: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CreateAndAttachBudget:
    user_id: str
    budget_id: str
    create_data: Mapping[str, Any]


MemberBudgetWrite = DisconnectBudget | UpdateBudget | CreateAndAttachBudget


@dataclass(frozen=True, slots=True)
class MemberBudgetWritePlan:
    writes: tuple[MemberBudgetWrite, ...]


def _limit_fields_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in _TEAM_MEMBER_BUDGET_LIMIT_FIELDS if field in row}


def budget_snapshot_from_row(budget_id: str, row: Mapping[str, Any]) -> BudgetFieldSnapshot:
    return BudgetFieldSnapshot(budget_id=budget_id, fields=_limit_fields_from_row(row))


def _build_create_data(
    *,
    actor_user_id: str,
    write_data: Mapping[str, Any],
    shared_default_fields: Mapping[str, Any] | None,
) -> dict[str, Any]:
    create_data: dict[str, Any] = {
        "created_by": actor_user_id,
        "updated_by": actor_user_id,
    }
    if shared_default_fields is not None:
        for field in _TEAM_MEMBER_BUDGET_LIMIT_FIELDS:
            value = shared_default_fields.get(field)
            if _is_set_budget_value(value):
                create_data[field] = value
    create_data.update(write_data)
    if create_data.get("budget_duration") is not None:
        create_data["budget_reset_at"] = get_budget_reset_time(budget_duration=create_data["budget_duration"])
    else:
        create_data.pop("budget_reset_at", None)
    return create_data


def plan_member_budget_writes(
    *,
    memberships: Sequence[MembershipBudgetSnapshot],
    budgets_by_id: Mapping[str, BudgetFieldSnapshot],
    budget_patch: Mapping[str, Any],
    team_default_budget_id: str | None,
    actor_user_id: str,
    new_budget_id_factory: Callable[[], str] = lambda: str(uuid4()),
) -> MemberBudgetWritePlan:
    if not budget_patch:
        return MemberBudgetWritePlan(writes=())

    write_data = _budget_patch_to_write_data(dict(budget_patch))
    writes: list[MemberBudgetWrite] = []

    for membership in memberships:
        existing_budget_id = membership.budget_id
        is_shared_default = (
            existing_budget_id is not None
            and team_default_budget_id is not None
            and existing_budget_id == team_default_budget_id
        )

        if existing_budget_id is not None and not is_shared_default:
            existing = budgets_by_id.get(existing_budget_id)
            merged = dict(existing.fields) if existing is not None else {}
            merged.update(write_data)
            if not _has_meaningful_budget_limit(merged):
                writes.append(DisconnectBudget(user_id=membership.user_id))
                continue
            writes.append(
                UpdateBudget(
                    budget_id=existing_budget_id,
                    write_data={"updated_by": actor_user_id, **write_data},
                )
            )
            continue

        shared_fields = None
        if is_shared_default and existing_budget_id is not None:
            default_snap = budgets_by_id.get(existing_budget_id)
            if default_snap is not None:
                shared_fields = default_snap.fields

        create_data = _build_create_data(
            actor_user_id=actor_user_id,
            write_data=write_data,
            shared_default_fields=shared_fields,
        )
        if not _has_meaningful_budget_limit(create_data):
            if existing_budget_id is not None:
                writes.append(DisconnectBudget(user_id=membership.user_id))
            continue

        new_budget_id = new_budget_id_factory()
        writes.append(
            CreateAndAttachBudget(
                user_id=membership.user_id,
                budget_id=new_budget_id,
                create_data={**create_data, "budget_id": new_budget_id},
            )
        )

    return MemberBudgetWritePlan(writes=tuple(writes))


class TeamMemberBudgetDb(Protocol):
    async def create_budgets(self, rows: Sequence[Mapping[str, Any]]) -> None: ...

    async def update_budget(self, budget_id: str, data: Mapping[str, Any]) -> None: ...

    async def disconnect_membership_budget(self, *, team_id: str, user_id: str) -> None: ...

    async def attach_membership_budget(self, *, team_id: str, user_id: str, budget_id: str) -> None: ...

    async def update_team_members_with_roles(self, *, team_id: str, members_with_roles_json: str) -> Any: ...


async def apply_member_budget_write_plan(
    *,
    db: TeamMemberBudgetDb,
    team_id: str,
    plan: MemberBudgetWritePlan,
    members_with_roles_json: str | None,
) -> Any | None:
    creates = tuple(write for write in plan.writes if isinstance(write, CreateAndAttachBudget))
    updates = tuple(write for write in plan.writes if isinstance(write, UpdateBudget))
    disconnects = tuple(write for write in plan.writes if isinstance(write, DisconnectBudget))

    if creates:
        await db.create_budgets(tuple(write.create_data for write in creates))
        for write in creates:
            await db.attach_membership_budget(
                team_id=team_id,
                user_id=write.user_id,
                budget_id=write.budget_id,
            )

    for write in updates:
        await db.update_budget(write.budget_id, write.write_data)

    for write in disconnects:
        await db.disconnect_membership_budget(team_id=team_id, user_id=write.user_id)

    if members_with_roles_json is None:
        return None
    return await db.update_team_members_with_roles(
        team_id=team_id,
        members_with_roles_json=members_with_roles_json,
    )


class PrismaTeamMemberBudgetDb:
    def __init__(self, tx: Any):
        self._tx = tx

    async def create_budgets(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            return
        await self._tx.litellm_budgettable.create_many(data=list(rows))

    async def update_budget(self, budget_id: str, data: Mapping[str, Any]) -> None:
        await self._tx.litellm_budgettable.update(where={"budget_id": budget_id}, data=dict(data))

    async def disconnect_membership_budget(self, *, team_id: str, user_id: str) -> None:
        await self._tx.litellm_teammembership.update(
            where={"user_id_team_id": {"user_id": user_id, "team_id": team_id}},
            data={"litellm_budget_table": {"disconnect": True}},
        )

    async def attach_membership_budget(self, *, team_id: str, user_id: str, budget_id: str) -> None:
        await self._tx.litellm_teammembership.upsert(
            where={"user_id_team_id": {"user_id": user_id, "team_id": team_id}},
            data={
                "create": {
                    "user_id": user_id,
                    "team_id": team_id,
                    "litellm_budget_table": {"connect": {"budget_id": budget_id}},
                },
                "update": {
                    "litellm_budget_table": {"connect": {"budget_id": budget_id}},
                },
            },
        )

    async def update_team_members_with_roles(self, *, team_id: str, members_with_roles_json: str) -> Any:
        return await self._tx.litellm_teamtable.update(
            where={"team_id": team_id},
            data={"members_with_roles": members_with_roles_json},
            include={"object_permission": True},
        )


async def invalidate_team_membership_caches(
    *,
    user_ids: Sequence[str],
    team_id: str,
    user_api_key_cache: Any,
) -> None:
    for user_id in user_ids:
        await user_api_key_cache.async_delete_cache(f"team_membership:{user_id}:{team_id}")
        await user_api_key_cache.async_delete_cache(f"{team_id}_{user_id}")
