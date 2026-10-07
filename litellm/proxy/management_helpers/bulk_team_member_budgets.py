"""Batched per-member limit writes behind `POST /management/v1/teams/{team_id}/members/bulk_update`.

Every read runs on the writer inside the batch transaction, so the write plan can never be
built from a lagging read replica. Any budget row that more than one membership points at,
the team's shared default included, is cloned before it is written, so raising one member's
cap never moves another member's.
"""

from collections.abc import Sequence
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict

from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy._types import (
    LiteLLM_TeamTable,
    LitellmTableNames,
    LitellmUserRoles,
    Member,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import invalidate_team_member_spend_state
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.db.routing_prisma_wrapper import WriterPinnedClient
from litellm.proxy.management_endpoints.common_utils import (
    _is_user_org_admin_for_team,  # pyright: ignore[reportPrivateUsage]  # same check /team/member_update uses
    _is_user_team_admin,  # pyright: ignore[reportPrivateUsage]  # same check /team/member_update uses
    _upsert_budget_and_membership,  # pyright: ignore[reportPrivateUsage]  # the single-member write, shared so the two surfaces cannot drift
    member_budget_patch,
)
from litellm.proxy.management_helpers.audit_logs import create_object_audit_log
from litellm.proxy.management_helpers.bulk_user_deletion import (
    _duplicate_member_indexes,  # pyright: ignore[reportPrivateUsage]  # same duplicate rule as members/bulk_delete
    _eq_filter,  # pyright: ignore[reportPrivateUsage]  # same prisma filter shape as members/bulk_delete
    _forbidden,  # pyright: ignore[reportPrivateUsage]  # same problem shape as members/bulk_delete
    _in_filter,  # pyright: ignore[reportPrivateUsage]  # same prisma filter shape as members/bulk_delete
    _team_not_found,  # pyright: ignore[reportPrivateUsage]  # same problem shape as members/bulk_delete
    _team_users_filter,  # pyright: ignore[reportPrivateUsage]  # same prisma filter shape as members/bulk_delete
)
from litellm.proxy.utils import PrismaClient
from litellm.repositories.team_repository import TeamRepository
from litellm.types.proxy.management_endpoints.team_endpoints import (
    BulkTeamMemberBudgetUpdateRequest,
    TeamMemberBudgetPatch,
    TeamMemberBudgetUpdateResult,
)

if TYPE_CHECKING:
    from prisma import Prisma
    from prisma import models as prisma_models

    from litellm.repositories.prisma_protocols import TableActions

_BATCH_TX_TIMEOUT: Final = timedelta(seconds=60)
_NO_METADATA: Final = MappingProxyType({})
_WITH_BUDGET: Final = MappingProxyType({"litellm_budget_table": True})


def _membership_tx_db(tx: "Prisma") -> "TableActions[prisma_models.LiteLLM_TeamMembership]":
    return tx.litellm_teammembership  # pyright: ignore[reportReturnType]  # TableActions widens the generated inputs to Mapping, as the repositories do


def _budget_tx_db(tx: "Prisma") -> "TableActions[prisma_models.LiteLLM_BudgetTable]":
    return tx.litellm_budgettable  # pyright: ignore[reportReturnType]  # TableActions widens the generated inputs to Mapping, as the repositories do


def _roster_user_id(member: TeamMemberBudgetPatch, roster: Sequence[Member]) -> str | None:
    """The team member this row addresses, or None when it names nobody on the team."""
    if member.user_id is not None:
        return member.user_id if any(m.user_id == member.user_id for m in roster) else None
    return next((m.user_id for m in roster if m.user_email is not None and m.user_email == member.user_email), None)


def _team_default_budget_id(team: LiteLLM_TeamTable) -> str | None:
    raw: Final = (team.metadata or _NO_METADATA).get("team_member_budget_id")
    return raw if isinstance(raw, str) else None


async def _shared_budget_ids(tx: "Prisma", budget_ids: frozenset[str]) -> frozenset[str]:
    """The rows in ``budget_ids`` more than one membership points at, counted across every
    team so a row shared with another team is protected too."""
    if not budget_ids:
        return frozenset()
    rows: Final = await _membership_tx_db(tx).find_many(where=_in_filter("budget_id", budget_ids))
    return frozenset(budget_id for budget_id in budget_ids if sum(1 for row in rows if row.budget_id == budget_id) > 1)


class _AuditedMemberBudget(BaseModel):
    """One member's limits as the audit log's before/after values record them."""

    model_config = ConfigDict(frozen=True)

    user_id: str
    budget_id: str | None = None
    max_budget: float | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    budget_duration: str | None = None
    budget_reset_at: datetime | None = None
    allowed_models: tuple[str, ...] | None = None


class _AuditedMemberBudgets(BaseModel):
    """The audit-log columns hold a JSON object, so the per-member list is nested under a key."""

    model_config = ConfigDict(frozen=True)

    team_member_budgets: tuple[_AuditedMemberBudget, ...]


def _audited_member_budget(row: "prisma_models.LiteLLM_TeamMembership") -> _AuditedMemberBudget:
    budget: Final = row.litellm_budget_table
    if budget is None:
        return _AuditedMemberBudget(user_id=row.user_id, budget_id=row.budget_id)
    return _AuditedMemberBudget(
        user_id=row.user_id,
        budget_id=row.budget_id,
        max_budget=budget.max_budget,
        tpm_limit=budget.tpm_limit,
        rpm_limit=budget.rpm_limit,
        budget_duration=budget.budget_duration,
        budget_reset_at=budget.budget_reset_at,
        allowed_models=tuple(budget.allowed_models),
    )


def _limits_audit_value(rows: "Sequence[prisma_models.LiteLLM_TeamMembership]") -> str:
    """Serialize the members' limits for an audit-log value, dropping the limits they do not set."""
    return safe_dumps(
        _AuditedMemberBudgets(
            team_member_budgets=tuple(_audited_member_budget(row) for row in sorted(rows, key=lambda row: row.user_id))
        ).model_dump(exclude_none=True, mode="json")
    )


def _result(
    member: TeamMemberBudgetPatch,
    user_id: str | None,
    error: str | None,
    budget_of: "MappingProxyType[str, prisma_models.LiteLLM_BudgetTable | None]",
    team_default_max_budget: float | None,
) -> TeamMemberBudgetUpdateResult:
    if error is not None or user_id is None:
        return TeamMemberBudgetUpdateResult(
            user_id=member.user_id,
            user_email=member.user_email,
            success=False,
            error=error or "User not found in team",
        )
    budget: Final = budget_of.get(user_id)
    own_max_budget: Final = budget.max_budget if budget is not None else None
    inherits: Final = own_max_budget is None and team_default_max_budget is not None and team_default_max_budget > 0
    return TeamMemberBudgetUpdateResult(
        user_id=user_id,
        user_email=member.user_email,
        success=True,
        budget_id=budget.budget_id if budget is not None else None,
        max_budget=team_default_max_budget if inherits else own_max_budget,
        max_budget_source=("team_default" if inherits else "member" if own_max_budget is not None else None),
        tpm_limit=budget.tpm_limit if budget is not None else None,
        rpm_limit=budget.rpm_limit if budget is not None else None,
        budget_duration=budget.budget_duration if budget is not None else None,
        allowed_models=tuple(budget.allowed_models) if budget is not None else None,
    )


async def bulk_update_team_member_budgets(
    team_id: str,
    data: BulkTeamMemberBudgetUpdateRequest,
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: PrismaClient,
    user_api_key_cache: UserApiKeyCache,
    litellm_proxy_admin_name: str,
    litellm_changed_by: str | None = None,
) -> tuple[TeamMemberBudgetUpdateResult, ...]:
    """Apply one merge patch of per-member limits per requested member, in one transaction."""
    team: Final = await TeamRepository(WriterPinnedClient(prisma_client.db)).find_by_id(team_id)
    if team is None:
        raise _team_not_found(team_id)

    if (
        user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN.value
        and not _is_user_team_admin(user_api_key_dict=user_api_key_dict, team_obj=team)
        and not await _is_user_org_admin_for_team(user_api_key_dict=user_api_key_dict, team_obj=team)
    ):
        raise _forbidden(
            "Call not allowed. User not proxy admin OR team admin OR org admin for this team. "
            f"route='/management/v1/teams/{team_id}/members/bulk_update'"
        )

    roster: Final = team.members_with_roles or ()
    named: Final = tuple(_roster_user_id(member, roster) for member in data.members)
    duplicates: Final = _duplicate_member_indexes(data.members) | frozenset(
        index for index, user_id in enumerate(named) if user_id is not None and user_id in named[:index]
    )
    applied: Final = tuple(
        (index, user_id) for index, user_id in enumerate(named) if user_id is not None and index not in duplicates
    )
    if not applied:
        return tuple(
            _result(
                member, None, "Duplicate member in request" if index in duplicates else None, MappingProxyType({}), None
            )
            for index, member in enumerate(data.members)
        )

    user_ids: Final = sorted(user_id for _, user_id in applied)
    default_budget_id: Final = _team_default_budget_id(team)
    team_members_filter: Final = _team_users_filter(team_id, user_ids)

    async with prisma_client.tx(timeout=_BATCH_TX_TIMEOUT) as tx:
        memberships: Final = await _membership_tx_db(tx).find_many(where=team_members_filter, include=_WITH_BUDGET)
        budget_id_of: Final = MappingProxyType({m.user_id: m.budget_id for m in memberships})
        shared: Final = await _shared_budget_ids(
            tx, frozenset(budget_id for budget_id in budget_id_of.values() if budget_id is not None)
        )
        for index, user_id in applied:
            await _upsert_budget_and_membership(
                tx=tx,
                team_id=team_id,
                user_id=user_id,
                existing_budget_id=budget_id_of.get(user_id),
                user_api_key_dict=user_api_key_dict,
                budget_patch=member_budget_patch(data.members[index]),
                team_default_budget_id=default_budget_id,
                shared_budget_ids=shared,
            )
        written: Final = await _membership_tx_db(tx).find_many(where=team_members_filter, include=_WITH_BUDGET)
        team_default: Final = (
            await _budget_tx_db(tx).find_unique(where=_eq_filter("budget_id", default_budget_id))
            if default_budget_id is not None
            else None
        )

    for user_id in user_ids:
        await invalidate_team_member_spend_state(
            user_id=user_id, team_id=team_id, user_api_key_cache=user_api_key_cache
        )

    await create_object_audit_log(
        object_id=team_id,
        action="updated",
        litellm_changed_by=litellm_changed_by,
        user_api_key_dict=user_api_key_dict,
        litellm_proxy_admin_name=litellm_proxy_admin_name,
        table_name=LitellmTableNames.TEAM_TABLE_NAME,
        before_value=_limits_audit_value(memberships),
        after_value=_limits_audit_value(written),
    )

    budget_of: Final = MappingProxyType({m.user_id: m.litellm_budget_table for m in written})
    return tuple(
        _result(
            member,
            named[index],
            "Duplicate member in request" if index in duplicates else None,
            budget_of,
            team_default.max_budget if team_default is not None else None,
        )
        for index, member in enumerate(data.members)
    )
