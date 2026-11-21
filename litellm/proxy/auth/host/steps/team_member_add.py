"""Adds the user of a JWT to the team gateway-auth-core resolved for it, as a `user`. The Rust repository writes
the team onto the user, the member's budget, the membership row and the team's roster in one transaction. This
module decides the member's budget before that write, and after it evicts the caches, updates the team members
gauge and schedules the audit log, the way `/team/member_add` does.

TODO: moves to Rust one piece at a time, together with member budgets and the audit log.

Blocked on:
- `resolve_member_budget_id`, shared with `/team/member_add`, which Rust has no port of
  - links the team's default member budget, or plans a budget of the member's own for a team with default member models
  - computes that budget's reset time from Python-owned settings
- the audit log writer and the Prometheus team members gauge, which are Python-owned
- the auth cache, which Python evicts until the Rust host owns it

After the move:
- the core decides the member's budget, so the table this module hands `resolve_member_budget_id` is deleted
- this module, `PythonAuthHost.add_team_member`, its `RustAuthStep` and its `answer.py` case are deleted"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict

from litellm._logging import verbose_proxy_logger
from litellm.models.team import LiteLLM_TeamTable
from litellm.proxy.auth.auth_checks import delete_cache_team_object, get_team_object
from litellm.proxy.common_utils.auth_cache_invalidation_pubsub import evict_and_broadcast
from litellm.repositories.budget_repository import BudgetRepository
from litellm.repositories.team_membership_repository import MemberAdded, NewMemberBudget, TeamMembershipRepository
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.types.proxy.auth.user_roles import LitellmUserRoles

if TYPE_CHECKING:
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.utils import PrismaClient, ProxyLogging


@dataclass(slots=True)
class _StoredBudget:
    budget_id: str
    row: Mapping[str, object]

    def model_dump(self) -> Mapping[str, object]:
        return self.row


class _MemberBudgetTable:
    """The budget table `resolve_member_budget_id` reads and writes through. A read goes to the database now,
    through the Rust repository. The budget it creates is kept in `planned` for the member add's transaction to
    insert, so it is stored only together with the membership it belongs to."""

    __slots__ = ("_prisma_client", "planned")

    def __init__(self, prisma_client: PrismaClient) -> None:
        self._prisma_client: Final = prisma_client
        self.planned: NewMemberBudget | None = None

    async def find_unique(self, *, where: Mapping[str, object]) -> _StoredBudget | None:
        budget_id: Final = str(where["budget_id"])
        row: Final = await BudgetRepository(self._prisma_client).find_team_member_default_row(budget_id)
        return None if row is None else _StoredBudget(budget_id, row)

    async def create(self, *, data: Mapping[str, object]) -> NewMemberBudget:
        self.planned = NewMemberBudget.model_validate({**data, "budget_id": str(uuid.uuid4())})
        return self.planned


class _MemberBudgets:
    __slots__ = ("litellm_budgettable",)

    def __init__(self, prisma_client: PrismaClient) -> None:
        self.litellm_budgettable: Final = _MemberBudgetTable(prisma_client)


class _TeamMetadata(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)

    metadata: Mapping[str, object] | None = None


def _default_member_budget_id(team: LiteLLM_TeamTable) -> str | None:
    budget_id: Final = (_TeamMetadata.model_validate(team).metadata or {}).get("team_member_budget_id")
    return budget_id if isinstance(budget_id, str) else None


async def add_team_member(
    prisma_client: PrismaClient,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging,
    litellm_proxy_admin_name: str,
    user_id: str,
    team_id: str,
) -> None:
    from litellm.proxy.management_endpoints.team_endpoints import (  # noqa: PLC0415  # team endpoints import auth
        emit_team_members_metric,
        evict_created_membership_caches,
        schedule_team_member_add_audit_logs,
    )
    from litellm.proxy.management_helpers.utils import (  # noqa: PLC0415  # management helpers import auth
        resolve_member_budget_id,
    )

    stored_team: Final = await get_team_object(
        team_id=team_id,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=None,
        proxy_logging_obj=proxy_logging_obj,
        check_cache_only=False,
        check_db_only=True,
    )
    team: Final = LiteLLM_TeamTable.model_validate(stored_team.model_dump())
    if any(member.user_id == user_id for member in team.members_with_roles):
        verbose_proxy_logger.debug("User %s is already a member of team %s", user_id, team_id)
        return

    admin: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    budgets: Final = _MemberBudgets(prisma_client)
    budget_id: Final = await resolve_member_budget_id(
        prisma_client=prisma_client,
        user_api_key_dict=admin,
        litellm_proxy_admin_name=litellm_proxy_admin_name,
        max_budget_in_team=None,
        allowed_models=team.default_team_member_models or None,
        budget_duration=None,
        default_team_budget_id=_default_member_budget_id(team),
        tx=budgets,
    )
    added: Final = await TeamMembershipRepository(prisma_client).add_member(
        user_id, team_id, budget_id=budget_id, own_budget=budgets.litellm_budgettable.planned
    )
    if not isinstance(added, MemberAdded):
        if added == "already_member":
            verbose_proxy_logger.debug("User %s is already a member of team %s", user_id, team_id)
            return
        gone: Final = f"Team={team_id}" if added == "team_missing" else f"User={user_id}"
        raise HTTPException(status_code=404, detail={"error": f"{gone} was deleted while this member add was running"})

    await delete_cache_team_object(
        team_id=team_id,
        team_alias=team.team_alias,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )
    await evict_and_broadcast(cache_keys=(user_id,), user_api_key_cache=user_api_key_cache)
    await evict_created_membership_caches(user_ids=(user_id,), team_id=team_id, user_api_key_cache=user_api_key_cache)
    emit_team_members_metric(team.model_copy(update={"members_with_roles": list(added.roster)}))
    schedule_team_member_add_audit_logs(
        team_id=team_id,
        team_alias=team.team_alias,
        updated_users=(),
        existing_user_ids=frozenset((user_id,)),
        before_members=added.roster[:-1],
        after_members=added.roster,
        user_api_key_dict=admin,
        litellm_proxy_admin_name=litellm_proxy_admin_name,
    )
    verbose_proxy_logger.debug("Successfully added user %s to team %s", user_id, team_id)
