"""One locked transaction brings a team's roster to a target set or applies a member delta, whatever its size.

The SCIM group endpoints write through here. Each member of a group push used to go through
/team/member_add or /team/member_delete on its own, so a 500-member group was thousands of
statements and outlasted the edge in front of the proxy.
"""

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict, assert_never

from litellm._uuid import uuid
from litellm.proxy._types import LiteLLM_TeamTable, Member, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import (
    delete_cache_key_objects,
    delete_cache_team_object,
    get_jwt_key_mapping_cache_keys_for_tokens,
    invalidate_team_member_spend_state,
)
from litellm.proxy.common_utils.auth_cache_invalidation_pubsub import evict_and_broadcast
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.hooks.key_management_event_hooks import KeyManagementEventHooks
from litellm.proxy.management_endpoints.key_management_endpoints import (
    _persist_deleted_verification_tokens,  # pyright: ignore[reportPrivateUsage]  # same audit path /key/delete uses
)
from litellm.proxy.management_endpoints.team_endpoints import (
    _emit_team_members_metric,  # pyright: ignore[reportPrivateUsage]  # same gauge /team/member_add moves
    _schedule_team_membership_audit_log,  # pyright: ignore[reportPrivateUsage]  # same audit row /team/member_add writes
)
from litellm.proxy.management_helpers.access_group_team_sync import TEAM_ADVISORY_LOCK_SQL
from litellm.proxy.utils import PrismaClient, ProxyLogging

if TYPE_CHECKING:
    from prisma import Prisma
    from prisma import models as prisma_models

    from litellm.repositories.prisma_protocols import TableActions

_SYNC_TX_TIMEOUT: Final = timedelta(seconds=60)
_DETACH_TEAM_SQL: Final = (
    'UPDATE "LiteLLM_UserTable" SET teams = array_remove(teams, $1) WHERE user_id = ANY($2::text[])'
)
_JSON_OBJECT: Final = TypeAdapter(dict[str, object])


@dataclass(frozen=True, slots=True)
class RosterTarget:
    member_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class RosterDelta:
    add: frozenset[str]
    remove: frozenset[str]


RosterPlan = RosterTarget | RosterDelta


@dataclass(frozen=True, slots=True)
class RosterSync:
    team: LiteLLM_TeamTable
    added: frozenset[str]
    removed: frozenset[str]


@dataclass(frozen=True, slots=True)
class TeamGone:
    team_id: str


@dataclass(frozen=True, slots=True)
class MembersMissing:
    team_id: str
    user_ids: tuple[str, ...]


RosterSyncOutcome = RosterSync | TeamGone | MembersMissing


class _MembershipData(TypedDict):
    team_id: ReadOnly[str]
    user_id: ReadOnly[str]
    budget_id: ReadOnly[str | None]


class _BudgetData(TypedDict):
    budget_id: ReadOnly[str]
    allowed_models: ReadOnly[tuple[str, ...]]
    created_by: ReadOnly[str]
    updated_by: ReadOnly[str]


class _RosterData(TypedDict):
    members_with_roles: ReadOnly[str]


@dataclass(frozen=True, slots=True)
class _Removal:
    deleted_keys: tuple["prisma_models.LiteLLM_VerificationToken", ...]
    jwt_mapping_cache_keys: tuple[str, ...]


def _team_tx_db(tx: "Prisma") -> "TableActions[prisma_models.LiteLLM_TeamTable]":
    return tx.litellm_teamtable  # pyright: ignore[reportReturnType]  # TableActions widens the generated inputs to Mapping, as the repositories do


def _user_tx_db(tx: "Prisma") -> "TableActions[prisma_models.LiteLLM_UserTable]":
    return tx.litellm_usertable  # pyright: ignore[reportReturnType]  # TableActions widens the generated inputs to Mapping, as the repositories do


def _membership_tx_db(tx: "Prisma") -> "TableActions[prisma_models.LiteLLM_TeamMembership]":
    return tx.litellm_teammembership  # pyright: ignore[reportReturnType]  # TableActions widens the generated inputs to Mapping, as the repositories do


def _budget_tx_db(tx: "Prisma") -> "TableActions[prisma_models.LiteLLM_BudgetTable]":
    return tx.litellm_budgettable  # pyright: ignore[reportReturnType]  # TableActions widens the generated inputs to Mapping, as the repositories do


def _token_tx_db(tx: "Prisma") -> "TableActions[prisma_models.LiteLLM_VerificationToken]":
    return tx.litellm_verificationtoken  # pyright: ignore[reportReturnType]  # TableActions widens the generated inputs to Mapping, as the repositories do


def _planned_changes(plan: RosterPlan, roster_ids: frozenset[str]) -> tuple[frozenset[str], frozenset[str]]:
    match plan:
        case RosterTarget(member_ids=target):
            return target - roster_ids, roster_ids - target
        case RosterDelta(add=add, remove=remove):
            return add - roster_ids, remove & roster_ids
        case _:
            assert_never(plan)


def _default_member_budget_id(team: LiteLLM_TeamTable) -> str | None:
    metadata: Final = (
        _JSON_OBJECT.validate_python(
            team.metadata  # pyright: ignore[reportUnknownMemberType]  # LiteLLM_TeamTable.metadata is a bare dict; validated by the adapter
        )
        if team.metadata  # pyright: ignore[reportUnknownMemberType]  # same bare dict
        else None
    )
    budget_id: Final = metadata.get("team_member_budget_id") if metadata is not None else None
    return budget_id if isinstance(budget_id, str) else None


async def _member_budget_ids(
    tx: "Prisma",
    team: LiteLLM_TeamTable,
    user_ids: Sequence[str],
    user_api_key_dict: UserAPIKeyAuth,
    litellm_proxy_admin_name: str,
) -> Mapping[str, str | None]:
    allowed_models: Final = tuple(team.default_team_member_models or ())
    if allowed_models:
        author: Final = user_api_key_dict.user_id or litellm_proxy_admin_name
        budgets: Final = tuple(
            _BudgetData(
                budget_id=str(uuid.uuid4()), allowed_models=allowed_models, created_by=author, updated_by=author
            )
            for _ in user_ids
        )
        await _budget_tx_db(tx).create_many(data=budgets)
        return MappingProxyType(dict(zip(user_ids, (budget["budget_id"] for budget in budgets), strict=True)))
    default_budget_id: Final = _default_member_budget_id(team)
    if default_budget_id is None:
        return MappingProxyType({})
    default_budget: Final = await _budget_tx_db(tx).find_unique(where={"budget_id": default_budget_id})
    return MappingProxyType(dict.fromkeys(user_ids, default_budget_id if default_budget is not None else None))


async def _add_members(
    tx: "Prisma",
    team: LiteLLM_TeamTable,
    user_ids: Sequence[str],
    user_api_key_dict: UserAPIKeyAuth,
    litellm_proxy_admin_name: str,
) -> tuple[Member, ...] | MembersMissing:
    if not user_ids:
        return ()
    rows: Final = await _user_tx_db(tx).find_many(where={"user_id": {"in": user_ids}})
    email_of: Final = MappingProxyType({row.user_id: row.user_email for row in rows})
    missing: Final = tuple(user_id for user_id in user_ids if user_id not in email_of)
    if missing:
        return MembersMissing(team_id=team.team_id, user_ids=missing)
    budget_ids: Final = await _member_budget_ids(tx, team, user_ids, user_api_key_dict, litellm_proxy_admin_name)
    await _membership_tx_db(tx).create_many(
        data=tuple(
            _MembershipData(team_id=team.team_id, user_id=user_id, budget_id=budget_ids.get(user_id))
            for user_id in user_ids
        ),
        skip_duplicates=True,
    )
    await _user_tx_db(tx).update_many(
        where={"user_id": {"in": user_ids}, "NOT": {"teams": {"has": team.team_id}}},
        data={"teams": {"push": [team.team_id]}},
    )
    return tuple(Member(user_id=user_id, user_email=email_of[user_id], role="user") for user_id in user_ids)


async def _remove_members(
    tx: "Prisma",
    prisma_client: PrismaClient,
    team_id: str,
    user_ids: Sequence[str],
    user_api_key_dict: UserAPIKeyAuth,
) -> _Removal:
    if not user_ids:
        return _Removal(deleted_keys=(), jwt_mapping_cache_keys=())
    team_rows_of_users: Final[Mapping[str, object]] = {"team_id": team_id, "user_id": {"in": user_ids}}
    keys: Final = tuple(await _token_tx_db(tx).find_many(where=team_rows_of_users))
    jwt_mapping_cache_keys: Final = await get_jwt_key_mapping_cache_keys_for_tokens(
        hashed_tokens=tuple(key.token for key in keys), prisma_client=prisma_client
    )
    await tx.execute_raw(_DETACH_TEAM_SQL, team_id, list(user_ids))
    await _membership_tx_db(tx).delete_many(where=team_rows_of_users)
    if keys:
        await _persist_deleted_verification_tokens(
            keys=keys,  # pyright: ignore[reportArgumentType]  # generated row model carries the same columns as LiteLLM_VerificationToken
            prisma_client=prisma_client,
            user_api_key_dict=user_api_key_dict,
            litellm_changed_by=None,
            tx=tx,
        )
        await _token_tx_db(tx).delete_many(where=team_rows_of_users)
    return _Removal(deleted_keys=keys, jwt_mapping_cache_keys=jwt_mapping_cache_keys)


async def _settle_caches(
    team: LiteLLM_TeamTable,
    changed_user_ids: Sequence[str],
    removal: _Removal,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging | None,
) -> None:
    await delete_cache_team_object(
        team_id=team.team_id,
        team_alias=team.team_alias,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )
    await delete_cache_key_objects(
        hashed_tokens=tuple(key.token for key in removal.deleted_keys),
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )
    await evict_and_broadcast(cache_keys=removal.jwt_mapping_cache_keys, user_api_key_cache=user_api_key_cache)
    await evict_and_broadcast(cache_keys=changed_user_ids, user_api_key_cache=user_api_key_cache)
    await asyncio.gather(
        *(
            invalidate_team_member_spend_state(
                user_id=user_id, team_id=team.team_id, user_api_key_cache=user_api_key_cache
            )
            for user_id in changed_user_ids
        )
    )


async def sync_team_roster(
    prisma_client: PrismaClient,
    team_id: str,
    plan: RosterPlan,
    user_api_key_dict: UserAPIKeyAuth,
    litellm_proxy_admin_name: str,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging | None,
) -> RosterSyncOutcome:
    """Reconcile the team's roster under its advisory lock in one transaction.

    The roster, every member's ``teams`` array, the membership rows and the removed
    members' team keys change together or not at all, so a failure leaves nothing
    half-applied for the caller's retry to trip over. ``TeamGone`` means the team row
    went away before the lock was taken; ``MembersMissing`` means a user the plan adds
    has no row, and nothing was written.
    """
    async with prisma_client.tx(timeout=_SYNC_TX_TIMEOUT) as tx:
        await tx.query_raw(TEAM_ADVISORY_LOCK_SQL, team_id)
        team_row: Final = await _team_tx_db(tx).find_unique(where={"team_id": team_id})
        if team_row is None:
            return TeamGone(team_id=team_id)
        team: Final = LiteLLM_TeamTable.model_validate(team_row.model_dump())
        roster: Final = tuple(team.members_with_roles)
        roster_ids: Final = frozenset(member.user_id for member in roster if member.user_id is not None)
        to_add, to_remove = _planned_changes(plan, roster_ids)
        added: Final = await _add_members(tx, team, sorted(to_add), user_api_key_dict, litellm_proxy_admin_name)
        if isinstance(added, MembersMissing):
            return added
        removal: Final = await _remove_members(tx, prisma_client, team_id, sorted(to_remove), user_api_key_dict)
        after: Final = (*(member for member in roster if member.user_id not in to_remove), *added)
        if to_add or to_remove:
            await _team_tx_db(tx).update(
                where={"team_id": team_id},
                data=_RosterData(members_with_roles=json.dumps(tuple(member.model_dump() for member in after))),
            )

    synced: Final = team.model_copy(update={"members_with_roles": list(after)})
    if removal.deleted_keys:
        KeyManagementEventHooks.create_key_deleted_audit_logs(
            keys_being_deleted=removal.deleted_keys,
            user_api_key_dict=user_api_key_dict,
            litellm_changed_by=None,
        )
    await _settle_caches(synced, sorted(to_add | to_remove), removal, user_api_key_cache, proxy_logging_obj)
    _emit_team_members_metric(synced)
    _schedule_team_membership_audit_log(
        team_id=team_id,
        team_alias=team.team_alias,
        before_members=roster,
        after_members=after,
        user_api_key_dict=user_api_key_dict,
        litellm_proxy_admin_name=litellm_proxy_admin_name,
    )
    return RosterSync(team=synced, added=to_add, removed=to_remove)
