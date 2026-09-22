"""Batched deletes behind `POST /management/v1/users/bulk_delete` and
`POST /management/v1/teams/{team_id}/members/bulk_delete`.

Each team a batch touches is rewritten exactly once, under the same advisory lock
`/team/member_delete` takes and from a roster re-read under that lock, so a concurrent
member_add on the team is never overwritten from a stale read. A user batch runs in one
transaction, taking its team locks in sorted order, so either every team rewrite and every
user row delete lands or none of them does.
"""

import asyncio
import json
from collections.abc import Awaitable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from fastapi import HTTPException
from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.integrations.prometheus import PrometheusLogger
from litellm.proxy._types import (
    LiteLLM_TeamTable,
    LitellmUserRoles,
    Member,
    MemberDeleteRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import delete_cache_key_objects, get_jwt_key_mapping_cache_keys_for_tokens
from litellm.proxy.common_utils.auth_cache_invalidation_pubsub import evict_and_broadcast
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.hooks.key_management_event_hooks import KeyManagementEventHooks
from litellm.proxy.hooks.user_management_event_hooks import UserManagementEventHooks
from litellm.proxy.list_api.common import PROBLEM_TYPE_BASE, ManagementProblem
from litellm.proxy.management_endpoints.common_utils import (
    _is_user_org_admin_for_team,  # pyright: ignore[reportPrivateUsage]  # same check /team/member_delete uses
    _is_user_team_admin,  # pyright: ignore[reportPrivateUsage]  # same check /team/member_delete uses
)
from litellm.proxy.management_endpoints.key_management_endpoints import (
    _persist_deleted_verification_tokens,  # pyright: ignore[reportPrivateUsage]  # same audit path /key/delete uses
)
from litellm.proxy.management_helpers.access_group_team_sync import TEAM_ADVISORY_LOCK_SQL
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.repositories.table_repositories import (
    OrganizationMembershipRepository,
    TeamMembershipRepository,
)
from litellm.repositories.team_repository import TeamRepository
from litellm.repositories.user_repository import UserRepository
from litellm.types.proxy.management_endpoints.internal_user_endpoints import (
    BulkDeleteUserRequest,
    UserDeleteResult,
)
from litellm.types.proxy.management_endpoints.management_v1 import ProblemDetail
from litellm.types.proxy.management_endpoints.team_endpoints import (
    BulkTeamMemberDeleteRequest,
    TeamMemberDeleteResult,
)

if TYPE_CHECKING:
    from prisma import Prisma
    from prisma import models as prisma_models

    from litellm.repositories.prisma_protocols import TableActions

_AUDIT_LOG_CONCURRENCY: Final = 10
_BATCH_TX_TIMEOUT: Final = timedelta(seconds=60)


class _OrgAdminFilter(TypedDict):
    user_id: ReadOnly[str]
    user_role: ReadOnly[str]


class _RosterData(TypedDict):
    members_with_roles: ReadOnly[str]


class _TeamsSet(TypedDict):
    set: ReadOnly[tuple[str, ...]]


class _TeamsData(TypedDict):
    teams: ReadOnly[_TeamsSet]


@dataclass(frozen=True, slots=True)
class _TeamRemoval:
    """One team's rewrite. `removed` holds the user ids taken off the team (roster, `teams` array, or both);
    `matched` holds the indexes into the requested members that named at least one of them."""

    team: LiteLLM_TeamTable
    removed: frozenset[str]
    matched: frozenset[int]
    deleted_keys: tuple["prisma_models.LiteLLM_VerificationToken", ...]
    jwt_mapping_cache_keys: tuple[str, ...]

    @property
    def deleted_key_tokens(self) -> tuple[str, ...]:
        return tuple(k.token for k in self.deleted_keys)


@dataclass(frozen=True, slots=True)
class _UserBatchDeletion:
    removals: Mapping[str, _TeamRemoval]
    deleted_keys: tuple["prisma_models.LiteLLM_VerificationToken", ...]
    jwt_mapping_cache_keys: tuple[str, ...]

    @property
    def deleted_key_tokens(self) -> tuple[str, ...]:
        return tuple(k.token for k in self.deleted_keys)


@dataclass(frozen=True, slots=True)
class _DeletedKeys:
    keys: tuple["prisma_models.LiteLLM_VerificationToken", ...]
    jwt_mapping_cache_keys: tuple[str, ...]


def _team_not_found(team_id: str) -> ManagementProblem:
    return ManagementProblem(
        ProblemDetail(
            type=f"{PROBLEM_TYPE_BASE}team-not-found",
            title="Team not found",
            status=404,
            detail=f"Team id={team_id} does not exist in db",
        )
    )


def _forbidden(detail: str) -> ManagementProblem:
    return ManagementProblem(
        ProblemDetail(type=f"{PROBLEM_TYPE_BASE}forbidden", title="Forbidden", status=403, detail=detail)
    )


def _in_filter(field: str, values: Iterable[str]) -> Mapping[str, object]:
    return {field: {"in": sorted(values)}}  # mutable-ok: Prisma query filters are dict-shaped


def _eq_filter(field: str, value: str) -> Mapping[str, object]:
    return {field: value}  # mutable-ok: Prisma query filters are dict-shaped


def _team_users_filter(team_id: str, user_ids: Iterable[str]) -> Mapping[str, object]:
    return {"team_id": team_id, **_in_filter("user_id", user_ids)}  # mutable-ok: Prisma query filters are dict-shaped


def _any_filter(*clauses: Mapping[str, object]) -> Mapping[str, object]:
    return {"OR": clauses}  # mutable-ok: Prisma query filters are dict-shaped


def _team_tx_db(tx: "Prisma") -> "TableActions[prisma_models.LiteLLM_TeamTable]":
    return tx.litellm_teamtable  # pyright: ignore[reportReturnType]  # TableActions widens the generated inputs to Mapping, as the repositories do


def _user_tx_db(tx: "Prisma") -> "TableActions[prisma_models.LiteLLM_UserTable]":
    return tx.litellm_usertable  # pyright: ignore[reportReturnType]  # TableActions widens the generated inputs to Mapping, as the repositories do


def _membership_tx_db(tx: "Prisma") -> "TableActions[prisma_models.LiteLLM_TeamMembership]":
    return tx.litellm_teammembership  # pyright: ignore[reportReturnType]  # TableActions widens the generated inputs to Mapping, as the repositories do


def _token_tx_db(tx: "Prisma") -> "TableActions[prisma_models.LiteLLM_VerificationToken]":
    return tx.litellm_verificationtoken  # pyright: ignore[reportReturnType]  # TableActions widens the generated inputs to Mapping, as the repositories do


def _invitation_tx_db(tx: "Prisma") -> "TableActions[prisma_models.LiteLLM_InvitationLink]":
    return tx.litellm_invitationlink  # pyright: ignore[reportReturnType]  # TableActions widens the generated inputs to Mapping, as the repositories do


def _org_membership_tx_db(tx: "Prisma") -> "TableActions[prisma_models.LiteLLM_OrganizationMembership]":
    return tx.litellm_organizationmembership  # pyright: ignore[reportReturnType]  # TableActions widens the generated inputs to Mapping, as the repositories do


def _same_email(email: str | None, request: MemberDeleteRequest) -> bool:
    return request.user_email is not None and request.user_email == email


def _addresses_member(member: Member, request: MemberDeleteRequest) -> bool:
    if request.user_id is None:
        return _same_email(member.user_email, request)
    return request.user_id == member.user_id or (member.user_id is None and _same_email(member.user_email, request))


def _with_row_email(request: MemberDeleteRequest, email_of: Mapping[str, str]) -> MemberDeleteRequest:
    if request.user_id is None or request.user_email is not None:
        return request
    return MemberDeleteRequest(user_id=request.user_id, user_email=email_of.get(request.user_id))


def _addresses_user(user: "prisma_models.LiteLLM_UserTable", request: MemberDeleteRequest) -> bool:
    if request.user_id is None:
        return _same_email(user.user_email, request)
    return request.user_id == user.user_id


def _error_message(exc: BaseException) -> str:
    if isinstance(exc, ManagementProblem):
        return exc.problem.detail
    if isinstance(exc, HTTPException) and isinstance(exc.detail, dict):
        return str(exc.detail.get("error", exc.detail))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # HTTPException.detail is untyped
    if isinstance(exc, HTTPException):
        return str(exc.detail)  # pyright: ignore[reportUnknownArgumentType]  # HTTPException.detail is untyped
    return str(exc) or type(exc).__name__


async def _bounded(awaitables: Iterable[Awaitable[object]]) -> tuple[object | BaseException, ...]:
    semaphore: Final = asyncio.Semaphore(_AUDIT_LOG_CONCURRENCY)

    async def run(awaitable: Awaitable[object]) -> object:
        async with semaphore:
            return await awaitable

    return tuple(await asyncio.gather(*(run(a) for a in awaitables), return_exceptions=True))


async def _remove_members_from_team(
    prisma_client: PrismaClient,
    tx: "Prisma",
    team_id: str,
    members: Sequence[MemberDeleteRequest],
    user_api_key_dict: UserAPIKeyAuth,
) -> _TeamRemoval:
    await tx.query_raw(TEAM_ADVISORY_LOCK_SQL, team_id)
    roster: Final = await TeamRepository(prisma_client).get_members_with_roles_locked(tx, team_id)
    if roster is None:
        raise _team_not_found(team_id)

    requested_ids: Final = frozenset(r.user_id for r in members if r.user_id is not None)
    requested_emails: Final = frozenset(r.user_email for r in members if r.user_id is None and r.user_email)
    requested_rows: Final = await _user_tx_db(tx).find_many(
        where=_any_filter(_in_filter("user_id", requested_ids), _in_filter("user_email", requested_emails))
    )
    email_of: Final = MappingProxyType(
        {u.user_id: u.user_email for u in requested_rows if u.user_email is not None and team_id in u.teams}
    )
    requests: Final = tuple(_with_row_email(r, email_of) for r in members)
    removed_members: Final = tuple(m for m in roster if any(_addresses_member(m, r) for r in requests))
    kept_members: Final = tuple(m for m in roster if not any(_addresses_member(m, r) for r in requests))
    removed_ids: Final = frozenset(m.user_id for m in removed_members if m.user_id is not None)
    unfetched_ids: Final = removed_ids - frozenset(u.user_id for u in requested_rows)
    removed_rows: Final = (
        await _user_tx_db(tx).find_many(where=_in_filter("user_id", unfetched_ids)) if unfetched_ids else ()
    )
    stale_rows: Final = tuple(u for u in (*requested_rows, *removed_rows) if team_id in u.teams)
    cleanup_ids: Final = removed_ids | frozenset(u.user_id for u in stale_rows)
    matched: Final = frozenset(
        i
        for i, r in enumerate(requests)
        if any(_addresses_member(m, r) for m in removed_members) or any(_addresses_user(u, r) for u in stale_rows)
    )
    keys: Final = await _token_tx_db(tx).find_many(where=_team_users_filter(team_id, cleanup_ids))
    jwt_mapping_cache_keys: Final = await get_jwt_key_mapping_cache_keys_for_tokens(
        hashed_tokens=tuple(k.token for k in keys),
        prisma_client=prisma_client,
    )

    if removed_members:
        roster_data: Final[_RosterData] = {
            "members_with_roles": json.dumps(tuple(m.model_dump() for m in kept_members))
        }
        await _team_tx_db(tx).update(where=_eq_filter("team_id", team_id), data=roster_data)
    for row in stale_rows:
        teams_data: _TeamsData = {"teams": {"set": tuple(t for t in row.teams if t != team_id)}}
        await _user_tx_db(tx).update(where=_eq_filter("user_id", row.user_id), data=teams_data)
    await _membership_tx_db(tx).delete_many(where=_team_users_filter(team_id, cleanup_ids))
    if keys:
        await _persist_deleted_verification_tokens(
            keys=keys,  # pyright: ignore[reportArgumentType]  # generated row model carries the same columns as LiteLLM_VerificationToken
            prisma_client=prisma_client,
            user_api_key_dict=user_api_key_dict,
            litellm_changed_by=None,
            tx=tx,
        )
        await _token_tx_db(tx).delete_many(where=_team_users_filter(team_id, cleanup_ids))

    return _TeamRemoval(
        team=LiteLLM_TeamTable(
            team_id=team_id,
            members_with_roles=kept_members,  # pyright: ignore[reportArgumentType]  # pydantic coerces the tuple into the list field
        ),
        removed=cleanup_ids,
        matched=matched,
        deleted_keys=tuple(keys),
        jwt_mapping_cache_keys=jwt_mapping_cache_keys,
    )


def _emit_team_members_metric(team: LiteLLM_TeamTable) -> None:
    prometheus_logger: Final = PrometheusLogger.get_instance()
    if prometheus_logger is None:
        return
    try:
        prometheus_logger.set_team_members_metric(team)
    except Exception as e:
        verbose_proxy_logger.debug("Prometheus: failed to emit team members metric: %s", str(e))


def _duplicate_member_indexes(members: Sequence[MemberDeleteRequest]) -> frozenset[int]:
    return frozenset(
        i
        for i, m in enumerate(members)
        if any(
            (m.user_id is not None and m.user_id == earlier.user_id)
            or (m.user_email is not None and m.user_email == earlier.user_email)
            for earlier in members[:i]
        )
    )


async def bulk_remove_team_members(
    team_id: str,
    data: BulkTeamMemberDeleteRequest,
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: PrismaClient,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging | None,
) -> tuple[TeamMemberDeleteResult, ...]:
    team: Final = await TeamRepository(prisma_client).find_by_id(team_id)
    if team is None:
        raise _team_not_found(team_id)

    if (
        user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN.value
        and not _is_user_team_admin(user_api_key_dict=user_api_key_dict, team_obj=team)
        and not await _is_user_org_admin_for_team(user_api_key_dict=user_api_key_dict, team_obj=team)
    ):
        raise _forbidden(
            "Call not allowed. User not proxy admin OR team admin OR org admin for this team. "
            f"route='/management/v1/teams/{team_id}/members/bulk_delete'"
        )

    duplicates: Final = _duplicate_member_indexes(data.members)
    kept_indexes: Final = tuple(i for i in range(len(data.members)) if i not in duplicates)
    members: Final = tuple(data.members[i] for i in kept_indexes)
    async with prisma_client.tx(timeout=_BATCH_TX_TIMEOUT) as tx:
        removal: Final = await _remove_members_from_team(prisma_client, tx, team_id, members, user_api_key_dict)
    if removal.deleted_keys:
        KeyManagementEventHooks.create_key_deleted_audit_logs(
            keys_being_deleted=removal.deleted_keys,
            user_api_key_dict=user_api_key_dict,
            litellm_changed_by=None,
        )
    await delete_cache_key_objects(
        hashed_tokens=removal.deleted_key_tokens,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )
    await evict_and_broadcast(cache_keys=removal.jwt_mapping_cache_keys, user_api_key_cache=user_api_key_cache)
    _emit_team_members_metric(removal.team)

    matched: Final = frozenset(kept_indexes[j] for j in removal.matched)

    def error(index: int) -> str | None:
        if index in duplicates:
            return "Duplicate member in request"
        return None if index in matched else "User not found in team"

    return tuple(
        TeamMemberDeleteResult(
            user_id=member.user_id,
            user_email=member.user_email,
            success=i in matched,
            error=error(i),
        )
        for i, member in enumerate(data.members)
    )


async def _caller_admin_org_ids(prisma_client: PrismaClient, user_api_key_dict: UserAPIKeyAuth) -> frozenset[str]:
    if user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value or not user_api_key_dict.user_id:
        return frozenset()
    where: Final[_OrgAdminFilter] = {
        "user_id": user_api_key_dict.user_id,
        "user_role": LitellmUserRoles.ORG_ADMIN.value,
    }
    memberships: Final = await OrganizationMembershipRepository(prisma_client).table.find_many(where=where)
    return frozenset(m.organization_id for m in memberships if m.organization_id)


def _scope_error(user_id: str, target_org_ids: frozenset[str], caller_admin_org_ids: frozenset[str]) -> str | None:
    if target_org_ids and target_org_ids <= caller_admin_org_ids:
        return None
    return (
        f"User {user_id} is not within your admin scope. "
        "Only PROXY_ADMIN may delete users outside your administered organizations."
    )


async def _delete_user_rows(
    prisma_client: PrismaClient,
    tx: "Prisma",
    user_ids: frozenset[str],
    user_api_key_dict: UserAPIKeyAuth,
    litellm_changed_by: str | None,
) -> _DeletedKeys:
    keys: Final = await _token_tx_db(tx).find_many(where=_in_filter("user_id", user_ids))
    jwt_mapping_cache_keys: Final = await get_jwt_key_mapping_cache_keys_for_tokens(
        hashed_tokens=tuple(k.token for k in keys),
        prisma_client=prisma_client,
    )
    if keys:
        await _persist_deleted_verification_tokens(
            keys=keys,  # pyright: ignore[reportArgumentType]  # generated row model carries the same columns as LiteLLM_VerificationToken
            prisma_client=prisma_client,
            user_api_key_dict=user_api_key_dict,
            litellm_changed_by=litellm_changed_by,
            tx=tx,
        )
        await _token_tx_db(tx).delete_many(where=_in_filter("user_id", user_ids))
    await _invitation_tx_db(tx).delete_many(
        where=_any_filter(
            _in_filter("user_id", user_ids),
            _in_filter("created_by", user_ids),
            _in_filter("updated_by", user_ids),
        )
    )
    await _org_membership_tx_db(tx).delete_many(where=_in_filter("user_id", user_ids))
    await _membership_tx_db(tx).delete_many(where=_in_filter("user_id", user_ids))
    await _user_tx_db(tx).delete_many(where=_in_filter("user_id", user_ids))
    return _DeletedKeys(keys=tuple(keys), jwt_mapping_cache_keys=jwt_mapping_cache_keys)


async def _delete_users_tx(
    prisma_client: PrismaClient,
    users: Sequence["prisma_models.LiteLLM_UserTable"],
    teams_of: Mapping[str, frozenset[str]],
    user_api_key_dict: UserAPIKeyAuth,
    litellm_changed_by: str | None,
) -> _UserBatchDeletion:
    """Rewrites every team the users belong to and deletes their rows in one transaction, so a
    failure anywhere rolls back the whole batch. Teams a user still names but which no longer exist
    are skipped; the user row goes away regardless."""
    async with prisma_client.tx(timeout=_BATCH_TX_TIMEOUT) as tx:
        team_rows: Final = await _team_tx_db(tx).find_many(
            where=_in_filter("team_id", frozenset(t for teams in teams_of.values() for t in teams))
        )
        team_ids: Final = tuple(sorted(t.team_id for t in team_rows))
        removals: Final = MappingProxyType(
            {
                tid: await _remove_members_from_team(
                    prisma_client,
                    tx,
                    tid,
                    tuple(
                        MemberDeleteRequest(user_id=u.user_id, user_email=u.user_email)
                        for u in users
                        if tid in teams_of[u.user_id]
                    ),
                    user_api_key_dict,
                )
                for tid in team_ids
            }
        )
        deleted_keys: Final = await _delete_user_rows(
            prisma_client, tx, frozenset(u.user_id for u in users), user_api_key_dict, litellm_changed_by
        )
    return _UserBatchDeletion(
        removals=removals,
        deleted_keys=deleted_keys.keys + tuple(k for r in removals.values() for k in r.deleted_keys),
        jwt_mapping_cache_keys=deleted_keys.jwt_mapping_cache_keys
        + tuple(k for r in removals.values() for k in r.jwt_mapping_cache_keys),
    )


async def _delete_users(
    prisma_client: PrismaClient,
    users: Sequence["prisma_models.LiteLLM_UserTable"],
    teams_of: Mapping[str, frozenset[str]],
    user_api_key_dict: UserAPIKeyAuth,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging | None,
    litellm_proxy_admin_name: str | None,
    litellm_changed_by: str | None,
) -> _UserBatchDeletion | str:
    """Returns the error message when the transaction rolled back, in which case no row was touched."""
    user_ids: Final = frozenset(u.user_id for u in users)
    try:
        deletion: Final = await _delete_users_tx(prisma_client, users, teams_of, user_api_key_dict, litellm_changed_by)
    except Exception as e:  # noqa: BLE001  # the rolled-back batch is reported per row, not as a request failure
        verbose_proxy_logger.error("users/bulk_delete: failed to delete users %s: %s", sorted(user_ids), e)
        return _error_message(e)
    if deletion.deleted_keys:
        KeyManagementEventHooks.create_key_deleted_audit_logs(
            keys_being_deleted=deletion.deleted_keys,
            user_api_key_dict=user_api_key_dict,
            litellm_changed_by=litellm_changed_by,
        )
    await delete_cache_key_objects(
        hashed_tokens=deletion.deleted_key_tokens,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )
    await evict_and_broadcast(cache_keys=deletion.jwt_mapping_cache_keys, user_api_key_cache=user_api_key_cache)
    await evict_and_broadcast(cache_keys=sorted(user_ids), user_api_key_cache=user_api_key_cache)
    for removal in deletion.removals.values():
        _emit_team_members_metric(removal.team)
    audit_outcomes: Final = await _bounded(
        UserManagementEventHooks.create_internal_user_audit_log(
            user_id=u.user_id,
            action="deleted",
            litellm_changed_by=litellm_changed_by,
            user_api_key_dict=user_api_key_dict,
            litellm_proxy_admin_name=litellm_proxy_admin_name,
            before_value=u.model_dump_json(exclude_none=True),
        )
        for u in users
    )
    for u, outcome in zip(users, audit_outcomes, strict=True):
        if isinstance(outcome, BaseException):
            verbose_proxy_logger.warning("Failed to create audit log for user %s: %s", u.user_id, outcome)
    return deletion


async def bulk_delete_users(
    data: BulkDeleteUserRequest,
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: PrismaClient,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging | None,
    litellm_proxy_admin_name: str | None,
    litellm_changed_by: str | None,
) -> tuple[UserDeleteResult, ...]:
    caller_is_proxy_admin: Final = user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
    caller_admin_org_ids: Final = await _caller_admin_org_ids(prisma_client, user_api_key_dict)
    if not caller_is_proxy_admin and not caller_admin_org_ids:
        raise _forbidden("Only PROXY_ADMIN or ORG_ADMIN users may delete users.")

    unique_ids: Final = frozenset(data.user_ids)
    rows: Final = await UserRepository(prisma_client).table.find_many(where=_in_filter("user_id", unique_ids))
    rows_by_id: Final = MappingProxyType({row.user_id: row for row in rows})
    target_memberships: Final = (
        ()
        if caller_is_proxy_admin
        else await OrganizationMembershipRepository(prisma_client).table.find_many(
            where=_in_filter("user_id", unique_ids)
        )
    )

    def precheck_error(user_id: str) -> str | None:
        if user_id not in rows_by_id:
            return f"User id={user_id} not found"
        if caller_is_proxy_admin:
            return None
        org_ids: Final = frozenset(
            m.organization_id for m in target_memberships if m.user_id == user_id and m.organization_id
        )
        return _scope_error(user_id, org_ids, caller_admin_org_ids)

    precheck_errors: Final = MappingProxyType({uid: precheck_error(uid) for uid in unique_ids})
    candidates: Final = tuple(rows_by_id[uid] for uid in sorted(unique_ids) if precheck_errors[uid] is None)
    candidate_ids: Final = frozenset(u.user_id for u in candidates)

    memberships: Final = await TeamMembershipRepository(prisma_client).table.find_many(
        where=_in_filter("user_id", candidate_ids)
    )
    teams_of: Final = MappingProxyType(
        {
            u.user_id: frozenset(u.teams) | frozenset(m.team_id for m in memberships if m.user_id == u.user_id)
            for u in candidates
        }
    )
    deletion: Final = (
        await _delete_users(
            prisma_client,
            candidates,
            teams_of,
            user_api_key_dict,
            user_api_key_cache,
            proxy_logging_obj,
            litellm_proxy_admin_name,
            litellm_changed_by,
        )
        if candidates
        else _UserBatchDeletion(removals=MappingProxyType({}), deleted_keys=(), jwt_mapping_cache_keys=())
    )

    def result(index: int, user_id: str) -> UserDeleteResult:
        if user_id in data.user_ids[:index]:
            return UserDeleteResult(user_id=user_id, success=False, error=f"Duplicate user_id in request: {user_id}")
        error: Final = precheck_errors[user_id]
        if error is not None:
            return UserDeleteResult(user_id=user_id, success=False, error=error)
        if isinstance(deletion, str):
            return UserDeleteResult(
                user_id=user_id,
                user_email=rows_by_id[user_id].user_email,
                success=False,
                error=f"Failed to delete user: {deletion}",
            )
        return UserDeleteResult(
            user_id=user_id,
            user_email=rows_by_id[user_id].user_email,
            success=True,
            teams_removed=tuple(tid for tid, r in deletion.removals.items() if user_id in r.removed),
        )

    return tuple(result(i, uid) for i, uid in enumerate(data.user_ids))
