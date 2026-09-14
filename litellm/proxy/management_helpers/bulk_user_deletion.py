"""Batched deletes behind `POST /user/bulk_delete` and `POST /team/bulk_member_delete`.

Each team a batch touches is rewritten exactly once, under the same advisory lock
`/team/member_delete` takes and from a roster re-read under that lock, so a concurrent
member_add on the team is never overwritten from a stale read.
"""

import asyncio
import json
from collections.abc import Awaitable, Iterable, Mapping, Sequence
from dataclasses import dataclass
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
from litellm.proxy.hooks.user_management_event_hooks import UserManagementEventHooks
from litellm.proxy.management_endpoints.common_utils import (
    _is_user_org_admin_for_team,  # pyright: ignore[reportPrivateUsage]  # same check /team/member_delete uses
    _is_user_team_admin,  # pyright: ignore[reportPrivateUsage]  # same check /team/member_delete uses
)
from litellm.proxy.management_endpoints.key_management_endpoints import (
    _persist_deleted_verification_tokens,  # pyright: ignore[reportPrivateUsage]  # same audit path /key/delete uses
)
from litellm.proxy.management_helpers.access_group_team_sync import TEAM_ADVISORY_LOCK_SQL
from litellm.proxy.utils import PrismaClient
from litellm.repositories.table_repositories import (
    InvitationLinkRepository,
    OrganizationMembershipRepository,
    TeamMembershipRepository,
)
from litellm.repositories.team_repository import TeamRepository
from litellm.repositories.user_repository import UserRepository
from litellm.repositories.verification_token_repository import VerificationTokenRepository
from litellm.types.proxy.management_endpoints.internal_user_endpoints import (
    BulkDeleteUserRequest,
    BulkDeleteUserResponse,
    UserDeleteResult,
)
from litellm.types.proxy.management_endpoints.team_endpoints import (
    BulkTeamMemberDeleteRequest,
    BulkTeamMemberDeleteResponse,
    TeamMemberDeleteResult,
)

if TYPE_CHECKING:
    from prisma import Prisma
    from prisma import models as prisma_models

    from litellm.repositories.prisma_protocols import TableActions

_TEAM_WRITE_CONCURRENCY: Final = 10


class _ErrorDetail(TypedDict):
    error: ReadOnly[str]


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


def _http_error(status_code: int, message: str) -> HTTPException:
    detail: Final[_ErrorDetail] = {"error": message}
    return HTTPException(status_code=status_code, detail=detail)


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


def _addresses_member(member: Member, request: MemberDeleteRequest) -> bool:
    return (request.user_id is not None and request.user_id == member.user_id) or (
        request.user_email is not None and request.user_email == member.user_email
    )


def _addresses_user(user: "prisma_models.LiteLLM_UserTable", request: MemberDeleteRequest) -> bool:
    return (request.user_id is not None and request.user_id == user.user_id) or (
        request.user_email is not None and request.user_email == user.user_email
    )


def _error_message(exc: BaseException) -> str:
    if isinstance(exc, HTTPException) and isinstance(exc.detail, dict):
        return str(exc.detail.get("error", exc.detail))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # HTTPException.detail is untyped
    if isinstance(exc, HTTPException):
        return str(exc.detail)  # pyright: ignore[reportUnknownArgumentType]  # HTTPException.detail is untyped
    return str(exc) or type(exc).__name__


async def _bounded(awaitables: Iterable[Awaitable[object]]) -> tuple[object | BaseException, ...]:
    semaphore: Final = asyncio.Semaphore(_TEAM_WRITE_CONCURRENCY)

    async def run(awaitable: Awaitable[object]) -> object:
        async with semaphore:
            return await awaitable

    return tuple(await asyncio.gather(*(run(a) for a in awaitables), return_exceptions=True))


async def _remove_members_from_team(
    prisma_client: PrismaClient,
    team_id: str,
    members: Sequence[MemberDeleteRequest],
    user_api_key_dict: UserAPIKeyAuth,
) -> _TeamRemoval:
    async with prisma_client.tx() as tx:
        await tx.query_raw(TEAM_ADVISORY_LOCK_SQL, team_id)
        roster: Final = await TeamRepository(prisma_client).get_members_with_roles_locked(tx, team_id)
        if roster is None:
            raise _http_error(400, f"Team id={team_id} does not exist in db")

        removed_members: Final = tuple(m for m in roster if any(_addresses_member(m, r) for r in members))
        kept_members: Final = tuple(m for m in roster if not any(_addresses_member(m, r) for r in members))
        removed_ids: Final = frozenset(m.user_id for m in removed_members if m.user_id is not None)
        requested_ids: Final = frozenset(r.user_id for r in members if r.user_id is not None)
        requested_emails: Final = frozenset(r.user_email for r in members if r.user_id is None and r.user_email)
        user_rows: Final = await _user_tx_db(tx).find_many(
            where=_any_filter(
                _in_filter("user_id", removed_ids | requested_ids),
                _in_filter("user_email", requested_emails),
            )
        )
        stale_rows: Final = tuple(u for u in user_rows if team_id in u.teams)
        cleanup_ids: Final = removed_ids | requested_ids | frozenset(u.user_id for u in stale_rows)
        matched: Final = frozenset(
            i
            for i, r in enumerate(members)
            if any(_addresses_member(m, r) for m in removed_members) or any(_addresses_user(u, r) for u in stale_rows)
        )
        keys: Final = await _token_tx_db(tx).find_many(where=_team_users_filter(team_id, cleanup_ids))

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
        removed=removed_ids | frozenset(u.user_id for u in stale_rows),
        matched=matched,
    )


def _emit_team_members_metric(team: LiteLLM_TeamTable) -> None:
    prometheus_logger: Final = PrometheusLogger.get_instance()
    if prometheus_logger is None:
        return
    try:
        prometheus_logger.set_team_members_metric(team)
    except Exception as e:
        verbose_proxy_logger.debug("Prometheus: failed to emit team members metric: %s", str(e))


async def bulk_remove_team_members(
    data: BulkTeamMemberDeleteRequest,
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: PrismaClient,
) -> BulkTeamMemberDeleteResponse:
    team: Final = await TeamRepository(prisma_client).find_by_id(data.team_id)
    if team is None:
        raise _http_error(400, f"Team id={data.team_id} does not exist in db")

    if (
        user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN.value
        and not _is_user_team_admin(user_api_key_dict=user_api_key_dict, team_obj=team)
        and not await _is_user_org_admin_for_team(user_api_key_dict=user_api_key_dict, team_obj=team)
    ):
        raise _http_error(
            403,
            "Call not allowed. User not proxy admin OR team admin OR org admin for this team. "
            f"route='/team/bulk_member_delete', team_id={data.team_id}",
        )

    removal: Final = await _remove_members_from_team(prisma_client, data.team_id, data.members, user_api_key_dict)
    _emit_team_members_metric(removal.team)

    results: Final = tuple(
        TeamMemberDeleteResult(
            user_id=member.user_id,
            user_email=member.user_email,
            success=i in removal.matched,
            error=None if i in removal.matched else "User not found in team",
        )
        for i, member in enumerate(data.members)
    )
    successful: Final = sum(1 for r in results if r.success)
    return BulkTeamMemberDeleteResponse(
        team_id=data.team_id,
        results=results,
        total_requested=len(results),
        successful_deletions=successful,
        failed_deletions=len(results) - successful,
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
    users: Sequence["prisma_models.LiteLLM_UserTable"],
    user_api_key_dict: UserAPIKeyAuth,
    litellm_proxy_admin_name: str | None,
    litellm_changed_by: str | None,
) -> None:
    user_ids: Final = frozenset(u.user_id for u in users)
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
    keys: Final = await VerificationTokenRepository(prisma_client).table.find_many(
        where=_in_filter("user_id", user_ids)
    )
    if keys:
        await _persist_deleted_verification_tokens(
            keys=keys,  # pyright: ignore[reportArgumentType]  # generated row model carries the same columns as LiteLLM_VerificationToken
            prisma_client=prisma_client,
            user_api_key_dict=user_api_key_dict,
            litellm_changed_by=litellm_changed_by,
        )
    await VerificationTokenRepository(prisma_client).table.delete_many(where=_in_filter("user_id", user_ids))
    await InvitationLinkRepository(prisma_client).table.delete_many(
        where=_any_filter(
            _in_filter("user_id", user_ids),
            _in_filter("created_by", user_ids),
            _in_filter("updated_by", user_ids),
        )
    )
    await OrganizationMembershipRepository(prisma_client).table.delete_many(where=_in_filter("user_id", user_ids))
    await TeamMembershipRepository(prisma_client).table.delete_many(where=_in_filter("user_id", user_ids))
    await UserRepository(prisma_client).table.delete_many(where=_in_filter("user_id", user_ids))


async def bulk_delete_users(
    data: BulkDeleteUserRequest,
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: PrismaClient,
    litellm_proxy_admin_name: str | None,
    litellm_changed_by: str | None,
) -> BulkDeleteUserResponse:
    caller_is_proxy_admin: Final = user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
    caller_admin_org_ids: Final = await _caller_admin_org_ids(prisma_client, user_api_key_dict)
    if not caller_is_proxy_admin and not caller_admin_org_ids:
        raise _http_error(403, "Only PROXY_ADMIN or ORG_ADMIN users may delete users.")

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
    team_ids: Final = tuple(sorted(frozenset(t for u in candidates for t in teams_of[u.user_id])))
    members_by_team: Final = MappingProxyType(
        {
            tid: tuple(
                MemberDeleteRequest(user_id=u.user_id, user_email=u.user_email)
                for u in candidates
                if tid in teams_of[u.user_id]
            )
            for tid in team_ids
        }
    )
    outcomes: Final = await _bounded(
        _remove_members_from_team(prisma_client, tid, members_by_team[tid], user_api_key_dict) for tid in team_ids
    )
    removals: Final = MappingProxyType(
        {tid: o for tid, o in zip(team_ids, outcomes, strict=True) if isinstance(o, _TeamRemoval)}
    )
    team_failures: Final = MappingProxyType(
        {tid: _error_message(o) for tid, o in zip(team_ids, outcomes, strict=True) if isinstance(o, BaseException)}
    )
    for tid, err in team_failures.items():
        verbose_proxy_logger.error("/user/bulk_delete: failed to remove users from team %s: %s", tid, err)
    for removal in removals.values():
        _emit_team_members_metric(removal.team)

    def team_errors(user_id: str) -> tuple[str, ...]:
        return tuple(
            f"Failed to remove from team {tid}: {err}" for tid, err in team_failures.items() if tid in teams_of[user_id]
        )

    deletable: Final = tuple(u for u in candidates if not team_errors(u.user_id))
    if deletable:
        await _delete_user_rows(
            prisma_client, deletable, user_api_key_dict, litellm_proxy_admin_name, litellm_changed_by
        )

    def result(index: int, user_id: str) -> UserDeleteResult:
        if user_id in data.user_ids[:index]:
            return UserDeleteResult(user_id=user_id, success=False, error=f"Duplicate user_id in request: {user_id}")
        error: Final = precheck_errors[user_id]
        if error is not None:
            return UserDeleteResult(user_id=user_id, success=False, error=error)
        errors: Final = team_errors(user_id)
        return UserDeleteResult(
            user_id=user_id,
            user_email=rows_by_id[user_id].user_email,
            success=not errors,
            teams_removed=tuple(tid for tid in team_ids if tid in removals and user_id in removals[tid].removed),
            error="; ".join(errors) or None,
        )

    results: Final = tuple(result(i, uid) for i, uid in enumerate(data.user_ids))
    successful: Final = sum(1 for r in results if r.success)
    return BulkDeleteUserResponse(
        results=results,
        total_requested=len(results),
        successful_deletions=successful,
        failed_deletions=len(results) - successful,
    )
