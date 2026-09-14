"""Batched internal user creation behind `/user/bulk_new`.

The batch is validated with set queries, user rows land in one `create_many`, and every
referenced team is written once under its advisory lock instead of once per user.
"""

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, TypeVar

from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.integrations.prometheus import PrometheusLogger
from litellm.proxy._types import (
    LiteLLM_TeamTable,
    LitellmUserRoles,
    Member,
    NewUserRequestTeam,
    OrganizationMemberAddRequest,
    OrgMember,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import invalidate_team_member_spend_state
from litellm.proxy.auth.litellm_license import LicenseCheck
from litellm.proxy.common_utils.timezone_utils import get_budget_reset_time
from litellm.proxy.hooks.user_management_event_hooks import UserManagementEventHooks
from litellm.proxy.management_endpoints.common_utils import (
    _is_user_org_admin_for_team,  # pyright: ignore[reportPrivateUsage]  # same team-admin check /user/new uses
    _is_user_team_admin,  # pyright: ignore[reportPrivateUsage]  # same team-admin check /user/new uses
    validate_budget_duration,
)
from litellm.proxy.management_endpoints.internal_user_endpoints import (
    _update_internal_new_user_params,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # /user/new defaults; result validated below
    check_if_default_team_set,
)
from litellm.proxy.management_endpoints.key_management_endpoints import (
    _check_permissions_caller_permission,  # pyright: ignore[reportPrivateUsage]  # same permission check /user/new uses
    generate_key_helper_fn,  # pyright: ignore[reportUnknownVariableType]  # legacy untyped helper; result validated by _KEY_RESPONSE
    metadata_json_with_limits,
)
from litellm.proxy.management_endpoints.organization_endpoints import organization_member_add
from litellm.proxy.management_helpers.access_group_team_sync import TEAM_ADVISORY_LOCK_SQL
from litellm.proxy.management_helpers.object_permission_utils import (
    _set_object_permission,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # shared with /user/new; result validated below
)
from litellm.proxy.management_helpers.utils import (
    _resolve_member_budget_id,  # pyright: ignore[reportPrivateUsage]  # shared with /team/member_add
)
from litellm.proxy.utils import PrismaClient
from litellm.repositories.prisma_protocols import TableActions
from litellm.repositories.team_repository import TeamRepository
from litellm.repositories.user_repository import UserRepository
from litellm.types.proxy.management_endpoints.internal_user_endpoints import (
    BulkNewUserItem,
    BulkNewUserResponse,
    UserCreateResult,
)

if TYPE_CHECKING:
    from prisma import Prisma
    from prisma import models as prisma_models

    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

BULK_NEW_USER_CONCURRENCY: Final = 10

TeamRole: TypeAlias = Literal["user", "admin"]
KeyGenerator: TypeAlias = Callable[..., Awaitable[object]]
_T: Final = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class _RowFailure:
    index: int
    user_id: str | None
    user_email: str | None
    error: str


@dataclass(frozen=True, slots=True)
class _PendingUser:
    index: int
    request: BulkNewUserItem
    user_id: str
    teams: tuple[NewUserRequestTeam, ...]


class _UserRow(BaseModel):
    """The `/user/new` body after defaults and object permission were applied."""

    model_config = ConfigDict(extra="ignore")

    user_id: str
    user_email: str | None = None
    user_alias: str | None = None
    user_role: str | None = None
    team_id: str | None = None
    max_budget: float | None = None
    spend: float | None = 0.0
    models: tuple[str, ...] | None = None
    metadata: Mapping[str, object] | None = None
    max_parallel_requests: int | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    budget_duration: str | None = None
    allowed_cache_controls: tuple[str, ...] | None = None
    sso_user_id: str | None = None
    object_permission_id: str | None = None
    model_max_budget: Mapping[str, object] | None = None
    model_rpm_limit: Mapping[str, object] | None = None
    model_tpm_limit: Mapping[str, object] | None = None
    mcp_rpm_limit: Mapping[str, int] | None = None
    tag_rpm_limit: Mapping[str, int] | None = None
    guardrails: tuple[str, ...] | None = None
    policies: tuple[str, ...] | None = None
    prompts: tuple[str, ...] | None = None
    duration: str | None = None
    key_alias: str | None = None
    aliases: Mapping[str, object] | None = None
    config: Mapping[str, object] | None = None
    permissions: Mapping[str, object] | None = None
    blocked: bool | None = None
    agent_id: str | None = None
    budget_fallbacks: Mapping[str, tuple[str, ...]] | None = None
    budget_limits: tuple[Mapping[str, object], ...] | None = None
    organizations: tuple[str, ...] | None = None


_USER_ROW: Final = TypeAdapter(_UserRow)


@dataclass(frozen=True, slots=True)
class _PreparedUser:
    pending: _PendingUser
    row: _UserRow


@dataclass(frozen=True, slots=True)
class _TeamAssignment:
    user_id: str
    user_email: str | None
    role: TeamRole
    max_budget_in_team: float | None


@dataclass(frozen=True, slots=True)
class _TeamWrite:
    """Outcome of one locked roster write. `failed` maps user ids to the reason they were not added."""

    team_id: str
    after: tuple[Member, ...]
    added: frozenset[str]
    failed: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _CreatedUser:
    prepared: _PreparedUser
    teams: tuple[str, ...]
    key: str | None
    errors: tuple[str, ...]


_ERROR_DETAIL: Final = TypeAdapter(Mapping[str, object])
_JSON_OBJECT: Final = TypeAdapter(dict[str, object])


class _KeyResponse(BaseModel):
    token: str


_KEY_RESPONSE: Final = TypeAdapter(_KeyResponse)


def _error_message(exc: BaseException) -> str:
    if not isinstance(exc, HTTPException):
        return str(exc)
    try:
        detail: Final = _ERROR_DETAIL.validate_python(exc.detail)
    except ValidationError:
        return str(exc.detail)
    return str(detail.get("error", detail))


def _requested_teams(item: BulkNewUserItem) -> tuple[NewUserRequestTeam, ...]:
    if item.team_id is not None:
        return (NewUserRequestTeam(team_id=item.team_id),)
    teams: Final = item.teams if item.teams is not None else check_if_default_team_set()
    if teams is None:
        return ()
    return tuple(team if isinstance(team, NewUserRequestTeam) else NewUserRequestTeam(team_id=team) for team in teams)


def _row_error(item: BulkNewUserItem, user_api_key_dict: UserAPIKeyAuth) -> str | None:
    if (
        item.user_role in (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)
        and user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN
    ):
        return (
            "Only proxy admins can create administrative users (proxy_admin, proxy_admin_viewer). "
            f"Attempted to create user with role: {item.user_role}. Your role: {user_api_key_dict.user_role}"
        )
    try:
        validate_budget_duration(item.budget_duration)
        _check_permissions_caller_permission(data=item, user_api_key_dict=user_api_key_dict)
    except Exception as exc:  # noqa: BLE001  # any validation failure is reported on this row only
        return _error_message(exc)
    return None


def _normalized_email(email: str | None) -> str | None:
    return email.strip().lower() if email else None


def _partition_rows(
    users: Sequence[BulkNewUserItem], user_api_key_dict: UserAPIKeyAuth
) -> tuple[tuple[_PendingUser, ...], tuple[_RowFailure, ...]]:
    """Assign ids, run the per-row checks and fail later rows that repeat an earlier row's id or email."""
    user_ids: Final = tuple(item.user_id or str(uuid.uuid4()) for item in users)
    first_index_by_id: Final = MappingProxyType(
        {user_id: index for index, user_id in reversed(tuple(enumerate(user_ids)))}
    )
    first_index_by_email: Final = MappingProxyType(
        {
            email: index
            for index, email in reversed(tuple(enumerate(_normalized_email(item.user_email) for item in users)))
            if email is not None
        }
    )

    def classify(index: int, item: BulkNewUserItem) -> _PendingUser | _RowFailure:
        user_id: Final = user_ids[index]
        email: Final = _normalized_email(item.user_email)
        if first_index_by_id[user_id] != index:
            return _RowFailure(index, user_id, item.user_email, f"Duplicate user_id in request: {user_id}")
        if email is not None and first_index_by_email[email] != index:
            return _RowFailure(index, user_id, item.user_email, f"Duplicate user_email in request: {item.user_email}")
        error: Final = _row_error(item, user_api_key_dict)
        if error is not None:
            return _RowFailure(index, user_id, item.user_email, error)
        return _PendingUser(index, item, user_id, _requested_teams(item))

    outcomes: Final = tuple(classify(index, item) for index, item in enumerate(users))
    return (
        tuple(outcome for outcome in outcomes if isinstance(outcome, _PendingUser)),
        tuple(outcome for outcome in outcomes if isinstance(outcome, _RowFailure)),
    )


def _user_table(prisma_client: PrismaClient) -> "TableActions[prisma_models.LiteLLM_UserTable]":
    return UserRepository(prisma_client).table


async def _existing_user_conflicts(
    prisma_client: PrismaClient, pending: Sequence[_PendingUser]
) -> tuple[frozenset[str], frozenset[str]]:
    """Return the requested user ids and (lowercased) emails that already exist, using one query each."""
    user_ids: Final = sorted(user.user_id for user in pending)
    emails: Final = sorted(frozenset(user.request.user_email for user in pending if user.request.user_email))
    if not user_ids:
        return frozenset(), frozenset()
    table: Final = _user_table(prisma_client)
    id_filter: Final = {"user_id": {"in": user_ids}}  # mutable-ok: Prisma query filters are dict-shaped
    email_filter: Final = {"user_email": {"in": emails, "mode": "insensitive"}}  # mutable-ok: Prisma filter
    id_rows: Final = await table.find_many(where=id_filter)
    email_rows: Final = await table.find_many(where=email_filter) if emails else ()
    return (
        frozenset(row.user_id for row in id_rows),
        frozenset(lowered for row in email_rows if (lowered := _normalized_email(row.user_email)) is not None),
    )


async def _load_teams(prisma_client: PrismaClient, team_ids: frozenset[str]) -> Mapping[str, LiteLLM_TeamTable]:
    if not team_ids:
        return MappingProxyType({})
    rows: Final = await TeamRepository(prisma_client).table.find_many(
        where={"team_id": {"in": sorted(team_ids)}}  # mutable-ok: Prisma query filters are dict-shaped
    )
    return MappingProxyType({row.team_id: LiteLLM_TeamTable.model_validate(row.model_dump()) for row in rows})


async def _team_permission_error(team: LiteLLM_TeamTable, user_api_key_dict: UserAPIKeyAuth) -> str | None:
    if user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value:
        return None
    if _is_user_team_admin(user_api_key_dict=user_api_key_dict, team_obj=team):
        return None
    if await _is_user_org_admin_for_team(user_api_key_dict=user_api_key_dict, team_obj=team):
        return None
    return f"Call not allowed. User not proxy admin OR team admin. team_id={team.team_id}"


async def _unusable_teams(
    prisma_client: PrismaClient,
    pending: Sequence[_PendingUser],
    user_api_key_dict: UserAPIKeyAuth,
) -> tuple[Mapping[str, LiteLLM_TeamTable], Mapping[str, str]]:
    """Load every referenced team once and explain, per team id, why rows naming it cannot proceed."""
    team_ids: Final = frozenset(team.team_id for user in pending for team in user.teams)
    teams: Final = await _load_teams(prisma_client, team_ids)
    permission_errors: Final = await asyncio.gather(
        *(_team_permission_error(team, user_api_key_dict) for team in teams.values())
    )
    missing: Final = tuple(
        (team_id, f"Team id={team_id} does not exist") for team_id in team_ids if team_id not in teams
    )
    denied: Final = tuple(
        (team.team_id, error)
        for team, error in zip(teams.values(), permission_errors, strict=True)
        if error is not None
    )
    return teams, MappingProxyType({team_id: error for team_id, error in (*missing, *denied)})


def _db_failure(
    user: _PendingUser,
    existing_ids: frozenset[str],
    existing_emails: frozenset[str],
    team_errors: Mapping[str, str],
) -> _RowFailure | None:
    email: Final = _normalized_email(user.request.user_email)
    if user.user_id in existing_ids:
        return _RowFailure(user.index, user.user_id, user.request.user_email, f"User id={user.user_id} already exists")
    if email is not None and email in existing_emails:
        return _RowFailure(
            user.index, user.user_id, user.request.user_email, f"User email={user.request.user_email} already exists"
        )
    errors: Final = tuple(team_errors[team.team_id] for team in user.teams if team.team_id in team_errors)
    if errors:
        return _RowFailure(user.index, user.user_id, user.request.user_email, "; ".join(errors))
    return None


async def _prepare_user(user: _PendingUser, prisma_client: PrismaClient) -> _PreparedUser | _RowFailure:
    try:
        dumped: Final = user.request.model_dump(exclude={"user_id"})  # mutable-ok: pydantic IncEx takes a set
        data: Final = {**dumped, "user_id": user.user_id}  # mutable-ok: /user/new defaults helper mutates in place
        data_json: Final = _JSON_OBJECT.validate_python(_update_internal_new_user_params(data, user.request))
        with_permission: Final = _JSON_OBJECT.validate_python(
            await _set_object_permission(data_json=data_json, prisma_client=prisma_client)  # pyright: ignore[reportUnknownArgumentType]  # validated by the adapter
        )
        return _PreparedUser(user, _USER_ROW.validate_python(with_permission))
    except Exception as exc:  # noqa: BLE001  # any preparation failure is reported on this row only
        verbose_proxy_logger.warning("/user/bulk_new: could not prepare row %d - %s", user.index, type(exc).__name__)
        return _RowFailure(user.index, user.user_id, user.request.user_email, _error_message(exc))


class _UserCreateData(TypedDict):
    """One `LiteLLM_UserTable` row as `create_many` takes it; JSON columns are pre-serialized."""

    user_id: ReadOnly[str]
    user_email: ReadOnly[str | None]
    user_alias: ReadOnly[str | None]
    user_role: ReadOnly[str | None]
    team_id: ReadOnly[str | None]
    max_budget: ReadOnly[float | None]
    spend: ReadOnly[float]
    models: ReadOnly[tuple[str, ...]]
    metadata: ReadOnly[str]
    max_parallel_requests: ReadOnly[int | None]
    tpm_limit: ReadOnly[int | None]
    rpm_limit: ReadOnly[int | None]
    budget_duration: ReadOnly[str | None]
    budget_reset_at: ReadOnly[datetime | None]
    allowed_cache_controls: ReadOnly[tuple[str, ...]]
    sso_user_id: ReadOnly[str | None]
    object_permission_id: ReadOnly[str | None]
    teams: ReadOnly[tuple[str, ...]]
    model_max_budget: ReadOnly[str]


def _user_create_payload(prepared: _PreparedUser) -> _UserCreateData:
    row: Final = prepared.row
    metadata_json: Final = metadata_json_with_limits(
        row.metadata,
        model_rpm_limit=row.model_rpm_limit,
        model_tpm_limit=row.model_tpm_limit,
        mcp_rpm_limit=row.mcp_rpm_limit,
        tag_rpm_limit=row.tag_rpm_limit,
        guardrails=row.guardrails,
        policies=row.policies,
        prompts=row.prompts,
    )
    payload: Final[_UserCreateData] = {
        "user_id": row.user_id,
        "user_email": row.user_email,
        "user_alias": row.user_alias,
        "user_role": row.user_role,
        "team_id": row.team_id,
        "max_budget": row.max_budget,
        "spend": row.spend or 0.0,
        "models": row.models or (),
        "metadata": metadata_json,
        "max_parallel_requests": row.max_parallel_requests,
        "tpm_limit": row.tpm_limit,
        "rpm_limit": row.rpm_limit,
        "budget_duration": row.budget_duration,
        "budget_reset_at": get_budget_reset_time(row.budget_duration) if row.budget_duration else None,
        "allowed_cache_controls": row.allowed_cache_controls or (),
        "sso_user_id": row.sso_user_id,
        "object_permission_id": row.object_permission_id,
        "teams": tuple(team.team_id for team in prepared.pending.teams),
        "model_max_budget": json.dumps(row.model_max_budget) if row.model_max_budget else "{}",
    }
    return payload


async def _bounded(limit: int, awaitables: Sequence[Awaitable[_T]]) -> tuple[_T | BaseException, ...]:
    semaphore: Final = asyncio.Semaphore(limit)

    async def run(awaitable: Awaitable[_T]) -> _T:
        async with semaphore:
            return await awaitable

    return tuple(await asyncio.gather(*(run(awaitable) for awaitable in awaitables), return_exceptions=True))


async def _insert_users(
    prisma_client: PrismaClient, prepared: Sequence[_PreparedUser]
) -> tuple[tuple[_PreparedUser, ...], tuple[_RowFailure, ...]]:
    """Insert every row in one statement. If that fails, retry rows one at a time so the error lands on its row."""
    if not prepared:
        return (), ()
    table: Final = _user_table(prisma_client)
    payloads: Final = tuple(_user_create_payload(user) for user in prepared)
    try:
        await table.create_many(data=payloads)
        return tuple(prepared), ()
    except Exception as exc:  # noqa: BLE001  # fall back to per-row inserts so the failing row can be identified
        verbose_proxy_logger.warning("/user/bulk_new: create_many failed, retrying rows individually - %s", exc)
    landed_rows: Final = await table.find_many(
        where={"user_id": {"in": [payload["user_id"] for payload in payloads]}}  # mutable-ok: Prisma filter
    )
    landed: Final = frozenset(row.user_id for row in landed_rows)
    retried: Final = tuple(user for user in prepared if user.row.user_id not in landed)
    outcomes: Final = await _bounded(
        BULK_NEW_USER_CONCURRENCY, tuple(table.create(data=_user_create_payload(user)) for user in retried)
    )
    failed: Final = MappingProxyType(
        {
            user.row.user_id: _RowFailure(
                user.pending.index, user.pending.user_id, user.row.user_email, _error_message(outcome)
            )
            for user, outcome in zip(retried, outcomes, strict=True)
            if isinstance(outcome, BaseException)
        }
    )
    return (
        tuple(user for user in prepared if user.row.user_id not in failed),
        tuple(failed.values()),
    )


def _assignments_by_team(created: Sequence[_PreparedUser]) -> Mapping[str, tuple[_TeamAssignment, ...]]:
    team_ids: Final = tuple(dict.fromkeys(team.team_id for user in created for team in user.pending.teams))
    return MappingProxyType(
        {
            team_id: tuple(
                _TeamAssignment(user.pending.user_id, user.row.user_email, team.user_role, team.max_budget_in_team)
                for user in created
                for team in user.pending.teams
                if team.team_id == team_id
            )
            for team_id in team_ids
        }
    )


class _MembershipData(TypedDict):
    team_id: ReadOnly[str]
    user_id: ReadOnly[str]
    budget_id: ReadOnly[str | None]


class _RosterData(TypedDict):
    members_with_roles: ReadOnly[str]


class _TeamsData(TypedDict):
    teams: ReadOnly[tuple[str, ...]]


def _default_member_budget_id(team: LiteLLM_TeamTable) -> str | None:
    metadata: Final = (
        _JSON_OBJECT.validate_python(
            team.metadata  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # LiteLLM_TeamTable.metadata is a bare dict; validated by the adapter
        )
        if team.metadata  # pyright: ignore[reportUnknownMemberType]  # same bare dict
        else None
    )
    budget_id: Final = metadata.get("team_member_budget_id") if metadata is not None else None
    return budget_id if isinstance(budget_id, str) else None


def _team_tx_db(tx: "Prisma") -> "TableActions[prisma_models.LiteLLM_TeamTable]":
    return tx.litellm_teamtable  # pyright: ignore[reportReturnType]  # TableActions widens the generated inputs to Mapping, as the repositories do


def _membership_tx_db(tx: "Prisma") -> "TableActions[prisma_models.LiteLLM_TeamMembership]":
    return tx.litellm_teammembership  # pyright: ignore[reportReturnType]  # TableActions widens the generated inputs to Mapping, as the repositories do


async def _write_team_roster(
    prisma_client: PrismaClient,
    team: LiteLLM_TeamTable,
    members: Sequence[_TeamAssignment],
    user_api_key_dict: UserAPIKeyAuth,
    litellm_proxy_admin_name: str,
) -> _TeamWrite:
    """Add every new member to one team under its advisory lock: one roster rewrite and one membership insert."""
    try:
        async with prisma_client.tx() as tx:
            await tx.query_raw(TEAM_ADVISORY_LOCK_SQL, team.team_id)
            roster: Final = await TeamRepository(prisma_client).get_members_with_roles_locked(tx, team.team_id)
            if roster is None:
                raise ValueError(f"Team id={team.team_id} does not exist")
            already_present: Final = frozenset(member.user_id for member in roster if member.user_id)
            new_members: Final = tuple(member for member in members if member.user_id not in already_present)
            budget_ids: Final = tuple(
                [  # mutable-ok: budgets are created one at a time on the transaction's single connection
                    await _resolve_member_budget_id(
                        prisma_client=prisma_client,
                        user_api_key_dict=user_api_key_dict,
                        litellm_proxy_admin_name=litellm_proxy_admin_name,
                        max_budget_in_team=member.max_budget_in_team,
                        allowed_models=team.default_team_member_models or None,
                        budget_duration=None,
                        default_team_budget_id=_default_member_budget_id(team),
                        tx=tx,  # pyright: ignore[reportArgumentType]  # MemberWriteTx lags the generated Prisma signatures, same as /team/member_add
                    )
                    for member in new_members
                ]
            )
            await _membership_tx_db(tx).create_many(
                data=tuple(
                    _MembershipData(team_id=team.team_id, user_id=member.user_id, budget_id=budget_id)
                    for member, budget_id in zip(new_members, budget_ids, strict=True)
                ),
                skip_duplicates=True,
            )
            after: Final = (
                *roster,
                *(Member(user_id=m.user_id, user_email=m.user_email, role=m.role) for m in new_members),
            )
            await _team_tx_db(tx).update(
                where={"team_id": team.team_id},  # mutable-ok: Prisma query filters are dict-shaped
                data=_RosterData(members_with_roles=json.dumps(tuple(member.model_dump() for member in after))),
            )
        return _TeamWrite(
            team_id=team.team_id,
            after=after,
            added=frozenset(member.user_id for member in new_members),
            failed=MappingProxyType({}),
        )
    except Exception as exc:  # noqa: BLE001  # the team write failure is reported on each affected row
        verbose_proxy_logger.exception("/user/bulk_new: failed to add %d members to a team", len(members))
        message: Final = f"Failed to add user to team {team.team_id}: {_error_message(exc)}"
        return _TeamWrite(
            team_id=team.team_id,
            after=(),
            added=frozenset(),
            failed=MappingProxyType({member.user_id: message for member in members}),
        )


async def _detach_failed_teams(
    prisma_client: PrismaClient, created: Sequence[_PreparedUser], writes: Mapping[str, _TeamWrite]
) -> None:
    """Users are inserted with `teams` already set; drop the teams whose roster write did not take them."""
    table: Final = _user_table(prisma_client)
    updates: Final = tuple(
        table.update(
            where={"user_id": user.row.user_id},  # mutable-ok: Prisma query filters are dict-shaped
            data=_TeamsData(teams=landed),
        )
        for user in created
        if (landed := _row_teams(user, writes)[0]) != tuple(team.team_id for team in user.pending.teams)
    )
    for outcome in await _bounded(BULK_NEW_USER_CONCURRENCY, updates):
        if isinstance(outcome, BaseException):
            verbose_proxy_logger.warning("/user/bulk_new: could not detach failed teams from user - %s", outcome)


async def _publish_team_writes(writes: Sequence[_TeamWrite], user_api_key_cache: "UserApiKeyCache") -> None:
    prometheus_logger: Final = PrometheusLogger.get_instance()
    for write in writes:
        if prometheus_logger is None or not write.added:
            continue
        try:
            prometheus_logger.set_team_members_metric(
                LiteLLM_TeamTable(
                    team_id=write.team_id,
                    members_with_roles=write.after,  # pyright: ignore[reportArgumentType]  # pydantic coerces the tuple into the declared list
                )
            )
        except Exception as exc:  # noqa: BLE001  # metrics are best-effort and must not fail the request
            verbose_proxy_logger.debug("Prometheus: failed to emit team members metric: %s", exc)
    evictions: Final = await _bounded(
        BULK_NEW_USER_CONCURRENCY,
        tuple(
            invalidate_team_member_spend_state(
                user_id=user_id, team_id=write.team_id, user_api_key_cache=user_api_key_cache
            )
            for write in writes
            for user_id in write.added
        ),
    )
    for eviction in evictions:
        if isinstance(eviction, BaseException):
            verbose_proxy_logger.warning("/user/bulk_new: cache eviction failed - %s", eviction)


_KEY_FIELDS: Final = MappingProxyType(
    {
        name: True
        for name in (
            "user_id",
            "team_id",
            "agent_id",
            "duration",
            "key_alias",
            "models",
            "aliases",
            "config",
            "permissions",
            "blocked",
            "spend",
            "budget_fallbacks",
            "budget_limits",
            "metadata",
            "max_parallel_requests",
            "tpm_limit",
            "rpm_limit",
            "allowed_cache_controls",
            "model_max_budget",
            "model_rpm_limit",
            "model_tpm_limit",
            "mcp_rpm_limit",
            "tag_rpm_limit",
            "guardrails",
            "policies",
            "prompts",
            "object_permission_id",
        )
    }
)


async def _generate_key(prepared: _PreparedUser, generate_key: KeyGenerator) -> str:
    response: Final = _KEY_RESPONSE.validate_python(
        await generate_key(
            request_type="key", table_name="key", **prepared.row.model_dump(include=_KEY_FIELDS, exclude_none=True)
        )
    )
    return response.token


async def _add_to_organizations(
    prepared: _PreparedUser, organizations: Sequence[str], user_api_key_dict: UserAPIKeyAuth
) -> None:
    for organization_id in organizations:
        await organization_member_add(
            data=OrganizationMemberAddRequest(
                organization_id=organization_id,
                member=OrgMember(user_id=prepared.row.user_id, role=LitellmUserRoles.INTERNAL_USER),
            ),
            http_request=Request(scope={"type": "http", "path": "/user/bulk_new"}),  # mutable-ok: ASGI scopes are dicts
            user_api_key_dict=user_api_key_dict,
        )


async def _run_per_user(
    created: Sequence[_PreparedUser],
    select: Callable[[_PreparedUser], bool],
    action: Callable[[_PreparedUser], Awaitable[_T]],
) -> Mapping[str, _T | BaseException]:
    chosen: Final = tuple(user for user in created if select(user))
    outcomes: Final = await _bounded(BULK_NEW_USER_CONCURRENCY, tuple(action(user) for user in chosen))
    return MappingProxyType({user.row.user_id: outcome for user, outcome in zip(chosen, outcomes, strict=True)})


async def _write_audit_logs(
    prisma_client: PrismaClient,
    created: Sequence[_PreparedUser],
    user_api_key_dict: UserAPIKeyAuth,
    litellm_proxy_admin_name: str,
) -> None:
    if not created:
        return
    created_ids: Final = sorted(user.row.user_id for user in created)
    created_filter: Final = {"user_id": {"in": created_ids}}  # mutable-ok: Prisma query filters are dict-shaped
    rows: Final = await _user_table(prisma_client).find_many(where=created_filter)
    outcomes: Final = await _bounded(
        BULK_NEW_USER_CONCURRENCY,
        tuple(
            UserManagementEventHooks.create_internal_user_audit_log(
                user_id=row.user_id,
                action="created",
                litellm_changed_by=user_api_key_dict.user_id,
                user_api_key_dict=user_api_key_dict,
                litellm_proxy_admin_name=litellm_proxy_admin_name,
                before_value=None,
                after_value=row.model_dump_json(exclude_none=True),
            )
            for row in rows
        ),
    )
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            verbose_proxy_logger.warning("Unable to create audit log for user on `/user/bulk_new` - %s", outcome)


def _row_teams(prepared: _PreparedUser, writes: Mapping[str, _TeamWrite]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split a user's requested teams into the ones they landed in and the errors for the ones they did not."""
    requested: Final = tuple(team.team_id for team in prepared.pending.teams)
    return (
        tuple(team_id for team_id in requested if prepared.row.user_id in writes[team_id].added),
        tuple(
            writes[team_id].failed[prepared.row.user_id]
            for team_id in requested
            if prepared.row.user_id in writes[team_id].failed
        ),
    )


def _to_result(created: _CreatedUser) -> UserCreateResult:
    return UserCreateResult(
        user_id=created.prepared.row.user_id,
        user_email=created.prepared.row.user_email,
        success=True,
        teams=created.teams,
        key=created.key,
        error="; ".join(created.errors) if created.errors else None,
    )


def _failure_result(failure: _RowFailure) -> UserCreateResult:
    return UserCreateResult(user_id=failure.user_id, user_email=failure.user_email, success=False, error=failure.error)


async def bulk_create_users(
    users: Sequence[BulkNewUserItem],
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: PrismaClient,
    license_check: LicenseCheck,
    litellm_proxy_admin_name: str,
    user_api_key_cache: "UserApiKeyCache",
    generate_key: KeyGenerator = generate_key_helper_fn,
) -> BulkNewUserResponse:
    """Create every valid row in `users`; rows that fail validation or a write are reported, not raised.

    Raises `HTTPException(403)` only when the whole batch would push the deployment over its license seat limit.
    """
    pending, request_failures = _partition_rows(users, user_api_key_dict)
    existing_ids, existing_emails = await _existing_user_conflicts(prisma_client, pending)
    teams, team_errors = await _unusable_teams(prisma_client, pending, user_api_key_dict)
    db_failures: Final = tuple(
        failure
        for user in pending
        if (failure := _db_failure(user, existing_ids, existing_emails, team_errors)) is not None
    )
    failed_indexes: Final = frozenset(failure.index for failure in db_failures)
    creatable: Final = tuple(user for user in pending if user.index not in failed_indexes)

    billable_users: Final = await UserRepository(prisma_client).count_billable_users()
    if creatable and license_check.is_over_limit(total_users=billable_users + len(creatable)):
        raise HTTPException(
            status_code=403,
            detail="License is over limit. Please contact support@berri.ai to upgrade your license.",
        )

    prepared_outcomes: Final = tuple([await _prepare_user(user, prisma_client) for user in creatable])
    prepare_failures: Final = tuple(o for o in prepared_outcomes if isinstance(o, _RowFailure))
    created, insert_failures = await _insert_users(
        prisma_client, tuple(o for o in prepared_outcomes if isinstance(o, _PreparedUser))
    )

    team_writes: Final = MappingProxyType(
        {
            team_id: await _write_team_roster(
                prisma_client, teams[team_id], members, user_api_key_dict, litellm_proxy_admin_name
            )
            for team_id, members in _assignments_by_team(created).items()
        }
    )
    await _detach_failed_teams(prisma_client, created, team_writes)
    await _publish_team_writes(tuple(team_writes.values()), user_api_key_cache)

    keys: Final = await _run_per_user(
        created, lambda user: user.pending.request.auto_create_key, lambda user: _generate_key(user, generate_key)
    )
    org_outcomes: Final = await _run_per_user(
        created,
        lambda user: bool(user.row.organizations),
        lambda user: _add_to_organizations(user, user.row.organizations or (), user_api_key_dict),
    )
    await _write_audit_logs(prisma_client, created, user_api_key_dict, litellm_proxy_admin_name)

    def finish(prepared: _PreparedUser) -> _CreatedUser:
        landed, team_failures = _row_teams(prepared, team_writes)
        key_outcome: Final = keys.get(prepared.row.user_id)
        org_outcome: Final = org_outcomes.get(prepared.row.user_id)
        return _CreatedUser(
            prepared=prepared,
            teams=landed,
            key=key_outcome if isinstance(key_outcome, str) else None,
            errors=(
                *team_failures,
                *(
                    (f"Failed to create key: {_error_message(key_outcome)}",)
                    if isinstance(key_outcome, BaseException)
                    else ()
                ),
                *(
                    (f"Failed to add user to organizations: {_error_message(org_outcome)}",)
                    if isinstance(org_outcome, BaseException)
                    else ()
                ),
            ),
        )

    failures: Final = MappingProxyType(
        {
            failure.index: _failure_result(failure)
            for failure in (*request_failures, *db_failures, *prepare_failures, *insert_failures)
        }
    )
    successes_by_index: Final = MappingProxyType({user.pending.index: _to_result(finish(user)) for user in created})
    results: Final = tuple(
        failures[index] if index in failures else successes_by_index[index] for index in range(len(users))
    )
    successes: Final = sum(1 for result in results if result.success)
    return BulkNewUserResponse(
        results=results,
        total_requested=len(users),
        successful_creations=successes,
        failed_creations=len(users) - successes,
    )
