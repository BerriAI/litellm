from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias

from fastapi import HTTPException

from litellm.proxy._types import KeyManagementRoutes, LiteLLM_TeamTable, UserAPIKeyAuth


@dataclass(frozen=True, slots=True)
class AllLogs:
    """Unrestricted reads, granted by the consuming endpoint's role checks."""


@dataclass(frozen=True, slots=True)
class UserLogs:
    user_id: str | None


@dataclass(frozen=True, slots=True)
class TeamLogs:
    team_id: str


LogGrant: TypeAlias = UserLogs | TeamLogs


@dataclass(frozen=True, slots=True)
class AnyOf:
    grants: tuple[LogGrant, ...]


LogReadScope: TypeAlias = AllLogs | LogGrant | AnyOf


def any_of(*grants: LogGrant) -> AnyOf:
    return AnyOf(grants)


async def resolve_log_read_scope(
    user_id: str | None,
    permitted_team_lookup: Callable[[], Awaitable[Sequence[str]]],
) -> UserLogs | AnyOf:
    """Resolve own-user and permitted-team reads, falling back to own-user on lookup failure."""
    try:
        team_ids: Final = tuple(await permitted_team_lookup())
    except Exception:  # noqa: BLE001  # preserve spend-log own-user fallback for every permission lookup failure
        return UserLogs(user_id)
    return any_of(UserLogs(user_id), *(TeamLogs(team_id) for team_id in team_ids)) if team_ids else UserLogs(user_id)


def can_read_team_logs(auth: UserAPIKeyAuth, team: LiteLLM_TeamTable) -> bool:
    from litellm.proxy.management.teams.access import is_team_admin
    from litellm.proxy.management_endpoints.common_utils import (
        _team_member_has_permission,  # pyright: ignore[reportPrivateUsage]  # reuse existing team permission policy
    )

    return is_team_admin(user_api_key_dict=auth, team_obj=team) or _team_member_has_permission(
        user_api_key_dict=auth,
        team_obj=team,
        permission=KeyManagementRoutes.SPEND_LOGS.value,
    )


def permitted_log_team_ids(auth: UserAPIKeyAuth, teams: Iterable[LiteLLM_TeamTable]) -> tuple[str, ...]:
    return tuple(team.team_id for team in teams if can_read_team_logs(auth, team))


async def can_read_log_owner(
    user_id: str | None,
    owner_user: str | None,
    owner_team_id: str | None,
    team_permission_lookup: Callable[[str], Awaitable[bool]],
) -> bool:
    """Authorize stored ownership without swallowing direct team-lookup failures."""
    if owner_user is not None and owner_user == user_id:
        return True
    if owner_team_id:
        return await team_permission_lookup(owner_team_id)
    return False


@dataclass(frozen=True, slots=True)
class LogVisibility:
    all_teams: bool = False
    user_id: str = ""
    team_ids: tuple[str, ...] = ()
    api_key_hash: str = ""


async def permitted_log_teams(auth: UserAPIKeyAuth) -> tuple[str, ...]:
    from litellm.proxy.proxy_server import prisma_client
    from litellm.proxy.spend_tracking.spend_management_endpoints import (
        _get_permitted_team_ids_for_spend_logs_or_empty,  # pyright: ignore[reportPrivateUsage]  # Reuse request-log policy
    )

    if prisma_client is None:
        return ()
    return await _get_permitted_team_ids_for_spend_logs_or_empty(prisma_client=prisma_client, user_api_key_dict=auth)


async def log_visibility(
    auth: UserAPIKeyAuth,
    team_lookup: Callable[[UserAPIKeyAuth], Awaitable[tuple[str, ...]]] = permitted_log_teams,
) -> LogVisibility:
    from litellm.proxy.spend_tracking.spend_management_endpoints import (
        _is_admin_view_safe,  # pyright: ignore[reportPrivateUsage]  # Reuse request-log policy
    )

    if _is_admin_view_safe(user_api_key_dict=auth):
        return LogVisibility(all_teams=True)
    if auth.user_id:
        team_ids: Final = await team_lookup(auth)
        return LogVisibility(user_id=auth.user_id, team_ids=team_ids, api_key_hash=auth.token or "")
    if auth.token:
        return LogVisibility(api_key_hash=auth.token)
    raise HTTPException(status_code=403, detail="Not allowed to view logs")
