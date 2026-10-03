from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias

from litellm.proxy._types import KeyManagementRoutes, LiteLLM_TeamTable, LitellmUserRoles, UserAPIKeyAuth


@dataclass(frozen=True, slots=True)
class AllRows:
    """Unrestricted reads, granted by the consuming endpoint's role checks."""


@dataclass(frozen=True, slots=True)
class OwnedRows:
    """Rows owned by ``user_id`` or by any of ``team_ids``; a ``None`` user grants no own-user rows."""

    user_id: str | None
    team_ids: tuple[str, ...] = ()


ReadScope: TypeAlias = AllRows | OwnedRows


async def resolve_owned_read_scope(
    user_id: str | None,
    permitted_team_lookup: Callable[[], Awaitable[Sequence[str]]],
) -> OwnedRows:
    """Resolve own-user and permitted-team reads, falling back to own-user on lookup failure."""
    if user_id is None:
        return OwnedRows(None)
    try:
        team_ids: Final = tuple(await permitted_team_lookup())
    except Exception:  # noqa: BLE001  # preserve spend-log own-user fallback for every permission lookup failure
        return OwnedRows(user_id)
    return OwnedRows(user_id, team_ids)


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


async def resolve_trace_read_scope(
    auth: UserAPIKeyAuth,
    permitted_team_lookup: Callable[[], Awaitable[Sequence[str]]],
) -> ReadScope | None:
    if auth.user_role in (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY):
        return AllRows()
    if not auth.user_id:
        return None
    return await resolve_owned_read_scope(auth.user_id, permitted_team_lookup)
