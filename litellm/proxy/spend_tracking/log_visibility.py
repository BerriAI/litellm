from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Final

from fastapi import HTTPException

from litellm.proxy._types import UserAPIKeyAuth


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
