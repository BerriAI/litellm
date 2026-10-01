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


async def log_visibility(user_api_key_dict: UserAPIKeyAuth) -> LogVisibility:
    from litellm.proxy.spend_tracking.spend_management_endpoints import (
        _get_permitted_team_ids_for_spend_logs_or_empty,  # pyright: ignore[reportPrivateUsage]  # Reuse log policy
        _is_admin_view_safe,  # pyright: ignore[reportPrivateUsage]  # Reuse log policy
    )

    if _is_admin_view_safe(user_api_key_dict=user_api_key_dict):
        return LogVisibility(all_teams=True)

    user_id: Final = user_api_key_dict.user_id
    if user_id:
        from litellm.proxy.proxy_server import prisma_client

        team_ids: Final = (
            ()
            if prisma_client is None
            else await _get_permitted_team_ids_for_spend_logs_or_empty(
                prisma_client=prisma_client,
                user_api_key_dict=user_api_key_dict,
            )
        )
        return LogVisibility(user_id=user_id, team_ids=team_ids)

    if user_api_key_dict.token:
        return LogVisibility(api_key_hash=user_api_key_dict.token)

    raise HTTPException(status_code=403, detail="Not allowed to view logs")
