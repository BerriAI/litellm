"""
Shared team scoping for the search tool authorization checks.
"""

from collections.abc import Awaitable, Callable
from typing import Final, TypeAlias

from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
from litellm.proxy._types import LiteLLM_TeamTable, UserAPIKeyAuth

TeamObjectLookup: TypeAlias = Callable[[str, UserAPIKeyAuth], Awaitable[LiteLLM_TeamTable]]


async def team_object_from_db(team_id: str, user_api_key_dict: UserAPIKeyAuth) -> LiteLLM_TeamTable:
    from litellm.proxy.auth.auth_checks import get_team_object
    from litellm.proxy.proxy_server import (
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    return await get_team_object(
        team_id=team_id,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=user_api_key_dict.parent_otel_span,
        proxy_logging_obj=proxy_logging_obj,
    )


async def resolve_allowlist_team(
    user_api_key_dict: UserAPIKeyAuth,
    lookup_team_object: TeamObjectLookup = team_object_from_db,
) -> LiteLLM_TeamTable | None:
    """
    The team whose object_permission allowlist scopes this caller, or None when there is none.

    Every Admin UI session key is stamped with UI_SESSION_TOKEN_TEAM_ID, a reserved sentinel that
    never has a row in LiteLLM_TeamTable (`/team/new` rejects it as a real team id), so looking it
    up would raise 404 instead of resolving a team. It carries no allowlist of its own, so the
    caller is scoped by its key-level allowlist alone. Any other team id is looked up for real and
    a failed lookup still surfaces.
    """
    team_id: Final = user_api_key_dict.team_id
    if not team_id or team_id == UI_SESSION_TOKEN_TEAM_ID:
        return None
    return await lookup_team_object(team_id, user_api_key_dict)
