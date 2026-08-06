"""
Shared team scoping for the search tool authorization checks.
"""

import asyncio
from collections.abc import Sequence
from typing import Final, Literal, Protocol

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
from litellm.proxy._types import LiteLLM_TeamTable, ProxyException, UserAPIKeyAuth


class TeamObjectLookup(Protocol):
    async def __call__(self, team_id: str, user_api_key_dict: UserAPIKeyAuth) -> LiteLLM_TeamTable: ...


class SessionTeamIdsLookup(Protocol):
    async def __call__(self, user_api_key_dict: UserAPIKeyAuth) -> Sequence[str]: ...


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


async def session_team_ids_from_db(user_api_key_dict: UserAPIKeyAuth) -> Sequence[str]:
    from litellm.proxy.auth.auth_checks import get_user_object
    from litellm.proxy.proxy_server import (
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    user_object: Final = await get_user_object(
        user_id=user_api_key_dict.user_id,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        user_id_upsert=False,
        parent_otel_span=user_api_key_dict.parent_otel_span,
        proxy_logging_obj=proxy_logging_obj,
    )
    if user_object is None:
        raise ValueError("Cannot resolve the teams of a dashboard session key that carries no user id.")
    return tuple(user_object.teams or ())


async def _load_session_team(
    team_id: str,
    user_api_key_dict: UserAPIKeyAuth,
    lookup_team_object: TeamObjectLookup,
) -> LiteLLM_TeamTable | HTTPException:
    try:
        return await lookup_team_object(team_id, user_api_key_dict)
    except HTTPException as exc:
        verbose_proxy_logger.warning(
            "Search tool scoping: dropping team %s for dashboard session user %s: %s",
            team_id,
            user_api_key_dict.user_id,
            exc.detail,
        )
        return exc


async def _session_teams(
    user_api_key_dict: UserAPIKeyAuth,
    lookup_team_object: TeamObjectLookup,
    lookup_session_team_ids: SessionTeamIdsLookup,
) -> tuple[LiteLLM_TeamTable, ...]:
    team_ids: Final = await lookup_session_team_ids(user_api_key_dict)
    loaded: Final = await asyncio.gather(
        *(_load_session_team(team_id, user_api_key_dict, lookup_team_object) for team_id in team_ids)
    )
    teams: Final = tuple(entry for entry in loaded if isinstance(entry, LiteLLM_TeamTable))
    if loaded and not teams:
        raise next(entry for entry in loaded if isinstance(entry, HTTPException))
    return teams


async def resolve_allowlist_teams(
    user_api_key_dict: UserAPIKeyAuth,
    lookup_team_object: TeamObjectLookup = team_object_from_db,
    lookup_session_team_ids: SessionTeamIdsLookup = session_team_ids_from_db,
) -> tuple[LiteLLM_TeamTable, ...]:
    """
    The teams whose object_permission allowlists scope this caller.

    A virtual key names exactly one team and that lookup still surfaces its failure, so a key
    pointing at a team that does not exist is rejected rather than treated as unscoped.

    An Admin UI session key instead carries UI_SESSION_TOKEN_TEAM_ID, a reserved sentinel that
    never has a row in LiteLLM_TeamTable (`/team/new` rejects it as a team id), so looking it up
    would raise 404 for every dashboard caller. It resolves to the real teams backing the session
    user, the same identity the MCP dashboard surfaces resolve, which keeps the team allowlists
    binding on the dashboard instead of dropping them.

    Resolving that set never fails open. Loading the session user surfaces its own failure, and one
    unloadable team is dropped only because `/team/delete` leaves its id behind on the user row, so
    a stale membership is ordinary and dropping it can only narrow a union the other teams still
    scope. Once no team survives there is no union left to narrow, so the first failure is raised
    rather than handing back the empty set that means "belongs to no team".
    """
    team_id: Final = user_api_key_dict.team_id
    if not team_id:
        return ()
    if team_id != UI_SESSION_TOKEN_TEAM_ID:
        return (await lookup_team_object(team_id, user_api_key_dict),)
    return await _session_teams(user_api_key_dict, lookup_team_object, lookup_session_team_ids)


async def _team_denial(search_tool_name: str, team: LiteLLM_TeamTable) -> ProxyException | None:
    from litellm.proxy.auth.auth_checks import can_team_call_search_tool

    try:
        await can_team_call_search_tool(search_tool_name=search_tool_name, team_object=team)
    except ProxyException as denial:
        return denial
    return None


async def authorize_search_tool(
    search_tool_name: str,
    user_api_key_dict: UserAPIKeyAuth,
    teams: Sequence[LiteLLM_TeamTable],
) -> Literal[True]:
    """
    Enforce the key and team object_permission allowlists for one search tool.

    The caller is authorized when its key permits the tool and at least one of its teams does.
    A caller with no teams is unrestricted at the team level, matching a key that belongs to no
    team, and is still bound by its key allowlist.
    """
    from litellm.proxy.auth.auth_checks import (
        can_key_call_search_tool,
        can_team_call_search_tool,
    )

    await can_key_call_search_tool(search_tool_name=search_tool_name, valid_token=user_api_key_dict)
    if not teams:
        return await can_team_call_search_tool(search_tool_name=search_tool_name, team_object=None)

    denials: Final = await asyncio.gather(*(_team_denial(search_tool_name, team) for team in teams))
    if any(denial is None for denial in denials):
        return True
    raise next(denial for denial in denials if denial is not None)


async def can_view_search_tool(
    search_tool_name: str,
    user_api_key_dict: UserAPIKeyAuth,
    teams: Sequence[LiteLLM_TeamTable],
) -> bool:
    """
    Boolean variant of authorize_search_tool, so a listing surface shows exactly the tools the
    caller may invoke.
    """
    try:
        await authorize_search_tool(
            search_tool_name=search_tool_name,
            user_api_key_dict=user_api_key_dict,
            teams=teams,
        )
    except ProxyException:
        return False
    return True
