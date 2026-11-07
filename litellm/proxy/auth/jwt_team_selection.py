from collections.abc import Awaitable, Callable, Sequence
from typing import Final, NamedTuple

from fastapi import HTTPException

from litellm.proxy._types import LiteLLM_TeamTable, ProxyException
from litellm.proxy.auth.auth_checks import can_team_access_model
from litellm.proxy.auth.team_grants import team_model_aliases


class JWTTeamCandidates(NamedTuple):
    loaded: tuple[LiteLLM_TeamTable, ...]
    matching: tuple[LiteLLM_TeamTable, ...]
    preferred: tuple[tuple[str, LiteLLM_TeamTable], ...]


async def model_preferred_teams(
    team_ids: Sequence[str],
    requested_model: str | None,
    load_team: Callable[[str], Awaitable[LiteLLM_TeamTable | None]],
    *,
    check_model: Callable[..., Awaitable[bool]] = can_team_access_model,
    tolerate_lookup_errors: bool = False,
) -> JWTTeamCandidates:
    from litellm.proxy.proxy_server import llm_router

    async def resolved(team_id: str) -> LiteLLM_TeamTable | None:
        try:
            return await load_team(team_id)
        except HTTPException:
            return None
        except Exception:
            if tolerate_lookup_errors:
                return None
            raise

    async def matches(team: LiteLLM_TeamTable) -> bool:
        if not requested_model:
            return True
        try:
            return await check_model(
                model=requested_model,
                team_object=team,
                llm_router=llm_router,
                team_model_aliases=dict(team_model_aliases(team) or {}),
            )
        except ProxyException:
            return False

    resolved_teams: Final = tuple([await resolved(team_id) for team_id in team_ids])
    loaded: Final = tuple(team for team in resolved_teams if team is not None)
    preferences: Final = tuple([(team, await matches(team)) for team in loaded])
    preferred: Final = tuple((team.team_id, team) for team, eligible in preferences if eligible)
    return JWTTeamCandidates(
        loaded=loaded,
        matching=tuple(team for team, eligible in preferences if eligible),
        preferred=preferred + tuple((team.team_id, team) for team, eligible in preferences if not eligible),
    )
