"""The human behind an agent's own proxy calls.

``/a2a/{agent}`` forwards the invoking key's ``X-LiteLLM-User-Id`` / ``X-LiteLLM-Team-Id`` to the
agent backend. When the agent echoes them back on requests made with its own key, the proxy caps
that key at what the invoking user and team may reach. The cap is intersected with, never
substituted for, the agent key's own grants and the agent's access group ceiling, so the headers
can only narrow access and need no trust.
"""

from collections.abc import Awaitable, Callable, Mapping
from typing import Final, TypeAlias

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LiteLLM_TeamTable, LiteLLM_UserTable, UserAPIKeyAuth
from litellm.types.agents import (
    AGENT_CALLER_TEAM_ID_HEADER,
    AGENT_CALLER_USER_ID_HEADER,
    AgentCaller,
)

LoadedCallerTeam: TypeAlias = LiteLLM_TeamTable | None
LoadedCallerUser: TypeAlias = LiteLLM_UserTable | None
CallerTeamLoader: TypeAlias = Callable[[UserAPIKeyAuth], Awaitable[LoadedCallerTeam]]  # mutable-ok: Callable params
CallerUserLoader: TypeAlias = Callable[[UserAPIKeyAuth], Awaitable[LoadedCallerUser]]  # mutable-ok: Callable params
CallerResolver: TypeAlias = Callable[[UserAPIKeyAuth], Awaitable[bool]]  # mutable-ok: Callable params


def _header(headers: Mapping[str, str], name: str) -> str | None:
    value: Final = next((raw for key, raw in headers.items() if key.lower() == name), None)
    return value.strip() or None if value is not None else None


def _log_safe(value: str | None) -> str:
    return (value or "").replace("\r", "").replace("\n", "")


def agent_caller_from_headers(headers: Mapping[str, str], user_api_key_auth: UserAPIKeyAuth) -> AgentCaller | None:
    """The caller an agent key is acting for, or ``None`` when the key is not an agent's or no id was echoed."""
    if not user_api_key_auth.agent_id:
        return None
    user_id: Final = _header(headers, AGENT_CALLER_USER_ID_HEADER)
    team_id: Final = _header(headers, AGENT_CALLER_TEAM_ID_HEADER)
    if user_id is None and team_id is None:
        return None
    return AgentCaller(user_id=user_id, team_id=team_id)


def agent_caller_auth(user_api_key_auth: UserAPIKeyAuth) -> UserAPIKeyAuth | None:
    """A minimal auth context standing for the invoking user and team, so the shared key/team/user
    resolvers can be reused unchanged to compute what the caller may reach."""
    caller: Final = user_api_key_auth.agent_caller
    if caller is None:
        return None
    return UserAPIKeyAuth(
        user_id=caller.user_id,
        team_id=caller.team_id,
        parent_otel_span=user_api_key_auth.parent_otel_span,
    )


async def load_agent_caller_team(user_api_key_auth: UserAPIKeyAuth) -> LiteLLM_TeamTable | None:
    """The invoking team's row, or ``None`` when no team id was echoed. Raises when the id names a team
    that cannot be loaded, since a caller we cannot resolve must not be treated as unrestricted."""
    caller: Final = user_api_key_auth.agent_caller
    if caller is None or caller.team_id is None:
        return None
    from litellm.proxy.auth.auth_checks import get_team_object
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    return await get_team_object(
        team_id=caller.team_id,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=user_api_key_auth.parent_otel_span,
        proxy_logging_obj=proxy_logging_obj,
    )


async def load_agent_caller_user(user_api_key_auth: UserAPIKeyAuth) -> LiteLLM_UserTable | None:
    """The invoking user's row, or ``None`` when no user id was echoed. Raises ``UserNotFoundError`` when
    the id names no user, so an unresolvable caller is denied rather than left uncapped."""
    caller: Final = user_api_key_auth.agent_caller
    if caller is None or caller.user_id is None:
        return None
    from litellm.proxy.auth.auth_checks import get_user_object
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    return await get_user_object(
        user_id=caller.user_id,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        user_id_upsert=False,
        parent_otel_span=user_api_key_auth.parent_otel_span,
        proxy_logging_obj=proxy_logging_obj,
    )


async def agent_caller_resolves(
    user_api_key_auth: UserAPIKeyAuth,
    load_team: CallerTeamLoader = load_agent_caller_team,
    load_user: CallerUserLoader = load_agent_caller_user,
) -> bool:
    """Whether every echoed id names a row the proxy can load. The grant resolvers answer "no
    grants" for a missing row just as for an unrestricted one, so callers ask this first and deny
    outright when the invoking identity cannot be resolved."""
    caller: Final = user_api_key_auth.agent_caller
    if caller is None:
        return True
    try:
        await load_team(user_api_key_auth)
        await load_user(user_api_key_auth)
    except Exception as error:  # noqa: BLE001  # any failure to resolve the caller must deny, never widen
        verbose_proxy_logger.warning(
            "agent caller user=%s team=%s could not be resolved (%s), denying",
            _log_safe(caller.user_id),
            _log_safe(caller.team_id),
            type(error).__name__,
        )
        return False
    return True
