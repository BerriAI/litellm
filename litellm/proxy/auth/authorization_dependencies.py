from __future__ import annotations

from collections.abc import Awaitable, Callable
from functools import partial
from typing import TYPE_CHECKING, Annotated, Final, TypeAlias

from fastapi import Depends
from pydantic import TypeAdapter

from litellm.constants import PERMITTED_LOG_TEAMS_CACHE_TTL
from litellm.proxy._types import LiteLLM_TeamTable, UserAPIKeyAuth
from litellm.proxy.auth.authorization import permitted_log_team_ids

if TYPE_CHECKING:
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.utils import PrismaClient, ProxyLogging


LogTeamLookup: TypeAlias = Callable[[UserAPIKeyAuth], Awaitable[tuple[str, ...]]]

_TEAM_IDS: Final = TypeAdapter(tuple[str, ...])


def _permitted_log_teams_cache_key(user_id: str) -> str:
    return f"permitted_log_team_ids:{user_id}"


async def load_permitted_log_team_ids(
    auth: UserAPIKeyAuth,
    *,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging,
) -> tuple[str, ...]:
    if prisma_client is None or auth.user_id is None:
        return ()
    cache_key: Final = _permitted_log_teams_cache_key(auth.user_id)
    cached: Final = await user_api_key_cache.async_get_cache(cache_key, local_only=True)
    if cached is not None:
        return _TEAM_IDS.validate_python(cached)
    team_ids: Final = await _query_permitted_log_team_ids(
        auth, prisma_client=prisma_client, user_api_key_cache=user_api_key_cache, proxy_logging_obj=proxy_logging_obj
    )
    await user_api_key_cache.async_set_cache(cache_key, team_ids, local_only=True, ttl=PERMITTED_LOG_TEAMS_CACHE_TTL)
    return team_ids


async def _query_permitted_log_team_ids(
    auth: UserAPIKeyAuth,
    *,
    prisma_client: PrismaClient,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging,
) -> tuple[str, ...]:
    from litellm.proxy.auth.auth_checks import get_user_object
    from litellm.repositories.team_repository import TeamRepository

    user_obj: Final = await get_user_object(
        user_id=auth.user_id,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        user_id_upsert=False,
        proxy_logging_obj=proxy_logging_obj,
    )
    if user_obj is None or not user_obj.teams:
        return ()
    team_rows: Final = await TeamRepository(prisma_client).table.find_many(where={"team_id": {"in": user_obj.teams}})
    return permitted_log_team_ids(auth, (LiteLLM_TeamTable.model_validate(row.model_dump()) for row in team_rows))


async def get_log_team_lookup() -> LogTeamLookup:
    """Bind infrastructure without performing permission I/O before the handler's checks."""
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    return partial(
        load_permitted_log_team_ids,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )


LogTeamLookupDependency: TypeAlias = Annotated[LogTeamLookup, Depends(get_log_team_lookup)]
