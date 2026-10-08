from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Final

import pytest

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import PERMITTED_LOG_TEAMS_CACHE_TTL
from litellm.proxy._types import LiteLLM_UserTable, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.authorization_dependencies import load_permitted_log_team_ids
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import ProxyLogging


@dataclass
class _TeamRow:
    team_id: str
    members_with_roles: tuple[dict[str, str], ...]

    def model_dump(self) -> dict[str, object]:
        return {"team_id": self.team_id, "members_with_roles": list(self.members_with_roles)}


@dataclass
class _TeamTable:
    rows: tuple[_TeamRow, ...]
    reads: list[object] = field(default_factory=list)

    async def find_many(self, where: object) -> tuple[_TeamRow, ...]:
        self.reads.append(where)
        return self.rows


def _admin_of(team_id: str, user_id: str) -> _TeamRow:
    return _TeamRow(team_id, ({"user_id": user_id, "role": "admin"},))


@dataclass
class _Clock:
    now: float = 1_000.0

    def __call__(self) -> float:
        return self.now


async def _cache_with_users(*user_ids: str, clock: _Clock | None = None) -> UserApiKeyCache:
    cache: Final = UserApiKeyCache(in_memory_cache=InMemoryCache(clock=clock))
    for user_id in user_ids:
        await cache.async_set_cache(
            key=user_id,
            value=LiteLLM_UserTable(user_id=user_id, teams=["team-a", "team-b"]),
            model_type=LiteLLM_UserTable,
        )
    return cache


def _caller(user_id: str) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_id=user_id, user_role=LitellmUserRoles.INTERNAL_USER)


@pytest.mark.asyncio
async def test_repeated_trace_reads_reuse_the_permitted_team_lookup() -> None:
    table: Final = _TeamTable((_admin_of("team-a", "alice"), _admin_of("team-b", "bob")))
    prisma: Final = SimpleNamespace(db=SimpleNamespace(litellm_teamtable=table))
    cache: Final = await _cache_with_users("alice", "bob")
    logging: Final = ProxyLogging(user_api_key_cache=cache)

    async def lookup(user_id: str) -> tuple[str, ...]:
        return await load_permitted_log_team_ids(
            _caller(user_id), prisma_client=prisma, user_api_key_cache=cache, proxy_logging_obj=logging
        )

    assert await lookup("alice") == ("team-a",)
    assert await lookup("alice") == ("team-a",)
    assert len(table.reads) == 1
    assert await lookup("bob") == ("team-b",)
    assert await lookup("bob") == ("team-b",)
    assert len(table.reads) == 2


@pytest.mark.asyncio
async def test_permitted_teams_are_reread_after_the_cache_entry_expires() -> None:
    table: Final = _TeamTable((_admin_of("team-a", "alice"),))
    prisma: Final = SimpleNamespace(db=SimpleNamespace(litellm_teamtable=table))
    clock: Final = _Clock()
    cache: Final = await _cache_with_users("alice", clock=clock)
    logging: Final = ProxyLogging(user_api_key_cache=cache)

    async def lookup() -> tuple[str, ...]:
        return await load_permitted_log_team_ids(
            _caller("alice"), prisma_client=prisma, user_api_key_cache=cache, proxy_logging_obj=logging
        )

    assert await lookup() == ("team-a",)
    table.rows = ()
    clock.now += PERMITTED_LOG_TEAMS_CACHE_TTL - 1
    assert await lookup() == ("team-a",)
    assert len(table.reads) == 1
    clock.now += 2
    assert await lookup() == ()
    assert len(table.reads) == 2
