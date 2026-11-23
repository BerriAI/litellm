from collections.abc import Awaitable, Callable
from typing import Final, TypeAlias

import pytest
from pydantic import BaseModel, ConfigDict

from litellm.caching.redis_cache import RedisCache
from litellm.proxy.common_utils.hook_cache import HookCache
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.redis.stores.auth_cache import AuthObject

_TEAM_OBJECT: Final = AuthObject.team("t1")
_TEAM_KEY: Final = _TEAM_OBJECT.key
_TEAM: Final = {"team_id": "t1", "models": ["gpt-5"]}
_TOKEN: Final = "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b"


def _caches() -> tuple[HookCache, UserApiKeyCache]:
    auth: Final = UserApiKeyCache()
    return HookCache(auth_cache=lambda: auth), auth


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stored_as", "value"),
    [(_TEAM_OBJECT, _TEAM), (AuthObject.virtual_key(_TOKEN), {"token": _TOKEN})],
    ids=["team", "virtual-key"],
)
async def test_a_hook_reads_an_auth_object_by_the_key_the_auth_cache_holds_it_under(
    stored_as: AuthObject, value: object
) -> None:
    hooks, auth = _caches()
    await auth.async_set_cache(stored_as, value)

    assert await hooks.async_get_cache(stored_as.key) == value
    assert hooks.get_cache(stored_as.key) == value


@pytest.mark.asyncio
async def test_a_hook_reads_back_the_very_object_it_stored() -> None:
    hooks, auth = _caches()
    state: Final = {1: ("not", "json")}

    await hooks.async_set_cache("hook-state", state)

    assert await hooks.async_get_cache("hook-state") is state
    assert hooks.get_cache("hook-state") is state
    assert auth.get_local_by_key("hook-state") is None


async def _delete(hooks: HookCache, key: str) -> None:
    await hooks.async_delete_cache(key)


async def _delete_sync(hooks: HookCache, key: str) -> None:
    hooks.delete_cache(key)


async def _delete_many(hooks: HookCache, key: str) -> None:
    await hooks.async_delete_cache_keys([key])


_Delete: TypeAlias = Callable[[HookCache, str], Awaitable[None]]
_DELETES: Final = (_delete, _delete_sync, _delete_many)


@pytest.mark.asyncio
@pytest.mark.parametrize("delete", _DELETES)
async def test_a_hook_that_deletes_an_auth_object_drops_it_from_the_auth_cache(delete: _Delete) -> None:
    hooks, auth = _caches()
    await auth.async_set_cache(_TEAM_OBJECT, _TEAM)

    await delete(hooks, _TEAM_KEY)

    assert await auth.async_get_cache(_TEAM_OBJECT) is None
    assert await hooks.async_get_cache(_TEAM_KEY) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("delete", _DELETES)
async def test_a_hook_that_deletes_its_own_state_no_longer_reads_it(delete: _Delete) -> None:
    hooks, _ = _caches()
    await hooks.async_set_cache("hook-state", 1)

    await delete(hooks, "hook-state")

    assert await hooks.async_get_cache("hook-state") is None


class _HookState(BaseModel):
    model_config = ConfigDict(frozen=True)

    count: int


class _SharedTier(RedisCache):
    """The writes a hook cache hands to the Redis every worker shares."""

    def __init__(self) -> None:
        self.written: tuple[tuple[str, object], ...] = ()

    def set_cache(self, key: str, value: object, **kwargs: object) -> None:
        self.written = (*self.written, (key, value))

    async def async_set_cache(self, key: str | None, value: object, **kwargs: object) -> None:
        self.written = (*self.written, (str(key), value))

    async def async_set_cache_pipeline(
        self, cache_list: list[tuple[str, object]], ttl: float | None = None, **kwargs: object
    ) -> None:
        self.written = (*self.written, *cache_list)


async def _set(hooks: HookCache, key: str, value: object) -> None:
    await hooks.async_set_cache(key, value)


async def _set_sync(hooks: HookCache, key: str, value: object) -> None:
    hooks.set_cache(key, value)


async def _set_many(hooks: HookCache, key: str, value: object) -> None:
    await hooks.async_set_cache_pipeline([(key, value)])


_Set: TypeAlias = Callable[[HookCache, str, object], Awaitable[None]]


@pytest.mark.asyncio
@pytest.mark.parametrize("write", [_set, _set_sync, _set_many])
async def test_a_model_a_hook_stores_reaches_the_shared_tier_as_json_and_stays_the_model_in_memory(
    write: _Set,
) -> None:
    hooks, _ = _caches()
    shared: Final = _SharedTier()
    hooks.attach_redis_cache(shared)
    state: Final = _HookState(count=1)

    await write(hooks, "hook-state", state)

    assert shared.written == (("hook-state", {"count": 1}),)
    assert await hooks.async_get_cache("hook-state") is state
