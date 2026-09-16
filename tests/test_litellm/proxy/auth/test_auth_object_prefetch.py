"""Counts the Redis round trips and DB queries auth object reads cost per cache regime, and checks that the
per-object getters still enforce on their own when the prefetch cannot help."""

import json
from collections.abc import Sequence
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache
from litellm.proxy._types import (
    LiteLLM_OrganizationTable,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTableCachedObj,
    LiteLLM_UserTable,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import (
    get_org_object,
    get_team_membership,
    get_team_object,
    get_user_object,
)
from litellm.proxy.auth.auth_object_prefetch import AuthObjectRefs, prefetch_auth_objects
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

USER_ID = "prefetch-user"
TEAM_ID = "prefetch-team"
ORG_ID = "prefetch-org"

USER_ROW = {
    "user_id": USER_ID,
    "max_budget": 50.0,
    "spend": 1.0,
    "models": ["gpt-5.4-mini"],
    "organization_memberships": [],
}
TEAM_ROW = {
    "team_id": TEAM_ID,
    "organization_id": ORG_ID,
    "max_budget": 500.0,
    "spend": 2.0,
    "models": [],
    "blocked": False,
    "members_with_roles": {},
}
MEMBERSHIP_ROW = {
    "user_id": USER_ID,
    "team_id": TEAM_ID,
    "spend": 3.0,
    "budget_id": "b1",
    "litellm_budget_table": {"budget_id": "b1", "max_budget": 20.0},
}
ORG_ROW = {
    "organization_id": ORG_ID,
    "organization_alias": "org",
    "budget_id": "b2",
    "created_by": "admin",
    "updated_by": "admin",
    "models": [],
    "spend": 4.0,
    "litellm_budget_table": {"budget_id": "b2", "max_budget": 1000.0},
}
ALL_ROWS = {
    "user_row": USER_ROW,
    "team_row": TEAM_ROW,
    "membership_row": MEMBERSHIP_ROW,
    "organization_row": ORG_ROW,
    "project_row": None,
}


class CountingRedis(RedisCache):
    """Redis fake that counts commands and round trips (an MGET or a pipeline is one round trip)."""

    def __init__(self, store: dict[str, str] | None = None, fail: bool = False) -> None:
        self.store: dict[str, str] = dict(store or {})
        self.fail = fail
        self.round_trips = 0
        self.commands: list[str] = []

    def _trip(self, *commands: str) -> None:
        if self.fail:
            raise ConnectionError("redis down")
        self.round_trips += 1
        self.commands.extend(commands)

    async def async_get_cache(self, key: str, **kwargs: object) -> object:
        self._trip(f"GET {key}")
        raw = self.store.get(key)
        return json.loads(raw) if raw is not None else None

    async def async_batch_get_cache(self, key_list: Sequence[str], **kwargs: object) -> dict[str, object]:
        self._trip(f"MGET {' '.join(key_list)}")
        return {key: (json.loads(self.store[key]) if key in self.store else None) for key in key_list}

    async def async_set_cache(self, key: str, value: object, **kwargs: object) -> None:
        self._trip(f"SET {key}")
        self.store[key] = json.dumps(value)

    async def async_set_cache_pipeline(self, cache_list: Sequence[tuple[str, object]], **kwargs: object) -> None:
        self._trip(*(f"SET {key}" for key, _ in cache_list))
        for key, value in cache_list:
            self.store[key] = json.dumps(value)

    async def async_set_cache_pipeline_with_ttls(self, cache_list: Sequence[tuple[str, object, float | None]]) -> None:
        self._trip(*(f"SET {key} ttl={ttl}" for key, _, ttl in cache_list))
        for key, value, _ in cache_list:
            self.store[key] = json.dumps(value)

    async def async_delete_cache(self, key: str) -> None:
        self._trip(f"DEL {key}")
        self.store.pop(key, None)


def _prisma(rows: dict[str, object] | None = ALL_ROWS) -> MagicMock:
    prisma = MagicMock(name="prisma_client")
    prisma.db.query_first = AsyncMock(return_value=rows)
    return prisma


def _non_prefetch_db_calls(prisma: MagicMock) -> list[str]:
    return [str(call) for call in prisma.db.mock_calls if not str(call).startswith("call.query_first(")]


def _cache(redis: RedisCache | None) -> UserApiKeyCache:
    return UserApiKeyCache(in_memory_cache=InMemoryCache(), redis_cache=redis)


def _refs() -> AuthObjectRefs:
    return AuthObjectRefs.from_token(UserAPIKeyAuth(token="t", user_id=USER_ID, team_id=TEAM_ID, org_id=ORG_ID))


async def _read_all_through_getters(
    cache: UserApiKeyCache, prisma: MagicMock
) -> tuple[
    LiteLLM_UserTable | None,
    LiteLLM_TeamTableCachedObj,
    LiteLLM_TeamMembership | None,
    LiteLLM_OrganizationTable | None,
]:
    return (
        await get_user_object(user_id=USER_ID, prisma_client=prisma, user_api_key_cache=cache, user_id_upsert=False),
        await get_team_object(team_id=TEAM_ID, prisma_client=prisma, user_api_key_cache=cache),
        await get_team_membership(user_id=USER_ID, team_id=TEAM_ID, prisma_client=prisma, user_api_key_cache=cache),
        await get_org_object(org_id=ORG_ID, prisma_client=prisma, user_api_key_cache=cache, include_budget_table=True),
    )


def test_refs_from_token_only_names_membership_when_both_ids_present():
    assert AuthObjectRefs.from_token(UserAPIKeyAuth(token="t", team_id=TEAM_ID)).membership_user_id is None
    assert AuthObjectRefs.from_token(UserAPIKeyAuth(token="t", user_id=USER_ID)).membership_user_id is None
    assert AuthObjectRefs.from_token(UserAPIKeyAuth(token="t", user_id=USER_ID, team_id=TEAM_ID)) == AuthObjectRefs(
        user_id=USER_ID, team_id=TEAM_ID, membership_user_id=USER_ID
    )


@pytest.mark.asyncio
async def test_cold_regime_is_one_mget_one_query_and_the_getters_never_touch_io_again():
    redis = CountingRedis()
    prisma = _prisma()
    cache = _cache(redis)

    await prefetch_auth_objects(refs=_refs(), user_api_key_cache=cache, prisma_client=prisma)

    assert prisma.db.query_first.await_count == 1
    assert prisma.db.query_first.await_args.args[1:] == (USER_ID, TEAM_ID, USER_ID, ORG_ID, None)
    mgets = [c for c in redis.commands if c.startswith("MGET")]
    assert len(mgets) == 1
    assert set(mgets[0].split()[1:]) == {
        USER_ID,
        f"team_id:{TEAM_ID}",
        f"{TEAM_ID}_{USER_ID}",
        f"team_membership:{USER_ID}:{TEAM_ID}",
        f"org_id:{ORG_ID}",
        f"org_id:{ORG_ID}:with_budget",
    }
    sets = sorted(c for c in redis.commands if c.startswith("SET"))
    assert sets == sorted(
        [
            f"SET {TEAM_ID}_{USER_ID} ttl=5",
            f"SET org_id:{ORG_ID} ttl=60",
            f"SET org_id:{ORG_ID}:with_budget ttl=60",
            f"SET {USER_ID} ttl=60",
            f"SET team_id:{TEAM_ID} ttl=60",
            f"SET team_membership:{USER_ID}:{TEAM_ID} ttl=None",
        ]
    )
    assert redis.round_trips == 2, "one MGET, one pipeline"

    before = (redis.round_trips, prisma.db.query_first.await_count)
    user, team, membership, org = await _read_all_through_getters(cache, prisma)
    assert (redis.round_trips, prisma.db.query_first.await_count) == before
    assert _non_prefetch_db_calls(prisma) == []

    assert isinstance(user, LiteLLM_UserTable) and user.max_budget == 50.0
    assert isinstance(team, LiteLLM_TeamTableCachedObj) and team.organization_id == ORG_ID
    assert team.last_refreshed_at is not None
    assert isinstance(membership, LiteLLM_TeamMembership) and membership.litellm_budget_table is not None
    assert membership.litellm_budget_table.max_budget == 20.0
    assert isinstance(org, LiteLLM_OrganizationTable) and org.litellm_budget_table is not None
    assert org.litellm_budget_table.max_budget == 1000.0


@pytest.mark.asyncio
async def test_redis_warm_regime_is_exactly_one_mget_and_zero_queries():
    seeded = CountingRedis()
    await prefetch_auth_objects(refs=_refs(), user_api_key_cache=_cache(seeded), prisma_client=_prisma())

    redis = CountingRedis(store=seeded.store)
    prisma = _prisma()
    cache = _cache(redis)
    await prefetch_auth_objects(refs=_refs(), user_api_key_cache=cache, prisma_client=prisma)

    assert redis.round_trips == 1
    assert redis.commands[0].startswith("MGET")
    assert prisma.db.query_first.await_count == 0

    user, team, membership, org = await _read_all_through_getters(cache, prisma)
    assert redis.round_trips == 1
    assert prisma.db.mock_calls == []
    assert (user.user_id, team.team_id, membership.team_id, org.organization_id) == (USER_ID, TEAM_ID, TEAM_ID, ORG_ID)


@pytest.mark.asyncio
async def test_hot_regime_costs_nothing():
    redis = CountingRedis()
    prisma = _prisma()
    cache = _cache(redis)
    await prefetch_auth_objects(refs=_refs(), user_api_key_cache=cache, prisma_client=prisma)
    redis.round_trips, redis.commands = 0, []
    prisma.db.reset_mock()

    await prefetch_auth_objects(refs=_refs(), user_api_key_cache=cache, prisma_client=prisma)
    await _read_all_through_getters(cache, prisma)

    assert redis.round_trips == 0
    assert prisma.db.mock_calls == []


def test_org_json_columns_that_arrive_as_strings_still_validate():
    org = LiteLLM_OrganizationTable.model_validate({**ORG_ROW, "metadata": "{}", "model_spend": "{}"})
    assert org.organization_id == ORG_ID
    assert org.metadata == {}
    assert org.model_spend == {}


@pytest.mark.asyncio
async def test_org_row_that_arrives_as_a_json_string_is_written_to_cache():
    prisma = _prisma({**ALL_ROWS, "organization_row": json.dumps(ORG_ROW)})
    cache = _cache(None)
    await prefetch_auth_objects(refs=_refs(), user_api_key_cache=cache, prisma_client=prisma)
    assert cache.in_memory_cache.get_cache(key=f"org_id:{ORG_ID}") is not None


@pytest.mark.asyncio
async def test_partial_redis_hit_queries_only_the_missing_objects():
    seeded = CountingRedis()
    await prefetch_auth_objects(refs=_refs(), user_api_key_cache=_cache(seeded), prisma_client=_prisma())
    for key in (f"team_id:{TEAM_ID}", f"{TEAM_ID}_{USER_ID}", f"team_membership:{USER_ID}:{TEAM_ID}"):
        del seeded.store[key]

    prisma = _prisma()
    await prefetch_auth_objects(refs=_refs(), user_api_key_cache=_cache(seeded), prisma_client=prisma)

    assert prisma.db.query_first.await_count == 1
    assert prisma.db.query_first.await_args.args[1:] == (None, TEAM_ID, USER_ID, None, None)

    del seeded.store[f"org_id:{ORG_ID}:with_budget"]
    await prefetch_auth_objects(refs=_refs(), user_api_key_cache=_cache(seeded), prisma_client=prisma)
    assert prisma.db.query_first.await_args.args[1:] == (None, None, None, ORG_ID, None)


@pytest.mark.asyncio
async def test_deleted_team_cache_entry_is_refetched_and_the_update_is_visible():
    redis = CountingRedis()
    prisma = _prisma()
    cache = _cache(redis)
    await prefetch_auth_objects(refs=_refs(), user_api_key_cache=cache, prisma_client=prisma)

    await cache.async_delete_cache(f"team_id:{TEAM_ID}")
    prisma.db.query_first.return_value = {**ALL_ROWS, "team_row": {**TEAM_ROW, "blocked": True}}
    await prefetch_auth_objects(refs=_refs(), user_api_key_cache=cache, prisma_client=prisma)

    team = await get_team_object(team_id=TEAM_ID, prisma_client=prisma, user_api_key_cache=cache)
    assert team.blocked is True
    assert prisma.db.query_first.await_count == 2
    assert prisma.db.query_first.await_args.args[1:] == (None, TEAM_ID, None, None, None)


@pytest.mark.asyncio
async def test_row_missing_a_required_column_is_not_cached_so_the_getter_still_fails_closed():
    prisma = _prisma({**ALL_ROWS, "team_row": {"max_budget": 1.0}})
    prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=None)
    cache = _cache(CountingRedis())

    await prefetch_auth_objects(refs=_refs(), user_api_key_cache=cache, prisma_client=prisma)

    with pytest.raises(HTTPException) as exc:
        await get_team_object(team_id=TEAM_ID, prisma_client=prisma, user_api_key_cache=cache)
    assert exc.value.status_code == 404
    assert prisma.db.litellm_teamtable.find_unique.await_count == 1
    assert cache.in_memory_cache.get_cache(USER_ID) is not None


@pytest.mark.asyncio
async def test_absent_rows_are_not_cached_as_present():
    redis = CountingRedis()
    prisma = _prisma({key: None for key in ALL_ROWS})
    cache = _cache(redis)

    await prefetch_auth_objects(refs=_refs(), user_api_key_cache=cache, prisma_client=prisma)

    assert [c for c in redis.commands if c.startswith("SET")] == []
    assert cache.in_memory_cache.get_cache(f"team_id:{TEAM_ID}") is None


@pytest.mark.asyncio
async def test_redis_failure_is_swallowed_and_getters_fall_back_to_their_own_reads():
    prisma = _prisma()
    prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=None)
    cache = _cache(CountingRedis(fail=True))

    await prefetch_auth_objects(refs=_refs(), user_api_key_cache=cache, prisma_client=prisma)

    assert prisma.db.query_first.await_count == 0
    assert cache.in_memory_cache.get_cache(USER_ID) is None


@pytest.mark.asyncio
async def test_no_prisma_still_uses_redis_but_never_queries():
    seeded = CountingRedis()
    await prefetch_auth_objects(refs=_refs(), user_api_key_cache=_cache(seeded), prisma_client=_prisma())
    redis = CountingRedis(store=seeded.store)
    cache = _cache(redis)

    await prefetch_auth_objects(refs=_refs(), user_api_key_cache=cache, prisma_client=None)

    assert redis.round_trips == 1
    assert cache.in_memory_cache.get_cache(f"org_id:{ORG_ID}") is not None


@pytest.mark.asyncio
async def test_no_redis_goes_straight_to_one_query():
    prisma = _prisma()
    cache = _cache(None)

    await prefetch_auth_objects(refs=_refs(), user_api_key_cache=cache, prisma_client=prisma)

    assert prisma.db.query_first.await_count == 1
    assert cache.in_memory_cache.get_cache(f"team_membership:{USER_ID}:{TEAM_ID}") is not None
