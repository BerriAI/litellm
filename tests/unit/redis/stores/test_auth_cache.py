import datetime
import json
from typing import Final
from unittest.mock import MagicMock

import pytest

from litellm.caching.redis_batch import Batch, request_redis_batch_scope
from litellm.caching.redis_cache import RedisCache
from litellm.proxy.common_utils.user_api_key_cache import is_user_key_cache_key
from litellm.redis.stores.auth_cache import AuthCacheSettings, AuthCacheStore
from litellm.redis.stores.connection import connect
from litellm.rust_bridge._native import LocalSizes, NativeAuthCache
from tests.unit.caching.redis_batch_double import FakeNatives, fake_send

from .auth_cache_double import FakeAuthCache

_KEY_OBJECT: Final = "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b"
_TEAM_RECORD: Final = {
    "team_id": "team-é",
    "models": ["gpt-5", "claude-opus-5"],
    "max_budget": None,
    "spend": 0.30000000000000004,
    "blocked": False,
    "tpm_limit": 10000000000,
    "metadata": {"tags": [], "quoted": 'it\'s "x"'},
}
_TEAM_JSON: Final = (
    r'{"team_id": "team-\u00e9", "models": ["gpt-5", "claude-opus-5"], "max_budget": null, '
    r'"spend": 0.30000000000000004, "blocked": false, "tpm_limit": 10000000000, '
    r""""metadata": {"tags": [], "quoted": "it's \"x\""}}"""
)
_TEAM_STR: Final = (
    "{'team_id': 'team-é', 'models': ['gpt-5', 'claude-opus-5'], 'max_budget': None, "
    "'spend': 0.30000000000000004, 'blocked': False, 'tpm_limit': 10000000000, "
    """'metadata': {'tags': [], 'quoted': 'it\\'s "x"'}}"""
)
_WRITTEN: Final[tuple[tuple[object, str, str], ...]] = (
    (_TEAM_RECORD, _TEAM_JSON, _TEAM_STR),
    (
        {"token": _KEY_OBJECT, "spend": 1e-05, "rpm_limit": 5, "expires": "2026-11-03T00:00:00Z"},
        '{"token": "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b", "spend": 1e-05, '
        '"rpm_limit": 5, "expires": "2026-11-03T00:00:00Z"}',
        "{'token': '88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b', 'spend': 1e-05, "
        "'rpm_limit': 5, 'expires': '2026-11-03T00:00:00Z'}",
    ),
    ("__no_team_membership__", '"__no_team_membership__"', "__no_team_membership__"),
    (["mcp-a", "mcp-b"], '["mcp-a", "mcp-b"]', "['mcp-a', 'mcp-b']"),
    (12.5, "12.5", "12.5"),
    (7, "7", "7"),
    (True, "true", "True"),
)


@pytest.mark.parametrize(
    ("key", "key_object"),
    [
        (_KEY_OBJECT, True),
        ("88DC28D0F030C55ED4AB77ED8FAF098196CB1C05DF778539800C9F1243FE6B4B", False),
        ("88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4", False),
        ("team_id:88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b", False),
        ("team_membership:user-1:team-1", False),
        ("end_user_restricted_registry", False),
    ],
)
def test_the_native_partition_rule_is_is_user_key_cache_key(key: str, key_object: bool) -> None:
    assert is_user_key_cache_key(key) is key_object


@pytest.mark.parametrize(("value", "encoded", "printed"), _WRITTEN, ids=[encoded for _, encoded, _ in _WRITTEN])
def test_writes_are_what_redis_cache_writes(value: object, encoded: str, printed: str) -> None:
    assert json.dumps(value) == encoded
    assert str(value) == printed


@pytest.mark.parametrize(
    ("stored", "value"),
    [
        (_TEAM_JSON, _TEAM_RECORD),
        (_TEAM_STR, _TEAM_RECORD),
        ("['mcp-a', 'mcp-b']", ["mcp-a", "mcp-b"]),
        (
            "{'user_id': 'u1', 'teams': ['t1'], 'active': True, 'budget': None}",
            {"user_id": "u1", "teams": ["t1"], "active": True, "budget": None},
        ),
        ('"__NO_MAPPING__"', "__NO_MAPPING__"),
    ],
    ids=str,
)
def test_reads_are_what_redis_cache_reads(stored: str, value: object) -> None:
    assert RedisCache._get_cache_logic(MagicMock(spec=RedisCache), stored.encode()) == value  # pyright: ignore[reportPrivateUsage]  # main's own decoder is the contract


@pytest.mark.parametrize("value", [written for written, _, _ in _WRITTEN], ids=str)
def test_plain_values_cross_the_bridge_unchanged(value: object) -> None:
    store: Final = NativeAuthCache(None, memory_entries=200)

    store.set_local("team_id:t1", value)

    assert store.get_local("team_id:t1") == value
    assert type(store.get_local("team_id:t1")) is type(value)


def test_a_live_object_is_refused_rather_than_cached_as_something_else() -> None:
    store: Final = NativeAuthCache(None, memory_entries=200)

    with pytest.raises(TypeError):
        store.set_local("team_id:t1", {"created_at": datetime.datetime(2026, 1, 1)})

    assert store.get_local("team_id:t1") is None


def test_user_key_objects_live_in_their_own_memory_partition() -> None:
    store: Final = NativeAuthCache(None, memory_entries=2)
    store.set_local(_KEY_OBJECT, {"token": _KEY_OBJECT})

    for index in range(5):
        store.set_local(f"team_id:t{index}", {"team_id": index})

    assert store.get_local(_KEY_OBJECT) == {"token": _KEY_OBJECT}
    assert store.len_local() == LocalSizes(objects=2, key_objects=1)


@pytest.mark.asyncio
async def test_a_key_the_request_read_found_absent_is_answered_without_another_read_until_written() -> None:
    fake: Final = FakeAuthCache(redis={"team_id:hit": {"team_id": "hit"}})
    store: Final = AuthCacheStore(fake)

    with request_redis_batch_scope(native_factory=FakeNatives(), native_send_factory=fake_send) as request:  # pyright: ignore[reportArgumentType]  # test double
        found = await store.read_missing(["team_id:absent", "team_id:hit"], batch=Batch.PRE_CALL)
        absent = await store.get("team_id:absent")
        await store.set("team_id:absent", {"team_id": "now"}, None)
        store.delete_local("team_id:absent")
        refilled = await store.get("team_id:absent")
        store.declare_delete("team_id:hit")
        deleted = await store.get("team_id:hit")
        await request.flush_pre_call()

    assert found == {"team_id:absent": None, "team_id:hit": {"team_id": "hit"}}
    assert absent is None
    assert refilled == {"team_id": "now"}
    assert deleted is None
    assert [command[:2] for command in fake.commands] == [("SET", "team_id:absent"), ("GET", "team_id:absent")]


@pytest.mark.asyncio
async def test_outside_a_request_a_read_is_a_round_trip_of_its_own() -> None:
    fake: Final = FakeAuthCache(redis={"team_id:t1": {"team_id": "t1"}})
    store: Final = AuthCacheStore(fake)

    assert await store.read_missing(["team_id:t1", "team_id:t2"], batch=Batch.PRE_CALL) == {
        "team_id:t1": {"team_id": "t1"},
        "team_id:t2": None,
    }
    assert store.declare_set("team_id:t3", {}, None) is None
    assert fake.commands == [("MGET", ("team_id:t1", "team_id:t2"))]


def test_a_new_memory_configuration_keeps_what_memory_holds() -> None:
    store: Final = AuthCacheStore(connect(None).auth_cache(AuthCacheSettings(200, 60)))
    store.set_local("team_id:t1", {"team_id": "t1"}, None)

    store.reconfigure(AuthCacheSettings(500, 5))

    assert store.get_local("team_id:t1") == {"team_id": "t1"}
