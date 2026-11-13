import datetime
import json
from typing import Final
from unittest.mock import MagicMock

import pytest

from litellm.caching.redis_batch import request_redis_batch_scope
from litellm.caching.redis_cache import RedisCache
from litellm.redis.stores.auth_cache import AuthCacheSettings, AuthCacheStore, AuthObject
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
    ("auth_object", "key"),
    [
        (AuthObject.virtual_key(_KEY_OBJECT), _KEY_OBJECT),
        (AuthObject.user("user-1"), "user-1"),
        (AuthObject.team("team-1"), "team_id:team-1"),
        (AuthObject.team_alias("alpha"), "team_alias:alpha"),
        (AuthObject.team_membership(team_id="team_1", user_id="user_1"), "team_1_user_1"),
        (AuthObject.team_membership_reservation(user_id="user-1", team_id="team-1"), "team_membership:user-1:team-1"),
        (AuthObject.org("org-1"), "org_id:org-1"),
        (AuthObject.org_with_budget("org-1"), "org_id:org-1:with_budget"),
        (AuthObject.org_with_budget_last_known("org-1"), "org_id:org-1:with_budget:last_known"),
        (AuthObject.org_alias("acme"), "org_alias:acme"),
        (AuthObject.project("p-1"), "project_id:p-1"),
        (AuthObject.end_user("eu-1"), "end_user_id:eu-1"),
        (AuthObject.end_user_restricted_registry(), "end_user_restricted_registry"),
        (AuthObject.end_user_validation("eu 1"), "end_user_validation:eu 1"),
        (AuthObject.default_end_user_budget("b-1"), "default_end_user_budget:b-1"),
        (AuthObject.team_member_default_budget("b-1"), "team_member_default_budget:b-1"),
        (AuthObject.tag("prod"), "tag:prod"),
        (AuthObject.tag_registry(), "tag_registry"),
        (AuthObject.model_access_group("beta"), "model_access_group:beta"),
        (AuthObject.model_access_group_registry(), "model_access_group_registry"),
        (AuthObject.access_group("ag-1"), "access_group_id:ag-1"),
        (AuthObject.access_group_rule_name("rule"), "access_group_rule_name:rule"),
        (AuthObject.object_permission("op-1"), "object_permission_id:op-1"),
        (AuthObject.user_object_permission("user-1"), "user_object_permission_id:user-1"),
        (AuthObject.agent_object_permission("agent-1"), "agent_object_permission_id:agent-1"),
        (AuthObject.managed_vector_store("vs-1"), "managed_vector_store_id:vs-1"),
        (AuthObject.jwt_key_mapping("sub", "a:b"), "jwt_key_mapping:sub:a:b"),
        (AuthObject.jwt_key_mapping("sub", "a", issuer=""), "jwt_key_mapping:sub:a"),
        (AuthObject.jwt_key_mapping("sub", "a", issuer="https://idp"), "jwt_key_mapping:https://idp:sub:a"),
        (AuthObject.agent_identity("t\u00e9", 'c"1'), 'agent_identity:["t\\u00e9", "c\\"1"]'),
        (AuthObject.global_proxy_spend(), "default_user_id:spend"),
    ],
    ids=lambda case: case.key if isinstance(case, AuthObject) else None,
)
def test_each_object_is_cached_under_mains_key(auth_object: AuthObject, key: str) -> None:
    assert auth_object.key == key


def test_mains_agent_identity_key_is_json_dumps_of_the_pair() -> None:
    pair: Final = ("t\u00e9", 'c"1')

    assert AuthObject.agent_identity(*pair).key == "agent_identity:" + json.dumps(pair)


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

    store.set_local(AuthObject.team("t1"), value)

    assert store.get_local(AuthObject.team("t1")) == value
    assert type(store.get_local(AuthObject.team("t1"))) is type(value)


def test_a_live_object_is_refused_rather_than_cached_as_something_else() -> None:
    store: Final = NativeAuthCache(None, memory_entries=200)

    with pytest.raises(TypeError):
        store.set_local(AuthObject.team("t1"), {"created_at": datetime.datetime(2026, 1, 1)})

    assert store.get_local(AuthObject.team("t1")) is None


def test_user_key_objects_live_in_their_own_memory_partition() -> None:
    store: Final = NativeAuthCache(None, memory_entries=2)
    store.set_local(AuthObject.virtual_key(_KEY_OBJECT), {"token": _KEY_OBJECT})

    for index in range(5):
        store.set_local(AuthObject.team(f"t{index}"), {"team_id": index})

    assert store.get_local(AuthObject.virtual_key(_KEY_OBJECT)) == {"token": _KEY_OBJECT}
    assert store.len_local() == LocalSizes(objects=2, key_objects=1)


@pytest.mark.asyncio
async def test_reads_and_writes_hand_the_native_store_the_requests_batches_once_built() -> None:
    fake: Final = FakeAuthCache()
    store: Final = AuthCacheStore(fake)
    natives: Final = FakeNatives()

    await store.get(AuthObject.team("t1"))
    with request_redis_batch_scope(native_factory=natives, native_send_factory=fake_send) as request:  # pyright: ignore[reportArgumentType]  # test double
        await store.get(AuthObject.team("t1"))
        built: Final = request.native()
        await store.get(AuthObject.team("t1"))
        await store.set(AuthObject.team("t1"), {"team_id": "t1"}, None)
        store.set_sync(AuthObject.team("t1"), {"team_id": "t1"}, None)
        await store.set_many([(AuthObject.team("t1"), {"team_id": "t1"})], None)

    assert fake.requests == [
        ("get", None),
        ("get", None),
        ("get", built),
        ("set", built),
        ("set_sync", built),
        ("set_many", built),
    ]


@pytest.mark.asyncio
async def test_outside_a_request_a_read_is_a_round_trip_of_its_own() -> None:
    fake: Final = FakeAuthCache(redis={"team_id:t1": {"team_id": "t1"}})
    store: Final = AuthCacheStore(fake)

    assert await store.read_missing([AuthObject.team("t1"), AuthObject.team("t2")]) == {
        AuthObject.team("t1"): {"team_id": "t1"},
        AuthObject.team("t2"): None,
    }
    assert store.declare_set(AuthObject.team("t3"), {}, None) is None
    assert fake.commands == [("MGET", ("team_id:t1", "team_id:t2"))]


def test_a_new_memory_configuration_keeps_what_memory_holds() -> None:
    store: Final = AuthCacheStore(connect(None).auth_cache(AuthCacheSettings(200, 60)))
    store.set_local(AuthObject.team("t1"), {"team_id": "t1"}, None)

    store.reconfigure(AuthCacheSettings(500, 5))

    assert store.get_local(AuthObject.team("t1")) == {"team_id": "t1"}
