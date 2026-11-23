from typing import Final

import pytest

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.types.agents import AgentCaller
from litellm.proxy.model_listing_cache import (
    ListingScope,
    ModelListingCache,
    listing_cache_key,
    listing_cache_scope,
    listing_cache_ttl,
)


@pytest.mark.parametrize(
    ("settings", "expected"),
    (
        ({}, 0.0),
        ({"model_listing_cache_ttl_seconds": 0}, 0.0),
        ({"model_listing_cache_ttl_seconds": -5}, 0.0),
        ({"model_listing_cache_ttl_seconds": True}, 0.0),
        ({"model_listing_cache_ttl_seconds": "30"}, 0.0),
        ({"model_listing_cache_ttl_seconds": 30}, 30.0),
        ({"model_listing_cache_ttl_seconds": 2.5}, 2.5),
    ),
)
def test_listing_cache_ttl_reads_only_a_positive_number(settings: dict[str, object], expected: float) -> None:
    assert listing_cache_ttl(settings) == expected


def test_listing_cache_key_separates_callers_routes_and_queries() -> None:
    admin: Final = UserAPIKeyAuth(token="hash-a", user_role=LitellmUserRoles.PROXY_ADMIN)
    team_key: Final = UserAPIKeyAuth(token="hash-b", team_id="team-1", models=["gpt-4o"])
    assert listing_cache_key("model_info_v1", admin, {"teamId": None}) == listing_cache_key(
        "model_info_v1", admin, {"teamId": None}
    )
    assert listing_cache_key("model_info_v1", admin, {}) != listing_cache_key("model_info_v1", team_key, {})
    assert listing_cache_key("model_info_v1", admin, {}) != listing_cache_key("model_group_info", admin, {})
    assert listing_cache_key("model_group_info", admin, {"model_group": "gpt-4o"}) != listing_cache_key(
        "model_group_info", admin, {"model_group": None}
    )


@pytest.mark.parametrize(
    "narrower",
    (
        {"agent_id": "agent-b"},
        {"team_models": ["gpt-4o"]},
        {"access_group_ids": ["group-b"]},
        {"aliases": {"fast": "gpt-4o-mini"}},
        {"team_model_aliases": {"fast": "gpt-4o-mini"}},
        {"models": ["gpt-4o-mini"]},
    ),
)
def test_listing_cache_key_separates_every_field_the_listing_is_filtered_by(narrower: dict[str, object]) -> None:
    broad: Final = UserAPIKeyAuth(token="hash-a", team_id="team-1", user_id="user-1")
    narrow: Final = UserAPIKeyAuth(token="hash-a", team_id="team-1", user_id="user-1", **narrower)
    assert listing_cache_key("model_info_v1", broad, {}) != listing_cache_key("model_info_v1", narrow, {})


def test_listing_cache_key_separates_the_user_and_team_that_invoked_the_agent() -> None:
    broad: Final = UserAPIKeyAuth(token="hash-a", agent_id="agent-a")
    narrow: Final = UserAPIKeyAuth(token="hash-a", agent_id="agent-a")
    narrow.agent_caller = AgentCaller(user_id="user-2", team_id="team-1")  # set per request, after validation
    assert listing_cache_key("model_group_info", broad, {}) != listing_cache_key("model_group_info", narrow, {})


def test_listing_cache_scope_skips_a_caller_without_a_credential_hash() -> None:
    settings: Final = {"model_listing_cache_ttl_seconds": 30}
    keyed: Final = UserAPIKeyAuth(token="hash-a", user_id="user-1")
    subject: Final = UserAPIKeyAuth(user_id="user-1")
    assert listing_cache_scope(settings, "model_info_v1", keyed, {}) == ListingScope(
        listing_cache_key("model_info_v1", keyed, {}), 30.0
    )
    assert listing_cache_scope(settings, "model_info_v1", subject, {}) is None, "told apart only by claims"
    assert listing_cache_scope({}, "model_info_v1", keyed, {}) is None, "the setting is off"


def test_listing_is_served_until_the_ttl_passes_the_registry_moves_or_the_cache_is_cleared() -> None:
    now: Final[list[float]] = [100.0]  # mutable-ok: the fake clock the test advances
    cache: Final = ModelListingCache[tuple[str, ...]](clock=lambda: now[0])

    assert cache.get("k", registry_version=3) is None
    cache.put("k", ("gpt-4o",), registry_version=3, ttl_seconds=30)
    assert cache.get("k", registry_version=3) == ("gpt-4o",)
    assert cache.get("k", registry_version=4) is None, "a deployment change moves the registry version"

    cache.put("k", ("gpt-4o",), registry_version=4, ttl_seconds=30)
    now[0] = 129.0
    assert cache.get("k", registry_version=4) == ("gpt-4o",)
    now[0] = 131.0
    assert cache.get("k", registry_version=4) is None, "a listing older than the TTL is recomputed"

    now[0] = 200.0
    cache.put("k", ("gpt-4o", "gpt-4.1"), registry_version=4, ttl_seconds=30)
    cache.clear()
    assert cache.get("k", registry_version=4) is None, "a replaced router clears every listing"
