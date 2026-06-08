"""Tests for the opt-in team-scoped cache key (Cache.add_team_id_to_cache_key).

On a multi-tenant proxy the response-cache key is otherwise derived only from the
request params, so two teams sending the same request share cache entries - one
team can be served another's cached response. With add_team_id_to_cache_key=True
the requesting team id is folded into the cache key so entries are not reused
across teams; same-team requests still share the cache. The flag defaults to
False, preserving the existing behavior.
"""

from litellm.caching.caching import Cache


def _key(cache: Cache, team_id: str) -> str:
    return cache.get_cache_key(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hello"}],
        metadata={"user_api_key_team_id": team_id},
    )


def test_team_scoped_cache_key_isolates_teams():
    cache = Cache(add_team_id_to_cache_key=True)
    assert _key(cache, "team-a") != _key(
        cache, "team-b"
    )  # different teams -> different keys
    assert _key(cache, "team-a") == _key(cache, "team-a")  # same team -> same key


def test_cache_key_shared_across_teams_by_default():
    cache = Cache()  # flag defaults to False -> existing behavior preserved
    assert _key(cache, "team-a") == _key(cache, "team-b")  # team ignored -> shared key


def test_team_scoped_cache_key_falls_back_to_api_key_when_no_team():
    cache = Cache(add_team_id_to_cache_key=True)

    def key_for(api_key: str) -> str:
        return cache.get_cache_key(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hello"}],
            metadata={"user_api_key": api_key},
        )

    # no team -> fall back to the api key, so different keys are still isolated
    assert key_for("key-1") != key_for("key-2")
