"""Tests for the opt-in team-scoped cache key (Cache.add_team_id_to_cache_key).

On a multi-tenant proxy the response-cache key is otherwise derived only from the
request params, so two teams sending the same request share cache entries - one
team can be served another's cached response. With add_team_id_to_cache_key=True
the authenticated team id (or, with no team, the hashed virtual key) is folded
into the cache key so entries are not reused across tenants; same-team requests
still share the cache. The flag defaults to False, preserving existing behavior.

The scope is read only from the proxy-trusted litellm_params["metadata"] (which
the proxy populates from the authenticated key and strips of any client-supplied
user_api_key_* fields), never from the caller-supplied top-level metadata, so a
client cannot forge the team/key used for scoping.
"""

from litellm.caching.caching import Cache


def _key_for_team(cache: Cache, team_id: str) -> str:
    return cache.get_cache_key(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hello"}],
        litellm_params={"metadata": {"user_api_key_team_id": team_id}},
    )


def test_team_scoped_cache_key_isolates_teams():
    cache = Cache(add_team_id_to_cache_key=True)
    assert _key_for_team(cache, "team-a") != _key_for_team(
        cache, "team-b"
    )  # different teams -> different keys
    assert _key_for_team(cache, "team-a") == _key_for_team(
        cache, "team-a"
    )  # same team -> same key


def test_cache_key_shared_across_teams_by_default():
    cache = Cache()  # flag defaults to False -> existing behavior preserved
    assert _key_for_team(cache, "team-a") == _key_for_team(
        cache, "team-b"
    )  # team ignored -> shared key


def test_team_scoped_cache_key_falls_back_to_api_key_when_no_team():
    cache = Cache(add_team_id_to_cache_key=True)

    def key_for(api_key: str) -> str:
        return cache.get_cache_key(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hello"}],
            litellm_params={"metadata": {"user_api_key": api_key}},
        )

    # no team -> fall back to the hashed virtual key, so callers stay isolated
    assert key_for("hashed-key-1") != key_for("hashed-key-2")


def test_team_scope_ignores_caller_supplied_top_level_metadata():
    # Security: the scope must come only from the proxy-trusted litellm_params
    # metadata, never from caller-supplied top-level metadata. A client must not
    # be able to forge a team by putting user_api_key_team_id in the request body.
    cache = Cache(add_team_id_to_cache_key=True)
    base = cache.get_cache_key(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hello"}],
    )
    forged = cache.get_cache_key(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hello"}],
        metadata={"user_api_key_team_id": "victim-team"},
    )
    assert base == forged  # caller-supplied team is ignored -> no cross-tenant forge


def test_team_scope_reads_authenticated_user_api_key_auth_object():
    # The proxy attaches the authenticated UserAPIKeyAuth as
    # litellm_params["metadata"]["user_api_key_auth"]; its team_id is the
    # un-forgeable source and takes precedence over the flat field.
    cache = Cache(add_team_id_to_cache_key=True)

    class _Auth:
        def __init__(self, team_id: str) -> None:
            self.team_id = team_id
            self.api_key = "hashed-key"

    def key_for(team_id: str) -> str:
        return cache.get_cache_key(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hello"}],
            litellm_params={"metadata": {"user_api_key_auth": _Auth(team_id)}},
        )

    assert key_for("team-a") != key_for("team-b")
