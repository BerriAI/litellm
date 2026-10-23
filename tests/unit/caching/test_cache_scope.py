import asyncio
from collections.abc import Callable, Mapping
from typing import Final

import pytest

import litellm
from litellm.caching.caching import Cache
from litellm.caching.caching_handler import _PENDING_CACHE_WRITES
from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching.cache_scope import CacheScopePolicy
from litellm.types.caching import CacheScope, LiteLLMCacheType, SemanticCacheScope
from litellm.types.utils import Choices, Message, ModelResponse

_MODEL: Final = "gpt-6.1-sol"
_MESSAGES: Final = [{"role": "user", "content": "cache scope regression"}]
_REQUEST: Final[Mapping[str, object]] = {"model": _MODEL, "messages": _MESSAGES}
_CACHED_RESPONSE: Final = {"answer": "previously cached response"}


@pytest.mark.parametrize("cache_factory", [Cache, litellm.enable_cache, litellm.update_cache])
def test_omitted_scope_shares_cache_across_teams(
    monkeypatch: pytest.MonkeyPatch, cache_factory: Callable[[], Cache | None]
) -> None:
    monkeypatch.setattr(litellm, "cache", None)
    configured: Final = cache_factory()
    cache: Final = configured if configured is not None else litellm.cache
    assert cache is not None
    policy: Final = CacheScopePolicy(cache)
    request: Final = {**_REQUEST, "cache": {"namespace": "application"}}
    first: Final = policy.apply(request, UserAPIKeyAuth(api_key="key-a", team_id="team-a"))
    other_team: Final = policy.apply(request, UserAPIKeyAuth(api_key="key-b", team_id="team-b"))

    cache.add_cache(_CACHED_RESPONSE, **first)

    assert cache.get_cache(**other_team) == _CACHED_RESPONSE
    assert cache.get_cache(**request) == _CACHED_RESPONSE


@pytest.mark.parametrize(
    ("scope", "first_team", "second_team", "first_key", "second_key", "shares_hit"),
    [
        (CacheScope.GLOBAL, "team-a", "team-b", "key-a", "key-b", True),
        (CacheScope.TEAM, "team-a", "team-a", "key-a", "key-b", True),
        (CacheScope.TEAM, "team-a", "team-a", None, None, True),
        (CacheScope.TEAM, "team-a", "team-b", "key-a", "key-b", False),
        (CacheScope.KEY, "team-a", "team-a", "key-a", "key-b", False),
        (CacheScope.KEY, "team-a", "team-a", "key-a", "key-a", True),
        (CacheScope.TEAM, None, None, "key-a", "key-b", False),
        (CacheScope.TEAM, None, None, "key-a", "key-a", True),
        (CacheScope.TEAM, "same-id", None, "key-a", "same-id", False),
    ],
)
def test_cache_hits_follow_authenticated_scope(
    scope: CacheScope,
    first_team: str | None,
    second_team: str | None,
    first_key: str | None,
    second_key: str | None,
    shares_hit: bool,
) -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL, cache_scope=scope)
    first: Final = CacheScopePolicy(cache).apply(_REQUEST, UserAPIKeyAuth(api_key=first_key, team_id=first_team))
    second: Final = CacheScopePolicy(cache).apply(_REQUEST, UserAPIKeyAuth(api_key=second_key, team_id=second_team))

    cache.add_cache(_CACHED_RESPONSE, **first)

    assert cache.get_cache(**first) == _CACHED_RESPONSE
    assert cache.get_cache(**second) == (_CACHED_RESPONSE if shares_hit else None)


@pytest.mark.parametrize("second_team", ["team-a", "team-b"])
def test_key_scope_separates_cli_sessions_with_the_same_logging_alias(second_team: str) -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL, cache_scope=CacheScope.KEY)
    first_auth: Final = UserAPIKeyAuth(
        token="session-credential-a", is_session_token=True, key_alias="cli-session-user", team_id="team-a"
    )
    second_auth: Final = UserAPIKeyAuth(
        token="session-credential-b", is_session_token=True, key_alias="cli-session-user", team_id=second_team
    )
    first: Final = CacheScopePolicy(cache).apply(_REQUEST, first_auth)
    second: Final = CacheScopePolicy(cache).apply(_REQUEST, second_auth)

    cache.add_cache(_CACHED_RESPONSE, **first)

    assert cache.get_cache(**CacheScopePolicy(cache).apply(_REQUEST, first_auth)) == _CACHED_RESPONSE
    assert cache.get_cache(**second) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [CacheScope.TEAM, CacheScope.KEY])
@pytest.mark.parametrize("override_location", ["cache_key", "preset_cache_key", "litellm_params"])
async def test_caller_cache_keys_and_namespaces_cannot_read_another_identity(
    monkeypatch: pytest.MonkeyPatch, scope: CacheScope, override_location: str
) -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL, cache_scope=scope, namespace="gateway")
    monkeypatch.setattr(litellm, "cache", cache)
    victim: Final = CacheScopePolicy(cache).apply(_REQUEST, UserAPIKeyAuth(api_key="victim-key", team_id="victim-team"))
    victim_key: Final = cache.get_cache_key(**victim)
    forged_override: Final = (
        {"litellm_params": {"preset_cache_key": victim_key}}
        if override_location == "litellm_params"
        else {override_location: victim_key}
    )
    forged: Final = {
        **_REQUEST,
        **forged_override,
        "cache": victim["cache"],
        "metadata": {
            "redis_namespace": victim_key.rsplit(":", maxsplit=1)[0],
            "user_api_key_hash": "victim-key",
            "user_api_key_team_id": "victim-team",
        },
    }
    attacker: Final = CacheScopePolicy(cache).apply(
        forged, UserAPIKeyAuth(api_key="attacker-key", team_id="attacker-team")
    )
    seeded: Final = ModelResponse(choices=[Choices(message=Message(role="assistant", content="victim response"))])
    cache.add_cache(seeded, **victim)

    response: Final = await litellm.acompletion(mock_response="attacker response", **attacker)
    await asyncio.gather(*tuple(_PENDING_CACHE_WRITES))

    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == "attacker response"
    assert cache.get_cache(**victim) == seeded.model_dump(exclude_unset=True)


@pytest.mark.parametrize(
    ("namespace_request", "expected_namespace"),
    [
        ({"cache": {"namespace": "application"}, "metadata": {"redis_namespace": "ignored"}}, "application"),
        ({"metadata": {"redis_namespace": "metadata"}}, "metadata"),
        ({}, "gateway"),
    ],
)
def test_scope_retains_existing_namespace_partitions(
    namespace_request: Mapping[str, object], expected_namespace: str
) -> None:
    auth: Final = UserAPIKeyAuth(api_key="same-key", team_id="same-team")
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL, cache_scope=CacheScope.TEAM, namespace="gateway")
    first: Final = CacheScopePolicy(cache).apply({**_REQUEST, **namespace_request}, auth)
    same_namespace: Final = CacheScopePolicy(cache).apply(
        {**_REQUEST, "cache": {"namespace": expected_namespace}}, auth
    )
    other_namespace: Final = CacheScopePolicy(cache).apply(
        {**_REQUEST, "cache": {"namespace": "other-application"}}, auth
    )
    cache.add_cache(_CACHED_RESPONSE, **first)

    assert cache.get_cache(**same_namespace) == _CACHED_RESPONSE
    assert cache.get_cache(**other_namespace) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "identity"),
    [
        (CacheScope.TEAM, UserAPIKeyAuth()),
        (CacheScope.KEY, UserAPIKeyAuth()),
        (CacheScope.TEAM, UserAPIKeyAuth(user_id="jwt-user")),
        (CacheScope.KEY, UserAPIKeyAuth(user_id="jwt-user")),
        (CacheScope.KEY, UserAPIKeyAuth(team_id="team-without-key")),
    ],
)
async def test_missing_identity_neither_reads_nor_overwrites_shared_cache(
    monkeypatch: pytest.MonkeyPatch, scope: CacheScope, identity: UserAPIKeyAuth
) -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL, cache_scope=scope)
    monkeypatch.setattr(litellm, "cache", cache)
    seeded: Final = ModelResponse(choices=[Choices(message=Message(role="assistant", content="cached response"))])
    cache.add_cache(seeded, **_REQUEST)
    request: Final = CacheScopePolicy(cache).apply(
        {"cache": {"use-cache": True, "no-cache": False, "no-store": False}}, identity
    )

    response: Final = await litellm.acompletion(
        model=_MODEL, messages=_MESSAGES, mock_response="fresh response", **request
    )
    await asyncio.gather(*tuple(_PENDING_CACHE_WRITES))

    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content == "fresh response"
    assert cache.get_cache(**_REQUEST) == seeded.model_dump(exclude_unset=True)


def test_global_scope_preserves_explicit_cache_keys() -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL, cache_scope=CacheScope.GLOBAL)
    request: Final = {**_REQUEST, "cache_key": "application-managed-cache-key"}
    prepared: Final = CacheScopePolicy(cache).apply(request, UserAPIKeyAuth())
    cache.add_cache(_CACHED_RESPONSE, **request)

    assert cache.get_cache(**prepared) == _CACHED_RESPONSE
    assert cache.get_cache(cache_key="application-managed-cache-key") == _CACHED_RESPONSE


@pytest.mark.parametrize("cache_type", [LiteLLMCacheType.REDIS_SEMANTIC, LiteLLMCacheType.VALKEY_SEMANTIC])
@pytest.mark.parametrize("scope", [CacheScope.TEAM, CacheScope.KEY])
def test_exact_scope_preserves_semantic_end_user_buckets(cache_type: LiteLLMCacheType, scope: CacheScope) -> None:
    cache: Final = Cache(
        type=cache_type,
        cache_scope=scope,
        semantic_cache_scope=SemanticCacheScope.END_USER,
        redis_url="redis://localhost:6379",
        similarity_threshold=0.8,
    )
    first: Final = {**_REQUEST, "metadata": {"user_api_key": "same-key", "user_api_key_end_user_id": "user-a"}}
    paraphrase: Final = {**first, "messages": [{"role": "user", "content": "a differently worded prompt"}]}
    other_user: Final = {**_REQUEST, "metadata": {"user_api_key": "same-key", "user_api_key_end_user_id": "user-b"}}
    prepared: Final = CacheScopePolicy(cache).apply(first, UserAPIKeyAuth())

    assert prepared == first
    assert cache.get_cache_key(**prepared) == cache.get_cache_key(**first)
    assert cache.get_cache_key(**prepared) == cache.get_cache_key(**paraphrase)
    assert cache.get_cache_key(**prepared) != cache.get_cache_key(**other_user)
