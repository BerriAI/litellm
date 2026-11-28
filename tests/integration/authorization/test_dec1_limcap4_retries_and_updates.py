"""Key windows under router retries and fallbacks, and limit changes reaching a second instance.

Retries and fallbacks. The v3 limiter's pre-call hook (``litellm/proxy/hooks/parallel_request_limiter_v3.py``) runs
once per proxy request, before the router. Router retries of the same model group happen inside the router call, so
the key, team and per-model windows count a retried request once, and settlement charges the usage of the attempt
that answered. A cross-group fallback after a provider error is different: ``RouterFallbackRateLimitCheck``
(``litellm/proxy/auth/fallback_rate_limit.py:38``) calls ``async_fallback_target_allowed`` (:4188), which re-checks
and increments the caller's per-model descriptors on the fallback group only.

Limit changes. ``/key/update`` and ``/team/update`` write the row and evict the cached object here and on every other
instance through the Redis channel ``litellm_proxy.auth_cache_invalidation``
(``litellm/proxy/common_utils/auth_cache_invalidation_pubsub.py``, ``evict_and_broadcast``). The counters already
live in the shared Redis, so a second instance enforces the new limit against the same window within seconds,
without a restart. Single-instance updates are pinned by ``test_dec1_limits_updates_and_windows.py`` and
``test_dec1_ratelim_owner_updates.py``; these tests use the shared gateway as the writer and an owned proxy on the
same database and Redis as the peer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.wire import wire_server
from tests.integration.authorization.dec1_limcap4_support import (
    PEER_PICKUP_SECONDS,
    PROVIDER_TOTAL,
    FlakyProvider,
    counter,
    counter_settles,
    unique,
    until_served,
)
from tests.integration.authorization.dec1_limits_support import (
    assert_limited,
    chat,
    hashed,
    limits_proxy,
    provider,
    scope_headers,
    shared_redis,
    statuses,
)

RETRIES: Final = {"num_retries": 2}


def test_a_request_the_router_retried_counts_once_in_the_key_request_window(gateway: Gateway, tmp_path: Path) -> None:
    """On a proxy with ``num_retries: 2`` the provider fails the first attempt with 500 and answers the retry. The
    caller gets 200, the provider saw at least two attempts, and ``{api_key:<hash>}:requests`` holds 1: the
    limiter's pre-call hook ran once for the proxy request, not once per attempt."""
    flaky: Final = FlakyProvider(failures=1)
    with (
        wire_server(flaky) as wire,
        limits_proxy(gateway, tmp_path, router_settings=RETRIES, shared=True) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=10)
        answered: Final = chat(candidate, model, key)
        assert answered.status_code == 200, answered.text
        assert flaky.attempts >= 2
        assert counter_settles("api_key", hashed(key), "requests", 1) == 1


def test_retries_do_not_use_up_the_key_rpm_limit(gateway: Gateway, tmp_path: Path) -> None:
    """A key with ``rpm_limit: 2`` makes one request that the router retries once, then a second request: both are
    200, and only the third is refused for ``api_key: <hash>`` with ``Current limit: 2``."""
    with (
        wire_server(FlakyProvider(failures=1)) as wire,
        limits_proxy(gateway, tmp_path, router_settings=RETRIES, shared=True) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=2)
        assert [chat(candidate, model, key).status_code for _ in range(2)] == [200, 200]
        assert_limited(chat(candidate, model, key), "api_key", hashed(key), "requests", 2)


def test_a_retried_request_settles_only_the_answering_attempts_tokens(gateway: Gateway, tmp_path: Path) -> None:
    """A key with ``tpm_limit: 1000`` makes one request whose first attempt fails. ``{api_key:<hash>}:tokens``
    settles to 18, the usage of the attempt that answered (11 prompt + 7 completion): the failed attempt reported no
    usage and the reservation is replaced, not added to."""
    with (
        wire_server(FlakyProvider(failures=1)) as wire,
        limits_proxy(gateway, tmp_path, router_settings=RETRIES, shared=True) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=1000)
        assert chat(candidate, model, key).status_code == 200
        assert counter_settles("api_key", hashed(key), "tokens", PROVIDER_TOTAL) == PROVIDER_TOTAL


def test_a_request_that_fails_after_every_retry_counts_once_and_releases_its_tokens(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The provider fails every attempt. With ``num_retries: 2`` the caller gets the provider's error after three
    attempts. ``{api_key:<hash>}:requests`` holds 1 and ``{api_key:<hash>}:tokens`` ends at 0 or absent: the request
    still used its request slot, and the failure hook gave back the reserved tokens."""
    flaky: Final = FlakyProvider(failures=-1)
    with (
        wire_server(flaky) as wire,
        limits_proxy(gateway, tmp_path, router_settings=RETRIES, shared=True) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=10, tpm_limit=1000)
        failed: Final = chat(candidate, model, key)
        assert failed.status_code >= 500, failed.text
        assert flaky.attempts >= 3
        assert counter("api_key", hashed(key), "requests") == 1
        assert counter_settles("api_key", hashed(key), "tokens", 0) in (None, 0)


def test_a_per_model_key_limit_counts_a_retried_request_once(gateway: Gateway, tmp_path: Path) -> None:
    """A key with ``model_rpm_limit: {<model>: 2}`` sends a request that is retried once, then a second request; both
    are 200 and the third is refused for ``model_per_key: <hash>:<model>``. Retries stay inside one model group, so
    the per-model window is checked once per proxy request."""
    with (
        wire_server(FlakyProvider(failures=1)) as wire,
        limits_proxy(gateway, tmp_path, router_settings=RETRIES, shared=True) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], model_rpm_limit={model: 2})
        assert [chat(candidate, model, key).status_code for _ in range(2)] == [200, 200]
        assert_limited(chat(candidate, model, key), "model_per_key", f"{hashed(key)}:{model}", "requests", 2)


def test_a_provider_error_fallback_takes_one_request_of_the_fallback_models_per_key_window(
    gateway: Gateway, tmp_path: Path
) -> None:
    """``primary`` fails every attempt and falls back to ``spare``. A key with ``model_rpm_limit: {<spare>: 5}``
    is served by ``spare``; ``{model_per_key:<hash>:<spare>}:requests`` then holds 1 (``async_fallback_target_allowed``
    increments the per-model descriptors of the target) while ``{api_key:<hash>}:requests`` still holds 1 for the
    whole request. UNVERIFIED: that ``fallbacks`` given in the owned proxy's ``router_settings`` name DB models."""
    primary: Final = unique("primary")
    spare: Final = unique("spare")
    with wire_server(FlakyProvider(failures=-1)) as broken, wire_server(provider()) as healthy:
        settings: Final = {"num_retries": 0, "fallbacks": [{primary: [spare]}]}
        with (
            limits_proxy(gateway, tmp_path, router_settings=settings, shared=True) as candidate,
            candidate.scenario() as scenario,
        ):
            scenario.model(name=primary, api_base=f"{broken.url}/v1")
            scenario.model(name=spare, api_base=f"{healthy.url}/v1")
            key: Final = scenario.key(models=[primary, spare], rpm_limit=10, model_rpm_limit={spare: 5})
            served: Final = chat(candidate, primary, key)
            assert served.status_code == 200, served.text
            assert counter_settles("model_per_key", f"{hashed(key)}:{spare}", "requests", 1) == 1
            assert counter("api_key", hashed(key), "requests") == 1


def test_raising_a_keys_rpm_on_one_instance_admits_the_key_on_the_other(gateway: Gateway, tmp_path: Path) -> None:
    """A key with ``rpm_limit: 1`` is served once by the shared gateway and then refused by the peer, which now
    caches the key. ``/key/update`` through the shared gateway raises ``rpm_limit`` to 3. Within seconds the peer
    serves the key again in the same minute: the update evicted the peer's cached key over Redis and the window
    already holds the first request."""
    with (
        wire_server(provider()) as wire,
        gateway.scenario() as scenario,
        limits_proxy(gateway, tmp_path, shared=True) as peer,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        until_served(peer, model, peer.key)
        key: Final = scenario.key(models=[model], rpm_limit=1)
        assert chat(gateway, model, key).status_code == 200
        assert chat(peer, model, key).status_code == 429
        gateway.post("/key/update", {"key": key, "rpm_limit": 3})
        served: Final = eventually(
            lambda: chat(peer, model, key), lambda response: response.status_code == 200, seconds=PEER_PICKUP_SECONDS
        )
        assert scope_headers(served, "api_key")["limit-requests"] == "3", dict(served.headers)


def test_lowering_a_keys_rpm_on_one_instance_refuses_the_key_on_the_other(gateway: Gateway, tmp_path: Path) -> None:
    """A key with ``rpm_limit: 10`` is served twice by the peer. ``/key/update`` through the shared gateway lowers
    ``rpm_limit`` to 2; the peer's next request is refused 429 for ``api_key: <hash>`` with ``Current limit: 2``
    within seconds, instead of the peer enforcing the stale 10 until its key cache expires."""
    with (
        wire_server(provider()) as wire,
        gateway.scenario() as scenario,
        limits_proxy(gateway, tmp_path, shared=True) as peer,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        until_served(peer, model, peer.key)
        key: Final = scenario.key(models=[model], rpm_limit=10)
        assert statuses([chat(peer, model, key) for _ in range(2)]) == [200, 200]
        gateway.post("/key/update", {"key": key, "rpm_limit": 2})
        refused: Final = eventually(
            lambda: chat(peer, model, key), lambda response: response.status_code == 429, seconds=PEER_PICKUP_SECONDS
        )
        assert_limited(refused, "api_key", hashed(key), "requests", 2)


def test_raising_a_teams_rpm_on_one_instance_admits_its_key_on_the_other(gateway: Gateway, tmp_path: Path) -> None:
    """A team with ``rpm_limit: 1`` has a key that used its request on the shared gateway and is refused by the peer
    for ``team: <team_id>``. ``/team/update`` through the shared gateway raises it to 3 and the peer serves the key
    again within seconds."""
    with (
        wire_server(provider()) as wire,
        gateway.scenario() as scenario,
        limits_proxy(gateway, tmp_path, shared=True) as peer,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        until_served(peer, model, peer.key)
        team: Final = scenario.team(models=[model], rpm_limit=1)
        key: Final = scenario.key(team_id=team, models=[model])
        assert chat(gateway, model, key).status_code == 200
        assert_limited(chat(peer, model, key), "team", team, "requests", 1)
        gateway.post("/team/update", {"team_id": team, "rpm_limit": 3})
        served: Final = eventually(
            lambda: chat(peer, model, key), lambda response: response.status_code == 200, seconds=PEER_PICKUP_SECONDS
        )
        assert served.status_code == 200, served.text


def test_a_per_model_limit_added_on_one_instance_is_enforced_on_the_other(gateway: Gateway, tmp_path: Path) -> None:
    """A key without limits is served by the peer. ``/key/update`` through the shared gateway adds
    ``model_rpm_limit: {<model>: 1}``; within seconds the peer serves one more request and then refuses for
    ``model_per_key: <hash>:<model>`` with ``Current limit: 1``."""
    with (
        wire_server(provider()) as wire,
        gateway.scenario() as scenario,
        limits_proxy(gateway, tmp_path, shared=True) as peer,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        until_served(peer, model, peer.key)
        key: Final = scenario.key(models=[model])
        assert chat(peer, model, key).status_code == 200
        gateway.post("/key/update", {"key": key, "model_rpm_limit": {model: 1}})
        refused: Final = eventually(
            lambda: chat(peer, model, key), lambda response: response.status_code == 429, seconds=PEER_PICKUP_SECONDS
        )
        assert_limited(refused, "model_per_key", f"{hashed(key)}:{model}", "requests", 1)


def test_a_key_update_is_announced_on_the_auth_cache_invalidation_channel(gateway: Gateway) -> None:
    """A Redis client subscribes to ``litellm_proxy.auth_cache_invalidation`` (no cache namespace in the suite
    config). ``/key/update`` on the shared gateway publishes at least one message whose ``cache_key`` names the key's
    hash, which is how other instances learn to drop it. UNVERIFIED: the cache key format is assumed to contain the
    sha256 of the key (``user_api_key_cache`` stores keys under the hashed token)."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario, shared_redis() as redis:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=5)
        assert chat(gateway, model, key).status_code == 200
        subscription: Final = redis.pubsub()
        subscription.subscribe("litellm_proxy.auth_cache_invalidation")
        try:
            gateway.post("/key/update", {"key": key, "rpm_limit": 6})

            def heard() -> str:
                message = subscription.get_message(ignore_subscribe_messages=True, timeout=1.0)
                return "" if message is None else str(message.get("data"))

            announced: Final = eventually(heard, lambda data: hashed(key) in data, seconds=20)
            assert hashed(key) in announced
        finally:
            subscription.close()
