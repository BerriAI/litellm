"""Refresher suite.

Deliberately small. The live-proxy run in the PR body is the workflow evidence; what is here is the set of
behaviors whose regression would harm the customer's own traffic and would not be obvious from a replay
succeeding: warming must not double-charge a key's TPM, must not reset a key's rate-limit window, must not
collide the operator's hanging-request tracking, and must be refused by every ceiling the request path
enforces. The remaining two cover the shape of a replay on each surface and the two brakes on cost.
"""

import time
from datetime import datetime, timedelta, timezone

import pytest

from litellm.caching.dual_cache import DualCache
from litellm.router_strategy.complexity_router.cache_warming.refresher import filter_cache_warmable
from litellm.router_strategy.complexity_router.cache_warming.store import CacheWarmingStore
from litellm.router_strategy.complexity_router.cache_warming.types import (
    CACHE_WARMING_REPLAY_MARKER_KEY,
    CACHE_WARMING_REPLAY_TAG,
)

from tests.test_litellm.router_strategy.complexity_router.cache_warming.test_store import FakeRedisCache
from tests.test_litellm.router_strategy.complexity_router.cache_warming.warming_rig import (
    UNIFORM_POOL,
    FakeKeyDirectory,
    FakeLeaseLock,
    ReplayRouter,
    affinity_check,
    key_state,
    priced_rig,
    proxy_logging_with_hooks,
    real_limiter,
    refresher,
    registered_callbacks,
    replayed_models,
    seed_session,
    team,
    tick,
    warming_rig,
    warmth_stamp,
)

_PAST = (datetime.now(timezone.utc) - timedelta(hours=1)).replace(microsecond=0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arm,warmed",
    [
        ("healthy", True),
        ("blocked-key", False),
        ("expired-key", False),
        ("unparseable-expiry", False),
        ("blocked-team", False),
        ("team-model-denied", False),
        ("key-model-denied", False),
        ("key-model-over-budget", False),
        ("over-budget", False),
        ("rate-limited", False),
        ("scim-deactivated-owner", False),
    ],
)
async def test_every_ceiling_the_request_path_enforces_gates_warming(arm, warmed):
    """A replay is admitted through the request path's own entry points, so each of these refusals comes from
    that path's owner rather than a second implementation here: the blocked and expired key checks mirroring
    the canonical auth checks, then both of production's own authorization enumerations
    (_enforce_key_and_fallback_model_access for the key-level model allowlist, common_checks for team, member,
    user, project, every budget scope and the tool and vector-store allowlists), the budget reservation on the
    shared spend counters, and the v3 limiter's own counters. The key-model-denied arm is the one that proves
    warming runs the key-level enumeration and not only the team one; the unparseable-expiry arm is the
    fail-closed half of a value that stays a plain string on a cached key object.
    """
    from litellm.proxy.proxy_server import spend_counter_cache

    redis = FakeRedisCache()
    limiter, counters = real_limiter()
    key_cache = DualCache()
    fields: dict = {}
    token = "k"

    if arm == "blocked-key":
        fields = {"blocked": True}
    elif arm == "expired-key":
        fields = {"expires": _PAST.isoformat().replace("+00:00", "Z")}
    elif arm == "unparseable-expiry":
        fields = {"expires": "not-a-timestamp"}
    elif arm in ("blocked-team", "team-model-denied"):
        team_object = team("t", blocked=arm == "blocked-team", models=[] if arm == "blocked-team" else ["fast-claude"])
        await key_cache.async_set_cache(key="team_id:t", value=team_object)
        fields = {"team_id": "t"}
    elif arm == "key-model-denied":
        fields = {"models": ["fast-claude"]}
    elif arm == "key-model-over-budget":
        from litellm.proxy.hooks.model_max_budget_limiter import VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX
        from litellm.proxy.proxy_server import user_api_key_cache as proxy_key_cache

        fields = {"model_max_budget": {"smart-claude": {"budget_limit": 1.0, "time_period": "1d"}}}
        await proxy_key_cache.async_set_cache(
            key=f"{VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX}:{token}:smart-claude:1d", value=5.0
        )
    elif arm == "over-budget":
        fields = {"spend": 100.0, "max_budget": 100.0}
    elif arm == "rate-limited":
        fields = {"rpm_limit": 1}
        await counters.async_set_cache(key="{api_key:k}:window", value=str(int(time.time())))
        await counters.async_set_cache(key="{api_key:k}:requests", value=1)
    elif arm == "scim-deactivated-owner":
        from litellm.proxy._types import LiteLLM_UserTable

        await key_cache.async_set_cache(
            key="u",
            value=LiteLLM_UserTable(user_id="u", max_budget=None, spend=0.0, metadata={"scim_active": False}),
        )
        fields = {"user_id": "u"}

    priced = arm == "over-budget"
    llm_router = priced_rig(redis) if priced else warming_rig(redis=redis)[0]
    served = "claude-haiku-4-5" if priced else "fast-claude"
    target = "claude-sonnet-4-5" if priced else "smart-claude"
    counter_key = f"spend:key:{token}"
    if arm == "over-budget":
        await spend_counter_cache.async_set_cache(key=counter_key, value=100.0)
    seed_session(redis, user_api_key=token, served_model=served, warmth={served: time.time()})
    keys = FakeKeyDirectory({token: key_state(token=token, **fields)})
    try:
        with registered_callbacks(limiter):
            await tick(llm_router, active=refresher(keys=keys, limiter=limiter), user_api_key_cache=key_cache)
        assert (target in replayed_models(llm_router)) is warmed
    finally:
        spend_counter_cache.in_memory_cache.delete_cache(key=counter_key)


@pytest.mark.asyncio
async def test_warming_does_not_double_charge_a_keys_tpm():
    """Both halves of the limiter's TPM contract on one replay: pre_call_hook reserves upfront, and the
    limiter's own success callback finds that reservation through the request metadata the replay carried and
    settles the counter to actual usage. Without the stash riding along the counter ends at reservation plus
    actual, so the customer's next real request is throttled against tokens nobody used."""
    from datetime import datetime as _datetime

    from litellm.types.utils import ModelResponse, Usage

    limiter, counters = real_limiter()
    llm_router, redis = warming_rig(redis=FakeRedisCache())
    seed_session(redis, user_api_key="tpm", warmth={"fast-claude": time.time()})
    keys = FakeKeyDirectory({"tpm": key_state(token="tpm", tpm_limit=100_000)})
    with registered_callbacks(limiter):
        await tick(llm_router, active=refresher(keys=keys, limiter=limiter))
    tokens_key = limiter.create_rate_limit_keys("api_key", "tpm", "tokens")
    assert int(await counters.async_get_cache(key=tokens_key) or 0) > 0
    replay_metadata = llm_router.completion_calls[0]["metadata"]
    assert "_litellm_tpm_reserved_tokens" in replay_metadata
    await limiter.async_log_success_event(
        kwargs={"metadata": replay_metadata, "standard_logging_object": {"metadata": {"user_api_key_hash": "tpm"}}},
        response_obj=ModelResponse(usage=Usage(prompt_tokens=2000, completion_tokens=1, total_tokens=2001)),
        start_time=_datetime.now(),
        end_time=_datetime.now(),
    )
    assert int(await counters.async_get_cache(key=tokens_key)) == 2001


@pytest.mark.asyncio
async def test_warming_does_not_reset_a_keys_rate_limit_window():
    """The descriptor carries the limiter's own configured window, so an hour-long window opened two minutes
    ago keeps accumulating. A hardcoded 60 would read as expired and overwrite the shared counters,
    permanently un-enforcing the operator's real limit for that key."""
    limiter, counters = real_limiter(window_size=3600)
    assert limiter.window_size == 3600
    llm_router, redis = warming_rig(redis=FakeRedisCache())
    seed_session(redis, user_api_key="hourly", warmth={"fast-claude": time.time()})
    keys = FakeKeyDirectory({"hourly": key_state(token="hourly", rpm_limit=100)})
    opened_at = int(time.time()) - 120
    await counters.async_set_cache(key="{api_key:hourly}:window", value=str(opened_at))
    await counters.async_set_cache(key="{api_key:hourly}:requests", value=5)
    with registered_callbacks(limiter):
        await tick(llm_router, active=refresher(keys=keys, limiter=limiter))
    assert len(llm_router.completion_calls) == 1
    assert int(await counters.async_get_cache(key="{api_key:hourly}:window")) == opened_at
    assert int(await counters.async_get_cache(key="{api_key:hourly}:requests")) == 6


@pytest.mark.asyncio
async def test_warming_does_not_collide_hanging_request_tracking():
    """The hanging-request checker keys its cache on litellm_call_id and clears an entry only when a request
    status is recorded. Replays sharing the empty default would collide on one entry and alert the operator
    about a request that never existed, indefinitely."""
    proxy_logging = proxy_logging_with_hooks()
    proxy_logging.alerting = ["slack"]
    llm_router, redis = warming_rig(redis=FakeRedisCache())
    seed_session(redis)
    await tick(llm_router, active=refresher(proxy_logging=proxy_logging))
    call_ids = [call["litellm_call_id"] for call in llm_router.completion_calls]
    assert len(call_ids) == 2 and len(set(call_ids)) == 2
    for call_id in call_ids:
        assert (
            await proxy_logging.internal_usage_cache.async_get_cache(
                key=f"request_status:{call_id}", litellm_parent_otel_span=None, local_only=True
            )
            == "success"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("surface,channel", [("chat_completions", "metadata"), ("anthropic_messages", "litellm_metadata")])
async def test_a_due_session_is_replayed_on_its_own_surface_and_stamped_warm(surface, channel):
    """The replay reaches the surface it was captured from, through that surface's own metadata channel, with
    generation held at the floor and the session_id that lets deployment affinity pin the replay to the same
    member real traffic will hit. Anthropic invalidates a cached prefix when tool_choice changes, so it and
    the system block ride along there. The warmth stamp is what paces the next tick."""
    llm_router, redis = warming_rig(redis=FakeRedisCache())
    record_key = seed_session(
        redis, call_surface=surface, warmth={"fast-claude": time.time()}, tool_choice={"type": "auto"}
    )
    await tick(llm_router)
    calls = llm_router.anthropic_calls if surface == "anthropic_messages" else llm_router.completion_calls
    other = llm_router.completion_calls if surface == "anthropic_messages" else llm_router.anthropic_calls
    assert len(calls) == 1 and other == []
    call = calls[0]
    assert call["model"] == "smart-claude"
    assert call["max_tokens"] == 1 and call["stream"] is False
    assert call["tool_choice"] == {"type": "auto"}
    assert call[channel][CACHE_WARMING_REPLAY_MARKER_KEY] is True
    assert call[channel]["session_id"] == "sess-1"
    assert call[channel]["user_api_key"] == "hash-1"
    assert call[channel]["spend_logs_metadata"] == {CACHE_WARMING_REPLAY_TAG: "true"}
    assert "tags" not in call[channel]
    if surface == "anthropic_messages":
        assert call["system"] == "You are a policy assistant"
    stamp = warmth_stamp(redis, record_key, "smart-claude")
    assert stamp is not None and stamp.at > 0 and stamp.warmed is True


@pytest.mark.asyncio
async def test_a_replay_the_provider_rejected_is_stamped_cold_and_still_paced():
    """A stamp is the claim that the provider holds this session's prefix on that model, so a replay that
    failed must not write a warm one: the pick would then send real traffic to a cold model for the provider's
    whole cache TTL, which is the outcome warming exists to avoid. The attempt is still recorded, so a model
    failing every replay is retried on the refresh interval rather than on every tick."""
    llm_router, redis = warming_rig(redis=FakeRedisCache())
    llm_router.failing_message_marker = "deployment policy"
    record_key = seed_session(redis, warmth={"fast-claude": time.time()})
    await tick(llm_router)
    assert llm_router.completion_calls == [] and len(llm_router.failed_calls) == 1
    stamp = warmth_stamp(redis, record_key, "smart-claude")
    assert stamp is not None and stamp.at > 0 and stamp.warmed is False

    await tick(llm_router)
    assert len(llm_router.failed_calls) == 1


@pytest.mark.asyncio
async def test_a_replay_refused_by_admission_is_stamped_cold_rather_than_warm():
    """Same claim, for the replay that never reached the provider at all: a refusal from the request path's
    own ceilings leaves the cache exactly as cold as a provider failure does."""
    limiter, counters = real_limiter()
    llm_router, redis = warming_rig(redis=FakeRedisCache())
    record_key = seed_session(redis, user_api_key="k", warmth={"fast-claude": time.time()})
    keys = FakeKeyDirectory({"k": key_state(token="k", rpm_limit=1)})
    await counters.async_set_cache(key="{api_key:k}:window", value=str(int(time.time())))
    await counters.async_set_cache(key="{api_key:k}:requests", value=1)
    with registered_callbacks(limiter):
        await tick(llm_router, active=refresher(keys=keys, limiter=limiter))
    assert llm_router.completion_calls == []
    stamp = warmth_stamp(redis, record_key, "smart-claude")
    assert stamp is not None and stamp.warmed is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "last_activity_offset,warmth_offset,replayed",
    [(-601, None, []), (None, -10, []), (None, -300, ["smart-claude"])],
    ids=["idle-session-drops-out", "recently-warmed-model-waits", "stale-warmth-is-refreshed"],
)
async def test_session_pacing_bounds_how_often_warming_spends(last_activity_offset, warmth_offset, replayed):
    """Two independent brakes on cost: a session idle past its timeout stops being warmed at all, and a model
    warmed inside the refresh interval is not warmed again."""
    llm_router, redis = warming_rig(redis=FakeRedisCache())
    now = time.time()
    seed_session(
        redis,
        last_activity=now + last_activity_offset if last_activity_offset is not None else None,
        warmth={"fast-claude": now, "smart-claude": now + warmth_offset} if warmth_offset is not None else None,
    )
    await tick(llm_router)
    assert replayed_models(llm_router) == replayed


@pytest.mark.parametrize("with_affinity,warmable", [(False, False), (True, True)])
def test_a_multi_deployment_group_is_only_warmed_when_deployment_affinity_pins_it(with_affinity, warmable):
    """Warming a pool without affinity pays the cache-write premium against 1/N routing odds, which is worse
    than not warming, so such a group is skipped rather than degrading silently."""
    llm_router = ReplayRouter(model_list=UNIFORM_POOL)
    if with_affinity:
        llm_router.optional_callbacks = [affinity_check()]
    assert filter_cache_warmable(llm_router, ["uniform"]) == (("uniform",) if warmable else ())


@pytest.mark.asyncio
@pytest.mark.parametrize("unavailable", ["database", "key-cache", "rate-limiter", "lock-backend"])
async def test_no_session_is_warmed_while_any_enforcement_dependency_is_unavailable(unavailable, caplog):
    """Every ceiling warming claims to respect is enforced by one of these, so losing any single one turns
    admission into a no-op that still spends. They are checked together, once, before any session is
    considered, rather than as per-dependency arms that each fail open on their own path. The warning is
    asserted as well as the silence, because a replay that dies further down on a missing dependency also
    makes no call and would let a removed guard pass unnoticed."""
    import logging

    from litellm.proxy.utils import ProxyLogging
    from litellm.router_strategy.complexity_router.cache_warming.types import warn_once

    warn_once.cache_clear()
    llm_router, redis = warming_rig(redis=FakeRedisCache())
    seed_session(redis, user_api_key="k")
    with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
        if unavailable == "database":
            await tick(llm_router, prisma=None)
        elif unavailable == "key-cache":
            await refresher().run_tick(llm_router=llm_router, prisma_client=object(), user_api_key_cache=None)
        elif unavailable == "rate-limiter":
            await tick(llm_router, active=refresher(proxy_logging=ProxyLogging(user_api_key_cache=DualCache())))
        else:
            from litellm.proxy._experimental.mcp_server.outbound_credentials.redis_refresh_coordinator import (
                LockAcquisition,
            )

            await tick(llm_router, active=refresher(lock=FakeLeaseLock(acquisition=LockAcquisition.ERROR)))
    assert llm_router.completion_calls == []
    assert any("cannot enforce a replay's ceilings" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_warmth_is_not_shared_between_two_auto_routers_on_one_redis():
    """Warmth keys derive from the session's scoped identity, which includes the auto-router. Without that a
    second warming router reads the first one's stamps and skips replays it never made, so its tiers go
    cold while it believes them warm."""
    redis = FakeRedisCache()
    first, _ = warming_rig(redis=redis)
    seed_session(redis, warmth={"fast-claude": time.time(), "smart-claude": time.time()})
    await tick(first)
    assert first.completion_calls == []

    other = CacheWarmingStore(redis_cache=redis, auto_router_model_name="other-router")
    record = other.record_key("other-router", "hash-1", "sess-1")
    assert await other.get_warmth(record, ("fast-claude", "smart-claude")) == {}




@pytest.mark.asyncio
@pytest.mark.parametrize("team_blocked,warmed", [(True, False), (False, True)])
async def test_a_keyless_proxy_caller_is_authorized_through_its_reconstructed_tenancy(team_blocked, warmed):
    """A JWT caller has api_key None, so before this it fell through to the unattributed path and every tenant
    control was skipped. Its tenancy is recorded at capture, so the principal is rebuilt from it and the team
    gate binds: a blocked team stops the replays, an open team still warms."""
    llm_router, redis = warming_rig(redis=FakeRedisCache())
    key_cache = DualCache()
    await key_cache.async_set_cache(
        key="team_id:jwt-team", value=team("jwt-team", blocked=team_blocked, models=[])
    )
    seed_session(
        redis,
        user_api_key=None,
        caller_scope="jwt-user",
        team_id="jwt-team",
        warmth={"fast-claude": time.time()},
    )
    await tick(llm_router, active=refresher(keys=FakeKeyDirectory({})), user_api_key_cache=key_cache)
    assert bool(llm_router.completion_calls) is warmed
    if warmed:
        assert llm_router.completion_calls[0]["metadata"]["user_api_key_team_id"] == "jwt-team"


@pytest.mark.asyncio
async def test_a_direct_sdk_session_with_no_recorded_identity_still_warms_unattributed():
    """No proxy auth object means no tenancy to preserve, so warming stays unattributed as before."""
    llm_router, redis = warming_rig(redis=FakeRedisCache())
    seed_session(redis, user_api_key=None, caller_scope="unscoped", warmth={"fast-claude": time.time()})
    await tick(llm_router, active=refresher(keys=FakeKeyDirectory({})))
    assert replayed_models(llm_router) == ["smart-claude"]
    assert llm_router.completion_calls[0]["metadata"]["user_api_key_team_id"] is None


@pytest.mark.asyncio
async def test_a_replay_never_falls_back_to_a_group_warming_did_not_validate():
    """Eligibility, every-member cacheability, affinity and pricing are properties of the target GROUP, so a
    fallback would substitute a group carrying none of them and spend the customer's money warming nothing."""
    llm_router, redis = warming_rig(redis=FakeRedisCache())
    seed_session(redis)
    await tick(llm_router)
    assert llm_router.completion_calls, "expected a replay"
    assert all(call["disable_fallbacks"] is True for call in llm_router.completion_calls)


@pytest.mark.asyncio
async def test_the_concurrency_bound_bounds_decompressed_payloads_not_just_replays():
    """Every session is started at once, so anything a session materializes before acquiring its slot scales
    with max_sessions instead of with the concurrency setting. Payloads are held decompressed for the whole
    replay, and capture admits them up to eight times the compressed cap, so inflating above the semaphore let
    one tick hold a thousand of them. Pins both halves of the bound: replays in flight and payloads inflated."""
    from litellm.router_strategy.complexity_router.cache_warming import refresher as refresher_module

    llm_router, redis = warming_rig(redis=FakeRedisCache(), replay_delay=0.02)
    for index in range(6):
        seed_session(redis, session_id=f"sess-{index}", caller_scope=f"hash-{index}", user_api_key=f"hash-{index}")
    real_decompress = refresher_module.decompress_payload
    inflated_before_first_replay_completed: list[int] = []
    inflated = 0

    def counting_decompress(blob):
        nonlocal inflated
        inflated += 1
        if not llm_router.completion_calls:
            inflated_before_first_replay_completed.append(inflated)
        return real_decompress(blob)

    refresher_module.decompress_payload = counting_decompress
    try:
        await tick(llm_router, active=refresher(max_concurrent_replays=2))
    finally:
        refresher_module.decompress_payload = real_decompress

    assert len(llm_router.completion_calls) == 12, "every seeded session should warm both due models"
    assert llm_router.max_concurrent <= 2, "replays in flight must respect the bound"
    assert max(inflated_before_first_replay_completed) <= 2, "payloads inflated must respect the same bound"
