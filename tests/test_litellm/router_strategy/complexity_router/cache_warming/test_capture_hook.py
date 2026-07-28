import gc
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from litellm.router_strategy.complexity_router.cache_warming.capture_hook import (
    _WARMING_STRATEGIES,
    ComplexityCacheWarmingCaptureHook,
    _warn_payload_too_large,
    _warn_privacy_gate_blocked,
)
from litellm.router_strategy.complexity_router.cache_warming.store import CacheWarmingStore
from litellm.router_strategy.complexity_router.cache_warming.types import (
    CACHE_WARMING_MARKER_KEY,
    CACHE_WARMING_REPLAY_MARKER_KEY,
    decompress_payload,
)
from litellm.router_strategy.complexity_router.complexity_router import ComplexityRouter
from litellm.types.utils import CallTypes

from tests.test_litellm.router_strategy.complexity_router.cache_warming.test_store import FakeRedisCache

LONG_SYSTEM = "All deployment manifests must declare resource ceilings before rollout. " * 200


def _complexity_router(redis: FakeRedisCache | None, **cache_warming_overrides: object) -> ComplexityRouter:
    router_instance = MagicMock()
    router_instance.cache = SimpleNamespace(redis_cache=redis)
    return ComplexityRouter(
        model_name="smart-router",
        litellm_router_instance=router_instance,
        complexity_router_config={
            "tiers": {"SIMPLE": "gpt-5-mini", "COMPLEX": "claude-sonnet-4-5"},
            "cache_warming": {"enabled": True, **cache_warming_overrides},
        },
    )


def _hook(allow_privacy: bool = True) -> ComplexityCacheWarmingCaptureHook:
    return ComplexityCacheWarmingCaptureHook(privacy_gate=lambda _kwargs: allow_privacy)


def _kwargs(router: ComplexityRouter, **overrides: object) -> dict:
    base: dict = {
        "model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "messages": [
            {"role": "system", "content": LONG_SYSTEM},
            {"role": "user", "content": "summarize rule 7"},
        ],
        "metadata": {
            CACHE_WARMING_MARKER_KEY: {
                "auto_router_model_name": "smart-router",
                "routed_model": "claude-sonnet-4-5",
                "strategy_ref": router._cache_warming_ref,
            },
            "session_id": "sess-1",
            "user_api_key_hash": "hash-1",
            "user_api_key": "hash-1",
            "user_api_key_team_id": "team-9",
        },
    }
    return {**base, **overrides}


SESSIONS_KEY = "{cache_warm:v1:smart-router}:sessions"


def _stored_records(redis: FakeRedisCache) -> list[dict]:
    return [json.loads(value) for value in redis.hashes.get(SESSIONS_KEY, {}).values()]


@pytest.mark.asyncio
async def test_captures_whitelisted_fields_only_never_credentials():
    redis = FakeRedisCache()
    router = _complexity_router(redis)
    result = await _hook().async_pre_call_deployment_hook(
        _kwargs(router, api_key="sk-live-secret", litellm_params={"api_key": "sk-live-secret"}),
        CallTypes.acompletion,
    )
    assert result is None
    records = _stored_records(redis)
    assert len(records) == 1
    payload = decompress_payload(records[0]["payload_compressed"])
    assert payload.model == "claude-sonnet-4-5"
    assert payload.call_surface == "chat_completions"
    assert "sk-live-secret" not in json.dumps(records[0])


@pytest.mark.asyncio
async def test_skips_unstamped_request():
    redis = FakeRedisCache()
    _complexity_router(redis)
    await _hook().async_pre_call_deployment_hook(
        {"messages": [{"role": "user", "content": "x"}], "metadata": {"session_id": "s"}}, CallTypes.acompletion
    )
    assert redis.hashes.get(SESSIONS_KEY, {}) == {}


@pytest.mark.asyncio
async def test_skips_unknown_strategy_ref():
    redis = FakeRedisCache()
    router = _complexity_router(redis)
    kwargs = _kwargs(router)
    kwargs["metadata"][CACHE_WARMING_MARKER_KEY]["strategy_ref"] = "deadbeef"
    await _hook().async_pre_call_deployment_hook(kwargs, CallTypes.acompletion)
    assert redis.hashes.get(SESSIONS_KEY, {}) == {}


@pytest.mark.asyncio
async def test_skips_when_stamping_strategy_was_replaced_and_collected():
    redis = FakeRedisCache()
    router = _complexity_router(redis)
    kwargs = _kwargs(router)
    ref = router._cache_warming_ref
    del router
    gc.collect()
    assert ref is not None and _WARMING_STRATEGIES.get(ref) is None
    await _hook().async_pre_call_deployment_hook(kwargs, CallTypes.acompletion)
    assert redis.hashes.get(SESSIONS_KEY, {}) == {}


@pytest.mark.asyncio
async def test_skips_stamp_whose_name_mismatches_resolved_strategy():
    redis = FakeRedisCache()
    router = _complexity_router(redis)
    kwargs = _kwargs(router)
    kwargs["metadata"][CACHE_WARMING_MARKER_KEY]["auto_router_model_name"] = "other"
    await _hook().async_pre_call_deployment_hook(kwargs, CallTypes.acompletion)
    assert redis.hashes.get(SESSIONS_KEY, {}) == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("slot", ["metadata", "litellm_metadata"])
async def test_skips_replay_marker_in_either_metadata_slot(slot):
    redis = FakeRedisCache()
    router = _complexity_router(redis)
    kwargs = _kwargs(router)
    kwargs[slot] = {**kwargs.pop("metadata"), CACHE_WARMING_REPLAY_MARKER_KEY: True}
    await _hook().async_pre_call_deployment_hook(kwargs, CallTypes.acompletion)
    assert redis.hashes.get(SESSIONS_KEY, {}) == {}


@pytest.mark.asyncio
async def test_skips_when_no_session_id():
    redis = FakeRedisCache()
    router = _complexity_router(redis)
    kwargs = _kwargs(router)
    kwargs["metadata"].pop("session_id")
    await _hook().async_pre_call_deployment_hook(kwargs, CallTypes.acompletion)
    assert redis.hashes.get(SESSIONS_KEY, {}) == {}


@pytest.mark.asyncio
async def test_skips_unsupported_call_type():
    redis = FakeRedisCache()
    router = _complexity_router(redis)
    await _hook().async_pre_call_deployment_hook(_kwargs(router), CallTypes.aembedding)
    assert redis.hashes.get(SESSIONS_KEY, {}) == {}


@pytest.mark.asyncio
async def test_privacy_gate_off_blocks_capture_and_warns():
    _warn_privacy_gate_blocked.cache_clear()
    redis = FakeRedisCache()
    router = _complexity_router(redis)
    await _hook(allow_privacy=False).async_pre_call_deployment_hook(_kwargs(router), CallTypes.acompletion)
    assert redis.hashes.get(SESSIONS_KEY, {}) == {}


@pytest.mark.asyncio
async def test_captures_system_and_surface_for_anthropic_messages():
    redis = FakeRedisCache()
    router = _complexity_router(redis)
    kwargs = _kwargs(router, system=LONG_SYSTEM, messages=[{"role": "user", "content": "hi there"}])
    await _hook().async_pre_call_deployment_hook(kwargs, CallTypes.aanthropic_messages)
    records = _stored_records(redis)
    assert len(records) == 1
    payload = decompress_payload(records[0]["payload_compressed"])
    assert payload.call_surface == "anthropic_messages"
    assert payload.system == LONG_SYSTEM


@pytest.mark.asyncio
async def test_chat_surface_ignores_stray_system_kwarg():
    redis = FakeRedisCache()
    router = _complexity_router(redis)
    await _hook().async_pre_call_deployment_hook(_kwargs(router, system=LONG_SYSTEM), CallTypes.acompletion)
    payload = decompress_payload(_stored_records(redis)[0]["payload_compressed"])
    assert payload.system is None


@pytest.mark.asyncio
async def test_skips_below_min_prompt_cache_tokens():
    redis = FakeRedisCache()
    router = _complexity_router(redis)
    kwargs = _kwargs(router, messages=[{"role": "user", "content": "tiny"}])
    await _hook().async_pre_call_deployment_hook(kwargs, CallTypes.acompletion)
    assert redis.hashes.get(SESSIONS_KEY, {}) == {}


@pytest.mark.asyncio
async def test_skips_oversized_payload():
    _warn_payload_too_large.cache_clear()
    redis = FakeRedisCache()
    router = _complexity_router(redis, max_payload_bytes=64)
    await _hook().async_pre_call_deployment_hook(_kwargs(router), CallTypes.acompletion)
    assert redis.hashes.get(SESSIONS_KEY, {}) == {}


@pytest.mark.asyncio
async def test_captures_attribution_subset():
    redis = FakeRedisCache()
    router = _complexity_router(redis)
    await _hook().async_pre_call_deployment_hook(_kwargs(router), CallTypes.acompletion)
    record = _stored_records(redis)[0]
    assert record["attribution"]["user_api_key"] == "hash-1"
    assert record["attribution"]["user_api_key_team_id"] == "team-9"
    assert record["attribution"]["user_api_key_user_id"] is None


@pytest.mark.asyncio
async def test_returns_none_and_never_mutates_kwargs():
    redis = FakeRedisCache()
    router = _complexity_router(redis)
    kwargs = _kwargs(router)
    snapshot = json.dumps(kwargs, sort_keys=True, default=str)
    result = await _hook().async_pre_call_deployment_hook(kwargs, CallTypes.acompletion)
    assert result is None
    assert json.dumps(kwargs, sort_keys=True, default=str) == snapshot


@pytest.mark.asyncio
async def test_swallows_store_exceptions():
    class ExplodingRedis(FakeRedisCache):
        async def async_set_cache(self, key: str, value: object, **kwargs: object) -> None:
            raise RuntimeError("redis down")

    router = _complexity_router(ExplodingRedis())
    result = await _hook().async_pre_call_deployment_hook(_kwargs(router), CallTypes.acompletion)
    assert result is None


@pytest.mark.asyncio
async def test_same_name_routers_capture_only_via_their_own_strategy():
    redis_a = FakeRedisCache()
    redis_b = FakeRedisCache()
    router_a = _complexity_router(redis_a)
    router_b = _complexity_router(redis_b)
    await _hook().async_pre_call_deployment_hook(_kwargs(router_a), CallTypes.acompletion)
    assert len(_stored_records(redis_a)) == 1
    assert _stored_records(redis_b) == []
    await _hook().async_pre_call_deployment_hook(_kwargs(router_b), CallTypes.acompletion)
    assert len(_stored_records(redis_b)) == 1


@pytest.mark.asyncio
async def test_second_turn_overwrites_payload_and_preserves_other_model_warmth():
    redis = FakeRedisCache()
    router = _complexity_router(redis)
    hook = _hook()
    await hook.async_pre_call_deployment_hook(_kwargs(router), CallTypes.acompletion)
    key = CacheWarmingStore.record_key("smart-router", "hash-1", "sess-1")
    first = json.loads(redis.hashes[SESSIONS_KEY][key])
    redis.data[CacheWarmingStore.warmth_key(key, "gpt-5-mini")] = json.dumps(123.0)
    turn2 = _kwargs(router)
    turn2["messages"] = turn2["messages"] + [{"role": "user", "content": "and rule 8?"}]
    await hook.async_pre_call_deployment_hook(turn2, CallTypes.acompletion)
    second = json.loads(redis.hashes[SESSIONS_KEY][key])
    assert json.loads(redis.data[CacheWarmingStore.warmth_key(key, "gpt-5-mini")]) == 123.0
    assert json.loads(redis.data[CacheWarmingStore.warmth_key(key, "claude-sonnet-4-5")]) > 0
    assert second["payload_sha256"] != first["payload_sha256"]


@pytest.mark.asyncio
async def test_skips_highly_compressible_payload_on_uncompressed_bound():
    _warn_payload_too_large.cache_clear()
    redis = FakeRedisCache()
    router = _complexity_router(redis, max_payload_bytes=1024)
    kwargs = _kwargs(router, messages=[{"role": "user", "content": "a" * 20_000}])
    await _hook().async_pre_call_deployment_hook(kwargs, CallTypes.acompletion)
    assert redis.hashes.get(SESSIONS_KEY, {}) == {}


@pytest.mark.asyncio
async def test_global_message_redaction_blocks_capture():
    import litellm

    from litellm.router_strategy.complexity_router.cache_warming.capture_hook import _capture_allowed

    redis = FakeRedisCache()
    router = _complexity_router(redis)
    hook = ComplexityCacheWarmingCaptureHook(privacy_gate=_capture_allowed)
    previous = litellm.turn_off_message_logging
    litellm.turn_off_message_logging = True
    try:
        await hook.async_pre_call_deployment_hook(_kwargs(router), CallTypes.acompletion)
    finally:
        litellm.turn_off_message_logging = previous
    assert redis.hashes.get(SESSIONS_KEY, {}) == {}


@pytest.mark.asyncio
async def test_per_request_redaction_header_blocks_capture():
    from litellm.router_strategy.complexity_router.cache_warming.capture_hook import _capture_allowed

    redis = FakeRedisCache()
    router = _complexity_router(redis)
    hook = ComplexityCacheWarmingCaptureHook(privacy_gate=_capture_allowed)
    kwargs = _kwargs(router)
    kwargs["metadata"]["headers"] = {"x-litellm-enable-message-redaction": True}
    await hook.async_pre_call_deployment_hook(kwargs, CallTypes.acompletion)
    assert redis.hashes.get(SESSIONS_KEY, {}) == {}
