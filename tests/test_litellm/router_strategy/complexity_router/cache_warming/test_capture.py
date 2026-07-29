import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import litellm
from litellm.router_strategy.complexity_router.cache_warming.capture import (
    _warn_payload_too_large,
    _warn_privacy_gate_blocked,
)
from litellm.router_strategy.complexity_router.cache_warming.store import CacheWarmingStore
from litellm.router_strategy.complexity_router.cache_warming.types import (
    CACHE_WARMING_REPLAY_MARKER_KEY,
    decompress_payload,
)
from litellm.router_strategy.complexity_router.complexity_router import ComplexityRouter

from tests.test_litellm.router_strategy.complexity_router.cache_warming.test_store import FakeRedisCache

LONG_SYSTEM = "All deployment manifests must declare resource ceilings before rollout. " * 200

SESSIONS_KEY = "{cache_warm:v1:smart-router}:sessions"


@pytest.fixture(autouse=True)
def _prompt_retention_consent(monkeypatch):
    monkeypatch.setenv("STORE_PROMPTS_IN_SPEND_LOGS", "true")


def anthropic_messages(**kwargs: object) -> None:
    raise AssertionError("marker function; never called")


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


MESSAGES = [
    {"role": "system", "content": LONG_SYSTEM},
    {"role": "user", "content": "summarize rule 7"},
]


def _kwargs(**overrides: object) -> dict:
    base: dict = {
        "model": "smart-router",
        "metadata": {
            "session_id": "sess-1",
            "user_api_key_hash": "hash-1",
            "user_api_key": "hash-1",
            "user_api_key_team_id": "team-9",
        },
    }
    return {**base, **overrides}


def _stored_records(redis: FakeRedisCache) -> list[dict]:
    return [json.loads(value) for value in redis.hashes.get(SESSIONS_KEY, {}).values()]


























@pytest.mark.asyncio
async def test_second_turn_overwrites_payload_and_preserves_other_model_warmth():
    redis = FakeRedisCache()
    router = _complexity_router(redis)
    await router._capture_session(_kwargs(), MESSAGES, "claude-sonnet-4-5")
    key = CacheWarmingStore.record_key("smart-router", "hash-1", "sess-1")
    first = json.loads(redis.hashes[SESSIONS_KEY][key])
    redis.data[CacheWarmingStore.warmth_key(key, "gpt-5-mini")] = json.dumps(123.0)
    await router._capture_session(
        _kwargs(), MESSAGES + [{"role": "user", "content": "and rule 8?"}], "claude-sonnet-4-5"
    )
    second = json.loads(redis.hashes[SESSIONS_KEY][key])
    assert json.loads(redis.data[CacheWarmingStore.warmth_key(key, "gpt-5-mini")]) == 123.0
    assert json.loads(redis.data[CacheWarmingStore.warmth_key(key, "claude-sonnet-4-5")]) > 0
    assert second["payload_sha256"] != first["payload_sha256"]
































@pytest.mark.asyncio
@pytest.mark.parametrize("opt_out", ["no-retention-consent", "global-redaction", "per-request-redaction"])
async def test_capture_honors_every_form_of_the_operators_prompt_retention_policy(opt_out, monkeypatch):
    """Capture persists full prompts, so it requires the retention opt-in and respects message redaction in
    every form it can be expressed."""
    redis = FakeRedisCache()
    router = _complexity_router(redis)
    kwargs = _kwargs()
    _warn_privacy_gate_blocked.cache_clear()
    if opt_out == "no-retention-consent":
        monkeypatch.delenv("STORE_PROMPTS_IN_SPEND_LOGS", raising=False)
    elif opt_out == "global-redaction":
        monkeypatch.setattr(litellm, "turn_off_message_logging", True)
    else:
        kwargs["metadata"]["headers"] = {"x-litellm-enable-message-redaction": True}
    await router._capture_session(kwargs, MESSAGES, "claude-sonnet-4-5")
    assert redis.hashes.get(SESSIONS_KEY, {}) == {}


@pytest.mark.asyncio
async def test_warming_does_not_accept_a_forged_billing_identity():
    """Identity comes from the one proxy-stamped slot, marked by user_api_key_hash which a caller cannot
    inject because the proxy strips the user_api_key_ prefixed fields. A bare user_api_key survives that
    strip and the slots are read litellm_metadata first, so merging them would let a caller name the
    master-key alias and have the refresher skip its key-state verification entirely."""
    from litellm.constants import LITELLM_PROXY_MASTER_KEY_ALIAS

    redis = FakeRedisCache()
    router = _complexity_router(redis)
    kwargs = _kwargs(litellm_metadata={"session_id": "sess-1", "user_api_key": LITELLM_PROXY_MASTER_KEY_ALIAS})
    await router._capture_session(kwargs, MESSAGES, "claude-sonnet-4-5")
    record = _stored_records(redis)[0]
    assert record["attribution"]["user_api_key"] == "hash-1"
    assert record["attribution"]["user_api_key_hash"] == "hash-1"
