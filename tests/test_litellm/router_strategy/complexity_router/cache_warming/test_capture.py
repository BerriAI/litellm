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
    WarmthStamp,
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


def _logging_obj(kwargs: dict) -> object:
    """The real Logging object the proxy threads into deployment selection, built from the request the way
    function_setup builds it, so the dynamic params under test are the ones production would carry."""
    from datetime import datetime

    from litellm.litellm_core_utils.litellm_logging import Logging

    return Logging(
        model=str(kwargs.get("model")),
        messages=MESSAGES,
        stream=False,
        call_type="acompletion",
        start_time=datetime.now(),
        litellm_call_id="capture-test",
        function_id="capture-test",
        kwargs=kwargs,
    )


def _stored_records(redis: FakeRedisCache) -> list[dict]:
    return [json.loads(value) for value in redis.hashes.get(SESSIONS_KEY, {}).values()]


























@pytest.mark.asyncio
async def test_second_turn_overwrites_payload_and_preserves_other_model_warmth():
    redis = FakeRedisCache()
    router = _complexity_router(redis)
    await router._capture_session(_kwargs(), MESSAGES, "claude-sonnet-4-5")
    key = CacheWarmingStore.record_key("smart-router", "key:hash-1", "sess-1")
    first = json.loads(redis.hashes[SESSIONS_KEY][key])
    other_stamp = WarmthStamp(at=123.0, warmed=True).model_dump()
    redis.data[CacheWarmingStore.warmth_key(key, "gpt-5-mini")] = json.dumps(other_stamp)
    await router._capture_session(
        _kwargs(), MESSAGES + [{"role": "user", "content": "and rule 8?"}], "claude-sonnet-4-5"
    )
    second = json.loads(redis.hashes[SESSIONS_KEY][key])
    assert json.loads(redis.data[CacheWarmingStore.warmth_key(key, "gpt-5-mini")]) == other_stamp
    served_stamp = WarmthStamp.model_validate_json(redis.data[CacheWarmingStore.warmth_key(key, "claude-sonnet-4-5")])
    assert served_stamp.at > 0 and served_stamp.warmed is True
    assert second["payload_sha256"] != first["payload_sha256"]
































@pytest.mark.asyncio
@pytest.mark.parametrize(
    "opt_out",
    [
        "no-retention-consent",
        "global-redaction",
        "redaction-header",
        "per-request-body-root",
        "per-request-body-metadata",
        "per-request-body-litellm-metadata",
        None,
    ],
)
async def test_capture_honors_every_form_of_the_operators_prompt_retention_policy(opt_out, monkeypatch):
    """Capture persists full prompts, so it requires the retention opt-in and respects message redaction in
    every form it can be expressed. The per-request cases carry a real Logging object because the caller's own
    turn_off_message_logging is read from the dynamic params it owns rather than from the request body, so a
    stub would assert the wiring instead of the policy. The final case is the negative control: with no opt-out
    expressed anywhere, this same path must still capture, otherwise the gate proves nothing."""
    redis = FakeRedisCache()
    router = _complexity_router(redis)
    kwargs = _kwargs()
    _warn_privacy_gate_blocked.cache_clear()
    if opt_out == "no-retention-consent":
        monkeypatch.delenv("STORE_PROMPTS_IN_SPEND_LOGS", raising=False)
    elif opt_out == "global-redaction":
        monkeypatch.setattr(litellm, "turn_off_message_logging", True)
    elif opt_out == "redaction-header":
        kwargs["metadata"]["headers"] = {"x-litellm-enable-message-redaction": True}
    elif opt_out == "per-request-body-root":
        kwargs["turn_off_message_logging"] = True
    elif opt_out == "per-request-body-metadata":
        kwargs["metadata"]["turn_off_message_logging"] = True
    elif opt_out == "per-request-body-litellm-metadata":
        kwargs["litellm_metadata"] = {"session_id": "sess-1", "turn_off_message_logging": True}
    kwargs["litellm_logging_obj"] = _logging_obj(kwargs)
    await router._capture_session(kwargs, MESSAGES, "claude-sonnet-4-5")
    stored = redis.hashes.get(SESSIONS_KEY, {})
    assert stored == {} if opt_out is not None else stored != {}


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


def test_keyless_tenants_reusing_a_session_id_do_not_share_a_record():
    """JWT and other keyless proxy principals carry no key hash. Collapsing them onto one literal would merge
    distinct tenants that reuse a session id into a single record, so the last writer's payload and
    attribution would win and later replays could spend under the wrong team."""
    from litellm.litellm_core_utils.core_helpers import get_caller_scope

    tenant_a = {"litellm_metadata": {"session_id": "shared", "user_api_key_team_id": "team-a"}}
    tenant_b = {"litellm_metadata": {"session_id": "shared", "user_api_key_team_id": "team-b"}}
    assert get_caller_scope(tenant_a) != get_caller_scope(tenant_b)
    assert get_caller_scope({"metadata": {}}) == "unscoped"
    keyed = {"metadata": {"user_api_key_hash": "hash-1", "user_api_key_team_id": "team-a"}}
    assert get_caller_scope(keyed) == "key:hash-1"
