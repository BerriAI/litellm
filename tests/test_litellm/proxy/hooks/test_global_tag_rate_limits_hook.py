"""
Unit tests for the global-scope, model-independent tag rate limiter.
"""

import asyncio
import os
import time
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError
from starlette.requests import Request

import litellm
import litellm.proxy.common_request_processing
from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.global_tag_rate_limits_hook import (
    _bucket_key,
    _PROXY_GlobalTagRateLimitsHook,
)
from litellm.proxy.proxy_server import ProxyConfig
from litellm.proxy.route_llm_request import route_request
from litellm.proxy.utils import ProxyLogging
from litellm.router import Router
from litellm.types.router import TagRateLimitEntry


class TimeController:
    def __init__(self):
        self._current = datetime(2026, 1, 1, 0, 0, 0)

    def now(self) -> datetime:
        return self._current

    def advance(self, seconds: float) -> None:
        self._current += timedelta(seconds=seconds)


@pytest.fixture
def time_controller():
    return TimeController()


def _make_hook(time_controller: TimeController) -> _PROXY_GlobalTagRateLimitsHook:
    return _PROXY_GlobalTagRateLimitsHook(
        internal_usage_cache=DualCache(),
        time_provider=time_controller.now,
    )


def _key(alias: str | None = None, api_key: str = "hash") -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key=api_key, key_alias=alias)


def _data(tags: list[str], call_id: str = "call-1") -> dict:
    return {"metadata": {"tags": tags}, "litellm_call_id": call_id}


def _redis_hook(time_controller: TimeController):
    from litellm.caching.redis_cache import RedisCache

    redis_host = os.getenv("REDIS_HOST")
    redis_port = os.getenv("REDIS_PORT")
    if not redis_host or not redis_port:
        pytest.skip("Redis environment variables (REDIS_HOST, REDIS_PORT) not set")
    redis_cache = RedisCache(host=redis_host, port=int(redis_port), password=os.getenv("REDIS_PASSWORD"))
    dual_cache = DualCache(redis_cache=redis_cache)
    return _PROXY_GlobalTagRateLimitsHook(
        internal_usage_cache=dual_cache, time_provider=time_controller.now
    ), redis_cache


# ---------------------------------------------------------------------------
# No-op when unconfigured
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_op_when_no_config_set(time_controller, monkeypatch):
    monkeypatch.setattr(litellm, "global_tag_rate_limits", None)
    hook = _make_hook(time_controller)
    result = await hook.async_pre_call_hook(
        user_api_key_dict=_key(), cache=DualCache(), data=_data(["end_user_id:u1"]), call_type="completion"
    )
    assert result == _data(["end_user_id:u1"])


@pytest.mark.asyncio
async def test_malformed_config_raises_at_first_use(time_controller, monkeypatch):
    monkeypatch.setattr(
        litellm, "global_tag_rate_limits", {"dollar_limits": {"limits": [{"name": "bad", "limit": "not-a-number"}]}}
    )
    hook = _make_hook(time_controller)
    with pytest.raises(ValidationError):
        await hook.async_pre_call_hook(
            user_api_key_dict=_key(), cache=DualCache(), data=_data(["end_user_id:u1"]), call_type="completion"
        )


@pytest.mark.asyncio
async def test_admission_ignores_a_forged_empty_litellm_metadata_key(time_controller, monkeypatch):
    """
    get_metadata_variable_name_from_kwargs picks "litellm_metadata" whenever
    that key is merely present, regardless of its value. A caller adding an
    empty "litellm_metadata" alongside the real, populated "metadata" made
    admission read no tags at all, sailing past every configured limit.
    """
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "request_limits": {
                "limits": [{"name": "daily", "tag_id": "end_user_id", "limit": 1, "period_seconds": 86400}]
            }
        },
    )
    hook = _make_hook(time_controller)
    poisoned = {"metadata": {"tags": ["end_user_id:u1"]}, "litellm_metadata": {}}

    await hook.async_pre_call_hook(
        user_api_key_dict=_key(),
        cache=DualCache(),
        data={**poisoned, "litellm_call_id": "call-1"},
        call_type="completion",
    )
    with pytest.raises(ProxyRateLimitError):
        await hook.async_pre_call_hook(
            user_api_key_dict=_key(),
            cache=DualCache(),
            data={**poisoned, "litellm_call_id": "call-2"},
            call_type="completion",
        )


# ---------------------------------------------------------------------------
# Global scope: applies to every key by default
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_limit_shared_across_keys_by_default(time_controller, monkeypatch):
    """No apply_to_key_alias -> the entry is one shared bucket regardless of
    which key made the request."""
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "request_limits": {
                "limits": [{"name": "daily", "tag_id": "end_user_id", "limit": 1, "period_seconds": 86400}]
            }
        },
    )
    hook = _make_hook(time_controller)

    await hook.async_pre_call_hook(
        user_api_key_dict=_key(alias="key-a"), cache=DualCache(), data=_data(["end_user_id:u1"]), call_type="completion"
    )
    # A different key, identical tag value, and a distinct call_id (a
    # genuinely separate logical request, not a fallback retry of the same
    # one) -- must be rejected too, proving the bucket is genuinely shared,
    # not per-key by default.
    with pytest.raises(ProxyRateLimitError):
        await hook.async_pre_call_hook(
            user_api_key_dict=_key(alias="key-b"),
            cache=DualCache(),
            data=_data(["end_user_id:u1"], call_id="call-2"),
            call_type="completion",
        )


@pytest.mark.asyncio
async def test_request_limit_is_independent_of_model(time_controller, monkeypatch):
    """The hook never reads `data["model"]` for identity -- two different
    "models" (irrelevant to this hook) must still share the same bucket."""
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "request_limits": {
                "limits": [{"name": "daily", "tag_id": "end_user_id", "limit": 1, "period_seconds": 86400}]
            }
        },
    )
    hook = _make_hook(time_controller)

    data_model_a = {**_data(["end_user_id:u1"]), "model": "gpt-4o"}
    # A distinct call_id: this is a separate logical request, not the same
    # one retrying against a different model via _pre_call_with_fallbacks.
    data_model_b = {**_data(["end_user_id:u1"], call_id="call-2"), "model": "claude-3"}
    await hook.async_pre_call_hook(
        user_api_key_dict=_key(), cache=DualCache(), data=data_model_a, call_type="completion"
    )
    with pytest.raises(ProxyRateLimitError):
        await hook.async_pre_call_hook(
            user_api_key_dict=_key(), cache=DualCache(), data=data_model_b, call_type="completion"
        )


# ---------------------------------------------------------------------------
# apply_to_key_alias -- narrows which keys an entry applies to
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_to_key_alias_ignores_non_matching_keys(time_controller, monkeypatch):
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "request_limits": {
                "limits": [
                    {
                        "name": "daily",
                        "tag_id": "end_user_id",
                        "limit": 1,
                        "period_seconds": 86400,
                        "apply_to_key_alias": ["premium-key"],
                    }
                ]
            }
        },
    )
    hook = _make_hook(time_controller)

    for _ in range(3):
        result = await hook.async_pre_call_hook(
            user_api_key_dict=_key(alias="other-key"),
            cache=DualCache(),
            data=_data(["end_user_id:u1"]),
            call_type="completion",
        )
        assert result is not None


@pytest.mark.asyncio
async def test_apply_to_key_alias_enforces_for_the_listed_key(time_controller, monkeypatch):
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "request_limits": {
                "limits": [
                    {
                        "name": "daily",
                        "tag_id": "end_user_id",
                        "limit": 1,
                        "period_seconds": 86400,
                        "apply_to_key_alias": ["premium-key"],
                    }
                ]
            }
        },
    )
    hook = _make_hook(time_controller)

    await hook.async_pre_call_hook(
        user_api_key_dict=_key(alias="premium-key"),
        cache=DualCache(),
        data=_data(["end_user_id:u1"]),
        call_type="completion",
    )
    # A distinct call_id: a second, separate request from the same key, not
    # a fallback retry of the first.
    with pytest.raises(ProxyRateLimitError):
        await hook.async_pre_call_hook(
            user_api_key_dict=_key(alias="premium-key"),
            cache=DualCache(),
            data=_data(["end_user_id:u1"], call_id="call-2"),
            call_type="completion",
        )


@pytest.mark.asyncio
async def test_apply_to_key_alias_composes_with_scope_by_key_hash(time_controller, monkeypatch):
    """Both listed keys are subject to the entry, but scope_by_key_hash
    splits their buckets: exhausting one must not affect the other."""
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "request_limits": {
                "limits": [
                    {
                        "name": "daily",
                        "tag_id": "end_user_id",
                        "limit": 1,
                        "period_seconds": 86400,
                        "apply_to_key_alias": ["key-a", "key-b"],
                        "scope_by_key_hash": True,
                    }
                ]
            }
        },
    )
    hook = _make_hook(time_controller)

    await hook.async_pre_call_hook(
        user_api_key_dict=_key(alias="key-a", api_key="hashA"),
        cache=DualCache(),
        data=_data(["end_user_id:u1"]),
        call_type="completion",
    )
    # A distinct call_id: a second, separate request from the same key, not
    # a fallback retry of the first.
    with pytest.raises(ProxyRateLimitError):
        await hook.async_pre_call_hook(
            user_api_key_dict=_key(alias="key-a", api_key="hashA"),
            cache=DualCache(),
            data=_data(["end_user_id:u1"], call_id="call-2"),
            call_type="completion",
        )
    # key-b is unaffected by key-a's exhausted bucket.
    result = await hook.async_pre_call_hook(
        user_api_key_dict=_key(alias="key-b", api_key="hashB"),
        cache=DualCache(),
        data=_data(["end_user_id:u1"]),
        call_type="completion",
    )
    assert result is not None


# ---------------------------------------------------------------------------
# apply_to_models -- narrows which requested model an entry applies to,
# letting one entry rate-limit a whole fallback chain as a single unit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_to_models_ignores_non_matching_model(time_controller, monkeypatch):
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "request_limits": {
                "limits": [
                    {
                        "name": "chain_cap",
                        "tag_id": "end_user_id",
                        "limit": 1,
                        "period_seconds": 86400,
                        "apply_to_models": ["opus-chain"],
                    }
                ]
            }
        },
    )
    hook = _make_hook(time_controller)

    for i in range(3):
        data = {**_data(["end_user_id:u1"], call_id=f"call-{i}"), "model": "sonnet-chain"}
        result = await hook.async_pre_call_hook(
            user_api_key_dict=_key(), cache=DualCache(), data=data, call_type="completion"
        )
        assert result is not None


@pytest.mark.asyncio
async def test_apply_to_models_enforces_for_the_listed_model(time_controller, monkeypatch):
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "request_limits": {
                "limits": [
                    {
                        "name": "chain_cap",
                        "tag_id": "end_user_id",
                        "limit": 1,
                        "period_seconds": 86400,
                        "apply_to_models": ["opus-chain"],
                    }
                ]
            }
        },
    )
    hook = _make_hook(time_controller)

    data1 = {**_data(["end_user_id:u1"], call_id="call-1"), "model": "opus-chain"}
    await hook.async_pre_call_hook(user_api_key_dict=_key(), cache=DualCache(), data=data1, call_type="completion")
    data2 = {**_data(["end_user_id:u1"], call_id="call-2"), "model": "opus-chain"}
    with pytest.raises(ProxyRateLimitError):
        await hook.async_pre_call_hook(user_api_key_dict=_key(), cache=DualCache(), data=data2, call_type="completion")


@pytest.mark.asyncio
async def test_apply_to_models_shares_one_bucket_across_every_listed_model(time_controller, monkeypatch):
    """The core "rate limit the whole chain" use case: a single limit shared
    across every model named in apply_to_models, not one bucket per model."""
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "request_limits": {
                "limits": [
                    {
                        "name": "chain_cap",
                        "tag_id": "end_user_id",
                        "limit": 1,
                        "period_seconds": 86400,
                        "apply_to_models": ["opus-chain", "sonnet-chain"],
                    }
                ]
            }
        },
    )
    hook = _make_hook(time_controller)

    data1 = {**_data(["end_user_id:u1"], call_id="call-1"), "model": "opus-chain"}
    await hook.async_pre_call_hook(user_api_key_dict=_key(), cache=DualCache(), data=data1, call_type="completion")

    data2 = {**_data(["end_user_id:u1"], call_id="call-2"), "model": "sonnet-chain"}
    with pytest.raises(ProxyRateLimitError):
        await hook.async_pre_call_hook(user_api_key_dict=_key(), cache=DualCache(), data=data2, call_type="completion")


@pytest.mark.asyncio
async def test_apply_to_models_composes_with_apply_to_key_alias(time_controller, monkeypatch):
    """Both gates must pass -- the listed key requesting a non-listed model
    is unaffected, and only the listed key requesting the listed model is
    actually enforced."""
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "request_limits": {
                "limits": [
                    {
                        "name": "chain_cap",
                        "tag_id": "end_user_id",
                        "limit": 1,
                        "period_seconds": 86400,
                        "apply_to_models": ["opus-chain"],
                        "apply_to_key_alias": ["premium-key"],
                    }
                ]
            }
        },
    )
    hook = _make_hook(time_controller)

    # premium-key requesting a non-listed model: apply_to_models alone must
    # still exclude it, even though apply_to_key_alias matches.
    for i in range(3):
        data = {**_data(["end_user_id:u1"], call_id=f"wrong-model-{i}"), "model": "sonnet-chain"}
        result = await hook.async_pre_call_hook(
            user_api_key_dict=_key(alias="premium-key"), cache=DualCache(), data=data, call_type="completion"
        )
        assert result is not None

    # A non-listed key requesting the listed model: apply_to_key_alias alone
    # must still exclude it, even though apply_to_models matches.
    for i in range(3):
        data = {**_data(["end_user_id:u1"], call_id=f"wrong-key-{i}"), "model": "opus-chain"}
        result = await hook.async_pre_call_hook(
            user_api_key_dict=_key(alias="other-key"), cache=DualCache(), data=data, call_type="completion"
        )
        assert result is not None

    # Both gates match: enforced.
    data1 = {**_data(["end_user_id:u1"], call_id="call-1"), "model": "opus-chain"}
    await hook.async_pre_call_hook(
        user_api_key_dict=_key(alias="premium-key"), cache=DualCache(), data=data1, call_type="completion"
    )
    data2 = {**_data(["end_user_id:u1"], call_id="call-2"), "model": "opus-chain"}
    with pytest.raises(ProxyRateLimitError):
        await hook.async_pre_call_hook(
            user_api_key_dict=_key(alias="premium-key"), cache=DualCache(), data=data2, call_type="completion"
        )


@pytest.mark.asyncio
async def test_dollar_limit_respects_apply_to_models_at_accounting_time(time_controller, monkeypatch):
    """The entry only applies to opus-chain; a non-listed model's spend must
    not be charged against this bucket at all -- proves apply_to_models
    gates async_log_success_event's tokens/dollars accounting, not just
    admission."""
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "dollar_limits": {
                "limits": [
                    {
                        "name": "chain_spend",
                        "tag_id": "end_user_id",
                        "limit": 10.0,
                        "period_seconds": 86400,
                        "apply_to_models": ["opus-chain"],
                    }
                ]
            }
        },
    )
    hook = _make_hook(time_controller)

    data = {**_data(["end_user_id:u1"], call_id="call-1"), "model": "sonnet-chain"}
    await hook.async_pre_call_hook(user_api_key_dict=_key(), cache=DualCache(), data=data, call_type="completion")
    kwargs = {
        "litellm_call_id": "call-1",
        "metadata": {"tags": ["end_user_id:u1"]},
        "standard_logging_object": {"total_tokens": 0, "response_cost": 999.0},
    }
    await hook.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)
    await asyncio.sleep(0)

    # opus-chain was never charged -- still fully under its own limit.
    result = await hook.async_pre_call_hook(
        user_api_key_dict=_key(),
        cache=DualCache(),
        data={**_data(["end_user_id:u1"], call_id="call-2"), "model": "opus-chain"},
        call_type="completion",
    )
    assert result is not None


@pytest.mark.asyncio
async def test_apply_to_models_fallback_does_not_re_narrow_accounting_to_the_serving_model(
    time_controller, monkeypatch
):
    """
    Documented limitation, not a bug: apply_to_models is evaluated exactly
    once, at admission, against the caller-requested model -- it is never
    re-evaluated against whichever model a later fallback actually serves.
    This request names "opus-chain" at admission (the entry applies), but its
    response accounting reports "sonnet-chain" as the model that actually
    served it, simulating Router falling back after opus-chain failed. The
    spend must still land in the opus-chain-scoped bucket: the check already
    ran and decided at admission, and is not re-run for the fallback target.
    """
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "dollar_limits": {
                "limits": [
                    {
                        "name": "chain_spend",
                        "tag_id": "end_user_id",
                        "limit": 10.0,
                        "period_seconds": 86400,
                        "apply_to_models": ["opus-chain"],
                    }
                ]
            }
        },
    )
    hook = _make_hook(time_controller)

    data = {**_data(["end_user_id:u1"], call_id="call-1"), "model": "opus-chain"}
    await hook.async_pre_call_hook(user_api_key_dict=_key(), cache=DualCache(), data=data, call_type="completion")

    # The response accounting reports the fallback target as the model that
    # actually served the request -- not the "opus-chain" admission decided.
    kwargs = {
        "litellm_call_id": "call-1",
        "metadata": {"tags": ["end_user_id:u1"]},
        "model": "sonnet-chain",
        "standard_logging_object": {
            "total_tokens": 0,
            "response_cost": 12.0,
            "model": "sonnet-chain",
            "model_group": "sonnet-chain",
        },
    }
    await hook.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)
    await asyncio.sleep(0)

    # The spend landed in the opus-chain-scoped bucket regardless -- a fresh
    # opus-chain request is now over the limit.
    with pytest.raises(ProxyRateLimitError):
        await hook.async_pre_call_hook(
            user_api_key_dict=_key(),
            cache=DualCache(),
            data={**_data(["end_user_id:u1"], call_id="call-2"), "model": "opus-chain"},
            call_type="completion",
        )


@pytest.mark.asyncio
async def test_apply_to_models_accounts_when_a_fallback_retry_re_admits_with_a_different_model(
    time_controller, monkeypatch
):
    """
    Unlike the test above (one admission, a later Router-level fallback
    reported only at success time), this simulates _pre_call_with_fallbacks
    itself re-running async_pre_call_hook for the SAME call_id with a
    DIFFERENT model after some other hook rejected the original one. The
    first admission (opus-chain, in apply_to_models scope) must still get
    its success-time accounting even though the second admission
    (sonnet-chain, out of scope) is the one that actually proceeds --
    overwriting a single "last admitted model" field would silently drop
    the spend for the in-scope entry.
    """
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "dollar_limits": {
                "limits": [
                    {
                        "name": "chain_spend",
                        "tag_id": "end_user_id",
                        "limit": 10.0,
                        "period_seconds": 86400,
                        "apply_to_models": ["opus-chain"],
                    }
                ]
            }
        },
    )
    hook = _make_hook(time_controller)

    await hook.async_pre_call_hook(
        user_api_key_dict=_key(),
        cache=DualCache(),
        data={**_data(["end_user_id:u1"], call_id="call-1"), "model": "opus-chain"},
        call_type="completion",
    )
    # _pre_call_with_fallbacks retry: same call_id, fallback model outside apply_to_models.
    await hook.async_pre_call_hook(
        user_api_key_dict=_key(),
        cache=DualCache(),
        data={**_data(["end_user_id:u1"], call_id="call-1"), "model": "sonnet-chain"},
        call_type="completion",
    )

    kwargs = {
        "litellm_call_id": "call-1",
        "metadata": {"tags": ["end_user_id:u1"]},
        "model": "sonnet-chain",
        "standard_logging_object": {
            "total_tokens": 0,
            "response_cost": 12.0,
            "model": "sonnet-chain",
            "model_group": "sonnet-chain",
        },
    }
    await hook.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)
    await asyncio.sleep(0)

    # The $12 spend landed in the opus-chain-scoped bucket, so a fresh
    # opus-chain request is now over the $10 limit and rejected.
    with pytest.raises(ProxyRateLimitError):
        await hook.async_pre_call_hook(
            user_api_key_dict=_key(),
            cache=DualCache(),
            data={**_data(["end_user_id:u1"], call_id="call-2"), "model": "opus-chain"},
            call_type="completion",
        )


@pytest.mark.asyncio
async def test_apply_to_models_enforces_for_a_comma_separated_batch_model(time_controller, monkeypatch):
    """A caller can dodge a chain-wide apply_to_models cap by routing through
    Router.abatch_completion's comma-separated `model` -- admission must
    check the individual names, not the raw joined string, which would
    never match any configured apply_to_models entry."""
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "concurrency_limits": {
                "limits": [
                    {
                        "name": "chain_cap",
                        "tag_id": "end_user_id",
                        "limit": 1,
                        "period_seconds": 60,
                        "apply_to_models": ["opus-chain"],
                    }
                ]
            }
        },
    )
    hook = _make_hook(time_controller)

    await hook.async_pre_call_hook(
        user_api_key_dict=_key(),
        cache=DualCache(),
        data={**_data(["end_user_id:u1"], call_id="call-1"), "model": "opus-chain"},
        call_type="acompletion",
    )
    with pytest.raises(ProxyRateLimitError):
        await hook.async_pre_call_hook(
            user_api_key_dict=_key(),
            cache=DualCache(),
            data={**_data(["end_user_id:u1"], call_id="call-2"), "model": "opus-chain,other-model"},
            call_type="acompletion",
        )


@pytest.mark.asyncio
async def test_apply_to_models_accounts_dollar_spend_from_a_batch_admission(time_controller, monkeypatch):
    """Same gap as above, at success-time accounting: the batch admission's
    own admitted_models must record each individual name, not the joined
    string, or the spend never lands in the chain-scoped bucket at all."""
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "dollar_limits": {
                "limits": [
                    {
                        "name": "chain_spend",
                        "tag_id": "end_user_id",
                        "limit": 10.0,
                        "period_seconds": 86400,
                        "apply_to_models": ["opus-chain"],
                    }
                ]
            }
        },
    )
    hook = _make_hook(time_controller)

    await hook.async_pre_call_hook(
        user_api_key_dict=_key(),
        cache=DualCache(),
        data={**_data(["end_user_id:u1"], call_id="call-1"), "model": "opus-chain,other-model"},
        call_type="acompletion",
    )
    kwargs = {
        "litellm_call_id": "call-1",
        "metadata": {"tags": ["end_user_id:u1"]},
        "standard_logging_object": {"total_tokens": 0, "response_cost": 12.0},
    }
    await hook.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)
    await asyncio.sleep(0)

    with pytest.raises(ProxyRateLimitError):
        await hook.async_pre_call_hook(
            user_api_key_dict=_key(),
            cache=DualCache(),
            data={**_data(["end_user_id:u1"], call_id="call-2"), "model": "opus-chain"},
            call_type="acompletion",
        )


@pytest.mark.asyncio
async def test_apply_to_models_batch_split_requires_the_acompletion_call_type(time_controller, monkeypatch):
    """route_llm_request.py only splits a comma-separated model for
    call_type acompletion -- any other call type never reaches
    Router.abatch_completion, so a comma there is just a literal (if odd)
    model name and must not be split."""
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "concurrency_limits": {
                "limits": [
                    {
                        "name": "chain_cap",
                        "tag_id": "end_user_id",
                        "limit": 1,
                        "period_seconds": 60,
                        "apply_to_models": ["opus-chain"],
                    }
                ]
            }
        },
    )
    hook = _make_hook(time_controller)

    result = await hook.async_pre_call_hook(
        user_api_key_dict=_key(),
        cache=DualCache(),
        data={**_data(["end_user_id:u1"], call_id="call-1"), "model": "opus-chain,other-model"},
        call_type="embedding",
    )
    assert result is not None


@pytest.mark.asyncio
async def test_an_out_of_scope_fallback_retry_does_not_move_accounting_into_a_later_bucket(
    time_controller, monkeypatch
):
    """
    veria-ai finding: async_pre_call_hook overwrote stash.admission_time on
    every attempt, including an out-of-scope fallback retry that never
    matched any entry. If that retry lands in a different period bucket
    than the original in-scope attempt (e.g. straddling a period rollover),
    success-time accounting used the retry's timestamp to pick the bucket,
    charging spend against a bucket admission never actually checked and
    letting a caller dodge the limit by timing the retry across a rollover.
    """
    entry = TagRateLimitEntry(
        name="chain_spend",
        tag_id="end_user_id",
        limit=10.0,
        period_seconds=60,
        apply_to_models=("opus-chain",),
    )
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {"dollar_limits": {"limits": [entry.model_dump()]}},
    )
    hook = _make_hook(time_controller)
    admission_bucket_id = int(time_controller.now().timestamp()) // 60

    await hook.async_pre_call_hook(
        user_api_key_dict=_key(),
        cache=DualCache(),
        data={**_data(["end_user_id:u1"], call_id="call-1"), "model": "opus-chain"},
        call_type="completion",
    )
    # Crosses into the next 60s bucket before the out-of-scope fallback retry.
    time_controller.advance(65)
    retry_bucket_id = int(time_controller.now().timestamp()) // 60
    assert retry_bucket_id != admission_bucket_id
    await hook.async_pre_call_hook(
        user_api_key_dict=_key(),
        cache=DualCache(),
        data={**_data(["end_user_id:u1"], call_id="call-1"), "model": "sonnet-chain"},
        call_type="completion",
    )

    kwargs = {
        "litellm_call_id": "call-1",
        "metadata": {"tags": ["end_user_id:u1"]},
        "model": "sonnet-chain",
        "standard_logging_object": {
            "total_tokens": 0,
            "response_cost": 12.0,
            "model": "sonnet-chain",
            "model_group": "sonnet-chain",
        },
    }
    await hook.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)
    await asyncio.sleep(0)

    # The spend must have landed in the bucket the original, in-scope
    # admission actually checked, not the fresh bucket opened by the later,
    # out-of-scope retry.
    admission_key = _bucket_key(entry, "dollars", "u1", admission_bucket_id, key_hash=None)
    retry_key = _bucket_key(entry, "dollars", "u1", retry_bucket_id, key_hash=None)
    admission_bucket_value = await hook.internal_usage_cache.async_get_cache(
        key=admission_key, litellm_parent_otel_span=None
    )
    retry_bucket_value = await hook.internal_usage_cache.async_get_cache(key=retry_key, litellm_parent_otel_span=None)
    assert float(admission_bucket_value) == 12.0
    assert retry_bucket_value is None


@pytest.mark.asyncio
async def test_a_rejected_admission_attempts_model_does_not_drive_later_accounting(time_controller, monkeypatch):
    """
    A fallback retry's FIRST attempt can itself be rejected (by this same
    entry, or a different hook) before it ever admits. That rejected
    attempt's model must not join the stash's admitted-models history: a
    later, successful attempt against a different (out-of-scope) model must
    not have its accounting wrongly credited to an apply_to_models entry
    that never actually admitted this request.
    """
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "concurrency_limits": {
                "limits": [
                    {
                        "name": "conc-a",
                        "tag_id": "end_user_id",
                        "limit": 1,
                        "period_seconds": 60,
                        "apply_to_models": ["model-a"],
                    }
                ]
            },
            "dollar_limits": {
                "limits": [
                    {
                        "name": "chain_spend",
                        "tag_id": "end_user_id",
                        "limit": 10.0,
                        "period_seconds": 86400,
                        "apply_to_models": ["model-a"],
                    }
                ]
            },
        },
    )
    hook = _make_hook(time_controller)

    # Occupy model-a's only concurrency slot with an unrelated call.
    await hook.async_pre_call_hook(
        user_api_key_dict=_key(),
        cache=DualCache(),
        data={**_data(["end_user_id:u1"], call_id="occupier"), "model": "model-a"},
        call_type="completion",
    )

    # call-1 attempt #1: model-a, rejected (slot taken) -- never truly admitted.
    with pytest.raises(ProxyRateLimitError):
        await hook.async_pre_call_hook(
            user_api_key_dict=_key(),
            cache=DualCache(),
            data={**_data(["end_user_id:u1"], call_id="call-1"), "model": "model-a"},
            call_type="completion",
        )
    # call-1 attempt #2 (fallback retry): model-b, not in apply_to_models=[model-a], admits.
    await hook.async_pre_call_hook(
        user_api_key_dict=_key(),
        cache=DualCache(),
        data={**_data(["end_user_id:u1"], call_id="call-1"), "model": "model-b"},
        call_type="completion",
    )

    kwargs = {
        "litellm_call_id": "call-1",
        "metadata": {"tags": ["end_user_id:u1"]},
        "model": "model-b",
        "standard_logging_object": {
            "total_tokens": 0,
            "response_cost": 50.0,
            "model": "model-b",
            "model_group": "model-b",
        },
    }
    await hook.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)
    await asyncio.sleep(0)

    # Release the occupier's concurrency slot so the final check below is
    # gated only by chain_spend (dollars), not by conc-a still being full.
    await hook.async_log_success_event(
        kwargs={"litellm_call_id": "occupier", "metadata": {"tags": ["end_user_id:u1"]}},
        response_obj=None,
        start_time=0,
        end_time=0,
    )
    await asyncio.sleep(0)

    # chain_spend (apply_to_models=[model-a]) must still be empty: model-a's
    # own admission attempt was rejected, never admitted, so a fresh
    # model-a request is still allowed under the $10 limit.
    result = await hook.async_pre_call_hook(
        user_api_key_dict=_key(),
        cache=DualCache(),
        data={**_data(["end_user_id:u1"], call_id="call-2"), "model": "model-a"},
        call_type="completion",
    )
    assert result is not None


# ---------------------------------------------------------------------------
# _pre_call_with_fallbacks reruns admission for the same logical request:
# a repeat call_id must renew, not double-charge -- veria-ai finding on
# PR #36541
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repeat_admission_for_the_same_call_id_and_key_renews_instead_of_double_charging(
    time_controller, monkeypatch
):
    """
    ProxyBaseLLMRequestProcessing._pre_call_with_fallbacks reruns the whole
    pre-call pipeline (this hook included) once per fallback model on ANY
    ProxyRateLimitError, not only one this hook itself raised, but keeps the
    same litellm_call_id across every attempt (self.data is mutated in
    place; only "model" changes). Without this fix, an unrelated rejection
    (a different rate limiter, a budget cap) triggering N fallback attempts
    would charge this hook's own "requests" cap N times for one logical
    client call. A limit of 1 makes a double-charge directly observable: if
    the second admission (same call_id, same key) charged again instead of
    renewing, this would raise.
    """
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "request_limits": {
                "limits": [{"name": "daily", "tag_id": "end_user_id", "limit": 1, "period_seconds": 86400}]
            }
        },
    )
    hook = _make_hook(time_controller)
    key = _key(alias="key-a")

    await hook.async_pre_call_hook(
        user_api_key_dict=key, cache=DualCache(), data=_data(["end_user_id:u1"]), call_type="completion"
    )
    # Same call_id, same key, different model -- exactly what
    # _pre_call_with_fallbacks produces for a fallback attempt of the same
    # logical request.
    result = await hook.async_pre_call_hook(
        user_api_key_dict=key,
        cache=DualCache(),
        data={**_data(["end_user_id:u1"]), "model": "fallback-model"},
        call_type="completion",
    )
    assert result is not None


@pytest.mark.asyncio
async def test_a_forged_shared_call_id_from_a_different_key_does_not_get_a_free_renewal(time_controller, monkeypatch):
    """
    Security regression: litellm_call_id is caller-controlled via the
    x-litellm-call-id header (the same forgery vector
    model_based_tag_rate_limits_hook's own pending-reservations mirror was
    hardened against earlier in this PR). Two unrelated requests choosing
    the identical call_id must not be able to renew each other's charge --
    only a second admission carrying the SAME authenticated key_hash as
    whichever request first claimed that call_id may. A limit of 1 makes
    this observable: if the second, different-key admission wrongly
    renewed, it would succeed instead of raising.
    """
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "request_limits": {
                "limits": [{"name": "daily", "tag_id": "end_user_id", "limit": 1, "period_seconds": 86400}]
            }
        },
    )
    hook = _make_hook(time_controller)

    await hook.async_pre_call_hook(
        user_api_key_dict=_key(alias="key-a", api_key="hashA"),
        cache=DualCache(),
        data=_data(["end_user_id:u1"], call_id="forged-call-id"),
        call_type="completion",
    )
    with pytest.raises(ProxyRateLimitError):
        await hook.async_pre_call_hook(
            user_api_key_dict=_key(alias="key-b", api_key="hashB"),
            cache=DualCache(),
            data=_data(["end_user_id:u1"], call_id="forged-call-id"),
            call_type="completion",
        )


# ---------------------------------------------------------------------------
# Concurrency: reservation at admission, release on success/failure/disconnect
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrency_limit_rejects_second_admission_until_release(time_controller, monkeypatch):
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "concurrency_limits": {
                "limits": [{"name": "conc", "tag_id": "end_user_id", "limit": 1, "period_seconds": 60}]
            }
        },
    )
    hook = _make_hook(time_controller)

    await hook.async_pre_call_hook(
        user_api_key_dict=_key(),
        cache=DualCache(),
        data=_data(["end_user_id:u1"], call_id="call-1"),
        call_type="completion",
    )
    with pytest.raises(ProxyRateLimitError):
        await hook.async_pre_call_hook(
            user_api_key_dict=_key(),
            cache=DualCache(),
            data=_data(["end_user_id:u1"], call_id="call-2"),
            call_type="completion",
        )


@pytest.mark.asyncio
async def test_concurrency_reservation_released_on_success(time_controller, monkeypatch):
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "concurrency_limits": {
                "limits": [{"name": "conc", "tag_id": "end_user_id", "limit": 1, "period_seconds": 60}]
            }
        },
    )
    hook = _make_hook(time_controller)

    async def one_request(call_id: str) -> None:
        data = _data(["end_user_id:u1"], call_id=call_id)
        await hook.async_pre_call_hook(user_api_key_dict=_key(), cache=DualCache(), data=data, call_type="completion")
        kwargs = {"litellm_call_id": call_id, "metadata": {"tags": ["end_user_id:u1"]}}
        await hook.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)

    await one_request("call-1")
    await asyncio.sleep(0)  # let the fire-and-forget release task run

    # The slot was released, so a fresh request must be admitted again.
    result = await hook.async_pre_call_hook(
        user_api_key_dict=_key(),
        cache=DualCache(),
        data=_data(["end_user_id:u1"], call_id="call-2"),
        call_type="completion",
    )
    assert result is not None


@pytest.mark.asyncio
async def test_nested_call_success_does_not_release_the_outer_calls_reservation(time_controller, monkeypatch):
    """
    A nested LiteLLM call made inside the request (an LLM-judge guardrail, a
    silent experiment) mints its own fresh litellm_call_id but inherits the
    same ContextVar-held stash as the outer call, since it runs in the same
    task rather than a separate one. The outer call's own concurrency
    reservation must survive the nested call's admission and success
    callback: it belongs to a different call id and must not be touched by
    it.
    """
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "concurrency_limits": {
                "limits": [{"name": "conc", "tag_id": "end_user_id", "limit": 1, "period_seconds": 60}]
            }
        },
    )
    hook = _make_hook(time_controller)

    await hook.async_pre_call_hook(
        user_api_key_dict=_key(),
        cache=DualCache(),
        data=_data(["end_user_id:u1"], call_id="call-outer"),
        call_type="completion",
    )

    # Nested call, same task, different tag and a fresh call id -- admits
    # and completes entirely before the outer call's own success/failure
    # callback ever fires.
    await hook.async_pre_call_hook(
        user_api_key_dict=_key(),
        cache=DualCache(),
        data=_data(["end_user_id:u2"], call_id="call-nested"),
        call_type="completion",
    )
    await hook.async_log_success_event(
        kwargs={"litellm_call_id": "call-nested", "metadata": {"tags": ["end_user_id:u2"]}},
        response_obj=None,
        start_time=0,
        end_time=0,
    )
    await asyncio.sleep(0)

    # The outer call is still genuinely in flight -- its own reservation
    # must still be held, so a second end_user_id:u1 request is rejected.
    with pytest.raises(ProxyRateLimitError):
        await hook.async_pre_call_hook(
            user_api_key_dict=_key(),
            cache=DualCache(),
            data=_data(["end_user_id:u1"], call_id="call-outer-2"),
            call_type="completion",
        )


@pytest.mark.asyncio
async def test_concurrency_reservation_released_when_a_different_hook_rejects_the_request(time_controller, monkeypatch):
    """
    model_based_tag_rate_limits_hook raises the identical ProxyRateLimitError
    shape (detail["error"] == "tag_rate_limit_exceeded") this hook's own
    admission does, since both hooks share the same rejection marker.
    async_log_failure_event fires on every registered CustomLogger regardless
    of which one raised, so this hook must still release its own successfully
    reserved concurrency slot when the *other* hook is what rejected the
    request -- skipping release just because the marker matches would leak
    this hook's own slot until the safety TTL, even though nothing about this
    hook's own admission failed.
    """
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "concurrency_limits": {
                "limits": [{"name": "conc", "tag_id": "end_user_id", "limit": 1, "period_seconds": 60}]
            }
        },
    )
    hook = _make_hook(time_controller)

    data = _data(["end_user_id:u1"], call_id="call-1")
    await hook.async_pre_call_hook(user_api_key_dict=_key(), cache=DualCache(), data=data, call_type="completion")

    other_hooks_rejection = ProxyRateLimitError(
        detail={"error": "tag_rate_limit_exceeded", "type": "requests", "tag_id": "end_user_id"},
        headers={"retry-after": "60"},
        rate_limit_type=None,
        model="gpt-4o",
        llm_provider="litellm_proxy",
    )
    kwargs = {"litellm_call_id": "call-1", "exception": other_hooks_rejection}
    await hook.async_log_failure_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)

    # The slot was released despite the shared rejection marker, so a fresh
    # request must be admitted again.
    result = await hook.async_pre_call_hook(
        user_api_key_dict=_key(),
        cache=DualCache(),
        data=_data(["end_user_id:u1"], call_id="call-2"),
        call_type="completion",
    )
    assert result is not None


@pytest.mark.asyncio
async def test_concurrency_reservation_released_on_disconnect(time_controller, monkeypatch):
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "concurrency_limits": {
                "limits": [{"name": "conc", "tag_id": "end_user_id", "limit": 1, "period_seconds": 60}]
            }
        },
    )
    hook = _make_hook(time_controller)

    data = _data(["end_user_id:u1"], call_id="call-1")
    await hook.async_pre_call_hook(user_api_key_dict=_key(), cache=DualCache(), data=data, call_type="completion")
    await hook.async_release_disconnect_state_hook({"litellm_call_id": "call-1"})

    result = await hook.async_pre_call_hook(
        user_api_key_dict=_key(),
        cache=DualCache(),
        data=_data(["end_user_id:u1"], call_id="call-2"),
        call_type="completion",
    )
    assert result is not None


@pytest.mark.asyncio
async def test_concurrency_reservation_released_when_every_fallback_is_exhausted(time_controller, monkeypatch):
    """
    When _pre_call_with_fallbacks exhausts every fallback model (another
    hook rejects each one) and re-raises, the request never reaches the
    real LLM call -- neither async_log_success_event nor
    async_log_failure_event, both tied to that call's own wrapper, ever
    fires. async_post_call_failure_hook is the only remaining release path
    for a reservation this hook already admitted earlier in that same
    fallback chain.
    """
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "concurrency_limits": {
                "limits": [{"name": "conc", "tag_id": "end_user_id", "limit": 1, "period_seconds": 60}]
            }
        },
    )
    hook = _make_hook(time_controller)

    # This hook admits and reserves the only slot.
    await hook.async_pre_call_hook(
        user_api_key_dict=_key(),
        cache=DualCache(),
        data={**_data(["end_user_id:u1"], call_id="call-1"), "model": "model-a"},
        call_type="completion",
    )
    # _pre_call_with_fallbacks eventually gives up (every fallback rejected
    # by some other hook) and reports the failure via post_call_failure_hook.
    await hook.async_post_call_failure_hook(
        request_data={"litellm_call_id": "call-1"},
        original_exception=ProxyRateLimitError(
            detail={"error": "some_other_hooks_limit"}, headers={}, rate_limit_type=None
        ),
        user_api_key_dict=_key(),
    )

    result = await hook.async_pre_call_hook(
        user_api_key_dict=_key(),
        cache=DualCache(),
        data={**_data(["end_user_id:u1"], call_id="call-2"), "model": "model-a"},
        call_type="completion",
    )
    assert result is not None


@pytest.mark.asyncio
async def test_concurrent_requests_do_not_share_each_others_reservation_state(time_controller, monkeypatch):
    """Two logically distinct requests running as separate asyncio Tasks must
    not see each other's pending-concurrency stash, even though both share
    this hook instance -- the whole point of the ContextVar-based stash."""
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "concurrency_limits": {
                "limits": [{"name": "conc", "tag_id": "end_user_id", "limit": 5, "period_seconds": 60}]
            }
        },
    )
    hook = _make_hook(time_controller)

    async def one_request(call_id: str) -> int:
        data = _data(["end_user_id:u1"], call_id=call_id)
        await hook.async_pre_call_hook(user_api_key_dict=_key(), cache=DualCache(), data=data, call_type="completion")
        kwargs = {"litellm_call_id": call_id, "metadata": {"tags": ["end_user_id:u1"]}}
        await hook.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)
        return 1

    results = await asyncio.gather(one_request("call-a"), one_request("call-b"))
    await asyncio.sleep(0)
    assert results == [1, 1]


# ---------------------------------------------------------------------------
# Batch dispatch (Router.abatch_completion/abatch_completion_fastest_response
# via a comma-separated `model`): every branch genuinely runs concurrently
# as a real, independent LLM call below this hook entirely (it never re-runs
# async_pre_call_hook per branch the way model_based_tag_rate_limits_hook's
# per-Router-hop admission does) -- confirmed live: branches sharing an
# identical mock delay complete in that one delay's time, not N times it --
# so admission reserves one unit per real branch to keep the cap honest
# about that concurrent load.
#
# Every branch also shares the identical litellm_logging_obj/model_call_details
# the proxy attaches before the dispatch, and Logging.should_run_logging
# gates each event type on that shared dict's own has_logged_{event_type}
# flag (confirmed against the real pipeline: common_processing_pre_call_logic
# -> route_request -> abatch_completion), so at most one success event and
# at most one failure event ever fire for the whole dispatch, never one per
# branch (fastest_response cancels every losing branch instead, which never
# fires a terminal event either way). Releasing everything pending on
# whichever of those fires is therefore correct, even though it may release
# the batch's units before every real branch has finished -- an accepted
# tradeoff, since no per-branch completion signal is ever available here.
# ---------------------------------------------------------------------------


def _batch_data(tags: list[str], model: str, call_id: str = "call-1", fastest_response: bool = False) -> dict:
    data = {**_data(tags, call_id=call_id), "model": model}
    if fastest_response:
        data["fastest_response"] = True
    return data


@pytest.mark.asyncio
async def test_batch_dispatch_reserves_one_unit_per_real_branch(time_controller, monkeypatch):
    """A caller whose tag is capped at N concurrent calls can submit
    model=a,b,c and drive 3 real simultaneous provider calls -- admission
    must reserve 3 units, not 1, or the cap doesn't reflect real load."""
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "concurrency_limits": {
                "limits": [{"name": "conc", "tag_id": "end_user_id", "limit": 3, "period_seconds": 60}]
            }
        },
    )
    hook = _make_hook(time_controller)

    await hook.async_pre_call_hook(
        user_api_key_dict=_key(),
        cache=DualCache(),
        data={**_data(["end_user_id:u1"], call_id="call-1"), "model": "model-a,model-b,model-c"},
        call_type="acompletion",
    )
    # The batch alone already occupies every unit of the cap of 3.
    with pytest.raises(ProxyRateLimitError):
        await hook.async_pre_call_hook(
            user_api_key_dict=_key(),
            cache=DualCache(),
            data=_data(["end_user_id:u1"], call_id="call-2"),
            call_type="completion",
        )


@pytest.mark.asyncio
async def test_batch_dispatch_accounts_for_nested_message_lists(time_controller, monkeypatch):
    """Router.abatch_completion's "N requests to M models" mode (a nested
    messages: list[list[...]]) dispatches one real branch per (message,
    model) pair, not one per model -- admission must reserve
    model_count * message_list_count units."""
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "concurrency_limits": {
                "limits": [{"name": "conc", "tag_id": "end_user_id", "limit": 4, "period_seconds": 60}]
            }
        },
    )
    hook = _make_hook(time_controller)

    data = {
        **_data(["end_user_id:u1"], call_id="call-1"),
        "model": "model-a,model-b",
        "messages": [[{"role": "user", "content": "q1"}], [{"role": "user", "content": "q2"}]],
    }
    await hook.async_pre_call_hook(user_api_key_dict=_key(), cache=DualCache(), data=data, call_type="acompletion")

    # 2 models x 2 message-lists = 4 real branches, exactly at the cap of 4.
    with pytest.raises(ProxyRateLimitError):
        await hook.async_pre_call_hook(
            user_api_key_dict=_key(),
            cache=DualCache(),
            data=_data(["end_user_id:u1"], call_id="call-2"),
            call_type="completion",
        )


@pytest.mark.asyncio
async def test_fastest_response_batch_also_reserves_one_unit_per_branch(time_controller, monkeypatch):
    """abatch_completion_fastest_response's own comma-separated `model` also
    starts every branch as a real provider call before the race is
    resolved, so it must reserve one unit per branch too."""
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "concurrency_limits": {
                "limits": [{"name": "conc", "tag_id": "end_user_id", "limit": 2, "period_seconds": 60}]
            }
        },
    )
    hook = _make_hook(time_controller)

    await hook.async_pre_call_hook(
        user_api_key_dict=_key(),
        cache=DualCache(),
        data={
            **_data(["end_user_id:u1"], call_id="call-1"),
            "model": "model-a,model-b",
            "fastest_response": True,
        },
        call_type="acompletion",
    )
    with pytest.raises(ProxyRateLimitError):
        await hook.async_pre_call_hook(
            user_api_key_dict=_key(),
            cache=DualCache(),
            data=_data(["end_user_id:u1"], call_id="call-2"),
            call_type="completion",
        )


@pytest.mark.asyncio
async def test_batch_dispatch_release_is_safe_across_both_real_terminal_events(time_controller, monkeypatch):
    """has_logged_async_success and has_logged_async_failure gate
    independently, so a mixed-outcome dispatch (one branch succeeds,
    another fails) can fire both a real success event and a real failure
    event for the same shared reservation. Releasing everything on the
    first must leave the second nothing to release, not double-release or
    error."""
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "concurrency_limits": {
                "limits": [{"name": "conc", "tag_id": "end_user_id", "limit": 2, "period_seconds": 60}]
            }
        },
    )
    hook = _make_hook(time_controller)

    await hook.async_pre_call_hook(
        user_api_key_dict=_key(),
        cache=DualCache(),
        data={**_data(["end_user_id:u1"], call_id="call-1"), "model": "model-a,model-b"},
        call_type="acompletion",
    )

    kwargs = {"litellm_call_id": "call-1", "metadata": {"tags": ["end_user_id:u1"]}}
    await hook.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)
    await asyncio.sleep(0)

    result = await hook.async_pre_call_hook(
        user_api_key_dict=_key(),
        cache=DualCache(),
        data=_data(["end_user_id:u1"], call_id="call-2"),
        call_type="completion",
    )
    assert result is not None

    await hook.async_log_failure_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)


@pytest.mark.asyncio
async def test_batch_dispatch_disconnect_releases_every_reserved_unit(time_controller, monkeypatch):
    """A client disconnect mid-batch must release every unit the whole
    dispatch reserved (one per real branch), not just one."""
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "concurrency_limits": {
                "limits": [{"name": "conc", "tag_id": "end_user_id", "limit": 3, "period_seconds": 60}]
            }
        },
    )
    hook = _make_hook(time_controller)

    await hook.async_pre_call_hook(
        user_api_key_dict=_key(),
        cache=DualCache(),
        data=_batch_data(["end_user_id:u1"], model="model-a,model-b,model-c", call_id="call-1"),
        call_type="acompletion",
    )
    await hook.async_release_disconnect_state_hook({"litellm_call_id": "call-1"})

    result = await hook.async_pre_call_hook(
        user_api_key_dict=_key(),
        cache=DualCache(),
        data={**_data(["end_user_id:u1"], call_id="call-2"), "model": "model-x,model-y,model-z"},
        call_type="acompletion",
    )
    assert result is not None


@pytest.mark.asyncio
async def test_real_abatch_completion_with_a_shared_logging_obj_does_not_leak(time_controller, monkeypatch):
    """End-to-end against the real Router.abatch_completion with a real,
    shared litellm_logging_obj attached the same way common_request_processing.py
    does it before Router ever sees the request. Every branch here
    genuinely succeeds, but Logging.should_run_logging still suppresses
    every callback after the first, so this proves all 3 units this hook
    reserves for the 3-branch dispatch are still fully released together
    by that one surviving event, not left leaking."""
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "concurrency_limits": {
                "limits": [{"name": "conc", "tag_id": "end_user_id", "limit": 3, "period_seconds": 60}]
            }
        },
    )
    hook = _make_hook(time_controller)
    monkeypatch.setattr(litellm, "callbacks", [hook])

    router = Router(
        model_list=[
            {
                "model_name": "model-fast",
                "litellm_params": {"model": "gpt-3.5-turbo", "mock_response": "fast-done", "mock_delay": 0.01},
            },
            {
                "model_name": "model-slow-1",
                "litellm_params": {"model": "gpt-3.5-turbo", "mock_response": "slow-1-done", "mock_delay": 0.05},
            },
            {
                "model_name": "model-slow-2",
                "litellm_params": {"model": "gpt-3.5-turbo", "mock_response": "slow-2-done", "mock_delay": 0.05},
            },
        ]
    )

    admitted_data = await hook.async_pre_call_hook(
        user_api_key_dict=_key(),
        cache=DualCache(),
        data={
            **_batch_data(["end_user_id:u1"], model="model-fast,model-slow-1,model-slow-2", call_id="call-1"),
            "messages": [{"role": "user", "content": "hello"}],
        },
        call_type="acompletion",
    )

    logging_obj, admitted_data = litellm.utils.function_setup(
        original_function="acompletion",
        rules_obj=litellm.utils.Rules(),
        start_time=time_controller.now(),
        **admitted_data,
    )
    admitted_data["litellm_logging_obj"] = logging_obj

    models = [m.strip() for m in admitted_data.pop("model").split(",")]
    responses = await router.abatch_completion(models=models, **admitted_data)
    await asyncio.sleep(0.1)

    assert [r.choices[0].message.content for r in responses] == ["fast-done", "slow-1-done", "slow-2-done"]

    # No leak: all 3 reserved units are back, so a fresh request needing
    # the full cap of 3 is admitted, not wrongly rejected by leftover
    # phantom reservations.
    result = await hook.async_pre_call_hook(
        user_api_key_dict=_key(),
        cache=DualCache(),
        data={**_data(["end_user_id:u1"], call_id="call-2"), "model": "model-p,model-q,model-r"},
        call_type="acompletion",
    )
    assert result is not None


@pytest.mark.asyncio
async def test_real_proxy_pipeline_batch_dispatch_fires_one_terminal_event_not_one_per_branch(
    time_controller, monkeypatch
):
    """Goes through the actual proxy request pipeline this hook runs inside
    -- ProxyBaseLLMRequestProcessing.common_processing_pre_call_logic, then
    route_request, exactly like a real incoming HTTP request -- rather than
    reconstructing the shared litellm_logging_obj by hand. Settles a live
    disagreement about whether Router.abatch_completion(models=[...], **data)
    fires one terminal callback per branch or one for the whole dispatch:
    calling that same function directly, with a bare kwargs dict that skips
    common_processing_pre_call_logic entirely, gives each branch its own
    fresh litellm_call_id and its own fresh Logging instance (no
    should_run_logging suppression is possible without a shared one) --
    but that is not what a real request goes through. common_processing_pre_call_logic
    always attaches one litellm_logging_obj (and one litellm_call_id) to
    `data` before route_request ever dispatches into Router, and every
    branch's own acompletion() call receives that same object via **data,
    so should_run_logging suppresses every callback after the first for the
    whole call_id -- confirmed here by asserting on litellm_call_id/kwargs
    identity, not just a fire count.
    """
    fire_log: list[
        tuple[str, str | None]
    ] = []  # mutable-ok: appended by the probe callback below, read once at the end

    class _ProbeLogger(CustomLogger):
        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            fire_log.append(("success", kwargs.get("litellm_call_id")))  # mutable-ok: see fire_log's own comment

    monkeypatch.setattr(litellm, "callbacks", [_ProbeLogger()])

    router = Router(
        model_list=[
            {"model_name": "m1", "litellm_params": {"model": "gpt-3.5-turbo", "mock_response": "ok1"}},
            {"model_name": "m2", "litellm_params": {"model": "gpt-3.5-turbo", "mock_response": "ok2"}},
            {"model_name": "m3", "litellm_params": {"model": "gpt-3.5-turbo", "mock_response": "ok3"}},
        ]
    )

    processing_obj = ProxyBaseLLMRequestProcessing(
        data={"model": "m1,m2,m3", "messages": [{"role": "user", "content": "hi"}]}
    )
    mock_request = MagicMock(spec=Request)
    mock_request.headers = {}

    async def passthrough_pre_call_hook(user_api_key_dict, data, call_type):
        return data

    async def passthrough_add_litellm_data_to_request(*args, **kwargs):
        return kwargs.get("data", {})

    monkeypatch.setattr(
        litellm.proxy.common_request_processing,
        "add_litellm_data_to_request",
        passthrough_add_litellm_data_to_request,
    )
    mock_proxy_logging_obj = MagicMock(spec=ProxyLogging)
    mock_proxy_logging_obj.pre_call_hook = AsyncMock(side_effect=passthrough_pre_call_hook)
    mock_proxy_logging_obj.during_call_hook = AsyncMock(return_value=None)
    mock_proxy_config = MagicMock(spec=ProxyConfig)
    mock_proxy_config._get_hierarchical_router_settings = AsyncMock(return_value={})

    returned_data, logging_obj = await processing_obj.common_processing_pre_call_logic(
        request=mock_request,
        general_settings={},
        user_api_key_dict=_key(),
        proxy_logging_obj=mock_proxy_logging_obj,
        proxy_config=mock_proxy_config,
        route_type="acompletion",
        llm_router=router,
    )
    # This is the exact mechanism the module docstring and this hook's own
    # admission rely on: one shared object, attached before Router ever
    # sees the request.
    assert returned_data.get("litellm_logging_obj") is logging_obj

    llm_call = await route_request(
        data=returned_data,
        llm_router=router,
        user_model=None,
        route_type="acompletion",
        user_api_key_dict=_key(),
    )
    responses = await llm_call
    await asyncio.sleep(0.1)

    assert [r.choices[0].message.content for r in responses] == ["ok1", "ok2", "ok3"]
    assert len(fire_log) == 1
    assert fire_log[0][1] == returned_data.get("litellm_call_id")


# ---------------------------------------------------------------------------
# Accounting: tokens/dollars via async_log_success_event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dollar_limit_accounts_usage_and_rejects_once_over(time_controller, monkeypatch):
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "dollar_limits": {
                "limits": [{"name": "daily_spend", "tag_id": "end_user_id", "limit": 10.0, "period_seconds": 86400}]
            }
        },
    )
    hook = _make_hook(time_controller)

    data = _data(["end_user_id:u1"], call_id="call-1")
    await hook.async_pre_call_hook(user_api_key_dict=_key(), cache=DualCache(), data=data, call_type="completion")
    kwargs = {
        "litellm_call_id": "call-1",
        "metadata": {"tags": ["end_user_id:u1"]},
        "standard_logging_object": {"total_tokens": 0, "response_cost": 12.0},
    }
    await hook.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)
    await asyncio.sleep(0)

    with pytest.raises(ProxyRateLimitError):
        await hook.async_pre_call_hook(
            user_api_key_dict=_key(),
            cache=DualCache(),
            data=_data(["end_user_id:u1"], call_id="call-2"),
            call_type="completion",
        )


@pytest.mark.asyncio
async def test_log_success_event_accounts_when_litellm_params_carries_a_null_litellm_metadata_key(
    time_controller, monkeypatch
):
    """
    kwargs at async_log_success_event time is Logging.model_call_details, not
    the flat dict admission sees -- for a plain (non LITELLM_METADATA_ROUTES)
    chat completion, kwargs["litellm_params"] carries a "litellm_metadata" key
    that is always present but set to None, alongside the real, populated
    "metadata" dict. get_metadata_variable_name_from_kwargs only checks key
    presence, so it always resolved to "litellm_metadata" here and read no
    tags/identity at all, silently dropping every token/dollar/key-hash/alias
    accounting for this route shape.
    """
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "dollar_limits": {
                "limits": [{"name": "daily_spend", "tag_id": "end_user_id", "limit": 10.0, "period_seconds": 86400}]
            }
        },
    )
    hook = _make_hook(time_controller)

    data = _data(["end_user_id:u1"], call_id="call-1")
    await hook.async_pre_call_hook(user_api_key_dict=_key(), cache=DualCache(), data=data, call_type="completion")
    kwargs = {
        "litellm_call_id": "call-1",
        "litellm_params": {
            "litellm_metadata": None,
            "metadata": {"tags": ["end_user_id:u1"], "user_api_key": "hash"},
        },
        "standard_logging_object": {"total_tokens": 0, "response_cost": 12.0},
    }
    await hook.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)
    await asyncio.sleep(0)

    with pytest.raises(ProxyRateLimitError):
        await hook.async_pre_call_hook(
            user_api_key_dict=_key(),
            cache=DualCache(),
            data=_data(["end_user_id:u1"], call_id="call-2"),
            call_type="completion",
        )


@pytest.mark.asyncio
async def test_log_success_event_reads_tags_when_top_level_kwargs_carries_a_null_metadata_key(
    time_controller, monkeypatch
):
    """
    Some call paths populate a top-level "metadata" (or "litellm_metadata")
    key on kwargs (Logging.model_call_details) set to None, alongside the
    real, populated dict nested under kwargs["litellm_params"].
    _get_tags_from_request_kwargs checks the top-level key first; passing it
    raw kwargs made a present-but-None top-level key win, reading no tags at
    all even though metadata_variable_name correctly resolved to "metadata".
    """
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "dollar_limits": {
                "limits": [{"name": "daily_spend", "tag_id": "end_user_id", "limit": 10.0, "period_seconds": 86400}]
            }
        },
    )
    hook = _make_hook(time_controller)

    kwargs = {
        "litellm_call_id": "call-1",
        "metadata": None,
        "litellm_params": {
            "metadata": {"tags": ["end_user_id:u1"], "user_api_key": "hash"},
        },
        "standard_logging_object": {"total_tokens": 0, "response_cost": 12.0},
    }
    await hook.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)
    await asyncio.sleep(0)

    with pytest.raises(ProxyRateLimitError):
        await hook.async_pre_call_hook(
            user_api_key_dict=_key(),
            cache=DualCache(),
            data=_data(["end_user_id:u1"], call_id="call-2"),
            call_type="completion",
        )


@pytest.mark.asyncio
async def test_dollar_limit_respects_apply_to_key_alias_at_accounting_time(time_controller, monkeypatch):
    """The entry only applies to `premium-key`; a non-listed key's spend must
    not be charged against this bucket at all."""
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "dollar_limits": {
                "limits": [
                    {
                        "name": "daily_spend",
                        "tag_id": "end_user_id",
                        "limit": 10.0,
                        "period_seconds": 86400,
                        "apply_to_key_alias": ["premium-key"],
                    }
                ]
            }
        },
    )
    hook = _make_hook(time_controller)

    data = _data(["end_user_id:u1"], call_id="call-1")
    await hook.async_pre_call_hook(
        user_api_key_dict=_key(alias="other-key"), cache=DualCache(), data=data, call_type="completion"
    )
    kwargs = {
        "litellm_call_id": "call-1",
        "metadata": {"tags": ["end_user_id:u1"], "user_api_key_alias": "other-key"},
        "standard_logging_object": {"total_tokens": 0, "response_cost": 999.0},
    }
    await hook.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)
    await asyncio.sleep(0)

    # premium-key was never charged -- still fully under its own limit.
    result = await hook.async_pre_call_hook(
        user_api_key_dict=_key(alias="premium-key"),
        cache=DualCache(),
        data=_data(["end_user_id:u1"], call_id="call-2"),
        call_type="completion",
    )
    assert result is not None


# ---------------------------------------------------------------------------
# Config hot-reload: identity-based re-validation, no restart needed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_config_reload_takes_effect_on_next_request(time_controller, monkeypatch):
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "request_limits": {
                "limits": [{"name": "daily", "tag_id": "end_user_id", "limit": 100, "period_seconds": 86400}]
            }
        },
    )
    hook = _make_hook(time_controller)
    await hook.async_pre_call_hook(
        user_api_key_dict=_key(), cache=DualCache(), data=_data(["end_user_id:u1"]), call_type="completion"
    )

    # Reload to a stricter config -- a fresh dict object, matching how a
    # proxy config reload replaces litellm_settings.global_tag_rate_limits
    # wholesale via setattr(litellm, key, value). A changed `limit` folds
    # into the bucket's own policy fingerprint, so this is a fresh counter;
    # the new, stricter limit=1 is still reachable in exactly one more call.
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "request_limits": {
                "limits": [{"name": "daily", "tag_id": "end_user_id", "limit": 1, "period_seconds": 86400}]
            }
        },
    )
    await hook.async_pre_call_hook(
        user_api_key_dict=_key(),
        cache=DualCache(),
        data=_data(["end_user_id:u1"], call_id="call-2"),
        call_type="completion",
    )
    with pytest.raises(ProxyRateLimitError):
        await hook.async_pre_call_hook(
            user_api_key_dict=_key(),
            cache=DualCache(),
            data=_data(["end_user_id:u1"], call_id="call-3"),
            call_type="completion",
        )


# ---------------------------------------------------------------------------
# Identity resolution: policy-backed tags must win over caller-supplied ones
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_caller_supplied_tag_cannot_shadow_the_policy_backed_identity_tag(time_controller, monkeypatch):
    """
    Security regression: _merge_tags (litellm_pre_call_utils.py) keeps
    caller-supplied tags first in the merged tags list, appending key/team/
    project-contributed tags only if not already present. Since
    _extract_identity/_entry_applies resolve a tag_id by
    first-match-by-prefix, an authenticated caller could otherwise submit
    e.g. company_id:attacker-chosen ahead of the key's real
    company_id:real-company (surfaced via metadata.inherited_tags) and have
    every company_id-scoped entry resolve to the forged value instead of the
    real one -- letting the caller dodge the limit entirely by rotating
    fabricated identities, or evade being charged against their own real
    bucket. The hook must order tags so inherited_tags wins.
    """
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "request_limits": {
                "limits": [{"name": "per-company", "tag_id": "company_id", "limit": 1, "period_seconds": 86400}]
            }
        },
    )
    hook = _make_hook(time_controller)
    key = _key()

    poisoned_data = {
        "litellm_call_id": "attack-1",
        "metadata": {
            "tags": ["company_id:attacker-chosen", "company_id:real-company"],
            "inherited_tags": ["company_id:real-company"],
        },
    }
    await hook.async_pre_call_hook(user_api_key_dict=key, cache=DualCache(), data=poisoned_data, call_type="completion")

    # The real company's own bucket must have been charged by the attack
    # request, not a bucket keyed to the attacker's forged value -- so a
    # second, genuine company_id:real-company request is now rejected.
    victim_data = {
        "litellm_call_id": "victim-1",
        "metadata": {"tags": ["company_id:real-company"], "inherited_tags": ["company_id:real-company"]},
    }
    with pytest.raises(ProxyRateLimitError):
        await hook.async_pre_call_hook(
            user_api_key_dict=key, cache=DualCache(), data=victim_data, call_type="completion"
        )


@pytest.mark.asyncio
async def test_success_accounting_also_resolves_identity_from_the_policy_backed_tag(time_controller, monkeypatch):
    """Same forged-tag scenario as the admission-time test above, but for
    async_log_success_event's own identity resolution (tokens/dollars)."""
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "dollar_limits": {
                "limits": [
                    {"name": "per-company-spend", "tag_id": "company_id", "limit": 10.0, "period_seconds": 86400}
                ]
            }
        },
    )
    hook = _make_hook(time_controller)

    # kwargs at async_log_success_event time is Logging.model_call_details:
    # metadata/inherited_tags live nested under kwargs["litellm_params"],
    # never at the top level.
    kwargs = {
        "litellm_call_id": "attack-1",
        "litellm_params": {
            "metadata": {
                "tags": ["company_id:attacker-chosen", "company_id:real-company"],
                "inherited_tags": ["company_id:real-company"],
            },
        },
        "standard_logging_object": {"total_tokens": 0, "response_cost": 20.0},
    }
    await hook.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=0, end_time=0)
    await asyncio.sleep(0)

    # The $20 spend must have landed against company_id:real-company, so a
    # fresh request under the genuine identity is now over the $10 limit.
    with pytest.raises(ProxyRateLimitError):
        await hook.async_pre_call_hook(
            user_api_key_dict=_key(),
            cache=DualCache(),
            data={
                "litellm_call_id": "victim-1",
                "metadata": {"tags": ["company_id:real-company"], "inherited_tags": ["company_id:real-company"]},
            },
            call_type="completion",
        )


@pytest.mark.asyncio
async def test_rejection_detail_does_not_disclose_the_resolved_tag_value(time_controller, monkeypatch):
    """
    tag_value can resolve from inherited_tags (server-assigned key/team/
    project metadata via _order_tags_for_identity_resolution), so echoing it
    back in the client-facing 429 detail would disclose that identity to
    the caller who triggered the rejection.
    """
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "request_limits": {
                "limits": [{"name": "per-company", "tag_id": "company_id", "limit": 1, "period_seconds": 86400}]
            }
        },
    )
    hook = _make_hook(time_controller)
    key = _key()

    data = {
        "litellm_call_id": "call-1",
        "metadata": {
            "tags": ["company_id:secret-internal-name"],
            "inherited_tags": ["company_id:secret-internal-name"],
        },
    }
    await hook.async_pre_call_hook(user_api_key_dict=key, cache=DualCache(), data=data, call_type="completion")

    with pytest.raises(ProxyRateLimitError) as exc_info:
        await hook.async_pre_call_hook(
            user_api_key_dict=key,
            cache=DualCache(),
            data={
                "litellm_call_id": "call-2",
                "metadata": {
                    "tags": ["company_id:secret-internal-name"],
                    "inherited_tags": ["company_id:secret-internal-name"],
                },
            },
            call_type="completion",
        )

    assert "tag_value" not in exc_info.value.detail


# ---------------------------------------------------------------------------
# Concurrency ttl refresh -- veria-ai finding: this hook's own atomic checks
# never carried refresh_ttl through to TAG_RL_CHECK_AND_INCR_SCRIPT or
# InMemoryCache.set_cache, even though it shares that script with
# model_based_tag_rate_limits_hook (whose own call sites got fixed first)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_redis_backed_concurrency_ttl_refreshes_on_every_admission(time_controller):
    """
    A concurrency bucket isn't epoch-windowed like requests/tokens/dollars --
    its ttl exists only as a crash-safety net for a reservation whose
    explicit release never runs -- so a still-active bucket receiving
    continuous admissions must keep extending that ttl, or it expires
    mid-flight under sustained traffic, silently admitting past the cap.
    """
    hook, redis_cache = _redis_hook(time_controller)
    try:
        await redis_cache.ping()
    except Exception as e:
        pytest.skip(f"Redis connection failed: {e!s}")

    key = f"{{tag_rl:test:global-ttl-refresh:{uuid.uuid4().hex}}}:inflight"
    cache = hook.internal_usage_cache
    try:
        admitted, _ = await hook._check_and_increment_one(cache, key, limit=100, increment=1.0, ttl=3, refresh_ttl=True)
        assert admitted
        ttl_after_first_admission = await redis_cache.init_async_client().ttl(key)
        assert ttl_after_first_admission > 0

        await asyncio.sleep(2)

        admitted, _ = await hook._check_and_increment_one(cache, key, limit=100, increment=1.0, ttl=3, refresh_ttl=True)
        assert admitted
        ttl_after_second_admission = await redis_cache.init_async_client().ttl(key)
        assert ttl_after_second_admission >= 2
    finally:
        await redis_cache.async_delete_cache(key=key)


@pytest.mark.asyncio
async def test_in_memory_concurrency_ttl_refreshes_on_every_admission(time_controller):
    """
    In-memory mirror of the Redis test above: InMemoryCache.allow_ttl_override
    leaves a still-live ttl untouched, so without refresh_ttl reaching
    set_cache a concurrency counter's expiry stayed fixed from its first
    admission even under sustained traffic.
    """
    hook = _make_hook(time_controller)
    cache = hook.internal_usage_cache
    in_memory_cache = cache.dual_cache.in_memory_cache
    key = f"tag_rl:test:global-in-memory-ttl-refresh:{uuid.uuid4().hex}"

    admitted, _ = await hook._check_and_increment_one(cache, key, limit=100, increment=1.0, ttl=3, refresh_ttl=True)
    assert admitted
    ttl_after_first_admission = in_memory_cache.ttl_dict[key]

    admitted, _ = await hook._check_and_increment_one(cache, key, limit=100, increment=1.0, ttl=3, refresh_ttl=True)
    assert admitted
    ttl_after_second_admission = in_memory_cache.ttl_dict[key]

    assert ttl_after_second_admission > ttl_after_first_admission


@pytest.mark.asyncio
async def test_concurrency_limit_admission_refreshes_ttl_end_to_end(time_controller, monkeypatch):
    """
    End-to-end regression through async_pre_call_hook itself (not just the
    low-level _check_and_increment_one helper above): a concurrency entry's
    bucket key must carry a live ttl after admission, proving refresh_ttl is
    actually wired from the classified check through to the atomic batch,
    not just present on the helper's own signature.
    """
    monkeypatch.setattr(
        litellm,
        "global_tag_rate_limits",
        {
            "concurrency_limits": {
                "limits": [{"name": "conc", "tag_id": "end_user_id", "limit": 5, "period_seconds": 60}]
            }
        },
    )
    hook = _make_hook(time_controller)
    await hook.async_pre_call_hook(
        user_api_key_dict=_key(), cache=DualCache(), data=_data(["end_user_id:u1"]), call_type="completion"
    )

    in_memory_cache = hook.internal_usage_cache.dual_cache.in_memory_cache
    inflight_keys = [key for key in in_memory_cache.ttl_dict if key.endswith(":inflight")]
    assert len(inflight_keys) == 1
    assert in_memory_cache.ttl_dict[inflight_keys[0]] > time.time()
