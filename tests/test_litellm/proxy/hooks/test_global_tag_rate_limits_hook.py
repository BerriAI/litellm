"""
Unit tests for the global-scope, model-independent tag rate limiter.
"""

import asyncio
from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.global_tag_rate_limits_hook import (
    _PROXY_GlobalTagRateLimitsHook,
)


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
    # A different key, identical tag value: must be rejected too -- proves
    # the bucket is genuinely shared, not per-key by default.
    with pytest.raises(ProxyRateLimitError):
        await hook.async_pre_call_hook(
            user_api_key_dict=_key(alias="key-b"),
            cache=DualCache(),
            data=_data(["end_user_id:u1"]),
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
    data_model_b = {**_data(["end_user_id:u1"]), "model": "claude-3"}
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
    with pytest.raises(ProxyRateLimitError):
        await hook.async_pre_call_hook(
            user_api_key_dict=_key(alias="premium-key"),
            cache=DualCache(),
            data=_data(["end_user_id:u1"]),
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
    with pytest.raises(ProxyRateLimitError):
        await hook.async_pre_call_hook(
            user_api_key_dict=_key(alias="key-a", api_key="hashA"),
            cache=DualCache(),
            data=_data(["end_user_id:u1"]),
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
        {"concurrency_limits": {"limits": [{"name": "conc", "tag_id": "end_user_id", "limit": 1, "period_seconds": 60}]}},
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
