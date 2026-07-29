"""Pin ProxyLogging lifecycle: ``__init__``, ``startup_event``,
``update_values``, ``_add_proxy_hooks``, ``get_proxy_hook``, and
``_init_litellm_callbacks``.

Also covers ``update_request_status`` and ``_convert_user_api_key_auth_to_dict``
because they are direct dependents on the lifecycle state.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import (
    InternalUsageCache,
    ProxyLogging,
)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_proxy_logging_init_sets_default_state(mock_callbacks_disabled):
    cache = UserApiKeyCache()
    pl = ProxyLogging(user_api_key_cache=cache)
    snapshot = {
        "internal_usage_cache_type": type(pl.internal_usage_cache).__name__,
        "alerting_is_none": pl.alerting is None,
        "alerting_threshold": pl.alerting_threshold,
        "premium_user": pl.premium_user,
        "proxy_hook_mapping": pl.proxy_hook_mapping,
        "daily_report_started": pl.daily_report_started,
        "hanging_requests_check_started": pl.hanging_requests_check_started,
    }
    assert snapshot == {
        "internal_usage_cache_type": "InternalUsageCache",
        "alerting_is_none": True,
        "alerting_threshold": 300,
        "premium_user": False,
        "proxy_hook_mapping": {},
        "daily_report_started": False,
        "hanging_requests_check_started": False,
    }


def test_proxy_logging_init_premium_user_flag(mock_callbacks_disabled):
    pl = ProxyLogging(user_api_key_cache=UserApiKeyCache(), premium_user=True)
    assert pl.premium_user is True


def test_proxy_logging_init_missing_cache_raises():
    with pytest.raises(TypeError):
        ProxyLogging()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# update_values
# ---------------------------------------------------------------------------


def test_update_values_stores_alerting_state(proxy_logging):
    proxy_logging.slack_alerting_instance = MagicMock()
    proxy_logging.update_values(
        alerting=["slack"],
        alerting_threshold=42.0,
        alert_types=["llm_too_slow"],
        alert_to_webhook_url={"key": "value"},
    )
    snapshot = {
        "alerting": proxy_logging.alerting,
        "threshold": proxy_logging.alerting_threshold,
        "alert_types": proxy_logging.alert_types,
        "webhook_url": proxy_logging.alert_to_webhook_url,
    }
    assert snapshot == {
        "alerting": ["slack"],
        "threshold": 42.0,
        "alert_types": ["llm_too_slow"],
        "webhook_url": {"key": "value"},
    }


def test_update_values_with_only_redis_cache_does_not_touch_slack(proxy_logging):
    proxy_logging.slack_alerting_instance = MagicMock()
    redis = MagicMock()
    proxy_logging.update_values(redis_cache=redis)
    proxy_logging.slack_alerting_instance.update_values.assert_not_called()
    assert proxy_logging.internal_usage_cache.dual_cache.redis_cache is redis


def test_update_values_with_no_args_is_noop(proxy_logging):
    proxy_logging.slack_alerting_instance = MagicMock()
    proxy_logging.update_values()
    proxy_logging.slack_alerting_instance.update_values.assert_not_called()


def test_update_values_invalid_type_for_alerting_raises(proxy_logging):
    proxy_logging.slack_alerting_instance = MagicMock(
        update_values=MagicMock(side_effect=TypeError("bad type"))
    )
    with pytest.raises(TypeError):
        proxy_logging.update_values(alerting={"not": "a list"})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# startup_event
# ---------------------------------------------------------------------------


def test_startup_event_initializes_slack_and_callbacks(proxy_logging):
    proxy_logging.slack_alerting_instance = MagicMock()
    proxy_logging.slack_alerting_instance.alert_types = []
    proxy_logging._init_litellm_callbacks = MagicMock()
    proxy_logging.update_values = MagicMock()

    proxy_logging.startup_event(llm_router=None, redis_usage_cache=None)
    snapshot = {
        "update_called": proxy_logging.update_values.called,
        "init_called": proxy_logging._init_litellm_callbacks.called,
        "slack_update_called": proxy_logging.slack_alerting_instance.update_values.called,
    }
    assert snapshot == {
        "update_called": True,
        "init_called": True,
        "slack_update_called": True,
    }


def test_startup_event_propagates_init_callbacks_failure_raises(proxy_logging):
    proxy_logging.slack_alerting_instance = MagicMock()
    proxy_logging.slack_alerting_instance.alert_types = []
    proxy_logging._init_litellm_callbacks = MagicMock(side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        proxy_logging.startup_event(llm_router=None, redis_usage_cache=None)


# ---------------------------------------------------------------------------
# _add_proxy_hooks
# ---------------------------------------------------------------------------


def test_add_proxy_hooks_registers_callbacks(proxy_logging, monkeypatch):
    """Patch ``PROXY_HOOKS`` and the resolver so we control exactly
    what gets registered. Verifies that the resulting instances land in
    ``proxy_logging.proxy_hook_mapping`` keyed by hook name.
    """
    hook_keys = ["cache_control_check", "max_budget_limiter"]
    registered: List[Any] = []

    from litellm.proxy import utils as utils_mod

    def fake_get_proxy_hook(hook_name):
        class _Stub:
            __name__ = hook_name

            def __init__(self, **kwargs):
                self.hook_name = hook_name

        return _Stub

    monkeypatch.setattr(utils_mod, "PROXY_HOOKS", hook_keys)
    monkeypatch.setattr(utils_mod, "get_proxy_hook", fake_get_proxy_hook)
    monkeypatch.setattr(
        litellm.logging_callback_manager,
        "add_litellm_callback",
        lambda cb: registered.append(cb),
    )

    with patch("litellm.proxy.proxy_server.prisma_client", None):
        proxy_logging._add_proxy_hooks(llm_router=None)

    keys = list(proxy_logging.proxy_hook_mapping.keys())
    snapshot = {
        "mapping_keys": keys,
        "registered_count": len(registered),
        "registered_hook_names": [getattr(r, "hook_name", None) for r in registered],
    }
    assert snapshot == {
        "mapping_keys": hook_keys,
        "registered_count": len(hook_keys),
        "registered_hook_names": hook_keys,
    }


def test_add_proxy_hooks_unknown_hook_raises(proxy_logging, monkeypatch):
    from litellm.proxy import utils as utils_mod

    monkeypatch.setattr(utils_mod, "PROXY_HOOKS", ["bogus_hook"])

    def bad_resolver(name):
        raise KeyError(name)

    monkeypatch.setattr(utils_mod, "get_proxy_hook", bad_resolver)
    with pytest.raises(KeyError):
        proxy_logging._add_proxy_hooks(llm_router=None)


# ---------------------------------------------------------------------------
# get_proxy_hook
# ---------------------------------------------------------------------------


def test_get_proxy_hook_returns_registered_instance(proxy_logging):
    s_cache = MagicMock()
    s_budget = MagicMock()
    s_parallel = MagicMock()
    proxy_logging.proxy_hook_mapping = {
        "cache_control_check": s_cache,
        "max_budget_limiter": s_budget,
        "max_parallel_request_limiter": s_parallel,
    }
    snapshot = {
        "cache_control_check": proxy_logging.get_proxy_hook("cache_control_check") is s_cache,
        "max_budget_limiter": proxy_logging.get_proxy_hook("max_budget_limiter") is s_budget,
        "max_parallel_request_limiter": proxy_logging.get_proxy_hook("max_parallel_request_limiter") is s_parallel,
        "unknown_returns_none": proxy_logging.get_proxy_hook("unknown") is None,
    }
    assert snapshot == {
        "cache_control_check": True,
        "max_budget_limiter": True,
        "max_parallel_request_limiter": True,
        "unknown_returns_none": True,
    }


def test_get_proxy_hook_unknown_returns_none(proxy_logging):
    proxy_logging.proxy_hook_mapping = {}
    assert proxy_logging.get_proxy_hook("does-not-exist") is None


def test_get_proxy_hook_non_string_key_raises(proxy_logging):
    # ``dict.get`` doesn't raise on unhashable types — but ``None`` returns None.
    # The pin: passing an unhashable key blows up like dict access does.
    proxy_logging.proxy_hook_mapping = {"k": object()}
    with pytest.raises(TypeError):
        proxy_logging.get_proxy_hook({"unhashable": True})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _init_litellm_callbacks
# ---------------------------------------------------------------------------


def test_init_litellm_callbacks_replaces_string_with_instance(proxy_logging, monkeypatch):
    from litellm.proxy import utils as utils_mod

    sentinel_instance = MagicMock(spec=litellm.integrations.custom_logger.CustomLogger)
    sentinel_instance.__class__ = litellm.integrations.custom_logger.CustomLogger

    monkeypatch.setattr(litellm, "callbacks", ["some-string-logger"])
    monkeypatch.setattr(
        litellm.litellm_core_utils.litellm_logging,
        "_init_custom_logger_compatible_class",
        lambda *a, **kw: sentinel_instance,
    )

    monkeypatch.setattr(utils_mod, "PROXY_HOOKS", [])
    proxy_logging._init_litellm_callbacks(llm_router=None)
    snapshot = {
        "replaced_first_item": litellm.callbacks[0] is sentinel_instance,
        "callbacks_grew_with_service": len(litellm.callbacks) >= 2,
        "service_logging_appended": any(
            "ServiceLogging" in type(c).__name__ for c in litellm.callbacks
        ),
    }
    assert snapshot == {
        "replaced_first_item": True,
        "callbacks_grew_with_service": True,
        "service_logging_appended": True,
    }


def test_init_litellm_callbacks_string_resolution_failure_keeps_string(proxy_logging, monkeypatch):
    from litellm.proxy import utils as utils_mod

    monkeypatch.setattr(litellm, "callbacks", ["unknown-logger"])
    monkeypatch.setattr(
        litellm.litellm_core_utils.litellm_logging,
        "_init_custom_logger_compatible_class",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(utils_mod, "PROXY_HOOKS", [])
    proxy_logging._init_litellm_callbacks(llm_router=None)
    # Resolver returned None — original string remains in place at idx 0.
    assert litellm.callbacks[0] == "unknown-logger"


def test_init_litellm_callbacks_propagates_resolver_error_raises(proxy_logging, monkeypatch):
    from litellm.proxy import utils as utils_mod

    monkeypatch.setattr(litellm, "callbacks", ["raises-on-init"])
    monkeypatch.setattr(
        litellm.litellm_core_utils.litellm_logging,
        "_init_custom_logger_compatible_class",
        MagicMock(side_effect=RuntimeError("bad init")),
    )
    monkeypatch.setattr(utils_mod, "PROXY_HOOKS", [])
    with pytest.raises(RuntimeError):
        proxy_logging._init_litellm_callbacks(llm_router=None)


# ---------------------------------------------------------------------------
# update_request_status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_request_status_when_alerting_set_writes_cache(proxy_logging):
    proxy_logging.alerting = ["slack"]
    proxy_logging.alerting_threshold = 5.0
    captured: Dict[str, Any] = {}

    async def fake_set_cache(**kwargs):
        captured.update(kwargs)

    proxy_logging.internal_usage_cache.async_set_cache = fake_set_cache  # type: ignore[assignment]
    await proxy_logging.update_request_status(litellm_call_id="call-1", status="success")
    snapshot = {
        "key": captured["key"],
        "value": captured["value"],
        "local_only": captured["local_only"],
        "ttl": captured["ttl"],
    }
    assert snapshot == {
        "key": "request_status:call-1",
        "value": "success",
        "local_only": True,
        "ttl": 105.0,
    }


@pytest.mark.asyncio
async def test_update_request_status_no_alerting_skips_cache(proxy_logging):
    proxy_logging.alerting = None
    proxy_logging.internal_usage_cache.async_set_cache = AsyncMock()
    await proxy_logging.update_request_status(litellm_call_id="call-1", status="success")
    proxy_logging.internal_usage_cache.async_set_cache.assert_not_called()


@pytest.mark.asyncio
async def test_update_request_status_cache_error_raises(proxy_logging):
    proxy_logging.alerting = ["slack"]
    proxy_logging.internal_usage_cache.async_set_cache = AsyncMock(side_effect=ConnectionError("redis"))
    with pytest.raises(ConnectionError):
        await proxy_logging.update_request_status(litellm_call_id="x", status="fail")


# ---------------------------------------------------------------------------
# _convert_user_api_key_auth_to_dict
# ---------------------------------------------------------------------------


def test_convert_user_api_key_auth_to_dict_pydantic_uses_model_dump(proxy_logging, make_user_api_key_auth):
    auth = make_user_api_key_auth(user_id="u-1", team_id="t-1")
    result = proxy_logging._convert_user_api_key_auth_to_dict(auth)
    snapshot = {
        "user_id": result["user_id"],
        "team_id": result["team_id"],
        "is_dict": isinstance(result, dict),
    }
    assert snapshot == {"user_id": "u-1", "team_id": "t-1", "is_dict": True}


def test_convert_user_api_key_auth_to_dict_plain_object_uses_dict(proxy_logging):
    class Obj:
        pass

    obj = Obj()
    obj.a = 1
    obj.b = 2
    obj.c = 3
    result = proxy_logging._convert_user_api_key_auth_to_dict(obj)
    assert result == {"a": 1, "b": 2, "c": 3}


def test_convert_user_api_key_auth_to_dict_none_returns_empty_dict(proxy_logging):
    assert proxy_logging._convert_user_api_key_auth_to_dict(None) == {}


def test_convert_user_api_key_auth_to_dict_unconvertible_object_returns_empty(proxy_logging):
    class NoDict:
        __slots__ = ()

    assert proxy_logging._convert_user_api_key_auth_to_dict(NoDict()) == {}


def test_convert_user_api_key_auth_to_dict_pydantic_error_raises(proxy_logging):
    """A ``model_dump`` that raises propagates."""

    class _Boom:
        def model_dump(self):
            raise RuntimeError("model_dump failure")

    with pytest.raises(RuntimeError):
        proxy_logging._convert_user_api_key_auth_to_dict(_Boom())
