"""Sanity tests for the proxy_logging conftest fixtures.

Excluded from the pin-check by name.
"""

from __future__ import annotations

import pytest


def test_normalize_replaces_volatile_keys(normalize_fn):
    raw = {"id": 7, "name": "x", "nested": {"created_at": 1, "value": 2}}
    expected = {"id": "<VOLATILE>", "name": "x", "nested": {"created_at": "<VOLATILE>", "value": 2}}
    assert normalize_fn(raw) == expected


def test_normalize_handles_lists(normalize_fn):
    raw = [{"id": 1}, {"id": 2}]
    assert normalize_fn(raw) == [{"id": "<VOLATILE>"}, {"id": "<VOLATILE>"}]


def test_mock_dual_cache_is_dual_cache(mock_dual_cache):
    from litellm.caching.caching import DualCache

    assert isinstance(mock_dual_cache, DualCache)


def test_make_user_api_key_auth_returns_correct_type(make_user_api_key_auth):
    from litellm.proxy._types import UserAPIKeyAuth

    auth = make_user_api_key_auth()
    assert isinstance(auth, UserAPIKeyAuth)
    assert auth.user_id == "test-user"


def test_make_user_api_key_auth_overrides_apply(make_user_api_key_auth):
    auth = make_user_api_key_auth(user_id="custom-id")
    assert auth.user_id == "custom-id"


def test_proxy_logging_fixture_is_initialized(proxy_logging):
    from litellm.proxy.utils import InternalUsageCache, ProxyLogging

    assert isinstance(proxy_logging, ProxyLogging)
    assert isinstance(proxy_logging.internal_usage_cache, InternalUsageCache)
    assert proxy_logging.proxy_hook_mapping == {}


def test_make_mcp_request_obj_default(make_mcp_request_obj):
    obj = make_mcp_request_obj()
    assert obj.tool_name == "calculator"
    assert obj.arguments == {"x": 1, "y": 2}


def test_mock_router_has_guardrail_list(mock_router):
    assert mock_router.guardrail_list == []
