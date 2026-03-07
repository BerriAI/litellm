"""
Tests for custom_auth receiving request_data parameter.

Verifies that when a custom_auth function accepts a `request_data` parameter,
the parsed request body is passed to it. Also verifies backwards compatibility
with custom_auth functions that only accept (request, api_key).
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from starlette.datastructures import URL

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import _user_api_key_auth_builder


def _make_request(path: str) -> Request:
    """Create a mock Request with the given URL path."""
    request = Request(scope={"type": "http"})
    request._url = URL(url=path)
    return request


def _set_proxy_server_attrs(proxy_mod, overrides: dict) -> dict:
    """Set attributes on the proxy_server module and return originals."""
    defaults = {
        "prisma_client": None,
        "user_api_key_cache": MagicMock(),
        "proxy_logging_obj": MagicMock(),
        "master_key": "sk-master-key",
        "general_settings": {},
        "llm_model_list": [],
        "llm_router": None,
        "open_telemetry_logger": None,
        "model_max_budget_limiter": MagicMock(),
        "user_custom_auth": None,
        "jwt_handler": None,
        "litellm_proxy_admin_name": "admin",
    }
    defaults.update(overrides)
    original_values = {attr: getattr(proxy_mod, attr, None) for attr in defaults}
    for attr, val in defaults.items():
        setattr(proxy_mod, attr, val)
    return original_values


def _restore_proxy_server_attrs(proxy_mod, original_values: dict) -> None:
    """Restore original values on the proxy_server module."""
    for attr, val in original_values.items():
        setattr(proxy_mod, attr, val)


@pytest.mark.asyncio
@patch("litellm.proxy.auth.user_api_key_auth.enterprise_custom_auth", None)
async def test_custom_auth_receives_request_data():
    """
    When a custom_auth function has a `request_data` parameter,
    it should receive the parsed request body dict.
    """
    import litellm.proxy.proxy_server as _proxy_server_mod

    received_kwargs = {}

    async def custom_auth_with_request_data(request, api_key, request_data):
        received_kwargs["request"] = request
        received_kwargs["api_key"] = api_key
        received_kwargs["request_data"] = request_data
        return UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER)

    overrides = {"user_custom_auth": custom_auth_with_request_data}
    original_values = _set_proxy_server_attrs(_proxy_server_mod, overrides)

    try:
        request = _make_request("/chat/completions")
        test_request_data = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}

        await _user_api_key_auth_builder(
            request=request,
            api_key="Bearer sk-test",
            azure_api_key_header="",
            anthropic_api_key_header=None,
            google_ai_studio_api_key_header=None,
            azure_apim_header=None,
            request_data=test_request_data,
        )

        assert "request_data" in received_kwargs
        assert received_kwargs["request_data"] == test_request_data
        assert received_kwargs["request_data"]["model"] == "gpt-4"
    finally:
        _restore_proxy_server_attrs(_proxy_server_mod, original_values)


@pytest.mark.asyncio
@patch("litellm.proxy.auth.user_api_key_auth.enterprise_custom_auth", None)
async def test_custom_auth_backwards_compatible_without_request_data():
    """
    When a custom_auth function does NOT have a `request_data` parameter
    (old-style 2-arg signature), it should still work without errors.
    """
    import litellm.proxy.proxy_server as _proxy_server_mod

    received_kwargs = {}

    async def custom_auth_legacy(request, api_key):
        received_kwargs["request"] = request
        received_kwargs["api_key"] = api_key
        return UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER)

    overrides = {"user_custom_auth": custom_auth_legacy}
    original_values = _set_proxy_server_attrs(_proxy_server_mod, overrides)

    try:
        request = _make_request("/chat/completions")

        result = await _user_api_key_auth_builder(
            request=request,
            api_key="Bearer sk-test",
            azure_api_key_header="",
            anthropic_api_key_header=None,
            google_ai_studio_api_key_header=None,
            azure_apim_header=None,
            request_data={"model": "gpt-4"},
        )

        assert isinstance(result, UserAPIKeyAuth)
        # request_data should NOT be in received kwargs (not in signature)
        assert "request_data" not in received_kwargs
        assert "request" in received_kwargs
        assert "api_key" in received_kwargs
    finally:
        _restore_proxy_server_attrs(_proxy_server_mod, original_values)


@pytest.mark.asyncio
@patch("litellm.proxy.auth.user_api_key_auth.enterprise_custom_auth", None)
async def test_custom_auth_with_kwargs_receives_request_data():
    """
    When a custom_auth function accepts **kwargs, it should receive
    request_data since kwargs implies any parameter is accepted.
    """
    import litellm.proxy.proxy_server as _proxy_server_mod

    received_kwargs = {}

    async def custom_auth_with_kwargs(request, api_key, **kwargs):
        received_kwargs.update(kwargs)
        return UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER)

    overrides = {"user_custom_auth": custom_auth_with_kwargs}
    original_values = _set_proxy_server_attrs(_proxy_server_mod, overrides)

    try:
        request = _make_request("/chat/completions")
        test_data = {"model": "gpt-4"}

        await _user_api_key_auth_builder(
            request=request,
            api_key="Bearer sk-test",
            azure_api_key_header="",
            anthropic_api_key_header=None,
            google_ai_studio_api_key_header=None,
            azure_apim_header=None,
            request_data=test_data,
        )

        # **kwargs should capture request_data
        assert "request_data" in received_kwargs
        assert received_kwargs["request_data"] == test_data
    finally:
        _restore_proxy_server_attrs(_proxy_server_mod, original_values)


# ------------------------------------------------------------------ #
# E2E test: TestClient -> /chat/completions -> custom_auth
# ------------------------------------------------------------------ #


def test_e2e_custom_auth_request_data_via_http(monkeypatch):
    """
    E2E: Send a real HTTP POST to /chat/completions through TestClient and
    verify that the custom_auth function receives the parsed request_data
    (including the model name from the request body).

    We only assert that the auth layer correctly passes request_data;
    the downstream LLM call is mocked out.
    """
    from litellm.proxy.proxy_server import app

    captured = {}

    async def custom_auth_e2e(request, api_key, request_data):
        captured["model"] = request_data.get("model")
        captured["messages"] = request_data.get("messages")
        return UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER)

    monkeypatch.setattr("litellm.proxy.proxy_server.user_custom_auth", custom_auth_e2e)
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_api_key_cache", MagicMock())
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", AsyncMock())
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_model_list", [{"model_name": "gpt-4", "litellm_params": {"model": "gpt-4"}}])
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", AsyncMock())
    monkeypatch.setattr("litellm.proxy.proxy_server.open_telemetry_logger", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.model_max_budget_limiter", MagicMock())
    monkeypatch.setattr("litellm.proxy.proxy_server.jwt_handler", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin")
    monkeypatch.setattr("litellm.proxy.auth.user_api_key_auth.enterprise_custom_auth", None)

    client = TestClient(app, raise_server_exceptions=False)
    client.post(
        "/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
        headers={"Authorization": "Bearer sk-test-key"},
    )

    # The custom_auth should have been called with request_data
    assert "model" in captured, f"custom_auth was not called with request_data. captured={captured}"
    assert captured["model"] == "gpt-4"
    assert captured["messages"] == [{"role": "user", "content": "hello"}]
