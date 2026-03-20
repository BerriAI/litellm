"""
Integration tests for async_post_call_response_headers_hook.

Tests verify that CustomLogger callbacks can inject custom HTTP response headers
into success (streaming and non-streaming) and failure responses.
"""

import os
import sys
import pytest
from typing import Any, Dict, Optional
from unittest.mock import patch

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth


class HeaderInjectorLogger(CustomLogger):
    """Logger that injects custom headers into responses."""

    def __init__(self, headers: Optional[Dict[str, str]] = None):
        self.headers = headers
        self.called = False
        self.received_response = None
        self.received_data = None

    async def async_post_call_response_headers_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_headers: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, str]]:
        self.called = True
        self.received_response = response
        self.received_data = data
        return self.headers


@pytest.mark.asyncio
async def test_response_headers_hook_returns_headers():
    """Test that the hook returns headers from a single callback."""
    injector = HeaderInjectorLogger(headers={"x-custom-id": "abc123"})

    with patch("litellm.callbacks", [injector]):
        from litellm.proxy.utils import ProxyLogging
        from litellm.caching.caching import DualCache

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        result = await proxy_logging.post_call_response_headers_hook(
            data={"model": "test-model"},
            user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
            response={"id": "resp-1"},
        )

        assert injector.called is True
        assert result == {"x-custom-id": "abc123"}


@pytest.mark.asyncio
async def test_response_headers_hook_returns_none():
    """Test that returning None results in empty headers dict."""
    injector = HeaderInjectorLogger(headers=None)

    with patch("litellm.callbacks", [injector]):
        from litellm.proxy.utils import ProxyLogging
        from litellm.caching.caching import DualCache

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        result = await proxy_logging.post_call_response_headers_hook(
            data={"model": "test-model"},
            user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
            response={"id": "resp-1"},
        )

        assert injector.called is True
        assert result == {}


@pytest.mark.asyncio
async def test_response_headers_hook_multiple_callbacks_merge():
    """Test that headers from multiple callbacks are merged."""
    injector1 = HeaderInjectorLogger(headers={"x-header-a": "value-a"})
    injector2 = HeaderInjectorLogger(headers={"x-header-b": "value-b"})

    with patch("litellm.callbacks", [injector1, injector2]):
        from litellm.proxy.utils import ProxyLogging
        from litellm.caching.caching import DualCache

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        result = await proxy_logging.post_call_response_headers_hook(
            data={"model": "test-model"},
            user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
            response=None,
        )

        assert injector1.called is True
        assert injector2.called is True
        assert result == {"x-header-a": "value-a", "x-header-b": "value-b"}


@pytest.mark.asyncio
async def test_response_headers_hook_later_callback_overrides():
    """Test that later callbacks override earlier ones for the same header key."""
    injector1 = HeaderInjectorLogger(headers={"x-request-id": "first"})
    injector2 = HeaderInjectorLogger(headers={"x-request-id": "second"})

    with patch("litellm.callbacks", [injector1, injector2]):
        from litellm.proxy.utils import ProxyLogging
        from litellm.caching.caching import DualCache

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        result = await proxy_logging.post_call_response_headers_hook(
            data={"model": "test-model"},
            user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
            response=None,
        )

        assert result == {"x-request-id": "second"}


@pytest.mark.asyncio
async def test_response_headers_hook_receives_response_on_success():
    """Test that the hook receives the response object on success."""
    injector = HeaderInjectorLogger(headers={"x-ok": "1"})
    mock_response = {"id": "resp-success", "choices": []}

    with patch("litellm.callbacks", [injector]):
        from litellm.proxy.utils import ProxyLogging
        from litellm.caching.caching import DualCache

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        await proxy_logging.post_call_response_headers_hook(
            data={"model": "test-model"},
            user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
            response=mock_response,
        )

        assert injector.received_response is mock_response


@pytest.mark.asyncio
async def test_response_headers_hook_receives_none_response_on_failure():
    """Test that the hook receives None response for failure cases."""
    injector = HeaderInjectorLogger(headers={"x-error-id": "err-1"})

    with patch("litellm.callbacks", [injector]):
        from litellm.proxy.utils import ProxyLogging
        from litellm.caching.caching import DualCache

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        await proxy_logging.post_call_response_headers_hook(
            data={"model": "test-model"},
            user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
            response=None,
        )

        assert injector.received_response is None


@pytest.mark.asyncio
async def test_response_headers_hook_no_callbacks():
    """Test that no callbacks results in empty headers."""
    with patch("litellm.callbacks", []):
        from litellm.proxy.utils import ProxyLogging
        from litellm.caching.caching import DualCache

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        result = await proxy_logging.post_call_response_headers_hook(
            data={"model": "test-model"},
            user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
            response=None,
        )

        assert result == {}


@pytest.mark.asyncio
async def test_default_hook_returns_none():
    """Test that the base CustomLogger hook returns None by default."""
    logger = CustomLogger()
    result = await logger.async_post_call_response_headers_hook(
        data={},
        user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
        response=None,
    )
    assert result is None


# --- Tests for litellm_call_info parameter ---


class CallInfoInspectorLogger(CustomLogger):
    """Logger that captures litellm_call_info for inspection."""

    def __init__(self):
        self.called = False
        self.received_call_info = None

    async def async_post_call_response_headers_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_headers: Optional[Dict[str, str]] = None,
        litellm_call_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, str]]:
        self.called = True
        self.received_call_info = litellm_call_info
        return None


@pytest.mark.asyncio
async def test_litellm_call_info_from_hidden_params():
    """Test that litellm_call_info is built from response._hidden_params."""
    inspector = CallInfoInspectorLogger()

    class MockResponse:
        _hidden_params = {
            "custom_llm_provider": "openai",
            "api_base": "https://api.openai.com",
            "model_id": "model-abc",
        }

    with patch("litellm.callbacks", [inspector]):
        from litellm.proxy.utils import ProxyLogging
        from litellm.caching.caching import DualCache

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        await proxy_logging.post_call_response_headers_hook(
            data={"model": "gpt-4", "metadata": {"model_info": {"id": "model-abc", "provider": "HubSpot"}}},
            user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
            response=MockResponse(),
        )

        assert inspector.called is True
        assert inspector.received_call_info is not None
        assert inspector.received_call_info["custom_llm_provider"] == "openai"
        assert inspector.received_call_info["api_base"] == "https://api.openai.com"
        assert inspector.received_call_info["model_id"] == "model-abc"
        assert inspector.received_call_info["model_info"]["provider"] == "HubSpot"


@pytest.mark.asyncio
async def test_litellm_call_info_from_litellm_metadata():
    """Test that litellm_call_info finds model_info under litellm_metadata (responses API path)."""
    inspector = CallInfoInspectorLogger()

    class MockResponse:
        _hidden_params = {
            "custom_llm_provider": "azure",
            "api_base": "https://east.openai.azure.com",
            "model_id": "deploy-xyz",
        }

    with patch("litellm.callbacks", [inspector]):
        from litellm.proxy.utils import ProxyLogging
        from litellm.caching.caching import DualCache

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        await proxy_logging.post_call_response_headers_hook(
            data={"model": "gpt-4", "litellm_metadata": {"model_info": {"id": "deploy-xyz"}}},
            user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
            response=MockResponse(),
        )

        assert inspector.received_call_info["model_info"]["id"] == "deploy-xyz"
        assert inspector.received_call_info["custom_llm_provider"] == "azure"


@pytest.mark.asyncio
async def test_litellm_call_info_with_none_response():
    """Test that litellm_call_info handles None response (failure path)."""
    inspector = CallInfoInspectorLogger()

    with patch("litellm.callbacks", [inspector]):
        from litellm.proxy.utils import ProxyLogging
        from litellm.caching.caching import DualCache

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        await proxy_logging.post_call_response_headers_hook(
            data={"model": "gpt-4", "metadata": {}},
            user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
            response=None,
        )

        assert inspector.called is True
        assert inspector.received_call_info is not None
        assert inspector.received_call_info["custom_llm_provider"] is None
        assert inspector.received_call_info["model_info"] == {}


@pytest.mark.asyncio
async def test_litellm_call_info_backwards_compatible():
    """Test that existing callbacks without litellm_call_info parameter still work."""
    # HeaderInjectorLogger doesn't accept litellm_call_info — must not crash
    injector = HeaderInjectorLogger(headers={"x-test": "1"})

    class MockResponse:
        _hidden_params = {"custom_llm_provider": "openai", "api_base": "https://api.openai.com", "model_id": "m1"}

    with patch("litellm.callbacks", [injector]):
        from litellm.proxy.utils import ProxyLogging
        from litellm.caching.caching import DualCache

        proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

        result = await proxy_logging.post_call_response_headers_hook(
            data={"model": "gpt-4", "metadata": {}},
            user_api_key_dict=UserAPIKeyAuth(api_key="test-key"),
            response=MockResponse(),
        )

        assert result == {"x-test": "1"}
        assert injector.called is True
