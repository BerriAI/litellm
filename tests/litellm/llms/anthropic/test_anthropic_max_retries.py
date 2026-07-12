"""
Regression tests for `max_retries` support on non-OpenAI providers.

`max_retries` was previously silently dropped for every provider except
OpenAI/Azure. These tests lock in the fix that:

1. Declares `max_retries` as a supported param for providers with retry wiring (openai, azure, anthropic).
2. Maps it through `AnthropicConfig` without leaking it into the request body.
3. Wires it to LiteLLM's httpx transport so transient connection errors are
   retried.
"""

import httpx
import pytest

from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
    _get_httpx_client,
)


class TestAnthropicMaxRetriesSupported:
    def test_max_retries_in_supported_params(self):
        supported = AnthropicConfig().get_supported_openai_params(model="claude-3-5-sonnet-20241022")
        assert "max_retries" in supported

    def test_map_openai_params_keeps_max_retries(self):
        optional_params = AnthropicConfig().map_openai_params(
            non_default_params={"max_retries": 3, "temperature": 0.5},
            optional_params={},
            model="claude-3-5-sonnet-20241022",
            drop_params=False,
        )
        assert optional_params.get("max_retries") == 3

    def test_map_openai_params_ignores_non_int_max_retries(self):
        optional_params = AnthropicConfig().map_openai_params(
            non_default_params={"max_retries": "fast"},
            optional_params={},
            model="claude-3-5-sonnet-20241022",
            drop_params=False,
        )
        assert "max_retries" not in optional_params

    def test_transform_request_drops_max_retries_from_body(self):
        data = AnthropicConfig().transform_request(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={"max_retries": 3, "temperature": 0.5},
            litellm_params={},
            headers={},
        )
        assert "max_retries" not in data
        # sanity: real params are still present
        assert data["temperature"] == 0.5


class TestMaxRetriesHttpTransport:
    def test_async_handler_builds_retry_transport(self):
        handler = AsyncHTTPHandler(max_retries=3)
        transport = handler.client._transport
        assert isinstance(transport, httpx.AsyncHTTPTransport)
        assert getattr(transport._pool, "_retries", 0) == 3

    def test_sync_handler_builds_retry_transport(self):
        handler = HTTPHandler(max_retries=2, timeout=5)
        transport = handler.client._transport
        assert isinstance(transport, httpx.HTTPTransport)
        assert getattr(transport._pool, "_retries", 0) == 2

    def test_async_handler_retry_transport_with_force_ipv4(self):
        import litellm

        original_force_ipv4 = litellm.force_ipv4
        litellm.force_ipv4 = True
        litellm.disable_aiohttp_transport = True
        try:
            transport = AsyncHTTPHandler._create_async_transport(max_retries=3)
            assert isinstance(transport, httpx.AsyncHTTPTransport)
            assert getattr(transport._pool, "_retries", 0) == 3
            assert transport._pool._local_address == "0.0.0.0"
        finally:
            litellm.force_ipv4 = original_force_ipv4
            litellm.disable_aiohttp_transport = False

    def test_sync_handler_retry_transport_with_force_ipv4(self):
        import litellm

        original_force_ipv4 = litellm.force_ipv4
        litellm.force_ipv4 = True
        litellm.disable_aiohttp_transport = True
        try:
            handler = HTTPHandler(max_retries=3, timeout=5)
            transport = handler.client._transport
            assert isinstance(transport, httpx.HTTPTransport)
            assert getattr(transport._pool, "_retries", 0) == 3
            assert transport._pool._local_address == "0.0.0.0"
        finally:
            litellm.force_ipv4 = original_force_ipv4
            litellm.disable_aiohttp_transport = False
            handler.close()

    def test_async_handler_force_ipv4_without_max_retries(self):
        import litellm

        original_force_ipv4 = litellm.force_ipv4
        litellm.force_ipv4 = True
        litellm.disable_aiohttp_transport = True
        try:
            transport = AsyncHTTPHandler._create_async_transport()
            assert isinstance(transport, httpx.AsyncHTTPTransport)
            assert transport._pool._local_address == "0.0.0.0"
        finally:
            litellm.force_ipv4 = original_force_ipv4
            litellm.disable_aiohttp_transport = False

    def test_default_handler_does_not_use_retry_transport(self):
        # Without max_retries we do not use the retry transport.
        handler = AsyncHTTPHandler()
        transport = handler.client._transport

        # Check if aiohttp transport is available and enabled
        aiohttp_enabled = AsyncHTTPHandler._should_use_aiohttp_transport()

        # Import the aiohttp transport type for checking
        try:
            from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport

            HAS_AIOHTTP_TRANSPORT = True
        except ImportError:
            HAS_AIOHTTP_TRANSPORT = False
            LiteLLMAiohttpTransport = None

        if aiohttp_enabled and HAS_AIOHTTP_TRANSPORT:
            # When aiohttp is available and enabled, we should get the aiohttp transport
            # which has no built-in retry mechanism in LiteLLM
            assert isinstance(transport, LiteLLMAiohttpTransport)
        else:
            # When aiohttp is not available/disabled, we should get an httpx transport
            # with explicit retries=0 (no retry)
            assert isinstance(transport, httpx.AsyncHTTPTransport)
            assert getattr(transport._pool, "_retries", 0) == 0

    def test_async_handler_transport_none_when_aiohttp_disabled_and_ipv4_off(self):
        import litellm

        original_force_ipv4 = litellm.force_ipv4
        original_disable = litellm.disable_aiohttp_transport
        litellm.force_ipv4 = False
        litellm.disable_aiohttp_transport = True
        try:
            transport = AsyncHTTPHandler._create_async_transport()
            assert transport is None
        finally:
            litellm.force_ipv4 = original_force_ipv4
            litellm.disable_aiohttp_transport = original_disable


class TestMaxRetriesValidation:
    @pytest.mark.parametrize("provider,model", [("anthropic", "claude-3-5-sonnet-20241022")])
    def test_completion_accepts_max_retries_without_error(self, provider, model):
        """
        Reproduces the original bug: passing max_retries to a non-OpenAI provider
        used to raise UnsupportedParamsError (or silently drop the param). It must
        now be accepted as a valid param.
        """
        from litellm.utils import get_supported_openai_params

        supported = get_supported_openai_params(model=model, custom_llm_provider=provider)
        assert "max_retries" in supported

    def test_non_wired_provider_still_raises_for_max_retries(self):
        """
        Regression test: providers without retry wiring should still raise
        UnsupportedParamsError when max_retries is passed via completion().
        """
        from litellm.utils import get_supported_openai_params

        # Cohere doesn't have actual retry wiring yet
        # get_supported_openai_params should NOT include max_retries for cohere
        supported_params = get_supported_openai_params(model="command", custom_llm_provider="cohere")
        assert "max_retries" not in supported_params, "max_retries should not be in supported params for cohere"


class TestMaxRetriesHandlerPaths:
    def test_get_async_httpx_client_with_max_retries(self):
        """Test that get_async_httpx_client creates client with max_retries when provided."""
        from litellm.types.utils import LlmProviders
        import asyncio

        client = get_async_httpx_client(llm_provider=LlmProviders.ANTHROPIC, params={"max_retries": 5})
        try:
            transport = client.client._transport
            assert isinstance(transport, httpx.AsyncHTTPTransport)
            assert getattr(transport._pool, "_retries", 0) == 5
        finally:
            asyncio.run(client.close())

    def test_get_httpx_client_with_max_retries(self):
        """Test that _get_httpx_client creates client with max_retries when provided."""
        handler = _get_httpx_client(params={"max_retries": 4, "timeout": 30})
        try:
            transport = handler.client._transport
            assert isinstance(transport, httpx.HTTPTransport)
            assert getattr(transport._pool, "_retries", 0) == 4
        finally:
            handler.close()

    def test_get_async_httpx_client_without_max_retries(self):
        """Test that get_async_httpx_client creates client without max_retries when not provided."""
        from litellm.types.utils import LlmProviders
        from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport
        import asyncio

        client = get_async_httpx_client(llm_provider=LlmProviders.ANTHROPIC, params={})
        try:
            transport = client.client._transport
            # Should be LiteLLMAiohttpTransport when available (default)
            assert isinstance(transport, LiteLLMAiohttpTransport)
        finally:
            asyncio.run(client.close())
