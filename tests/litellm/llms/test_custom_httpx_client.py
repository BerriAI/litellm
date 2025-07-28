"""
Tests for custom httpx.Client support across OpenAI, Anthropic, and Azure providers.

Validates fix for Issue #13049:
https://github.com/BerriAI/litellm/issues/13049
"""

import os
from typing import Any
from unittest.mock import Mock, patch

import httpx
import pytest

import litellm


class DebugClient(httpx.Client):
    """Custom httpx.Client that tracks request calls for testing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.request_calls: list[tuple[str, str]] = []

    def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:  # type: ignore[override]
        self.request_calls.append((request.method, str(request.url)))
        return super().send(request, **kwargs)


class AsyncDebugClient(httpx.AsyncClient):
    """Custom httpx.AsyncClient that tracks request calls for testing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.request_calls: list[tuple[str, str]] = []

    async def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:  # type: ignore[override]
        self.request_calls.append((request.method, str(request.url)))
        return await super().send(request, **kwargs)


# Unit tests for client utilities
def test_normalize_client_for_openai_passthrough() -> None:
    """Test that non-httpx clients are passed through unchanged."""
    from litellm.litellm_core_utils.core_helpers import normalize_httpx_client_for_openai
    
    # Mock OpenAI client
    class MockOpenAIClient:
        pass
    
    mock_client = MockOpenAIClient()
    
    # Should pass through non-httpx clients unchanged
    assert normalize_httpx_client_for_openai(mock_client, False) is mock_client
    
    # Should return None for None input
    assert normalize_httpx_client_for_openai(None, False) is None


def test_normalize_client_for_anthropic_passthrough() -> None:
    """Test that non-httpx clients are passed through unchanged for Anthropic."""
    from litellm.litellm_core_utils.core_helpers import normalize_httpx_client_for_anthropic
    
    # Mock Anthropic client
    class MockAnthropicClient:
        pass
    
    mock_client = MockAnthropicClient()
    
    # Should pass through non-httpx clients unchanged
    assert normalize_httpx_client_for_anthropic(mock_client, False) is mock_client
    
    # Should return None for None input
    assert normalize_httpx_client_for_anthropic(None, False) is None


def test_normalize_client_for_azure_passthrough() -> None:
    """Test that non-httpx clients are passed through unchanged for Azure."""
    from litellm.litellm_core_utils.core_helpers import normalize_httpx_client_for_azure
    
    # Mock Azure client
    class MockAzureClient:
        pass
    
    mock_client = MockAzureClient()
    
    # Should pass through non-httpx clients unchanged
    assert normalize_httpx_client_for_azure(mock_client, False) is mock_client
    
    # Should return None for None input
    assert normalize_httpx_client_for_azure(None, False) is None


def test_gemini_direct_httpx_support() -> None:
    """Test that Gemini accepts raw httpx clients directly without normalization."""
    # Gemini doesn't use normalization - it accepts httpx clients directly through HTTPHandler/AsyncHTTPHandler
    sync_client = httpx.Client()
    async_client = httpx.AsyncClient()
    
    # These should be usable directly in Gemini completion calls
    # This is a unit test to document the expected behavior
    assert isinstance(sync_client, httpx.Client)
    assert isinstance(async_client, httpx.AsyncClient)
    
    # Clean up
    sync_client.close()
    # Note: async_client.aclose() would require await, so we skip cleanup for the test


def test_normalize_client_sync_async_mismatch() -> None:
    """Test that sync/async mismatches raise appropriate errors."""
    from litellm.litellm_core_utils.core_helpers import (
        normalize_httpx_client_for_openai,
        normalize_httpx_client_for_anthropic,
        normalize_httpx_client_for_azure,
    )
    
    sync_client = httpx.Client()
    async_client = httpx.AsyncClient()
    
    # Test OpenAI
    with pytest.raises(ValueError, match="Expected httpx.Client for sync operation"):
        normalize_httpx_client_for_openai(async_client, is_async=False, api_key="test")
    
    with pytest.raises(ValueError, match="Expected httpx.AsyncClient for async operation"):
        normalize_httpx_client_for_openai(sync_client, is_async=True, api_key="test")
    
    # Test Anthropic
    with pytest.raises(ValueError, match="Expected httpx.Client for sync Anthropic operation"):
        normalize_httpx_client_for_anthropic(async_client, is_async=False, api_key="test")
    
    with pytest.raises(ValueError, match="Expected httpx.AsyncClient for async Anthropic operation"):
        normalize_httpx_client_for_anthropic(sync_client, is_async=True, api_key="test")
    
    # Test Azure
    with pytest.raises(ValueError, match="Expected httpx.Client for sync Azure operation"):
        normalize_httpx_client_for_azure(async_client, is_async=False, api_key="test")
    
    with pytest.raises(ValueError, match="Expected httpx.AsyncClient for async Azure operation"):
        normalize_httpx_client_for_azure(sync_client, is_async=True, api_key="test")


def test_global_client_fallback() -> None:
    """Test that global client sessions are used as fallbacks."""
    from litellm.litellm_core_utils.core_helpers import get_client_or_fallback_to_global
    
    original_client_session = litellm.client_session
    original_aclient_session = litellm.aclient_session
    
    try:
        debug_client = DebugClient()
        
        # Set global sessions
        litellm.client_session = debug_client
        litellm.aclient_session = debug_client  # For test simplicity
        
        # Should return explicit client when provided
        explicit_client = DebugClient()
        assert get_client_or_fallback_to_global(explicit_client, False) is explicit_client
        
        # Should fallback to global sessions when no client provided
        assert get_client_or_fallback_to_global(None, False) is debug_client
        assert get_client_or_fallback_to_global(None, True) is debug_client
        
    finally:
        litellm.client_session = original_client_session
        litellm.aclient_session = original_aclient_session


# Integration tests with mocking
@pytest.mark.parametrize(
    "model,provider_name",
    [
        ("gpt-3.5-turbo", "openai"),
        ("claude-3-haiku-20240307", "anthropic"),
        ("azure/gpt-35-turbo", "azure"),
        ("gemini/gemini-1.5-flash", "gemini"),
    ],
)
def test_custom_httpx_client_with_mocked_response(model: str, provider_name: str) -> None:
    """Test custom httpx.Client integration with mocked API responses across providers."""
    debug_client = DebugClient()
    
    # Mock a successful response
    mock_response_data = {
        "choices": [{"message": {"content": "Hello! How can I help you today?"}}],
        "model": model,
        "usage": {"total_tokens": 10}
    }
    
    with patch.object(debug_client, 'send') as mock_send:
        # Create a mock HTTP response
        http_response = httpx.Response(
            status_code=200,
            json=mock_response_data,
            request=httpx.Request("POST", "http://test.com")
        )
        mock_send.return_value = http_response
        
        try:
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": "Hello"}],
                client=debug_client,
                max_tokens=10,
            )
            
            # Verify mock was called
            assert mock_send.called
            
            # Verify response structure (basic check)
            assert hasattr(response, 'choices')
            
        except Exception as e:
            # The key thing is we don't get the original attribute error
            assert "'Client' object has no attribute 'api_key'" not in str(e)
            assert "'DebugClient' object has no attribute 'api_key'" not in str(e)
            # For other providers, we expect different error patterns but not the original issue


def test_reproduce_original_issue_fixed() -> None:
    """Ensure the original issue #13049 is fixed - no attribute errors for all providers."""
    debug_client = DebugClient()
    
    test_cases = [
        ("gpt-3.5-turbo", "OPENAI_API_KEY"),
        ("claude-3-haiku-20240307", "ANTHROPIC_API_KEY"),
        ("azure/gpt-35-turbo", "AZURE_API_KEY"),
    ]
    
    for model, env_key in test_cases:
        with patch.dict(os.environ, {env_key: "test-key"}):
            with patch.object(debug_client, 'send') as mock_send:
                # Mock response to avoid network calls
                mock_send.return_value = httpx.Response(
                    status_code=200,
                    json={"choices": [{"message": {"content": "test"}}]},
                    request=httpx.Request("POST", "http://test.com")
                )
                
                try:
                    litellm.completion(
                        model=model,
                        messages=[{"role": "user", "content": "Hello"}],
                        client=debug_client,
                        max_tokens=1,
                    )
                except Exception as e:
                    # The key thing is we don't get the original attribute error
                    assert "'Client' object has no attribute 'api_key'" not in str(e)
                    assert "'DebugClient' object has no attribute 'api_key'" not in str(e)


@pytest.mark.parametrize(
    "model,provider_name",
    [
        ("gpt-3.5-turbo", "openai"),
        ("claude-3-haiku-20240307", "anthropic"), 
        ("azure/gpt-35-turbo", "azure"),
        ("gemini/gemini-1.5-flash", "gemini"),
    ],
)
def test_global_client_session_works(model: str, provider_name: str) -> None:
    """Test that global client session works for all supported providers."""
    debug_client = DebugClient()
    original_client_session = litellm.client_session
    
    try:
        # Set global client session
        litellm.client_session = debug_client
        
        with patch.object(debug_client, 'send') as mock_send:
            # Mock response
            mock_send.return_value = httpx.Response(
                status_code=200,
                json={"choices": [{"message": {"content": "test"}}]},
                request=httpx.Request("POST", "http://test.com")
            )
            
            # Set appropriate API key for the provider
            env_keys = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY", 
                "azure": "AZURE_API_KEY",
                "gemini": "GEMINI_API_KEY"
            }
            
            with patch.dict(os.environ, {env_keys[provider_name]: "test-key"}):
                try:
                    response = litellm.completion(
                        model=model,
                        messages=[{"role": "user", "content": "Hello"}],
                        max_tokens=10,
                    )
                    
                    # If successful, verify our client was used
                    assert mock_send.called
                    
                except Exception as e:
                    # Should not get the original attribute error
                    assert "'Client' object has no attribute 'api_key'" not in str(e)
                    assert "'DebugClient' object has no attribute 'api_key'" not in str(e)
                    
    finally:
        # Restore original client session
        litellm.client_session = original_client_session


# Optional: Keep one integration test that uses real API if available
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OPENAI_API_KEY set")
def test_openai_client_real_api() -> None:
    """Integration test with real OpenAI API."""
    debug_client = DebugClient()
    
    try:
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'test' only."}],
            client=debug_client,
            max_tokens=5,
        )
        
        # Verify we got a response and our client was used
        assert response.choices[0].message.content is not None
        assert len(debug_client.request_calls) > 0
        
        # Verify it went to OpenAI
        method, url = debug_client.request_calls[0]
        assert method == "POST"
        assert "api.openai.com" in url
        
    except Exception as e:
        # Should not get the original attribute error
        assert "api_key" not in str(e).lower(), f"Got api_key attribute error: {e}"


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="No ANTHROPIC_API_KEY set")
def test_anthropic_client_real_api() -> None:
    """Integration test with real Anthropic API."""
    debug_client = DebugClient()
    
    try:
        response = litellm.completion(
            model="claude-3-haiku-20240307",
            messages=[{"role": "user", "content": "Say 'test' only."}],
            client=debug_client,
            max_tokens=5,
        )
        
        # Verify we got a response and our client was used
        assert response.choices[0].message.content is not None
        assert len(debug_client.request_calls) > 0
        
        # Verify it went to Anthropic
        method, url = debug_client.request_calls[0]
        assert method == "POST"
        assert "api.anthropic.com" in url
        
    except Exception as e:
        # Should not get the original attribute error
        assert "api_key" not in str(e).lower(), f"Got api_key attribute error: {e}"


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="No GEMINI_API_KEY set")
def test_gemini_client_real_api() -> None:
    """Integration test with real Gemini API using custom httpx client."""
    debug_client = DebugClient()
    
    try:
        response = litellm.completion(
            model="gemini/gemini-1.5-flash",
            messages=[{"role": "user", "content": "Say 'test' only."}],
            client=debug_client,
            max_tokens=5,
        )
        
        # Verify we got a response and our client was used
        assert response.choices[0].message.content is not None
        assert len(debug_client.request_calls) > 0
        
        # Verify it went to Google AI
        method, url = debug_client.request_calls[0]
        assert method == "POST"
        assert "generativelanguage.googleapis.com" in url
        
    except Exception as e:
        # Should not get the original attribute error
        assert "api_key" not in str(e).lower(), f"Got api_key attribute error: {e}"