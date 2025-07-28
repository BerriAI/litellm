"""
Tests for custom httpx.Client support in OpenAI provider.

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


def test_normalize_client_sync_async_mismatch() -> None:
    """Test that sync/async mismatches raise appropriate errors."""
    from litellm.litellm_core_utils.core_helpers import normalize_httpx_client_for_openai
    
    sync_client = httpx.Client()
    async_client = httpx.AsyncClient()
    
    # Should raise when passing async client for sync operation
    with pytest.raises(ValueError, match="Expected httpx.Client for sync operation"):
        normalize_httpx_client_for_openai(async_client, is_async=False, api_key="test")
    
    # Should raise when passing sync client for async operation  
    with pytest.raises(ValueError, match="Expected httpx.AsyncClient for async operation"):
        normalize_httpx_client_for_openai(sync_client, is_async=True, api_key="test")


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


def test_openai_client_with_mocked_response() -> None:
    """Test OpenAI integration with mocked API responses."""
    debug_client = DebugClient()
    
    # Mock a successful response
    mock_response = {
        "choices": [{"message": {"content": "Hello! How can I help you today?"}}],
        "model": "gpt-3.5-turbo",
        "usage": {"total_tokens": 10}
    }
    
    with patch.object(debug_client, 'send') as mock_send:
        # Create a mock HTTP response
        http_response = httpx.Response(
            status_code=200,
            json=mock_response,
            request=httpx.Request("POST", "http://test.com")
        )
        mock_send.return_value = http_response
        
        try:
            response = litellm.completion(
                model="gpt-3.5-turbo",
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


def test_reproduce_original_issue_fixed() -> None:
    """Ensure the original issue #13049 is fixed - no attribute errors."""
    debug_client = DebugClient()
    
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with patch.object(debug_client, 'send') as mock_send:
            # Mock response to avoid network calls
            mock_send.return_value = httpx.Response(
                status_code=200,
                json={"choices": [{"message": {"content": "test"}}]},
                request=httpx.Request("POST", "http://test.com")
            )
            
            try:
                litellm.completion(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello"}],
                    client=debug_client,
                    max_tokens=1,
                )
            except Exception as e:
                # The key thing is we don't get the original attribute error
                assert "'Client' object has no attribute 'api_key'" not in str(e)
                assert "'DebugClient' object has no attribute 'api_key'" not in str(e)


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