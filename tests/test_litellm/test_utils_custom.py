import pytest
import sys
from unittest.mock import MagicMock, patch, AsyncMock
from litellm.proxy.utils import count_tokens_with_anthropic_api, _anthropic_async_clients

@pytest.mark.asyncio
async def test_count_tokens_caching():
    """
    Test that count_tokens_with_anthropic_api caches the client.
    """
    # Clear cache
    _anthropic_async_clients.clear()
    
    api_key = "sk-ant-test-key"
    messages = [{"role": "user", "content": "hello"}]
    model = "claude-3-opus-20240229"
    
    # Create a mock anthropic module
    mock_anthropic = MagicMock()
    mock_client = MagicMock()
    mock_anthropic.AsyncAnthropic.return_value = mock_client
    
    # Mock response
    mock_response = MagicMock()
    mock_response.input_tokens = 10
    
    # Setup async return for count_tokens
    mock_client.beta.messages.count_tokens = AsyncMock(return_value=mock_response)
    
    # Patch sys.modules to ensure our mock is used when anthropic is imported
    with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
        # First call
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": api_key}):
            await count_tokens_with_anthropic_api(model, messages)
            
        assert api_key in _anthropic_async_clients
        assert _anthropic_async_clients[api_key] == mock_client
        mock_anthropic.AsyncAnthropic.assert_called_once() # Should be called once
        
        # Second call
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": api_key}):
            await count_tokens_with_anthropic_api(model, messages)
            
        # Should still be called once (cached)
        mock_anthropic.AsyncAnthropic.assert_called_once()
