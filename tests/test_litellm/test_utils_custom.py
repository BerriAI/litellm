import pytest
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
    
    # Mock anthropic
    with patch("anthropic.AsyncAnthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        
        # Mock response
        mock_response = MagicMock()
        mock_response.input_tokens = 10
        
        # Setup async return for count_tokens
        mock_client.beta.messages.count_tokens = AsyncMock(return_value=mock_response)
        
        # First call
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": api_key}):
            await count_tokens_with_anthropic_api(model, messages)
            
        assert api_key in _anthropic_async_clients
        assert _anthropic_async_clients[api_key] == mock_client
        mock_cls.assert_called_once() # Should be called once
        
        # Second call
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": api_key}):
            await count_tokens_with_anthropic_api(model, messages)
            
        # Should still be called once (cached)
        mock_cls.assert_called_once()
