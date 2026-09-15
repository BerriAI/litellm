from unittest.mock import AsyncMock, patch

import pytest

from litellm.llms.anthropic.count_tokens.handler import AnthropicCountTokensHandler


@pytest.mark.asyncio
async def test_anthropic_count_tokens_custom_api_base():
    handler = AnthropicCountTokensHandler()

    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"input_tokens": 10}

    with patch("litellm.llms.anthropic.count_tokens.handler.get_async_httpx_client") as mock_client_factory:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client_factory.return_value = mock_client

        # Test custom api_base URL resolution and formatting
        res = await handler.handle_count_tokens_request(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "hello"}],
            api_key="test-key",
            api_base="https://custom-proxy.internal.com/v1",
        )

        assert res["input_tokens"] == 10
        mock_client.post.assert_called_once()
        called_url = mock_client.post.call_args[0][0]
        assert called_url == "https://custom-proxy.internal.com/v1/messages/count_tokens"
