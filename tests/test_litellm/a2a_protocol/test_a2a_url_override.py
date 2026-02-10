import pytest
from unittest.mock import MagicMock, patch, AsyncMock, ANY
from litellm.a2a_protocol.main import create_a2a_client

@pytest.mark.asyncio
async def test_create_a2a_client_url_override():
    """
    Test that create_a2a_client passes the base_url as an override to the A2A SDK's A2AClient.
    This ensures that the registered URL is used even if the agent card reports a different one.
    """
    base_url = "http://my-registered-agent:8080"
    broken_card_url = "http://[::]:8081/invoke"
    
    # Mock agent card
    mock_agent_card = MagicMock()
    mock_agent_card.url = broken_card_url
    mock_agent_card.name = "test-agent"
    
    # Mock A2ACardResolver
    with patch("litellm.a2a_protocol.main.A2ACardResolver") as mock_resolver_cls:
        mock_resolver = mock_resolver_cls.return_value
        mock_resolver.get_agent_card = AsyncMock(return_value=mock_agent_card)
        
        # Mock _A2AClient (the SDK client)
        with patch("litellm.a2a_protocol.main._A2AClient") as mock_sdk_client_cls:
            # Mock successful import if not available
            with patch("litellm.a2a_protocol.main.A2A_SDK_AVAILABLE", True):
                client = await create_a2a_client(base_url=base_url)
                
                # Verify SDK client was initialized with the override URL
                mock_sdk_client_cls.assert_called_once_with(
                    httpx_client=ANY, # We don't care about the exact client here
                    agent_card=mock_agent_card,
                    url=base_url # THIS IS THE CRITICAL ASSERTION
                )
                
                assert client == mock_sdk_client_cls.return_value

@pytest.fixture(autouse=True)
def mock_httpx_client():
    with patch("litellm.a2a_protocol.main.get_async_httpx_client") as mock_get_httpx:
        mock_handler = MagicMock()
        mock_handler.client = MagicMock()
        mock_get_httpx.return_value = mock_handler
        yield mock_handler.client
