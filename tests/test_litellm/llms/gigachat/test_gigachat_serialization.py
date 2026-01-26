
import json
import pytest
from unittest.mock import Mock, MagicMock, patch
from litellm.llms.gigachat.chat.transformation import GigaChatConfig

def test_gigachat_serialization_utf8():
    """
    Verify that GigaChatConfig returns correct json_dumps_params
    and that the serialization preserves UTF-8.
    """
    config = GigaChatConfig()
    params = config.get_json_dumps_params()
    assert params == {"ensure_ascii": False}
    
    # Test actual serialization with non-ASCII characters
    data = {"message": "Привет, мир!"} # "Hello, world!" in Russian
    serialized = json.dumps(data, **params)
    assert "Привет, мир!" in serialized
    # Verify that default serialization would escape it
    default_serialized = json.dumps(data)
    assert "\\u041f\\u0440\\u0438\\u0432\\u0435\\u0442" in default_serialized

@patch("litellm.llms.custom_httpx.llm_http_handler.get_async_httpx_client")
@pytest.mark.asyncio
async def test_make_common_async_call_serialization(mock_get_client):
    """
    Verify that BaseLLMHTTPHandler uses the provider's json_dumps_params.
    """
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
    from unittest.mock import AsyncMock
    import httpx
    
    handler = BaseLLMHTTPHandler()
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    
    # Mock response
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_client.post = AsyncMock(return_value=mock_response)
    
    provider_config = GigaChatConfig()
    data = {"messages": [{"role": "user", "content": "Привет"}]}
    
    # We need to mock a few more things or provide valid arguments for litellm_params and logging_obj
    await handler._make_common_async_call(
        async_httpx_client=mock_client,
        provider_config=provider_config,
        api_base="https://api.example.com",
        headers={},
        data=data,
        timeout=10,
        litellm_params={},
        logging_obj=Mock()
    )
    
    # Check that post was called with serialized data
    args, kwargs = mock_client.post.call_args
    sent_data = kwargs["data"]
    # Should contain literal "Привет" because ensure_ascii=False
    assert "Привет" in sent_data
    assert "\\u041f" not in sent_data
