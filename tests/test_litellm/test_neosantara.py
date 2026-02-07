import sys, os
import pytest
import litellm
import httpx
import json
from unittest.mock import MagicMock, patch

def test_neosantara_provider_info():
    """
    Test that Neosantara is correctly recognized as a provider and maps to the correct base URL.
    """
    model, provider, api_key, api_base = litellm.get_llm_provider("neosantara/claude-3-haiku")
    
    assert provider == "neosantara"
    assert api_base == "https://api.neosantara.xyz/v1"
    assert model == "claude-3-haiku"

def test_neosantara_auto_detection_api_base():
    """
    Test that Neosantara is automatically detected when the Neosantara API base URL is provided.
    """
    model, provider, api_key, api_base = litellm.get_llm_provider(
        model="claude-3-haiku", 
        api_base="https://api.neosantara.xyz/v1"
    )
    
    assert provider == "neosantara"
    assert api_base == "https://api.neosantara.xyz/v1"
    assert model == "claude-3-haiku"

def test_neosantara_auto_detection_api_base_no_protocol():
    """
    Test that Neosantara is automatically detected when the Neosantara API base URL is provided without protocol.
    """
    model, provider, api_key, api_base = litellm.get_llm_provider(
        model="claude-3-haiku", 
        api_base="api.neosantara.xyz/v1"
    )
    
    assert provider == "neosantara"
    assert api_base == "api.neosantara.xyz/v1"
    assert model == "claude-3-haiku"

@patch("httpx.Client.send")
def test_neosantara_completion_formatting(mock_send):
    """
    Test that a completion call to Neosantara formats the request correctly as an OpenAI-like call.
    """
    os.environ["NEOSANTARA_API_KEY"] = "sk-1234"
    
    def side_effect(request, **kwargs):
        # Verify request URL and headers
        assert "api.neosantara.xyz" in str(request.url)
        assert request.headers["Authorization"] == "Bearer sk-1234"
        
        # Verify request body is OpenAI format
        body = json.loads(request.read())
        assert "messages" in body
        assert body["model"] == "claude-3-haiku"

        return httpx.Response(
            200,
            content='{"id": "chatcmpl-123", "choices": [{"message": {"content": "Hello world", "role": "assistant"}, "finish_reason": "stop", "index": 0}], "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}, "object": "chat.completion", "created": 1677610602, "model": "claude-3-haiku"}'.encode(),
            request=request
        )

    mock_send.side_effect = side_effect

    response = litellm.completion(
        model="neosantara/claude-3-haiku",
        messages=[{"role": "user", "content": "hi"}],
        api_key="sk-1234"
    )
    
    assert response.choices[0].message.content == "Hello world"
    assert response._hidden_params["custom_llm_provider"] == "neosantara"
    assert response._hidden_params["api_base"] == "https://api.neosantara.xyz/v1"

@patch("httpx.Client.send")
def test_neosantara_embedding_formatting(mock_send):
    """
    Test that an embedding call to Neosantara formats the request correctly as an OpenAI-like call.
    """
    os.environ["NEOSANTARA_API_KEY"] = "sk-1234"
    
    def side_effect(request, **kwargs):
        # Verify request URL
        assert "api.neosantara.xyz" in str(request.url)
        assert "/v1/embeddings" in str(request.url)
        
        return httpx.Response(
            200,
            content='{"data": [{"embedding": [0.1, 0.2, 0.3], "index": 0, "object": "embedding"}], "model": "nusa-embedding-0001", "object": "list", "usage": {"prompt_tokens": 10, "total_tokens": 10}}'.encode(),
            request=request
        )

    mock_send.side_effect = side_effect

    response = litellm.embedding(
        model="neosantara/nusa-embedding-0001",
        input=["hi"],
        api_key="sk-1234"
    )
    
    assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]
    assert response._hidden_params["custom_llm_provider"] == "neosantara"
    assert response._hidden_params["api_base"] == "https://api.neosantara.xyz/v1"

@patch("httpx.Client.send")
def test_neosantara_responses_api_bridge(mock_send):
    """
    Test that Neosantara works with litellm.responses() API bridge.
    """
    os.environ["NEOSANTARA_API_KEY"] = "sk-1234"
    
    def side_effect(request, **kwargs):
        # Verify the bridge transformed 'input' into OpenAI 'messages'
        body = json.loads(request.read())
        assert "messages" in body
        assert body["messages"][0]["content"] == "hi from responses"
        assert "/v1/chat/completions" in str(request.url)

        return httpx.Response(
            200,
            content='{"id": "chatcmpl-123", "choices": [{"message": {"content": "Hello from bridge", "role": "assistant"}, "finish_reason": "stop", "index": 0}], "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}, "object": "chat.completion", "created": 1677610602, "model": "claude-3-haiku"}'.encode(),
            request=request
        )

    mock_send.side_effect = side_effect

    response = litellm.responses(
        model="neosantara/claude-3-haiku",
        input="hi from responses",
        api_key="sk-1234"
    )
    
    # Verify result is transformed to Responses API schema
    # Bridge transforms choices[0].message.content -> output[0].content[0].text
    assert response.model == "neosantara/claude-3-haiku"
    assert response.output[0].content[0].text == "Hello from bridge"
    assert hasattr(response, "created_at")
    assert response.created_at == 1677610602

@pytest.mark.anyio
async def test_neosantara_anthropic_messages_bridge():
    """
    Test that Neosantara works with litellm.anthropic_messages() API bridge.
    """
    os.environ["NEOSANTARA_API_KEY"] = "sk-1234"
    
    with patch("httpx.AsyncClient.send") as mock_async_send:
        async def side_effect(request, **kwargs):
            # Verify the bridge transformed Anthropic messages to OpenAI messages
            body = json.loads(request.read())
            assert "messages" in body
            assert body["messages"][0]["role"] == "user"
            assert "/v1/chat/completions" in str(request.url)

            return httpx.Response(
                200,
                content='{"id": "chatcmpl-123", "choices": [{"message": {"content": "Hello from anthropic bridge", "role": "assistant"}, "finish_reason": "stop", "index": 0}], "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}, "object": "chat.completion", "created": 1677610602, "model": "claude-3-haiku"}'.encode(),
                request=request
            )

        mock_async_send.side_effect = side_effect

        response = await litellm.anthropic_messages(
            model="neosantara/claude-3-haiku",
            messages=[{"role": "user", "content": "hi from anthropic"}],
            api_key="sk-1234",
            max_tokens=100
        )
        
        # Verify result is transformed to Anthropic Messages API schema
        # Bridge transforms choices[0].message.content -> content[0].text
        # If result is an object (AnthropicMessagesResponse), access via .content[0].text
        # If result is a dict, access via ["content"][0]["text"]
        if isinstance(response, dict):
            assert response["model"] == "neosantara/claude-3-haiku"
            assert response["role"] == "assistant"
            # Access text via object attribute if subscripting fails
            content_block = response["content"][0]
            if hasattr(content_block, "text"):
                assert content_block.text == "Hello from anthropic bridge"
            else:
                assert content_block["text"] == "Hello from anthropic bridge"
            assert response["type"] == "message"
            assert response["usage"]["input_tokens"] == 10
            assert response["usage"]["output_tokens"] == 10
        else:
            assert response.model == "neosantara/claude-3-haiku"
            assert response.role == "assistant"
            assert response.content[0].text == "Hello from anthropic bridge"
            assert response.type == "message"
            assert response.usage.input_tokens == 10
            assert response.usage.output_tokens == 10