import sys, os
import pytest
import litellm
import httpx

def test_neosantara_provider_info():
    """
    Test that Neosantara is correctly recognized as a provider and maps to the correct base URL.
    """
    model, provider, api_key, api_base = litellm.get_llm_provider("neosantara/claude-3-haiku")
    
    assert provider == "neosantara"
    assert api_base == "https://api.neosantara.xyz/v1"
    assert model == "claude-3-haiku"

def test_neosantara_completion_formatting():
    """
    Test that a completion call to Neosantara formats the request correctly as an OpenAI-like call.
    """
    litellm.set_verbose = True
    os.environ["NEOSANTARA_API_KEY"] = "sk-1234"
    
    # We mock the actual call to avoid network requests
    with pytest.MonkeyPatch().context() as m:
        def mock_send(self, request, **kwargs):
            return httpx.Response(
                200,
                content='{"choices": [{"message": {"content": "Hello world"}, "finish_reason": "stop", "index": 0}], "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}, "object": "chat.completion"}'.encode(),
                request=request
            )

        m.setattr("httpx.Client.send", mock_send)

        response = litellm.completion(
            model="neosantara/claude-3-haiku",
            messages=[{"role": "user", "content": "hi"}],
            api_key="sk-1234"
        )
        
        assert response.choices[0].message.content == "Hello world"
        assert response._hidden_params["custom_llm_provider"] == "neosantara"
        assert response._hidden_params["api_base"] == "https://api.neosantara.xyz/v1"

def test_neosantara_embedding_formatting():
    """
    Test that an embedding call to Neosantara formats the request correctly as an OpenAI-like call.
    """
    litellm.set_verbose = True
    os.environ["NEOSANTARA_API_KEY"] = "sk-1234"
    
    # We mock the actual call to avoid network requests
    with pytest.MonkeyPatch().context() as m:
        def mock_send(self, request, **kwargs):
            return httpx.Response(
                200,
                content='{"data": [{"embedding": [0.1, 0.2, 0.3], "index": 0, "object": "embedding"}], "model": "nusa-embedding-0001", "object": "list", "usage": {"prompt_tokens": 10, "total_tokens": 10}}'.encode(),
                request=request
            )

        m.setattr("httpx.Client.send", mock_send)

        response = litellm.embedding(
            model="neosantara/nusa-embedding-0001",
            input=["hi"],
            api_key="sk-1234"
        )
        
        assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]
        assert response._hidden_params["custom_llm_provider"] == "neosantara"
        assert response._hidden_params["api_base"] == "https://api.neosantara.xyz/v1"

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

def test_neosantara_responses_api_bridge():
    """
    Test that Neosantara works with litellm.responses() API bridge.
    """
    os.environ["NEOSANTARA_API_KEY"] = "sk-1234"
    
    with pytest.MonkeyPatch().context() as m:
        def mock_send(self, request, **kwargs):
            return httpx.Response(
                200,
                content='{"choices": [{"message": {"content": "Hello from responses API"}, "finish_reason": "stop", "index": 0}], "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}, "object": "chat.completion", "id": "resp-123", "created": 123456789}'.encode(),
                request=request
            )

        m.setattr("httpx.Client.send", mock_send)

        response = litellm.responses(
            model="neosantara/claude-3-haiku",
            input="hi",
            api_key="sk-1234"
        )
        
        assert response.model == "neosantara/claude-3-haiku"
        assert response.output[0].content[0].text == "Hello from responses API"

@pytest.mark.asyncio
async def test_neosantara_anthropic_messages_bridge():
    """
    Test that Neosantara works with litellm.anthropic_messages() API bridge.
    """
    os.environ["NEOSANTARA_API_KEY"] = "sk-1234"
    
    with pytest.MonkeyPatch().context() as m:
        async def mock_async_send(self, request, **kwargs):
            return httpx.Response(
                200,
                content='{"choices": [{"message": {"content": "Hello from messages API"}, "finish_reason": "stop", "index": 0}], "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}, "object": "chat.completion", "id": "msg-123", "created": 123456789}'.encode(),
                request=request
            )

        m.setattr("httpx.AsyncClient.send", mock_async_send)

        response = await litellm.anthropic_messages(
            model="neosantara/claude-3-haiku",
            messages=[{"role": "user", "content": "hi"}],
            api_key="sk-1234",
            max_tokens=100
        )
        
        # Check both dict and object access to be robust
        if isinstance(response, dict):
            assert response["content"][0].text == "Hello from messages API"
            assert response["role"] == "assistant"
        else:
            assert response.content[0].text == "Hello from messages API"
            assert response.role == "assistant"
