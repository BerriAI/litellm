import os
import pytest
import litellm
import httpx
import json
from unittest.mock import MagicMock, patch
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from openai.types.create_embedding_response import CreateEmbeddingResponse
from openai.types.embedding import Embedding

# This class mimics the structure of the `APIResponse` object returned by `openai`'s `with_raw_response`
class MockAPIResponse:
    def __init__(self, mock_obj):
        self._mock_obj = mock_obj
        self.headers = {"x-request-id": "some-id"}

    def parse(self):
        return self._mock_obj

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

@patch("openai.resources.chat.completions.Completions.create")
def test_neosantara_completion_formatting(mock_create):
    """
    Test that a completion call to Neosantara formats the request correctly as an OpenAI-like call.
    """
    os.environ["NEOSANTARA_API_KEY"] = "sk-1234"

    # Define the mock response object to be returned by the patched method
    mock_choice = Choice(
        finish_reason="stop",
        index=0,
        message=ChatCompletionMessage(
            content="Hello world",
            role="assistant",
            function_call=None,
            tool_calls=None,
        )
    )
    mock_usage = CompletionUsage(
        completion_tokens=10,
        prompt_tokens=10,
        total_tokens=20
    )
    mock_chat_completion = ChatCompletion(
        id="chatcmpl-123",
        choices=[mock_choice],
        created=1677610602,
        model="claude-3-haiku",
        object="chat.completion",
        usage=mock_usage,
        system_fingerprint=None
    )

    mock_create.return_value = MockAPIResponse(mock_chat_completion)

    response = litellm.completion(
        model="neosantara/claude-3-haiku",
        messages=[{"role": "user", "content": "hi"}],
        api_key="sk-1234"
    )

    # Assert that the mocked 'create' method was called with the correct parameters
    mock_create.assert_called_once()
    called_args, called_kwargs = mock_create.call_args
    assert called_kwargs["model"] == "claude-3-haiku"
    assert called_kwargs["messages"] == [{"role": "user", "content": "hi"}]

    # Assertions on the transformed response
    assert response.choices[0].message.content == "Hello world"
    assert response._hidden_params["custom_llm_provider"] == "neosantara"
    assert response._hidden_params["api_base"] == "https://api.neosantara.xyz/v1"
@patch("openai.resources.embeddings.Embeddings.create")
def test_neosantara_embedding_formatting(mock_create):
    """
    Test that an embedding call to Neosantara formats the request correctly as an OpenAI-like call.
    """
    os.environ["NEOSANTARA_API_KEY"] = "sk-1234"

    mock_embedding = Embedding(
        embedding=[0.1, 0.2, 0.3],
        index=0,
        object="embedding"
    )
    mock_response = CreateEmbeddingResponse(
        data=[mock_embedding],
        model="nusa-embedding-0001",
        object="list",
        usage={"prompt_tokens": 10, "total_tokens": 10}
    )

    mock_create.return_value = MockAPIResponse(mock_response)

    response = litellm.embedding(
        model="neosantara/nusa-embedding-0001",
        input=["hi"],
        api_key="sk-1234"
    )

    assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]
    assert response._hidden_params["custom_llm_provider"] == "neosantara"
    assert response._hidden_params["api_base"] == "https://api.neosantara.xyz/v1"
@patch("openai.resources.chat.completions.Completions.create")
def test_neosantara_responses_api_bridge(mock_create):
    """
    Test that Neosantara works with litellm.responses() API bridge.
    """
    os.environ["NEOSANTARA_API_KEY"] = "sk-1234"
    
    # Define the mock response object to be returned by the patched method
    mock_choice = Choice(
        finish_reason="stop",
        index=0,
        message=ChatCompletionMessage(
            content="Hello from bridge",
            role="assistant",
            function_call=None,
            tool_calls=None,
        )
    )
    mock_usage = CompletionUsage(
        completion_tokens=10,
        prompt_tokens=10,
        total_tokens=20
    )
    mock_chat_completion = ChatCompletion(
        id="chatcmpl-123",
        choices=[mock_choice],
        created=1677610602,
        model="claude-3-haiku",
        object="chat.completion",
        usage=mock_usage,
        system_fingerprint=None
    )

    mock_create.return_value = MockAPIResponse(mock_chat_completion)

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
@patch("openai.resources.chat.completions.AsyncCompletions.create")
async def test_neosantara_anthropic_messages_bridge(mock_async_create):
    """
    Test that Neosantara works with litellm.anthropic_messages() API bridge.
    """
    os.environ["NEOSANTARA_API_KEY"] = "sk-1234"
    
    # Define the mock response object to be returned by the patched method
    mock_choice = Choice(
        finish_reason="stop",
        index=0,
        message=ChatCompletionMessage(
            content="Hello from anthropic bridge",
            role="assistant",
            function_call=None,
            tool_calls=None,
        )
    )
    mock_usage = CompletionUsage(
        completion_tokens=10,
        prompt_tokens=10,
        total_tokens=20
    )
    mock_chat_completion = ChatCompletion(
        id="chatcmpl-123",
        choices=[mock_choice],
        created=1677610602,
        model="claude-3-haiku",
        object="chat.completion",
        usage=mock_usage,
        system_fingerprint=None
    )

    # For async, the mock's return value should be an awaitable
    async def side_effect(*args, **kwargs):
        return MockAPIResponse(mock_chat_completion)

    mock_async_create.side_effect = side_effect

    response = await litellm.anthropic_messages(
        model="neosantara/claude-3-haiku",
        messages=[{"role": "user", "content": "hi from anthropic"}],
        api_key="sk-1234",
        max_tokens=100
    )
    
    # Verify result is transformed to Anthropic Messages API schema
    assert response["model"] == "neosantara/claude-3-haiku"
    assert response["role"] == "assistant"
    assert response["content"][0].text == "Hello from anthropic bridge"
    assert response["type"] == "message"
    assert response["usage"]["input_tokens"] == 10
    assert response["usage"]["output_tokens"] == 10