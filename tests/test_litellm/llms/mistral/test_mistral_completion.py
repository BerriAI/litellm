import pytest
import litellm


@pytest.fixture(autouse=True)
def add_mistral_api_key_to_env(monkeypatch):
    """Add Mistral API key to environment for testing."""
    monkeypatch.setenv("MISTRAL_API_KEY", "fake-mistral-api-key-12345")


@pytest.fixture
def mistral_api_response():
    """Mock response data for Mistral API calls."""
    return {
        "id": "chatcmpl-mistral-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "mistral-medium-latest",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello from Mistral! How can I help you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
    }


@pytest.fixture
def mistral_api_response_with_empty_content():
    """Mock response data for Mistral API calls with empty content that should be converted to None."""
    return {
        "id": "chatcmpl-mistral-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "mistral-medium-latest",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",  # Empty string that should be converted to None
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    }


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_mistral_basic_completion(sync_mode, respx_mock, mistral_api_response):
    """Test basic Mistral completion functionality."""
    litellm.disable_aiohttp_transport = True
    
    model = "mistral/mistral-medium-latest"
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    
    # Mock the Mistral API endpoint
    respx_mock.post("https://api.mistral.ai/v1/chat/completions").respond(
        json=mistral_api_response
    )
    
    if sync_mode:
        response = litellm.completion(model=model, messages=messages)
    else:
        response = await litellm.acompletion(model=model, messages=messages)
    
    # Verify response
    assert response.choices[0].message.content == "Hello from Mistral! How can I help you today?"
    assert response.model == "mistral-medium-latest"
    assert response.usage.total_tokens == 25


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_mistral_transform_response_empty_content_conversion(sync_mode, respx_mock, mistral_api_response_with_empty_content):
    """
    Test that Mistral's transform_response method is being called by verifying
    the specific behavior of converting empty string content to None.
    
    This test verifies that the _handle_empty_content_response method in
    MistralConfig.transform_response is being applied.
    """
    litellm.disable_aiohttp_transport = True
    
    model = "mistral/mistral-medium-latest"
    messages = [{"role": "user", "content": "Generate an empty response"}]
    
    # Mock the Mistral API endpoint with empty content
    respx_mock.post("https://api.mistral.ai/v1/chat/completions").respond(
        json=mistral_api_response_with_empty_content
    )
    
    if sync_mode:
        response = litellm.completion(model=model, messages=messages)
    else:
        response = await litellm.acompletion(model=model, messages=messages)
    
    # Verify that the transform_response method was called by checking that
    # empty string content was converted to None (Mistral-specific behavior)
    assert response.choices[0].message.content is None
    assert response.model == "mistral-medium-latest"
    assert response.usage.total_tokens == 10


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_mistral_transform_request_name_field_removal(sync_mode, respx_mock, mistral_api_response):
    """
    Test that Mistral's transform_request method is being called by verifying
    the specific behavior of removing the 'name' field from non-tool messages.
    
    This test verifies that the _handle_name_in_message method in
    MistralConfig._transform_messages is being applied.
    """
    litellm.disable_aiohttp_transport = True
    
    model = "mistral/mistral-medium-latest"
    # Include a message with 'name' field that should be removed for non-tool messages
    messages = [
        {"role": "user", "content": "Hello", "name": "should_be_removed"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]
    
    # Mock the Mistral API endpoint
    respx_mock.post("https://api.mistral.ai/v1/chat/completions").respond(
        json=mistral_api_response
    )
    
    if sync_mode:
        response = litellm.completion(model=model, messages=messages)
    else:
        response = await litellm.acompletion(model=model, messages=messages)
    
    # Verify the response works (if transform_request wasn't called, the API would reject the request)
    assert response.choices[0].message.content == "Hello from Mistral! How can I help you today?"
    assert response.model == "mistral-medium-latest"
    
    # Verify that the request was made (if transform_request failed, this would fail)
    assert len(respx_mock.calls) == 1
    
    # Get the actual request that was made
    request = respx_mock.calls[0].request
    import json
    request_data = json.loads(request.content.decode('utf-8'))
    
    # Verify that the 'name' field was removed from the user message
    # (Mistral API only supports 'name' in tool messages)
    user_message = request_data["messages"][0]
    assert user_message["role"] == "user"
    assert user_message["content"] == "Hello"
    assert "name" not in user_message  # The 'name' field should have been removed

