import json

import pytest
import respx
import litellm
import httpx
from typing import Final


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
    respx_mock.post("https://api.mistral.ai/v1/chat/completions").respond(json=mistral_api_response)

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
async def test_mistral_transform_response_empty_content_conversion(
    sync_mode, respx_mock, mistral_api_response_with_empty_content
):
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
    respx_mock.post("https://api.mistral.ai/v1/chat/completions").respond(json=mistral_api_response_with_empty_content)

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
        {"role": "user", "content": "How are you?"},
    ]

    # Mock the Mistral API endpoint
    respx_mock.post("https://api.mistral.ai/v1/chat/completions").respond(json=mistral_api_response)

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

    request_data = json.loads(request.content.decode("utf-8"))

    # Verify that the 'name' field was removed from the user message
    # (Mistral API only supports 'name' in tool messages)
    user_message = request_data["messages"][0]
    assert user_message["role"] == "user"
    assert user_message["content"] == "Hello"
    assert "name" not in user_message  # The 'name' field should have been removed


def test_mistral_401_maps_to_authentication_error(respx_mock):
    route = respx_mock.post("https://api.mistral.ai/v1/chat/completions").mock(
        return_value=httpx.Response(401, json={"message": "invalid API key"})
    )

    with pytest.raises(litellm.AuthenticationError) as exc_info:
        litellm.completion(
            model="mistral/mistral-tiny",
            messages=[{"role": "user", "content": "hello"}],
            api_key="bad-mistral-key",
            max_retries=0,
        )

    assert exc_info.value.status_code == 401
    assert "invalid API key" in str(exc_info.value)
    assert route.calls.last.request.headers["Authorization"] == "Bearer bad-mistral-key"


def test_mistral_streaming_function_call_arguments_are_assembled(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    chunks: Final = (
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "mistral-medium-latest",
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "mistral-medium-latest",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_weather",
                                "type": "function",
                                "function": {
                                    "name": "get_current_weather",
                                    "arguments": '{"location":"',
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "mistral-medium-latest",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": 'Boston","unit":"fahrenheit"}'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "mistral-medium-latest",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        },
    )
    body: Final = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    respx_mock.post("https://api.mistral.ai/v1/chat/completions").respond(
        200,
        text=body,
        headers={"content-type": "text/event-stream"},
    )
    messages: Final = [{"role": "user", "content": "What is the weather in Boston?"}]
    stream: Final = litellm.completion(
        model="mistral/mistral-medium-latest",
        messages=messages,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {"type": "string"},
                        },
                        "required": ["location", "unit"],
                    },
                },
            }
        ],
        stream=True,
    )
    stream_chunks: Final = tuple(stream)
    assert {chunk._hidden_params["custom_llm_provider"] for chunk in stream_chunks} == {"mistral"}
    argument_fragments: Final = tuple(
        chunk.choices[0].delta.tool_calls[0].function.arguments
        for chunk in stream_chunks
        if chunk.choices and chunk.choices[0].delta.tool_calls is not None
    )
    assert "".join(argument or "" for argument in argument_fragments) == ('{"location":"Boston","unit":"fahrenheit"}')
    assembled: Final = litellm.stream_chunk_builder(chunks=list(stream_chunks), messages=messages)
    assert assembled is not None
    assert assembled.choices[0].message.tool_calls[0].function.name == "get_current_weather"
    assert json.loads(assembled.choices[0].message.tool_calls[0].function.arguments) == {
        "location": "Boston",
        "unit": "fahrenheit",
    }
    assert assembled.choices[0].finish_reason == "tool_calls"


def test_mistral_completion_cost_is_priced_from_the_returned_usage(
    respx_mock, mistral_api_response, local_model_cost_map
):
    route: Final = respx_mock.post("https://api.mistral.ai/v1/chat/completions").respond(json=mistral_api_response)

    response: Final = litellm.completion(
        model="mistral/mistral-medium-latest",
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
        max_tokens=5,
        seed=10,
        input_cost_per_token=4e-07,
        output_cost_per_token=2e-06,
    )

    body: Final = json.loads(route.calls.last.request.content)
    assert (body["random_seed"], body["max_tokens"]) == (10, 5)
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (10, 15)
    assert response._hidden_params["response_cost"] == pytest.approx(10 * 4e-07 + 15 * 2e-06)
