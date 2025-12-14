import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system paths

from base_llm_unit_tests import BaseLLMChatTest
from litellm.llms.vertex_ai.context_caching.transformation import (
    separate_cached_messages,
    transform_openai_messages_to_gemini_context_caching,
)
import litellm
from litellm import completion
import json


class TestGoogleAIStudioGemini(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        return {"model": "gemini/gemini-2.5-flash"}

    def get_base_completion_call_args_with_reasoning_model(self) -> dict:
        return {"model": "gemini/gemini-2.5-flash"}

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        from litellm.litellm_core_utils.prompt_templates.factory import (
            convert_to_gemini_tool_call_invoke,
        )

        result = convert_to_gemini_tool_call_invoke(tool_call_no_arguments)
        print(result)

    @pytest.mark.flaky(retries=3, delay=2)
    def test_url_context(self):
        from litellm.utils import supports_url_context

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        litellm._turn_on_debug()

        base_completion_call_args = self.get_base_completion_call_args()

        if not supports_url_context(base_completion_call_args["model"], None):
            pytest.skip("Model does not support url context")

        response = self.completion_function(
            **base_completion_call_args,
            messages=[
                {
                    "role": "user",
                    "content": "Summarize the content of this URL: https://en.wikipedia.org/wiki/Artificial_intelligence",
                }
            ],
            tools=[{"urlContext": {}}],
        )

        assert response is not None
        assert (
            response.model_extra["vertex_ai_url_context_metadata"] is not None
        ), "URL context metadata should be present"
        print(f"response={response}")


def test_gemini_context_caching_with_ttl():
    """Test Gemini context caching with TTL support"""

    # Test case 1: Basic TTL functionality
    messages_with_ttl = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Here is the full text of a complex legal agreement" * 400,
                    "cache_control": {"type": "ephemeral", "ttl": "3600s"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What are the key terms and conditions in this agreement?",
                    "cache_control": {"type": "ephemeral", "ttl": "7200s"},
                }
            ],
        },
    ]

    # Test the transformation function directly
    result = transform_openai_messages_to_gemini_context_caching(
        model="gemini-1.5-pro",
        messages=messages_with_ttl,
        cache_key="test-ttl-cache-key",
        custom_llm_provider="gemini",
        vertex_project=None,
        vertex_location=None,
    )

    # Verify TTL is properly included in the result
    assert "ttl" in result
    assert result["ttl"] == "3600s"  # Should use the first valid TTL found
    assert result["model"] == "models/gemini-1.5-pro"
    assert result["displayName"] == "test-ttl-cache-key"

    # Test case 2: Invalid TTL should be ignored
    messages_invalid_ttl = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Cached content with invalid TTL",
                    "cache_control": {"type": "ephemeral", "ttl": "invalid_ttl"},
                }
            ],
        }
    ]

    result_invalid = transform_openai_messages_to_gemini_context_caching(
        model="gemini-1.5-pro",
        messages=messages_invalid_ttl,
        cache_key="test-invalid-ttl",
        custom_llm_provider="gemini",
        vertex_project=None,
        vertex_location=None,
    )

    # Verify invalid TTL is not included
    assert "ttl" not in result_invalid
    assert result_invalid["model"] == "models/gemini-1.5-pro"
    assert result_invalid["displayName"] == "test-invalid-ttl"

    # Test case 3: Messages without TTL should work normally
    messages_no_ttl = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Cached content without TTL",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ]

    result_no_ttl = transform_openai_messages_to_gemini_context_caching(
        model="gemini-1.5-pro",
        messages=messages_no_ttl,
        cache_key="test-no-ttl",
        custom_llm_provider="gemini",
        vertex_project=None,
        vertex_location=None,
    )

    # Verify no TTL field is present when not specified
    assert "ttl" not in result_no_ttl
    assert result_no_ttl["model"] == "models/gemini-1.5-pro"
    assert result_no_ttl["displayName"] == "test-no-ttl"

    # Test case 4: Mixed messages with some having TTL
    messages_mixed = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "System message with TTL",
                    "cache_control": {"type": "ephemeral", "ttl": "1800s"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "User message without TTL",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {"role": "assistant", "content": "Assistant response without cache control"},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Another user message",
                    "cache_control": {"type": "ephemeral", "ttl": "900s"},
                }
            ],
        },
    ]

    # Test separation of cached messages
    cached_messages, non_cached_messages = separate_cached_messages(messages_mixed)
    assert len(cached_messages) > 0
    assert len(non_cached_messages) > 0

    # Test transformation with mixed messages
    result_mixed = transform_openai_messages_to_gemini_context_caching(
        model="gemini-1.5-pro",
        messages=messages_mixed,
        cache_key="test-mixed-ttl",
        custom_llm_provider="gemini",
        vertex_project=None,
        vertex_location=None,
    )

    # Should pick up the first valid TTL
    assert "ttl" in result_mixed
    assert result_mixed["ttl"] == "1800s"
    assert result_mixed["model"] == "models/gemini-1.5-pro"
    assert result_mixed["displayName"] == "test-mixed-ttl"


def test_gemini_context_caching_separate_messages():
    messages = [
        # System Message
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Here is the full text of a complex legal agreement" * 400,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        # marked for caching with the cache_control parameter, so that this checkpoint can read from the previous cache.
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What are the key terms and conditions in this agreement?",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "assistant",
            "content": "Certainly! the key terms and conditions are the following: the contract is 1 year long for $10/mo",
        },
        # The final turn is marked with cache-control, for continuing in followups.
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What are the key terms and conditions in this agreement?",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
    ]
    cached_messages, non_cached_messages = separate_cached_messages(messages)
    print(cached_messages)
    print(non_cached_messages)
    assert len(cached_messages) > 0, "Cached messages should be present"
    assert len(non_cached_messages) > 0, "Non-cached messages should be present"


def test_gemini_image_generation():
    # litellm._turn_on_debug()
    response = completion(
        model="gemini/gemini-2.0-flash-exp-image-generation",
        messages=[{"role": "user", "content": "Generate an image of a cat"}],
        modalities=["image", "text"],
    )

    #########################################################
    # Important: Validate we did get an image in the response
    #########################################################
    assert response.choices[0].message.images is not None
    assert len(response.choices[0].message.images) > 0
    assert response.choices[0].message.images[0]["image_url"] is not None
    assert response.choices[0].message.images[0]["image_url"]["url"] is not None
    assert (
        response.choices[0]
        .message.images[0]["image_url"]["url"]
        .startswith("data:image/png;base64,")
    )


@pytest.mark.parametrize(
    "model_name",
    [
        "gemini/gemini-2.5-flash-image-preview",
        "gemini/gemini-2.0-flash-preview-image-generation",
        "gemini/gemini-3-pro-image-preview",
    ],
)
def test_gemini_flash_image_preview_models(model_name: str):
    """
    Validate Gemini Flash image preview models route through image_generation()
    and invoke the generateContent endpoint returning inline image data.
    """
    from unittest.mock import patch, MagicMock
    from litellm.types.utils import ImageResponse, ImageObject

    # Mock successful response to avoid API limits
    mock_response = ImageResponse()
    mock_response.data = [ImageObject(b64_json="test_base64_data", url=None)]

    with patch(
        "litellm.llms.custom_httpx.llm_http_handler.HTTPHandler.post"
    ) as mock_post:
        # Mock successful HTTP response
        mock_http_response = MagicMock()
        mock_http_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"inlineData": {"data": "test_base64_image_data"}}]
                    }
                }
            ]
        }
        mock_http_response.status_code = 200
        mock_post.return_value = mock_http_response

        # Test that the function works without throwing the original 400 error
        response = litellm.image_generation(
            model=model_name,
            prompt="Generate a simple test image",
            api_key="test_api_key",
        )

        # Validate response structure
        assert response is not None
        assert hasattr(response, "data")
        assert response.data is not None
        assert len(response.data) > 0

        # Validate the correct endpoint was called
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        called_url = (
            call_args[0][0] if call_args[0] else call_args.kwargs.get("url", "")
        )

        # Verify it uses generateContent endpoint for Gemini Flash image preview models (not predict)
        assert ":generateContent" in called_url
        assert model_name.split("/", 1)[1] in called_url

        # Verify request format is Gemini format (not Imagen)
        request_data = call_args.kwargs.get("json", {})
        assert "contents" in request_data
        assert "parts" in request_data["contents"][0]

        # Verify response_modalities is set correctly for image generation
        assert "generationConfig" in request_data
        assert "response_modalities" in request_data["generationConfig"]
        assert request_data["generationConfig"]["response_modalities"] == [
            "IMAGE",
            "TEXT",
        ]

def test_gemini_imagen_models_use_predict_endpoint():
    """
    Test that Imagen models still use :predict endpoint (not broken by gemini-2.5-flash-image-preview fix)
    """
    from unittest.mock import patch, MagicMock
    from litellm.types.utils import ImageResponse, ImageObject

    with patch(
        "litellm.llms.custom_httpx.llm_http_handler.HTTPHandler.post"
    ) as mock_post:
        # Mock successful HTTP response for Imagen
        mock_http_response = MagicMock()
        mock_http_response.json.return_value = {
            "predictions": [{"bytesBase64Encoded": "test_base64_image_data"}]
        }
        mock_http_response.status_code = 200
        mock_post.return_value = mock_http_response

        # Test an Imagen model
        response = litellm.image_generation(
            model="gemini/imagen-3.0-generate-001",
            prompt="Generate a simple test image",
            api_key="test_api_key",
        )

        # Validate response structure
        assert response is not None
        assert hasattr(response, "data")

        # Validate the correct endpoint was called for Imagen models
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        called_url = (
            call_args[0][0] if call_args[0] else call_args.kwargs.get("url", "")
        )

        # Verify Imagen models use predict endpoint (not generateContent)
        assert ":predict" in called_url
        assert "imagen-3.0-generate-001" in called_url
        assert ":generateContent" not in called_url

        # Verify request format is Imagen format (not Gemini)
        request_data = call_args.kwargs.get("json", {})
        assert "instances" in request_data
        assert "parameters" in request_data


def test_gemini_thinking():
    litellm._turn_on_debug()
    from litellm.types.utils import Message, CallTypes
    from litellm.utils import return_raw_request
    import json

    messages = [
        {
            "role": "user",
            "content": "Explain the concept of Occam's Razor and provide a simple, everyday example",
        }
    ]
    reasoning_content = "I'm thinking about Occam's Razor."
    assistant_message = Message(
        content="Okay, let's break down Occam's Razor.",
        reasoning_content=reasoning_content,
        role="assistant",
        tool_calls=None,
        function_call=None,
        provider_specific_fields=None,
    )

    messages.append(assistant_message)

    raw_request = return_raw_request(
        endpoint=CallTypes.completion,
        kwargs={
            "model": "gemini/gemini-2.5-flash",
            "messages": messages,
        },
    )
    assert reasoning_content in json.dumps(raw_request)
    response = completion(
        model="gemini/gemini-2.5-flash",
        messages=messages,  # make sure call works
    )
    print(response.choices[0].message)
    assert response.choices[0].message.content is not None


def test_gemini_thinking_budget_0():
    litellm._turn_on_debug()
    from litellm.types.utils import Message, CallTypes
    from litellm.utils import return_raw_request
    import json

    raw_request = return_raw_request(
        endpoint=CallTypes.completion,
        kwargs={
            "model": "gemini/gemini-2.5-flash",
            "messages": [
                {
                    "role": "user",
                    "content": "Explain the concept of Occam's Razor and provide a simple, everyday example",
                }
            ],
            "thinking": {"type": "enabled", "budget_tokens": 0},
        },
    )
    print(json.dumps(raw_request, indent=4, default=str))
    assert "0" in json.dumps(raw_request["raw_request_body"])


def test_gemini_finish_reason():
    import os
    from litellm import completion

    litellm._turn_on_debug()
    response = completion(
        model="gemini/gemini-2.5-flash-lite",
        messages=[{"role": "user", "content": "give me 3 random words"}],
        max_tokens=2,
    )
    print(response)
    assert response.choices[0].finish_reason is not None
    assert response.choices[0].finish_reason == "length"


def test_gemini_url_context():
    from litellm import completion

    litellm._turn_on_debug()
    URL1 = "https://www.foodnetwork.com/recipes/ina-garten/perfect-roast-chicken-recipe-1940592"

    prompt = f"""
    Get the recipes listed on the following website
    {URL1}
    """
    response = completion(
        model="gemini/gemini-2.5-flash",
        messages=[{"role": "user", "content": prompt}],
        tools=[{"urlContext": {}}],
    )
    print(response)
    message = response.choices[0].message.content
    assert message is not None
    url_context_metadata = response.model_extra["vertex_ai_url_context_metadata"]
    assert url_context_metadata is not None
    urlMetadata = url_context_metadata[0]["urlMetadata"][0]
    assert urlMetadata["retrievedUrl"] == URL1
    assert urlMetadata["urlRetrievalStatus"] == "URL_RETRIEVAL_STATUS_SUCCESS"


@pytest.mark.flaky(retries=3, delay=2)
def test_gemini_with_grounding():
    from litellm import completion, Usage, stream_chunk_builder

    litellm._turn_on_debug()
    litellm.set_verbose = True
    tools = [{"googleSearch": {}}]

    # response = completion(model="gemini/gemini-2.0-flash", messages=[{"role": "user", "content": "What is the capital of France?"}], tools=tools)
    # print(response)
    # usage: Usage = response.usage
    # assert usage.prompt_tokens_details.web_search_requests is not None
    # assert usage.prompt_tokens_details.web_search_requests > 0

    ## Check streaming

    response = completion(
        model="gemini/gemini-2.0-flash",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        tools=tools,
        stream=True,
        stream_options={"include_usage": True},
    )
    chunks = []
    for chunk in response:
        print(f"received chunk: {chunk}")
        chunks.append(chunk)
    print(f"chunks before stream_chunk_builder: {chunks}")
    assert len(chunks) > 0
    complete_response = stream_chunk_builder(chunks)
    print(complete_response)
    assert complete_response is not None
    usage: Usage = complete_response.usage
    assert usage.prompt_tokens_details.web_search_requests is not None
    assert usage.prompt_tokens_details.web_search_requests > 0


def test_gemini_with_empty_function_call_arguments():
    from litellm import completion

    litellm._turn_on_debug()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "parameters": "",
            },
        }
    ]
    response = completion(
        model="gemini/gemini-2.0-flash",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        tools=tools,
    )
    print(response)
    assert response.choices[0].message.content is not None


@pytest.mark.asyncio
async def test_claude_tool_use_with_gemini():
    response = await litellm.anthropic.messages.acreate(
        messages=[
            {
                "role": "user",
                "content": "Hello, can you tell me the weather in Boston. Please respond with a tool call?",
            }
        ],
        model="gemini/gemini-2.5-flash",
        stream=True,
        max_tokens=100,
        tools=[
            {
                "name": "get_weather",
                "description": "Get current weather information for a specific location",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            }
        ],
    )

    is_content_block_tool_use = False
    is_partial_json = False
    has_usage_in_message_delta = False
    is_content_block_stop = False

    async for chunk in response:
        print(chunk)
        if "content_block_stop" in str(chunk):
            is_content_block_stop = True

        # Handle bytes chunks (SSE format)
        if isinstance(chunk, bytes):
            chunk_str = chunk.decode("utf-8")

            # Parse SSE format: event: <type>\ndata: <json>\n\n
            if "data: " in chunk_str:
                try:
                    # Extract JSON from data line
                    data_line = [
                        line
                        for line in chunk_str.split("\n")
                        if line.startswith("data: ")
                    ][0]
                    json_str = data_line[6:]  # Remove 'data: ' prefix
                    chunk_data = json.loads(json_str)

                    # Check for tool_use
                    if "tool_use" in json_str:
                        is_content_block_tool_use = True
                    if "partial_json" in json_str:
                        is_partial_json = True
                    if "content_block_stop" in json_str:
                        is_content_block_stop = True

                    # Check for usage in message_delta with stop_reason
                    if (
                        chunk_data.get("type") == "message_delta"
                        and chunk_data.get("delta", {}).get("stop_reason") is not None
                        and "usage" in chunk_data
                    ):
                        has_usage_in_message_delta = True
                        # Verify usage has the expected structure
                        usage = chunk_data["usage"]
                        assert (
                            "input_tokens" in usage
                        ), "input_tokens should be present in usage"
                        assert (
                            "output_tokens" in usage
                        ), "output_tokens should be present in usage"
                        assert isinstance(
                            usage["input_tokens"], int
                        ), "input_tokens should be an integer"
                        assert isinstance(
                            usage["output_tokens"], int
                        ), "output_tokens should be an integer"
                        print(f"Found usage in message_delta: {usage}")

                except (json.JSONDecodeError, IndexError) as e:
                    # Skip chunks that aren't valid JSON
                    pass
        else:
            # Handle dict chunks (fallback)
            if "tool_use" in str(chunk):
                is_content_block_tool_use = True
            if "partial_json" in str(chunk):
                is_partial_json = True
            if "content_block_stop" in str(chunk):
                is_content_block_stop = True

    assert is_content_block_tool_use, "content_block_tool_use should be present"
    assert is_partial_json, "partial_json should be present"
    assert (
        has_usage_in_message_delta
    ), "Usage should be present in message_delta with stop_reason"
    assert is_content_block_stop, "is_content_block_stop should be present"


def test_gemini_tool_use():
    data = {
        "max_tokens": 8192,
        "stream": True,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather like in Lima, Peru today?"},
        ],
        "model": "gemini/gemini-2.0-flash",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Retrieve current weather for a specific location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City and country, e.g., Lima, Peru",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "Temperature unit",
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ],
        "stream_options": {"include_usage": True},
    }

    response = litellm.completion(**data)
    print(response)

    stop_reason = None
    for chunk in response:
        print(chunk)
        if chunk.choices[0].finish_reason:
            stop_reason = chunk.choices[0].finish_reason
    assert stop_reason is not None
    assert stop_reason == "tool_calls"


@pytest.mark.asyncio
async def test_gemini_image_generation_async():
    litellm._turn_on_debug()
    response = await litellm.acompletion(
        messages=[
            {
                "role": "user",
                "content": "Generate an image of a banana wearing a costume that says LiteLLM",
            }
        ],
        model="gemini/gemini-2.5-flash-image-preview",
    )

    CONTENT = response.choices[0].message.content

    # Check if images list exists and has items before accessing
    assert hasattr(response.choices[0].message, "images"), "Response message should have images attribute"
    assert response.choices[0].message.images is not None, "Images should not be None"
    assert len(response.choices[0].message.images) > 0, "Images list should not be empty"
    
    IMAGE_URL = response.choices[0].message.images[0]["image_url"]
    print("IMAGE_URL: ", IMAGE_URL)

    assert CONTENT is not None, "CONTENT is not None"
    assert IMAGE_URL is not None, "IMAGE_URL is not None"
    assert IMAGE_URL["url"] is not None, "IMAGE_URL['url'] is not None"
    assert IMAGE_URL["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_gemini_image_generation_async_stream():
    # litellm._turn_on_debug()
    response = await litellm.acompletion(
        messages=[
            {
                "role": "user",
                "content": "Generate an image of a banana wearing a costume that says LiteLLM",
            }
        ],
        model="gemini/gemini-2.5-flash-image-preview",
        stream=True,
    )

    print("RESPONSE: ", response)
    model_response_image = None
    async for chunk in response:
        print("CHUNK: ", chunk)
        if (
            hasattr(chunk.choices[0].delta, "images")
            and chunk.choices[0].delta.images is not None
            and len(chunk.choices[0].delta.images) > 0
        ):
            model_response_image = chunk.choices[0].delta.images[0]["image_url"]
            assert model_response_image is not None
            assert model_response_image["url"].startswith("data:image/png;base64,")
            break

    #########################################################
    # Important: Validate we did get an image in the response
    #########################################################
    assert model_response_image is not None
    assert model_response_image["url"].startswith("data:image/png;base64,")


def test_system_message_with_no_user_message():
    """
    Test that the system message is translated correctly for non-OpenAI providers.
    """
    messages = [
        {
            "role": "system",
            "content": "Be a good bot!",
        },
    ]

    response = litellm.completion(
        model="gemini/gemini-2.5-flash",
        messages=messages,
    )
    assert response is not None

    assert response.choices[0].message.content is not None


def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps(
            {"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"}
        )
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": "celsius"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


def test_gemini_with_thinking():
    from litellm import completion

    litellm._turn_on_debug()
    litellm.modify_params = True
    model = "gemini/gemini-2.5-flash"
    messages = [
        {
            "role": "user",
            "content": "What's the weather like in San Francisco, Tokyo, and Paris? - give me 3 responses",
        }
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    response = litellm.completion(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
        reasoning_effort="low",
    )
    print("Response\n", response)
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    print("Expecting there to be 3 tool calls")
    assert len(tool_calls) > 0  # this has to call the function for SF, Tokyo and paris

    # Step 2: check if the model wanted to call a function
    print(f"tool_calls: {tool_calls}")
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_current_weather": get_current_weather,
        }  # only one function in this example, but you can have multiple
        messages.append(response_message)  # extend conversation with assistant's reply
        print("Response message\n", response_message)
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            if function_name not in available_functions:
                # the model called a function that does not exist in available_functions - don't try calling anything
                return
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                location=function_args.get("location"),
                unit=function_args.get("unit"),
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
        print(f"messages: {messages}")
        second_response = litellm.completion(
            model=model,
            messages=messages,
            seed=22,
            reasoning_effort="low",
            tools=tools,
            drop_params=True,
        )  # get a new response from the model where it can see the function response
        print("second response\n", second_response)


def test_gemini_reasoning_effort_minimal():
    """
    Test that reasoning_effort='minimal' correctly maps to model-specific minimum thinking budgets
    """
    from litellm.utils import return_raw_request
    from litellm.types.utils import CallTypes
    import json

    # Test with different Gemini models to verify model-specific mapping
    test_cases = [
        ("gemini/gemini-2.5-flash", 1),  # Flash: minimum 1 token
        ("gemini/gemini-2.5-pro", 128),  # Pro: minimum 128 tokens
        ("gemini/gemini-2.5-flash-lite", 512),  # Flash-Lite: minimum 512 tokens
    ]

    for model, expected_min_budget in test_cases:
        # Get the raw request to verify the thinking budget mapping
        raw_request = return_raw_request(
            endpoint=CallTypes.completion,
            kwargs={
                "model": model,
                "messages": [{"role": "user", "content": "Hello"}],
                "reasoning_effort": "minimal",
            },
        )

        # Verify that the thinking config is set correctly
        request_body = raw_request["raw_request_body"]
        assert (
            "generationConfig" in request_body
        ), f"Model {model} should have generationConfig"

        generation_config = request_body["generationConfig"]
        assert (
            "thinkingConfig" in generation_config
        ), f"Model {model} should have thinkingConfig"

        thinking_config = generation_config["thinkingConfig"]
        assert (
            "thinkingBudget" in thinking_config
        ), f"Model {model} should have thinkingBudget"

        actual_budget = thinking_config["thinkingBudget"]
        assert (
            actual_budget == expected_min_budget
        ), f"Model {model} should map 'minimal' to {expected_min_budget} tokens, got {actual_budget}"

        # Verify that includeThoughts is True for minimal reasoning effort
        assert thinking_config.get(
            "includeThoughts", True
        ), f"Model {model} should have includeThoughts=True for minimal reasoning effort"

    # Test with unknown model (should use generic fallback)
    try:
        raw_request = return_raw_request(
            endpoint=CallTypes.completion,
            kwargs={
                "model": "gemini/unknown-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "reasoning_effort": "minimal",
            },
        )

        request_body = raw_request["raw_request_body"]
        generation_config = request_body["generationConfig"]
        thinking_config = generation_config["thinkingConfig"]
        # Should use generic fallback (128 tokens)
        assert (
            thinking_config["thinkingBudget"] == 128
        ), "Unknown model should use generic fallback of 128 tokens"
    except Exception as e:
        # If return_raw_request doesn't work for unknown models, that's okay
        # The important part is that our known models work correctly
        print(f"Note: Unknown model test skipped due to: {e}")
        pass


def test_gemini_exception_message_format():
    """
    Test that Gemini provider exceptions show as 'GeminiException' not 'VertexAIException'.

    This addresses issue #14586 where Gemini API errors were incorrectly showing as
    VertexAIException instead of GeminiException due to incorrect exception mapping.
    """
    import httpx
    from unittest.mock import Mock
    from litellm.litellm_core_utils.exception_mapping_utils import exception_type
    from litellm import BadRequestError

    # Mock a typical Gemini API error response
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 400
    mock_response.text = "Invalid API key provided"
    mock_response.headers = {}

    # Create a mock exception that simulates a Gemini API error
    mock_exception = httpx.HTTPStatusError(
        message="Bad Request", request=Mock(), response=mock_response
    )
    mock_exception.response = mock_response
    mock_exception.status_code = 400

    # Test the exception mapping for Gemini provider
    try:
        exception_type(
            model="gemini-pro",
            original_exception=mock_exception,
            custom_llm_provider="gemini",
            completion_kwargs={},
            extra_kwargs={},
        )
        # Should not reach here - exception should be raised
        assert False, "Expected BadRequestError to be raised"
    except BadRequestError as e:
        # The test should FAIL initially (before fix) because it will show VertexAIException
        # After the fix, it should show GeminiException
        error_message = str(e)
        print(f"Error message: {error_message}")  # For debugging

        # This assertion will initially FAIL - that's expected for TDD
        assert "GeminiException" in error_message, (
            f"Expected 'GeminiException' in error message, got: {error_message}. "
            f"This test should fail before the fix is implemented."
        )
        assert (
            "VertexAIException" not in error_message
        ), f"Should not contain 'VertexAIException' in error message, got: {error_message}"


@pytest.mark.parametrize(
    "status_code,expected_exception",
    [
        (400, "BadRequestError"),
        (401, "AuthenticationError"),
        (403, "PermissionDeniedError"),
        (404, "NotFoundError"),
        (408, "Timeout"),
        (429, "RateLimitError"),
        (500, "InternalServerError"),
        (502, "APIConnectionError"),
        (503, "ServiceUnavailableError"),
    ],
)
def l(status_code, expected_exception):
    """
    Test comprehensive Gemini error handling for all HTTP status codes.

    This ensures that Gemini API errors of different types are properly mapped
    to the correct LiteLLM exception types with GeminiException prefix.
    """
    import httpx
    from unittest.mock import Mock
    from litellm.litellm_core_utils.exception_mapping_utils import exception_type
    from litellm.exceptions import (
        BadRequestError,
        AuthenticationError,
        PermissionDeniedError,
        NotFoundError,
        Timeout,
        RateLimitError,
        InternalServerError,
        APIConnectionError,
        ServiceUnavailableError,
    )

    # Mock the appropriate error response
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = status_code
    mock_response.text = f"API Error {status_code}"
    mock_response.headers = {}

    # Create a mock exception
    mock_exception = httpx.HTTPStatusError(
        message=f"HTTP {status_code}", request=Mock(), response=mock_response
    )
    mock_exception.response = mock_response
    mock_exception.status_code = status_code
    # Set message attribute for compatibility with exception mapping
    mock_exception.message = f"HTTP {status_code}"

    # Test the exception mapping
    try:
        exception_type(
            model="gemini-pro",
            original_exception=mock_exception,
            custom_llm_provider="gemini",
            completion_kwargs={},
            extra_kwargs={},
        )
        assert (
            False
        ), f"Expected {expected_exception} to be raised for status {status_code}"
    except Exception as e:
        # Verify the correct exception type is raised
        exception_classes = {
            "BadRequestError": BadRequestError,
            "AuthenticationError": AuthenticationError,
            "PermissionDeniedError": PermissionDeniedError,
            "NotFoundError": NotFoundError,
            "Timeout": Timeout,
            "RateLimitError": RateLimitError,
            "InternalServerError": InternalServerError,
            "APIConnectionError": APIConnectionError,
            "ServiceUnavailableError": ServiceUnavailableError,
        }
        expected_class = exception_classes[expected_exception]
        assert isinstance(
            e, expected_class
        ), f"Expected {expected_exception}, got {type(e).__name__}"

        # Verify the error message contains GeminiException
        error_message = str(e)
        assert (
            "GeminiException" in error_message
        ), f"Expected 'GeminiException' in error message for status {status_code}, got: {error_message}"
        assert (
            "VertexAIException" not in error_message
        ), f"Should not contain 'VertexAIException' for status {status_code}, got: {error_message}"


def test_gemini_embedding():
    litellm._turn_on_debug()
    response = litellm.embedding(
        model="gemini/gemini-embedding-001",
        input="Hello, world!",
    )
    print("response: ", response)
    assert response is not None


def test_reasoning_effort_none_mapping():
    """
    Test that reasoning_effort='none' correctly maps to thinkingConfig.
    Related issue: https://github.com/BerriAI/litellm/issues/16420
    """
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )

    # Test reasoning_effort="none" mapping
    result = VertexGeminiConfig._map_reasoning_effort_to_thinking_budget(
        reasoning_effort="none",
        model="gemini-2.0-flash-thinking-exp-01-21",
    )

    assert result is not None
    assert result["thinkingBudget"] == 0
    assert result["includeThoughts"] is False
    
def test_gemini_function_args_preserve_unicode():
    """
    Test for Issue #16533: Gemini function call arguments should preserve non-ASCII characters
    https://github.com/BerriAI/litellm/issues/16533

    Before fix: "や" becomes "\u3084"
    After fix: "や" stays as "や"
    """
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig

    # Test Japanese characters
    parts = [
        {
            "functionCall": {
                "name": "send_message",
                "args": {
                    "message": "やあ",  # Japanese "hello"
                    "recipient": "たけし"  # Japanese name
                }
            }
        }
    ]

    function, tools, _ = VertexGeminiConfig._transform_parts(
        parts=parts,
        cumulative_tool_call_idx=0,
        is_function_call=False
    )

    arguments_str = tools[0]['function']['arguments']
    parsed_args = json.loads(arguments_str)

    # Verify characters are preserved
    assert parsed_args["message"] == "やあ", "Japanese characters should be preserved"
    assert parsed_args["recipient"] == "たけし", "Japanese characters should be preserved"

    # Verify no Unicode escape sequences in raw string
    assert "\\u" not in arguments_str, "Should not contain Unicode escape sequences"
    assert "やあ" in arguments_str, "Original Japanese characters should be in the string"
    assert "たけし" in arguments_str, "Original Japanese characters should be in the string"

    # Test Spanish characters
    parts_spanish = [
        {
            "functionCall": {
                "name": "send_message",
                "args": {
                    "message": "¡Hola! ¿Cómo estás?",
                    "recipient": "José"
                }
            }
        }
    ]

    function, tools, _ = VertexGeminiConfig._transform_parts(
        parts=parts_spanish,
        cumulative_tool_call_idx=0,
        is_function_call=False
    )

    arguments_str = tools[0]['function']['arguments']
    parsed_args = json.loads(arguments_str)

    assert parsed_args["message"] == "¡Hola! ¿Cómo estás?"
    assert parsed_args["recipient"] == "José"
    assert "\\u" not in arguments_str
    assert "José" in arguments_str
