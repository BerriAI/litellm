import os
import sys

import pytest

from litellm.utils import supports_url_context

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
        return {"model": "gemini/gemini-2.0-flash"}

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
        model="gemini-1.5-pro", messages=messages_no_ttl, cache_key="test-no-ttl"
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
        model="gemini-1.5-pro", messages=messages_mixed, cache_key="test-mixed-ttl"
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
    assert response.choices[0].message.image is not None
    assert response.choices[0].message.image["url"] is not None
    assert response.choices[0].message.image["url"].startswith("data:image/png;base64,")


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
        model="gemini/gemini-1.5-pro",
        messages=[{"role": "user", "content": "give me 3 random words"}],
        max_tokens=2,
    )
    print(response)
    assert response.choices[0].finish_reason is not None
    assert response.choices[0].finish_reason == "length"


def test_gemini_url_context():
    from litellm import completion

    litellm._turn_on_debug()

    url = "https://ai.google.dev/gemini-api/docs/models"
    prompt = f"""
    Summarize this document:
    {url}
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
    assert urlMetadata["retrievedUrl"] == url
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

    IMAGE_URL = response.choices[0].message.image
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
            hasattr(chunk.choices[0].delta, "image")
            and chunk.choices[0].delta.image is not None
        ):
            model_response_image = chunk.choices[0].delta.image
            print("MODEL_RESPONSE_IMAGE: ", model_response_image)
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
