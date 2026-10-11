import os

import pytest

from base_llm_unit_tests import BaseLLMChatTest
from litellm.llms.vertex_ai.context_caching.transformation import (
    separate_cached_messages,
    transform_openai_messages_to_gemini_context_caching,
)
import litellm
from litellm import completion
import json

class TestGoogleAIStudioGemini(BaseLLMChatTest):
    test_async_pdf_handling_with_file_id = None
    test_function_calling_with_tool_response = None
    test_web_search = None

    def get_base_completion_call_args(self) -> dict:
        return {"model": "gemini/gemini-2.5-flash"}

    def get_base_completion_call_args_with_reasoning_model(self) -> dict:
        return {"model": "gemini/gemini-2.5-flash"}

    @pytest.mark.flaky(retries=3, delay=2)
    def test_url_context(self):
        from litellm.utils import supports_url_context

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        litellm.turn_on_debug()

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

def test_gemini_image_generation():
    # litellm.turn_on_debug()
    response = completion(
        model="gemini/gemini-2.5-flash-image",
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

def test_gemini_thinking():
    litellm.turn_on_debug()
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

def test_gemini_finish_reason():
    import os
    from litellm import completion

    litellm.turn_on_debug()
    response = completion(
        model="gemini/gemini-2.5-flash-lite",
        messages=[{"role": "user", "content": "give me 3 random words"}],
        max_tokens=2,
    )
    print(response)
    assert response.choices[0].finish_reason is not None
    assert response.choices[0].finish_reason == "length"

@pytest.mark.flaky(retries=3, delay=2)
def test_gemini_url_context():
    from litellm import completion

    litellm.turn_on_debug()
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

    litellm.turn_on_debug()
    litellm.set_verbose = True
    tools = [{"googleSearch": {}}]

    # response = completion(model="gemini/gemini-2.0-flash", messages=[{"role": "user", "content": "What is the capital of France?"}], tools=tools)
    # print(response)
    # usage: Usage = response.usage
    # assert usage.prompt_tokens_details.web_search_requests is not None
    # assert usage.prompt_tokens_details.web_search_requests > 0

    ## Check streaming

    response = completion(
        model="gemini/gemini-2.5-flash",
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

    litellm.turn_on_debug()
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
        model="gemini/gemini-2.5-flash",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        tools=tools,
    )
    print(response)
    assert response.choices[0].message.content is not None

def test_gemini_tool_use():
    data = {
        "max_tokens": 8192,
        "stream": True,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather like in Lima, Peru today?"},
        ],
        "model": "gemini/gemini-2.5-flash",
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
    litellm.turn_on_debug()
    response = await litellm.acompletion(
        messages=[
            {
                "role": "user",
                "content": "Generate an image of a banana wearing a costume that says LiteLLM",
            }
        ],
        model="gemini/gemini-2.5-flash-image",
    )

    CONTENT = response.choices[0].message.content

    # Check if images list exists and has items before accessing
    assert hasattr(
        response.choices[0].message, "images"
    ), "Response message should have images attribute"
    assert response.choices[0].message.images is not None, "Images should not be None"
    assert (
        len(response.choices[0].message.images) > 0
    ), "Images list should not be empty"

    IMAGE_URL = response.choices[0].message.images[0]["image_url"]
    print("IMAGE_URL: ", IMAGE_URL)

    # content may be None when the model returns only an image with no text
    assert IMAGE_URL is not None, "IMAGE_URL is not None"
    assert IMAGE_URL["url"] is not None, "IMAGE_URL['url'] is not None"
    assert IMAGE_URL["url"].startswith("data:image/png;base64,")

@pytest.mark.asyncio
async def test_gemini_image_generation_async_stream():
    # litellm.turn_on_debug()
    response = await litellm.acompletion(
        messages=[
            {
                "role": "user",
                "content": "Generate an image of a banana wearing a costume that says LiteLLM",
            }
        ],
        model="gemini/gemini-2.5-flash-image",
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

    litellm.turn_on_debug()
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

    # Test the exception mapping
    with pytest.raises(expected_class) as exc_info:
        exception_type(
            model="gemini-pro",
            original_exception=mock_exception,
            custom_llm_provider="gemini",
            completion_kwargs={},
            extra_kwargs={},
        )
    e = exc_info.value

    # Verify the error message contains GeminiException
    error_message = str(e)
    assert (
        "GeminiException" in error_message
    ), f"Expected 'GeminiException' in error message for status {status_code}, got: {error_message}"
    assert (
        "VertexAIException" not in error_message
    ), f"Should not contain 'VertexAIException' for status {status_code}, got: {error_message}"

def test_gemini_embedding():
    litellm.turn_on_debug()
    response = litellm.embedding(
        model="gemini/gemini-embedding-001",
        input="Hello, world!",
    )
    print("response: ", response)
    assert response is not None

@pytest.mark.asyncio
async def test_gemini_openai_web_search_tool_to_google_search():
    """
    Test that OpenAI-style web_search tools are transformed to Gemini's googleSearch.

    When passing {"type": "web_search"} or {"type": "web_search_preview"} to Gemini,
    these should be transformed to googleSearch, not silently ignored.
    """
    response = await litellm.acompletion(
        model="gemini/gemini-2.5-flash",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        tools=[{"type": "web_search"}],
    )
    print("response: ", response.model_dump_json(indent=4))
    assert hasattr(response, "vertex_ai_grounding_metadata")
    assert getattr(response, "vertex_ai_grounding_metadata") is not None
