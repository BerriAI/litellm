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


GEMINI_3_IMAGE_SIZE_MAPPINGS = [
    ("512x512", "1:1", "512"),
    ("1024x1024", "1:1", "1K"),
    ("2048x2048", "1:1", "2K"),
    ("4096x4096", "1:1", "4K"),
    ("256x1024", "1:4", "512"),
    ("512x2048", "1:4", "1K"),
    ("1024x4096", "1:4", "2K"),
    ("2048x8192", "1:4", "4K"),
    ("192x1536", "1:8", "512"),
    ("384x3072", "1:8", "1K"),
    ("768x6144", "1:8", "2K"),
    ("1536x12288", "1:8", "4K"),
    ("424x632", "2:3", "512"),
    ("848x1264", "2:3", "1K"),
    ("1696x2528", "2:3", "2K"),
    ("3392x5056", "2:3", "4K"),
    ("632x424", "3:2", "512"),
    ("1264x848", "3:2", "1K"),
    ("2528x1696", "3:2", "2K"),
    ("5056x3392", "3:2", "4K"),
    ("448x600", "3:4", "512"),
    ("896x1200", "3:4", "1K"),
    ("1792x2400", "3:4", "2K"),
    ("3584x4800", "3:4", "4K"),
    ("1024x256", "4:1", "512"),
    ("2048x512", "4:1", "1K"),
    ("4096x1024", "4:1", "2K"),
    ("8192x2048", "4:1", "4K"),
    ("600x448", "4:3", "512"),
    ("1200x896", "4:3", "1K"),
    ("2400x1792", "4:3", "2K"),
    ("4800x3584", "4:3", "4K"),
    ("464x576", "4:5", "512"),
    ("928x1152", "4:5", "1K"),
    ("1856x2304", "4:5", "2K"),
    ("3712x4608", "4:5", "4K"),
    ("576x464", "5:4", "512"),
    ("1152x928", "5:4", "1K"),
    ("2304x1856", "5:4", "2K"),
    ("4608x3712", "5:4", "4K"),
    ("1536x192", "8:1", "512"),
    ("3072x384", "8:1", "1K"),
    ("6144x768", "8:1", "2K"),
    ("12288x1536", "8:1", "4K"),
    ("384x688", "9:16", "512"),
    ("768x1376", "9:16", "1K"),
    ("1536x2752", "9:16", "2K"),
    ("3072x5504", "9:16", "4K"),
    ("688x384", "16:9", "512"),
    ("1376x768", "16:9", "1K"),
    ("2752x1536", "16:9", "2K"),
    ("5504x3072", "16:9", "4K"),
    ("792x336", "21:9", "512"),
    ("1584x672", "21:9", "1K"),
    ("3168x1344", "21:9", "2K"),
    ("6336x2688", "21:9", "4K"),
]

class TestGoogleAIStudioGemini(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        return {"model": "gemini/gemini-2.5-flash"}

    def get_base_completion_call_args_with_reasoning_model(self) -> dict:
        return {"model": "gemini/gemini-2.5-flash"}

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


def test_gemini_image_generation():
    # litellm._turn_on_debug()
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


def test_gemini_finish_reason():
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
    litellm._turn_on_debug()
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
    # litellm._turn_on_debug()
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
