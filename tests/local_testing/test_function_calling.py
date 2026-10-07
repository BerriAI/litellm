import traceback

from dotenv import load_dotenv

load_dotenv()
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm import RateLimitError, Timeout, completion_cost, embedding

litellm.num_retries = 0
litellm.cache = None
# litellm.set_verbose=True
import json

# litellm.success_callback = ["langfuse"]


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


# Example dummy function hard coded to return the same weather


# In production, this could be your backend API or an external API
# test_parallel_function_call()


from litellm.types.utils import ChatCompletionMessageToolCall, Function, Message

_PARALLEL_TOOL_HISTORY_MESSAGES = [
    {
        "role": "user",
        "content": "What's the weather like in San Francisco, Tokyo, and Paris? - give me 3 responses",
    },
    Message(
        content="Here are the current weather conditions for San Francisco, Tokyo, and Paris:",
        role="assistant",
        tool_calls=[
            ChatCompletionMessageToolCall(
                index=1,
                function=Function(
                    arguments='{"location": "San Francisco, CA", "unit": "fahrenheit"}',
                    name="get_current_weather",
                ),
                id="tooluse_Jj98qn6xQlOP_PiQr-w9iA",
                type="function",
            )
        ],
        function_call=None,
    ),
    {
        "tool_call_id": "tooluse_Jj98qn6xQlOP_PiQr-w9iA",
        "role": "tool",
        "name": "get_current_weather",
        "content": '{"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"}',
    },
]


@pytest.mark.parametrize(
    "model, messages",
    [
        # Anthropic Messages API: a dummy tool is injected without modify_params,
        # so tool history with no tools= completes instead of raising.
        ("claude-haiku-4-5-20251001", _PARALLEL_TOOL_HISTORY_MESSAGES),
        (
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            [
                {
                    "role": "user",
                    "content": "What's the weather like in San Francisco, Tokyo, and Paris? - give me 3 responses",
                }
            ],
        ),
        (
            "claude-haiku-4-5-20251001",
            [
                {
                    "role": "user",
                    "content": "What's the weather like in San Francisco, Tokyo, and Paris? - give me 3 responses",
                }
            ],
        ),
    ],
)
def test_parallel_function_call_anthropic_error_msg(model, messages):
    """
    Tool history without an explicit ``tools`` param must complete, not raise.

    Anthropic (and Bedrock Invoke via ``AnthropicConfig.transform_request``)
    inject a dummy tool so CLIs work with ``modify_params`` left off. Bedrock
    Converse's no-raise behavior is covered offline in
    ``tests/unit/llms/bedrock/chat/test_converse_transformation.py``
    (see #24158, #27138), which needs no live credentials.
    """
    # Force modify_params off as a clean baseline: it exercises the Anthropic
    # dummy-tool path, which injects regardless of modify_params
    # (other tests in this file set it to True and don't reset it)
    original_modify_params = litellm.modify_params
    litellm.modify_params = False
    try:
        litellm.set_verbose = True
        second_response = litellm.completion(
            model=model,
            messages=messages,
            temperature=0.2,
            seed=22,
            drop_params=True,
        )  # get a new response from the model where it can see the function response
        print("second response\n", second_response)
    except litellm.InternalServerError as e:
        print(e)
    except litellm.RateLimitError as e:
        print(e)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
    finally:
        litellm.modify_params = original_modify_params


def test_parallel_function_call_stream():
    try:
        litellm.set_verbose = True
        # Step 1: send the conversation and available functions to the model
        messages = [
            {
                "role": "user",
                "content": "What's the weather like in San Francisco, Tokyo, and Paris?",
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
                                "description": "The city and state, e.g. San Francisco, CA",
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
            model="gpt-6-luna",
            messages=messages,
            tools=tools,
            stream=True,
            tool_choice="auto",  # auto is default, but we'll be explicit
            complete_response=True,
        )
        print("Response\n", response)
        # for chunk in response:
        #     print(chunk)
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        print("length of tool calls", len(tool_calls))
        print("Expecting there to be 3 tool calls")
        assert (
            len(tool_calls) > 1
        )  # this has to call the function for SF, Tokyo and parise

        # Step 2: check if the model wanted to call a function
        if tool_calls:
            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors
            available_functions = {
                "get_current_weather": get_current_weather,
            }  # only one function in this example, but you can have multiple
            messages.append(
                response_message
            )  # extend conversation with assistant's reply
            print("Response message\n", response_message)
            # Step 4: send the info for each function call and function response to the model
            for tool_call in tool_calls:
                function_name = tool_call.function.name
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
                model="gpt-6-luna", messages=messages, temperature=0.2, seed=22, reasoning_effort="none"
            )  # get a new response from the model where it can see the function response
            print("second response\n", second_response)
            return second_response
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_parallel_function_call_stream()




@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
@pytest.mark.flaky(retries=6, delay=1)
async def test_watsonx_tool_choice(sync_mode, monkeypatch):
    import json

    from litellm import acompletion, completion
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

    # Mock the IAM token generation to avoid actual API calls
    monkeypatch.setenv("WATSONX_API_KEY", "mock-api-key")
    monkeypatch.setenv("WATSONX_TOKEN", "mock-watsonx-token")
    monkeypatch.setenv("WATSONX_API_BASE", "https://us-south.ml.cloud.ibm.com")
    monkeypatch.setenv("WATSONX_PROJECT_ID", "mock-project-id")

    litellm.set_verbose = True
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
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    messages = [{"role": "user", "content": "What is the weather in San Francisco?"}]

    client = HTTPHandler() if sync_mode else AsyncHTTPHandler()
    with patch.object(client, "post", return_value=MagicMock()) as mock_completion:
        try:
            if sync_mode:
                resp = completion(
                    model="watsonx/meta-llama/llama-3-1-8b-instruct",
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    client=client,
                )
            else:
                resp = await acompletion(
                    model="watsonx/meta-llama/llama-3-1-8b-instruct",
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    client=client,
                    stream=True,
                )

            print(resp)

            mock_completion.assert_called_once()
            print(mock_completion.call_args.kwargs)
            json_data = json.loads(mock_completion.call_args.kwargs["data"])
            assert json_data["tool_choice_option"] == "auto"
        except Exception as e:
            print(e)
            if "The read operation timed out" in str(e):
                pytest.skip("Skipping test due to timeout")
            else:
                raise e
