import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()
import io
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest

import litellm
from litellm import RateLimitError, Timeout, completion, completion_cost, embedding

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
@pytest.mark.parametrize(
    "model",
    [
        "gpt-3.5-turbo-1106",
        # "mistral/mistral-large-latest",
        "claude-3-haiku-20240307",
        "gemini/gemini-1.5-pro",
        "anthropic.claude-3-sonnet-20240229-v1:0",
        # "groq/llama3-8b-8192",
    ],
)
@pytest.mark.flaky(retries=3, delay=1)
def test_aaparallel_function_call(model):
    try:
        litellm.set_verbose = True
        litellm.modify_params = True
        # Step 1: send the conversation and available functions to the model
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
        )
        print("Response\n", response)
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        print("Expecting there to be 3 tool calls")
        assert (
            len(tool_calls) > 0
        )  # this has to call the function for SF, Tokyo and paris

        # Step 2: check if the model wanted to call a function
        print(f"tool_calls: {tool_calls}")
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
                temperature=0.2,
                seed=22,
                # tools=tools,
                drop_params=True,
            )  # get a new response from the model where it can see the function response
            print("second response\n", second_response)
    except litellm.InternalServerError as e:
        print(e)
    except litellm.RateLimitError as e:
        print(e)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_parallel_function_call()

from litellm.types.utils import ChatCompletionMessageToolCall, Function, Message


@pytest.mark.parametrize(
    "model, provider",
    [
        (
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "bedrock",
        ),
        ("claude-3-haiku-20240307", "anthropic"),
    ],
)
@pytest.mark.parametrize(
    "messages, expected_error_msg",
    [
        (
            [
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
            ],
            True,
        ),
        (
            [
                {
                    "role": "user",
                    "content": "What's the weather like in San Francisco, Tokyo, and Paris? - give me 3 responses",
                }
            ],
            False,
        ),
    ],
)
def test_parallel_function_call_anthropic_error_msg(
    model, provider, messages, expected_error_msg
):
    """
    Anthropic doesn't support tool calling without `tools=` param specified.

    Ensure this error is thrown when `tools=` param is not specified. But tool call requests are made.

    Reference Issue: https://github.com/BerriAI/litellm/issues/5747, https://github.com/BerriAI/litellm/issues/5388
    """
    try:
        litellm.set_verbose = True

        messages = messages

        if expected_error_msg:
            with pytest.raises(litellm.UnsupportedParamsError) as e:
                second_response = litellm.completion(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                    seed=22,
                    drop_params=True,
                )  # get a new response from the model where it can see the function response
                print("second response\n", second_response)
        else:
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
            model="gpt-3.5-turbo-1106",
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
                model="gpt-3.5-turbo-1106", messages=messages, temperature=0.2, seed=22
            )  # get a new response from the model where it can see the function response
            print("second response\n", second_response)
            return second_response
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_parallel_function_call_stream()


@pytest.mark.skip(
    reason="Flaky test. Groq function calling is not reliable for ci/cd testing."
)
def test_groq_parallel_function_call():
    litellm.set_verbose = True
    try:
        # Step 1: send the conversation and available functions to the model
        messages = [
            {
                "role": "system",
                "content": "You are a function calling LLM that uses the data extracted from get_current_weather to answer questions about the weather in San Francisco.",
            },
            {
                "role": "user",
                "content": "What's the weather like in San Francisco?",
            },
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
            model="groq/llama2-70b-4096",
            messages=messages,
            tools=tools,
            tool_choice="auto",  # auto is default, but we'll be explicit
        )
        print("Response\n", response)
        response_message = response.choices[0].message
        if hasattr(response_message, "tool_calls"):
            tool_calls = response_message.tool_calls

            assert isinstance(
                response.choices[0].message.tool_calls[0].function.name, str
            )
            assert isinstance(
                response.choices[0].message.tool_calls[0].function.arguments, str
            )

            print("length of tool calls", len(tool_calls))

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
                    model="groq/llama2-70b-4096", messages=messages
                )  # get a new response from the model where it can see the function response
                print("second response\n", second_response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize(
    "model",
    [
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "claude-3-haiku-20240307",
    ],
)
def test_anthropic_function_call_with_no_schema(model):
    """
    Relevant Issue: https://github.com/BerriAI/litellm/issues/6012
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in New York",
            },
        }
    ]
    messages = [
        {"role": "user", "content": "What is the current temperature in New York?"}
    ]
    completion(model=model, messages=messages, tools=tools, tool_choice="auto")
