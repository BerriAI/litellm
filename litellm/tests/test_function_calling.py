import sys, os
import traceback
from dotenv import load_dotenv

load_dotenv()
import os, io

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
import litellm
from litellm import embedding, completion, completion_cost, Timeout
from litellm import RateLimitError
import pytest

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
def test_parallel_function_call():
    try:
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
            tool_choice="auto",  # auto is default, but we'll be explicit
        )
        print("Response\n", response)
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


# test_parallel_function_call()


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
