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
import json

import pytest

import litellm
from litellm import RateLimitError, Timeout, completion, completion_cost, embedding

litellm.num_retries = 3


# FYI - cohere_chat looks quite unstable, even when testing locally
def test_chat_completion_cohere():
    try:
        litellm.set_verbose = True
        messages = [
            {
                "role": "user",
                "content": "Hey",
            },
        ]
        response = completion(
            model="cohere_chat/command-r",
            messages=messages,
            max_tokens=10,
        )
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_chat_completion_cohere_tool_calling():
    try:
        litellm.set_verbose = True
        messages = [
            {
                "role": "user",
                "content": "What is the weather like in Boston?",
            },
        ]
        response = completion(
            model="cohere_chat/command-r",
            messages=messages,
            tools=[
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
            ],
        )
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

    # def get_current_weather(location, unit="fahrenheit"):
    #     """Get the current weather in a given location"""
    #     if "tokyo" in location.lower():
    #         return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    #     elif "san francisco" in location.lower():
    #         return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
    #     elif "paris" in location.lower():
    #         return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    #     else:
    #         return json.dumps({"location": location, "temperature": "unknown"})

    # def test_chat_completion_cohere_tool_with_result_calling():
    #     # end to end cohere command-r with tool calling
    #     # Step 1 - Send available tools
    #     # Step 2 - Execute results
    #     # Step 3 - Send results to command-r
    #     try:
    #         litellm.set_verbose = True
    #         import json

    #         # Step 1 - Send available tools
    #         tools = [
    #                 {
    #                     "type": "function",
    #                     "function": {
    #                         "name": "get_current_weather",
    #                         "description": "Get the current weather in a given location",
    #                         "parameters": {
    #                             "type": "object",
    #                             "properties": {
    #                                 "location": {
    #                                     "type": "string",
    #                                     "description": "The city and state, e.g. San Francisco, CA",
    #                                 },
    #                                 "unit": {
    #                                     "type": "string",
    #                                     "enum": ["celsius", "fahrenheit"],
    #                                 },
    #                             },
    #                             "required": ["location"],
    #                         },
    #                     },
    #                 }
    #         ]

    #         messages = [
    #             {
    #                 "role": "user",
    #                 "content": "What is the weather like in Boston?",
    #             },
    #         ]
    #         response = completion(
    #             model="cohere_chat/command-r",
    #             messages=messages,
    #             tools=tools,
    #         )
    #         print("Response with tools to call", response)
    #         print(response)

    #         # step 2 - Execute results
    #         tool_calls = response.tool_calls

    #         available_functions = {
    #             "get_current_weather": get_current_weather,
    #         }  # only one function in this example, but you can have multiple

    #         for tool_call in tool_calls:
    #             function_name = tool_call.function.name
    #             function_to_call = available_functions[function_name]
    #             function_args = json.loads(tool_call.function.arguments)
    #             function_response = function_to_call(
    #                 location=function_args.get("location"),
    #                 unit=function_args.get("unit"),
    #             )
    #             messages.append(
    #                 {
    #                     "tool_call_id": tool_call.id,
    #                     "role": "tool",
    #                     "name": function_name,
    #                     "content": function_response,
    #                 }
    #             )  # extend conversation with function response

    #         print("messages with tool call results", messages)

    # messages = [
    #     {
    #         "role": "user",
    #         "content": "What is the weather like in Boston?",
    #     },
    #     {
    #             "tool_call_id": "tool_1",
    #             "role": "tool",
    #             "name": "get_current_weather",
    #             "content": {"location": "San Francisco, CA", "unit": "fahrenheit", "temperature": "72"},
    #     },
    # ]
    # respone = completion(
    #     model="cohere_chat/command-r",
    #     messages=messages,
    #     tools=[
    #         {
    #             "type": "function",
    #             "function": {
    #                 "name": "get_current_weather",
    #                 "description": "Get the current weather in a given location",
    #                 "parameters": {
    #                     "type": "object",
    #                     "properties": {
    #                         "location": {
    #                             "type": "string",
    #                             "description": "The city and state, e.g. San Francisco, CA",
    #                         },
    #                         "unit": {
    #                             "type": "string",
    #                             "enum": ["celsius", "fahrenheit"],
    #                         },
    #                     },
    #                     "required": ["location"],
    #                 },
    #             },
    #         }
    #     ],
    # )
    # print(respone)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
