import json
import sys
import os
import io, asyncio

import logging

logging.basicConfig(level=logging.DEBUG)
sys.path.insert(0, os.path.abspath("../.."))

from litellm import completion
import litellm

litellm.num_retries = 3
litellm.success_callback = ["langfuse"]
os.environ["LANGFUSE_DEBUG"] = "True"
import time
import pytest


def search_logs(log_file_path, num_good_logs=1):
    """
    Searches the given log file for logs containing the "/api/public" string.

    Parameters:
    - log_file_path (str): The path to the log file to be searched.

    Returns:
    - None

    Raises:
    - Exception: If there are any bad logs found in the log file.
    """
    import re

    print("\n searching logs")
    bad_logs = []
    good_logs = []
    all_logs = []
    try:
        with open(log_file_path, "r") as log_file:
            lines = log_file.readlines()
            print(f"searching logslines: {lines}")
            for line in lines:
                all_logs.append(line.strip())
                if "/api/public" in line:
                    print("Found log with /api/public:")
                    print(line.strip())
                    print("\n\n")
                    match = re.search(
                        r'"POST /api/public/ingestion HTTP/1.1" (\d+) (\d+)',
                        line,
                    )
                    if match:
                        status_code = int(match.group(1))
                        print("STATUS CODE", status_code)
                        if (
                            status_code != 200
                            and status_code != 201
                            and status_code != 207
                        ):
                            print("got a BAD log")
                            bad_logs.append(line.strip())
                        else:
                            good_logs.append(line.strip())
        print("\nBad Logs")
        print(bad_logs)
        if len(bad_logs) > 0:
            raise Exception(f"bad logs, Bad logs = {bad_logs}")
        assert (
            len(good_logs) == num_good_logs
        ), f"Did not get expected number of good logs, expected {num_good_logs}, got {len(good_logs)}. All logs \n {all_logs}"
        print("\nGood Logs")
        print(good_logs)
        if len(good_logs) <= 0:
            raise Exception(
                f"There were no Good Logs from Langfuse. No logs with /api/public status 200. \nAll logs:{all_logs}"
            )

    except Exception as e:
        raise e


def pre_langfuse_setup():
    """
    Set up the logging for the 'pre_langfuse_setup' function.
    """
    # sends logs to langfuse.log
    import logging

    # Configure the logging to write to a file
    logging.basicConfig(filename="langfuse.log", level=logging.DEBUG)
    logger = logging.getLogger()

    # Add a FileHandler to the logger
    file_handler = logging.FileHandler("langfuse.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    return


def test_langfuse_logging_async():
    # this tests time added to make langfuse logging calls, vs just acompletion calls
    try:
        pre_langfuse_setup()
        litellm.set_verbose = True

        # Make 5 calls with an empty success_callback
        litellm.success_callback = []
        start_time_empty_callback = asyncio.run(make_async_calls())
        print("done with no callback test")

        print("starting langfuse test")
        # Make 5 calls with success_callback set to "langfuse"
        litellm.success_callback = ["langfuse"]
        start_time_langfuse = asyncio.run(make_async_calls())
        print("done with langfuse test")

        # Compare the time for both scenarios
        print(f"Time taken with success_callback='langfuse': {start_time_langfuse}")
        print(f"Time taken with empty success_callback: {start_time_empty_callback}")

        # assert the diff is not more than 1 second - this was 5 seconds before the fix
        assert abs(start_time_langfuse - start_time_empty_callback) < 1

    except litellm.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"An exception occurred - {e}")


async def make_async_calls():
    tasks = []
    for _ in range(5):
        task = asyncio.create_task(
            litellm.acompletion(
                model="azure/chatgpt-v-2",
                messages=[{"role": "user", "content": "This is a test"}],
                max_tokens=5,
                temperature=0.7,
                timeout=5,
                user="langfuse_latency_test_user",
                mock_response="It's simple to use and easy to get started",
            )
        )
        tasks.append(task)

    # Measure the start time before running the tasks
    start_time = asyncio.get_event_loop().time()

    # Wait for all tasks to complete
    responses = await asyncio.gather(*tasks)

    # Print the responses when tasks return
    for idx, response in enumerate(responses):
        print(f"Response from Task {idx + 1}: {response}")

    # Calculate the total time taken
    total_time = asyncio.get_event_loop().time() - start_time

    return total_time


# def test_langfuse_logging_async_text_completion():
#     try:
#         pre_langfuse_setup()
#         litellm.set_verbose = False
#         litellm.success_callback = ["langfuse"]

#         async def _test_langfuse():
#             response = await litellm.atext_completion(
#                 model="gpt-3.5-turbo-instruct",
#                 prompt="this is a test",
#                 max_tokens=5,
#                 temperature=0.7,
#                 timeout=5,
#                 user="test_user",
#                 stream=True
#             )
#             async for chunk in response:
#                 print()
#                 print(chunk)
#             await asyncio.sleep(1)
#             return response

#         response = asyncio.run(_test_langfuse())
#         print(f"response: {response}")

#         # # check langfuse.log to see if there was a failed response
#         search_logs("langfuse.log")
#     except litellm.Timeout as e:
#         pass
#     except Exception as e:
#         pytest.fail(f"An exception occurred - {e}")


# test_langfuse_logging_async_text_completion()


@pytest.mark.skip(reason="beta test - checking langfuse output")
def test_langfuse_logging():
    try:
        pre_langfuse_setup()
        litellm.set_verbose = True
        response = completion(
            model="claude-instant-1.2",
            messages=[{"role": "user", "content": "Hi 👋 - i'm claude"}],
            max_tokens=10,
            temperature=0.2,
        )
        print(response)
        # time.sleep(5)
        # # check langfuse.log to see if there was a failed response
        # search_logs("langfuse.log")

    except litellm.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"An exception occurred - {e}")


# test_langfuse_logging()


@pytest.mark.skip(reason="beta test - checking langfuse output")
def test_langfuse_logging_stream():
    try:
        litellm.set_verbose = True
        response = completion(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": "this is a streaming test for llama2 + langfuse",
                }
            ],
            max_tokens=20,
            temperature=0.2,
            stream=True,
        )
        print(response)
        for chunk in response:
            pass
            # print(chunk)
    except litellm.Timeout as e:
        pass
    except Exception as e:
        print(e)


# test_langfuse_logging_stream()


@pytest.mark.skip(reason="beta test - checking langfuse output")
def test_langfuse_logging_custom_generation_name():
    try:
        litellm.set_verbose = True
        response = completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi 👋 - i'm claude"}],
            max_tokens=10,
            metadata={
                "langfuse/foo": "bar",
                "langsmith/fizz": "buzz",
                "prompt_hash": "asdf98u0j9131123",
                "generation_name": "ishaan-test-generation",
                "generation_id": "gen-id22",
                "trace_id": "trace-id22",
                "trace_user_id": "user-id2",
            },
        )
        print(response)
    except litellm.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"An exception occurred - {e}")
        print(e)


# test_langfuse_logging_custom_generation_name()


@pytest.mark.skip(reason="beta test - checking langfuse output")
def test_langfuse_logging_embedding():
    try:
        litellm.set_verbose = True
        litellm.success_callback = ["langfuse"]
        response = litellm.embedding(
            model="text-embedding-ada-002",
            input=["gm", "ishaan"],
        )
        print(response)
    except litellm.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"An exception occurred - {e}")
        print(e)


@pytest.mark.skip(reason="beta test - checking langfuse output")
def test_langfuse_logging_function_calling():
    litellm.set_verbose = True
    function1 = [
        {
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
        }
    ]
    try:
        response = completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "what's the weather in boston"}],
            temperature=0.1,
            functions=function1,
        )
        print(response)
    except litellm.Timeout as e:
        pass
    except Exception as e:
        print(e)


# test_langfuse_logging_function_calling()


def test_langfuse_logging_tool_calling():
    litellm.set_verbose = True

    def get_current_weather(location, unit="fahrenheit"):
        """Get the current weather in a given location"""
        if "tokyo" in location.lower():
            return json.dumps(
                {"location": "Tokyo", "temperature": "10", "unit": "celsius"}
            )
        elif "san francisco" in location.lower():
            return json.dumps(
                {"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"}
            )
        elif "paris" in location.lower():
            return json.dumps(
                {"location": "Paris", "temperature": "22", "unit": "celsius"}
            )
        else:
            return json.dumps({"location": location, "temperature": "unknown"})

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
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
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
    print("\nLLM Response1:\n", response)
    response_message = response.choices[0].message
    tool_calls = response.choices[0].message.tool_calls


# test_langfuse_logging_tool_calling()
