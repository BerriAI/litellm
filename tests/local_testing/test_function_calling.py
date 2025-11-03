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
from unittest.mock import patch, MagicMock, AsyncMock
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
        "mistral/mistral-large-latest",
        "claude-3-haiku-20240307",
        "gemini/gemini-2.5-flash-lite",
        "anthropic.claude-3-sonnet-20240229-v1:0",
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


@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-3-7-sonnet-20250219",
        "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    ],
)
@pytest.mark.flaky(retries=3, delay=1)
def test_aaparallel_function_call_with_anthropic_thinking(model):
    try:
        litellm._turn_on_debug()
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
            thinking={"type": "enabled", "budget_tokens": 1024},
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
                seed=22,
                # tools=tools,
                drop_params=True,
                thinking={"type": "enabled", "budget_tokens": 1024},
            )  # get a new response from the model where it can see the function response
            print("second response\n", second_response)

            ## THIRD RESPONSE
    except litellm.InternalServerError as e:
        print(e)
    except litellm.RateLimitError as e:
        print(e)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


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
        "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    ],
)
def test_passing_tool_result_as_list(model):
    litellm.set_verbose = True
    litellm._turn_on_debug()
    messages = [
        {
            "content": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant that have the ability to interact with a computer to solve tasks.",
                }
            ],
            "role": "system",
        },
        {
            "content": [
                {
                    "type": "text",
                    "text": "Write a git commit message for the current staging area and commit the changes.",
                }
            ],
            "role": "user",
        },
        {
            "content": [
                {
                    "type": "text",
                    "text": "I'll help you commit the changes. Let me first check the git status to see what changes are staged.",
                }
            ],
            "role": "assistant",
            "tool_calls": [
                {
                    "index": 1,
                    "function": {
                        "arguments": '{"command": "git status", "thought": "Checking git status to see staged changes"}',
                        "name": "execute_bash",
                    },
                    "id": "toolu_01V1paXrun4CVetdAGiQaZG5",
                    "type": "function",
                }
            ],
        },
        {
            "content": [
                {
                    "type": "text",
                    "text": 'OBSERVATION:\nOn branch master\r\n\r\nNo commits yet\r\n\r\nChanges to be committed:\r\n  (use "git rm --cached <file>..." to unstage)\r\n\tnew file:   hello.py\r\n\r\n\r\n[Python Interpreter: /openhands/poetry/openhands-ai-5O4_aCHf-py3.12/bin/python]\nroot@openhands-workspace:/workspace # \n[Command finished with exit code 0]',
                }
            ],
            "role": "tool",
            "tool_call_id": "toolu_01V1paXrun4CVetdAGiQaZG5",
            "name": "execute_bash"
        },
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "execute_bash",
                "description": 'Execute a bash command in the terminal.\n* Long running commands: For commands that may run indefinitely, it should be run in the background and the output should be redirected to a file, e.g. command = `python3 app.py > server.log 2>&1 &`.\n* Interactive: If a bash command returns exit code `-1`, this means the process is not yet finished. The assistant must then send a second call to terminal with an empty `command` (which will retrieve any additional logs), or it can send additional text (set `command` to the text) to STDIN of the running process, or it can send command=`ctrl+c` to interrupt the process.\n* Timeout: If a command execution result says "Command timed out. Sending SIGINT to the process", the assistant should retry running the command in the background.\n',
                "parameters": {
                    "type": "object",
                    "properties": {
                        "thought": {
                            "type": "string",
                            "description": "Reasoning about the action to take.",
                        },
                        "command": {
                            "type": "string",
                            "description": "The bash command to execute. Can be empty to view additional logs when previous exit code is `-1`. Can be `ctrl+c` to interrupt the currently running process.",
                        },
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "finish",
                "description": "Finish the interaction.\n* Do this if the task is complete.\n* Do this if the assistant cannot proceed further with the task.\n",
            },
        },
        {
            "type": "function",
            "function": {
                "name": "str_replace_editor",
                "description": "Custom editing tool for viewing, creating and editing files\n* State is persistent across command calls and discussions with the user\n* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep\n* The `create` command cannot be used if the specified `path` already exists as a file\n* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`\n* The `undo_edit` command will revert the last edit made to the file at `path`\n\nNotes for using the `str_replace` command:\n* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!\n* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique\n* The `new_str` parameter should contain the edited lines that should replace the `old_str`\n",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "description": "The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.",
                            "enum": [
                                "view",
                                "create",
                                "str_replace",
                                "insert",
                                "undo_edit",
                            ],
                            "type": "string",
                        },
                        "path": {
                            "description": "Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`.",
                            "type": "string",
                        },
                        "file_text": {
                            "description": "Required parameter of `create` command, with the content of the file to be created.",
                            "type": "string",
                        },
                        "old_str": {
                            "description": "Required parameter of `str_replace` command containing the string in `path` to replace.",
                            "type": "string",
                        },
                        "new_str": {
                            "description": "Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.",
                            "type": "string",
                        },
                        "insert_line": {
                            "description": "Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.",
                            "type": "integer",
                        },
                        "view_range": {
                            "description": "Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.",
                            "items": {"type": "integer"},
                            "type": "array",
                        },
                    },
                    "required": ["command", "path"],
                },
            },
        },
    ]
    for _ in range(2):
        resp = completion(model=model, messages=messages, tools=tools)
        print(resp)

    if model == "claude-sonnet-4-5-20250929":
        assert resp.usage.prompt_tokens_details.cached_tokens > 0


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
@pytest.mark.flaky(retries=6, delay=1)
async def test_watsonx_tool_choice(sync_mode):
    from litellm.llms.custom_httpx.http_handler import HTTPHandler, AsyncHTTPHandler
    import json
    from litellm import acompletion, completion

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
            json_data["tool_choice_option"] == "auto"
        except Exception as e:
            print(e)
            if "The read operation timed out" in str(e):
                pytest.skip("Skipping test due to timeout")
            else:
                raise e


