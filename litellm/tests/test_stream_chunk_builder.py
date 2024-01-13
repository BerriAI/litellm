import sys, os, time
import traceback, asyncio
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from litellm import completion, stream_chunk_builder
import litellm
import os, dotenv
from openai import OpenAI
import pytest

dotenv.load_dotenv()

user_message = "What is the current weather in Boston?"
messages = [{"content": user_message, "role": "user"}]

function_schema = {
    "name": "get_weather",
    "description": "gets the current weather",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
        },
        "required": ["location"],
    },
}


tools_schema = [
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

# def test_stream_chunk_builder_tools():
#     try:
#       litellm.set_verbose = False
#       response = client.chat.completions.create(
#           model="gpt-3.5-turbo",
#           messages=messages,
#           tools=tools_schema,
#           # stream=True,
#           # complete_response=True # runs stream_chunk_builder under-the-hood
#       )

#       print(f"response: {response}")
#       print(f"response usage: {response.usage}")
#     except Exception as e:
#        pytest.fail(f"An exception occurred - {str(e)}")

# test_stream_chunk_builder_tools()


def test_stream_chunk_builder_litellm_function_call():
    try:
        litellm.set_verbose = False
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=messages,
            functions=[function_schema],
            # stream=True,
            # complete_response=True # runs stream_chunk_builder under-the-hood
        )

        print(f"response: {response}")
    except Exception as e:
        pytest.fail(f"An exception occurred - {str(e)}")


# test_stream_chunk_builder_litellm_function_call()


def test_stream_chunk_builder_litellm_tool_call():
    try:
        litellm.set_verbose = True
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=messages,
            tools=tools_schema,
            stream=True,
            complete_response=True,
        )

        print(f"complete response: {response}")
        print(f"complete response usage: {response.usage}")
        assert response.usage.completion_tokens > 0
        assert response.usage.prompt_tokens > 0
        assert (
            response.usage.total_tokens
            == response.usage.completion_tokens + response.usage.prompt_tokens
        )
    except Exception as e:
        pytest.fail(f"An exception occurred - {str(e)}")


# test_stream_chunk_builder_litellm_tool_call()


def test_stream_chunk_builder_litellm_tool_call_regular_message():
    try:
        messages = [{"role": "user", "content": "Hey, how's it going?"}]
        # litellm.set_verbose = True
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=messages,
            tools=tools_schema,
            stream=True,
            complete_response=True,
        )

        print(f"complete response: {response}")
        print(f"complete response usage: {response.usage}")
        assert response.usage.completion_tokens > 0
        assert response.usage.prompt_tokens > 0
        assert (
            response.usage.total_tokens
            == response.usage.completion_tokens + response.usage.prompt_tokens
        )

        # check provider is in hidden params
        print("hidden params", response._hidden_params)
        assert response._hidden_params["custom_llm_provider"] == "openai"

    except Exception as e:
        pytest.fail(f"An exception occurred - {str(e)}")


# test_stream_chunk_builder_litellm_tool_call_regular_message()
