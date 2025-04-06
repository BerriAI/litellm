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
from unittest.mock import AsyncMock, patch
from litellm import RateLimitError, Timeout, completion, completion_cost, embedding
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

litellm.num_retries = 3


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.flaky(retries=3, delay=1)
@pytest.mark.asyncio
async def test_chat_completion_cohere_citations(stream):
    try:
        litellm.set_verbose = True
        messages = [
            {
                "role": "user",
                "content": "Which penguins are the tallest?",
            },
        ]
        response = await litellm.acompletion(
            model="cohere_chat/command-r",
            messages=messages,
            documents=[
                {"title": "Tall penguins", "text": "Emperor penguins are the tallest."},
                {
                    "title": "Penguin habitats",
                    "text": "Emperor penguins only live in Antarctica.",
                },
            ],
            stream=stream,
        )

        if stream:
            citations_chunk = False
            async for chunk in response:
                print("received chunk", chunk)
                if "citations" in chunk:
                    citations_chunk = True
                    break
            assert citations_chunk
        else:
            assert response.citations is not None
    except litellm.ServiceUnavailableError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_completion_cohere_command_r_plus_function_call():
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
    messages = [
        {
            "role": "user",
            "content": "What's the weather like in Boston today in Fahrenheit?",
        }
    ]
    try:
        # test without max tokens
        response = completion(
            model="command-r-plus",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        # Add any assertions, here to check response args
        print(response)
        assert isinstance(response.choices[0].message.tool_calls[0].function.name, str)
        assert isinstance(
            response.choices[0].message.tool_calls[0].function.arguments, str
        )

        messages.append(
            response.choices[0].message.model_dump()
        )  # Add assistant tool invokes
        tool_result = (
            '{"location": "Boston", "temperature": "72", "unit": "fahrenheit"}'
        )
        # Add user submitted tool results in the OpenAI format
        messages.append(
            {
                "tool_call_id": response.choices[0].message.tool_calls[0].id,
                "role": "tool",
                "name": response.choices[0].message.tool_calls[0].function.name,
                "content": tool_result,
            }
        )
        # In the second response, Cohere should deduce answer from tool results
        second_response = completion(
            model="command-r-plus",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            force_single_step=True,
        )
        print(second_response)
    except litellm.Timeout:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# @pytest.mark.skip(reason="flaky test, times out frequently")
@pytest.mark.flaky(retries=6, delay=1)
def test_completion_cohere():
    try:
        # litellm.set_verbose=True
        messages = [
            {"role": "system", "content": "You're a good bot"},
            {"role": "assistant", "content": [{"text": "2", "type": "text"}]},
            {"role": "assistant", "content": [{"text": "3", "type": "text"}]},
            {
                "role": "user",
                "content": "Hey",
            },
        ]
        response = completion(
            model="command-r",
            messages=messages,
        )
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# FYI - cohere_chat looks quite unstable, even when testing locally
@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_chat_completion_cohere(sync_mode):
    try:
        litellm.set_verbose = True
        messages = [
            {"role": "system", "content": "You're a good bot"},
            {
                "role": "user",
                "content": "Hey",
            },
        ]
        if sync_mode is False:
            response = await litellm.acompletion(
                model="cohere_chat/command-r",
                messages=messages,
                max_tokens=10,
            )
        else:
            response = completion(
                model="cohere_chat/command-r",
                messages=messages,
                max_tokens=10,
            )
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [False])
async def test_chat_completion_cohere_stream(sync_mode):
    try:
        litellm.set_verbose = True
        messages = [
            {"role": "system", "content": "You're a good bot"},
            {
                "role": "user",
                "content": "Hey",
            },
        ]
        if sync_mode is False:
            response = await litellm.acompletion(
                model="cohere_chat/command-r",
                messages=messages,
                max_tokens=10,
                stream=True,
            )
            print("async cohere stream response", response)
            async for chunk in response:
                print(chunk)
        else:
            response = completion(
                model="cohere_chat/command-r",
                messages=messages,
                max_tokens=10,
                stream=True,
            )
            print(response)
            for chunk in response:
                print(chunk)
    except litellm.APIConnectionError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio
async def test_cohere_request_body_with_allowed_params():
    """
    Test to validate that when allowed_openai_params is provided, the request body contains
    the correct response_format and reasoning_effort values.
    """
    # Define test parameters
    test_response_format = {"type": "json"}
    test_reasoning_effort = "low"
    test_tools = [{
        "type": "function", 
        "function": {
            "name": "get_current_time", 
            "description": "Get the current time in a given location.", 
            "parameters": {
                "type": "object", 
                "properties": {
                    "location": {"type": "string", "description": "The city name, e.g. San Francisco"}
                },
                "required": ["location"]
            }
        }
    }]

    client = AsyncHTTPHandler()

    # Mock the post method
    with patch.object(client, "post", new=AsyncMock()) as mock_post:
        try:
            await litellm.acompletion(
                model="cohere/command",
                messages=[{"content": "what llm are you", "role": "user"}],
                allowed_openai_params=["tools", "response_format", "reasoning_effort"],
                response_format=test_response_format,
                reasoning_effort=test_reasoning_effort,
                tools=test_tools,
                client=client
            )
        except Exception:
            pass  # We only care about the request body validation

        # Verify the API call was made
        mock_post.assert_called_once()
        
        # Get and parse the request body
        request_data = json.loads(mock_post.call_args.kwargs["data"])
        print(f"request_data: {request_data}")
        
        # Validate request contains our specified parameters
        assert "allowed_openai_params" not in request_data
        assert request_data["response_format"] == test_response_format
        assert request_data["reasoning_effort"] == test_reasoning_effort
