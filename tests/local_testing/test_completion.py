import json
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


import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm import RateLimitError, Timeout, completion, completion_cost, embedding
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.litellm_core_utils.prompt_templates.factory import anthropic_messages_pt

# litellm.num_retries=3

litellm.cache = None
litellm.success_callback = []
user_message = "Write a short poem about the sky"
messages = [{"content": user_message, "role": "user"}]


def logger_fn(user_model_dict):
    print(f"user_model_dict: {user_model_dict}")


@pytest.fixture(autouse=True)
def reset_callbacks():
    print("\npytest fixture - resetting callbacks")
    litellm.success_callback = []
    litellm._async_success_callback = []
    litellm.failure_callback = []
    litellm.callbacks = []


@pytest.mark.skip(reason="Local test")
def test_response_model_none():
    """
    Addresses:https://github.com/BerriAI/litellm/issues/2972
    """
    x = completion(
        model="mymodel",
        custom_llm_provider="openai",
        messages=[{"role": "user", "content": "Hello!"}],
        api_base="http://0.0.0.0:8080",
        api_key="my-api-key",
    )
    print(f"x: {x}")
    assert isinstance(x, litellm.ModelResponse)


def test_completion_custom_provider_model_name():
    try:
        litellm.cache = None
        response = completion(
            model="together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1",
            messages=messages,
            logger_fn=logger_fn,
        )
        # Add assertions here to check the-response
        print(response)
        print(response["choices"][0]["finish_reason"])
    except litellm.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def _openai_mock_response(*args, **kwargs) -> litellm.ModelResponse:
    new_response = MagicMock()
    new_response.headers = {"hello": "world"}

    response_object = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-3.5-turbo-0125",
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "\n\nHello there, how may I assist you today?",
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
    }
    from openai import OpenAI
    from openai.types.chat.chat_completion import ChatCompletion

    pydantic_obj = ChatCompletion(**response_object)  # type: ignore
    pydantic_obj.choices[0].message.role = None  # type: ignore
    new_response.parse.return_value = pydantic_obj
    return new_response


def test_null_role_response():
    """
    Test if the api returns 'null' role, 'assistant' role is still returned
    """
    import openai

    openai_client = openai.OpenAI()
    with patch.object(
        openai_client.chat.completions, "create", side_effect=_openai_mock_response
    ) as mock_response:
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hey! how's it going?"}],
            client=openai_client,
        )
        print(f"response: {response}")

        assert response.id == "chatcmpl-123"

        assert response.choices[0].message.role == "assistant"



def predibase_mock_post(url, data=None, json=None, headers=None, timeout=None):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.json.return_value = {
        "generated_text": " Is it to find happiness, to achieve success,",
        "details": {
            "finish_reason": "length",
            "prompt_tokens": 8,
            "generated_tokens": 10,
            "seed": None,
            "prefill": [],
            "tokens": [
                {"id": 2209, "text": " Is", "logprob": -1.7568359, "special": False},
                {"id": 433, "text": " it", "logprob": -0.2220459, "special": False},
                {"id": 311, "text": " to", "logprob": -0.6928711, "special": False},
                {"id": 1505, "text": " find", "logprob": -0.6425781, "special": False},
                {
                    "id": 23871,
                    "text": " happiness",
                    "logprob": -0.07519531,
                    "special": False,
                },
                {"id": 11, "text": ",", "logprob": -0.07110596, "special": False},
                {"id": 311, "text": " to", "logprob": -0.79296875, "special": False},
                {
                    "id": 11322,
                    "text": " achieve",
                    "logprob": -0.7602539,
                    "special": False,
                },
                {
                    "id": 2450,
                    "text": " success",
                    "logprob": -0.03656006,
                    "special": False,
                },
                {"id": 11, "text": ",", "logprob": -0.0011510849, "special": False},
            ],
        },
    }
    return mock_response


# @pytest.mark.skip(reason="local-only test")
@pytest.mark.asyncio
async def test_completion_predibase():
    try:
        litellm.set_verbose = True

        # with patch("requests.post", side_effect=predibase_mock_post):
        response = await litellm.acompletion(
            model="predibase/llama-3-8b-instruct",
            tenant_id="c4768f95",
            api_key=os.getenv("PREDIBASE_API_KEY"),
            messages=[{"role": "user", "content": "who are u?"}],
            max_tokens=10,
            timeout=5,
        )

        print(response)
    except litellm.Timeout as e:
        print("got a timeout error from predibase")
        pass
    except litellm.ServiceUnavailableError as e:
        pass
    except litellm.InternalServerError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_predibase()


# test_completion_claude()


@pytest.mark.skip(reason="No empower api key")
def test_completion_empower():
    litellm.set_verbose = True
    messages = [
        {
            "role": "user",
            "content": "\nWhat is the query for `console.log` => `console.error`\n",
        },
        {
            "role": "assistant",
            "content": "\nThis is the GritQL query for the given before/after examples:\n<gritql>\n`console.log` => `console.error`\n</gritql>\n",
        },
        {
            "role": "user",
            "content": "\nWhat is the query for `console.info` => `consdole.heaven`\n",
        },
    ]
    try:
        # test without max tokens
        response = completion(
            model="empower/empower-functions-small",
            messages=messages,
        )
        # Add any assertions, here to check response args
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_completion_github_api():
    litellm.set_verbose = True
    messages = [
        {
            "role": "user",
            "content": "\nWhat is the query for `console.log` => `console.error`\n",
        },
        {
            "role": "assistant",
            "content": "\nThis is the GritQL query for the given before/after examples:\n<gritql>\n`console.log` => `console.error`\n</gritql>\n",
        },
        {
            "role": "user",
            "content": "\nWhat is the query for `console.info` => `consdole.heaven`\n",
        },
    ]
    try:
        # test without max tokens
        response = completion(
            model="github/gpt-4o",
            messages=messages,
        )
        # Add any assertions, here to check response args
        print(response)
    except litellm.AuthenticationError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_completion_claude_3_empty_response():
    litellm.set_verbose = True

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are 2twNLGfqk4GMOn3ffp4p."}],
        },
        {"role": "user", "content": "Hi gm!", "name": "ishaan"},
        {"role": "assistant", "content": "Good morning! How are you doing today?"},
        {
            "role": "user",
            "content": "I was hoping we could chat a bit",
        },
    ]
    try:
        response = litellm.completion(model="claude-3-opus-20240229", messages=messages)
        print(response)
    except litellm.InternalServerError as e:
        pytest.skip(f"InternalServerError - {str(e)}")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_completion_claude_3():
    litellm.set_verbose = True
    messages = [
        {
            "role": "user",
            "content": "\nWhat is the query for `console.log` => `console.error`\n",
        },
        {
            "role": "assistant",
            "content": "\nThis is the GritQL query for the given before/after examples:\n<gritql>\n`console.log` => `console.error`\n</gritql>\n",
        },
        {
            "role": "user",
            "content": "\nWhat is the query for `console.info` => `consdole.heaven`\n",
        },
    ]
    try:
        # test without max tokens
        response = completion(
            model="anthropic/claude-3-opus-20240229",
            messages=messages,
        )
        # Add any assertions, here to check response args
        print(response)
    except litellm.InternalServerError as e:
        pytest.skip(f"InternalServerError - {str(e)}")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize(
    "model",
    ["anthropic/claude-3-opus-20240229", "anthropic.claude-3-sonnet-20240229-v1:0"],
)
def test_completion_claude_3_function_call(model):
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
            model=model,
            messages=messages,
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": "get_current_weather"},
            },
            drop_params=True,
        )

        # Add any assertions here to check response args
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
        # In the second response, Claude should deduce answer from tool results
        second_response = completion(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            drop_params=True,
        )
        print(second_response)
    except litellm.InternalServerError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize("sync_mode", [True])
@pytest.mark.parametrize(
    "model, api_key, api_base",
    [
        ("gpt-3.5-turbo", None, None),
        ("claude-3-opus-20240229", None, None),
        ("anthropic.claude-3-sonnet-20240229-v1:0", None, None),
        # (
        #     "azure_ai/command-r-plus",
        #     os.getenv("AZURE_COHERE_API_KEY"),
        #     os.getenv("AZURE_COHERE_API_BASE"),
        # ),
    ],
)
@pytest.mark.asyncio
async def test_model_function_invoke(model, sync_mode, api_key, api_base):
    try:
        litellm.set_verbose = True

        messages = [
            {
                "role": "system",
                "content": "Your name is Litellm Bot, you are a helpful assistant",
            },
            # User asks for their name and weather in San Francisco
            {
                "role": "user",
                "content": "Hello, what is your name and can you tell me the weather?",
            },
            # Assistant replies with a tool call
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "index": 0,
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "San Francisco, CA"}',
                        },
                    }
                ],
            },
            # The result of the tool call is added to the history
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": "27 degrees celsius and clear in San Francisco, CA",
            },
            # Now the assistant can reply with the result of the tool call.
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            }
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        data = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "api_key": api_key,
            "api_base": api_base,
        }
        if sync_mode:
            response = litellm.completion(**data)
        else:
            response = await litellm.acompletion(**data)

        print(f"response: {response}")
    except litellm.InternalServerError:
        pass
    except litellm.RateLimitError as e:
        pass
    except Exception as e:
        if "429 Quota exceeded" in str(e):
            pass
        else:
            pytest.fail("An unexpected exception occurred - {}".format(str(e)))


@pytest.mark.asyncio
async def test_anthropic_no_content_error():
    """
    https://github.com/BerriAI/litellm/discussions/3440#discussioncomment-9323402
    """
    try:
        litellm.drop_params = True
        response = await litellm.acompletion(
            model="anthropic/claude-3-opus-20240229",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            messages=[
                {
                    "role": "system",
                    "content": "You will be given a list of fruits. Use the submitFruit function to submit a fruit. Don't say anything after.",
                },
                {"role": "user", "content": "I like apples"},
                {
                    "content": "<thinking>The most relevant tool for this request is the submitFruit function.</thinking>",
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": '{"name": "Apple"}',
                                "name": "submitFruit",
                            },
                            "id": "toolu_012ZTYKWD4VqrXGXyE7kEnAK",
                            "type": "function",
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": '{"success":true}',
                    "tool_call_id": "toolu_012ZTYKWD4VqrXGXyE7kEnAK",
                },
            ],
            max_tokens=2000,
            temperature=1,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "submitFruit",
                        "description": "Submits a fruit",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The name of the fruit",
                                }
                            },
                            "required": ["name"],
                        },
                    },
                }
            ],
            frequency_penalty=0.8,
        )

        pass
    except litellm.InternalServerError:
        pass
    except litellm.APIError as e:
        assert e.status_code == 500
    except Exception as e:
        pytest.fail(f"An unexpected error occurred - {str(e)}")


def test_parse_xml_params():
    from litellm.litellm_core_utils.prompt_templates.factory import parse_xml_params

    ## SCENARIO 1 ## - W/ ARRAY
    xml_content = """<invoke><tool_name>return_list_of_str</tool_name>\n<parameters>\n<value>\n<item>apple</item>\n<item>banana</item>\n<item>orange</item>\n</value>\n</parameters></invoke>"""
    json_schema = {
        "properties": {
            "value": {
                "items": {"type": "string"},
                "title": "Value",
                "type": "array",
            }
        },
        "required": ["value"],
        "type": "object",
    }
    response = parse_xml_params(xml_content=xml_content, json_schema=json_schema)

    print(f"response: {response}")
    assert response["value"] == ["apple", "banana", "orange"]

    ## SCENARIO 2 ## - W/OUT ARRAY
    xml_content = """<invoke><tool_name>get_current_weather</tool_name>\n<parameters>\n<location>Boston, MA</location>\n<unit>fahrenheit</unit>\n</parameters></invoke>"""
    json_schema = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
    }

    response = parse_xml_params(xml_content=xml_content, json_schema=json_schema)

    print(f"response: {response}")
    assert response["location"] == "Boston, MA"
    assert response["unit"] == "fahrenheit"


def test_completion_claude_3_multi_turn_conversations():
    litellm.set_verbose = True
    litellm.modify_params = True
    messages = [
        {"role": "assistant", "content": "?"},  # test first user message auto injection
        {"role": "user", "content": "Hi!"},
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is the weather like today?"}],
        },
        {"role": "assistant", "content": "Hi! I am Claude. "},
        {"role": "assistant", "content": "Today is a sunny "},
    ]
    try:
        response = completion(
            model="anthropic/claude-3-opus-20240229",
            messages=messages,
        )
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_completion_claude_3_stream():
    litellm.set_verbose = False
    messages = [{"role": "user", "content": "Hello, world"}]
    try:
        # test without max tokens
        response = completion(
            model="anthropic/claude-3-opus-20240229",
            messages=messages,
            max_tokens=10,
            stream=True,
        )
        # Add any assertions, here to check response args
        print(response)
        for chunk in response:
            print(chunk)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def encode_image(image_path):
    import base64

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@pytest.mark.parametrize(
    "model",
    [
        "gpt-4o",
        "azure/gpt-4.1-mini",
        "anthropic/claude-3-opus-20240229",
    ],
)  #
def test_completion_base64(model):
    try:
        import base64

        import requests

        litellm.set_verbose = True
        url = "https://dummyimage.com/100/100/fff&text=Test+image"
        response = requests.get(url)
        file_data = response.content

        encoded_file = base64.b64encode(file_data).decode("utf-8")
        base64_image = f"data:image/png;base64,{encoded_file}"
        resp = litellm.completion(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Whats in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": base64_image},
                        },
                    ],
                }
            ],
        )
        print(f"\nResponse: {resp}")

        prompt_tokens = resp.usage.prompt_tokens
    except litellm.ServiceUnavailableError as e:
        print("got service unavailable error: ", e)
        pass
    except litellm.InternalServerError as e:
        print("got internal server error: ", e)
        pass
    except Exception as e:
        if "500 Internal error encountered.'" in str(e):
            pass
        else:
            pytest.fail(f"An exception occurred - {str(e)}")


def test_completion_mistral_api():
    try:
        litellm.set_verbose = True
        response = completion(
            model="mistral/mistral-tiny",
            max_tokens=5,
            messages=[
                {
                    "role": "user",
                    "content": "Hey, how's it going?",
                }
            ],
            seed=10,
        )
        # Add any assertions here to check the response
        print(response)

        cost = litellm.completion_cost(completion_response=response)
        print("cost to make mistral completion=", cost)
        assert cost > 0.0
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.skip(reason="backend api unavailable")
@pytest.mark.asyncio
async def test_completion_codestral_chat_api():
    try:
        litellm.set_verbose = True
        response = await litellm.acompletion(
            model="codestral/codestral-latest",
            messages=[
                {
                    "role": "user",
                    "content": "Hey, how's it going?",
                }
            ],
            temperature=0.0,
            top_p=1,
            max_tokens=10,
            safe_prompt=False,
            seed=12,
        )
        # Add any assertions here to-check the response
        print(response)

        # cost = litellm.completion_cost(completion_response=response)
        # print("cost to make mistral completion=", cost)
        # assert cost > 0.0
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_completion_mistral_api_mistral_large_function_call():
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
            model="mistral/mistral-medium-latest",
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
        # In the second response, Mistral should deduce answer from tool results
        second_response = completion(
            model="mistral/mistral-large-latest",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        print(second_response)
    except litellm.RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.skip(
    reason="Since we already test mistral/mistral-tiny in test_completion_mistral_api. This is only for locally verifying azure mistral works"
)
def test_completion_mistral_azure():
    try:
        litellm.set_verbose = True
        response = completion(
            model="mistral/Mistral-large-nmefg",
            api_key=os.environ["MISTRAL_AZURE_API_KEY"],
            api_base=os.environ["MISTRAL_AZURE_API_BASE"],
            max_tokens=5,
            messages=[
                {
                    "role": "user",
                    "content": "Hi from litellm",
                }
            ],
        )
        # Add any assertions here to check, the response
        print(response)

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_mistral_api()


def test_completion_mistral_api_modified_input():
    try:
        litellm.set_verbose = True
        response = completion(
            model="mistral/mistral-tiny",
            max_tokens=5,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hey, how's it going?"}],
                }
            ],
        )
        # Add any assertions here to check the response
        print(response)

        cost = litellm.completion_cost(completion_response=response)
        print("cost to make mistral completion=", cost)
        assert cost > 0.0
    except Exception as e:
        if "500" in str(e):
            pass
        else:
            pytest.fail(f"Error occurred: {e}")


# def test_completion_oobabooga():
#     try:
#         response = completion(
#             model="oobabooga/vicuna-1.3b", messages=messages, api_base="http://127.0.0.1:5000"
#         )
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# test_completion_oobabooga()
# aleph alpha
# def test_completion_aleph_alpha():
#     try:
#         response = completion(
#             model="luminous-base", messages=messages, logger_fn=logger_fn
#         )
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")
# test_completion_aleph_alpha()


# def test_completion_aleph_alpha_control_models():
#     try:
#         response = completion(
#             model="luminous-base-control", messages=messages, logger_fn=logger_fn
#         )
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")
# test_completion_aleph_alpha_control_models()

import openai


def test_completion_gpt4_turbo():
    litellm.set_verbose = True
    try:
        response = completion(
            model="gpt-4-1106-preview",
            messages=messages,
            max_completion_tokens=10,
        )
        print(response)
    except openai.RateLimitError:
        print("got a rate liimt error")
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_gpt4_turbo()


def test_completion_gpt4_turbo_0125():
    try:
        response = completion(
            model="gpt-4-0125-preview",
            messages=messages,
            max_tokens=10,
        )
        print(response)
    except openai.RateLimitError:
        print("got a rate liimt error")
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.skip(reason="this test is flaky")
def test_completion_gpt4_vision():
    try:
        litellm.set_verbose = True
        response = completion(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Whats in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                            },
                        },
                    ],
                }
            ],
        )
        print(response)
    except openai.RateLimitError:
        print("got a rate liimt error")
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_gpt4_vision()


def test_completion_azure_gpt4_vision():
    # azure/gpt-4, vision takes 5-seconds to respond
    try:
        litellm.set_verbose = True
        response = completion(
            model="azure/gpt-4-vision",
            timeout=5,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Whats in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://avatars.githubusercontent.com/u/29436595?v=4"
                            },
                        },
                    ],
                }
            ],
            base_url="https://gpt-4-vision-resource.openai.azure.com/openai/deployments/gpt-4-vision/extensions",
            api_key=os.getenv("AZURE_VISION_API_KEY"),
            enhancements={"ocr": {"enabled": True}, "grounding": {"enabled": True}},
            dataSources=[
                {
                    "type": "AzureComputerVision",
                    "parameters": {
                        "endpoint": "https://gpt-4-vision-enhancement.cognitiveservices.azure.com/",
                        "key": os.environ["AZURE_VISION_ENHANCE_KEY"],
                    },
                }
            ],
        )
        print(response)
    except openai.APIError as e:
        pass
    except openai.APITimeoutError:
        print("got a timeout error")
        pass
    except openai.RateLimitError as e:
        print("got a rate liimt error", e)
        pass
    except openai.APIStatusError as e:
        print("got an api status error", e)
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_azure_gpt4_vision()


def test_completion_openai_response_headers():
    """
    Tests if LiteLLM reurns response hea
    """
    litellm.return_response_headers = True

    # /chat/completion
    messages = [
        {
            "role": "user",
            "content": "hi",
        }
    ]

    response = completion(
        model="gpt-4o-mini",
        messages=messages,
    )

    print(f"response: {response}")

    print("response_headers=", response._response_headers)
    assert response._response_headers is not None
    assert "x-ratelimit-remaining-tokens" in response._response_headers
    assert isinstance(
        response._hidden_params["additional_headers"][
            "llm_provider-x-ratelimit-remaining-requests"
        ],
        str,
    )

    # /chat/completion - with streaming

    streaming_response = litellm.completion(
        model="gpt-4o-mini",
        messages=messages,
        stream=True,
    )
    response_headers = streaming_response._response_headers
    print("streaming response_headers=", response_headers)
    assert response_headers is not None
    assert "x-ratelimit-remaining-tokens" in response_headers
    assert isinstance(
        response._hidden_params["additional_headers"][
            "llm_provider-x-ratelimit-remaining-requests"
        ],
        str,
    )

    for chunk in streaming_response:
        print("chunk=", chunk)

    # embedding
    embedding_response = litellm.embedding(
        model="text-embedding-ada-002",
        input="hello",
    )

    embedding_response_headers = embedding_response._response_headers
    print("embedding_response_headers=", embedding_response_headers)
    assert embedding_response_headers is not None
    assert "x-ratelimit-remaining-tokens" in embedding_response_headers
    assert isinstance(
        response._hidden_params["additional_headers"][
            "llm_provider-x-ratelimit-remaining-requests"
        ],
        str,
    )

    litellm.return_response_headers = False


@pytest.mark.asyncio()
async def test_async_completion_openai_response_headers():
    """
    Tests if LiteLLM reurns response hea
    """
    litellm.return_response_headers = True

    # /chat/completion
    messages = [
        {
            "role": "user",
            "content": "hi",
        }
    ]

    response = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=messages,
    )

    print(f"response: {response}")

    print("response_headers=", response._response_headers)
    assert response._response_headers is not None
    assert "x-ratelimit-remaining-tokens" in response._response_headers

    # /chat/completion with streaming

    streaming_response = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=messages,
        stream=True,
    )
    response_headers = streaming_response._response_headers
    print("streaming response_headers=", response_headers)
    assert response_headers is not None
    assert "x-ratelimit-remaining-tokens" in response_headers

    async for chunk in streaming_response:
        print("chunk=", chunk)

    # embedding
    embedding_response = await litellm.aembedding(
        model="text-embedding-ada-002",
        input="hello",
    )

    embedding_response_headers = embedding_response._response_headers
    print("embedding_response_headers=", embedding_response_headers)
    assert embedding_response_headers is not None
    assert "x-ratelimit-remaining-tokens" in embedding_response_headers

    litellm.return_response_headers = False


@pytest.mark.parametrize("model", ["gpt-3.5-turbo", "gpt-4", "gpt-4o"])
def test_completion_openai_params(model):
    litellm.drop_params = True
    messages = [
        {
            "role": "user",
            "content": """Generate JSON about Bill Gates: { "full_name": "", "title": "" }""",
        }
    ]

    response = completion(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )

    print(f"response: {response}")


def test_completion_fireworks_ai():
    try:
        litellm.set_verbose = True
        messages = [
            {"role": "system", "content": "You're a good bot"},
            {
                "role": "user",
                "content": "Hey",
            },
        ]
        response = completion(
            model="fireworks_ai/llama4-maverick-instruct-basic",
            messages=messages,
        )
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize(
    "api_key, api_base", [(None, "my-bad-api-base"), ("my-bad-api-key", None)]
)
def test_completion_fireworks_ai_dynamic_params(api_key, api_base):
    try:
        litellm.set_verbose = True
        messages = [
            {"role": "system", "content": "You're a good bot"},
            {
                "role": "user",
                "content": "Hey",
            },
        ]
        response = completion(
            model="fireworks_ai/accounts/fireworks/models/mixtral-8x7b-instruct",
            messages=messages,
            api_base=api_base,
            api_key=api_key,
        )
        pytest.fail(f"This call should have failed!")
    except Exception as e:
        pass


# @pytest.mark.skip(reason="this test is flaky")
def test_completion_perplexity_api():
    try:
        response_object = {
            "id": "a8f37485-026e-45da-81a9-cf0184896840",
            "model": "llama-3-sonar-small-32k-online",
            "created": 1722186391,
            "usage": {"prompt_tokens": 17, "completion_tokens": 65, "total_tokens": 82},
            "citations": [
                "https://www.sciencedirect.com/science/article/pii/S007961232200156X",
                "https://www.britannica.com/event/World-War-II",
                "https://www.loc.gov/classroom-materials/united-states-history-primary-source-timeline/great-depression-and-world-war-ii-1929-1945/world-war-ii/",
                "https://www.nationalww2museum.org/war/topics/end-world-war-ii-1945",
                "https://en.wikipedia.org/wiki/World_War_II",
            ],
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "World War II was won by the Allied powers, which included the United States, the Soviet Union, Great Britain, France, China, and other countries. The war concluded with the surrender of Germany on May 8, 1945, and Japan on September 2, 1945[2][3][4].",
                    },
                    "delta": {"role": "assistant", "content": ""},
                }
            ],
        }

        from openai import OpenAI
        from openai.types.chat.chat_completion import ChatCompletion

        pydantic_obj = ChatCompletion(**response_object)

        def _return_pydantic_obj(*args, **kwargs):
            new_response = MagicMock()
            new_response.headers = {"hello": "world"}

            new_response.parse.return_value = pydantic_obj
            return new_response

        openai_client = OpenAI()

        with patch.object(
            openai_client.chat.completions.with_raw_response,
            "create",
            side_effect=_return_pydantic_obj,
        ) as mock_client:
            # litellm.set_verbose= True
            messages = [
                {"role": "system", "content": "You're a good bot"},
                {
                    "role": "user",
                    "content": "Hey",
                },
                {
                    "role": "user",
                    "content": "Hey",
                },
            ]
            response = completion(
                model="mistral-7b-instruct",
                messages=messages,
                api_base="https://api.perplexity.ai",
                client=openai_client,
            )
            print(response)
            assert hasattr(response, "citations")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_perplexity_api()


@pytest.mark.skip(reason="this test is flaky")
def test_completion_perplexity_api_2():
    try:
        # litellm.set_verbose=True
        messages = [
            {"role": "system", "content": "You're a good bot"},
            {
                "role": "user",
                "content": "Hey",
            },
            {
                "role": "user",
                "content": "Hey",
            },
        ]
        response = completion(model="perplexity/mistral-7b-instruct", messages=messages)
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_perplexity_api_2()

# commenting out as this is a flaky test on circle-ci
# def test_completion_nlp_cloud():
#     try:
#         messages = [
#             {"role": "system", "content": "You are a helpful assistant."},
#             {
#                 "role": "user",
#                 "content": "how does a court case get to the Supreme Court?",
#             },
#         ]
#         response = completion(model="dolphin", messages=messages, logger_fn=logger_fn)
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# test_completion_nlp_cloud()

######### HUGGING FACE TESTS ########################
#####################################################
"""
HF Tests we should pass
- TGI:
    - Pro Inference API
    - Deployed Endpoint
- Coversational
    - Free Inference API
    - Deployed Endpoint
- Neither TGI or Coversational
    - Free Inference API
    - Deployed Endpoint
"""


@pytest.mark.parametrize(
    "provider", ["openai", "hosted_vllm", "lm_studio", "llamafile"]
)  # "vertex_ai",
@pytest.mark.asyncio
async def test_openai_compatible_custom_api_base(provider):
    litellm.set_verbose = True
    messages = [
        {
            "role": "user",
            "content": "Hello world",
        }
    ]
    from openai import OpenAI

    openai_client = OpenAI(api_key="fake-key")

    with patch.object(
        openai_client.chat.completions, "create", new=MagicMock()
    ) as mock_call:
        try:
            completion(
                model="{provider}/my-vllm-model".format(provider=provider),
                messages=messages,
                response_format={"type": "json_object"},
                client=openai_client,
                api_base="my-custom-api-base",
                hello="world",
            )
        except Exception as e:
            print(e)

        mock_call.assert_called_once()

        print("Call KWARGS - {}".format(mock_call.call_args.kwargs))

        assert "hello" in mock_call.call_args.kwargs["extra_body"]


@pytest.mark.parametrize(
    "provider",
    [
        "openai",
        "hosted_vllm",
        "llamafile",
    ],
)  # "vertex_ai",
@pytest.mark.asyncio
async def test_openai_compatible_custom_api_video(provider):
    litellm.set_verbose = True
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What do you see in this video?",
                },
                {
                    "type": "video_url",
                    "video_url": {"url": "https://www.youtube.com/watch?v=29_ipKNI8I0"},
                },
            ],
        }
    ]
    from openai import OpenAI

    openai_client = OpenAI(api_key="fake-key")

    with patch.object(
        openai_client.chat.completions, "create", new=MagicMock()
    ) as mock_call:
        try:
            completion(
                model="{provider}/my-vllm-model".format(provider=provider),
                messages=messages,
                response_format={"type": "json_object"},
                client=openai_client,
                api_base="my-custom-api-base",
            )
        except Exception as e:
            print(e)

        mock_call.assert_called_once()


def test_lm_studio_completion(monkeypatch):
    monkeypatch.delenv("LM_STUDIO_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    litellm._turn_on_debug()
    try:
        completion(
            api_key="fake-key",
            model="lm_studio/typhoon2-quen2.5-7b-instruct",
            messages=[
                {"role": "user", "content": "What's the weather like in San Francisco?"}
            ],
            api_base="https://exampleopenaiendpoint-production.up.railway.app/",
        )
    except litellm.AuthenticationError as e:
        pytest.fail(f"Error occurred: {e}")
    except litellm.APIError as e:
        print(e)


# ################### Hugging Face Conversational models ########################
# def hf_test_completion_conv():
#     try:
#         response = litellm.completion(
#             model="huggingface/facebook/blenderbot-3B",
#             messages=[{ "content": "Hello, how are you?","role": "user"}],
#         )
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")
# hf_test_completion_conv()

# ################### Hugging Face Neither TGI or Conversational models ########################
# # Neither TGI or Conversational task
# def hf_test_completion_none_task():
#     try:
#         user_message = "My name is Merve and my favorite"
#         messages = [{ "content": user_message,"role": "user"}]
#         response = completion(
#             model="huggingface/roneneldan/TinyStories-3M",
#             messages=messages,
#             api_base="https://p69xlsj6rpno5drq.us-east-1.aws.endpoints.huggingface.cloud",
#         )
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")
# hf_test_completion_none_task()


def mock_post(url, **kwargs):
    print(f"url={url}")
    if "text-classification" in url:
        raise Exception("Model not found")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.json.return_value = [
        [
            {"label": "LABEL_0", "score": 0.9990691542625427},
            {"label": "LABEL_1", "score": 0.0009308889275416732},
        ]
    ]
    return mock_response


def test_ollama_image():
    """
    Test that datauri prefixes are removed, JPEG/PNG images are passed
    through, and other image formats are converted to JPEG.  Non-image
    data is untouched.
    """

    import base64
    import io

    from PIL import Image

    def mock_post(url, **kwargs):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        data_json = json.loads(kwargs["data"])
        mock_response.json.return_value = {
            # return the image in the response so that it can be tested
            # against the original
            "response": data_json["images"]
        }
        return mock_response

    def make_b64image(format):
        image = Image.new(mode="RGB", size=(1, 1))
        image_buffer = io.BytesIO()
        image.save(image_buffer, format)
        return base64.b64encode(image_buffer.getvalue()).decode("utf-8")

    jpeg_image = make_b64image("JPEG")
    webp_image = make_b64image("WEBP")
    png_image = make_b64image("PNG")

    base64_data = base64.b64encode(b"some random data")
    datauri_base64_data = f"data:text/plain;base64,{base64_data}"

    tests = [
        # input                                    expected
        [jpeg_image, jpeg_image],
        [webp_image, None],
        [png_image, png_image],
        [f"data:image/jpeg;base64,{jpeg_image}", jpeg_image],
        [f"data:image/webp;base64,{webp_image}", None],
        [f"data:image/png;base64,{png_image}", png_image],
        [datauri_base64_data, datauri_base64_data],
    ]

    client = HTTPHandler()
    for test in tests:
        try:
            with patch.object(client, "post", side_effect=mock_post):
                response = completion(
                    model="ollama/llava",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Whats in this image?"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": test[0]},
                                },
                            ],
                        }
                    ],
                    client=client,
                )
                if not test[1]:
                    # the conversion process may not always generate the same image,
                    # so just check for a JPEG image when a conversion was done.
                    image_data = response["choices"][0]["message"]["content"][0]
                    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
                    assert image.format == "JPEG"
                else:
                    assert response["choices"][0]["message"]["content"][0] == test[1]
        except Exception as e:
            pytest.fail(f"Error occurred: {e}")


########################### End of Hugging Face Tests ##############################################
# def test_completion_hf_api():
# # failing on circle-ci commenting out
#     try:
#         user_message = "write some code to find the sum of two numbers"
#         messages = [{ "content": user_message,"role": "user"}]
#         api_base = "https://a8l9e3ucxinyl3oj.us-east-1.aws.endpoints.huggingface.cloud"
#         response = completion(model="huggingface/meta-llama/Llama-2-7b-chat-hf", messages=messages, api_base=api_base)
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         if "loading" in str(e):
#             pass
#         pytest.fail(f"Error occurred: {e}")

# test_completion_hf_api()

# def test_completion_hf_api_best_of():
# # failing on circle ci commenting out
#     try:
#         user_message = "write some code to find the sum of two numbers"
#         messages = [{ "content": user_message,"role": "user"}]
#         api_base = "https://a8l9e3ucxinyl3oj.us-east-1.aws.endpoints.huggingface.cloud"
#         response = completion(model="huggingface/meta-llama/Llama-2-7b-chat-hf", messages=messages, api_base=api_base, n=2)
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         if "loading" in str(e):
#             pass
#         pytest.fail(f"Error occurred: {e}")

# test_completion_hf_api_best_of()

# def test_completion_hf_deployed_api():
#     try:
#         user_message = "There's a llama in my garden 😱 What should I do?"
#         messages = [{ "content": user_message,"role": "user"}]
#         response = completion(model="huggingface/https://ji16r2iys9a8rjk2.us-east-1.aws.endpoints.huggingface.cloud", messages=messages, logger_fn=logger_fn)
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")


# this should throw an exception, to trigger https://logs.litellm.ai/
# def hf_test_error_logs():
#     try:
#         litellm.set_verbose=True
#         user_message = "My name is Merve and my favorite"
#         messages = [{ "content": user_message,"role": "user"}]
#         response = completion(
#             model="huggingface/roneneldan/TinyStories-3M",
#             messages=messages,
#             api_base="https://p69xlsj6rpno5drq.us-east-1.aws.endpoints.huggingface.cloud",

#         )
#         # Add any assertions here to check the response
#         print(response)

#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# hf_test_error_logs()


def test_completion_openai():
    try:
        litellm.set_verbose = True
        litellm.drop_params = True
        print(f"api key: {os.environ['OPENAI_API_KEY']}")
        litellm.api_key = os.environ["OPENAI_API_KEY"]
        response = completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hey"}],
            max_tokens=10,
            metadata={"hi": "bye"},
        )
        print("This is the response object\n", response)

        response_str = response["choices"][0]["message"]["content"]
        response_str_2 = response.choices[0].message.content

        cost = completion_cost(completion_response=response)
        print("Cost for completion call with gpt-3.5-turbo: ", f"${float(cost):.10f}")
        assert response_str == response_str_2
        assert type(response_str) == str
        assert len(response_str) > 1

        litellm.api_key = None
    except Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize(
    "model, api_version",
    [
        # ("gpt-4o-2024-08-06", None),
        # ("azure/gpt-4.1-mini", None),
        ("bedrock/anthropic.claude-3-sonnet-20240229-v1:0", None),
        # ("azure/gpt-4o-new-test", "2024-08-01-preview"),
    ],
)
@pytest.mark.flaky(retries=3, delay=1)
def test_completion_openai_pydantic(model, api_version):
    try:
        litellm._turn_on_debug()
        from pydantic import BaseModel

        messages = [
            {"role": "user", "content": "List 5 important events in the XIX century"}
        ]

        class CalendarEvent(BaseModel):
            name: str
            date: str
            participants: list[str]

        class EventsList(BaseModel):
            events: list[CalendarEvent]

        litellm.enable_json_schema_validation = True
        for _ in range(3):
            try:
                response = completion(
                    model=model,
                    messages=messages,
                    metadata={"hi": "bye"},
                    response_format=EventsList,
                    api_version=api_version,
                )
                break
            except litellm.JSONSchemaValidationError:
                pytest.fail("ERROR OCCURRED! INVALID JSON")

        print("This is the response object\n", response)

        response_str = response["choices"][0]["message"]["content"]

        print(f"response_str: {response_str}")
        json.loads(response_str)  # check valid json is returned

    except Timeout:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_completion_openai_organization():
    try:
        litellm.set_verbose = True
        try:
            response = completion(
                model="gpt-3.5-turbo", messages=messages, organization="org-ikDc4ex8NB"
            )
            pytest.fail("Request should have failed - This organization does not exist")
        except Exception as e:
            assert "header should match organization for API key" in str(e)

    except Exception as e:
        print(e)
        pytest.fail(f"Error occurred: {e}")


def test_completion_text_openai():
    try:
        # litellm.set_verbose =True
        response = completion(model="gpt-3.5-turbo-instruct", messages=messages)
        print(response["choices"][0]["message"]["content"])
    except Exception as e:
        print(e)
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio
async def test_completion_text_openai_async():
    try:
        # litellm.set_verbose =True
        response = await litellm.acompletion(
            model="gpt-3.5-turbo-instruct", messages=messages
        )
        print(response["choices"][0]["message"]["content"])
    except Exception as e:
        print(e)
        pytest.fail(f"Error occurred: {e}")


def custom_callback(
    kwargs,  # kwargs to completion
    completion_response,  # response from completion
    start_time,
    end_time,  # start/end time
):
    # Your custom code here
    try:
        print("LITELLM: in custom callback function")
        print("\nkwargs\n", kwargs)
        model = kwargs["model"]
        messages = kwargs["messages"]
        user = kwargs.get("user")

        #################################################

        print(
            f"""
                Model: {model},
                Messages: {messages},
                User: {user},
                Seed: {kwargs["seed"]},
                temperature: {kwargs["temperature"]},
            """
        )

        assert kwargs["user"] == "ishaans app"
        assert kwargs["model"] == "gpt-3.5-turbo-1106"
        assert kwargs["seed"] == 12
        assert kwargs["temperature"] == 0.5
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_completion_openai_with_optional_params():
    # [Proxy PROD TEST] WARNING: DO NOT DELETE THIS TEST
    # assert that `user` gets passed to the completion call
    # Note: This tests that we actually send the optional params to the completion call
    # We use custom callbacks to test this
    try:
        litellm.set_verbose = True
        litellm.success_callback = [custom_callback]
        response = completion(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "user", "content": "respond in valid, json - what is the day"}
            ],
            temperature=0.5,
            top_p=0.1,
            seed=12,
            response_format={"type": "json_object"},
            logit_bias=None,
            user="ishaans app",
        )
        # Add any assertions here to check the response

        print(response)
        litellm.success_callback = []  # unset callbacks

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_openai_with_optional_params()


def test_completion_logprobs():
    """
    This function is used to test the litellm.completion logprobs functionality.

    Parameters:
        None

    Returns:
        None
    """
    try:
        litellm.set_verbose = True
        response = completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "what is the time"}],
            temperature=0.5,
            top_p=0.1,
            seed=12,
            logit_bias=None,
            user="ishaans app",
            logprobs=True,
            top_logprobs=3,
        )
        # Add any assertions here to check the response

        print(response)
        print(len(response.choices[0].logprobs["content"][0]["top_logprobs"]))
        assert "logprobs" in response.choices[0]
        assert "content" in response.choices[0]["logprobs"]
        assert len(response.choices[0].logprobs["content"][0]["top_logprobs"]) == 3

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_logprobs()


def test_completion_logprobs_stream():
    """
    This function is used to test the litellm.completion logprobs functionality.

    Parameters:
        None

    Returns:
        None
    """
    try:
        litellm.set_verbose = False
        response = completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "what is the time"}],
            temperature=0.5,
            top_p=0.1,
            seed=12,
            max_tokens=5,
            logit_bias=None,
            user="ishaans app",
            logprobs=True,
            top_logprobs=3,
            stream=True,
        )
        # Add any assertions here to check the response

        print(response)

        found_logprob = False
        for chunk in response:
            # check if atleast one chunk has log probs
            print(chunk)
            print(f"chunk.choices[0]: {chunk.choices[0]}")
            if "logprobs" in chunk.choices[0]:
                # assert we got a valid logprob in the choices
                assert len(chunk.choices[0].logprobs.content[0].top_logprobs) == 3
                found_logprob = True
                break
            print(chunk)
        assert found_logprob == True
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_logprobs_stream()


def test_completion_openai_litellm_key():
    try:
        litellm.set_verbose = True
        litellm.num_retries = 0
        litellm.api_key = os.environ["OPENAI_API_KEY"]

        # ensure key is set to None in .env and in openai.api_key
        os.environ["OPENAI_API_KEY"] = ""
        import openai

        openai.api_key = ""
        ##########################################################

        response = completion(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.5,
            top_p=0.1,
            max_tokens=10,
            user="ishaan_dev@berri.ai",
        )
        # Add any assertions here to check the response
        print(response)

        ###### reset environ key
        os.environ["OPENAI_API_KEY"] = litellm.api_key

        ##### unset litellm var
        litellm.api_key = None
    except Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_ completion_openai_litellm_key()


@pytest.mark.skip(reason="Unresponsive endpoint.[TODO] Rehost this somewhere else")
def test_completion_ollama_hosted():
    try:
        litellm.request_timeout = 20  # give ollama 20 seconds to response
        litellm.set_verbose = True
        response = completion(
            model="ollama/phi",
            messages=messages,
            max_tokens=20,
            # api_base="https://test-ollama-endpoint.onrender.com",
        )
        # Add any assertions here to check the response
        print(response)
    except openai.APITimeoutError as e:
        print("got a timeout error. Passed ! ")
        litellm.request_timeout = None
        pass
    except Exception as e:
        if "try pulling it first" in str(e):
            return
        pytest.fail(f"Error occurred: {e}")


# test_completion_ollama_hosted()


@pytest.mark.skip(reason="Local test")
@pytest.mark.parametrize(
    ("model"),
    [
        "ollama/llama2",
        "ollama_chat/llama2",
    ],
)
def test_completion_ollama_function_call(model):
    messages = [
        {"role": "user", "content": "What's the weather like in San Francisco?"}
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
    try:
        litellm.set_verbose = True
        response = litellm.completion(model=model, messages=messages, tools=tools)
        print(response)
        assert response.choices[0].message.tool_calls
        assert (
            response.choices[0].message.tool_calls[0].function.name
            == "get_current_weather"
        )
        assert response.choices[0].finish_reason == "tool_calls"
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.skip(reason="Local test")
@pytest.mark.parametrize(
    ("model"),
    [
        "ollama/llama2",
        "ollama_chat/llama2",
    ],
)
def test_completion_ollama_function_call_stream(model):
    messages = [
        {"role": "user", "content": "What's the weather like in San Francisco?"}
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
    try:
        litellm.set_verbose = True
        response = litellm.completion(
            model=model, messages=messages, tools=tools, stream=True
        )
        print(response)
        first_chunk = next(response)
        assert first_chunk.choices[0].delta.tool_calls
        assert (
            first_chunk.choices[0].delta.tool_calls[0].function.name
            == "get_current_weather"
        )
        assert first_chunk.choices[0].finish_reason == "tool_calls"
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.skip(reason="local test")
@pytest.mark.parametrize(
    ("model"),
    [
        "ollama/llama2",
        "ollama_chat/llama2",
    ],
)
@pytest.mark.asyncio
async def test_acompletion_ollama_function_call(model):
    messages = [
        {"role": "user", "content": "What's the weather like in San Francisco?"}
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
    try:
        litellm.set_verbose = True
        response = await litellm.acompletion(
            model=model, messages=messages, tools=tools
        )
        print(response)
        assert response.choices[0].message.tool_calls
        assert (
            response.choices[0].message.tool_calls[0].function.name
            == "get_current_weather"
        )
        assert response.choices[0].finish_reason == "tool_calls"
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.skip(reason="local test")
@pytest.mark.parametrize(
    ("model"),
    [
        "ollama/llama2",
        "ollama_chat/llama2",
    ],
)
@pytest.mark.asyncio
async def test_acompletion_ollama_function_call_stream(model):
    messages = [
        {"role": "user", "content": "What's the weather like in San Francisco?"}
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
    try:
        litellm.set_verbose = True
        response = await litellm.acompletion(
            model=model, messages=messages, tools=tools, stream=True
        )
        print(response)
        first_chunk = await anext(response)
        assert first_chunk.choices[0].delta.tool_calls
        assert (
            first_chunk.choices[0].delta.tool_calls[0].function.name
            == "get_current_weather"
        )
        assert first_chunk.choices[0].finish_reason == "tool_calls"
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_completion_openrouter1():
    try:
        litellm.set_verbose = True
        response = completion(
            model="openrouter/mistralai/mistral-tiny",
            messages=messages,
            max_tokens=5,
        )
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_completion_openrouter_reasoning_effort():
    try:
        litellm.set_verbose = True
        response = completion(
            model="openrouter/deepseek/deepseek-r1",
            messages=messages,
            include_reasoning=True,
            max_tokens=5,
        )
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_openrouter1()


def test_completion_hf_model_no_provider():
    try:
        response = completion(
            model="WizardLM/WizardLM-70B-V1.0",
            messages=messages,
            max_tokens=5,
        )
        # Add any assertions here to check the response
        print(response)
        pytest.fail(f"Error occurred: {e}")
    except Exception as e:
        pass


# test_completion_hf_model_no_provider()


def gemini_mock_post(*args, **kwargs):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.json = MagicMock(
        return_value={
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_current_weather",
                                    "args": {"location": "Boston, MA"},
                                }
                            }
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                    "index": 0,
                    "safetyRatings": [
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "probability": "NEGLIGIBLE",
                        },
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "probability": "NEGLIGIBLE",
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "probability": "NEGLIGIBLE",
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "probability": "NEGLIGIBLE",
                        },
                    ],
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 86,
                "candidatesTokenCount": 19,
                "totalTokenCount": 105,
            },
        }
    )

    return mock_response


@pytest.mark.asyncio
async def test_completion_functions_param():
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
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

        messages = [{"role": "user", "content": "What is the weather like in Boston?"}]

        client = AsyncHTTPHandler(concurrent_limit=1)

        with patch.object(client, "post", side_effect=gemini_mock_post) as mock_client:
            response: litellm.ModelResponse = await litellm.acompletion(
                model="gemini/gemini-1.5-pro",
                messages=messages,
                functions=function1,
                client=client,
            )
            print(response)
            # Add any assertions here to check the response
            mock_client.assert_called()
            print(f"mock_client.call_args.kwargs: {mock_client.call_args.kwargs}")
            assert "tools" in mock_client.call_args.kwargs["json"]
            assert (
                "litellm_param_is_function_call"
                not in mock_client.call_args.kwargs["json"]
            )
            assert (
                "litellm_param_is_function_call"
                not in mock_client.call_args.kwargs["json"]["generationConfig"]
            )
            assert response.choices[0].message.function_call is not None
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_anyscale_with_functions()


def test_completion_azure_extra_headers():
    # this tests if we can pass api_key to completion, when it's not in the env.
    # DO NOT REMOVE THIS TEST. No MATTER WHAT Happens!
    # If you want to remove it, speak to Ishaan!
    # Ishaan will be very disappointed if this test is removed -> this is a standard way to pass api_key + the router + proxy use this
    from httpx import Client
    from openai import AzureOpenAI

    from litellm.llms.custom_httpx.httpx_handler import HTTPHandler

    http_client = Client()

    with patch.object(http_client, "send", new=MagicMock()) as mock_client:
        litellm.client_session = http_client
        try:
            response = completion(
                model="azure/gpt-4.1-mini",
                messages=messages,
                api_base=os.getenv("AZURE_API_BASE"),
                api_version="2023-07-01-preview",
                api_key=os.getenv("AZURE_API_KEY"),
                extra_headers={
                    "Authorization": "my-bad-key",
                    "Ocp-Apim-Subscription-Key": "hello-world-testing",
                },
            )
            print(response)
            pytest.fail("Expected this to fail")
        except Exception as e:
            pass

        mock_client.assert_called()

        print(f"mock_client.call_args: {mock_client.call_args}")
        request = mock_client.call_args[0][0]
        print(request.method)  # This will print 'POST'
        print(request.url)  # This will print the full URL
        print(request.headers)  # This will print the full URL
        auth_header = request.headers.get("Authorization")
        apim_key = request.headers.get("Ocp-Apim-Subscription-Key")
        print(auth_header)
        assert auth_header == "my-bad-key"
        assert apim_key == "hello-world-testing"


def test_completion_azure_ad_token():
    # this tests if we can pass api_key to completion, when it's not in the env.
    # DO NOT REMOVE THIS TEST. No MATTER WHAT Happens!
    # If you want to remove it, speak to Ishaan!
    # Ishaan will be very disappointed if this test is removed -> this is a standard way to pass api_key + the router + proxy use this
    from httpx import Client

    from litellm import completion

    litellm.set_verbose = True

    old_key = os.environ["AZURE_API_KEY"]
    os.environ.pop("AZURE_API_KEY", None)

    http_client = Client()

    with patch.object(http_client, "send", new=MagicMock()) as mock_client:
        litellm.client_session = http_client
        try:
            response = completion(
                model="azure/gpt-4.1-mini",
                messages=messages,
                azure_ad_token="my-special-token",
            )
            print(response)
        except Exception as e:
            pass
        finally:
            os.environ["AZURE_API_KEY"] = old_key

        mock_client.assert_called_once()
        request = mock_client.call_args[0][0]
        print(request.method)  # This will print 'POST'
        print(request.url)  # This will print the full URL
        print(request.headers)  # This will print the full URL
        auth_header = request.headers.get("Authorization")
        assert auth_header == "Bearer my-special-token"


def test_completion_azure_key_completion_arg():
    # this tests if we can pass api_key to completion, when it's not in the env.
    # DO NOT REMOVE THIS TEST. No MATTER WHAT Happens!
    # If you want to remove it, speak to Ishaan!
    # Ishaan will be very disappointed if this test is removed -> this is a standard way to pass api_key + the router + proxy use this
    old_key = os.environ["AZURE_API_KEY"]
    os.environ.pop("AZURE_API_KEY", None)
    try:
        print("azure gpt-3.5 test\n\n")
        litellm.set_verbose = True
        ## Test azure call
        response = completion(
            model="azure/gpt-4.1-mini",
            messages=messages,
            api_key=old_key,
            logprobs=True,
            max_tokens=10,
        )

        print(f"response: {response}")

        print("Hidden Params", response._hidden_params)
        assert response._hidden_params["custom_llm_provider"] == "azure"
        os.environ["AZURE_API_KEY"] = old_key
    except Exception as e:
        os.environ["AZURE_API_KEY"] = old_key
        pytest.fail(f"Error occurred: {e}")


async def test_re_use_azure_async_client():
    try:
        print("azure gpt-3.5 ASYNC with clie nttest\n\n")
        litellm.set_verbose = True
        import openai

        client = openai.AsyncAzureOpenAI(
            azure_endpoint=os.environ["AZURE_API_BASE"],
            api_key=os.environ["AZURE_API_KEY"],
            api_version="2023-07-01-preview",
        )
        ## Test azure call
        for _ in range(3):
            response = await litellm.acompletion(
                model="azure/gpt-4.1-mini", messages=messages, client=client
            )
            print(f"response: {response}")
    except Exception as e:
        pytest.fail("got Exception", e)


def test_re_use_openaiClient():
    try:
        print("gpt-3.5  with client test\n\n")
        litellm.set_verbose = True
        import openai

        client = openai.OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
        )
        ## Test OpenAI call
        for _ in range(2):
            response = litellm.completion(
                model="gpt-3.5-turbo", messages=messages, client=client
            )
            print(f"response: {response}")
    except Exception as e:
        pytest.fail("got Exception", e)


@pytest.mark.skip(
    reason="this is bad test. It doesn't actually fail if the token is not set in the header. "
)
def test_azure_openai_ad_token():
    import time

    # this tests if the azure ad token is set in the request header
    # the request can fail since azure ad tokens expire after 30 mins, but the header MUST have the azure ad token
    # we use litellm.input_callbacks for this test
    def tester(
        kwargs,  # kwargs to completion
    ):
        print("inside kwargs")
        print(kwargs["additional_args"])
        if kwargs["additional_args"]["headers"]["Authorization"] != "Bearer gm":
            pytest.fail("AZURE AD TOKEN Passed but not set in request header")
        return

    litellm.input_callback = [tester]
    try:
        response = litellm.completion(
            model="azure/gpt-4.1-mini",  # e.g. gpt-35-instant
            messages=[
                {
                    "role": "user",
                    "content": "what is your name",
                },
            ],
            azure_ad_token="gm",
        )
        print("azure ad token respoonse\n")
        print(response)
        litellm.input_callback = []
    except Exception as e:
        litellm.input_callback = []
        pass

    time.sleep(1)


# test_azure_openai_ad_token()


# test_completion_azure()
def test_completion_azure2():
    # test if we can pass api_base, api_version and api_key in compleition()
    try:
        print("azure gpt-3.5 test\n\n")
        litellm.set_verbose = False
        api_base = os.environ["AZURE_API_BASE"]
        api_key = os.environ["AZURE_API_KEY"]
        api_version = os.environ["AZURE_API_VERSION"]

        os.environ["AZURE_API_BASE"] = ""
        os.environ["AZURE_API_VERSION"] = ""
        os.environ["AZURE_API_KEY"] = ""

        ## Test azure call
        response = completion(
            model="azure/gpt-4.1-mini",
            messages=messages,
            api_base=api_base,
            api_key=api_key,
            api_version=api_version,
            max_tokens=10,
        )

        # Add any assertions here to check the response
        print(response)

        os.environ["AZURE_API_BASE"] = api_base
        os.environ["AZURE_API_VERSION"] = api_version
        os.environ["AZURE_API_KEY"] = api_key

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_azure2()


def test_completion_azure3():
    # test if we can pass api_base, api_version and api_key in compleition()
    try:
        print("azure gpt-3.5 test\n\n")
        litellm.set_verbose = True
        litellm.api_base = os.environ["AZURE_API_BASE"]
        litellm.api_key = os.environ["AZURE_API_KEY"]
        litellm.api_version = os.environ["AZURE_API_VERSION"]

        os.environ["AZURE_API_BASE"] = ""
        os.environ["AZURE_API_VERSION"] = ""
        os.environ["AZURE_API_KEY"] = ""

        ## Test azure call
        response = completion(
            model="azure/gpt-4.1-mini",
            messages=messages,
            max_tokens=10,
        )

        # Add any assertions here to check the response
        print(response)

        os.environ["AZURE_API_BASE"] = litellm.api_base
        os.environ["AZURE_API_VERSION"] = litellm.api_version
        os.environ["AZURE_API_KEY"] = litellm.api_key

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_azure3()


# new azure test for using litellm. vars,
# use the following vars in this test and make an azure_api_call
#  litellm.api_type = self.azure_api_type
#  litellm.api_base = self.azure_api_base
#  litellm.api_version = self.azure_api_version
#  litellm.api_key = self.api_key
def test_completion_azure_with_litellm_key():
    try:
        print("azure gpt-3.5 test\n\n")
        import openai

        #### set litellm vars
        litellm.api_type = "azure"
        litellm.api_base = os.environ["AZURE_API_BASE"]
        litellm.api_version = os.environ["AZURE_API_VERSION"]
        litellm.api_key = os.environ["AZURE_API_KEY"]

        ######### UNSET ENV VARs for this ################
        os.environ["AZURE_API_BASE"] = ""
        os.environ["AZURE_API_VERSION"] = ""
        os.environ["AZURE_API_KEY"] = ""

        ######### UNSET OpenAI vars for this ##############
        openai.api_type = ""
        openai.api_base = "gm"
        openai.api_version = "333"
        openai.api_key = "ymca"

        response = completion(
            model="azure/gpt-4.1-mini",
            messages=messages,
        )
        # Add any assertions here to check the response
        print(response)

        ######### RESET ENV VARs for this ################
        os.environ["AZURE_API_BASE"] = litellm.api_base
        os.environ["AZURE_API_VERSION"] = litellm.api_version
        os.environ["AZURE_API_KEY"] = litellm.api_key

        ######### UNSET litellm vars
        litellm.api_type = None
        litellm.api_base = None
        litellm.api_version = None
        litellm.api_key = None

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_azure()


import asyncio


@pytest.mark.skip(reason="replicate endpoints are extremely flaky")
@pytest.mark.parametrize("sync_mode", [False, True])
@pytest.mark.asyncio
async def test_completion_replicate_llama3(sync_mode):
    litellm.set_verbose = True
    model_name = "replicate/meta/meta-llama-3-8b-instruct"
    try:
        if sync_mode:
            response = completion(
                model=model_name,
                messages=messages,
                max_tokens=10,
            )
        else:
            response = await litellm.acompletion(
                model=model_name,
                messages=messages,
                max_tokens=10,
            )
            print(f"ASYNC REPLICATE RESPONSE - {response}")
        print(f"REPLICATE RESPONSE - {response}")
        # Add any assertions here to check the response
        assert isinstance(response, litellm.ModelResponse)
        assert len(response.choices[0].message.content.strip()) > 0
        response_format_tests(response=response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.skip(reason="replicate endpoints take +2 mins just for this request")
def test_completion_replicate_vicuna():
    print("TESTING REPLICATE")
    litellm.set_verbose = True
    model_name = "replicate/meta/llama-2-7b-chat:f1d50bb24186c52daae319ca8366e53debdaa9e0ae7ff976e918df752732ccc4"
    try:
        response = completion(
            model=model_name,
            messages=messages,
            temperature=0.5,
            top_k=20,
            repetition_penalty=1,
            min_tokens=1,
            seed=-1,
            max_tokens=2,
        )
        print(response)
        # Add any assertions here to check the response
        response_str = response["choices"][0]["message"]["content"]
        print("RESPONSE STRING\n", response_str)
        if type(response_str) != str:
            pytest.fail(f"Error occurred: {e}")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_replicate_vicuna()


def test_replicate_custom_prompt_dict():
    litellm.set_verbose = True
    model_name = "replicate/meta/llama-2-7b"
    litellm.register_prompt_template(
        model="replicate/meta/llama-2-7b",
        initial_prompt_value="You are a good assistant",  # [OPTIONAL]
        roles={
            "system": {
                "pre_message": "[INST] <<SYS>>\n",  # [OPTIONAL]
                "post_message": "\n<</SYS>>\n [/INST]\n",  # [OPTIONAL]
            },
            "user": {
                "pre_message": "[INST] ",  # [OPTIONAL]
                "post_message": " [/INST]",  # [OPTIONAL]
            },
            "assistant": {
                "pre_message": "\n",  # [OPTIONAL]
                "post_message": "\n",  # [OPTIONAL]
            },
        },
        final_prompt_value="Now answer as best you can:",  # [OPTIONAL]
    )
    try:
        response = completion(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": "what is yc write 1 paragraph",
                }
            ],
            mock_response="Hello world",
            repetition_penalty=0.1,
            num_retries=3,
        )

    except litellm.APIError as e:
        pass
    except litellm.APIConnectionError as e:
        pass
    except Exception as e:
        pytest.fail(f"An exception occurred - {str(e)}")
    print(f"response: {response}")
    litellm.custom_prompt_dict = {}  # reset


def test_bedrock_deepseek_custom_prompt_dict():
    model = "llama/arn:aws:bedrock:us-east-1:1234:imported-model/45d34re"
    litellm.register_prompt_template(
        model=model,
        tokenizer_config={
            "add_bos_token": True,
            "add_eos_token": False,
            "bos_token": {
                "__type": "AddedToken",
                "content": "<｜begin▁of▁sentence｜>",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False,
            },
            "clean_up_tokenization_spaces": False,
            "eos_token": {
                "__type": "AddedToken",
                "content": "<｜end▁of▁sentence｜>",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False,
            },
            "legacy": True,
            "model_max_length": 16384,
            "pad_token": {
                "__type": "AddedToken",
                "content": "<｜end▁of▁sentence｜>",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False,
            },
            "sp_model_kwargs": {},
            "unk_token": None,
            "tokenizer_class": "LlamaTokenizerFast",
            "chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\\n'}}{% endif %}",
        },
    )
    assert model in litellm.known_tokenizer_config
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()

    messages = [
        {"role": "system", "content": "You are a good assistant"},
        {"role": "user", "content": "What is the weather in Copenhagen?"},
    ]

    with patch.object(client, "post") as mock_post:
        try:
            completion(
                model="bedrock/" + model,
                messages=messages,
                client=client,
            )
        except Exception as e:
            pass

        mock_post.assert_called_once()
        print(mock_post.call_args.kwargs)
        json_data = json.loads(mock_post.call_args.kwargs["data"])
        assert (
            json_data["prompt"].rstrip()
            == """<｜begin▁of▁sentence｜>You are a good assistant<｜User｜>What is the weather in Copenhagen?<｜Assistant｜><think>"""
        )


def test_bedrock_deepseek_known_tokenizer_config(monkeypatch):
    model = (
        "deepseek_r1/arn:aws:bedrock:us-west-2:888602223428:imported-model/bnnr6463ejgf"
    )
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    from unittest.mock import Mock
    import httpx

    monkeypatch.setenv("AWS_REGION", "us-east-1")

    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.headers = {
        "x-amzn-bedrock-input-token-count": "20",
        "x-amzn-bedrock-output-token-count": "30",
    }

    # The response format for deepseek_r1
    response_data = {
        "generation": "The weather in Copenhagen is currently sunny with a temperature of 20°C (68°F). The forecast shows clear skies throughout the day with a gentle breeze from the northwest.",
        "stop_reason": "stop",
        "stop_sequence": None,
    }

    mock_response.json.return_value = response_data
    mock_response.text = json.dumps(response_data)

    client = HTTPHandler()

    messages = [
        {"role": "system", "content": "You are a good assistant"},
        {"role": "user", "content": "What is the weather in Copenhagen?"},
    ]

    with patch.object(client, "post", return_value=mock_response) as mock_post:
        completion(
            model="bedrock/" + model,
            messages=messages,
            client=client,
        )

        mock_post.assert_called_once()
        print(mock_post.call_args.kwargs)
        url = mock_post.call_args.kwargs["url"]
        assert "deepseek_r1" not in url
        assert "us-east-1" not in url
        assert "us-west-2" in url
        json_data = json.loads(mock_post.call_args.kwargs["data"])
        assert (
            json_data["prompt"].rstrip()
            == """<｜begin▁of▁sentence｜>You are a good assistant<｜User｜>What is the weather in Copenhagen?<｜Assistant｜><think>"""
        )


# test_replicate_custom_prompt_dict()

# commenthing this out since we won't be always testing a custom, replicate deployment
# def test_completion_replicate_deployments():
#     print("TESTING REPLICATE")
#     litellm.set_verbose=False
#     model_name = "replicate/deployments/ishaan-jaff/ishaan-mistral"
#     try:
#         response = completion(
#             model=model_name,
#             messages=messages,
#             temperature=0.5,
#             seed=-1,
#         )
#         print(response)
#         # Add any assertions here to check the response
#         response_str = response["choices"][0]["message"]["content"]
#         print("RESPONSE STRING\n", response_str)
#         if type(response_str) != str:
#             pytest.fail(f"Error occurred: {e}")
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")
# test_completion_replicate_deployments()


######## Test TogetherAI ########
@pytest.mark.skip(reason="Skip flaky test")
def test_completion_together_ai_mixtral():
    model_name = "together_ai/DiscoResearch/DiscoLM-mixtral-8x7b-v2"
    try:
        messages = [
            {"role": "user", "content": "Who are you"},
            {"role": "assistant", "content": "I am your helpful assistant."},
            {"role": "user", "content": "Tell me a joke"},
        ]
        response = completion(
            model=model_name,
            messages=messages,
            max_tokens=256,
            n=1,
            logger_fn=logger_fn,
        )
        # Add any assertions here to check the response
        print(response)
        cost = completion_cost(completion_response=response)
        assert cost > 0.0
        print(
            "Cost for completion call together-computer/llama-2-70b: ",
            f"${float(cost):.10f}",
        )
    except litellm.Timeout as e:
        pass
    except litellm.ServiceUnavailableError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_together_ai_mixtral()


def test_completion_together_ai_llama():
    litellm.set_verbose = True
    model_name = "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    try:
        messages = [
            {"role": "user", "content": "What llm are you?"},
        ]
        response = completion(model=model_name, messages=messages, max_tokens=5)
        # Add any assertions here to check the response
        print(response)
        cost = completion_cost(completion_response=response)
        assert cost > 0.0
        print(
            "Cost for completion call together-computer/llama-2-70b: ",
            f"${float(cost):.10f}",
        )
    except litellm.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_together_ai_yi_chat()


# test_completion_together_ai()
def test_customprompt_together_ai():
    try:
        litellm.set_verbose = False
        litellm.num_retries = 0
        print("in test_customprompt_together_ai")
        print(litellm.success_callback)
        print(litellm._async_success_callback)
        response = completion(
            model="together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1",
            messages=messages,
            roles={
                "system": {
                    "pre_message": "<|im_start|>system\n",
                    "post_message": "<|im_end|>",
                },
                "assistant": {
                    "pre_message": "<|im_start|>assistant\n",
                    "post_message": "<|im_end|>",
                },
                "user": {
                    "pre_message": "<|im_start|>user\n",
                    "post_message": "<|im_end|>",
                },
            },
        )
        print(response)
    except litellm.exceptions.Timeout as e:
        print(f"Timeout Error")
        pass
    except Exception as e:
        print(f"ERROR TYPE {type(e)}")
        pytest.fail(f"Error occurred: {e}")


# test_customprompt_together_ai()


def response_format_tests(response: litellm.ModelResponse):
    assert isinstance(response.id, str)
    assert response.id != ""

    assert isinstance(response.object, str)
    assert response.object != ""

    assert isinstance(response.created, int)

    assert isinstance(response.model, str)
    assert response.model != ""

    assert isinstance(response.choices, list)
    assert len(response.choices) == 1
    choice = response.choices[0]
    assert isinstance(choice, litellm.Choices)
    assert isinstance(choice.get("index"), int)

    message = choice.get("message")
    assert isinstance(message, litellm.Message)
    assert isinstance(message.get("role"), str)
    assert message.get("role") != ""
    assert isinstance(message.get("content"), str)
    assert message.get("content") != ""

    assert choice.get("logprobs") is None
    assert isinstance(choice.get("finish_reason"), str)
    assert choice.get("finish_reason") != ""

    assert isinstance(response.usage, litellm.Usage)  # type: ignore
    assert isinstance(response.usage.prompt_tokens, int)  # type: ignore
    assert isinstance(response.usage.completion_tokens, int)  # type: ignore
    assert isinstance(response.usage.total_tokens, int)  # type: ignore


@pytest.mark.parametrize(
    "model",
    [
        "bedrock/mistral.mistral-large-2407-v1:0",
        "bedrock/cohere.command-r-plus-v1:0",
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "mistral.mistral-7b-instruct-v0:2",
        # "bedrock/amazon.titan-tg1-large",
        "meta.llama3-8b-instruct-v1:0",
    ],
)
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_completion_bedrock_httpx_models(sync_mode, model):
    litellm.set_verbose = True
    try:

        if sync_mode:
            response = completion(
                model=model,
                messages=[{"role": "user", "content": "Hey! how's it going?"}],
                temperature=0.2,
                max_tokens=200,
            )

            assert isinstance(response, litellm.ModelResponse)

            response_format_tests(response=response)
        else:
            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": "Hey! how's it going?"}],
                temperature=0.2,
                max_tokens=100,
            )

            assert isinstance(response, litellm.ModelResponse)

            print(f"response: {response}")
            response_format_tests(response=response)

        print(f"response: {response}")
    except litellm.RateLimitError as e:
        print("got rate limit error=", e)
        pass
    except Exception as e:
        pytest.fail(f"An error occurred - {str(e)}")


def test_completion_bedrock_titan_null_response():
    try:
        response = completion(
            model="bedrock/amazon.titan-text-lite-v1",
            messages=[
                {
                    "role": "user",
                    "content": "Hello!",
                },
                {
                    "role": "assistant",
                    "content": "Hello! How can I help you?",
                },
                {
                    "role": "user",
                    "content": "What model are you?",
                },
            ],
        )
        # Add any assertions here to check the response
        print(f"response: {response}")
    except Exception as e:
        pytest.fail(f"An error occurred - {str(e)}")


# test_completion_bedrock_titan()


# test_completion_bedrock_claude()


# def test_completion_bedrock_claude_stream():
#     print("calling claude")
#     litellm.set_verbose = False
#     try:
#         response = completion(
#             model="bedrock/anthropic.claude-instant-v1",
#             messages=messages,
#             stream=True
#         )
#         # Add any assertions here to check the response
#         print(response)
#         for chunk in response:
#             print(chunk)
#     except RateLimitError:
#         pass
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")
# test_completion_bedrock_claude_stream()


######## Test VLLM ########
# def test_completion_vllm():
#     try:
#         response = completion(
#             model="vllm/facebook/opt-125m",
#             messages=messages,
#             temperature=0.2,
#             max_tokens=80,
#         )
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# test_completion_vllm()

# def test_completion_hosted_chatCompletion():
#     # this tests calling a server where vllm is hosted
#     # this should make an openai.Completion() call to the specified api_base
#     # send a request to this proxy server: https://replit.com/@BerriAI/openai-proxy#main.py
#     # it checks if model == facebook/opt-125m and returns test passed
#     try:
#         litellm.set_verbose = True
#         response = completion(
#             model="facebook/opt-125m",
#             messages=messages,
#             temperature=0.2,
#             max_tokens=80,
#             api_base="https://openai-proxy.berriai.repl.co",
#             custom_llm_provider="openai"
#         )
#         print(response)

#         if response['choices'][0]['message']['content'] != "passed":
#             # see https://replit.com/@BerriAI/openai-proxy#main.py
#             pytest.fail(f"Error occurred: proxy server did not respond")
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# test_completion_hosted_chatCompletion()

# def test_completion_custom_api_base():
#     try:
#         response = completion(
#             model="custom/meta-llama/Llama-2-13b-hf",
#             messages=messages,
#             temperature=0.2,
#             max_tokens=10,
#             api_base="https://api.autoai.dev/inference",
#             request_timeout=300,
#         )
#         # Add any assertions here to check the response
#         print("got response\n", response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# test_completion_custom_api_base()


def test_completion_with_fallbacks():
    print(f"RUNNING TEST COMPLETION WITH FALLBACKS -  test_completion_with_fallbacks")
    fallbacks = ["gpt-3.5-turbo", "gpt-3.5-turbo", "command-nightly"]
    try:
        response = completion(
            model="bad-model", messages=messages, force_timeout=120, fallbacks=fallbacks
        )
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_with_fallbacks()


# @pytest.mark.parametrize(
#     "function_call",
#     [
#         [{"role": "function", "name": "get_capital", "content": "Kokoko"}],
#         [
#             {"role": "function", "name": "get_capital", "content": "Kokoko"},
#             {"role": "function", "name": "get_capital", "content": "Kokoko"},
#         ],
#     ],
# )
# @pytest.mark.parametrize(
#     "tool_call",
#     [
#         [{"role": "tool", "tool_call_id": "1234", "content": "Kokoko"}],
#         [
#             {"role": "tool", "tool_call_id": "12344", "content": "Kokoko"},
#             {"role": "tool", "tool_call_id": "1214", "content": "Kokoko"},
#         ],
#     ],
# )
def test_completion_anthropic_hanging():
    litellm.set_verbose = True
    litellm.modify_params = True
    messages = [
        {
            "role": "user",
            "content": "What's the capital of fictional country Ubabababababaaba? Use your tools.",
        },
        {
            "role": "assistant",
            "function_call": {
                "name": "get_capital",
                "arguments": '{"country": "Ubabababababaaba"}',
            },
        },
        {"role": "function", "name": "get_capital", "content": "Kokoko"},
    ]

    converted_messages = anthropic_messages_pt(
        messages, model="claude-3-sonnet-20240229", llm_provider="anthropic"
    )

    print(f"converted_messages: {converted_messages}")

    ## ENSURE USER / ASSISTANT ALTERNATING
    for i, msg in enumerate(converted_messages):
        if i < len(converted_messages) - 1:
            assert msg["role"] != converted_messages[i + 1]["role"]


@pytest.mark.skip(reason="anyscale stopped serving public api endpoints")
def test_completion_anyscale_api():
    try:
        # litellm.set_verbose = True
        messages = [
            {"role": "system", "content": "You're a good bot"},
            {
                "role": "user",
                "content": "Hey",
            },
            {
                "role": "user",
                "content": "Hey",
            },
        ]
        response = completion(
            model="anyscale/meta-llama/Llama-2-7b-chat-hf",
            messages=messages,
        )
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")



@pytest.mark.skip(reason="anyscale stopped serving public api endpoints")
def test_completion_anyscale_2():
    try:
        # litellm.set_verbose = True
        messages = [
            {"role": "system", "content": "You're a good bot"},
            {
                "role": "user",
                "content": "Hey",
            },
            {
                "role": "user",
                "content": "Hey",
            },
        ]
        response = completion(
            model="anyscale/meta-llama/Llama-2-7b-chat-hf", messages=messages
        )
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.skip(reason="anyscale stopped serving public api endpoints")
def test_mistral_anyscale_stream():
    litellm.set_verbose = False
    response = completion(
        model="anyscale/mistralai/Mistral-7B-Instruct-v0.1",
        messages=[{"content": "hello, good morning", "role": "user"}],
        stream=True,
    )
    for chunk in response:
        # print(chunk)
        print(chunk["choices"][0]["delta"].get("content", ""), end="")


# test_completion_anyscale_2()
# def test_completion_with_fallbacks_multiple_keys():
#     print(f"backup key 1: {os.getenv('BACKUP_OPENAI_API_KEY_1')}")
#     print(f"backup key 2: {os.getenv('BACKUP_OPENAI_API_KEY_2')}")
#     backup_keys = [{"api_key": os.getenv("BACKUP_OPENAI_API_KEY_1")}, {"api_key": os.getenv("BACKUP_OPENAI_API_KEY_2")}]
#     try:
#         api_key = "bad-key"
#         response = completion(
#             model="gpt-3.5-turbo", messages=messages, force_timeout=120, fallbacks=backup_keys, api_key=api_key
#         )
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         error_str = traceback.format_exc()
#         pytest.fail(f"Error occurred: {error_str}")


# test_completion_with_fallbacks_multiple_keys()
def test_petals():
    try:
        from litellm.llms.custom_httpx.http_handler import HTTPHandler

        client = HTTPHandler()
        with patch.object(client, "post") as mock_post:
            try:
                completion(
                    model="petals-team/StableBeluga2",
                    messages=messages,
                    client=client,
                    api_base="https://api.petals.dev",
                )
            except Exception as e:
                print(f"Error occurred: {e}")
            mock_post.assert_called_once()
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# def test_baseten():
#     try:

#         response = completion(model="baseten/7qQNLDB", messages=messages, logger_fn=logger_fn)
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# test_baseten()
# def test_baseten_falcon_7bcompletion():
#     model_name = "qvv0xeq"
#     try:
#         response = completion(model=model_name, messages=messages, custom_llm_provider="baseten")
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# def test_baseten_falcon_7bcompletion_withbase():
#     model_name = "qvv0xeq"
#     litellm.api_base = "https://app.baseten.co"
#     try:
#         response = completion(model=model_name, messages=messages)
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")
#     litellm.api_base = None

# test_baseten_falcon_7bcompletion_withbase()


# def test_baseten_wizardLMcompletion_withbase():
#     model_name = "q841o8w"
#     litellm.api_base = "https://app.baseten.co"
#     try:
#         response = completion(model=model_name, messages=messages)
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# test_baseten_wizardLMcompletion_withbase()

# def test_baseten_mosaic_ML_completion_withbase():
#     model_name = "31dxrj3",
#     litellm.api_base = "https://app.baseten.co"
#     try:
#         response = completion(model=model_name, messages=messages)
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")


# test_completion_ai21()
# test_completion_ai21()
## test deep infra
@pytest.mark.parametrize("drop_params", [True, False])
def test_completion_deep_infra(drop_params):
    litellm.set_verbose = False
    model_name = "deepinfra/meta-llama/Llama-2-70b-chat-hf"
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
        response = completion(
            model=model_name,
            messages=messages,
            temperature=0,
            max_tokens=10,
            tools=tools,
            tool_choice={
                "type": "function",
                "function": {"name": "get_current_weather"},
            },
            drop_params=drop_params,
        )
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        if drop_params is True:
            pytest.fail(f"Error occurred: {e}")


# test_completion_deep_infra()


def test_completion_deep_infra_mistral():
    print("deep infra test with temp=0")
    model_name = "deepinfra/mistralai/Mistral-7B-Instruct-v0.1"
    try:
        response = completion(
            model=model_name,
            messages=messages,
            temperature=0.01,  # mistrail fails with temperature=0
            max_tokens=10,
        )
        # Add any assertions here to check the response
        print(response)
    except litellm.exceptions.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_deep_infra_mistral()


@pytest.mark.skip(reason="Local test - don't have a volcengine account as yet")
def test_completion_volcengine():
    litellm.set_verbose = True
    model_name = "volcengine/<OUR_ENDPOINT_ID>"
    try:
        response = completion(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather like in Boston today in Fahrenheit?",
                }
            ],
            api_key="<OUR_API_KEY>",
        )
        # Add any assertions here to check the response
        print(response)

    except litellm.exceptions.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# Gemini tests
@pytest.mark.parametrize(
    "model",
    [
        # "gemini-1.0-pro",
        "gemini-2.5-flash-lite",
    ],
)
@pytest.mark.flaky(retries=3, delay=1)
def test_completion_gemini(model):
    litellm.set_verbose = True
    model_name = "gemini/{}".format(model)
    messages = [
        {"role": "system", "content": "Be a good bot!"},
        {"role": "user", "content": "Hey, how's it going?"},
    ]
    try:
        response = completion(
            model=model_name,
            messages=messages,
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ],
        )
        # Add any assertions,here to check the response
        print(response)
        assert response.choices[0]["index"] == 0
    except litellm.RateLimitError:
        pass
    except litellm.APIError:
        pass
    except Exception as e:
        if "InternalServerError" in str(e):
            pass
        else:
            pytest.fail(f"Error occurred:{e}")


# test_completion_gemini()


@pytest.mark.asyncio
async def test_acompletion_gemini():
    litellm.set_verbose = True
    model_name = "gemini/gemini-2.5-flash-lite"
    messages = [{"role": "user", "content": "Hey, how's it going?"}]
    try:
        response = await litellm.acompletion(model=model_name, messages=messages)
        # Add any assertions here to check the response
        print(f"response: {response}")
    except litellm.Timeout as e:
        pass
    except litellm.APIError as e:
        pass
    except Exception as e:
        if "InternalServerError" in str(e):
            pass
        else:
            pytest.fail(f"Error occurred: {e}")


# Deepseek tests
def test_completion_deepseek():
    litellm.set_verbose = True
    model_name = "deepseek/deepseek-chat"
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather of an location, the user shoud supply a location first",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        }
                    },
                    "required": ["location"],
                },
            },
        },
    ]
    messages = [{"role": "user", "content": "How's the weather in Hangzhou?"}]
    try:
        response = completion(model=model_name, messages=messages, tools=tools)
        # Add any assertions here to check the response
        print(response)
    except litellm.APIError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.skip(reason="Account deleted by IBM.")
def test_completion_watsonx_error():
    litellm.set_verbose = True
    model_name = "watsonx_text/ibm/granite-13b-chat-v2"

    response = completion(
        model=model_name,
        messages=messages,
        stop=["stop"],
        max_tokens=20,
        stream=True,
    )

    for chunk in response:
        print(chunk)
    # Add any assertions here to check the response
    print(response)


@pytest.mark.skip(reason="Skip test. account deleted.")
def test_completion_stream_watsonx():
    litellm.set_verbose = True
    model_name = "watsonx/ibm/granite-13b-chat-v2"
    try:
        response = completion(
            model=model_name,
            messages=messages,
            stop=["stop"],
            max_tokens=20,
            stream=True,
        )
        for chunk in response:
            print(chunk)
    except litellm.APIError as e:
        pass
    except litellm.RateLimitError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize(
    "provider, model, project, region_name, token",
    [
        ("azure", "chatgpt-v-3", None, None, "test-token"),
        ("vertex_ai", "anthropic-claude-3", "adroit-crow-1", "us-east1", None),
        ("watsonx", "ibm/granite", "96946574", "dallas", "1234"),
        ("bedrock", "anthropic.claude-3", None, "us-east-1", None),
    ],
)
def test_unified_auth_params(provider, model, project, region_name, token):
    """
    Check if params = ["project", "region_name", "token"]
    are correctly translated for = ["azure", "vertex_ai", "watsonx", "aws"]

    tests get_optional_params
    """
    data = {
        "project": project,
        "region_name": region_name,
        "token": token,
        "custom_llm_provider": provider,
        "model": model,
    }

    translated_optional_params = litellm.utils.get_optional_params(**data)

    if provider == "azure":
        special_auth_params = (
            litellm.AzureOpenAIConfig().get_mapped_special_auth_params()
        )
    elif provider == "bedrock":
        special_auth_params = (
            litellm.AmazonBedrockGlobalConfig().get_mapped_special_auth_params()
        )
    elif provider == "vertex_ai":
        special_auth_params = litellm.VertexAIConfig().get_mapped_special_auth_params()
    elif provider == "watsonx":
        special_auth_params = (
            litellm.IBMWatsonXAIConfig().get_mapped_special_auth_params()
        )

    for param, value in special_auth_params.items():
        assert param in data
        assert value in translated_optional_params


@pytest.mark.skip(reason="Local test")
@pytest.mark.asyncio
async def test_acompletion_watsonx():
    litellm.set_verbose = True
    model_name = "watsonx/ibm/granite-13b-chat-v2"
    print("testing watsonx")
    try:
        response = await litellm.acompletion(
            model=model_name,
            messages=messages,
            temperature=0.2,
            max_tokens=80,
        )
        # Add any assertions here to check the response
        print(response)
    except litellm.RateLimitError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.skip(reason="Local test")
@pytest.mark.asyncio
async def test_acompletion_stream_watsonx():
    litellm.set_verbose = True
    model_name = "watsonx/ibm/granite-13b-chat-v2"
    print("testing watsonx")
    try:
        response = await litellm.acompletion(
            model=model_name,
            messages=messages,
            temperature=0.2,
            max_tokens=80,
            stream=True,
        )
        # Add any assertions here to check the response
        async for chunk in response:
            print(chunk)
    except litellm.RateLimitError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_palm_stream()

# test_completion_deep_infra()
# test_completion_ai21()
# test config file with completion #
# def test_completion_openai_config():
#     try:
#         litellm.config_path = "../config.json"
#         litellm.set_verbose = True
#         response = litellm.config_completion(messages=messages)
#         # Add any assertions here to check the response
#         print(response)
#         litellm.config_path = None
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")


# def test_maritalk():
#     messages = [{"role": "user", "content": "Hey"}]
#     try:
#         response = completion("maritalk", messages=messages)
#         print(f"response: {response}")
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")
# test_maritalk()


def test_completion_together_ai_stream():
    litellm.set_verbose = True
    user_message = "Write 1pg about YC & litellm"
    messages = [{"content": user_message, "role": "user"}]
    try:
        response = completion(
            model="together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1",
            messages=messages,
            stream=True,
            max_tokens=5,
        )
        print(response)
        for chunk in response:
            print(chunk)
        # print(string_response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_together_ai_stream()


def test_moderation():
    response = litellm.moderation(input="i'm ishaan cto of litellm")
    print(response)
    output = response.results[0]
    print(output)
    return output


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("sync_mode", [False, True])
@pytest.mark.asyncio
async def test_dynamic_azure_params(stream, sync_mode):
    """
    If dynamic params are given, which are different from the initialized client, use a new client
    """
    from openai import AsyncAzureOpenAI, AzureOpenAI

    if sync_mode:
        client = AzureOpenAI(
            api_key="my-test-key",
            base_url="my-test-base",
            api_version="my-test-version",
        )
        mock_client = MagicMock(return_value="Hello world!")
    else:
        client = AsyncAzureOpenAI(
            api_key="my-test-key",
            base_url="my-test-base",
            api_version="my-test-version",
        )
        mock_client = AsyncMock(return_value="Hello world!")

    ## CHECK IF CLIENT IS USED (NO PARAM CHANGE)
    with patch.object(
        client.chat.completions.with_raw_response, "create", new=mock_client
    ) as mock_client:
        try:
            # client.chat.completions.with_raw_response.create = mock_client
            if sync_mode:
                _ = completion(
                    model="azure/chatgpt-v2",
                    messages=[{"role": "user", "content": "Hello world"}],
                    client=client,
                    stream=stream,
                )
            else:
                _ = await litellm.acompletion(
                    model="azure/chatgpt-v2",
                    messages=[{"role": "user", "content": "Hello world"}],
                    client=client,
                    stream=stream,
                )
        except Exception:
            pass

        mock_client.assert_called()

    ## recreate mock client
    if sync_mode:
        new_mock_client = MagicMock(return_value="Hello world!")
    else:
        new_mock_client = AsyncMock(return_value="Hello world!")

    ## CHECK IF NEW CLIENT IS USED (PARAM CHANGE)
    with patch.object(
        client.chat.completions.with_raw_response, "create", new=new_mock_client
    ) as new_mock_client:
        try:
            if sync_mode:
                _ = completion(
                    model="azure/chatgpt-v2",
                    messages=[{"role": "user", "content": "Hello world"}],
                    client=client,
                    api_version="my-new-version",
                    stream=stream,
                )
            else:
                _ = await litellm.acompletion(
                    model="azure/chatgpt-v2",
                    messages=[{"role": "user", "content": "Hello world"}],
                    client=client,
                    api_version="my-new-version",
                    stream=stream,
                )
        except Exception:
            pass

        try:
            new_mock_client.assert_called()
        except Exception as e:
            raise e


@pytest.mark.asyncio()
@pytest.mark.flaky(retries=3, delay=1)
async def test_completion_ai21_chat():
    litellm.set_verbose = True
    try:
        response = await litellm.acompletion(
            model="ai21_chat/jamba-mini",
            user="ishaan",
            tool_choice="auto",
            seed=123,
            messages=[{"role": "user", "content": "what does the document say"}],
            documents=[
                {
                    "content": "hello world",
                    "metadata": {"source": "google", "author": "ishaan"},
                }
            ],
        )
    except litellm.InternalServerError:
        pytest.skip("Model is overloaded")


@pytest.mark.parametrize(
    "model",
    ["gpt-4o", "azure/gpt-4.1-mini"],
)
@pytest.mark.parametrize(
    "stream",
    [False, True],
)
@pytest.mark.flaky(retries=3, delay=1)
def test_completion_response_ratelimit_headers(model, stream):
    response = completion(
        model=model,
        messages=[{"role": "user", "content": "Hello world"}],
        stream=stream,
    )
    hidden_params = response._hidden_params
    additional_headers = hidden_params.get("additional_headers", {})

    print(additional_headers)
    for k, v in additional_headers.items():
        assert v != "None" and v is not None
    assert "x-ratelimit-remaining-requests" in additional_headers
    assert "x-ratelimit-remaining-tokens" in additional_headers

    if model == "azure/gpt-4.1-mini":
        # Azure OpenAI header
        assert "llm_provider-azureml-model-session" in additional_headers
    if model == "claude-3-sonnet-20240229":
        # anthropic header
        assert "llm_provider-anthropic-ratelimit-requests-reset" in additional_headers


def _openai_hallucinated_tool_call_mock_response(
    *args, **kwargs
) -> litellm.ModelResponse:
    new_response = MagicMock()
    new_response.headers = {"hello": "world"}

    response_object = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-3.5-turbo-0125",
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [
            {
                "index": 0,
                "message": {
                    "content": None,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": '{"tool_uses":[{"recipient_name":"product_title","parameters":{"content":"Story Scribe"}},{"recipient_name":"one_liner","parameters":{"content":"Transform interview transcripts into actionable user stories"}}]}',
                                "name": "multi_tool_use.parallel",
                            },
                            "id": "call_IzGXwVa5OfBd9XcCJOkt2q0s",
                            "type": "function",
                        }
                    ],
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
    }
    from openai import OpenAI
    from openai.types.chat.chat_completion import ChatCompletion

    pydantic_obj = ChatCompletion(**response_object)  # type: ignore
    pydantic_obj.choices[0].message.role = None  # type: ignore
    new_response.parse.return_value = pydantic_obj
    return new_response


def test_openai_hallucinated_tool_call():
    """
    Patch for this issue: https://community.openai.com/t/model-tries-to-call-unknown-function-multi-tool-use-parallel/490653

    Handle openai invalid tool calling response.

    OpenAI assistant will sometimes return an invalid tool calling response, which needs to be parsed

    -           "arguments": "{\"tool_uses\":[{\"recipient_name\":\"product_title\",\"parameters\":{\"content\":\"Story Scribe\"}},{\"recipient_name\":\"one_liner\",\"parameters\":{\"content\":\"Transform interview transcripts into actionable user stories\"}}]}",

    To extract actual tool calls:

    1. Parse arguments JSON object
    2. Iterate over tool_uses array to call functions:
        - get function name from recipient_name value
        - parameters will be JSON object for function arguments
    """
    import openai

    openai_client = openai.OpenAI()
    with patch.object(
        openai_client.chat.completions,
        "create",
        side_effect=_openai_hallucinated_tool_call_mock_response,
    ) as mock_response:
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hey! how's it going?"}],
            client=openai_client,
        )
        print(f"response: {response}")

        response_dict = response.model_dump()

        tool_calls = response_dict["choices"][0]["message"]["tool_calls"]

        print(f"tool_calls: {tool_calls}")

        for idx, tc in enumerate(tool_calls):
            if idx == 0:
                print(f"tc in test_openai_hallucinated_tool_call: {tc}")
                assert tc == {
                    "function": {
                        "arguments": '{"content": "Story Scribe"}',
                        "name": "product_title",
                    },
                    "id": "call_IzGXwVa5OfBd9XcCJOkt2q0s_0",
                    "type": "function",
                }
            elif idx == 1:
                assert tc == {
                    "function": {
                        "arguments": '{"content": "Transform interview transcripts into actionable user stories"}',
                        "name": "one_liner",
                    },
                    "id": "call_IzGXwVa5OfBd9XcCJOkt2q0s_1",
                    "type": "function",
                }


@pytest.mark.parametrize(
    "function_name, expect_modification",
    [
        ("multi_tool_use.parallel", True),
        ("my-fake-function", False),
    ],
)
def test_openai_hallucinated_tool_call_util(function_name, expect_modification):
    """
    Patch for this issue: https://community.openai.com/t/model-tries-to-call-unknown-function-multi-tool-use-parallel/490653

    Handle openai invalid tool calling response.

    OpenAI assistant will sometimes return an invalid tool calling response, which needs to be parsed

    -           "arguments": "{\"tool_uses\":[{\"recipient_name\":\"product_title\",\"parameters\":{\"content\":\"Story Scribe\"}},{\"recipient_name\":\"one_liner\",\"parameters\":{\"content\":\"Transform interview transcripts into actionable user stories\"}}]}",

    To extract actual tool calls:

    1. Parse arguments JSON object
    2. Iterate over tool_uses array to call functions:
        - get function name from recipient_name value
        - parameters will be JSON object for function arguments
    """
    from litellm.utils import _handle_invalid_parallel_tool_calls
    from litellm.types.utils import ChatCompletionMessageToolCall

    response = _handle_invalid_parallel_tool_calls(
        tool_calls=[
            ChatCompletionMessageToolCall(
                **{
                    "function": {
                        "arguments": '{"tool_uses":[{"recipient_name":"product_title","parameters":{"content":"Story Scribe"}},{"recipient_name":"one_liner","parameters":{"content":"Transform interview transcripts into actionable user stories"}}]}',
                        "name": function_name,
                    },
                    "id": "call_IzGXwVa5OfBd9XcCJOkt2q0s",
                    "type": "function",
                }
            )
        ]
    )

    print(f"response: {response}")

    if expect_modification:
        for idx, tc in enumerate(response):
            if idx == 0:
                assert tc.model_dump() == {
                    "function": {
                        "arguments": '{"content": "Story Scribe"}',
                        "name": "product_title",
                    },
                    "id": "call_IzGXwVa5OfBd9XcCJOkt2q0s_0",
                    "type": "function",
                }
            elif idx == 1:
                assert tc.model_dump() == {
                    "function": {
                        "arguments": '{"content": "Transform interview transcripts into actionable user stories"}',
                        "name": "one_liner",
                    },
                    "id": "call_IzGXwVa5OfBd9XcCJOkt2q0s_1",
                    "type": "function",
                }
    else:
        assert len(response) == 1
        assert response[0].function.name == function_name


def test_langfuse_completion(monkeypatch):
    monkeypatch.setenv(
        "LANGFUSE_PUBLIC_KEY", "pk-lf-b3db7e8e-c2f6-4fc7-825c-a541a8fbe003"
    )
    monkeypatch.setenv(
        "LANGFUSE_SECRET_KEY", "sk-lf-b11ef3a8-361c-4445-9652-12318b8596e4"
    )
    monkeypatch.setenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
    litellm.set_verbose = True
    resp = litellm.completion(
        model="langfuse/gpt-3.5-turbo",
        langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        langfuse_host="https://us.cloud.langfuse.com",
        prompt_id="test-chat-prompt",
        prompt_variables={"user_message": "this is used"},
        messages=[{"role": "user", "content": "this is ignored"}],
    )


def test_completion_novita_ai():
    litellm.set_verbose = True
    messages = [
        {"role": "system", "content": "You're a good bot"},
        {
            "role": "user",
            "content": "Hey",
        },
    ]

    from openai import OpenAI

    openai_client = OpenAI(api_key="fake-key")

    with patch.object(
        openai_client.chat.completions, "create", new=MagicMock()
    ) as mock_call:
        try:
            completion(
                model="novita/meta-llama/llama-3.3-70b-instruct",
                messages=messages,
                client=openai_client,
                api_base="https://api.novita.ai/v3/openai",
            )

            mock_call.assert_called_once()

            # Verify model is passed correctly
            assert (
                mock_call.call_args.kwargs["model"]
                == "meta-llama/llama-3.3-70b-instruct"
            )
            # Verify messages are passed correctly
            assert mock_call.call_args.kwargs["messages"] == messages

        except Exception as e:
            pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize("api_key", ["my-bad-api-key"])
def test_completion_novita_ai_dynamic_params(api_key):
    try:
        litellm.set_verbose = True
        messages = [
            {"role": "system", "content": "You're a good bot"},
            {
                "role": "user",
                "content": "Hey",
            },
        ]

        from openai import OpenAI

        openai_client = OpenAI(api_key="fake-key")

        with patch.object(
            openai_client.chat.completions,
            "create",
            side_effect=Exception("Invalid API key"),
        ) as mock_call:
            try:
                completion(
                    model="novita/meta-llama/llama-3.3-70b-instruct",
                    messages=messages,
                    api_key=api_key,
                    client=openai_client,
                    api_base="https://api.novita.ai/v3/openai",
                )
                pytest.fail(f"This call should have failed!")
            except Exception as e:
                # This should fail with the mocked exception
                assert "Invalid API key" in str(e)

            mock_call.assert_called_once()
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")


def test_deepseek_reasoning_content_completion():
    try:
        litellm.set_verbose = True
        litellm._turn_on_debug()
        resp = litellm.completion(
            timeout=5,
            model="deepseek/deepseek-reasoner",
            messages=[{"role": "user", "content": "Tell me a joke."}],
        )

        assert resp.choices[0].message.reasoning_content is not None
    except litellm.Timeout:
        pytest.skip("Model is timing out")


def test_qwen_text_completion():
    # litellm._turn_on_debug()
    resp = litellm.completion(
        model="gpt-3.5-turbo-instruct",
        messages=[{"content": "hello", "role": "user"}],
        stream=False,
        logprobs=1,
    )
    assert resp.choices[0].message.content is not None
    assert resp.choices[0].logprobs.token_logprobs[0] is not None
    print(
        f"resp.choices[0].logprobs.token_logprobs[0]: {resp.choices[0].logprobs.token_logprobs[0]}"
    )


@pytest.mark.parametrize(
    "enable_preview_features",
    [True, False],
)
def test_completion_openai_metadata(monkeypatch, enable_preview_features):
    from openai import OpenAI

    client = OpenAI()

    litellm.set_verbose = True

    monkeypatch.setattr(litellm, "enable_preview_features", enable_preview_features)
    with patch.object(
        client.chat.completions.with_raw_response, "create", return_value=MagicMock()
    ) as mock_completion:
        try:
            resp = litellm.completion(
                model="openai/gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello world"}],
                metadata={"my-test-key": "my-test-value"},
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_completion.assert_called_once()
        if enable_preview_features:
            assert mock_completion.call_args.kwargs["metadata"] == {
                "my-test-key": "my-test-value"
            }
        else:
            assert "metadata" not in mock_completion.call_args.kwargs


def test_completion_o3_mini_temperature():
    try:
        litellm.set_verbose = True
        resp = litellm.completion(
            model="o3-mini",
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": "Hello, world!",
                }
            ],
            drop_params=True,
        )
        assert resp.choices[0].message.content is not None
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_completion_gpt_4o_empty_str():
    litellm._turn_on_debug()
    from openai import OpenAI
    from unittest.mock import MagicMock

    client = OpenAI()

    # Create response object matching OpenAI's format
    mock_response_data = {
        "id": "chatcmpl-B0W3vmiM78Xkgx7kI7dr7PC949DMS",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": "",
                    "refusal": None,
                    "role": "assistant",
                    "audio": None,
                    "function_call": None,
                    "tool_calls": None,
                },
            }
        ],
        "created": 1739462947,
        "model": "gpt-4o-mini-2024-07-18",
        "object": "chat.completion",
        "service_tier": "default",
        "system_fingerprint": "fp_bd83329f63",
        "usage": {
            "completion_tokens": 1,
            "prompt_tokens": 121,
            "total_tokens": 122,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "audio_tokens": 0,
                "reasoning_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
            "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
        },
    }

    # Create a mock response object
    mock_raw_response = MagicMock()
    mock_raw_response.headers = {
        "x-request-id": "123",
        "openai-organization": "org-123",
        "x-ratelimit-limit-requests": "100",
        "x-ratelimit-remaining-requests": "99",
    }
    mock_raw_response.parse.return_value = mock_response_data

    # Set up the mock completion
    mock_completion = MagicMock()
    mock_completion.return_value = mock_raw_response

    with patch.object(
        client.chat.completions.with_raw_response, "create", mock_completion
    ) as mock_create:
        resp = litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": ""}],
        )
        assert resp.choices[0].message.content is not None


def test_edit_note():
    litellm.callbacks = ["langfuse_otel"]
    response = completion(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Your only job is to call the edit_note tool with the content specified in the user's message.",
            },
            {
                "role": "user",
                "content": "Edit the note with the content: 'This is a test note.'",
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "edit_note",
                    "description": "Edit the note with the content specified in the user's message.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                        },
                    },
                },
            },
        ],
    )

    return response
