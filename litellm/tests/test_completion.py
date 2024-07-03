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
)  # Adds-the parent directory to the system path

import os
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm import RateLimitError, Timeout, completion, completion_cost, embedding
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.prompt_templates.factory import anthropic_messages_pt

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
    Addresses: https://github.com/BerriAI/litellm/issues/2972
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
    _data = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-3.5-turbo-0125",
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": None,
                    "content": "\n\nHello there, how may I assist you today?",
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
    }
    return litellm.ModelResponse(**_data)


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


def test_completion_azure_ai_command_r():
    try:
        import os

        litellm.set_verbose = True

        os.environ["AZURE_AI_API_BASE"] = os.getenv("AZURE_COHERE_API_BASE", "")
        os.environ["AZURE_AI_API_KEY"] = os.getenv("AZURE_COHERE_API_KEY", "")

        response: litellm.ModelResponse = completion(
            model="azure_ai/command-r-plus",
            messages=[{"role": "user", "content": "What is the meaning of life?"}],
        )  # type: ignore

        assert "azure_ai" in response.model
    except litellm.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_completion_azure_command_r():
    try:
        litellm.set_verbose = True

        response = completion(
            model="azure/command-r-plus",
            api_base=os.getenv("AZURE_COHERE_API_BASE"),
            api_key=os.getenv("AZURE_COHERE_API_KEY"),
            messages=[{"role": "user", "content": "What is the meaning of life?"}],
        )

        print(response)
    except litellm.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_completion_databricks(sync_mode):
    litellm.set_verbose = True

    if sync_mode:
        response: litellm.ModelResponse = completion(
            model="databricks/databricks-dbrx-instruct",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
        )  # type: ignore

    else:
        response: litellm.ModelResponse = await litellm.acompletion(
            model="databricks/databricks-dbrx-instruct",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
        )  # type: ignore
    print(f"response: {response}")

    response_format_tests(response=response)


def predibase_mock_post(url, data=None, json=None, headers=None):
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


# @pytest.mark.skip(reason="local only test")
@pytest.mark.asyncio
async def test_completion_predibase():
    try:
        litellm.set_verbose = True

        with patch("requests.post", side_effect=predibase_mock_post):
            response = completion(
                model="predibase/llama-3-8b-instruct",
                tenant_id="c4768f95",
                api_key=os.getenv("PREDIBASE_API_KEY"),
                messages=[{"role": "user", "content": "What is the meaning of life?"}],
                max_tokens=10,
            )

            print(response)
    except litellm.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_predibase()


def test_completion_claude():
    litellm.set_verbose = True
    litellm.cache = None
    litellm.AnthropicTextConfig(max_tokens_to_sample=200, metadata={"user_id": "1224"})
    messages = [
        {
            "role": "system",
            "content": """You are an upbeat, enthusiastic personal fitness coach named Sam. Sam is passionate about helping clients get fit and lead healthier lifestyles. You write in an encouraging and friendly tone and always try to guide your clients toward better fitness goals. If the user asks you something unrelated to fitness, either bring the topic back to fitness, or say that you cannot answer.""",
        },
        {"content": user_message, "role": "user"},
    ]
    try:
        # test without max tokens
        response = completion(
            model="claude-instant-1", messages=messages, request_timeout=10
        )
        # Add any assertions here to check response args
        print(response)
        print(response.usage)
        print(response.usage.completion_tokens)
        print(response["usage"]["completion_tokens"])
        # print("new cost tracking")
    except Exception as e:
        if "overloaded_error" in str(e):
            pass
        pytest.fail(f"Error occurred: {e}")


# test_completion_claude()


def test_completion_claude_3_empty_response():
    litellm.set_verbose = True

    messages = [
        {
            "role": "system",
            "content": "You are 2twNLGfqk4GMOn3ffp4p.",
        },
        {"role": "user", "content": "Hi gm!", "name": "ishaan"},
        {"role": "assistant", "content": "Good morning! How are you doing today?"},
        {
            "role": "user",
            "content": "I was hoping we could chat a bit",
        },
    ]
    response = litellm.completion(model="claude-3-opus-20240229", messages=messages)
    print(response)


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
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


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
    except litellm.APIError as e:
        assert e.status_code == 500
    except Exception as e:
        pytest.fail(f"An unexpected error occurred - {str(e)}")


def test_gemini_completion_call_error():
    try:
        print("test completion + streaming")
        litellm.num_retries = 3
        litellm.set_verbose = True
        messages = [{"role": "user", "content": "what is the capital of congo?"}]
        response = completion(
            model="gemini/gemini-1.5-pro-latest",
            messages=messages,
            stream=True,
            max_tokens=10,
        )
        print(f"response: {response}")
        for chunk in response:
            print(chunk)
    except Exception as e:
        pytest.fail(f"error occurred: {str(e)}")


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
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_parse_xml_params():
    from litellm.llms.prompt_templates.factory import parse_xml_params

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


@pytest.mark.skip(
    reason="we already test claude-3, this is just another way to pass images"
)
def test_completion_claude_3_base64():
    try:
        litellm.set_verbose = True
        litellm.num_retries = 3
        image_path = "../proxy/cached_logo.jpg"
        # Getting the base64 string
        base64_image = encode_image(image_path)
        resp = litellm.completion(
            model="anthropic/claude-3-opus-20240229",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Whats in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64," + base64_image
                            },
                        },
                    ],
                }
            ],
        )
        print(f"\nResponse: {resp}")

        prompt_tokens = resp.usage.prompt_tokens
        raise Exception("it worked!")
    except Exception as e:
        if "500 Internal error encountered.'" in str(e):
            pass
        else:
            pytest.fail(f"An exception occurred - {str(e)}")


@pytest.mark.parametrize(
    "model", ["gemini/gemini-1.5-flash"]  # "claude-3-sonnet-20240229",
)
def test_completion_function_plus_image(model):
    litellm.set_verbose = True

    image_content = [
        {"type": "text", "text": "What’s in this image?"},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://litellm-listing.s3.amazonaws.com/litellm_logo.png"
            },
        },
    ]
    image_message = {"role": "user", "content": image_content}

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

    tool_choice = {"type": "function", "function": {"name": "get_current_weather"}}
    messages = [
        {
            "role": "user",
            "content": "What's the weather like in Boston today in Fahrenheit?",
        }
    ]

    response = completion(
        model=model,
        messages=[image_message],
        tool_choice=tool_choice,
        tools=tools,
        stream=False,
    )

    print(response)


@pytest.mark.parametrize(
    "provider",
    ["azure", "azure_ai"],
)
def test_completion_azure_mistral_large_function_calling(provider):
    """
    This primarily tests if the 'Function()' pydantic object correctly handles argument param passed in as a dict vs. string
    """
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

    response = completion(
        model="{}/mistral-large-latest".format(provider),
        api_base=os.getenv("AZURE_MISTRAL_API_BASE"),
        api_key=os.getenv("AZURE_MISTRAL_API_KEY"),
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    # Add any assertions, here to check response args
    print(response)
    assert isinstance(response.choices[0].message.tool_calls[0].function.name, str)
    assert isinstance(response.choices[0].message.tool_calls[0].function.arguments, str)


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
        assert response.model == "mistral/mistral-tiny"
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


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
            model="mistral/mistral-large-latest",
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


def test_completion_claude2_1():
    try:
        litellm.set_verbose = True
        print("claude2.1 test request")
        messages = [
            {
                "role": "system",
                "content": "Your goal is generate a joke on the topic user gives.",
            },
            {"role": "user", "content": "Generate a 3 liner joke for me"},
        ]
        # test without max tokens
        response = completion(model="claude-2.1", messages=messages)
        # Add any assertions here to check the response
        print(response)
        print(response.usage)
        print(response.usage.completion_tokens)
        print(response["usage"]["completion_tokens"])
        # print("new cost tracking")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_claude2_1()


@pytest.mark.asyncio
async def test_acompletion_claude2_1():
    try:
        litellm.set_verbose = True
        print("claude2.1 test request")
        messages = [
            {
                "role": "system",
                "content": "Your goal is generate a joke on the topic user gives.",
            },
            {"role": "user", "content": "Generate a 3 liner joke for me"},
        ]
        # test without max-tokens
        response = await litellm.acompletion(model="claude-2.1", messages=messages)
        # Add any assertions here to check the response
        print(response)
        print(response.usage)
        print(response.usage.completion_tokens)
        print(response["usage"]["completion_tokens"])
        # print("new cost tracking")
    except Exception as e:
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
    try:
        response = completion(
            model="gpt-4-1106-preview",
            messages=messages,
            max_tokens=10,
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
            model="fireworks_ai/accounts/fireworks/models/mixtral-8x7b-instruct",
            messages=messages,
        )
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.skip(reason="this test is flaky")
def test_completion_perplexity_api():
    try:
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
        )
        print(response)
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


#####################################################
#####################################################
# Test util to sort models to TGI, conv, None
def test_get_hf_task_for_model():
    model = "glaiveai/glaive-coder-7b"
    model_type, _ = litellm.llms.huggingface_restapi.get_hf_task_for_model(model)
    print(f"model:{model}, model type: {model_type}")
    assert model_type == "text-generation-inference"

    model = "meta-llama/Llama-2-7b-hf"
    model_type, _ = litellm.llms.huggingface_restapi.get_hf_task_for_model(model)
    print(f"model:{model}, model type: {model_type}")
    assert model_type == "text-generation-inference"

    model = "facebook/blenderbot-400M-distill"
    model_type, _ = litellm.llms.huggingface_restapi.get_hf_task_for_model(model)
    print(f"model:{model}, model type: {model_type}")
    assert model_type == "conversational"

    model = "facebook/blenderbot-3B"
    model_type, _ = litellm.llms.huggingface_restapi.get_hf_task_for_model(model)
    print(f"model:{model}, model type: {model_type}")
    assert model_type == "conversational"

    # neither Conv or None
    model = "roneneldan/TinyStories-3M"
    model_type, _ = litellm.llms.huggingface_restapi.get_hf_task_for_model(model)
    print(f"model:{model}, model type: {model_type}")
    assert model_type == "text-generation"


# test_get_hf_task_for_model()
# litellm.set_verbose=False
# ################### Hugging Face TGI models ########################
# # TGI model
# # this is a TGI model https://huggingface.co/glaiveai/glaive-coder-7b
def tgi_mock_post(url, data=None, json=None, headers=None):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.json.return_value = [
        {
            "generated_text": "<|assistant|>\nI'm",
            "details": {
                "finish_reason": "length",
                "generated_tokens": 10,
                "seed": None,
                "prefill": [],
                "tokens": [
                    {
                        "id": 28789,
                        "text": "<",
                        "logprob": -0.025222778,
                        "special": False,
                    },
                    {
                        "id": 28766,
                        "text": "|",
                        "logprob": -0.000003695488,
                        "special": False,
                    },
                    {
                        "id": 489,
                        "text": "ass",
                        "logprob": -0.0000019073486,
                        "special": False,
                    },
                    {
                        "id": 11143,
                        "text": "istant",
                        "logprob": -0.000002026558,
                        "special": False,
                    },
                    {
                        "id": 28766,
                        "text": "|",
                        "logprob": -0.0000015497208,
                        "special": False,
                    },
                    {
                        "id": 28767,
                        "text": ">",
                        "logprob": -0.0000011920929,
                        "special": False,
                    },
                    {
                        "id": 13,
                        "text": "\n",
                        "logprob": -0.00009703636,
                        "special": False,
                    },
                    {"id": 28737, "text": "I", "logprob": -0.1953125, "special": False},
                    {
                        "id": 28742,
                        "text": "'",
                        "logprob": -0.88183594,
                        "special": False,
                    },
                    {
                        "id": 28719,
                        "text": "m",
                        "logprob": -0.00032639503,
                        "special": False,
                    },
                ],
            },
        }
    ]
    return mock_response


def test_hf_test_completion_tgi():
    litellm.set_verbose = True
    try:
        with patch("requests.post", side_effect=tgi_mock_post):
            response = completion(
                model="huggingface/HuggingFaceH4/zephyr-7b-beta",
                messages=[{"content": "Hello, how are you?", "role": "user"}],
                max_tokens=10,
            )
            # Add any assertions-here to check the response
            print(response)
    except litellm.ServiceUnavailableError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# hf_test_completion_tgi()


@pytest.mark.parametrize("provider", ["vertex_ai_beta"])  # "vertex_ai",
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
            response = completion(
                model="openai/my-vllm-model",
                messages=messages,
                response_format={"type": "json_object"},
                client=openai_client,
                api_base="my-custom-api-base",
                hello="world",
            )
        except Exception as e:
            pass

        mock_call.assert_called_once()

        print("Call KWARGS - {}".format(mock_call.call_args.kwargs))

        assert "hello" in mock_call.call_args.kwargs["extra_body"]


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


def mock_post(url, data=None, json=None, headers=None):
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


def test_hf_classifier_task():
    try:
        with patch("requests.post", side_effect=mock_post):
            litellm.set_verbose = True
            user_message = "I like you. I love you"
            messages = [{"content": user_message, "role": "user"}]
            response = completion(
                model="huggingface/text-classification/shahrukhx01/question-vs-statement-classifier",
                messages=messages,
            )
            print(f"response: {response}")
            assert isinstance(response, litellm.ModelResponse)
            assert isinstance(response.choices[0], litellm.Choices)
            assert response.choices[0].message.content is not None
            assert isinstance(response.choices[0].message.content, str)
    except Exception as e:
        pytest.fail(f"Error occurred: {str(e)}")


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
        mock_response.json.return_value = {
            # return the image in the response so that it can be tested
            # against the original
            "response": kwargs["json"]["images"]
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

    for test in tests:
        try:
            with patch("requests.post", side_effect=mock_post):
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


# def test_completion_cohere():  # commenting out,for now as the cohere endpoint is being flaky
#     try:
#         litellm.CohereConfig(max_tokens=10, stop_sequences=["a"])
#         response = completion(
#             model="command-nightly", messages=messages, logger_fn=logger_fn
#         )
#         # Add any assertions here to check the response
#         print(response)
#         response_str = response["choices"][0]["message"]["content"]
#         response_str_2 = response.choices[0].message.content
#         if type(response_str) != str:
#             pytest.fail(f"Error occurred: {e}")
#         if type(response_str_2) != str:
#             pytest.fail(f"Error occurred: {e}")
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")


# test_completion_cohere()


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


def test_completion_openai_organization():
    try:
        litellm.set_verbose = True
        try:
            response = completion(
                model="gpt-3.5-turbo", messages=messages, organization="org-ikDc4ex8NB"
            )
            pytest.fail("Request should have failed - This organization does not exist")
        except Exception as e:
            assert "No such organization: org-ikDc4ex8NB" in str(e)

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
            max_tokens=2,
            api_base="https://test-ollama-endpoint.onrender.com",
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


def test_completion_anyscale_with_functions():
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
        messages = [{"role": "user", "content": "What is the weather like in Boston?"}]
        response = completion(
            model="anyscale/mistralai/Mistral-7B-Instruct-v0.1",
            messages=messages,
            functions=function1,
        )
        # Add any assertions here to check the response
        print(response)

        cost = litellm.completion_cost(completion_response=response)
        print("cost to make anyscale completion=", cost)
        assert cost > 0.0
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_anyscale_with_functions()


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
            model="azure/chatgpt-v-2",
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


# test_completion_azure_key_completion_arg()


def test_azure_instruct():
    litellm.set_verbose = True
    response = completion(
        model="azure_text/instruct-model",
        messages=[{"role": "user", "content": "What is the weather like in Boston?"}],
        max_tokens=10,
    )
    print("response", response)


@pytest.mark.asyncio
async def test_azure_instruct_stream():
    litellm.set_verbose = False
    response = await litellm.acompletion(
        model="azure_text/instruct-model",
        messages=[{"role": "user", "content": "What is the weather like in Boston?"}],
        max_tokens=10,
        stream=True,
    )
    print("response", response)
    async for chunk in response:
        print(chunk)


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
                model="azure/chatgpt-v-2", messages=messages, client=client
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


def test_completion_azure():
    try:
        print("azure gpt-3.5 test\n\n")
        litellm.set_verbose = False
        ## Test azure call
        response = completion(
            model="azure/chatgpt-v-2",
            messages=messages,
            api_key="os.environ/AZURE_API_KEY",
        )
        print(f"response: {response}")
        ## Test azure flag for backwards-compat
        # response = completion(
        #     model="chatgpt-v-2",
        #     messages=messages,
        #     azure=True,
        #     max_tokens=10
        # )
        # Add any assertions here to check the response
        print(response)

        cost = completion_cost(completion_response=response)
        assert cost > 0.0
        print("Cost for azure completion request", cost)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_azure()


def test_azure_openai_ad_token():
    # this tests if the azure ad token is set in the request header
    # the request can fail since azure ad tokens expire after 30 mins, but the header MUST have the azure ad token
    # we use litellm.input_callbacks for this test
    def tester(
        kwargs,  # kwargs to completion
    ):
        print(kwargs["additional_args"])
        if kwargs["additional_args"]["headers"]["Authorization"] != "Bearer gm":
            pytest.fail("AZURE AD TOKEN Passed but not set in request header")
        return

    litellm.input_callback = [tester]
    try:
        response = litellm.completion(
            model="azure/chatgpt-v-2",  # e.g. gpt-35-instant
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
        pytest.fail(f"An exception occurs - {str(e)}")


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
            model="azure/chatgpt-v-2",
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
            model="azure/chatgpt-v-2",
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
            model="azure/chatgpt-v-2",
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


def test_completion_azure_deployment_id():
    try:
        litellm.set_verbose = True
        response = completion(
            deployment_id="chatgpt-v-2",
            model="gpt-3.5-turbo",
            messages=messages,
        )
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_azure_deployment_id()

import asyncio


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
            )
        else:
            response = await litellm.acompletion(
                model=model_name,
                messages=messages,
            )
            print(f"ASYNC REPLICATE RESPONSE - {response}")
        print(response)
        # Add any assertions here to check the response
        assert isinstance(response, litellm.ModelResponse)
        response_format_tests(response=response)
    except litellm.APIError as e:
        pass
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


def test_completion_together_ai_yi_chat():
    litellm.set_verbose = True
    model_name = "together_ai/zero-one-ai/Yi-34B-Chat"
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


@pytest.mark.skip(reason="AWS Suspended Account")
def test_completion_sagemaker():
    try:
        litellm.set_verbose = True
        print("testing sagemaker")
        response = completion(
            model="sagemaker/jumpstart-dft-hf-llm-mistral-7b-ins-20240329-150233",
            model_id="huggingface-llm-mistral-7b-instruct-20240329-150233",
            messages=messages,
            temperature=0.2,
            max_tokens=80,
            aws_region_name=os.getenv("AWS_REGION_NAME_2"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID_2"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY_2"),
            input_cost_per_second=0.000420,
        )
        # Add any assertions here to check the response
        print(response)
        cost = completion_cost(completion_response=response)
        print("calculated cost", cost)
        assert (
            cost > 0.0 and cost < 1.0
        )  # should never be > $1 for a single completion call
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_sagemaker()


@pytest.mark.skip(reason="AWS Suspended Account")
@pytest.mark.asyncio
async def test_acompletion_sagemaker():
    try:
        litellm.set_verbose = True
        print("testing sagemaker")
        response = await litellm.acompletion(
            model="sagemaker/jumpstart-dft-hf-llm-mistral-7b-ins-20240329-150233",
            model_id="huggingface-llm-mistral-7b-instruct-20240329-150233",
            messages=messages,
            temperature=0.2,
            max_tokens=80,
            aws_region_name=os.getenv("AWS_REGION_NAME_2"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID_2"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY_2"),
            input_cost_per_second=0.000420,
        )
        # Add any assertions here to check the response
        print(response)
        cost = completion_cost(completion_response=response)
        print("calculated cost", cost)
        assert (
            cost > 0.0 and cost < 1.0
        )  # should never be > $1 for a single completion call
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.skip(reason="AWS Suspended Account")
def test_completion_chat_sagemaker():
    try:
        messages = [{"role": "user", "content": "Hey, how's it going?"}]
        litellm.set_verbose = True
        response = completion(
            model="sagemaker/berri-benchmarking-Llama-2-70b-chat-hf-4",
            messages=messages,
            max_tokens=100,
            temperature=0.7,
            stream=True,
        )
        # Add any assertions here to check the response
        complete_response = ""
        for chunk in response:
            complete_response += chunk.choices[0].delta.content or ""
        print(f"complete_response: {complete_response}")
        assert len(complete_response) > 0
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_chat_sagemaker()


@pytest.mark.skip(reason="AWS Suspended Account")
def test_completion_chat_sagemaker_mistral():
    try:
        messages = [{"role": "user", "content": "Hey, how's it going?"}]

        response = completion(
            model="sagemaker/jumpstart-dft-hf-llm-mistral-7b-instruct",
            messages=messages,
            max_tokens=100,
        )
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"An error occurred: {str(e)}")


# test_completion_chat_sagemaker_mistral()


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


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize(
    "model",
    [
        "bedrock/cohere.command-r-plus-v1:0",
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "anthropic.claude-instant-v1",
        "bedrock/ai21.j2-mid",
        "mistral.mistral-7b-instruct-v0:2",
        "bedrock/amazon.titan-tg1-large",
        "meta.llama3-8b-instruct-v1:0",
        "cohere.command-text-v14",
    ],
)
@pytest.mark.asyncio
async def test_completion_bedrock_httpx_models(sync_mode, model):
    litellm.set_verbose = True

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


# test_completion_bedrock_cohere()


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

    converted_messages = anthropic_messages_pt(messages)

    print(f"converted_messages: {converted_messages}")

    ## ENSURE USER / ASSISTANT ALTERNATING
    for i, msg in enumerate(converted_messages):
        if i < len(converted_messages) - 1:
            assert msg["role"] != converted_messages[i + 1]["role"]


def test_completion_anyscale_api():
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
        response = completion(
            model="anyscale/meta-llama/Llama-2-7b-chat-hf",
            messages=messages,
        )
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_anyscale_api()


@pytest.mark.skip(reason="flaky test, times out frequently")
def test_completion_cohere():
    try:
        # litellm.set_verbose=True
        messages = [
            {"role": "system", "content": "You're a good bot"},
            {
                "role": "user",
                "content": "Hey",
            },
        ]
        response = completion(
            model="command-nightly",
            messages=messages,
        )
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# FYI - cohere_chat looks quite unstable, even when testing locally
def test_chat_completion_cohere():
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
            model="cohere_chat/command-r",
            messages=messages,
            max_tokens=10,
        )
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_chat_completion_cohere_stream():
    try:
        litellm.set_verbose = False
        messages = [
            {"role": "system", "content": "You're a good bot"},
            {
                "role": "user",
                "content": "Hey",
            },
        ]
        response = completion(
            model="cohere_chat/command-r",
            messages=messages,
            max_tokens=10,
            stream=True,
        )
        print(response)
        for chunk in response:
            print(chunk)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_azure_cloudflare_api():
    litellm.set_verbose = True
    try:
        messages = [
            {
                "role": "user",
                "content": "How do I output all files in a directory using Python?",
            },
        ]
        response = completion(
            model="azure/gpt-turbo",
            messages=messages,
            base_url=os.getenv("CLOUDFLARE_AZURE_BASE_URL"),
            api_key=os.getenv("AZURE_FRANCE_API_KEY"),
        )
        print(f"response: {response}")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
        traceback.print_exc()
        pass


# test_azure_cloudflare_api()


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
# def test_petals():
#     try:
#         response = completion(model="petals-team/StableBeluga2", messages=messages)
#         # Add any assertions here to check the response
#         print(response)

#         response = completion(model="petals-team/StableBeluga2", messages=messages)
#         # Add any assertions here to check the response
#         print(response)
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

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


#### Test A121 ###################
@pytest.mark.skip(reason="Local test")
def test_completion_ai21():
    print("running ai21 j2light test")
    litellm.set_verbose = True
    model_name = "j2-light"
    try:
        response = completion(
            model=model_name, messages=messages, max_tokens=100, temperature=0.8
        )
        # Add any assertions here to check the response
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


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


def test_completion_nvidia_nim():
    model_name = "nvidia_nim/databricks/dbrx-instruct"
    try:
        response = completion(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather like in Boston today in Fahrenheit?",
                }
            ],
        )
        # Add any assertions here to check the response
        print(response)
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0
    except litellm.exceptions.Timeout as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# Gemini tests
@pytest.mark.parametrize(
    "model",
    [
        # "gemini-1.0-pro",
        "gemini-1.5-pro",
        # "gemini-1.5-flash",
    ],
)
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
    except litellm.APIError as e:
        pass
    except Exception as e:
        if "InternalServerError" in str(e):
            pass
        else:
            pytest.fail(f"Error occurred: {e}")


# test_completion_gemini()


@pytest.mark.asyncio
async def test_acompletion_gemini():
    litellm.set_verbose = True
    model_name = "gemini/gemini-pro"
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
    messages = [{"role": "user", "content": "Hey, how's it going?"}]
    try:
        response = completion(model=model_name, messages=messages)
        # Add any assertions here to check the response
        print(response)
    except litellm.APIError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# Palm tests
def test_completion_palm():
    litellm.set_verbose = True
    model_name = "palm/chat-bison"
    messages = [{"role": "user", "content": "Hey, how's it going?"}]
    try:
        response = completion(model=model_name, messages=messages)
        # Add any assertions here to check the response
        print(response)
    except litellm.APIError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_palm()


# test palm with streaming
def test_completion_palm_stream():
    # litellm.set_verbose = True
    model_name = "palm/chat-bison"
    try:
        response = completion(
            model=model_name,
            messages=messages,
            stop=["stop"],
            stream=True,
            max_tokens=20,
        )
        # Add any assertions here to check the response
        for chunk in response:
            print(chunk)
    except litellm.APIError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.skip(reason="Account deleted by IBM.")
def test_completion_watsonx():
    litellm.set_verbose = True
    model_name = "watsonx/ibm/granite-13b-chat-v2"
    try:
        response = completion(
            model=model_name,
            messages=messages,
            stop=["stop"],
            max_tokens=20,
        )
        # Add any assertions here to check the response
        print(response)
    except litellm.APIError as e:
        pass
    except litellm.RateLimitError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


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
        ("azure", "chatgpt-v-2", None, None, "test-token"),
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


# Cloud flare AI tests
@pytest.mark.skip(reason="Flaky test-cloudflare is very unstable")
def test_completion_cloudflare():
    try:
        litellm.set_verbose = True
        response = completion(
            model="cloudflare/@cf/meta/llama-2-7b-chat-int8",
            messages=[{"content": "what llm are you", "role": "user"}],
            max_tokens=15,
            num_retries=3,
        )
        print(response)

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_cloudflare()


def test_moderation():
    response = litellm.moderation(input="i'm ishaan cto of litellm")
    print(response)
    output = response.results[0]
    print(output)
    return output


# test_moderation()
