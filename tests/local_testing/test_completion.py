import json
import os

from dotenv import load_dotenv

load_dotenv()
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import OpenAI

import litellm
from litellm import Timeout, completion, completion_cost
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from tests.fake_openai_endpoint import FAKE_OPENAI_API_BASE

# litellm.num_retries=3

litellm.cache = None
litellm.success_callback = []
user_message = "Write a short poem about the sky"
messages = [{"content": user_message, "role": "user"}]




@pytest.fixture(autouse=True)
def reset_callbacks():
    print("\npytest fixture - resetting callbacks")
    litellm.success_callback = []
    litellm._async_success_callback = []
    litellm.failure_callback = []
    litellm.callbacks = []








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


# test_completion_predibase()


# test_completion_claude()




@pytest.mark.asyncio
async def test_anthropic_no_content_error():
    """
    https://github.com/BerriAI/litellm/discussions/3440#discussioncomment-9323402
    """
    try:
        litellm.drop_params = True
        response = await litellm.acompletion(
            model="anthropic/claude-sonnet-4-5-20250929",
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
        if e.status_code != 500:
            raise
    except Exception as e:
        pytest.fail(f"An unexpected error occurred - {str(e)}")




def encode_image(image_path):
    import base64

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@pytest.mark.parametrize(
    "model",
    [
        "gpt-4o",
        "azure/gpt-4.1-mini",
        "anthropic/claude-sonnet-4-5-20250929",
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
    """
    Mocked so it does not depend on Fireworks' rotating serverless catalog
    (no externally-verifiable model list exists). Asserts the request is
    built correctly and the OpenAI-compatible response is parsed back.
    """
    litellm.set_verbose = True
    messages = [
        {"role": "system", "content": "You're a good bot"},
        {"role": "user", "content": "Hey"},
    ]

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "accounts/fireworks/models/deepseek-v3p1",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello there!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }
    mock_response.text = json.dumps(mock_response.json.return_value)

    client = HTTPHandler()
    with patch.object(client, "post", return_value=mock_response) as mock_post:
        response = completion(
            model="fireworks_ai/accounts/fireworks/models/deepseek-v3p1",
            messages=messages,
            client=client,
        )

    mock_post.assert_called_once()
    request_body = json.loads(mock_post.call_args.kwargs["data"])
    assert "deepseek-v3p1" in request_body["model"]
    assert request_body["messages"] == messages
    assert response.choices[0].message.content == "Hello there!"
    assert response.usage.total_tokens == 12


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




# test_completion_perplexity_api()




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






def test_lm_studio_completion(monkeypatch):
    monkeypatch.delenv("LM_STUDIO_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    litellm.turn_on_debug()
    try:
        completion(
            api_key="fake-key",
            model="lm_studio/typhoon2-quen2.5-7b-instruct",
            messages=[
                {"role": "user", "content": "What's the weather like in San Francisco?"}
            ],
            api_base=FAKE_OPENAI_API_BASE,
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
        ("bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0", None),
        # ("azure/gpt-4o-new-test", "2024-08-01-preview"),
    ],
)
@pytest.mark.flaky(retries=3, delay=1)
def test_completion_openai_pydantic(model, api_version):
    try:
        litellm.turn_on_debug()
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


def test_completion_text_openai():
    try:
        # litellm.set_verbose =True
        response = completion(model="text-completion-openai/gpt-5.4-nano", messages=messages)
        print(response["choices"][0]["message"]["content"])
    except Exception as e:
        print(e)
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio
async def test_completion_text_openai_async():
    try:
        # litellm.set_verbose =True
        response = await litellm.acompletion(
            model="text-completion-openai/gpt-5.4-nano", messages=messages
        )
        print(response["choices"][0]["message"]["content"])
    except Exception as e:
        print(e)
        pytest.fail(f"Error occurred: {e}")


def test_completion_openai_with_optional_params():
    # [Proxy PROD TEST] WARNING: DO NOT DELETE THIS TEST
    on_request = MagicMock()
    client = OpenAI(http_client=httpx.Client(event_hooks={"request": [on_request]}))
    response = completion(
        model="gpt-6-luna",
        reasoning_effort="none",
        messages=[{"role": "user", "content": "respond in valid, json - what is the day"}],
        temperature=0.5,
        top_p=0.1,
        seed=12,
        response_format={"type": "json_object"},
        logit_bias=None,
        user="ishaans app",
        client=client,
    )

    assert response.choices[0].message.content
    on_request.assert_called_once()
    sent = json.loads(on_request.call_args.args[0].content)
    assert sent["model"] == "gpt-6-luna"
    assert sent["user"] == "ishaans app"
    assert sent["seed"] == 12
    assert sent["temperature"] == 0.5
    assert sent["top_p"] == 0.1
    assert sent["response_format"] == {"type": "json_object"}
    assert "logit_bias" not in sent


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
            if len(chunk.choices) == 0:
                continue
            print(f"chunk.choices[0]: {chunk.choices[0]}")
            if (
                "logprobs" in chunk.choices[0]
                and chunk.choices[0].logprobs is not None
                and len(chunk.choices[0].logprobs.content) > 0
            ):
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




# test_completion_ollama_hosted()










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




# test_completion_hf_model_no_provider()






# test_completion_anyscale_with_functions()


def test_completion_azure_extra_headers():
    # this tests if we can pass api_key to completion, when it's not in the env.
    # DO NOT REMOVE THIS TEST. No MATTER WHAT Happens!
    # If you want to remove it, speak to Ishaan!
    # Ishaan will be very disappointed if this test is removed -> this is a standard way to pass api_key + the router + proxy use this
    from httpx import Client


    http_client = Client()

    with patch.object(http_client, "send", new=MagicMock()) as mock_client:
        litellm.client_session = http_client
        try:
            response = completion(
                model="azure/gpt-4.1-mini",
                messages=messages,
                api_base=os.getenv("AZURE_AI_API_BASE"),
                api_version="2023-07-01-preview",
                api_key=os.getenv("AZURE_AI_API_KEY"),
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

    old_key = os.environ["AZURE_AI_API_KEY"]
    os.environ.pop("AZURE_AI_API_KEY", None)

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
            os.environ["AZURE_AI_API_KEY"] = old_key

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
    old_key = os.environ["AZURE_AI_API_KEY"]
    os.environ.pop("AZURE_AI_API_KEY", None)
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
        os.environ["AZURE_AI_API_KEY"] = old_key
    except Exception as e:
        os.environ["AZURE_AI_API_KEY"] = old_key
        pytest.fail(f"Error occurred: {e}")


async def test_re_use_azure_async_client():
    try:
        print("azure gpt-3.5 ASYNC with clie nttest\n\n")
        litellm.set_verbose = True
        import openai

        client = openai.AsyncAzureOpenAI(
            azure_endpoint=os.environ["AZURE_AI_API_BASE"],
            api_key=os.environ["AZURE_AI_API_KEY"],
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




# test_azure_openai_ad_token()


def test_completion_azure2():
    # test if we can pass api_base, api_version and api_key in compleition()
    try:
        print("azure gpt-3.5 test\n\n")
        litellm.set_verbose = False
        api_base = os.environ["AZURE_AI_API_BASE"]
        api_key = os.environ["AZURE_AI_API_KEY"]
        api_version = os.environ["AZURE_API_VERSION"]

        os.environ["AZURE_AI_API_BASE"] = ""
        os.environ["AZURE_API_VERSION"] = ""
        os.environ["AZURE_AI_API_KEY"] = ""

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

        os.environ["AZURE_AI_API_BASE"] = api_base
        os.environ["AZURE_API_VERSION"] = api_version
        os.environ["AZURE_AI_API_KEY"] = api_key

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_azure2()


def test_completion_azure3():
    # test if we can pass api_base, api_version and api_key in compleition()
    try:
        print("azure gpt-3.5 test\n\n")
        litellm.set_verbose = True
        litellm.api_base = os.environ["AZURE_AI_API_BASE"]
        litellm.api_key = os.environ["AZURE_AI_API_KEY"]
        litellm.api_version = os.environ["AZURE_API_VERSION"]

        os.environ["AZURE_AI_API_BASE"] = ""
        os.environ["AZURE_API_VERSION"] = ""
        os.environ["AZURE_AI_API_KEY"] = ""

        ## Test azure call
        response = completion(
            model="azure/gpt-4.1-mini",
            messages=messages,
            max_tokens=10,
        )

        # Add any assertions here to check the response
        print(response)

        os.environ["AZURE_AI_API_BASE"] = litellm.api_base
        os.environ["AZURE_API_VERSION"] = litellm.api_version
        os.environ["AZURE_AI_API_KEY"] = litellm.api_key

    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_azure3()


# new azure test for using litellm. vars,
# use the following vars in this test and make an azure_api_call
#  litellm.api_type = self.azure_api_type
#  litellm.api_base = self.AZURE_AI_API_BASE
#  litellm.api_version = self.azure_api_version
#  litellm.api_key = self.api_key
def test_completion_azure_with_litellm_key():
    try:
        print("azure gpt-3.5 test\n\n")
        import openai

        #### set litellm vars
        litellm.api_type = "azure"
        litellm.api_base = os.environ["AZURE_AI_API_BASE"]
        litellm.api_version = os.environ["AZURE_API_VERSION"]
        litellm.api_key = os.environ["AZURE_AI_API_KEY"]

        ######### UNSET ENV VARs for this ################
        os.environ["AZURE_AI_API_BASE"] = ""
        os.environ["AZURE_API_VERSION"] = ""
        os.environ["AZURE_AI_API_KEY"] = ""

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
        os.environ["AZURE_AI_API_BASE"] = litellm.api_base
        os.environ["AZURE_API_VERSION"] = litellm.api_version
        os.environ["AZURE_AI_API_KEY"] = litellm.api_key

        ######### UNSET litellm vars
        litellm.api_type = None
        litellm.api_base = None
        litellm.api_version = None
        litellm.api_key = None

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


# test_completion_together_ai_mixtral()


def test_completion_together_ai_llama():
    litellm.set_verbose = True
    model_name = "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"
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
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "mistral.mistral-7b-instruct-v0:2",
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


# test_completion_deep_infra()




# test_completion_deep_infra_mistral()




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


# Deepseek tests










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


def test_moderation():
    response = litellm.moderation(input="i'm ishaan cto of litellm")
    print(response)
    output = response.results[0]
    print(output)
    return output




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






def test_deepseek_reasoning_content_completion():
    try:
        litellm.set_verbose = True
        litellm.turn_on_debug()
        resp = litellm.completion(
            timeout=5,
            model="deepseek/deepseek-reasoner",
            messages=[{"role": "user", "content": "Tell me a joke."}],
        )

        assert resp.choices[0].message.reasoning_content is not None
    except litellm.Timeout:
        pytest.skip("Model is timing out")


def test_qwen_text_completion():
    # litellm.turn_on_debug()
    resp = litellm.completion(
        model="text-completion-openai/gpt-5.4-nano",
        messages=[{"content": "hello", "role": "user"}],
        stream=False,
        logprobs=1,
    )
    assert resp.choices[0].message.content is not None
    assert resp.choices[0].logprobs.token_logprobs[0] is not None
    print(
        f"resp.choices[0].logprobs.token_logprobs[0]: {resp.choices[0].logprobs.token_logprobs[0]}"
    )




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
    litellm.turn_on_debug()
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
