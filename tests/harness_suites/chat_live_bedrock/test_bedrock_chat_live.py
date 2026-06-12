"""
Tests Bedrock Completion + Rerank endpoints
"""

# @pytest.mark.skip(reason="AWS Suspended Account")
import os
import sys

from dotenv import load_dotenv

import litellm.types

load_dotenv()

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from unittest.mock import patch

import pytest

import litellm
from litellm import (
    ModelResponse,
    RateLimitError,
    ServiceUnavailableError,
    completion,
    completion_cost,
)
from base_llm_unit_tests import BaseLLMChatTest, BaseAnthropicChatTest

# litellm.num_retries = 3
litellm.cache = None
litellm.success_callback = []
user_message = "Write a short poem about the sky"
messages = [{"content": user_message, "role": "user"}]

def process_stream_response(res, messages):

    if isinstance(res, litellm.utils.CustomStreamWrapper):
        chunks = []
        for part in res:
            chunks.append(part)
            text = part.choices[0].delta.content or ""
            print(text, end="")
        res = litellm.stream_chunk_builder(chunks, messages=messages)
    else:
        raise ValueError("Response object is not a streaming response")

    return res


def encode_image(image_path):
    import base64

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def test_completion_bedrock_claude_completion_auth():
    print("calling bedrock claude completion params auth")
    import os

    aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    aws_region_name = os.environ["AWS_REGION_NAME"]

    os.environ.pop("AWS_ACCESS_KEY_ID", None)
    os.environ.pop("AWS_SECRET_ACCESS_KEY", None)
    os.environ.pop("AWS_REGION_NAME", None)

    try:
        response = completion(
            model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            messages=messages,
            max_tokens=10,
            temperature=0.1,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_region_name=aws_region_name,
        )
        # Add any assertions here to check the response
        print(response)

        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
        os.environ["AWS_REGION_NAME"] = aws_region_name
    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize("streaming", [True, False])
def test_completion_bedrock_guardrails(streaming):

    litellm.set_verbose = True


    # verbose_logger.setLevel(logging.DEBUG)
    try:
        if streaming is False:
            response = completion(
                model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                messages=[
                    {
                        "content": "where do i buy coffee from? ",
                        "role": "user",
                    }
                ],
                max_tokens=10,
                guardrailConfig={
                    "guardrailIdentifier": "ff6ujrregl1q",
                    "guardrailVersion": "DRAFT",
                    "trace": "enabled",
                },
            )
            # Add any assertions here to check the response
            print(response)
            assert (
                "Sorry, the model cannot answer this question. coffee guardrail applied"
                in response.choices[0].message.content
            )

            assert "trace" in response
            assert response.trace is not None

            print("TRACE=", response.trace)
        else:
            litellm.set_verbose = True
            response = completion(
                model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                messages=[
                    {
                        "content": "where do i buy coffee from? ",
                        "role": "user",
                    }
                ],
                stream=True,
                max_tokens=10,
                guardrailConfig={
                    "guardrailIdentifier": "ff6ujrregl1q",
                    "guardrailVersion": "DRAFT",
                    "trace": "enabled",
                },
            )

            saw_trace = False

            for chunk in response:
                if "trace" in chunk:
                    saw_trace = True
                print(chunk)

            assert (
                saw_trace is True
            ), "Did not see trace in response even when trace=enabled sent in the guardrailConfig"

    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_completion_bedrock_claude_external_client_auth():
    print("\ncalling bedrock claude external client auth")
    import os

    aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    aws_region_name = os.environ["AWS_REGION_NAME"]

    os.environ.pop("AWS_ACCESS_KEY_ID", None)
    os.environ.pop("AWS_SECRET_ACCESS_KEY", None)
    os.environ.pop("AWS_REGION_NAME", None)

    try:
        import boto3

        litellm.set_verbose = True

        bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name=aws_region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            endpoint_url=f"https://bedrock-runtime.{aws_region_name}.amazonaws.com",
        )

        response = completion(
            model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            messages=messages,
            max_tokens=10,
            temperature=0.1,
            aws_bedrock_client=bedrock,
        )
        # Add any assertions here to check the response
        print(response)

        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
        os.environ["AWS_REGION_NAME"] = aws_region_name
    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.skip(reason="Cannot run without being in CircleCI Runner")
def test_completion_bedrock_claude_aws_session_token(bedrock_session_token_creds):
    print("\ncalling bedrock claude with aws_session_token auth")

    import os

    aws_region_name = os.environ["AWS_REGION_NAME"]
    aws_access_key_id = bedrock_session_token_creds.access_key
    aws_secret_access_key = bedrock_session_token_creds.secret_key
    aws_session_token = bedrock_session_token_creds.token

    try:
        litellm.set_verbose = True

        response_1 = completion(
            model="bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            messages=messages,
            max_tokens=10,
            temperature=0.1,
            aws_region_name=aws_region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
        )
        print(response_1)
        assert len(response_1.choices) > 0
        assert len(response_1.choices[0].message.content) > 0

        # This second call is to verify that the cache isn't breaking anything
        response_2 = completion(
            model="bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            messages=messages,
            max_tokens=5,
            temperature=0.2,
            aws_region_name=aws_region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
        )
        print(response_2)
        assert len(response_2.choices) > 0
        assert len(response_2.choices[0].message.content) > 0

        # This third call is to verify that the cache isn't used for a different region
        response_3 = completion(
            model="bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            messages=messages,
            max_tokens=6,
            temperature=0.3,
            aws_region_name="us-east-1",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
        )
        print(response_3)
        assert len(response_3.choices) > 0
        assert len(response_3.choices[0].message.content) > 0

        # This fourth call is to verify streaming api works
        response_4 = completion(
            model="bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            messages=messages,
            max_tokens=6,
            temperature=0.3,
            aws_region_name="us-east-1",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            stream=True,
        )
        response_4 = process_stream_response(response_4, messages)
        print(response_4)
        assert len(response_4.choices) > 0
        assert len(response_4.choices[0].message.content) > 0

    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.skip(reason="Cannot run without being in CircleCI Runner")
def test_completion_bedrock_claude_aws_bedrock_client(bedrock_session_token_creds):
    print("\ncalling bedrock claude with aws_session_token auth")

    import os

    import boto3
    from botocore.client import Config

    aws_region_name = os.environ["AWS_REGION_NAME"]
    aws_access_key_id = bedrock_session_token_creds.access_key
    aws_secret_access_key = bedrock_session_token_creds.secret_key
    aws_session_token = bedrock_session_token_creds.token

    aws_bedrock_client_west = boto3.client(
        service_name="bedrock-runtime",
        region_name=aws_region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        config=Config(read_timeout=600),
    )

    try:
        litellm.set_verbose = True

        response_1 = completion(
            model="bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            messages=messages,
            max_tokens=10,
            temperature=0.1,
            aws_bedrock_client=aws_bedrock_client_west,
        )
        print(response_1)
        assert len(response_1.choices) > 0
        assert len(response_1.choices[0].message.content) > 0

        # This second call is to verify that the cache isn't breaking anything
        response_2 = completion(
            model="bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            messages=messages,
            max_tokens=5,
            temperature=0.2,
            aws_bedrock_client=aws_bedrock_client_west,
        )
        print(response_2)
        assert len(response_2.choices) > 0
        assert len(response_2.choices[0].message.content) > 0

        # This third call is to verify that the cache isn't used for a different region
        aws_bedrock_client_east = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            config=Config(read_timeout=600),
        )

        response_3 = completion(
            model="bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            messages=messages,
            max_tokens=6,
            temperature=0.3,
            aws_bedrock_client=aws_bedrock_client_east,
        )
        print(response_3)
        assert len(response_3.choices) > 0
        assert len(response_3.choices[0].message.content) > 0

        # This fourth call is to verify streaming api works
        response_4 = completion(
            model="bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            messages=messages,
            max_tokens=6,
            temperature=0.3,
            aws_bedrock_client=aws_bedrock_client_east,
            stream=True,
        )
        response_4 = process_stream_response(response_4, messages)
        print(response_4)
        assert len(response_4.choices) > 0
        assert len(response_4.choices[0].message.content) > 0

    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize(
    "image_url",
    [
        "data:image/png;base64,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",
        "https://avatars.githubusercontent.com/u/29436595?v=",
    ],
)
def test_bedrock_claude_3(image_url):
    try:
        litellm.set_verbose = True
        data = {
            "max_tokens": 100,
            "stream": False,
            "temperature": 0.3,
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hi"},
                {
                    "role": "user",
                    "content": [
                        {"text": "describe this image", "type": "text"},
                        {
                            "image_url": {
                                "detail": "high",
                                "url": image_url,
                            },
                            "type": "image_url",
                        },
                    ],
                },
            ],
        }
        response: ModelResponse = completion(
            model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            num_retries=3,
            **data,
        )  # type: ignore
        # Add any assertions here to check the response
        assert len(response.choices) > 0
        assert len(response.choices[0].message.content) > 0

    except litellm.InternalServerError:
        pass
    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize(
    "stop",
    [""],
)
@pytest.mark.parametrize(
    "model",
    [
        "anthropic.claude-3-sonnet-20240229-v1:0",
        # "meta.llama3-70b-instruct-v1:0",
        # "anthropic.claude-v2",
        # "mistral.mixtral-8x7b-instruct-v0:1",
    ],
)
def test_bedrock_stop_value(stop, model):
    try:
        litellm.set_verbose = True
        data = {
            "max_tokens": 100,
            "stream": False,
            "temperature": 0.3,
            "messages": [
                {"role": "user", "content": "hey, how's it going?"},
            ],
            "stop": stop,
        }
        response: ModelResponse = completion(
            model="bedrock/{}".format(model),
            **data,
        )  # type: ignore
        # Add any assertions here to check the response
        assert len(response.choices) > 0
        assert len(response.choices[0].message.content) > 0

    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize(
    "system",
    ["You are an AI", [{"type": "text", "text": "You are an AI"}], ""],
)
@pytest.mark.parametrize(
    "model",
    [
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "mistral.mixtral-8x7b-instruct-v0:1",
    ],
)
def test_bedrock_system_prompt(system, model):
    try:
        litellm.set_verbose = True
        data = {
            "max_tokens": 100,
            "stream": False,
            "temperature": 0.3,
            "messages": [
                {"role": "system", "content": system},
                {"role": "assistant", "content": "hey, how's it going?"},
            ],
            "user_continue_message": {"role": "user", "content": "Be a good bot!"},
        }
        response: ModelResponse = completion(
            model="bedrock/{}".format(model),
            **data,
        )  # type: ignore
        # Add any assertions here to check the response
        assert len(response.choices) > 0
        assert len(response.choices[0].message.content) > 0

    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_bedrock_claude_3_tool_calling():
    try:
        litellm.set_verbose = True
        litellm._turn_on_debug()
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
        messages = [
            {
                "role": "user",
                "content": "What's the weather like in Boston today in fahrenheit?",
            }
        ]
        response: ModelResponse = completion(
            model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )  # type: ignore
        print(f"response: {response}")
        # Add any assertions here to check the response
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
            model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        print(f"second response: {second_response}")
        assert isinstance(second_response.choices[0].message.content, str)
    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


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
            model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
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

        prompt_tokens = resp.usage.prompt_tokens
        raise Exception("it worked!")
    except Exception as e:
        if "500 Internal error encountered.'" in str(e):
            pass
        else:
            pytest.fail(f"An exception occurred - {str(e)}")


def test_completion_bedrock_mistral_completion_auth():
    print("calling bedrock mistral completion params auth")


    litellm._turn_on_debug()

    # aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
    # aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    # aws_region_name = os.environ["AWS_REGION_NAME"]
    # os.environ.pop("AWS_ACCESS_KEY_ID", None)
    # os.environ.pop("AWS_SECRET_ACCESS_KEY", None)
    # os.environ.pop("AWS_REGION_NAME", None)
    try:
        response: ModelResponse = completion(
            model="bedrock/mistral.mistral-7b-instruct-v0:2",
            messages=messages,
            max_tokens=10,
            temperature=0.1,
        )  # type: ignore
        # Add any assertions here to check the response
        print(f"response: {response}")
        assert len(response.choices) > 0
        assert len(response.choices[0].message.content) > 0

        # os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
        # os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
        # os.environ["AWS_REGION_NAME"] = aws_region_name
    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_bedrock_tool_calling():
    """
    # related issue: https://github.com/BerriAI/litellm/issues/5007
    # Bedrock tool names must satisfy regular expression pattern: [a-zA-Z][a-zA-Z0-9_]* ensure this is true
    """
    litellm.set_verbose = True
    response = litellm.completion(
        model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
        fallbacks=["bedrock/meta.llama3-1-8b-instruct-v1:0"],
        messages=[
            {
                "role": "user",
                "content": "What's the weather like in Boston today in Fahrenheit?",
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "-DoSomethingVeryCool-forLitellm_Testin999229291-0293993",
                    "description": "use this to get the current weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    print("bedrock response")
    print(response)

    # Assert that the tools in response have the same function name as the input
    _choice_1 = response.choices[0]
    if _choice_1.message.tool_calls is not None:
        print(_choice_1.message.tool_calls)
        for tool_call in _choice_1.message.tool_calls:
            _tool_Call_name = tool_call.function.name
            if _tool_Call_name is not None and "DoSomethingVeryCool" in _tool_Call_name:
                assert (
                    _tool_Call_name
                    == "-DoSomethingVeryCool-forLitellm_Testin999229291-0293993"
                )


def test_not_found_error():
    with pytest.raises(litellm.NotFoundError):
        completion(
            model="bedrock/bad_model",
            messages=[
                {
                    "role": "user",
                    "content": "What is the meaning of life",
                }
            ],
        )


@pytest.mark.parametrize(
    "model",
    [
        "bedrock/us.anthropic.claude-3-haiku-20240307-v1:0",
        "bedrock/us.meta.llama3-2-11b-instruct-v1:0",
    ],
)
def test_bedrock_cross_region_inference(model):
    litellm.set_verbose = True
    response = completion(
        model=model,
        messages=messages,
        max_tokens=10,
        temperature=0.1,
    )


def test_base_aws_llm_get_credentials():
    import time

    import boto3

    from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

    start_time = time.time()
    session = boto3.Session(
        aws_access_key_id="test",
        aws_secret_access_key="test2",
        region_name="test3",
    )
    credentials = session.get_credentials().get_frozen_credentials()
    end_time = time.time()

    print(
        "Total time for credentials - {}. Credentials - {}".format(
            end_time - start_time, credentials
        )
    )

    start_time = time.time()
    credentials = BaseAWSLLM().get_credentials(
        aws_access_key_id="test",
        aws_secret_access_key="test2",
        aws_region_name="test3",
    )

    end_time = time.time()

    print(
        "Total time for credentials - {}. Credentials - {}".format(
            end_time - start_time, credentials.get_frozen_credentials()
        )
    )


def test_bedrock_converse_route():
    litellm.set_verbose = True
    try:
        litellm.completion(
            model="bedrock/converse/us.amazon.nova-pro-v1:0",
            messages=[{"role": "user", "content": "Hello, world!"}],
        )
    except ServiceUnavailableError as e:
        if "Too many requests" in str(e):
            pytest.skip("Skipping test due to AWS Bedrock rate limiting")
        else:
            raise


def test_bedrock_mapped_converse_models():
    litellm.set_verbose = True
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.add_known_models()
    litellm.completion(
        model="bedrock/us.amazon.nova-pro-v1:0",
        messages=[{"role": "user", "content": "Hello, world!"}],
    )


class TestBedrockConverseChatCrossRegion(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")
        litellm.add_known_models()
        return {
            "model": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        }

    def test_prompt_caching(self):
        """
        Remove override once we have access to Bedrock prompt caching
        """
        pass

    def test_completion_cost(self):
        """
        Test if region models info is correctly used for cost calculation. Using the base model info for cost calculation.
        """
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")
        bedrock_model = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        litellm.model_cost.pop(bedrock_model, None)
        model = f"bedrock/{bedrock_model}"

        litellm.set_verbose = True
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )
        cost = completion_cost(response)

        assert cost > 0


class TestBedrockConverseAnthropicUnitTests(BaseAnthropicChatTest):
    def get_base_completion_call_args(self) -> dict:
        return {
            "model": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        }

    def get_base_completion_call_args_with_thinking(self) -> dict:
        return {
            "model": "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "thinking": {"type": "enabled", "budget_tokens": 16000},
        }


class TestBedrockConverseChatNormal(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")
        litellm.add_known_models()
        return {
            "model": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "aws_region_name": "us-east-1",
        }


class TestBedrockConverseNovaTestSuite(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")
        litellm.add_known_models()
        return {
            "model": "bedrock/us.amazon.nova-lite-v1:0",
            "aws_region_name": "us-east-1",
        }

    def test_prompt_caching(self):
        """
        TODO: Ensure this test passes our base llm test suite
        """


def test_bedrock_empty_content_real_call():
    completion(
        model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "",
                    },
                    {"type": "text", "text": "Hey, how's it going?"},
                ],
            }
        ],
    )


@pytest.mark.skip(
    reason="Skipping test due to bedrock changing their response schema support. Come back to this."
)
def test_nova_optional_params_tool_choice():
    try:
        litellm.drop_params = True
        litellm.set_verbose = True
        litellm.completion(
            messages=[
                {"role": "user", "content": "A WWII competitive game for 4-8 players"}
            ],
            model="bedrock/us.amazon.nova-pro-v1:0",
            temperature=0.3,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "GameDefinition",
                        "description": "Correctly extracted `GameDefinition` with all the required parameters with correct types",
                        "parameters": {
                            "$defs": {
                                "TurnDurationEnum": {
                                    "enum": [
                                        "action",
                                        "encounter",
                                        "battle",
                                        "operation",
                                    ],
                                    "title": "TurnDurationEnum",
                                    "type": "string",
                                }
                            },
                            "properties": {
                                "id": {
                                    "anyOf": [{"type": "integer"}, {"type": "null"}],
                                    "default": None,
                                    "title": "Id",
                                },
                                "prompt": {"title": "Prompt", "type": "string"},
                                "name": {"title": "Name", "type": "string"},
                                "description": {
                                    "title": "Description",
                                    "type": "string",
                                },
                                "competitve": {
                                    "title": "Competitve",
                                    "type": "boolean",
                                },
                                "players_min": {
                                    "title": "Players Min",
                                    "type": "integer",
                                },
                                "players_max": {
                                    "title": "Players Max",
                                    "type": "integer",
                                },
                                "turn_duration": {
                                    "$ref": "#/$defs/TurnDurationEnum",
                                    "description": "how long the passing of a turn should represent for a game at this scale",
                                },
                            },
                            "required": [
                                "competitve",
                                "description",
                                "name",
                                "players_max",
                                "players_min",
                                "prompt",
                                "turn_duration",
                            ],
                            "type": "object",
                        },
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": "GameDefinition"}},
        )
    except litellm.APIConnectionError:
        pass


@pytest.mark.parametrize(
    "image_url",
    [
        "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        # "https://raw.githubusercontent.com/datasets/gdp/master/data/gdp.csv",
        "https://www.cmu.edu/blackboard/files/evaluate/tests-example.xls",
        # "https://raw.githubusercontent.com/datasets/sample-data/master/README.txt", # invalid url
        "https://raw.githubusercontent.com/mdn/content/main/README.md",
    ],
)
@pytest.mark.asyncio
async def test_bedrock_document_understanding(image_url):
    from litellm import acompletion

    litellm._turn_on_debug()
    model = "bedrock/us.amazon.nova-pro-v1:0"

    image_content = [
        {"type": "text", "text": f"What's this file about?"},
        {
            "type": "image_url",
            "image_url": image_url,
        },
    ]

    try:
        response = await acompletion(
            model=model,
            messages=[{"role": "user", "content": image_content}],
        )
        assert response is not None
        assert response.choices[0].message.content != ""
    except litellm.ServiceUnavailableError as e:
        pytest.skip("Skipping test due to ServiceUnavailableError")


@pytest.mark.asyncio
async def test_bedrock_stream_thinking_content_openwebui():
    """
    When merge_reasoning_content_in_choices=True

    The content should be collected as

    ```
    <think>
    I am a helpful assistant, the user wants to know who I am
    </think>

    Hi I am Anthropic, I am a helpful assistant

    ```
    """
    response = await litellm.acompletion(
        model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        messages=[{"role": "user", "content": "Hello who is this?"}],
        stream=True,
        max_tokens=1080,
        thinking={"type": "enabled", "budget_tokens": 1024},
        merge_reasoning_content_in_choices=True,
    )
    content = ""
    async for chunk in response:
        content += chunk.choices[0].delta.content or ""

        # OpenWebUI expects the reasoning_content to be removed, otherwise this will appear as duplicate thinking blocks
        assert getattr(chunk.choices[0].delta, "reasoning_content", None) is None
        print(chunk)

    print("collected content", content)

    # Assert that the content follows the expected format with exactly one thinking section
    think_open_pos = content.find("<think>")
    think_close_pos = content.find("</think>")

    # Assert there's exactly one opening and closing tag
    assert think_open_pos >= 0, "Opening <think> tag not found"
    assert think_close_pos > 0, "Closing </think> tag not found"
    assert (
        content.count("<think>") == 1
    ), "There should be exactly one opening <think> tag"
    assert (
        content.count("</think>") == 1
    ), "There should be exactly one closing </think> tag"

    # Assert the opening tag comes before the closing tag
    assert (
        think_open_pos < think_close_pos
    ), "Opening tag should come before closing tag"

    # Assert there's content between the tags
    thinking_content = content[think_open_pos + 7 : think_close_pos]
    assert (
        len(thinking_content.strip()) > 0
    ), "There should be content between thinking tags"

    # Assert there's content after the closing tag
    assert (
        len(content) > think_close_pos + 8
    ), "There should be content after the thinking tags"
    response_content = content[think_close_pos + 8 :].strip()
    assert (
        len(response_content) > 0
    ), "There should be non-empty content after thinking tags"


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_bedrock_passthrough(sync_mode: bool):
    import litellm

    litellm._turn_on_debug()

    data = {
        "max_tokens": 512,
        "messages": [{"role": "user", "content": "Hey"}],
        "system": [
            {
                "type": "text",
                "text": "Analyze if this message indicates a new conversation topic. If it does, extract a 2-3 word title that captures the new topic. Format your response as a JSON object with two fields: 'isNewTopic' (boolean) and 'title' (string, or null if isNewTopic is false). Only include these fields, no other text.",
            }
        ],
        "temperature": 0,
        "metadata": {
            "user_id": "5dd07c33da27e6d2968d94ea20bf47a7b090b6b158b82328d54da2909a108e84"
        },
        "anthropic_version": "bedrock-2023-05-31",
        "anthropic_beta": ["claude-code-20250219"],
    }

    if sync_mode:
        response = litellm.llm_passthrough_route(
            model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            method="POST",
            endpoint="/model/us.anthropic.claude-haiku-4-5-20251001-v1:0/invoke",
            data=data,
        )
    else:
        response = await litellm.allm_passthrough_route(
            model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            method="POST",
            endpoint="/model/us.anthropic.claude-haiku-4-5-20251001-v1:0/invoke",
            data=data,
        )

    print(response.text)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_bedrock_passthrough_router():
    """
    Test bedrock passthrough using litellm.Router with async mode.
    Tests that the router:
    1. Resolves the router model name to the actual deployment
    2. Replaces the router model name in the endpoint with the actual deployment model
    """
    import litellm
    from litellm import Router

    litellm._turn_on_debug()

    router = Router(
        model_list=[
            {
                "model_name": "special-bedrock-model",
                "litellm_params": {
                    "model": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                },
            }
        ]
    )

    data = {
        "max_tokens": 512,
        "messages": [{"role": "user", "content": "Hey"}],
        "system": [
            {
                "type": "text",
                "text": "Analyze if this message indicates a new conversation topic. If it does, extract a 2-3 word title that captures the new topic. Format your response as a JSON object with two fields: 'isNewTopic' (boolean) and 'title' (string, or null if isNewTopic is false). Only include these fields, no other text.",
            }
        ],
        "temperature": 0,
        "metadata": {
            "user_id": "5dd07c33da27e6d2968d94ea20bf47a7b090b6b158b82328d54da2909a108e84"
        },
        "anthropic_version": "bedrock-2023-05-31",
        "anthropic_beta": ["claude-code-20250219"],
    }

    # Endpoint uses the router model name which should be replaced with actual deployment
    response = await router.allm_passthrough_route(
        model="special-bedrock-model",
        method="POST",
        endpoint="/model/special-bedrock-model/invoke",
        data=data,
    )

    print(response.text)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_bedrock_converse__streaming_passthrough(monkeypatch):
    import litellm
    from litellm.integrations.custom_logger import CustomLogger
    import asyncio

    if os.environ.get("LITELLM_RUN_LIVE_BEDROCK_PASSTHROUGH_TESTS") != "1":
        pytest.skip("Live Bedrock passthrough E2E tests are opt-in")
    if os.environ.get("CASSETTE_REDIS_URL"):
        pytest.skip("Live Bedrock passthrough E2E tests cannot run under VCR replay")

    class MockCustomLogger(CustomLogger):
        pass

    mock_custom_logger = MockCustomLogger()
    monkeypatch.setattr(litellm, "callbacks", [mock_custom_logger])

    litellm._turn_on_debug()

    data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": "Write an article about impact of high inflation to GDP of a country"
                    }
                ],
            }
        ],
        "system": [{"text": "You are an economist with access to lots of data"}],
        "inferenceConfig": {"maxTokens": 100, "temperature": 0.5},
    }
    with patch.object(mock_custom_logger, "async_log_success_event") as mock_callback:
        response = await litellm.allm_passthrough_route(
            model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            method="POST",
            endpoint="/model/us.anthropic.claude-haiku-4-5-20251001-v1:0/converse-stream",
            data=data,
        )
        async for chunk in response:
            print(chunk)

        await asyncio.sleep(1)

        mock_callback.assert_called_once()
        print(mock_callback.call_args.kwargs.keys())
        assert "response_cost" in mock_callback.call_args.kwargs["kwargs"]
        response_cost = mock_callback.call_args.kwargs["kwargs"]["response_cost"]
        assert response_cost is not None and response_cost > 0
        assert "standard_logging_object" in mock_callback.call_args.kwargs["kwargs"]


@pytest.mark.asyncio
async def test_bedrock_streaming_passthrough_test2(monkeypatch):
    import litellm
    import asyncio
    from litellm.integrations.custom_logger import CustomLogger

    class MockCustomLogger(CustomLogger):
        pass

    mock_custom_logger = MockCustomLogger()
    monkeypatch.setattr(litellm, "callbacks", [mock_custom_logger])

    litellm._turn_on_debug()

    data = {
        "max_tokens": 512,
        "messages": [{"role": "user", "content": "Hey"}],
        "system": [
            {
                "type": "text",
                "text": "Analyze if this message indicates a new conversation topic. If it does, extract a 2-3 word title that captures the new topic. Format your response as a JSON object with two fields: 'isNewTopic' (boolean) and 'title' (string, or null if isNewTopic is false). Only include these fields, no other text.",
            }
        ],
        "temperature": 0,
        "metadata": {
            "user_id": "5dd07c33da27e6d2968d94ea20bf47a7b090b6b158b82328d54da2909a108e84"
        },
        "anthropic_version": "bedrock-2023-05-31",
        "anthropic_beta": ["claude-code-20250219"],
    }

    with patch.object(mock_custom_logger, "async_log_success_event") as mock_callback:
        response = await litellm.allm_passthrough_route(
            model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            method="POST",
            endpoint="/model/us.anthropic.claude-haiku-4-5-20251001-v1:0/invoke-with-response-stream",
            data=data,
        )
        async for chunk in response:
            print(chunk)

        await asyncio.sleep(5)

        mock_callback.assert_called_once()
        # check standard logging payload created
        print(mock_callback.call_args.kwargs.keys())
        assert "standard_logging_object" in mock_callback.call_args.kwargs["kwargs"]
        assert "response_cost" in mock_callback.call_args.kwargs["kwargs"]


@pytest.mark.asyncio
async def test_bedrock_streaming_passthrough_test1(monkeypatch):
    import litellm
    import asyncio
    from litellm.integrations.custom_logger import CustomLogger

    class MockCustomLogger(CustomLogger):
        pass

    mock_custom_logger = MockCustomLogger()
    monkeypatch.setattr(litellm, "callbacks", [mock_custom_logger])

    litellm._turn_on_debug()

    data = {
        "max_tokens": 512,
        "messages": [{"role": "user", "content": "Hey"}],
        "system": [
            {
                "type": "text",
                "text": "Analyze if this message indicates a new conversation topic. If it does, extract a 2-3 word title that captures the new topic. Format your response as a JSON object with two fields: 'isNewTopic' (boolean) and 'title' (string, or null if isNewTopic is false). Only include these fields, no other text.",
            }
        ],
        "temperature": 0,
        "metadata": {
            "user_id": "5dd07c33da27e6d2968d94ea20bf47a7b090b6b158b82328d54da2909a108e84"
        },
        "anthropic_version": "bedrock-2023-05-31",
        "anthropic_beta": ["claude-code-20250219"],
    }

    with patch.object(mock_custom_logger, "async_log_success_event") as mock_callback:
        response = await litellm.allm_passthrough_route(
            model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            method="POST",
            endpoint="/model/us.anthropic.claude-haiku-4-5-20251001-v1:0/invoke-with-response-stream",
            data=data,
        )
        async for chunk in response:
            print(chunk)

        await asyncio.sleep(5)

        mock_callback.assert_called_once()
        # check standard logging payload created
        print(mock_callback.call_args.kwargs.keys())
        assert "standard_logging_object" in mock_callback.call_args.kwargs["kwargs"]
        assert "response_cost" in mock_callback.call_args.kwargs["kwargs"]
