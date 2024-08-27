# @pytest.mark.skip(reason="AWS Suspended Account")
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
from unittest.mock import AsyncMock, Mock, patch

import pytest

import litellm
from litellm import (
    ModelResponse,
    RateLimitError,
    Timeout,
    completion,
    completion_cost,
    embedding,
)
from litellm.llms.bedrock_httpx import BedrockLLM, ToolBlock
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.prompt_templates.factory import _bedrock_tools_pt

# litellm.num_retries = 3
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
            model="bedrock/anthropic.claude-instant-v1",
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


# test_completion_bedrock_claude_completion_auth()


@pytest.mark.parametrize("streaming", [True, False])
def test_completion_bedrock_guardrails(streaming):
    import os

    litellm.set_verbose = True
    import logging

    from litellm._logging import verbose_logger

    # verbose_logger.setLevel(logging.DEBUG)
    try:
        if streaming is False:
            response = completion(
                model="anthropic.claude-v2",
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

            response = completion(
                model="anthropic.claude-v2",
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


def test_completion_bedrock_claude_2_1_completion_auth():
    print("calling bedrock claude 2.1 completion params auth")
    import os

    aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    aws_region_name = os.environ["AWS_REGION_NAME"]

    os.environ.pop("AWS_ACCESS_KEY_ID", None)
    os.environ.pop("AWS_SECRET_ACCESS_KEY", None)
    os.environ.pop("AWS_REGION_NAME", None)
    try:
        response = completion(
            model="bedrock/anthropic.claude-v2:1",
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


# test_completion_bedrock_claude_2_1_completion_auth()


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
            model="bedrock/anthropic.claude-instant-v1",
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


# test_completion_bedrock_claude_external_client_auth()


@pytest.mark.skip(reason="Expired token, need to renew")
def test_completion_bedrock_claude_sts_client_auth():
    print("\ncalling bedrock claude external client auth")
    import os

    aws_access_key_id = os.environ["AWS_TEMP_ACCESS_KEY_ID"]
    aws_secret_access_key = os.environ["AWS_TEMP_SECRET_ACCESS_KEY"]
    aws_region_name = os.environ["AWS_REGION_NAME"]
    aws_role_name = os.environ["AWS_TEMP_ROLE_NAME"]

    try:
        import boto3

        litellm.set_verbose = True

        response = completion(
            model="bedrock/anthropic.claude-instant-v1",
            messages=messages,
            max_tokens=10,
            temperature=0.1,
            aws_region_name=aws_region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_role_name=aws_role_name,
            aws_session_name="my-test-session",
        )

        response = embedding(
            model="cohere.embed-multilingual-v3",
            input=["hello world"],
            aws_region_name="us-east-1",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_role_name=aws_role_name,
            aws_session_name="my-test-session",
        )

        response = completion(
            model="gpt-3.5-turbo",
            messages=messages,
            aws_region_name="us-east-1",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_role_name=aws_role_name,
            aws_session_name="my-test-session",
        )
        # Add any assertions here to check the response
        print(response)
    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.fixture()
def bedrock_session_token_creds():
    print("\ncalling oidc auto to get aws_session_token credentials")
    import os

    aws_region_name = os.environ["AWS_REGION_NAME"]
    aws_session_token = os.environ.get("AWS_SESSION_TOKEN")

    bllm = BedrockLLM()
    if aws_session_token is not None:
        # For local testing
        creds = bllm.get_credentials(
            aws_region_name=aws_region_name,
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            aws_session_token=aws_session_token,
        )
    else:
        # For circle-ci testing
        # aws_role_name = os.environ["AWS_TEMP_ROLE_NAME"]
        # TODO: This is using ai.moda's IAM role, we should use LiteLLM's IAM role eventually
        aws_role_name = (
            "arn:aws:iam::335785316107:role/litellm-github-unit-tests-circleci"
        )
        aws_web_identity_token = "oidc/circleci_v2/"

        creds = bllm.get_credentials(
            aws_region_name=aws_region_name,
            aws_web_identity_token=aws_web_identity_token,
            aws_role_name=aws_role_name,
            aws_session_name="my-test-session",
        )
    return creds


def process_stream_response(res, messages):
    import types

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


@pytest.mark.skipif(
    os.environ.get("CIRCLE_OIDC_TOKEN_V2") is None,
    reason="Cannot run without being in CircleCI Runner",
)
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


@pytest.mark.skipif(
    os.environ.get("CIRCLE_OIDC_TOKEN_V2") is None,
    reason="Cannot run without being in CircleCI Runner",
)
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


# test_completion_bedrock_claude_sts_client_auth()


@pytest.mark.skipif(
    os.environ.get("CIRCLE_OIDC_TOKEN_V2") is None,
    reason="Cannot run without being in CircleCI Runner",
)
def test_completion_bedrock_claude_sts_oidc_auth():
    print("\ncalling bedrock claude with oidc auth")
    import os

    aws_web_identity_token = "oidc/circleci_v2/"
    aws_region_name = os.environ["AWS_REGION_NAME"]
    # aws_role_name = os.environ["AWS_TEMP_ROLE_NAME"]
    # TODO: This is using ai.moda's IAM role, we should use LiteLLM's IAM role eventually
    aws_role_name = "arn:aws:iam::335785316107:role/litellm-github-unit-tests-circleci"

    try:
        litellm.set_verbose = True

        response_1 = completion(
            model="bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            messages=messages,
            max_tokens=10,
            temperature=0.1,
            aws_region_name=aws_region_name,
            aws_web_identity_token=aws_web_identity_token,
            aws_role_name=aws_role_name,
            aws_session_name="my-test-session",
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
            aws_web_identity_token=aws_web_identity_token,
            aws_role_name=aws_role_name,
            aws_session_name="my-test-session",
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
            aws_web_identity_token=aws_web_identity_token,
            aws_role_name=aws_role_name,
            aws_session_name="my-test-session",
        )
        print(response_3)
        assert len(response_3.choices) > 0
        assert len(response_3.choices[0].message.content) > 0

    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.skipif(
    os.environ.get("CIRCLE_OIDC_TOKEN_V2") is None,
    reason="Cannot run without being in CircleCI Runner",
)
def test_completion_bedrock_httpx_command_r_sts_oidc_auth():
    print("\ncalling bedrock httpx command r with oidc auth")
    import os

    aws_web_identity_token = "oidc/circleci_v2/"
    aws_region_name = "us-west-2"
    # aws_role_name = os.environ["AWS_TEMP_ROLE_NAME"]
    # TODO: This is using ai.moda's IAM role, we should use LiteLLM's IAM role eventually
    aws_role_name = "arn:aws:iam::335785316107:role/litellm-github-unit-tests-circleci"

    try:
        litellm.set_verbose = True

        response = completion(
            model="bedrock/cohere.command-r-v1:0",
            messages=messages,
            max_tokens=10,
            temperature=0.1,
            aws_region_name=aws_region_name,
            aws_web_identity_token=aws_web_identity_token,
            aws_role_name=aws_role_name,
            aws_session_name="my-test-session",
            aws_sts_endpoint="https://sts-fips.us-west-2.amazonaws.com",
            aws_bedrock_runtime_endpoint="https://bedrock-runtime-fips.us-west-2.amazonaws.com",
        )
        # Add any assertions here to check the response
        print(response)
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
        "meta.llama3-70b-instruct-v1:0",
        "anthropic.claude-v2",
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
    import os

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
        assert len(response.choices) > 0
        assert len(response.choices[0].message.content) > 0

        # os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
        # os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
        # os.environ["AWS_REGION_NAME"] = aws_region_name
    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# test_completion_bedrock_mistral_completion_auth()


def test_bedrock_ptu():
    """
    Check if a url with 'modelId' passed in, is created correctly

    Reference: https://github.com/BerriAI/litellm/issues/3805
    """
    client = HTTPHandler()

    with patch.object(client, "post", new=Mock()) as mock_client_post:
        litellm.set_verbose = True
        from openai.types.chat import ChatCompletion

        model_id = (
            "arn:aws:bedrock:us-west-2:888602223428:provisioned-model/8fxff74qyhs3"
        )
        try:
            response = litellm.completion(
                model="bedrock/anthropic.claude-instant-v1",
                messages=[{"role": "user", "content": "What's AWS?"}],
                model_id=model_id,
                client=client,
            )
        except Exception as e:
            pass

        assert "url" in mock_client_post.call_args.kwargs
        assert (
            mock_client_post.call_args.kwargs["url"]
            == "https://bedrock-runtime.us-west-2.amazonaws.com/model/arn%3Aaws%3Abedrock%3Aus-west-2%3A888602223428%3Aprovisioned-model%2F8fxff74qyhs3/converse"
        )
        mock_client_post.assert_called_once()


@pytest.mark.asyncio
async def test_bedrock_extra_headers():
    """
    Check if a url with 'modelId' passed in, is created correctly

    Reference: https://github.com/BerriAI/litellm/issues/3805, https://github.com/BerriAI/litellm/issues/5389#issuecomment-2313677977

    """
    client = AsyncHTTPHandler()

    with patch.object(client, "post", new=AsyncMock()) as mock_client_post:
        litellm.set_verbose = True
        from openai.types.chat import ChatCompletion

        try:
            response = await litellm.acompletion(
                model="anthropic.claude-3-sonnet-20240229-v1:0",
                messages=[{"role": "user", "content": "What's AWS?"}],
                client=client,
                extra_headers={"test": "hello world", "Authorization": "my-test-key"},
            )
        except Exception as e:
            pass

        print(f"mock_client_post.call_args: {mock_client_post.call_args}")
        assert "test" in mock_client_post.call_args.kwargs["headers"]
        assert mock_client_post.call_args.kwargs["headers"]["test"] == "hello world"
        assert (
            mock_client_post.call_args.kwargs["headers"]["Authorization"]
            == "my-test-key"
        )
        mock_client_post.assert_called_once()


@pytest.mark.asyncio
async def test_bedrock_custom_prompt_template():
    """
    Check if custom prompt template used for bedrock models

    Reference: https://github.com/BerriAI/litellm/issues/4415
    """
    client = AsyncHTTPHandler()

    with patch.object(client, "post", new=AsyncMock()) as mock_client_post:
        import json

        try:
            response = await litellm.acompletion(
                model="bedrock/mistral.OpenOrca",
                messages=[{"role": "user", "content": "What's AWS?"}],
                client=client,
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
                bos_token="<s>",
                eos_token="<|im_end|>",
            )
        except Exception as e:
            pass

        print(f"mock_client_post.call_args: {mock_client_post.call_args}")
        assert "prompt" in mock_client_post.call_args.kwargs["data"]

        prompt = json.loads(mock_client_post.call_args.kwargs["data"])["prompt"]
        assert prompt == "<|im_start|>user\nWhat's AWS?<|im_end|>"
        mock_client_post.assert_called_once()


def test_completion_bedrock_external_client_region():
    print("\ncalling bedrock claude external client auth")
    import os

    aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    aws_region_name = "us-east-1"

    os.environ.pop("AWS_ACCESS_KEY_ID", None)
    os.environ.pop("AWS_SECRET_ACCESS_KEY", None)

    client = HTTPHandler()

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
        with patch.object(client, "post", new=Mock()) as mock_client_post:
            try:
                response = completion(
                    model="bedrock/anthropic.claude-instant-v1",
                    messages=messages,
                    max_tokens=10,
                    temperature=0.1,
                    aws_bedrock_client=bedrock,
                    client=client,
                )
                # Add any assertions here to check the response
                print(response)
            except Exception as e:
                pass

            print(f"mock_client_post.call_args: {mock_client_post.call_args}")
            assert "us-east-1" in mock_client_post.call_args.kwargs["url"]

            mock_client_post.assert_called_once()

        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
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


def test_bedrock_tools_pt_valid_names():
    """
    # related issue: https://github.com/BerriAI/litellm/issues/5007
    # Bedrock tool names must satisfy regular expression pattern: [a-zA-Z][a-zA-Z0-9_]* ensure this is true

    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_restaurants",
                "description": "Search for restaurants",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cuisine": {"type": "string"},
                    },
                    "required": ["cuisine"],
                },
            },
        },
    ]

    result = _bedrock_tools_pt(tools)

    assert len(result) == 2
    assert result[0]["toolSpec"]["name"] == "get_current_weather"
    assert result[1]["toolSpec"]["name"] == "search_restaurants"


def test_bedrock_tools_pt_invalid_names():
    """
    # related issue: https://github.com/BerriAI/litellm/issues/5007
    # Bedrock tool names must satisfy regular expression pattern: [a-zA-Z][a-zA-Z0-9_]* ensure this is true

    """

    tools = [
        {
            "type": "function",
            "function": {
                "name": "123-invalid@name",
                "description": "Invalid name test",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "test": {"type": "string"},
                    },
                    "required": ["test"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "another@invalid#name",
                "description": "Another invalid name test",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "test": {"type": "string"},
                    },
                    "required": ["test"],
                },
            },
        },
    ]

    result = _bedrock_tools_pt(tools)

    print("bedrock tools after prompt formatting=", result)

    assert len(result) == 2
    assert result[0]["toolSpec"]["name"] == "a123_invalid_name"
    assert result[1]["toolSpec"]["name"] == "another_invalid_name"


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
