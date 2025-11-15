"""
Tests Bedrock Completion + Rerank endpoints
"""

# @pytest.mark.skip(reason="AWS Suspended Account")
import os
import sys
import traceback

from dotenv import load_dotenv

import litellm.types

load_dotenv()
import io
import os
import json

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from unittest.mock import AsyncMock, Mock, patch

import pytest

import litellm
from litellm import (
    ModelResponse,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
    completion,
    completion_cost,
    embedding,
)
from litellm.llms.bedrock.chat import BedrockLLM
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.litellm_core_utils.prompt_templates.factory import _bedrock_tools_pt
from base_llm_unit_tests import BaseLLMChatTest, BaseAnthropicChatTest
from base_rerank_unit_tests import BaseLLMRerankTest
from base_embedding_unit_tests import BaseLLMEmbeddingTest

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
            model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
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
                model="anthropic.claude-3-5-sonnet-20240620-v1:0",
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
                model="anthropic.claude-3-5-sonnet-20240620-v1:0",
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
            model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
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
            model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
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
            aws_session_name="cross-region-test",
            aws_sts_endpoint="https://sts-fips.us-east-2.amazonaws.com",
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
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAL0AAAC9CAMAAADRCYwCAAAAh1BMVEX///8AAAD8/Pz5+fkEBAT39/cJCQn09PRNTU3y8vIMDAwzMzPe3t7v7+8QEBCOjo7FxcXR0dHn5+elpaWGhoYYGBivr686OjocHBy0tLQtLS1TU1PY2Ni6urpaWlpERER3d3ecnJxoaGiUlJRiYmIlJSU4ODhBQUFycnKAgIDBwcFnZ2chISE7EjuwAAAI/UlEQVR4nO1caXfiOgz1bhJIyAJhX1JoSzv8/9/3LNlpYd4rhX6o4/N8Z2lKM2cURZau5JsQEhERERERERERERERERERERHx/wBjhDPC3OGN8+Cc5JeMuheaETSdO8vZFyCScHtmz2CsktoeMn7rLM1u3h0PMAEhyYX7v/Q9wQvoGdB0hlbzm45lEq/wd6y6G9aezvBk9AXwp1r3LHJIRsh6s2maxaJpmvqgvkC7WFS3loUnaFJtKRVUCEoV/RpCnHRvAsesVQ1hw+vd7Mpo+424tLs72NplkvQgcdrsvXkW/zJWqH/fA0FT84M/xnQJt4to3+ZLuanbM6X5lfXKHosO9COgREqpCR5i86pf2zPS7j9tTj+9nO7bQz3+xGEyGW9zqgQ1tyQ/VsxEDvce/4dcUPNb5OD9yXvR4Z2QisuP0xiGWPnemgugU5q/troHhGEjIF5sTOyW648aC0TssuaaCEsYEIkGzjWXOp3A0vVsf6kgRyqaDk+T7DIVWrb58b2tT5xpUucKwodOD/5LbrZC1ws6YSaBZJ/8xlh+XZSYXaMJ2ezNqjB3IPXuehPcx2U6b4t1dS/xNdFzguUt8ie7arnPeyCZroxLHzGgGdqVcspwafizPWEXBee+9G1OaufGdvNng/9C+gwgZ3PH3r87G6zXTZ5D5De2G2DeFoANXfbACkT+fxBQ22YFsTTJF9hjFVO6VbqxZXko4WJ8s52P4PnuxO5KRzu0/hlix1ySt8iXjgaQ+4IHPA9nVzNkdduM9LFT/Aacj4FtKrHA7iAw602Vnht6R8Vq1IOS+wNMKLYqayAYfRuufQPGeGb7sZogQQoLZrGPgZ6KoYn70Iw30O92BNEDpvwouCFn6wH2uS+EhRb3WF/HObZk3HuxfRQM3Y/Of/VH0n4MKNHZDiZvO9+m/ABALfkOcuar/7nOo7B95ACGVAFaz4jMiJwJhdaHBkySmzlGTu82gr6FSTik2kJvLnY9nOd/D90qcH268m3I/cgI1xg1maE5CuZYaWLH+UHANCIck0yt7Mx5zBm5vVHXHwChsZ35kKqUpmo5Svq5/fzfAI5g2vDtFPYo1HiEA85QrDeGm9g//LG7K0scO3sdpj2CBDgCa+0OFs0bkvVgnnM/QBDwllOMm+cN7vMSHlB7Uu4haHKaTwgGkv8tlK+hP8fzmFuK/RQTpaLPWvbd58yWIo66HHM0OsPoPhVqmtaEVL7N+wYcTLTbb0DLdgp23Eyy2VYJ2N7bkLFAAibtoLPe5sLt6Oa2bvU+zyeMa8wrixO0gRTn9tO9NCSThTLGqcqtsDvphlfmx/cPBZVvw24jg1LE2lPuEo35Mhi58U0I/Ga8n5w+NS8i34MAQLos5B1u0xL1ZvCVYVRw/Fs2q53KLaXJMWwOZZ/4MPYV19bAHmgGDKB6f01xoeJKFbl63q9J34KdaVNPJWztQyRkzA3KNs1AdAEDowMxh10emXTCx75CkurtbY/ZpdNDGdsn2UcHKHsQ8Ai3WZi48IfkvtjOhsLpuIRSKZTX9FA4o+0d6o/zOWqQzVJMynL9NsxhSJOaourq6nBVQBueMSyubsX2xHrmuABZN2Ns9jr5nwLFlLF/2R6atjW/67Yd11YQ1Z+kA9Zk9dPTM/o6dVo6HHVgC0JR8oUfmI93T9u3gvTG94bAH02Y5xeqRcjuwnKCK6Q2+ajl8KXJ3GSh22P3Zfx6S+n008ROhJn+JRIUVu6o7OXl8w1SeyhuqNDwNI7SjbK08QrqPxS95jy4G7nCXVq6G3HNu0LtK5J0e226CfC005WKK9sVvfxI0eUbcnzutfhWe3rpZHM0nZ/ny/N8tanKYlQ6VEW5Xuym8yV1zZX58vwGhZp/5tFfhybZabdbrQYOs8F+xEhmPsb0/nki6kIyVvzZzUASiOrTfF+Sj9bXC7DoJxeiV8tjQL6loSd0yCx7YyB6rPdLx31U2qCG3F/oXIuDuqd6LFO+4DNIJuxFZqSsU0ea88avovFnWKRYFYRQDfCfcGaBCLn4M4A1ntJ5E57vicwqq2enaZEF5nokCYu9TbKqCC5yCDfL+GhLxT4w4xEJs+anqgou8DOY2q8FMryjb2MehC1dRJ9s4g9NXeTwPkWON4RH+FhIe0AWR/S9ekvQ+t70XHeimGF78LzuU7d7PwrswdIG2VpgF8C53qVQsTDtBJc4CdnkQPbnZY9mbPdDFra3PCXBBQ5QBn2aQqtyhvlyYM4Hb2/mdhsxCUen04GZVvIJZw5PAamMOmjzq8Q+dzAKLXDQ3RUZItWsg4t7W2DP+JDrJDymoMH7E5zQtuEpG03GTIjGCW3LQqOYEsXgFc78x76NeRwY6SNM+IfQoh6myJKRBIcLYxZcwscJ/gI2isTBty2Po9IkYzP0/SS4hGlxRjFAG5z1Jt1LckiB57yWvo35EaolbvA+6fBa24xodL2YjsPpTnj3JgJOqhcgOeLVsYYwoK0wjY+m1D3rGc40CukkaHnkEjarlXrF1B9M6ECQ6Ow0V7R7N4G3LfOHAXtymoyXOb4QhaYHJ/gNBJUkxclpSs7DNcgWWDDmM7Ke5MJpGuioe7w5EOvfTunUKRzOh7G2ylL+6ynHrD54oQO3//cN3yVO+5qMVsPZq0CZIOx4TlcJ8+Vz7V5waL+7WekzUpRFMTnnTlSCq3X5usi8qmIleW/rit1+oQZn1WGSU/sKBYEqMNh1mBOc6PhK8yCfKHdUNQk8o/G19ZPTs5MYfai+DLs5vmee37zEyyH48WW3XA6Xw6+Az8lMhci7N/KleToo7PtTKm+RA887Kqc6E9dyqL/QPTugzMHLbLZtJKqKLFfzVWRNJ63c+95uWT/F7R0U5dDVvuS409AJXhJvD0EwWaWdW8UN11u/7+umaYjT8mJtzZwP/MD4r57fihiHlC5fylHfaqnJdro+Dr7DajvO+vi2EwyD70s8nCH71nzIO1l5Zl+v1DMCb5ebvCMkGHvobXy/hPumGLyX0218/3RyD1GRLOuf9u/OGQyDmto32yMiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIiIv7GP8YjWPR/czH2AAAAAElFTkSuQmCC",
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
                model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
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
async def test_bedrock_custom_api_base():
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
                api_base="https://gateway.ai.cloudflare.com/v1/fa4cdcab1f32b95ca3b53fd36043d691/test/aws-bedrock/bedrock-runtime/us-east-1",
            )
        except Exception as e:
            pass

        print(f"mock_client_post.call_args.kwargs: {mock_client_post.call_args.kwargs}")
        assert (
            mock_client_post.call_args.kwargs["url"]
            == "https://gateway.ai.cloudflare.com/v1/fa4cdcab1f32b95ca3b53fd36043d691/test/aws-bedrock/bedrock-runtime/us-east-1/model/anthropic.claude-3-sonnet-20240229-v1%3A0/converse"
        )
        assert "test" in mock_client_post.call_args.kwargs["headers"]
        assert mock_client_post.call_args.kwargs["headers"]["test"] == "hello world"
        assert (
            mock_client_post.call_args.kwargs["headers"]["Authorization"]
            == "my-test-key"
        )
        mock_client_post.assert_called_once()


@pytest.mark.parametrize(
    "model",
    [
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "bedrock/invoke/anthropic.claude-3-sonnet-20240229-v1:0",
    ],
)
@pytest.mark.asyncio
async def test_bedrock_extra_headers(model):
    """
    Relevant Issue: https://github.com/BerriAI/litellm/issues/9106
    """
    client = AsyncHTTPHandler()

    with patch.object(client, "post", new=AsyncMock()) as mock_client_post:
        litellm.set_verbose = True
        from openai.types.chat import ChatCompletion

        try:
            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": "What's AWS?"}],
                client=client,
                extra_headers={"test": "hello world", "Authorization": "my-test-key"},
            )
        except Exception as e:
            print(f"error: {e}")

        print(f"mock_client_post.call_args.kwargs: {mock_client_post.call_args.kwargs}")
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
        assert "prompt" in json.loads(mock_client_post.call_args.kwargs["data"])

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
                    model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
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


def test_bedrock_tools_transformation_valid_params():
    from litellm.types.llms.bedrock import ToolJsonSchemaBlock

    tools = [
        {
            "type": "function",
            "function": {
                "name": "123-invalid@name",
                "description": "Invalid name test",
                "parameters": {
                    "$id": "https://some/internal/name",
                    "type": "object",
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "properties": {
                        "test": {"type": "string"},
                    },
                    "required": ["test"],
                },
            },
        }
    ]

    result = _bedrock_tools_pt(tools)

    print("bedrock tools after prompt formatting=", result)
    # Ensure the keys for properties in the response is a subset of keys in ToolJsonSchemaBlock
    toolJsonSchema = result[0]["toolSpec"]["inputSchema"]["json"]
    assert toolJsonSchema is not None
    print("transformed toolJsonSchema keys=", toolJsonSchema.keys())
    print(
        "allowed ToolJsonSchemaBlock keys=", ToolJsonSchemaBlock.__annotations__.keys()
    )
    assert set(toolJsonSchema.keys()).issubset(
        set(ToolJsonSchemaBlock.__annotations__.keys())
    )

    assert isinstance(result, list)
    assert len(result) == 1
    assert "toolSpec" in result[0]
    assert result[0]["toolSpec"]["name"] == "a123_invalid_name"
    assert result[0]["toolSpec"]["description"] == "Invalid name test"
    assert "inputSchema" in result[0]["toolSpec"]
    assert "json" in result[0]["toolSpec"]["inputSchema"]
    assert (
        result[0]["toolSpec"]["inputSchema"]["json"]["properties"]["test"]["type"]
        == "string"
    )
    assert "test" in result[0]["toolSpec"]["inputSchema"]["json"]["required"]


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


@pytest.mark.parametrize(
    "model, expected_base_model",
    [
        (
            "apac.anthropic.claude-3-5-sonnet-20240620-v1:0",
            "anthropic.claude-3-5-sonnet-20240620-v1:0",
        ),
    ],
)
def test_bedrock_get_base_model(model, expected_base_model):
    from litellm.llms.bedrock.common_utils import BedrockModelInfo

    assert BedrockModelInfo.get_base_model(model) == expected_base_model


from litellm.litellm_core_utils.prompt_templates.factory import (
    _bedrock_converse_messages_pt,
)


def test_bedrock_converse_translation_tool_message():
    from litellm.types.utils import ChatCompletionMessageToolCall, Function

    litellm.set_verbose = True

    messages = [
        {
            "role": "user",
            "content": "What's the weather like in San Francisco, Tokyo, and Paris? - give me 3 responses",
        },
        {
            "tool_call_id": "tooluse_DnqEmD5qR6y2-aJ-Xd05xw",
            "role": "tool",
            "name": "get_current_weather",
            "content": [
                {
                    "text": '{"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"}',
                    "type": "text",
                }
            ],
        },
    ]

    translated_msg = _bedrock_converse_messages_pt(
        messages=messages, model="", llm_provider=""
    )

    print(translated_msg)
    assert translated_msg == [
        {
            "role": "user",
            "content": [
                {
                    "text": "What's the weather like in San Francisco, Tokyo, and Paris? - give me 3 responses"
                },
                {
                    "toolResult": {
                        "content": [
                            {
                                "text": '{"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"}'
                            }
                        ],
                        "toolUseId": "tooluse_DnqEmD5qR6y2-aJ-Xd05xw",
                    }
                },
            ],
        }
    ]


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


def test_bedrock_completion_test_2():
    litellm.set_verbose = True
    data = {
        "model": "bedrock/anthropic.claude-3-opus-20240229-v1:0",
        "messages": [
            {
                "role": "system",
                "content": "You are Claude Dev, a highly skilled software developer with extensive knowledge in many programming languages, frameworks, design patterns, and best practices.\n\n====\n \nCAPABILITIES\n\n- You can read and analyze code in various programming languages, and can write clean, efficient, and well-documented code.\n- You can debug complex issues and providing detailed explanations, offering architectural insights and design patterns.\n- You have access to tools that let you execute CLI commands on the user's computer, list files, view source code definitions, regex search, inspect websites, read and write files, and ask follow-up questions. These tools help you effectively accomplish a wide range of tasks, such as writing code, making edits or improvements to existing files, understanding the current state of a project, performing system operations, and much more.\n- When the user initially gives you a task, a recursive list of all filepaths in the current working directory ('/Users/hongbo-miao/Clouds/Git/hongbomiao.com') will be included in environment_details. This provides an overview of the project's file structure, offering key insights into the project from directory/file names (how developers conceptualize and organize their code) and file extensions (the language used). This can also guide decision-making on which files to explore further. If you need to further explore directories such as outside the current working directory, you can use the list_files tool. If you pass 'true' for the recursive parameter, it will list files recursively. Otherwise, it will list files at the top level, which is better suited for generic directories where you don't necessarily need the nested structure, like the Desktop.\n- You can use search_files to perform regex searches across files in a specified directory, outputting context-rich results that include surrounding lines. This is particularly useful for understanding code patterns, finding specific implementations, or identifying areas that need refactoring.\n- You can use the list_code_definition_names tool to get an overview of source code definitions for all files at the top level of a specified directory. This can be particularly useful when you need to understand the broader context and relationships between certain parts of the code. You may need to call this tool multiple times to understand various parts of the codebase related to the task.\n\t- For example, when asked to make edits or improvements you might analyze the file structure in the initial environment_details to get an overview of the project, then use list_code_definition_names to get further insight using source code definitions for files located in relevant directories, then read_file to examine the contents of relevant files, analyze the code and suggest improvements or make necessary edits, then use the write_to_file tool to implement changes. If you refactored code that could affect other parts of the codebase, you could use search_files to ensure you update other files as needed.\n- You can use the execute_command tool to run commands on the user's computer whenever you feel it can help accomplish the user's task. When you need to execute a CLI command, you must provide a clear explanation of what the command does. Prefer to execute complex CLI commands over creating executable scripts, since they are more flexible and easier to run. Interactive and long-running commands are allowed, since the commands are run in the user's VSCode terminal. The user may keep commands running in the background and you will be kept updated on their status along the way. Each command you execute is run in a new terminal instance.\n- You can use the inspect_site tool to capture a screenshot and console logs of the initial state of a website (including html files and locally running development servers) when you feel it is necessary in accomplishing the user's task. This tool may be useful at key stages of web development tasks-such as after implementing new features, making substantial changes, when troubleshooting issues, or to verify the result of your work. You can analyze the provided screenshot to ensure correct rendering or identify errors, and review console logs for runtime issues.\n\t- For example, if asked to add a component to a react website, you might create the necessary files, use execute_command to run the site locally, then use inspect_site to verify there are no runtime errors on page load.\n\n====\n\nRULES\n\n- Your current working directory is: /Users/hongbo-miao/Clouds/Git/hongbomiao.com\n- You cannot `cd` into a different directory to complete a task. You are stuck operating from '/Users/hongbo-miao/Clouds/Git/hongbomiao.com', so be sure to pass in the correct 'path' parameter when using tools that require a path.\n- Do not use the ~ character or $HOME to refer to the home directory.\n- Before using the execute_command tool, you must first think about the SYSTEM INFORMATION context provided to understand the user's environment and tailor your commands to ensure they are compatible with their system. You must also consider if the command you need to run should be executed in a specific directory outside of the current working directory '/Users/hongbo-miao/Clouds/Git/hongbomiao.com', and if so prepend with `cd`'ing into that directory && then executing the command (as one command since you are stuck operating from '/Users/hongbo-miao/Clouds/Git/hongbomiao.com'). For example, if you needed to run `npm install` in a project outside of '/Users/hongbo-miao/Clouds/Git/hongbomiao.com', you would need to prepend with a `cd` i.e. pseudocode for this would be `cd (path to project) && (command, in this case npm install)`.\n- When using the search_files tool, craft your regex patterns carefully to balance specificity and flexibility. Based on the user's task you may use it to find code patterns, TODO comments, function definitions, or any text-based information across the project. The results include context, so analyze the surrounding code to better understand the matches. Leverage the search_files tool in combination with other tools for more comprehensive analysis. For example, use it to find specific code patterns, then use read_file to examine the full context of interesting matches before using write_to_file to make informed changes.\n- When creating a new project (such as an app, website, or any software project), organize all new files within a dedicated project directory unless the user specifies otherwise. Use appropriate file paths when writing files, as the write_to_file tool will automatically create any necessary directories. Structure the project logically, adhering to best practices for the specific type of project being created. Unless otherwise specified, new projects should be easily run without additional setup, for example most projects can be built in HTML, CSS, and JavaScript - which you can open in a browser.\n- You must try to use multiple tools in one request when possible. For example if you were to create a website, you would use the write_to_file tool to create the necessary files with their appropriate contents all at once. Or if you wanted to analyze a project, you could use the read_file tool multiple times to look at several key files. This will help you accomplish the user's task more efficiently.\n- Be sure to consider the type of project (e.g. Python, JavaScript, web application) when determining the appropriate structure and files to include. Also consider what files may be most relevant to accomplishing the task, for example looking at a project's manifest file would help you understand the project's dependencies, which you could incorporate into any code you write.\n- When making changes to code, always consider the context in which the code is being used. Ensure that your changes are compatible with the existing codebase and that they follow the project's coding standards and best practices.\n- Do not ask for more information than necessary. Use the tools provided to accomplish the user's request efficiently and effectively. When you've completed your task, you must use the attempt_completion tool to present the result to the user. The user may provide feedback, which you can use to make improvements and try again.\n- You are only allowed to ask the user questions using the ask_followup_question tool. Use this tool only when you need additional details to complete a task, and be sure to use a clear and concise question that will help you move forward with the task. However if you can use the available tools to avoid having to ask the user questions, you should do so. For example, if the user mentions a file that may be in an outside directory like the Desktop, you should use the list_files tool to list the files in the Desktop and check if the file they are talking about is there, rather than asking the user to provide the file path themselves.\n- When executing commands, if you don't see the expected output, assume the terminal executed the command successfully and proceed with the task. The user's terminal may be unable to stream the output back properly. If you absolutely need to see the actual terminal output, use the ask_followup_question tool to request the user to copy and paste it back to you.\n- Your goal is to try to accomplish the user's task, NOT engage in a back and forth conversation.\n- NEVER end completion_attempt with a question or request to engage in further conversation! Formulate the end of your result in a way that is final and does not require further input from the user. \n- NEVER start your responses with affirmations like \"Certainly\", \"Okay\", \"Sure\", \"Great\", etc. You should NOT be conversational in your responses, but rather direct and to the point.\n- Feel free to use markdown as much as you'd like in your responses. When using code blocks, always include a language specifier.\n- When presented with images, utilize your vision capabilities to thoroughly examine them and extract meaningful information. Incorporate these insights into your thought process as you accomplish the user's task.\n- At the end of each user message, you will automatically receive environment_details. This information is not written by the user themselves, but is auto-generated to provide potentially relevant context about the project structure and environment. While this information can be valuable for understanding the project context, do not treat it as a direct part of the user's request or response. Use it to inform your actions and decisions, but don't assume the user is explicitly asking about or referring to this information unless they clearly do so in their message. When using environment_details, explain your actions clearly to ensure the user understands, as they may not be aware of these details.\n- CRITICAL: When editing files with write_to_file, ALWAYS provide the COMPLETE file content in your response. This is NON-NEGOTIABLE. Partial updates or placeholders like '// rest of code unchanged' are STRICTLY FORBIDDEN. You MUST include ALL parts of the file, even if they haven't been modified. Failure to do so will result in incomplete or broken code, severely impacting the user's project.\n\n====\n\nOBJECTIVE\n\nYou accomplish a given task iteratively, breaking it down into clear steps and working through them methodically.\n\n1. Analyze the user's task and set clear, achievable goals to accomplish it. Prioritize these goals in a logical order.\n2. Work through these goals sequentially, utilizing available tools as necessary. Each goal should correspond to a distinct step in your problem-solving process. It is okay for certain steps to take multiple iterations, i.e. if you need to create many files, it's okay to create a few files at a time as each subsequent iteration will keep you informed on the work completed and what's remaining. \n3. Remember, you have extensive capabilities with access to a wide range of tools that can be used in powerful and clever ways as necessary to accomplish each goal. Before calling a tool, do some analysis within <thinking></thinking> tags. First, analyze the file structure provided in environment_details to gain context and insights for proceeding effectively. Then, think about which of the provided tools is the most relevant tool to accomplish the user's task. Next, go through each of the required parameters of the relevant tool and determine if the user has directly provided or given enough information to infer a value. When deciding if the parameter can be inferred, carefully consider all the context to see if it supports a specific value. If all of the required parameters are present or can be reasonably inferred, close the thinking tag and proceed with the tool call. BUT, if one of the values for a required parameter is missing, DO NOT invoke the function (not even with fillers for the missing params) and instead, ask the user to provide the missing parameters using the ask_followup_question tool. DO NOT ask for more information on optional parameters if it is not provided.\n4. Once you've completed the user's task, you must use the attempt_completion tool to present the result of the task to the user. You may also provide a CLI command to showcase the result of your task; this can be particularly useful for web development tasks, where you can run e.g. `open index.html` to show the website you've built.\n5. The user may provide feedback, which you can use to make improvements and try again. But DO NOT continue in pointless back and forth conversations, i.e. don't end your responses with questions or offers for further assistance.\n\n====\n\nSYSTEM INFORMATION\n\nOperating System: macOS\nDefault Shell: /bin/zsh\nHome Directory: /Users/hongbo-miao\nCurrent Working Directory: /Users/hongbo-miao/Clouds/Git/hongbomiao.com\n",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<task>\nHello\n</task>"},
                    {
                        "type": "text",
                        "text": "<environment_details>\n# VSCode Visible Files\ncomputer-vision/hm-open3d/src/main.py\n\n# VSCode Open Tabs\ncomputer-vision/hm-open3d/src/main.py\n../../../.vscode/extensions/continue.continue-0.8.52-darwin-arm64/continue_tutorial.py\n\n# Current Working Directory (/Users/hongbo-miao/Clouds/Git/hongbomiao.com) Files\n.ansible-lint\n.clang-format\n.cmakelintrc\n.dockerignore\n.editorconfig\n.gitignore\n.gitmodules\n.hadolint.yaml\n.isort.cfg\n.markdownlint-cli2.jsonc\n.mergify.yml\n.npmrc\n.nvmrc\n.prettierignore\n.rubocop.yml\n.ruby-version\n.ruff.toml\n.shellcheckrc\n.solhint.json\n.solhintignore\n.sqlfluff\n.sqlfluffignore\n.stylelintignore\n.yamllint.yaml\nCODE_OF_CONDUCT.md\ncommitlint.config.js\nGemfile\nGemfile.lock\nLICENSE\nlint-staged.config.js\nMakefile\nmiss_hit.cfg\nmypy.ini\npackage-lock.json\npackage.json\npoetry.lock\npoetry.toml\nprettier.config.js\npyproject.toml\nREADME.md\nrelease.config.js\nrenovate.json\nSECURITY.md\nstylelint.config.js\naerospace/\naerospace/air-defense-system/\naerospace/hm-aerosandbox/\naerospace/hm-openaerostruct/\naerospace/px4/\naerospace/quadcopter-pd-controller/\naerospace/simulate-satellite/\naerospace/simulated-and-actual-flights/\naerospace/toroidal-propeller/\nansible/\nansible/inventory.yaml\nansible/Makefile\nansible/requirements.yml\nansible/hm_macos_group/\nansible/hm_ubuntu_group/\nansible/hm_windows_group/\napi-go/\napi-go/buf.yaml\napi-go/go.mod\napi-go/go.sum\napi-go/Makefile\napi-go/api/\napi-go/build/\napi-go/cmd/\napi-go/config/\napi-go/internal/\napi-node/\napi-node/.env.development\napi-node/.env.development.local.example\napi-node/.env.development.local.example.docker\napi-node/.env.production\napi-node/.env.production.local.example\napi-node/.env.test\napi-node/.eslintignore\napi-node/.eslintrc.js\napi-node/.npmrc\napi-node/.nvmrc\napi-node/babel.config.js\napi-node/docker-compose.cypress.yaml\napi-node/docker-compose.development.yaml\napi-node/Dockerfile\napi-node/Dockerfile.development\napi-node/jest.config.js\napi-node/Makefile\napi-node/package-lock.json\napi-node/package.json\napi-node/Procfile\napi-node/stryker.conf.js\napi-node/tsconfig.json\napi-node/bin/\napi-node/postgres/\napi-node/scripts/\napi-node/src/\napi-python/\napi-python/.flaskenv\napi-python/docker-entrypoint.sh\napi-python/Dockerfile\napi-python/Makefile\napi-python/poetry.lock\napi-python/poetry.toml\napi-python/pyproject.toml\napi-python/flaskr/\nasterios/\nasterios/led-blinker/\nauthorization/\nauthorization/hm-opal-client/\nauthorization/ory-hydra/\nautomobile/\nautomobile/build-map-by-lidar-point-cloud/\nautomobile/detect-lane-by-lidar-point-cloud/\nbin/\nbin/clean.sh\nbin/count_code_lines.sh\nbin/lint_javascript_fix.sh\nbin/lint_javascript.sh\nbin/set_up.sh\nbiology/\nbiology/compare-nucleotide-sequences/\nbusybox/\nbusybox/Makefile\ncaddy/\ncaddy/Caddyfile\ncaddy/Makefile\ncaddy/bin/\ncloud-computing/\ncloud-computing/hm-ray/\ncloud-computing/hm-skypilot/\ncloud-cost/\ncloud-cost/komiser/\ncloud-infrastructure/\ncloud-infrastructure/hm-pulumi/\ncloud-infrastructure/karpenter/\ncloud-infrastructure/terraform/\ncloud-platform/\ncloud-platform/aws/\ncloud-platform/google-cloud/\ncloud-security/\ncloud-security/hm-prowler/\ncomputational-fluid-dynamics/\ncomputational-fluid-dynamics/matlab/\ncomputational-fluid-dynamics/openfoam/\ncomputer-vision/\ncomputer-vision/hm-open3d/\ncomputer-vision/hm-pyvista/\ndata-analytics/\ndata-analytics/hm-geopandas/\ndata-distribution-service/\ndata-distribution-service/dummy_test.py\ndata-distribution-service/hm_message.idl\ndata-distribution-service/hm_message.xml\ndata-distribution-service/Makefile\ndata-distribution-service/poetry.lock\ndata-distribution-service/poetry.toml\ndata-distribution-service/publish.py\ndata-ingestion/\ndata-orchestration/\ndata-processing/\ndata-storage/\ndata-transformation/\ndata-visualization/\ndesktop-qt/\nembedded/\nethereum/\ngit/\ngolang-migrate/\nhardware-in-the-loop/\nhasura-graphql-engine/\nhigh-performance-computing/\nhm-alpine/\nhm-kafka/\nhm-locust/\nhm-rust/\nhm-traefik/\nhm-xxhash/\nkubernetes/\nmachine-learning/\nmatlab/\nmobile/\nnetwork-programmability/\noperating-system/\nparallel-computing/\nphysics/\nquantum-computing/\nrclone/\nrestic/\nreverse-engineering/\nrobotics/\nsubmodules/\ntrino/\nvagrant/\nvalgrind/\nvhdl/\nvim/\nweb/\nweb-cypress/\nwireless-network/\n\n(File list truncated. Use list_files on specific subdirectories if you need to explore further.)\n</environment_details>",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": '<thinking>\nThe user has simply said "Hello" without providing any specific task or request. There is not enough information to determine which tools would be relevant or necessary to respond.\n\nThe environment details show a list of files and directories in the current working directory, but without a clear task from the user, it\'s not apparent which of these, if any, are relevant.\n\nSince no specific request has been made, there are no required parameters to analyze for any of the available tools. Asking a follow-up question seems to be the most appropriate action to get clarification on what the user needs help with.\n</thinking>',
                "tool_calls": [
                    {
                        "id": "tooluse_OPznXwZaRzCfPaQF2dxRSA",
                        "type": "function",
                        "function": {
                            "name": "ask_followup_question",
                            "arguments": '{"question":"Hello! How can I assist you today? Do you have a specific task or request you need help with? I\'d be happy to help, but I\'ll need some more details on what you\'re looking to accomplish."}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "tooluse_OPznXwZaRzCfPaQF2dxRSA",
                "content": "<answer>\nExplain this file\n</answer>",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "<environment_details>\n# VSCode Visible Files\ncomputer-vision/hm-open3d/src/main.py\n\n# VSCode Open Tabs\ncomputer-vision/hm-open3d/src/main.py\n../../../.vscode/extensions/continue.continue-0.8.52-darwin-arm64/continue_tutorial.py\n</environment_details>",
                    }
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "execute_command",
                    "description": "Execute a CLI command on the system. Use this when you need to perform system operations or run specific commands to accomplish any step in the user's task. You must tailor your command to the user's system and provide a clear explanation of what the command does. Prefer to execute complex CLI commands over creating executable scripts, as they are more flexible and easier to run. Commands will be executed in the current working directory: /Users/hongbo-miao/Clouds/Git/hongbomiao.com",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The CLI command to execute. This should be valid for the current operating system. Ensure the command is properly formatted and does not contain any harmful instructions.",
                            }
                        },
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file at the specified path. Use this when you need to examine the contents of an existing file, for example to analyze code, review text files, or extract information from configuration files. Automatically extracts raw text from PDF and DOCX files. May not be suitable for other types of binary files, as it returns the raw content as a string.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The path of the file to read (relative to the current working directory /Users/hongbo-miao/Clouds/Git/hongbomiao.com)",
                            }
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_to_file",
                    "description": "Write content to a file at the specified path. If the file exists, it will be overwritten with the provided content. If the file doesn't exist, it will be created. Always provide the full intended content of the file, without any truncation. This tool will automatically create any directories needed to write the file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The path of the file to write to (relative to the current working directory /Users/hongbo-miao/Clouds/Git/hongbomiao.com)",
                            },
                            "content": {
                                "type": "string",
                                "description": "The full content to write to the file.",
                            },
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "description": "Perform a regex search across files in a specified directory, providing context-rich results. This tool searches for patterns or specific content across multiple files, displaying each match with encapsulating context.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The path of the directory to search in (relative to the current working directory /Users/hongbo-miao/Clouds/Git/hongbomiao.com). This directory will be recursively searched.",
                            },
                            "regex": {
                                "type": "string",
                                "description": "The regular expression pattern to search for. Uses Rust regex syntax.",
                            },
                            "filePattern": {
                                "type": "string",
                                "description": "Optional glob pattern to filter files (e.g., '*.ts' for TypeScript files). If not provided, it will search all files (*).",
                            },
                        },
                        "required": ["path", "regex"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files and directories within the specified directory. If recursive is true, it will list all files and directories recursively. If recursive is false or not provided, it will only list the top-level contents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The path of the directory to list contents for (relative to the current working directory /Users/hongbo-miao/Clouds/Git/hongbomiao.com)",
                            },
                            "recursive": {
                                "type": "string",
                                "enum": ["true", "false"],
                                "description": "Whether to list files recursively. Use 'true' for recursive listing, 'false' or omit for top-level only.",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_code_definition_names",
                    "description": "Lists definition names (classes, functions, methods, etc.) used in source code files at the top level of the specified directory. This tool provides insights into the codebase structure and important constructs, encapsulating high-level concepts and relationships that are crucial for understanding the overall architecture.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The path of the directory (relative to the current working directory /Users/hongbo-miao/Clouds/Git/hongbomiao.com) to list top level source code definitions for",
                            }
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "inspect_site",
                    "description": "Captures a screenshot and console logs of the initial state of a website. This tool navigates to the specified URL, takes a screenshot of the entire page as it appears immediately after loading, and collects any console logs or errors that occur during page load. It does not interact with the page or capture any state changes after the initial load.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL of the site to inspect. This should be a valid URL including the protocol (e.g. http://localhost:3000/page, file:///path/to/file.html, etc.)",
                            }
                        },
                        "required": ["url"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ask_followup_question",
                    "description": "Ask the user a question to gather additional information needed to complete the task. This tool should be used when you encounter ambiguities, need clarification, or require more details to proceed effectively. It allows for interactive problem-solving by enabling direct communication with the user. Use this tool judiciously to maintain a balance between gathering necessary information and avoiding excessive back-and-forth.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The question to ask the user. This should be a clear, specific question that addresses the information you need.",
                            }
                        },
                        "required": ["question"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "attempt_completion",
                    "description": "Once you've completed the task, use this tool to present the result to the user. Optionally you may provide a CLI command to showcase the result of your work, but avoid using commands like 'echo' or 'cat' that merely print text. They may respond with feedback if they are not satisfied with the result, which you can use to make improvements and try again.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "A CLI command to execute to show a live demo of the result to the user. For example, use 'open index.html' to display a created website. This command should be valid for the current operating system. Ensure the command is properly formatted and does not contain any harmful instructions.",
                            },
                            "result": {
                                "type": "string",
                                "description": "The result of the task. Formulate this result in a way that is final and does not require further input from the user. Don't end your result with questions or offers for further assistance.",
                            },
                        },
                        "required": ["result"],
                    },
                },
            },
        ],
    }

    from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig

    request = AmazonConverseConfig()._transform_request(
        model=data["model"],
        messages=data["messages"],
        optional_params={"tools": data["tools"]},
        litellm_params={},
    )

    """
    Iterate through the messages

    ensure 'role' is always alternating b/w 'user' and 'assistant'
    """
    _messages = request["messages"]
    for i in range(len(_messages) - 1):
        assert _messages[i]["role"] != _messages[i + 1]["role"]


def test_bedrock_completion_test_3():
    """
    Check if content in tool result is formatted correctly
    """
    from litellm.types.utils import ChatCompletionMessageToolCall, Function, Message
    from litellm.litellm_core_utils.prompt_templates.factory import (
        _bedrock_converse_messages_pt,
    )

    messages = [
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
                    id="tooluse_EF8PwJ1dSMSh6tLGKu9VdA",
                    type="function",
                )
            ],
            function_call=None,
        ).model_dump(),
        {
            "tool_call_id": "tooluse_EF8PwJ1dSMSh6tLGKu9VdA",
            "role": "tool",
            "name": "get_current_weather",
            "content": '{"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"}',
        },
    ]

    transformed_messages = _bedrock_converse_messages_pt(
        messages=messages, model="", llm_provider=""
    )
    print(transformed_messages)

    assert transformed_messages[-1]["role"] == "user"
    assert transformed_messages[-1]["content"] == [
        {
            "toolResult": {
                "content": [
                    {
                        "text": '{"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"}'
                    }
                ],
                "toolUseId": "tooluse_EF8PwJ1dSMSh6tLGKu9VdA",
            }
        }
    ]


@pytest.mark.skip(reason="Skipping this test as Bedrock now supports this behavior.")
@pytest.mark.parametrize("modify_params", [True, False])
def test_bedrock_completion_test_4(modify_params):
    litellm.set_verbose = True
    litellm.modify_params = modify_params

    data = {
        "model": "anthropic.claude-3-opus-20240229-v1:0",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<task>\nWhat is this file?\n</task>"},
                    {
                        "type": "text",
                        "text": "<environment_details>\n# VSCode Visible Files\ncomputer-vision/hm-open3d/src/main.py\n\n# VSCode Open Tabs\ncomputer-vision/hm-open3d/src/main.py\n\n# Current Working Directory (/Users/hongbo-miao/Clouds/Git/hongbomiao.com) Files\n.ansible-lint\n.clang-format\n.cmakelintrc\n.dockerignore\n.editorconfig\n.gitignore\n.gitmodules\n.hadolint.yaml\n.isort.cfg\n.markdownlint-cli2.jsonc\n.mergify.yml\n.npmrc\n.nvmrc\n.prettierignore\n.rubocop.yml\n.ruby-version\n.ruff.toml\n.shellcheckrc\n.solhint.json\n.solhintignore\n.sqlfluff\n.sqlfluffignore\n.stylelintignore\n.yamllint.yaml\nCODE_OF_CONDUCT.md\ncommitlint.config.js\nGemfile\nGemfile.lock\nLICENSE\nlint-staged.config.js\nMakefile\nmiss_hit.cfg\nmypy.ini\npackage-lock.json\npackage.json\npoetry.lock\npoetry.toml\nprettier.config.js\npyproject.toml\nREADME.md\nrelease.config.js\nrenovate.json\nSECURITY.md\nstylelint.config.js\naerospace/\naerospace/air-defense-system/\naerospace/hm-aerosandbox/\naerospace/hm-openaerostruct/\naerospace/px4/\naerospace/quadcopter-pd-controller/\naerospace/simulate-satellite/\naerospace/simulated-and-actual-flights/\naerospace/toroidal-propeller/\nansible/\nansible/inventory.yaml\nansible/Makefile\nansible/requirements.yml\nansible/hm_macos_group/\nansible/hm_ubuntu_group/\nansible/hm_windows_group/\napi-go/\napi-go/buf.yaml\napi-go/go.mod\napi-go/go.sum\napi-go/Makefile\napi-go/api/\napi-go/build/\napi-go/cmd/\napi-go/config/\napi-go/internal/\napi-node/\napi-node/.env.development\napi-node/.env.development.local.example\napi-node/.env.development.local.example.docker\napi-node/.env.production\napi-node/.env.production.local.example\napi-node/.env.test\napi-node/.eslintignore\napi-node/.eslintrc.js\napi-node/.npmrc\napi-node/.nvmrc\napi-node/babel.config.js\napi-node/docker-compose.cypress.yaml\napi-node/docker-compose.development.yaml\napi-node/Dockerfile\napi-node/Dockerfile.development\napi-node/jest.config.js\napi-node/Makefile\napi-node/package-lock.json\napi-node/package.json\napi-node/Procfile\napi-node/stryker.conf.js\napi-node/tsconfig.json\napi-node/bin/\napi-node/postgres/\napi-node/scripts/\napi-node/src/\napi-python/\napi-python/.flaskenv\napi-python/docker-entrypoint.sh\napi-python/Dockerfile\napi-python/Makefile\napi-python/poetry.lock\napi-python/poetry.toml\napi-python/pyproject.toml\napi-python/flaskr/\nasterios/\nasterios/led-blinker/\nauthorization/\nauthorization/hm-opal-client/\nauthorization/ory-hydra/\nautomobile/\nautomobile/build-map-by-lidar-point-cloud/\nautomobile/detect-lane-by-lidar-point-cloud/\nbin/\nbin/clean.sh\nbin/count_code_lines.sh\nbin/lint_javascript_fix.sh\nbin/lint_javascript.sh\nbin/set_up.sh\nbiology/\nbiology/compare-nucleotide-sequences/\nbusybox/\nbusybox/Makefile\ncaddy/\ncaddy/Caddyfile\ncaddy/Makefile\ncaddy/bin/\ncloud-computing/\ncloud-computing/hm-ray/\ncloud-computing/hm-skypilot/\ncloud-cost/\ncloud-cost/komiser/\ncloud-infrastructure/\ncloud-infrastructure/hm-pulumi/\ncloud-infrastructure/karpenter/\ncloud-infrastructure/terraform/\ncloud-platform/\ncloud-platform/aws/\ncloud-platform/google-cloud/\ncloud-security/\ncloud-security/hm-prowler/\ncomputational-fluid-dynamics/\ncomputational-fluid-dynamics/matlab/\ncomputational-fluid-dynamics/openfoam/\ncomputer-vision/\ncomputer-vision/hm-open3d/\ncomputer-vision/hm-pyvista/\ndata-analytics/\ndata-analytics/hm-geopandas/\ndata-distribution-service/\ndata-distribution-service/dummy_test.py\ndata-distribution-service/hm_message.idl\ndata-distribution-service/hm_message.xml\ndata-distribution-service/Makefile\ndata-distribution-service/poetry.lock\ndata-distribution-service/poetry.toml\ndata-distribution-service/publish.py\ndata-ingestion/\ndata-orchestration/\ndata-processing/\ndata-storage/\ndata-transformation/\ndata-visualization/\ndesktop-qt/\nembedded/\nethereum/\ngit/\ngolang-migrate/\nhardware-in-the-loop/\nhasura-graphql-engine/\nhigh-performance-computing/\nhm-alpine/\nhm-kafka/\nhm-locust/\nhm-rust/\nhm-traefik/\nhm-xxhash/\nkubernetes/\nmachine-learning/\nmatlab/\nmobile/\nnetwork-programmability/\noperating-system/\nparallel-computing/\nphysics/\nquantum-computing/\nrclone/\nrestic/\nreverse-engineering/\nrobotics/\nsubmodules/\ntrino/\nvagrant/\nvalgrind/\nvhdl/\nvim/\nweb/\nweb-cypress/\nwireless-network/\n\n(File list truncated. Use list_files on specific subdirectories if you need to explore further.)\n</environment_details>",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": '<thinking>\nThe user is asking about a specific file: main.py. Based on the environment details provided, this file is located in the computer-vision/hm-open3d/src/ directory and is currently open in a VSCode tab.\n\nTo answer the question of what this file is, the most relevant tool would be the read_file tool. This will allow me to examine the contents of main.py to determine its purpose.\n\nThe read_file tool requires the "path" parameter. I can infer this path based on the environment details:\npath: "computer-vision/hm-open3d/src/main.py"\n\nSince I have the necessary parameter, I can proceed with calling the read_file tool.\n</thinking>',
                "tool_calls": [
                    {
                        "id": "tooluse_qCt-KEyWQlWiyHl26spQVA",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"computer-vision/hm-open3d/src/main.py"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "tooluse_qCt-KEyWQlWiyHl26spQVA",
                "content": 'import numpy as np\nimport open3d as o3d\n\n\ndef main():\n    ply_point_cloud = o3d.data.PLYPointCloud()\n    pcd = o3d.io.read_point_cloud(ply_point_cloud.path)\n    print(pcd)\n    print(np.asarray(pcd.points))\n\n    demo_crop_data = o3d.data.DemoCropPointCloud()\n    vol = o3d.visualization.read_selection_polygon_volume(\n        demo_crop_data.cropped_json_path\n    )\n    chair = vol.crop_point_cloud(pcd)\n\n    dists = pcd.compute_point_cloud_distance(chair)\n    dists = np.asarray(dists)\n    idx = np.where(dists > 0.01)[0]\n    pcd_without_chair = pcd.select_by_index(idx)\n\n    axis_aligned_bounding_box = chair.get_axis_aligned_bounding_box()\n    axis_aligned_bounding_box.color = (1, 0, 0)\n\n    oriented_bounding_box = chair.get_oriented_bounding_box()\n    oriented_bounding_box.color = (0, 1, 0)\n\n    o3d.visualization.draw_geometries(\n        [pcd_without_chair, chair, axis_aligned_bounding_box, oriented_bounding_box],\n        zoom=0.3412,\n        front=[0.4, -0.2, -0.9],\n        lookat=[2.6, 2.0, 1.5],\n        up=[-0.10, -1.0, 0.2],\n    )\n\n\nif __name__ == "__main__":\n    main()\n',
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "<environment_details>\n# VSCode Visible Files\ncomputer-vision/hm-open3d/src/main.py\n\n# VSCode Open Tabs\ncomputer-vision/hm-open3d/src/main.py\n</environment_details>",
                    }
                ],
            },
        ],
        "temperature": 0.2,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "execute_command",
                    "description": "Execute a CLI command on the system. Use this when you need to perform system operations or run specific commands to accomplish any step in the user's task. You must tailor your command to the user's system and provide a clear explanation of what the command does. Prefer to execute complex CLI commands over creating executable scripts, as they are more flexible and easier to run. Commands will be executed in the current working directory: /Users/hongbo-miao/Clouds/Git/hongbomiao.com",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The CLI command to execute. This should be valid for the current operating system. Ensure the command is properly formatted and does not contain any harmful instructions.",
                            }
                        },
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file at the specified path. Use this when you need to examine the contents of an existing file, for example to analyze code, review text files, or extract information from configuration files. Automatically extracts raw text from PDF and DOCX files. May not be suitable for other types of binary files, as it returns the raw content as a string.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The path of the file to read (relative to the current working directory /Users/hongbo-miao/Clouds/Git/hongbomiao.com)",
                            }
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_to_file",
                    "description": "Write content to a file at the specified path. If the file exists, it will be overwritten with the provided content. If the file doesn't exist, it will be created. Always provide the full intended content of the file, without any truncation. This tool will automatically create any directories needed to write the file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The path of the file to write to (relative to the current working directory /Users/hongbo-miao/Clouds/Git/hongbomiao.com)",
                            },
                            "content": {
                                "type": "string",
                                "description": "The full content to write to the file.",
                            },
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "description": "Perform a regex search across files in a specified directory, providing context-rich results. This tool searches for patterns or specific content across multiple files, displaying each match with encapsulating context.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The path of the directory to search in (relative to the current working directory /Users/hongbo-miao/Clouds/Git/hongbomiao.com). This directory will be recursively searched.",
                            },
                            "regex": {
                                "type": "string",
                                "description": "The regular expression pattern to search for. Uses Rust regex syntax.",
                            },
                            "filePattern": {
                                "type": "string",
                                "description": "Optional glob pattern to filter files (e.g., '*.ts' for TypeScript files). If not provided, it will search all files (*).",
                            },
                        },
                        "required": ["path", "regex"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files and directories within the specified directory. If recursive is true, it will list all files and directories recursively. If recursive is false or not provided, it will only list the top-level contents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The path of the directory to list contents for (relative to the current working directory /Users/hongbo-miao/Clouds/Git/hongbomiao.com)",
                            },
                            "recursive": {
                                "type": "string",
                                "enum": ["true", "false"],
                                "description": "Whether to list files recursively. Use 'true' for recursive listing, 'false' or omit for top-level only.",
                            },
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_code_definition_names",
                    "description": "Lists definition names (classes, functions, methods, etc.) used in source code files at the top level of the specified directory. This tool provides insights into the codebase structure and important constructs, encapsulating high-level concepts and relationships that are crucial for understanding the overall architecture.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The path of the directory (relative to the current working directory /Users/hongbo-miao/Clouds/Git/hongbomiao.com) to list top level source code definitions for",
                            }
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "inspect_site",
                    "description": "Captures a screenshot and console logs of the initial state of a website. This tool navigates to the specified URL, takes a screenshot of the entire page as it appears immediately after loading, and collects any console logs or errors that occur during page load. It does not interact with the page or capture any state changes after the initial load.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL of the site to inspect. This should be a valid URL including the protocol (e.g. http://localhost:3000/page, file:///path/to/file.html, etc.)",
                            }
                        },
                        "required": ["url"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ask_followup_question",
                    "description": "Ask the user a question to gather additional information needed to complete the task. This tool should be used when you encounter ambiguities, need clarification, or require more details to proceed effectively. It allows for interactive problem-solving by enabling direct communication with the user. Use this tool judiciously to maintain a balance between gathering necessary information and avoiding excessive back-and-forth.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The question to ask the user. This should be a clear, specific question that addresses the information you need.",
                            }
                        },
                        "required": ["question"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "attempt_completion",
                    "description": "Once you've completed the task, use this tool to present the result to the user. Optionally you may provide a CLI command to showcase the result of your work, but avoid using commands like 'echo' or 'cat' that merely print text. They may respond with feedback if they are not satisfied with the result, which you can use to make improvements and try again.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "A CLI command to execute to show a live demo of the result to the user. For example, use 'open index.html' to display a created website. This command should be valid for the current operating system. Ensure the command is properly formatted and does not contain any harmful instructions.",
                            },
                            "result": {
                                "type": "string",
                                "description": "The result of the task. Formulate this result in a way that is final and does not require further input from the user. Don't end your result with questions or offers for further assistance.",
                            },
                        },
                        "required": ["result"],
                    },
                },
            },
        ],
        "tool_choice": "auto",
    }

    if modify_params:
        transformed_messages = _bedrock_converse_messages_pt(
            messages=data["messages"], model="", llm_provider=""
        )
        expected_messages = [
            {
                "role": "user",
                "content": [
                    {"text": "<task>\nWhat is this file?\n</task>"},
                    {
                        "text": "<environment_details>\n# VSCode Visible Files\ncomputer-vision/hm-open3d/src/main.py\n\n# VSCode Open Tabs\ncomputer-vision/hm-open3d/src/main.py\n\n# Current Working Directory (/Users/hongbo-miao/Clouds/Git/hongbomiao.com) Files\n.ansible-lint\n.clang-format\n.cmakelintrc\n.dockerignore\n.editorconfig\n.gitignore\n.gitmodules\n.hadolint.yaml\n.isort.cfg\n.markdownlint-cli2.jsonc\n.mergify.yml\n.npmrc\n.nvmrc\n.prettierignore\n.rubocop.yml\n.ruby-version\n.ruff.toml\n.shellcheckrc\n.solhint.json\n.solhintignore\n.sqlfluff\n.sqlfluffignore\n.stylelintignore\n.yamllint.yaml\nCODE_OF_CONDUCT.md\ncommitlint.config.js\nGemfile\nGemfile.lock\nLICENSE\nlint-staged.config.js\nMakefile\nmiss_hit.cfg\nmypy.ini\npackage-lock.json\npackage.json\npoetry.lock\npoetry.toml\nprettier.config.js\npyproject.toml\nREADME.md\nrelease.config.js\nrenovate.json\nSECURITY.md\nstylelint.config.js\naerospace/\naerospace/air-defense-system/\naerospace/hm-aerosandbox/\naerospace/hm-openaerostruct/\naerospace/px4/\naerospace/quadcopter-pd-controller/\naerospace/simulate-satellite/\naerospace/simulated-and-actual-flights/\naerospace/toroidal-propeller/\nansible/\nansible/inventory.yaml\nansible/Makefile\nansible/requirements.yml\nansible/hm_macos_group/\nansible/hm_ubuntu_group/\nansible/hm_windows_group/\napi-go/\napi-go/buf.yaml\napi-go/go.mod\napi-go/go.sum\napi-go/Makefile\napi-go/api/\napi-go/build/\napi-go/cmd/\napi-go/config/\napi-go/internal/\napi-node/\napi-node/.env.development\napi-node/.env.development.local.example\napi-node/.env.development.local.example.docker\napi-node/.env.production\napi-node/.env.production.local.example\napi-node/.env.test\napi-node/.eslintignore\napi-node/.eslintrc.js\napi-node/.npmrc\napi-node/.nvmrc\napi-node/babel.config.js\napi-node/docker-compose.cypress.yaml\napi-node/docker-compose.development.yaml\napi-node/Dockerfile\napi-node/Dockerfile.development\napi-node/jest.config.js\napi-node/Makefile\napi-node/package-lock.json\napi-node/package.json\napi-node/Procfile\napi-node/stryker.conf.js\napi-node/tsconfig.json\napi-node/bin/\napi-node/postgres/\napi-node/scripts/\napi-node/src/\napi-python/\napi-python/.flaskenv\napi-python/docker-entrypoint.sh\napi-python/Dockerfile\napi-python/Makefile\napi-python/poetry.lock\napi-python/poetry.toml\napi-python/pyproject.toml\napi-python/flaskr/\nasterios/\nasterios/led-blinker/\nauthorization/\nauthorization/hm-opal-client/\nauthorization/ory-hydra/\nautomobile/\nautomobile/build-map-by-lidar-point-cloud/\nautomobile/detect-lane-by-lidar-point-cloud/\nbin/\nbin/clean.sh\nbin/count_code_lines.sh\nbin/lint_javascript_fix.sh\nbin/lint_javascript.sh\nbin/set_up.sh\nbiology/\nbiology/compare-nucleotide-sequences/\nbusybox/\nbusybox/Makefile\ncaddy/\ncaddy/Caddyfile\ncaddy/Makefile\ncaddy/bin/\ncloud-computing/\ncloud-computing/hm-ray/\ncloud-computing/hm-skypilot/\ncloud-cost/\ncloud-cost/komiser/\ncloud-infrastructure/\ncloud-infrastructure/hm-pulumi/\ncloud-infrastructure/karpenter/\ncloud-infrastructure/terraform/\ncloud-platform/\ncloud-platform/aws/\ncloud-platform/google-cloud/\ncloud-security/\ncloud-security/hm-prowler/\ncomputational-fluid-dynamics/\ncomputational-fluid-dynamics/matlab/\ncomputational-fluid-dynamics/openfoam/\ncomputer-vision/\ncomputer-vision/hm-open3d/\ncomputer-vision/hm-pyvista/\ndata-analytics/\ndata-analytics/hm-geopandas/\ndata-distribution-service/\ndata-distribution-service/dummy_test.py\ndata-distribution-service/hm_message.idl\ndata-distribution-service/hm_message.xml\ndata-distribution-service/Makefile\ndata-distribution-service/poetry.lock\ndata-distribution-service/poetry.toml\ndata-distribution-service/publish.py\ndata-ingestion/\ndata-orchestration/\ndata-processing/\ndata-storage/\ndata-transformation/\ndata-visualization/\ndesktop-qt/\nembedded/\nethereum/\ngit/\ngolang-migrate/\nhardware-in-the-loop/\nhasura-graphql-engine/\nhigh-performance-computing/\nhm-alpine/\nhm-kafka/\nhm-locust/\nhm-rust/\nhm-traefik/\nhm-xxhash/\nkubernetes/\nmachine-learning/\nmatlab/\nmobile/\nnetwork-programmability/\noperating-system/\nparallel-computing/\nphysics/\nquantum-computing/\nrclone/\nrestic/\nreverse-engineering/\nrobotics/\nsubmodules/\ntrino/\nvagrant/\nvalgrind/\nvhdl/\nvim/\nweb/\nweb-cypress/\nwireless-network/\n\n(File list truncated. Use list_files on specific subdirectories if you need to explore further.)\n</environment_details>"
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "text": """<thinking>\nThe user is asking about a specific file: main.py. Based on the environment details provided, this file is located in the computer-vision/hm-open3d/src/ directory and is currently open in a VSCode tab.\n\nTo answer the question of what this file is, the most relevant tool would be the read_file tool. This will allow me to examine the contents of main.py to determine its purpose.\n\nThe read_file tool requires the "path" parameter. I can infer this path based on the environment details:\npath: "computer-vision/hm-open3d/src/main.py"\n\nSince I have the necessary parameter, I can proceed with calling the read_file tool.\n</thinking>"""
                    },
                    {
                        "toolUse": {
                            "input": {"path": "computer-vision/hm-open3d/src/main.py"},
                            "name": "read_file",
                            "toolUseId": "tooluse_qCt-KEyWQlWiyHl26spQVA",
                        }
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "content": [
                                {
                                    "text": 'import numpy as np\nimport open3d as o3d\n\n\ndef main():\n    ply_point_cloud = o3d.data.PLYPointCloud()\n    pcd = o3d.io.read_point_cloud(ply_point_cloud.path)\n    print(pcd)\n    print(np.asarray(pcd.points))\n\n    demo_crop_data = o3d.data.DemoCropPointCloud()\n    vol = o3d.visualization.read_selection_polygon_volume(\n        demo_crop_data.cropped_json_path\n    )\n    chair = vol.crop_point_cloud(pcd)\n\n    dists = pcd.compute_point_cloud_distance(chair)\n    dists = np.asarray(dists)\n    idx = np.where(dists > 0.01)[0]\n    pcd_without_chair = pcd.select_by_index(idx)\n\n    axis_aligned_bounding_box = chair.get_axis_aligned_bounding_box()\n    axis_aligned_bounding_box.color = (1, 0, 0)\n\n    oriented_bounding_box = chair.get_oriented_bounding_box()\n    oriented_bounding_box.color = (0, 1, 0)\n\n    o3d.visualization.draw_geometries(\n        [pcd_without_chair, chair, axis_aligned_bounding_box, oriented_bounding_box],\n        zoom=0.3412,\n        front=[0.4, -0.2, -0.9],\n        lookat=[2.6, 2.0, 1.5],\n        up=[-0.10, -1.0, 0.2],\n    )\n\n\nif __name__ == "__main__":\n    main()\n'
                                }
                            ],
                            "toolUseId": "tooluse_qCt-KEyWQlWiyHl26spQVA",
                        }
                    }
                ],
            },
            {"role": "assistant", "content": [{"text": "Please continue."}]},
            {
                "role": "user",
                "content": [
                    {
                        "text": "<environment_details>\n# VSCode Visible Files\ncomputer-vision/hm-open3d/src/main.py\n\n# VSCode Open Tabs\ncomputer-vision/hm-open3d/src/main.py\n</environment_details>"
                    }
                ],
            },
        ]
        assert transformed_messages == expected_messages
    else:
        with pytest.raises(Exception) as e:
            litellm.completion(**data)
        assert "litellm.modify_params" in str(e.value)


def test_bedrock_context_window_error():
    with pytest.raises(litellm.ContextWindowExceededError) as e:
        litellm.completion(
            model="bedrock/claude-3-5-sonnet-20240620",
            messages=[{"role": "user", "content": "Hello, world!"}],
            mock_response=Exception("prompt is too long"),
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


def test_bedrock_base_model_helper():
    from litellm.llms.bedrock.common_utils import BedrockModelInfo

    model = "us.amazon.nova-pro-v1:0"
    base_model = BedrockModelInfo.get_base_model(model)
    assert base_model == "amazon.nova-pro-v1:0"

    assert (
        BedrockModelInfo.get_base_model(
            "invoke/anthropic.claude-3-5-sonnet-20241022-v2:0"
        )
        == "anthropic.claude-3-5-sonnet-20241022-v2:0"
    )


@pytest.mark.parametrize(
    "model,expected_route",
    [
        # Test explicit route prefixes
        ("invoke/anthropic.claude-3-sonnet-20240229-v1:0", "invoke"),
        ("converse/anthropic.claude-3-sonnet-20240229-v1:0", "converse"),
        ("converse_like/anthropic.claude-3-sonnet-20240229-v1:0", "converse_like"),
        # Test models in BEDROCK_CONVERSE_MODELS list
        ("anthropic.claude-3-5-haiku-20241022-v1:0", "converse"),
        ("anthropic.claude-v2", "converse"),
        ("meta.llama3-70b-instruct-v1:0", "converse"),
        ("mistral.mistral-large-2407-v1:0", "converse"),
        # Test models with region prefixes
        ("us.anthropic.claude-3-sonnet-20240229-v1:0", "converse"),
        ("us.meta.llama3-70b-instruct-v1:0", "converse"),
        # Test default case (should return "invoke")
        ("amazon.titan-text-express-v1", "invoke"),
        ("cohere.command-text-v14", "invoke"),
        ("cohere.command-r-v1:0", "invoke"),
    ],
)
def test_bedrock_route_detection(model, expected_route):
    """Test all scenarios for BedrockModelInfo.get_bedrock_route"""
    from litellm.llms.bedrock.common_utils import BedrockModelInfo

    route = BedrockModelInfo.get_bedrock_route(model)
    assert (
        route == expected_route
    ), f"Expected route '{expected_route}' for model '{model}', but got '{route}'"


@pytest.mark.parametrize(
    "messages, expected_cache_control",
    [
        (
            [  # test system prompt cache
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are an AI assistant tasked with analyzing legal documents.",
                        },
                        {
                            "type": "text",
                            "text": "Here is the full text of a complex legal agreement",
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": "what are the key terms and conditions in this agreement?",
                },
            ],
            True,
        ),
        (
            [  # test user prompt cache
                {
                    "role": "user",
                    "content": "what are the key terms and conditions in this agreement?",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            True,
        ),
    ],
)
def test_bedrock_prompt_caching_message(messages, expected_cache_control):
    import litellm
    import json

    transformed_messages = litellm.AmazonConverseConfig()._transform_request(
        model="bedrock/anthropic.claude-3-5-haiku-20241022-v1:0",
        messages=messages,
        optional_params={},
        litellm_params={},
    )
    if expected_cache_control:
        assert "cachePoint" in json.dumps(transformed_messages)
    else:
        assert "cachePoint" not in json.dumps(transformed_messages)


@pytest.mark.parametrize(
    "model, expected_supports_tool_call",
    [
        ("bedrock/us.amazon.nova-pro-v1:0", True),
        ("bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0", True),
        ("bedrock/mistral.mistral-7b-instruct-v0.1:0", True),
        ("bedrock/meta.llama3-1-8b-instruct:0", True),
        ("bedrock/meta.llama3-2-70b-instruct:0", True),
        ("bedrock/meta.llama3-3-70b-instruct-v1:0", True),
        ("bedrock/amazon.titan-embed-text-v1:0", False),
    ],
)
def test_bedrock_supports_tool_call(model, expected_supports_tool_call):
    supported_openai_params = (
        litellm.AmazonConverseConfig().get_supported_openai_params(model=model)
    )
    if expected_supports_tool_call:
        assert "tools" in supported_openai_params
    else:
        assert "tools" not in supported_openai_params


class TestBedrockConverseChatCrossRegion(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")
        litellm.add_known_models()
        return {
            "model": "bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        }

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        pass

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
        bedrock_model = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
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
            "model": "bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        }

    def get_base_completion_call_args_with_thinking(self) -> dict:
        return {
            "model": "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "thinking": {"type": "enabled", "budget_tokens": 16000},
        }


class TestBedrockConverseChatNormal(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")
        litellm.add_known_models()
        return {
            "model": "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            "aws_region_name": "us-east-1",
        }

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        pass


class TestBedrockConverseNovaTestSuite(BaseLLMChatTest):
    def get_base_completion_call_args(self) -> dict:
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")
        litellm.add_known_models()
        return {
            "model": "bedrock/us.amazon.nova-lite-v1:0",
            "aws_region_name": "us-east-1",
        }

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        pass

    def test_prompt_caching(self):
        """
        TODO: Ensure this test passes our base llm test suite
        """


class TestBedrockRerank(BaseLLMRerankTest):
    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.BEDROCK

    def get_base_rerank_call_args(self) -> dict:
        return {
            "model": "bedrock/arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0",
        }


class TestBedrockCohereRerank(BaseLLMRerankTest):
    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.BEDROCK

    def get_base_rerank_call_args(self) -> dict:
        return {
            "model": "bedrock/arn:aws:bedrock:us-west-2::foundation-model/cohere.rerank-v3-5:0",
        }


@pytest.mark.parametrize(
    "messages, continue_message_index",
    [
        (
            [
                {"role": "user", "content": [{"type": "text", "text": ""}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]},
            ],
            0,
        ),
        (
            [
                {"role": "user", "content": [{"type": "text", "text": "Hello!"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "   "}]},
            ],
            1,
        ),
    ],
)
def test_bedrock_empty_content_handling(messages, continue_message_index):
    """
    Test that empty content in messages is handled correctly with default messages
    """
    # Test with default behavior (modify_params=True)
    litellm.modify_params = True
    formatted_messages = _bedrock_converse_messages_pt(
        messages=messages,
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        llm_provider="bedrock",
    )
    print(formatted_messages)
    # Verify assistant message with default text was inserted
    assert formatted_messages[0]["role"] == "user"
    assert formatted_messages[1]["role"] == "assistant"
    assert (
        formatted_messages[continue_message_index]["content"][0]["text"]
        == "Please continue."
    )


def test_bedrock_custom_continue_message():
    """
    Test that custom continue messages are used when provided
    """
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Hello!"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "   "}]},
    ]

    custom_continue = {
        "role": "assistant",
        "content": [{"text": "Custom continue message", "type": "text"}],
    }

    formatted_messages = _bedrock_converse_messages_pt(
        messages=messages,
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        llm_provider="bedrock",
        assistant_continue_message=custom_continue,
    )

    # Verify custom message was used
    assert formatted_messages[1]["role"] == "assistant"
    assert formatted_messages[1]["content"][0]["text"] == "Custom continue message"


def test_bedrock_no_default_message():
    """
    Test that empty content is replaced with placeholder when modify_params=False.
    AWS Bedrock doesn't allow empty or whitespace-only text content.
    """
    messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "Hi again"},
        {"role": "assistant", "content": "Valid response"},
    ]

    litellm.modify_params = False
    formatted_messages = _bedrock_converse_messages_pt(
        messages=messages,
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        llm_provider="bedrock",
    )

    # Verify empty message is replaced with placeholder and valid message remains
    assistant_messages = [
        msg for msg in formatted_messages if msg["role"] == "assistant"
    ]
    assert len(assistant_messages) == 2
    assert assistant_messages[0]["content"][0]["text"] == "."
    assert assistant_messages[1]["content"][0]["text"] == "Valid response"


@pytest.mark.parametrize("top_k_param", ["top_k", "topK"])
def test_bedrock_nova_topk(top_k_param):
    litellm.set_verbose = True
    data = {
        "model": "bedrock/us.amazon.nova-pro-v1:0",
        "messages": [{"role": "user", "content": "Hello, world!"}],
        top_k_param: 10,
    }
    original_transform = litellm.AmazonConverseConfig()._transform_request
    captured_data = None

    def mock_transform(*args, **kwargs):
        nonlocal captured_data
        result = original_transform(*args, **kwargs)
        captured_data = result
        return result

    with patch(
        "litellm.AmazonConverseConfig._transform_request", side_effect=mock_transform
    ):
        litellm.completion(**data)

        # Assert that additionalRequestParameters exists and contains topK
        assert "additionalModelRequestFields" in captured_data
        assert "inferenceConfig" in captured_data["additionalModelRequestFields"]
        assert (
            captured_data["additionalModelRequestFields"]["inferenceConfig"]["topK"]
            == 10
        )


def test_bedrock_cross_region_inference(monkeypatch):
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.add_known_models()

    litellm.set_verbose = True
    client = HTTPHandler()

    with patch.object(client, "post") as mock_post:
        try:
            completion(
                model="bedrock/us.meta.llama3-3-70b-instruct-v1:0",
                messages=[{"role": "user", "content": "Hello, world!"}],
                client=client,
            )
        except Exception as e:
            print(e)

        assert (
            mock_post.call_args.kwargs["url"]
            == "https://bedrock-runtime.us-west-2.amazonaws.com/model/us.meta.llama3-3-70b-instruct-v1%3A0/converse"
        )


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


def test_bedrock_process_empty_text_blocks():
    from litellm.litellm_core_utils.prompt_templates.factory import (
        process_empty_text_blocks,
    )

    message = {
        "message": {"role": "assistant", "content": [{"type": "text", "text": "   "}]},
        "assistant_continue_message": None,
    }
    modified_message = process_empty_text_blocks(**message)
    assert modified_message["content"][0]["text"] == "Please continue."


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


class TestBedrockEmbedding(BaseLLMEmbeddingTest):
    def get_base_embedding_call_args(self) -> dict:
        return {
            "model": "bedrock/amazon.titan-embed-image-v1",
        }

    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.BEDROCK

    def test_bedrock_image_embedding_transformation(self):
        from litellm.llms.bedrock.embed.amazon_titan_multimodal_transformation import (
            AmazonTitanMultimodalEmbeddingG1Config,
        )

        args = {
            "input": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkBAMAAACCzIhnAAAAG1BMVEURAAD///+ln5/h39/Dv79qX18uHx+If39MPz9oMSdmAAAACXBIWXMAAA7EAAAOxAGVKw4bAAABB0lEQVRYhe2SzWrEIBCAh2A0jxEs4j6GLDS9hqWmV5Flt0cJS+lRwv742DXpEjY1kOZW6HwHFZnPmVEBEARBEARB/jd0KYA/bcUYbPrRLh6amXHJ/K+ypMoyUaGthILzw0l+xI0jsO7ZcmCcm4ILd+QuVYgpHOmDmz6jBeJImdcUCmeBqQpuqRIbVmQsLCrAalrGpfoEqEogqbLTWuXCPCo+Ki1XGqgQ+jVVuhB8bOaHkvmYuzm/b0KYLWwoK58oFqi6XfxQ4Uz7d6WeKpna6ytUs5e8betMcqAv5YPC5EZB2Lm9FIn0/VP6R58+/GEY1X1egVoZ/3bt/EqF6malgSAIgiDIH+QL41409QMY0LMAAAAASUVORK5CYII=",
            "inference_params": {},
        }

        transformed_request = (
            AmazonTitanMultimodalEmbeddingG1Config()._transform_request(**args)
        )
        transformed_request[
            "inputImage"
        ] == "iVBORw0KGgoAAAANSUhEUgAAAGQAAABkBAMAAACCzIhnAAAAG1BMVEURAAD///+ln5/h39/Dv79qX18uHx+If39MPz9oMSdmAAAACXBIWXMAAA7EAAAOxAGVKw4bAAABB0lEQVRYhe2SzWrEIBCAh2A0jxEs4j6GLDS9hqWmV5Flt0cJS+lRwv742DXpEjY1kOZW6HwHFZnPmVEBEARBEARB/jd0KYA/bcUYbPrRLh6amXHJ/K+ypMoyUaGthILzw0l+xI0jsO7ZcmCcm4ILd+QuVYgpHOmDmz6jBeJImdcUCmeBqQpuqRIbVmQsLCrAalrGpfoEqEogqbLTWuXCPCo+Ki1XGqgQ+jVVuhB8bOaHkvmYuzm/b0KYLWwoK58oFqi6XfxQ4Uz7d6WeKpna6ytUs5e8betMcqAv5YPC5EZB2Lm9FIn0/VP6R58+/GEY1X1egVoZ/3bt/EqF6malgSAIgiDIH+QL41409QMY0LMAAAAASUVORK5CYII="


@pytest.mark.asyncio
async def test_bedrock_image_url_sync_client():
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    import logging
    from litellm import verbose_logger

    verbose_logger.setLevel(level=logging.DEBUG)

    litellm._turn_on_debug()
    client = AsyncHTTPHandler()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                    },
                },
            ],
        }
    ]

    with patch.object(client, "post") as mock_post:
        try:
            await litellm.acompletion(
                model="bedrock/us.amazon.nova-pro-v1:0",
                messages=messages,
                client=client,
            )
        except Exception as e:
            print(e)
        mock_post.assert_called_once()


def test_bedrock_error_handling_streaming():
    from litellm.llms.bedrock.chat.invoke_handler import (
        AWSEventStreamDecoder,
        BedrockError,
    )
    from unittest.mock import patch, Mock

    event = Mock()
    event.to_response_dict = Mock(
        return_value={
            "status_code": 400,
            "headers": {
                ":exception-type": "serviceUnavailableException",
                ":content-type": "application/json",
                ":message-type": "exception",
            },
            "body": b'{"message":"Bedrock is unable to process your request."}',
        }
    )

    decoder = AWSEventStreamDecoder(
        model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
    )
    with pytest.raises(Exception) as e:
        decoder._parse_message_from_event(event)
    assert isinstance(e.value, BedrockError)
    assert "Bedrock is unable to process your request." in e.value.message
    assert e.value.status_code == 400


@pytest.mark.parametrize(
    "image_url",
    [
        "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        # "https://raw.githubusercontent.com/datasets/gdp/master/data/gdp.csv",
        "https://www.cmu.edu/blackboard/files/evaluate/tests-example.xls",
        "http://www.krishdholakia.com/",
        # "https://raw.githubusercontent.com/datasets/sample-data/master/README.txt", # invalid url
        "https://raw.githubusercontent.com/mdn/content/main/README.md",
    ],
)
@pytest.mark.flaky(retries=6, delay=2)
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


def test_bedrock_custom_proxy():
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()

    with patch.object(client, "post") as mock_post:
        try:
            response = completion(
                model="bedrock/converse_like/us.amazon.nova-pro-v1:0",
                messages=[{"content": "Tell me a joke", "role": "user"}],
                api_key="Token",
                client=client,
                api_base="https://some-api-url/models",
            )
        except Exception as e:
            print(e)
        print(mock_post.call_args.kwargs)
        mock_post.assert_called_once()
        assert mock_post.call_args.kwargs["url"] == "https://some-api-url/models"

        assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer Token"


def test_bedrock_custom_deepseek():
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    import json

    litellm._turn_on_debug()
    client = HTTPHandler()

    with patch.object(client, "post") as mock_post:
        # Mock the response
        mock_response = Mock()
        mock_response.text = json.dumps(
            {"generation": "Here's a joke...", "stop_reason": "stop"}
        )
        mock_response.status_code = 200
        # Add required response attributes
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_post.return_value = mock_response

        try:
            response = completion(
                model="bedrock/llama/arn:aws:bedrock:us-east-1:086734376398:imported-model/r4c4kewx2s0n",  # Updated to specify provider
                messages=[{"role": "user", "content": "Tell me a joke"}],
                max_tokens=100,
                client=client,
            )

            # Print request details
            print("\nRequest Details:")
            print(f"URL: {mock_post.call_args.kwargs['url']}")

            # Verify the URL
            assert (
                mock_post.call_args.kwargs["url"]
                == "https://bedrock-runtime.us-east-1.amazonaws.com/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A086734376398%3Aimported-model%2Fr4c4kewx2s0n/invoke"
            )

            # Verify the request body format
            request_body = json.loads(mock_post.call_args.kwargs["data"])
            print("request_body=", json.dumps(request_body, indent=4, default=str))
            assert "prompt" in request_body
            assert request_body["prompt"] == "Tell me a joke"

            # follows the llama spec
            assert request_body["max_gen_len"] == 100

        except Exception as e:
            print(f"Error: {str(e)}")
            raise e


@pytest.mark.parametrize(
    "model, expected_output",
    [
        ("bedrock/anthropic.claude-3-sonnet-20240229-v1:0", {"top_k": 3}),
        ("bedrock/converse/us.amazon.nova-pro-v1:0", {"inferenceConfig": {"topK": 3}}),
        ("bedrock/meta.llama3-70b-instruct-v1:0", {}),
    ],
)
def test_handle_top_k_value_helper(model, expected_output):
    assert (
        litellm.AmazonConverseConfig()._handle_top_k_value(model, {"topK": 3})
        == expected_output
    )
    assert (
        litellm.AmazonConverseConfig()._handle_top_k_value(model, {"top_k": 3})
        == expected_output
    )


@pytest.mark.parametrize(
    "model, expected_params",
    [
        ("bedrock/anthropic.claude-3-sonnet-20240229-v1:0", {"top_k": 2}),
        ("bedrock/converse/us.amazon.nova-pro-v1:0", {"inferenceConfig": {"topK": 2}}),
        ("bedrock/meta.llama3-70b-instruct-v1:0", {}),
        ("bedrock/mistral.mistral-7b-instruct-v0:2", {}),
    ],
)
def test_bedrock_top_k_param(model, expected_params):
    import json

    client = HTTPHandler()

    with patch.object(client, "post") as mock_post:
        mock_response = Mock()

        if "mistral" in model:
            mock_response.text = json.dumps(
                {"outputs": [{"text": "Here's a joke...", "stop_reason": "stop"}]}
            )
        else:
            mock_response.text = json.dumps(
                {
                    "output": {
                        "message": {
                            "role": "assistant",
                            "content": [{"text": "Here's a joke..."}],
                        }
                    },
                    "usage": {"inputTokens": 12, "outputTokens": 6, "totalTokens": 18},
                    "stopReason": "stop",
                }
            )

        mock_response.status_code = 200
        # Add required response attributes
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_post.return_value = mock_response

        litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Hello, world!"}],
            top_k=2,
            client=client,
        )
        data = json.loads(mock_post.call_args.kwargs["data"])
        if "mistral" in model:
            assert data["top_k"] == 2
        else:
            assert data["additionalModelRequestFields"] == expected_params


def test_bedrock_invoke_provider():
    assert (
        litellm.AmazonInvokeConfig().get_bedrock_invoke_provider(
            "bedrock/invoke/us.anthropic.claude-3-5-sonnet-20240620-v1:0"
        )
        == "anthropic"
    )
    assert (
        litellm.AmazonInvokeConfig().get_bedrock_invoke_provider(
            "bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0"
        )
        == "anthropic"
    )
    assert (
        litellm.AmazonInvokeConfig().get_bedrock_invoke_provider(
            "bedrock/llama/arn:aws:bedrock:us-east-1:086734376398:imported-model/r4c4kewx2s0n"
        )
        == "llama"
    )
    assert (
        litellm.AmazonInvokeConfig().get_bedrock_invoke_provider(
            "us.amazon.nova-pro-v1:0"
        )
        == "nova"
    )


def test_bedrock_description_param():
    from litellm import completion
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()

    with patch.object(client, "post") as mock_post:
        try:
            response = completion(
                model="bedrock/us.amazon.nova-pro-v1:0",
                messages=[
                    {"role": "user", "content": "What is the meaning of this poem?"}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "meaning_reasoning",
                        "description": "Find the meaning inside a poem",
                        "schema": {
                            "type": "object",
                            "properties": {"meaning": {"type": "string"}},
                        },
                    },
                },
                client=client,
            )
        except Exception as e:
            print(e)
        mock_post.assert_called_once()

        request_body = json.loads(mock_post.call_args.kwargs["data"])
        request_body_str = json.dumps(request_body, indent=4, default=str)
        print("request_body=", request_body_str)

        assert (
            "Find the meaning inside a poem" in request_body_str
        )  # assert description is passed


@pytest.mark.parametrize(
    "sync_mode",
    [
        True,
        False,
    ],
)
@pytest.mark.asyncio
async def test_bedrock_thinking_in_assistant_message(sync_mode):
    litellm._turn_on_debug()
    from litellm.llms.custom_httpx.http_handler import HTTPHandler, AsyncHTTPHandler

    if sync_mode:
        client = HTTPHandler()
    else:
        client = AsyncHTTPHandler()

    params = {
        "model": "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Alright, let's get started with resolving this issue about implementing `__len__` for the Pipeline class in scikit-learn.\n\n### Step 1: Explore the repository structure to understand it better\n\nLet's first explore the repository structure to get a better understanding of the codebase:\n\n\nls -la /workspace/scikit-learn__scikit-learn__0.21\n\n\nThis will give us an overview of the top-level directories and files. From there, we can delve deeper into relevant parts of the code.\n\nLet's look at the main organization of the repository first, then we'll locate the Pipeline class, understand how it's currently implemented, and then add the `__len__` method to it.\n\nNext, let's look at the pipeline module, as the issue mentions `pipe[:len(pipe)]`, indicating that there is indexing support for Pipeline, but it lacks a `__len__` method.\n\nSince the issue mentions that Pipeline supports indexing but not `__len__`, I'll check the Pipeline class to understand its implementation.\n\nThe Pipeline class is likely to be in sklearn/pipeline.py or within a pipeline directory. Let's look for it:\n\n\nfind /workspace/scikit-learn__scikit-learn__0.21 -name \"*pipeline*\" -type f | grep -v __pycache__ | grep -v \"\\.pyc$\"\n\n\nThis will help us locate the file containing the Pipeline class.\n\nOnce we've located the file, we'll examine it to understand:\n1. How indexing is currently implemented\n2. Where we need to add the `__len__` method\n3. How the length of a pipeline should be determined\n\nThen, we'll implement the `__len__` method, test our changes to ensure they fix the issue, and verify that our changes don't break existing functionality.\n\nLet's begin by exploring the repository.\n\n### Step 2: Reproduce the error\n\nAfter exploring the repository, I'll create a script to reproduce the error described in the issue. This script will use the code provided in the \"Steps/Code to Reproduce\" section of the issue description.\n\nThen I'll run the script to confirm the error occurs as described, which will help us understand what needs to be fixed.\n\n### Step 3: Fix the issue\n\nOnce I've understood the problem and located the relevant files, I'll modify the code to add the `__len__` method to the Pipeline class. This method should return the number of steps in the pipeline, which is consistent with what a user would expect when calling `len(pipe)`.\n\n### Step 4: Test the fix\n\nAfter implementing the fix, I'll rerun the reproduction script to verify that the error is resolved and `len(pipe)` works correctly.\n\n### Step 5: Consider edge cases\n\nI'll think about potential edge cases like:\n- Empty pipelines\n- Very long pipelines\n- Any other special cases where the `__len__` implementation might behave unexpectedly\n\nI'll add tests for these cases to ensure our implementation handles them correctly.\n\n### Step 6: Comprehensive testing\n\nI'll run the existing tests to ensure our changes don't break any existing functionality, and I'll also write additional tests to specifically test our new `__len__` method.\n\nLet's start with exploring the repository.\n\nLet's start by exploring the repository structure:\n\n\nls -la /workspace/scikit-learn__scikit-learn__0.21\n\n\nNow, let's find the Pipeline class:\n\n\nfind /workspace/scikit-learn__scikit-learn__0.21 -name \"*pipeline*\" -type f | grep -v __pycache__ | grep -v \"\\.pyc$\"\n\n\nBased on the repository structure, let's look at the pipeline implementation:\n\n\ncat /workspace/scikit-learn__scikit-learn__0.21/sklearn/pipeline.py\n\n\nLet's check if there are tests for the Pipeline class:\n\n\nfind /workspace/scikit-learn__scikit-learn__0.21 -name \"*pipeline*test*\" -type f | grep -v __pycache__ | grep -v \"\\.pyc$\"\n\n\nNow, let's create a script to reproduce the error. I'll create a file in the workspace directory and add the code from the issue description:\n\n\ncat > /workspace/reproduce_pipeline_len_error.py << 'EOF'\nfrom sklearn import svm\nfrom sklearn.datasets import samples_generator\nfrom sklearn.feature_selection import SelectKBest\nfrom sklearn.feature_selection import f_regression\nfrom sklearn.pipeline import Pipeline\n\n# generate some data to play with\nX, y = samples_generator.make_classification(\n    n_informative=5, n_redundant=0, random_state=42)\n\nanova_filter = SelectKBest(f_regression, k=5)\nclf = svm.SVC(kernel='linear')\npipe = Pipeline([('anova', anova_filter), ('svc', clf)])\n\nprint(\"Trying to get length of pipeline...\")\ntry:\n    length = len(pipe)\n    print(f\"Length of pipeline: {length}\")\nexcept Exception as e:\n    print(f\"Error: {type(e).__name__}: {e}\")\nEOF\n\n\nLet's run this to confirm the error:\n\n\npython /workspace/reproduce_pipeline_len_error.py\n\n\nNow, let's implement the `__len__` method in the Pipeline class. First, I'll examine the Pipeline class to understand how it's structured and where to add the `__len__` method.\n\nFrom the error we saw above and looking at the pipeline.py file, we should add a `__len__` method to the Pipeline class.\n\nLet's modify the pipeline.py file:\n\n\ncp /workspace/scikit-learn__scikit-learn__0.21/sklearn/pipeline.py /workspace/pipeline.py.backup\n\n\nNow, let's edit the file to add the `__len__` method:\n\n\nsed -n -e '/class Pipeline/,/^class /p' /workspace/scikit-learn__scikit-learn__0.21/sklearn/pipeline.py\n\n\nThis will show us the Pipeline class definition so we can understand where to add the `__len__` method.\n\nBased on what we see, I'll add the `__len__` method to the Pipeline class:\n\n\nsed -i -e '/def __getitem__.*:/a\\\n    def __len__(self):\\\n        \"\"\"Returns the length of the Pipeline.\\\n\\\n        Returns\\\n        -------\\\n        length : int\\\n            The number of steps in the pipeline.\\\n        \"\"\"\\\n        return len(self.steps)\n' /workspace/scikit-learn__scikit-learn__0.21/sklearn/pipeline.py\n\n\nLet's verify that the changes were applied correctly:\n\n\ngrep -A 10 \"__len__\" /workspace/scikit-learn__scikit-learn__0.21/sklearn/pipeline.py\n\n\nNow, let's run our script again to see if the error is fixed:\n\n\npython /workspace/reproduce_pipeline_len_error.py\n\n\nIf the error is fixed, let's test some edge cases by modifying our script:\n\n\ncat > /workspace/test_pipeline_len.py << 'EOF'\nfrom sklearn.pipeline import Pipeline\nfrom sklearn import svm\nfrom sklearn.feature_selection import SelectKBest, f_regression\nfrom sklearn.datasets import samples_generator\nfrom sklearn.preprocessing import StandardScaler\n\n# Test case 1: Normal pipeline\nanova_filter = SelectKBest(f_regression, k=5)\nclf = svm.SVC(kernel='linear')\npipe = Pipeline([('anova', anova_filter), ('svc', clf)])\nprint(f\"Length of normal pipeline: {len(pipe)}\")\n\n# Test case 2: Empty pipeline\nempty_pipe = Pipeline([])\nprint(f\"Length of empty pipeline: {len(empty_pipe)}\")\n\n# Test case 3: Long pipeline\nX, y = samples_generator.make_classification(\n    n_informative=5, n_redundant=0, random_state=42)\n\nsteps = []\nfor i in range(10):\n    steps.append((f'scaler_{i}', StandardScaler()))\nlong_pipe = Pipeline(steps)\nprint(f\"Length of long pipeline: {len(long_pipe)}\")\n\n# Test case 4: Slicing with len\npipe = Pipeline([('anova', anova_filter), ('svc', clf)])\nsliced_pipe = pipe[:len(pipe)]\nprint(f\"Original pipe steps: {[name for name, _ in pipe.steps]}\")\nprint(f\"Sliced pipe steps: {[name for name, _ in sliced_pipe.steps]}\")\nEOF\n\n\nLet's run this test script:\n\n\npython /workspace/test_pipeline_len.py\n\n\nLet's also run the tests for the pipeline module to make sure we haven't broken anything:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest sklearn/tests/test_pipeline.py -v\n\n\nAfter testing the changes, let's update our script once more to test the edge case where we use `len(pipe)` in slicing:\n\n\ncat >> /workspace/test_pipeline_len.py << 'EOF'\n\n# Test case 5: Various slicing operations with len\npipe = Pipeline([\n    ('step1', StandardScaler()),\n    ('step2', SelectKBest(f_regression, k=5)),\n    ('step3', svm.SVC(kernel='linear'))\n])\nprint(f\"\\nOriginal pipe length: {len(pipe)}\")\n\n# Slice from beginning to end\nslice1 = pipe[:len(pipe)]\nprint(f\"pipe[:len(pipe)] steps: {[name for name, _ in slice1.steps]}\")\n\n# Slice from beginning to middle\nslice2 = pipe[:len(pipe)//2]\nprint(f\"pipe[:len(pipe)//2] steps: {[name for name, _ in slice2.steps]}\")\n\n# Slice from middle to end\nslice3 = pipe[len(pipe)//2:]\nprint(f\"pipe[len(pipe)//2:] steps: {[name for name, _ in slice3.steps]}\")\n\n# Slice a subset in the middle\nif len(pipe) >= 3:\n    slice4 = pipe[1:len(pipe)-1]\n    print(f\"pipe[1:len(pipe)-1] steps: {[name for name, _ in slice4.steps]}\")\nEOF\n\n\nLet's run this updated script:\n\n\npython /workspace/test_pipeline_len.py\n\n\nLet's also check all the tests in the sklearn directory to ensure our changes don't break anything:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest -xvs sklearn/tests/test_pipeline.py\n\n\nBefore we conclude, let's verify that our changes to scikit-learn match the expected functionality:\n\n1. We have implemented the `__len__` method in the Pipeline class\n2. The method returns the number of steps in the pipeline\n3. We have tested that it works for normal, empty, and long pipelines\n4. We have also tested that slicing operations using `len(pipe)` work correctly\n\nSince the issue description mentioned that the repository already supports indexing but not `__len__`, we didn't have to implement indexing. We only needed to add the `__len__` method to make `len(pipe)` work, which we've done.\n\nOur changes are minimal and focused on solving the specific issue described.\n\nFurthermore, to ensure we've done a thorough job, let's run all sklearn tests to make sure we haven't broken anything:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest sklearn/tests/\n\n\nLet's check the full implementation of the Pipeline to understand where `__len__` fits in compared to other methods like `__getitem__`:\n\n\nhead -n 300 /workspace/scikit-learn__scikit-learn__0.21/sklearn/pipeline.py\n\n\nLet's more specifically check the surrounding methods to ensure our addition fits well with the rest of the class:\n\n\ngrep -A 20 \"__getitem__\" /workspace/scikit-learn__scikit-learn__0.21/sklearn/pipeline.py\n\n\nNow, let's also test our changes against the base commit a62775e99f2a5ea3d51db7160fad783f6cd8a4c5, as mentioned in the requirements:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\ngit status\n\n\nHmm, we don't have access to the git repository directly. Let's focus on thoroughly testing our changes then.\n\nLet's run the full test suite for the sklearn module to make sure our changes don't break any existing functionality:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest -xvs sklearn/tests/test_pipeline.py\n\n\nGreat, now let's create a more comprehensive test script to ensure our `__len__` implementation works in various scenarios:\n\n\ncat > /workspace/comprehensive_pipeline_len_test.py << 'EOF'\n\"\"\"\nComprehensive tests for the __len__ method of Pipeline.\n\"\"\"\nfrom sklearn.pipeline import Pipeline, FeatureUnion\nfrom sklearn import svm\nfrom sklearn.feature_selection import SelectKBest, f_regression\nfrom sklearn.datasets import samples_generator\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.decomposition import PCA\n\n# Test case 1: Normal pipeline\nprint(\"Test case 1: Normal pipeline\")\nanova_filter = SelectKBest(f_regression, k=5)\nclf = svm.SVC(kernel='linear')\npipe = Pipeline([('anova', anova_filter), ('svc', clf)])\nprint(f\"Length of normal pipeline: {len(pipe)}\")\nassert len(pipe) == 2, \"Length of normal pipeline should be 2\"\n\n# Test case 2: Empty pipeline\nprint(\"\\nTest case 2: Empty pipeline\")\nempty_pipe = Pipeline([])\nprint(f\"Length of empty pipeline: {len(empty_pipe)}\")\nassert len(empty_pipe) == 0, \"Length of empty pipeline should be 0\"\n\n# Test case 3: Long pipeline\nprint(\"\\nTest case 3: Long pipeline\")\nX, y = samples_generator.make_classification(\n    n_informative=5, n_redundant=0, random_state=42)\n\nsteps = []\nfor i in range(10):\n    steps.append((f'scaler_{i}', StandardScaler()))\nlong_pipe = Pipeline(steps)\nprint(f\"Length of long pipeline: {len(long_pipe)}\")\nassert len(long_pipe) == 10, \"Length of long pipeline should be 10\"\n\n# Test case 4: Pipeline with FeatureUnion\nprint(\"\\nTest case 4: Pipeline with FeatureUnion\")\nunion = FeatureUnion([\n    ('pca', PCA(n_components=1)),\n    ('select', SelectKBest(k=1))\n])\npipe_with_union = Pipeline([\n    ('scaler', StandardScaler()),\n    ('union', union),\n    ('svc', svm.SVC(kernel='linear'))\n])\nprint(f\"Length of pipeline with FeatureUnion: {len(pipe_with_union)}\")\nassert len(pipe_with_union) == 3, \"Length of pipeline with FeatureUnion should be 3\"\n\n# Test case 5: Various slicing operations with len\nprint(\"\\nTest case 5: Various slicing operations with len\")\npipe = Pipeline([\n    ('step1', StandardScaler()),\n    ('step2', SelectKBest(f_regression, k=5)),\n    ('step3', svm.SVC(kernel='linear'))\n])\nprint(f\"Original pipe length: {len(pipe)}\")\nassert len(pipe) == 3, \"Original pipe length should be 3\"\n\n# Slice from beginning to end\nslice1 = pipe[:len(pipe)]\nprint(f\"pipe[:len(pipe)] steps: {[name for name, _ in slice1.steps]}\")\nassert len(slice1) == 3, \"Length of pipe[:len(pipe)] should be 3\"\n\n# Slice from beginning to middle\nslice2 = pipe[:len(pipe)//2]\nprint(f\"pipe[:len(pipe)//2] steps: {[name for name, _ in slice2.steps]}\")\nassert len(slice2) == 1, \"Length of pipe[:len(pipe)//2] should be 1\"\n\n# Slice from middle to end\nslice3 = pipe[len(pipe)//2:]\nprint(f\"pipe[len(pipe)//2:] steps: {[name for name, _ in slice3.steps]}\")\nassert len(slice3) == 2, \"Length of pipe[len(pipe)//2:] should be 2\"\n\n# Slice a subset in the middle\nslice4 = pipe[1:len(pipe)-1]\nprint(f\"pipe[1:len(pipe)-1] steps: {[name for name, _ in slice4.steps]}\")\nassert len(slice4) == 1, \"Length of pipe[1:len(pipe)-1] should be 1\"\n\nprint(\"\\nAll tests passed!\")\nEOF\n\n\nLet's run this comprehensive test:\n\n\npython /workspace/comprehensive_pipeline_len_test.py\n\n\nLet's also run specific tests from the scikit-learn test suite that relate to Pipeline indexing to ensure our changes integrate well:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest -xvs sklearn/tests/test_pipeline.py::test_pipeline_indexing\n\n\nOh, there might not be a specific test named `test_pipeline_indexing`. Let's look at the available tests in the test_pipeline.py file:\n\n\ngrep -r \"def test_\" /workspace/scikit-learn__scikit-learn__0.21/sklearn/tests/test_pipeline.py\n\n\nLet's run the tests that seem related to pipeline indexing or slicing:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest -xvs sklearn/tests/test_pipeline.py::test_pipeline_indexing || python -m pytest -xvs sklearn/tests/test_pipeline.py | grep -i \"index\"\n\n\nSince we might not have direct access to specific test names, let's examine the pipeline test file to identify relevant tests:\n\n\ncat /workspace/scikit-learn__scikit-learn__0.21/sklearn/tests/test_pipeline.py | grep -i \"index\\|slicing\\|getitem\"\n\n\nBased on our findings, let's run specific tests that are relevant to our changes:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest -xvs sklearn/tests/test_pipeline.py::test_pipeline_indexing || python -m pytest -xvs sklearn/tests/test_pipeline.py\n\n\nLet's check if the test_pipeline.py file contains any tests for slicing functionality:\n\n\ncat /workspace/scikit-learn__scikit-learn__0.21/sklearn/tests/test_pipeline.py | grep -A 20 -B 20 \"slicing\\|__getitem__\"\n\n\nLet's modify our comprehensive test script to include more edge cases and ensure it aligns with any existing tests:\n\n\ncat >> /workspace/comprehensive_pipeline_len_test.py << 'EOF'\n\n# Test case 6: Testing on pipeline with make_pipeline\nprint(\"\\nTest case 6: Testing on pipeline with make_pipeline\")\nfrom sklearn.pipeline import make_pipeline\n\npipe = make_pipeline(StandardScaler(), PCA(n_components=2), SelectKBest(k=1))\nprint(f\"Length of make_pipeline: {len(pipe)}\")\nassert len(pipe) == 3, \"Length of make_pipeline should be 3\"\n\n# Test case 7: Testing on nested pipelines\nprint(\"\\nTest case 7: Testing on nested pipelines\")\ninner_pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=2))])\nouter_pipe = Pipeline([('inner', inner_pipe), ('svc', svm.SVC())])\nprint(f\"Length of outer pipeline: {len(outer_pipe)}\")\nassert len(outer_pipe) == 2, \"Length of outer pipeline should be 2\"\n\n# Test case 8: Testing __len__ with negative indexing\nprint(\"\\nTest case 8: Testing __len__ with negative indexing\")\npipe = Pipeline([\n    ('step1', StandardScaler()),\n    ('step2', PCA(n_components=2)),\n    ('step3', SelectKBest(k=1))\n])\nneg_slice = pipe[-len(pipe):]\nprint(f\"pipe[-len(pipe):] steps: {[name for name, _ in neg_slice.steps]}\")\nassert len(neg_slice) == 3, \"Length of pipe[-len(pipe):] should be 3\"\n\nneg_slice2 = pipe[:-len(pipe)]\nprint(f\"pipe[:-len(pipe)] steps: {[name for name, _ in neg_slice2.steps]}\")\nassert len(neg_slice2) == 0, \"Length of pipe[:-len(pipe)] should be 0\"\n\n# Test case 9: Testing complex slicing expressions with len\nprint(\"\\nTest case 9: Testing complex slicing expressions with len\")\npipe = Pipeline([\n    ('step1', StandardScaler()),\n    ('step2', PCA(n_components=2)),\n    ('step3', SelectKBest(k=1)),\n    ('step4', svm.SVC())\n])\ncomplex_slice = pipe[len(pipe)//4:len(pipe)//4*3]\nprint(f\"pipe[len(pipe)//4:len(pipe)//4*3] steps: {[name for name, _ in complex_slice.steps]}\")\nassert len(complex_slice) == 2, \"Length of complex slice should be 2\"\n\nprint(\"\\nAll additional tests passed!\")\nEOF\n\n\nLet's run the updated comprehensive test:\n\n\npython /workspace/comprehensive_pipeline_len_test.py\n\n\nLet's also verify that our implementation doesn't cause any regressions in the scikit-learn test suite:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest -xvs sklearn/tests/test_pipeline.py\n\n\nNow, let's also run another specific test related to pipeline functionality:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest -xvs sklearn/tests/test_metaestimators.py | grep -i \"pipeline\"\n\n\nLet's get a deeper understanding of how the Pipeline class is used in tests by examining test files containing pipeline tests:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\ngrep -r \"Pipeline\" --include=\"*.py\" sklearn/tests/ | head -n 20\n\n\nLet's also look for any tests related to pipeline length or slicing:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\ngrep -r \"len(\" --include=\"*.py\" sklearn/tests/ | grep -i \"pipeline\" | head -n 20\n\n\nWith all our tests passing, it appears we've successfully implemented the `__len__` method for the Pipeline class, and it behaves as expected in various scenarios.\n\nLet's summarize what we've done:\n\n1. We located the Pipeline class in sklearn/pipeline.py\n2. We identified that it was missing a `__len__` method, which was causing the error described in the issue\n3. We implemented the `__len__` method to return the length of the pipeline's steps list\n4. We verified our fix with a reproduction script and comprehensive tests\n5. We confirmed that existing scikit-learn tests still pass with our changes\n\nOur implementation is minimal and focused on the specific issue, adding only the `__len__` method without changing any existing functionality.\n\nAt this point we can conclude that the issue has been successfully resolved. The Pipeline class now supports `len(pipe)`, which allows expressions like `pipe[:len(pipe)]` to work correctly.\n\nLet's make sure we've run the specific test for pipeline indexing:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\npython -m pytest -xvs sklearn/tests/test_pipeline.py | grep -i \"indexing\"\n\n\nLet's also check if there are any tests related to `__len__` or length in the pipeline tests:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\ngrep -r \"__len__\\|len(\" --include=\"*.py\" sklearn/tests/test_pipeline.py\n\n\nWith all our checks and tests, we're confident that our implementation of the `__len__` method in the Pipeline class correctly resolves the issue described. The method returns the number of steps in the pipeline, which is the expected behavior for `len(pipe)`.\n\nI believe our solution is complete and correct. The changes we made are minimal and focused on the specific issue, and we've thoroughly tested them to ensure they work as expected in various scenarios.\n\nLet's complete the task by running our comprehensive test one more time:\n\n\npython /workspace/comprehensive_pipeline_len_test.py\n\n\nI think we're good to go. Our implementation of the `__len__` method in the Pipeline class resolves the issue described, allowing expressions like `pipe[:len(pipe)]` to work correctly.\n\nLet's do a final check on the changes we made. The diff should show just the addition of the `__len__` method to the Pipeline class:\n\n\ncd /workspace/scikit-learn__scikit-learn__0.21\ndiff /workspace/pipeline.py.backup sklearn/pipeline.py\n\n\nWith all our checks and tests passing, we've successfully resolved the issue described in the problem statement.\n\nAlright, let's summarize our solution once more:\n\nThe issue was that Pipeline in scikit-learn didn't implement the `__len__` method, which was causing `pipe[:len(pipe)]` to raise an error, even though Pipeline supported indexing.\n\nTo resolve this, we added a `__len__` method to the Pipeline class that returns the number of steps in the pipeline. This allows expressions like `len(pipe)` and `pipe[:len(pipe)]` to work correctly, providing a more consistent and intuitive API for users.\n\nWe thoroughly tested our solution with:\n1. A script to reproduce the original error and confirm our fix\n2. Comprehensive tests for various scenarios, including edge cases\n3. Running the existing scikit-learn test suite to ensure no regressions\n\nThese tests confirm that our solution correctly resolves the issue while maintaining compatibility with the rest of the codebase.\n\nAnd that concludes our implementation.",
                        "signature": "ErUBCkYIARgCIkDnKEqW20HOxB9jS85ReLznKuKSV88wnAJf/M+y9ge7U+GOWAHd/trgmvpgHpJmOBkeUWUftR5g+81TNmAsDd2EEgxQHv0uMQBMvelCR+saDFvnSSfaNOj3NTVYpyIwG1ddKj5AVnXe382HERyVchmOTdn1ebogEFpXlRE66Mckq57xLSygusq6MwAV/6RvKh28HEuwcXNQn+BPIxQDVCwGGUh5g4fpZ6IgGk3mRg==",
                    },
                    {
                        "type": "text",
                        "text": "I'll help you implement the `__len__` method for Pipeline in scikit-learn to fix the issue described. Let's start by exploring the repository and understanding the problem better.\n\n## Step 1: Explore the repository structure\n\nLet's first look at the repository structure:",
                    },
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "Who do you know?"}]},
        ],
        "max_tokens": 32768,
        "thinking": {"type": "enabled", "budget_tokens": 30720},
    }

    with patch.object(client, "post") as mock_post:
        try:
            if sync_mode:
                response = litellm.completion(**params, client=client)
            else:
                response = await litellm.acompletion(**params, client=client)
        except Exception as e:
            print(e)

        mock_post.assert_called_once()

        print(mock_post.call_args.kwargs)
        json_data = mock_post.call_args.kwargs["data"]

        assert (
            "Alright, let's get started with resolving this issue about implementing"
            in json_data
        )


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
        model="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
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


def test_bedrock_application_inference_profile():
    from litellm.llms.custom_httpx.http_handler import HTTPHandler, AsyncHTTPHandler

    client = HTTPHandler()
    client2 = HTTPHandler()

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

    with patch.object(client, "post") as mock_post, patch.object(
        client2, "post"
    ) as mock_post2:
        try:
            resp = completion(
                model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                model_id="arn:aws:bedrock:eu-central-1:000000000000:application-inference-profile/a0a0a0a0a0a0",
                client=client,
                tools=tools,
            )
        except Exception as e:
            print(e)

        try:
            resp = completion(
                model="bedrock/converse/arn:aws:bedrock:eu-central-1:000000000000:application-inference-profile/a0a0a0a0a0a0",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                client=client2,
                tools=tools,
            )
        except Exception as e:
            print(e)

        mock_post.assert_called_once()
        mock_post2.assert_called_once()
        print(mock_post.call_args.kwargs)
        json_data = mock_post.call_args.kwargs["data"]
        assert mock_post.call_args.kwargs["url"].startswith(
            "https://bedrock-runtime.eu-central-1.amazonaws.com/"
        )
        assert mock_post2.call_args.kwargs["url"] == mock_post.call_args.kwargs["url"]


def return_mocked_response(model: str):
    if model == "bedrock/mistral.mistral-large-2407-v1:0":
        return {
            "metrics": {"latencyMs": 316},
            "output": {
                "message": {
                    "content": [{"text": "Hello! How are you doing today? How can"}],
                    "role": "assistant",
                }
            },
            "stopReason": "max_tokens",
            "usage": {"inputTokens": 5, "outputTokens": 10, "totalTokens": 15},
        }


@pytest.mark.parametrize(
    "model",
    [
        "bedrock/mistral.mistral-large-2407-v1:0",
    ],
)
@pytest.mark.asyncio()
async def test_bedrock_max_completion_tokens(model: str):
    """
    Tests that:
    - max_completion_tokens is passed as max_tokens to bedrock models
    """
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    litellm.set_verbose = True

    client = AsyncHTTPHandler()

    mock_response = return_mocked_response(model)
    _model = model.split("/")[1]
    print("\n\nmock_response: ", mock_response)

    with patch.object(client, "post") as mock_client:
        try:
            response = await litellm.acompletion(
                model=model,
                max_completion_tokens=10,
                messages=[{"role": "user", "content": "Hello!"}],
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_client.assert_called_once()
        request_body = json.loads(mock_client.call_args.kwargs["data"])

        print("request_body: ", request_body)

        assert request_body == {
            "messages": [{"role": "user", "content": [{"text": "Hello!"}]}],
            "additionalModelRequestFields": {},
            "system": [],
            "inferenceConfig": {"maxTokens": 10},
        }


def test_bedrock_meta_llama_function_calling():
    """
    Tests that:
    - meta llama models support function calling
    """
    from litellm.utils import return_raw_request
    from litellm.types.utils import CallTypes

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
    request_args = {
        "messages": messages,
        "tools": tools,
        "model": "bedrock/us.meta.llama4-scout-17b-instruct-v1:0",
    }

    response = return_raw_request(
        endpoint=CallTypes.completion,
        kwargs=request_args,
    )

    print(response)


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
            model="bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
            method="POST",
            endpoint="/model/us.anthropic.claude-3-5-sonnet-20240620-v1:0/invoke",
            data=data,
        )
    else:
        response = await litellm.allm_passthrough_route(
            model="bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
            method="POST",
            endpoint="/model/us.anthropic.claude-3-5-sonnet-20240620-v1:0/invoke",
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
                    "model": "bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
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
            model="bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
            method="POST",
            endpoint="/model/us.anthropic.claude-3-5-sonnet-20240620-v1:0/converse-stream",
            data=data,
        )
        async for chunk in response:
            print(chunk)

        await asyncio.sleep(1)

        mock_callback.assert_called_once()
        print(mock_callback.call_args.kwargs.keys())
        assert "response_cost" in mock_callback.call_args.kwargs["kwargs"]
        assert mock_callback.call_args.kwargs["kwargs"]["response_cost"] > 0
        assert "standard_logging_object" in mock_callback.call_args.kwargs["kwargs"]


@pytest.mark.asyncio
async def test_bedrock_streaming_passthrough_test2(monkeypatch):
    import litellm
    import time
    import asyncio
    from unittest.mock import MagicMock
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
            model="bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
            method="POST",
            endpoint="/model/us.anthropic.claude-3-5-sonnet-20240620-v1:0/invoke-with-response-stream",
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
    import time
    import asyncio
    from unittest.mock import MagicMock
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
            model="bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
            method="POST",
            endpoint="/model/us.anthropic.claude-3-5-sonnet-20240620-v1:0/invoke-with-response-stream",
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
