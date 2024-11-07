# @pytest.mark.skip(reason="AWS Suspended Account")
import os
import sys
import traceback

from dotenv import load_dotenv

import litellm.types

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
from litellm.llms.bedrock.chat import BedrockLLM
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
                api_base="https://gateway.ai.cloudflare.com/v1/fa4cdcab1f32b95ca3b53fd36043d691/test/aws-bedrock/bedrock-runtime/us-east-1",
            )
        except Exception as e:
            pass

        print(f"mock_client_post.call_args.kwargs: {mock_client_post.call_args.kwargs}")
        assert (
            mock_client_post.call_args.kwargs["url"]
            == "https://gateway.ai.cloudflare.com/v1/fa4cdcab1f32b95ca3b53fd36043d691/test/aws-bedrock/bedrock-runtime/us-east-1/model/anthropic.claude-3-sonnet-20240229-v1:0/converse"
        )
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


from litellm.llms.prompt_templates.factory import _bedrock_converse_messages_pt


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

    from litellm.llms.base_aws_llm import BaseAWSLLM

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
    from litellm.llms.prompt_templates.factory import _bedrock_converse_messages_pt

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
        ),
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
