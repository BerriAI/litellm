# tests/llm_translation/test_base_aws_llm.py
import os
import json
import pytest
from unittest.mock import patch
from botocore.credentials import Credentials
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from unittest.mock import Mock
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

import json
import pytest
from unittest.mock import patch, Mock

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM


def test_bedrock_completion_with_region_name():
    litellm._turn_on_debug()
    client = HTTPHandler()

    with patch.object(client, "post") as mock_post:
        mock_response = Mock()
        # Construct a response similar to our other tests.
        mock_response.text = json.dumps(
            {
                "response_id": "379ed018/60744aff-e741-4aad-bd10-74639a4ade79",
                "text": "Hello! How's it going? I hope you're having a fantastic day!",
                "generation_id": "38709bb9-f20f-42d9-9c61-13a73b7bbc12",
                "chat_history": [
                    {"role": "USER", "message": "Hello, world!"},
                    {
                        "role": "CHATBOT",
                        "message": "Hello! How's it going? I hope you're having a fantastic day!",
                    },
                ],
                "finish_reason": "COMPLETE",
            }
        )
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_post.return_value = mock_response

        # Pass the client so that the HTTP call will be intercepted.
        response = litellm.completion(
            model="cohere.command-r-v1:0",
            messages=[{"role": "user", "content": "Hello, world!"}],
            aws_region_name="us-west-12",
            client=client,
        )

        # Ensure our post method has been called.
        mock_post.assert_called_once()

        assert (
            mock_post.call_args.kwargs["url"]
            == "https://bedrock-runtime.us-west-12.amazonaws.com/model/cohere.command-r-v1:0/invoke"
        )
        assert (
            mock_post.call_args.kwargs["data"]
            == json.dumps({"message": "Hello, world!", "chat_history": []}).encode(
                "utf-8"
            )
        )

        # Print the URL and body of the HTTP request.
        # assert request was signed with the correct region
        _authorization_header = mock_post.call_args.kwargs["headers"]["Authorization"]
        import re

        # Ensure the authorization header contains the exact region segment "us-west-12/bedrock/aws4_request"
        pattern = r"us-west-12/bedrock/aws4_request"
        assert re.search(pattern, _authorization_header) is not None


def test_bedrock_completion_with_dynamic_authentication_params():
    litellm._turn_on_debug()
    client = HTTPHandler()

    with patch.object(client, "post") as mock_post:
        mock_response = Mock()
        # Construct a response similar to our other tests.
        mock_response.text = json.dumps(
            {
                "response_id": "379ed018/60744aff-e741-4aad-bd10-74639a4ade79",
                "text": "Hello! How's it going? I hope you're having a fantastic day!",
                "generation_id": "38709bb9-f20f-42d9-9c61-13a73b7bbc12",
                "chat_history": [
                    {"role": "USER", "message": "Hello, world!"},
                    {
                        "role": "CHATBOT",
                        "message": "Hello! How's it going? I hope you're having a fantastic day!",
                    },
                ],
                "finish_reason": "COMPLETE",
            }
        )
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_post.return_value = mock_response

        # Pass the client so that the HTTP call will be intercepted.
        response = litellm.completion(
            model="cohere.command-r-v1:0",
            messages=[{"role": "user", "content": "Hello, world!"}],
            aws_access_key_id="dynamically_generated_access_key_id",
            aws_secret_access_key="dynamically_generated_secret_access_key",
            client=client,
        )

        # Ensure our post method has been called.
        mock_post.assert_called_once()
        import re

        # Get authorization header
        _authorization_header = mock_post.call_args.kwargs["headers"]["Authorization"]

        # Check for exact credential pattern
        pattern = r"AWS4-HMAC-SHA256 Credential=dynamically_generated_access_key_id/\d{8}/[a-z0-9-]+/bedrock/aws4_request"
        assert re.search(pattern, _authorization_header) is not None


def test_bedrock_completion_with_dynamic_bedrock_runtime_endpoint():
    litellm._turn_on_debug()
    client = HTTPHandler()

    with patch.object(client, "post") as mock_post:
        mock_response = Mock()
        # Construct a response similar to our other tests.
        mock_response.text = json.dumps(
            {
                "response_id": "379ed018/60744aff-e741-4aad-bd10-74639a4ade79",
                "text": "Hello! How's it going? I hope you're having a fantastic day!",
                "generation_id": "38709bb9-f20f-42d9-9c61-13a73b7bbc12",
                "chat_history": [
                    {"role": "USER", "message": "Hello, world!"},
                    {
                        "role": "CHATBOT",
                        "message": "Hello! How's it going? I hope you're having a fantastic day!",
                    },
                ],
                "finish_reason": "COMPLETE",
            }
        )
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_post.return_value = mock_response

        # Pass the client so that the HTTP call will be intercepted.
        response = litellm.completion(
            model="cohere.command-r-v1:0",
            messages=[{"role": "user", "content": "Hello, world!"}],
            aws_bedrock_runtime_endpoint="https://my-fake-endpoint.com",
            client=client,
        )

        # Ensure our post method has been called.
        mock_post.assert_called_once()
        assert (
            mock_post.call_args.kwargs["url"]
            == "https://my-fake-endpoint.com/model/cohere.command-r-v1:0/invoke"
        )


# ------------------------------------------------------------------------------
# A dummy credentials object to return from get_credentials.
# (It must have attributes so that SigV4Auth.add_auth doesn't break.)
# ------------------------------------------------------------------------------
class DummyCredentials:
    access_key = "dummy_access"
    secret_key = "dummy_secret"
    token = "dummy_token"


# ------------------------------------------------------------------------------
# This test makes sure that a given dynamic parameter is passed into the call
# to BaseAWSLLM.get_credentials. (Some dynamic params—for example aws_region_name
# or aws_bedrock_runtime_endpoint—are already covered by other tests.)
# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "model",
    [
        "bedrock/converse/cohere.command-r-v1:0",
        "cohere.command-r-v1:0",
        "bedrock/cohere.command-r-v1:0",
        "bedrock/invoke/cohere.command-r-v1:0",
    ],
)
@pytest.mark.parametrize(
    "param_name, param_value",
    [
        ("aws_session_token", "dummy_session_token"),
        ("aws_session_name", "dummy_session_name"),
        ("aws_profile_name", "dummy_profile_name"),
        ("aws_role_name", "dummy_role_name"),
        ("aws_web_identity_token", "dummy_web_identity_token"),
        ("aws_sts_endpoint", "dummy_sts_endpoint"),
        ("aws_external_id", "dummy_external_id"),
    ],
)
def test_dynamic_aws_params_propagation(model, param_name, param_value):
    """
    When passed to litellm.completion, each dynamic AWS authentication parameter
    should propagate down to the get_credentials() call in BaseAWSLLM.

    Also tests different model parameter values.
    """
    client = HTTPHandler()

    # Base parameters required for the completion call.
    # (We include aws_access_key_id and aws_secret_access_key so that the correct auth
    # branch in get_credentials() is reached.)
    base_params = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello, world!"}],
        "aws_access_key_id": "dummy_access",
        "aws_secret_access_key": "dummy_secret",
        "client": client,
    }
    # For parameters such as aws_role_name or aws_web_identity_token a session name is required.
    if param_name in ("aws_role_name", "aws_web_identity_token"):
        base_params["aws_session_name"] = "dummy_session_name"
        if param_name == "aws_web_identity_token":
            # The web identity branch also requires a role name.
            base_params["aws_role_name"] = "dummy_role_name"
    # Inject the dynamic parameter under test.
    base_params[param_name] = param_value

    # Patch SigV4Auth in the signing (so that no actual signing is done).
    with patch("botocore.auth.SigV4Auth", autospec=True) as mock_sigv4:
        instance = mock_sigv4.return_value
        instance.add_auth.return_value = None

        # Patch BaseAWSLLM.get_credentials so that we can capture its kwargs.
        def dummy_get_credentials(**kwargs):
            dummy_get_credentials.called_kwargs = kwargs  # type: ignore[attr-defined]
            return DummyCredentials()

        with patch.object(
            BaseAWSLLM, "get_credentials", side_effect=dummy_get_credentials
        ):
            # Patch the HTTP client's post method to avoid an actual HTTP call.
            with patch.object(client, "post") as mock_post:
                mock_response = Mock()
                mock_response.text = json.dumps(
                    {
                        "response_id": "dummy_response",
                        "text": "Hello! world",
                        "generation_id": "dummy_gen",
                        "chat_history": [],
                        "finish_reason": "COMPLETE",
                    }
                )
                if "converse" in model:
                    mock_response.text = json.dumps(
                        {
                            "output": {
                                "message": {
                                    "role": "assistant",
                                    "content": [{"text": "Here's a joke..."}],
                                }
                            },
                            "usage": {
                                "inputTokens": 12,
                                "outputTokens": 6,
                                "totalTokens": 18,
                            },
                            "stopReason": "stop",
                        }
                    )

                mock_response.status_code = 200
                mock_response.headers = {"Content-Type": "application/json"}
                mock_response.json = lambda: json.loads(mock_response.text)
                mock_post.return_value = mock_response

                # Call litellm.completion with our base & dynamic parameters.
                litellm.completion(**base_params)

                print(
                    "get_credentials.called_kwargs",
                    json.dumps(dummy_get_credentials.called_kwargs, indent=4),
                )

                # We now assert that get_credentials() was called with the dynamic param.
                assert (
                    dummy_get_credentials.called_kwargs.get(param_name) == param_value
                )
