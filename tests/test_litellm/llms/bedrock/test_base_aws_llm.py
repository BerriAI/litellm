import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from botocore.credentials import Credentials
from botocore.awsrequest import AWSRequest, AWSPreparedRequest
import litellm
from litellm.llms.bedrock.base_aws_llm import (
    AwsAuthError,
    BaseAWSLLM,
    Boto3CredentialsInfo,
)
from litellm.caching.caching import DualCache

# Global variable for the base_aws_llm.py file path

BASE_AWS_LLM_PATH = os.path.join(
    os.path.dirname(__file__), "../../../../litellm/llms/bedrock/base_aws_llm.py"
)


def test_boto3_init_tracer_wrapping():
    """
    Test that all boto3 initializations are wrapped in tracer.trace or @tracer.wrap

    Ensures observability of boto3 calls in litellm.
    """
    # Get the source code of base_aws_llm.py
    with open(BASE_AWS_LLM_PATH, "r") as f:
        content = f.read()

    # List all boto3 initialization patterns we want to check
    boto3_init_patterns = ["boto3.client", "boto3.Session"]

    lines = content.split("\n")
    # Check each boto3 initialization is wrapped in tracer.trace
    for line_number, line in enumerate(lines, 1):
        for pattern in boto3_init_patterns:
            if pattern in line:
                # Look back up to 5 lines for decorator or trace block
                start_line = max(0, line_number - 5)
                context_lines = lines[start_line:line_number]

                has_trace = (
                    "tracer.trace" in line
                    or any("tracer.trace" in prev_line for prev_line in context_lines)
                    or any("@tracer.wrap" in prev_line for prev_line in context_lines)
                )

                if not has_trace:
                    print(f"\nContext for line {line_number}:")
                    for i, ctx_line in enumerate(context_lines, start=start_line + 1):
                        print(f"{i}: {ctx_line}")

                assert (
                    has_trace
                ), f"boto3 initialization '{pattern}' on line {line_number} is not wrapped with tracer.trace or @tracer.wrap"


def test_auth_functions_tracer_wrapping():
    """
    Test that all _auth functions in base_aws_llm.py are wrapped with @tracer.wrap

    Ensures observability of AWS authentication calls in litellm.
    """
    # Get the source code of base_aws_llm.py
    with open(BASE_AWS_LLM_PATH, "r") as f:
        content = f.read()

    lines = content.split("\n")
    # Check each line for _auth function definitions
    for line_number, line in enumerate(lines, 1):
        if line.strip().startswith("def _auth_"):
            # Look back up to 2 lines for the @tracer.wrap decorator
            start_line = max(0, line_number - 2)
            context_lines = lines[start_line:line_number]

            has_tracer_wrap = any(
                "@tracer.wrap" in prev_line for prev_line in context_lines
            )

            if not has_tracer_wrap:
                print(f"\nContext for line {line_number}:")
                for i, ctx_line in enumerate(context_lines, start=start_line + 1):
                    print(f"{i}: {ctx_line}")

            assert (
                has_tracer_wrap
            ), f"Auth function on line {line_number} is not wrapped with @tracer.wrap: {line.strip()}"


def test_get_aws_region_name_boto3_fallback():
    """
    Test the boto3 session fallback logic in _get_aws_region_name method.

    This tests the specific code block that tries to get the region from boto3.Session()
    when aws_region_name is None and not found in environment variables.
    """
    base_aws_llm = BaseAWSLLM()

    # Test case 1: boto3.Session() returns a configured region
    with patch("litellm.llms.bedrock.base_aws_llm.get_secret") as mock_get_secret:
        mock_get_secret.return_value = None  # No region in env vars

        with patch("boto3.Session") as mock_boto3_session:
            mock_session = MagicMock()
            mock_session.region_name = "us-east-1"
            mock_boto3_session.return_value = mock_session

            optional_params = {}
            result = base_aws_llm._get_aws_region_name(optional_params)

            assert result == "us-east-1"
            mock_boto3_session.assert_called_once()

    # Test case 2: boto3.Session() returns None for region (should default to us-west-2)
    with patch("litellm.llms.bedrock.base_aws_llm.get_secret") as mock_get_secret:
        mock_get_secret.return_value = None  # No region in env vars

        with patch("boto3.Session") as mock_boto3_session:
            mock_session = MagicMock()
            mock_session.region_name = None
            mock_boto3_session.return_value = mock_session

            optional_params = {}
            result = base_aws_llm._get_aws_region_name(optional_params)

            assert result == "us-west-2"
            mock_boto3_session.assert_called_once()

    # Test case 3: boto3 import/session creation raises exception (should default to us-west-2)
    with patch("litellm.llms.bedrock.base_aws_llm.get_secret") as mock_get_secret:
        mock_get_secret.return_value = None  # No region in env vars

        with patch("boto3.Session") as mock_boto3_session:
            mock_boto3_session.side_effect = Exception("boto3 not available")

            optional_params = {}
            result = base_aws_llm._get_aws_region_name(optional_params)

            assert result == "us-west-2"
            mock_boto3_session.assert_called_once()

    # Test case 4: aws_region_name is provided in optional_params (should not use boto3)
    with patch("boto3.Session") as mock_boto3_session:
        optional_params = {"aws_region_name": "eu-west-1"}
        result = base_aws_llm._get_aws_region_name(optional_params)

        assert result == "eu-west-1"
        mock_boto3_session.assert_not_called()

    # Test case 5: aws_region_name found in environment variables (should not use boto3)
    with patch("litellm.llms.bedrock.base_aws_llm.get_secret") as mock_get_secret:

        def side_effect(key, default=None):
            if key == "AWS_REGION_NAME":
                return "ap-southeast-1"
            return default

        mock_get_secret.side_effect = side_effect

        with patch("boto3.Session") as mock_boto3_session:
            optional_params = {}
            result = base_aws_llm._get_aws_region_name(optional_params)

            assert result == "ap-southeast-1"
            mock_boto3_session.assert_not_called()


def test_sign_request_with_env_var_bearer_token():
    # Create instance of actual class
    llm = BaseAWSLLM()

    # Test data
    service_name = "bedrock"
    headers = {"Custom-Header": "test"}
    optional_params = {}
    request_data = {"prompt": "test"}
    api_base = "https://api.example.com"

    # Mock environment variable
    with patch.dict(os.environ, {"AWS_BEARER_TOKEN_BEDROCK": "test_token"}):
        # Execute
        result_headers, result_body = llm._sign_request(
            service_name=service_name,
            headers=headers,
            optional_params=optional_params,
            request_data=request_data,
            api_base=api_base,
        )

        # Assert
        assert result_headers["Authorization"] == "Bearer test_token"
        assert result_headers["Content-Type"] == "application/json"
        assert result_headers["Custom-Header"] == "test"
        assert result_body == json.dumps(request_data).encode()


def test_sign_request_with_sigv4():
    llm = BaseAWSLLM()

    # Mock AWS credentials and SigV4 auth
    mock_credentials = Credentials("test_key", "test_secret", "test_token")
    mock_sigv4 = MagicMock()
    mock_request = MagicMock()
    mock_request.headers = {
        "Authorization": "AWS4-HMAC-SHA256 Credential=test",
        "Content-Type": "application/json",
    }
    mock_request.body = b'{"prompt": "test"}'

    # Test data
    service_name = "bedrock"
    headers = {"Custom-Header": "test"}
    optional_params = {
        "aws_access_key_id": "test_key",
        "aws_secret_access_key": "test_secret",
        "aws_region_name": "us-west-2",
    }
    request_data = {"prompt": "test"}
    api_base = "https://api.example.com"

    # Mock the necessary components
    with patch("botocore.auth.SigV4Auth", return_value=mock_sigv4), patch(
        "botocore.awsrequest.AWSRequest", return_value=mock_request
    ), patch.object(
        llm, "get_credentials", return_value=mock_credentials
    ), patch.object(
        llm, "_get_aws_region_name", return_value="us-west-2"
    ):
        result_headers, result_body = llm._sign_request(
            service_name=service_name,
            headers=headers,
            optional_params=optional_params,
            request_data=request_data,
            api_base=api_base,
        )

        # Assert
        assert "Authorization" in result_headers
        assert result_headers["Authorization"] != "Bearer test_token"
        assert result_headers["Content-Type"] == "application/json"
        assert result_body == mock_request.body


def test_sign_request_with_api_key_bearer_token():
    """
    Test that _sign_request uses the api_key parameter as a bearer token when provided
    """
    llm = BaseAWSLLM()

    # Test data
    service_name = "bedrock"
    headers = {"Custom-Header": "test"}
    optional_params = {}
    request_data = {"prompt": "test"}
    api_base = "https://api.example.com"
    api_key = "test_api_key"

    # Execute with api_key parameter
    result_headers, result_body = llm._sign_request(
        service_name=service_name,
        headers=headers,
        optional_params=optional_params,
        request_data=request_data,
        api_base=api_base,
        api_key=api_key,
    )

    # Assert
    assert result_headers["Authorization"] == f"Bearer {api_key}"
    assert result_headers["Content-Type"] == "application/json"
    assert result_headers["Custom-Header"] == "test"
    assert result_body == json.dumps(request_data).encode()


def test_get_request_headers_with_env_var_bearer_token():
    # Setup
    llm = BaseAWSLLM()
    credentials = Credentials("test_key", "test_secret", "test_token")
    headers = {"Content-Type": "application/json"}
    headers_dict = headers.copy()

    # Create mock request
    mock_prepared_request = MagicMock(spec=AWSPreparedRequest)
    mock_request = MagicMock(spec=AWSRequest)
    mock_request.headers = headers_dict
    mock_request.prepare.return_value = mock_prepared_request

    def mock_aws_request_init(method, url, data, headers):
        mock_request.headers.update(headers)
        return mock_request

    # Test with bearer token
    with patch.dict(os.environ, {"AWS_BEARER_TOKEN_BEDROCK": "test_token"}), patch(
        "botocore.awsrequest.AWSRequest", side_effect=mock_aws_request_init
    ):
        result = llm.get_request_headers(
            credentials=credentials,
            aws_region_name="us-west-2",
            extra_headers=None,
            endpoint_url="https://api.example.com",
            data='{"prompt": "test"}',
            headers=headers_dict,
        )

        # Assert
        assert mock_request.headers["Authorization"] == "Bearer test_token"
        assert result == mock_prepared_request


def test_get_request_headers_with_sigv4():
    # Setup
    llm = BaseAWSLLM()
    credentials = Credentials("test_key", "test_secret", "test_token")
    headers = {"Content-Type": "application/json"}

    # Create mock request and SigV4 instance
    mock_request = MagicMock(spec=AWSRequest)
    mock_request.headers = headers.copy()
    mock_request.prepare.return_value = MagicMock(spec=AWSPreparedRequest)

    mock_sigv4 = MagicMock()

    # Test without bearer token (should use SigV4)
    with patch.dict(os.environ, {}, clear=True), patch(
        "botocore.auth.SigV4Auth", return_value=mock_sigv4
    ) as mock_sigv4_class, patch(
        "botocore.awsrequest.AWSRequest", return_value=mock_request
    ):
        result = llm.get_request_headers(
            credentials=credentials,
            aws_region_name="us-west-2",
            extra_headers=None,
            endpoint_url="https://api.example.com",
            data='{"prompt": "test"}',
            headers=headers,
        )

        # Verify SigV4 authentication and result
        mock_sigv4_class.assert_called_once_with(credentials, "bedrock", "us-west-2")
        mock_sigv4.add_auth.assert_called_once_with(mock_request)
        assert result == mock_request.prepare.return_value


def test_get_request_headers_with_api_key_bearer_token():
    """
    Test that get_request_headers uses the api_key parameter as a bearer token when provided
    """
    # Setup
    llm = BaseAWSLLM()
    credentials = Credentials("test_key", "test_secret", "test_token")
    headers = {"Content-Type": "application/json"}
    headers_dict = headers.copy()
    api_key = "test_api_key"

    # Create mock request
    mock_prepared_request = MagicMock(spec=AWSPreparedRequest)
    mock_request = MagicMock(spec=AWSRequest)
    mock_request.headers = headers_dict
    mock_request.prepare.return_value = mock_prepared_request

    def mock_aws_request_init(method, url, data, headers):
        mock_request.headers.update(headers)
        return mock_request

    # Test with api_key parameter
    with patch.dict(os.environ, {}, clear=True), patch(
        "botocore.awsrequest.AWSRequest", side_effect=mock_aws_request_init
    ):
        result = llm.get_request_headers(
            credentials=credentials,
            aws_region_name="us-west-2",
            extra_headers=None,
            endpoint_url="https://api.example.com",
            data='{"prompt": "test"}',
            headers=headers_dict,
            api_key=api_key,
        )

        # Assert
        assert mock_request.headers["Authorization"] == f"Bearer {api_key}"
        assert result == mock_prepared_request


def test_role_assumption_without_session_name():
    """
    Test for issue 12583: Role assumption should work when only aws_role_name is provided
    without aws_session_name. The system should auto-generate a session name.
    """
    base_aws_llm = BaseAWSLLM()

    # Mock the boto3 STS client
    mock_sts_client = MagicMock()

    # Mock the STS response with proper expiration handling
    mock_expiry = MagicMock()
    mock_expiry.tzinfo = timezone.utc
    current_time = datetime.now(timezone.utc)
    # Create a timedelta object that returns 3600 when total_seconds() is called
    time_diff = MagicMock()
    time_diff.total_seconds.return_value = 3600
    mock_expiry.__sub__ = MagicMock(return_value=time_diff)

    mock_sts_response = {
        "Credentials": {
            "AccessKeyId": "assumed-access-key",
            "SecretAccessKey": "assumed-secret-key",
            "SessionToken": "assumed-session-token",
            "Expiration": mock_expiry,
        }
    }
    mock_sts_client.assume_role.return_value = mock_sts_response

    # Test case 1: aws_role_name provided without aws_session_name
    with patch("boto3.client", return_value=mock_sts_client):
        credentials = base_aws_llm.get_credentials(
            aws_role_name="arn:aws:iam::2222222222222:role/LitellmEvalBedrockRole"
        )

        # Verify assume_role was called
        mock_sts_client.assume_role.assert_called_once()

        # Check the call arguments
        call_args = mock_sts_client.assume_role.call_args
        assert (
            call_args[1]["RoleArn"]
            == "arn:aws:iam::2222222222222:role/LitellmEvalBedrockRole"
        )
        # Session name should be auto-generated with format "litellm-session-{timestamp}"
        assert call_args[1]["RoleSessionName"].startswith("litellm-session-")

        # Verify credentials are returned correctly
        assert isinstance(credentials, Credentials)
        assert credentials.access_key == "assumed-access-key"
        assert credentials.secret_key == "assumed-secret-key"
        assert credentials.token == "assumed-session-token"

    # Test case 2: Both aws_role_name and aws_session_name provided (existing behavior)
    mock_sts_client.reset_mock()
    with patch("boto3.client", return_value=mock_sts_client):
        credentials = base_aws_llm.get_credentials(
            aws_role_name="arn:aws:iam::2222222222222:role/LitellmEvalBedrockRole",
            aws_session_name="my-custom-session",
        )

        # Verify assume_role was called with custom session name
        mock_sts_client.assume_role.assert_called_once()
        call_args = mock_sts_client.assume_role.call_args
        assert call_args[1]["RoleSessionName"] == "my-custom-session"

    # Test case 3: Verify caching works with auto-generated session names
    # Clear the cache first
    base_aws_llm.iam_cache = DualCache()

    mock_sts_client.reset_mock()
    with patch("boto3.client", return_value=mock_sts_client):
        # First call
        credentials1 = base_aws_llm.get_credentials(
            aws_role_name="arn:aws:iam::2222222222222:role/LitellmEvalBedrockRole"
        )

        # Second call with same role should use cache (not call assume_role again)
        credentials2 = base_aws_llm.get_credentials(
            aws_role_name="arn:aws:iam::2222222222222:role/LitellmEvalBedrockRole"
        )

        # Should only be called once due to caching
        assert mock_sts_client.assume_role.call_count == 1
