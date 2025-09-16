import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


from datetime import datetime, timedelta, timezone
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


def test_cache_keys_are_different_for_different_roles():
    """
    Test that cache keys are different for different AWS roles.
    This ensures that credentials for different roles don't get mixed up.
    """
    base_aws_llm = BaseAWSLLM()
    
    # Create arguments for two different roles
    args1 = {
        "aws_access_key_id": None,
        "aws_secret_access_key": None,
        "aws_role_name": "arn:aws:iam::1111111111111:role/LitellmRole",
        "aws_session_name": "test-session-1"
    }
    
    args2 = {
        "aws_access_key_id": None,
        "aws_secret_access_key": None,
        "aws_role_name": "arn:aws:iam::2222222222222:role/LitellmEvalBedrockRole",
        "aws_session_name": "test-session-2"
    }
    
    # Generate cache keys
    cache_key1 = base_aws_llm.get_cache_key(args1)
    cache_key2 = base_aws_llm.get_cache_key(args2)
    
    # Cache keys should be different because the role names are different
    assert cache_key1 != cache_key2


def test_different_roles_without_session_names_should_not_share_cache():
    """
    Test that different roles with auto-generated session names don't share cache.
    This was the original issue where cache keys were the same for different roles.
    """
    base_aws_llm = BaseAWSLLM()
    
    # Create arguments for two different roles without session names
    args1 = {
        "aws_access_key_id": None,
        "aws_secret_access_key": None,
        "aws_role_name": "arn:aws:iam::1111111111111:role/LitellmRole",
        "aws_session_name": None
    }
    
    args2 = {
        "aws_access_key_id": None,
        "aws_secret_access_key": None,
        "aws_role_name": "arn:aws:iam::2222222222222:role/LitellmEvalBedrockRole",
        "aws_session_name": None
    }
    
    # Generate cache keys
    cache_key1 = base_aws_llm.get_cache_key(args1)
    cache_key2 = base_aws_llm.get_cache_key(args2)
    
    # Cache keys should be different because the role names are different
    assert cache_key1 != cache_key2


def test_eks_irsa_ambient_credentials_used():
    """
    Test that in EKS/IRSA environments, ambient credentials are used when no explicit keys provided.
    This allows web identity tokens to work automatically.
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
    
    with patch("boto3.client", return_value=mock_sts_client) as mock_boto3_client:
        
        # Call with no explicit credentials (EKS/IRSA scenario)
        credentials, ttl = base_aws_llm._auth_with_aws_role(
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_session_token=None,
            aws_role_name="arn:aws:iam::2222222222222:role/LitellmEvalBedrockRole",
            aws_session_name="test-session"
        )
        
        # Should create STS client without explicit credentials (using ambient credentials)
        mock_boto3_client.assert_called_once_with("sts")
        
        # Should call assume_role
        mock_sts_client.assume_role.assert_called_once_with(
            RoleArn="arn:aws:iam::2222222222222:role/LitellmEvalBedrockRole",
            RoleSessionName="test-session"
        )
        
        # Verify credentials are returned correctly
        assert credentials.access_key == "assumed-access-key"
        assert credentials.secret_key == "assumed-secret-key"
        assert credentials.token == "assumed-session-token"
        assert ttl is not None


def test_explicit_credentials_used_when_provided():
    """
    Test that explicit credentials are used when provided (non-EKS/IRSA scenario).
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
    
    with patch("boto3.client", return_value=mock_sts_client) as mock_boto3_client:
        
        # Call with explicit credentials
        credentials, ttl = base_aws_llm._auth_with_aws_role(
            aws_access_key_id="explicit-access-key",
            aws_secret_access_key="explicit-secret-key",
            aws_session_token="assumed-session-token",
            aws_role_name="arn:aws:iam::2222222222222:role/LitellmEvalBedrockRole",
            aws_session_name="test-session"
        )
        
        # Should create STS client with explicit credentials
        mock_boto3_client.assert_called_once_with(
            "sts",
            aws_access_key_id="explicit-access-key",
            aws_secret_access_key="explicit-secret-key",
            aws_session_token="assumed-session-token",
        )
        
        # Should call assume_role
        mock_sts_client.assume_role.assert_called_once_with(
            RoleArn="arn:aws:iam::2222222222222:role/LitellmEvalBedrockRole",
            RoleSessionName="test-session"
        )
        
        # Verify credentials are returned correctly
        assert credentials.access_key == "assumed-access-key"
        assert credentials.secret_key == "assumed-secret-key"
        assert credentials.token == "assumed-session-token"
        assert ttl is not None


def test_partial_credentials_still_use_ambient():
    """
    Test that if only one credential is provided, we still use ambient credentials.
    This handles edge cases where configuration might be incomplete.
    """
    base_aws_llm = BaseAWSLLM()
    
    # Mock the boto3 STS client
    mock_sts_client = MagicMock()
    
    # Mock the STS response
    mock_expiry = MagicMock()
    mock_expiry.tzinfo = timezone.utc
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
    
    with patch("boto3.client", return_value=mock_sts_client) as mock_boto3_client:
        
        # Call with only access key (missing secret key)
        credentials, ttl = base_aws_llm._auth_with_aws_role(
            aws_access_key_id="AKIAEXAMPLE",
            aws_secret_access_key=None,
            aws_session_token=None,
            aws_role_name="arn:aws:iam::2222222222222:role/LitellmEvalBedrockRole",
            aws_session_name="test-session"
        )
        
        # Should still pass partial credentials to boto3.client
        mock_boto3_client.assert_called_once_with(
            "sts",
            aws_access_key_id="AKIAEXAMPLE",
            aws_secret_access_key=None,
            aws_session_token=None,
        )
        
        # Should still call assume_role
        mock_sts_client.assume_role.assert_called_once_with(
            RoleArn="arn:aws:iam::2222222222222:role/LitellmEvalBedrockRole",
            RoleSessionName="test-session"
        )


def test_cross_account_role_assumption():
    """
    Test assuming a role in a different AWS account (common in multi-account setups).
    """
    base_aws_llm = BaseAWSLLM()
    
    # Mock the boto3 STS client
    mock_sts_client = MagicMock()
    
    # Mock the STS response for cross-account role
    mock_expiry = MagicMock()
    mock_expiry.tzinfo = timezone.utc
    time_diff = MagicMock()
    time_diff.total_seconds.return_value = 3600
    mock_expiry.__sub__ = MagicMock(return_value=time_diff)

    mock_sts_response = {
        "Credentials": {
            "AccessKeyId": "cross-account-access-key",
            "SecretAccessKey": "cross-account-secret-key",
            "SessionToken": "cross-account-session-token",
            "Expiration": mock_expiry,
        }
    }
    mock_sts_client.assume_role.return_value = mock_sts_response
    
    with patch("boto3.client", return_value=mock_sts_client) as mock_boto3_client:
        
        # Assume role in different account (EKS/IRSA scenario)
        credentials, ttl = base_aws_llm._auth_with_aws_role(
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_session_token=None,
            aws_role_name="arn:aws:iam::999999999999:role/CrossAccountRole",
            aws_session_name="cross-account-session"
        )
        
        # Should use ambient credentials
        mock_boto3_client.assert_called_once_with("sts")
        
        # Should call assume_role with cross-account role
        mock_sts_client.assume_role.assert_called_once_with(
            RoleArn="arn:aws:iam::999999999999:role/CrossAccountRole",
            RoleSessionName="cross-account-session"
        )
        
        # Verify cross-account credentials are returned
        assert credentials.access_key == "cross-account-access-key"
        assert credentials.secret_key == "cross-account-secret-key"
        assert credentials.token == "cross-account-session-token"
        assert ttl is not None


def test_role_assumption_with_custom_session_name():
    """
    Test role assumption with a custom session name.
    """
    base_aws_llm = BaseAWSLLM()
    
    # Mock the boto3 STS client
    mock_sts_client = MagicMock()
    
    # Mock the STS response
    mock_expiry = MagicMock()
    mock_expiry.tzinfo = timezone.utc
    time_diff = MagicMock()
    time_diff.total_seconds.return_value = 3600
    mock_expiry.__sub__ = MagicMock(return_value=time_diff)

    mock_sts_response = {
        "Credentials": {
            "AccessKeyId": "custom-session-access-key",
            "SecretAccessKey": "custom-session-secret-key",
            "SessionToken": "custom-session-token",
            "Expiration": mock_expiry,
        }
    }
    mock_sts_client.assume_role.return_value = mock_sts_response
    
    with patch("boto3.client", return_value=mock_sts_client):
        
        # Use custom session name
        credentials, ttl = base_aws_llm._auth_with_aws_role(
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_session_token=None,
            aws_role_name="arn:aws:iam::1111111111111:role/LitellmRole",
            aws_session_name="evals-bedrock-session"
        )
        
        # Should call assume_role with custom session name
        mock_sts_client.assume_role.assert_called_once_with(
            RoleArn="arn:aws:iam::1111111111111:role/LitellmRole",
            RoleSessionName="evals-bedrock-session"
        )
        
        # Verify credentials are returned
        assert credentials.access_key == "custom-session-access-key"
        assert credentials.secret_key == "custom-session-secret-key"
        assert credentials.token == "custom-session-token"


def test_role_assumption_ttl_calculation():
    """
    Test that TTL is calculated correctly from STS response expiration.
    """
    base_aws_llm = BaseAWSLLM()
    
    # Mock the boto3 STS client
    mock_sts_client = MagicMock()
    
    # Create a real datetime for expiration (1 hour from now)
    expiration_time = datetime.now(timezone.utc) + timedelta(hours=1)
    
    mock_sts_response = {
        "Credentials": {
            "AccessKeyId": "ttl-test-access-key",
            "SecretAccessKey": "ttl-test-secret-key",
            "SessionToken": "ttl-test-session-token",
            "Expiration": expiration_time,
        }
    }
    mock_sts_client.assume_role.return_value = mock_sts_response
    
    with patch("boto3.client", return_value=mock_sts_client):
        
        credentials, ttl = base_aws_llm._auth_with_aws_role(
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_session_token=None,
            aws_role_name="arn:aws:iam::1111111111111:role/LitellmRole",
            aws_session_name="ttl-test-session"
        )
        
        # TTL should be approximately 3540 seconds (1 hour - 60 second buffer)
        assert ttl is not None
        assert 3500 <= ttl <= 3600  # Allow some variance for test execution time


def test_role_assumption_error_handling():
    """
    Test that role assumption errors are properly propagated.
    """
    base_aws_llm = BaseAWSLLM()
    
    # Mock the boto3 STS client to raise an exception
    mock_sts_client = MagicMock()
    mock_sts_client.assume_role.side_effect = Exception("AccessDenied: User is not authorized to perform sts:AssumeRole")
    
    with patch("boto3.client", return_value=mock_sts_client):
        
        # Should raise the exception
        with pytest.raises(Exception) as exc_info:
            base_aws_llm._auth_with_aws_role(
                aws_access_key_id=None,
                aws_secret_access_key=None,
                aws_session_token=None,
                aws_role_name="arn:aws:iam::1111111111111:role/UnauthorizedRole",
                aws_session_name="error-test-session"
            )
        
        assert "AccessDenied" in str(exc_info.value)


def test_multiple_role_assumptions_in_sequence():
    """
    Test that multiple role assumptions work correctly in sequence.
    This simulates the scenario where different models use different roles.
    """
    base_aws_llm = BaseAWSLLM()
    
    # Mock the boto3 STS client
    mock_sts_client = MagicMock()
    
    # Mock different responses for different roles
    mock_expiry = MagicMock()
    mock_expiry.tzinfo = timezone.utc
    time_diff = MagicMock()
    time_diff.total_seconds.return_value = 3600
    mock_expiry.__sub__ = MagicMock(return_value=time_diff)

    # First role response
    mock_sts_response1 = {
        "Credentials": {
            "AccessKeyId": "role1-access-key",
            "SecretAccessKey": "role1-secret-key",
            "SessionToken": "role1-session-token",
            "Expiration": mock_expiry,
        }
    }
    
    # Second role response
    mock_sts_response2 = {
        "Credentials": {
            "AccessKeyId": "role2-access-key",
            "SecretAccessKey": "role2-secret-key",
            "SessionToken": "role2-session-token",
            "Expiration": mock_expiry,
        }
    }
    
    # Configure mock to return different responses
    mock_sts_client.assume_role.side_effect = [mock_sts_response1, mock_sts_response2]
    
    with patch("boto3.client", return_value=mock_sts_client):
        
        # First role assumption
        credentials1, ttl1 = base_aws_llm._auth_with_aws_role(
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_session_token=None,
            aws_role_name="arn:aws:iam::1111111111111:role/LitellmRole",
            aws_session_name="session-1"
        )
        
        # Second role assumption
        credentials2, ttl2 = base_aws_llm._auth_with_aws_role(
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_session_token=None,
            aws_role_name="arn:aws:iam::2222222222222:role/LitellmEvalBedrockRole",
            aws_session_name="session-2"
        )
        
        # Verify both role assumptions were made
        assert mock_sts_client.assume_role.call_count == 2
        
        # Verify first role credentials
        assert credentials1.access_key == "role1-access-key"
        assert credentials1.secret_key == "role1-secret-key"
        assert credentials1.token == "role1-session-token"
        
        # Verify second role credentials
        assert credentials2.access_key == "role2-access-key"
        assert credentials2.secret_key == "role2-secret-key"
        assert credentials2.token == "role2-session-token"


def test_auth_with_aws_role_irsa_environment():
    """Test that _auth_with_aws_role detects and uses IRSA environment variables"""
    base_llm = BaseAWSLLM()
    
    # Create a temporary file to simulate the web identity token
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write('test-web-identity-token')
        token_file = f.name
    
    try:
        # Set IRSA environment variables
        with patch.dict(os.environ, {
            'AWS_WEB_IDENTITY_TOKEN_FILE': token_file,
            'AWS_ROLE_ARN': 'arn:aws:iam::111111111111:role/eks-service-account-role',
            'AWS_REGION': 'us-east-1'
        }):
            # Mock the boto3 STS client
            mock_sts_client = MagicMock()
            mock_assume_web_identity_response = {
                'Credentials': {
                    'AccessKeyId': 'irsa-temp-access-key',
                    'SecretAccessKey': 'irsa-temp-secret-key',
                    'SessionToken': 'irsa-temp-session-token',
                    'Expiration': datetime.now() + timedelta(hours=1)
                }
            }
            mock_assume_role_response = {
                'Credentials': {
                    'AccessKeyId': 'irsa-access-key',
                    'SecretAccessKey': 'irsa-secret-key',
                    'SessionToken': 'irsa-session-token',
                    'Expiration': datetime.now() + timedelta(hours=1)
                }
            }
            mock_sts_client.assume_role_with_web_identity.return_value = mock_assume_web_identity_response
            mock_sts_client.assume_role.return_value = mock_assume_role_response
            
            with patch('boto3.client', return_value=mock_sts_client) as mock_boto3_client:
                # Call _auth_with_aws_role without explicit credentials
                creds, ttl = base_llm._auth_with_aws_role(
                    aws_access_key_id=None,
                    aws_secret_access_key=None,
                    aws_session_token=None,
                    aws_role_name='arn:aws:iam::222222222222:role/target-role',
                    aws_session_name='test-session'
                )
                
                # Verify boto3.client was called multiple times
                # First for manual IRSA, then with IRSA credentials
                assert mock_boto3_client.call_count >= 2
                
                # Verify assume_role_with_web_identity was called
                mock_sts_client.assume_role_with_web_identity.assert_called_once_with(
                    RoleArn='arn:aws:iam::111111111111:role/eks-service-account-role',
                    RoleSessionName='test-session',
                    WebIdentityToken='test-web-identity-token'
                )
                
                # Verify assume_role was called with correct parameters
                mock_sts_client.assume_role.assert_called_once_with(
                    RoleArn='arn:aws:iam::222222222222:role/target-role',
                    RoleSessionName='test-session'
                )
                
                # Verify the returned credentials
                assert creds.access_key == 'irsa-access-key'
                assert creds.secret_key == 'irsa-secret-key'
                assert creds.token == 'irsa-session-token'
                assert ttl > 0  # TTL should be positive
    finally:
        # Clean up the temporary file
        os.unlink(token_file)


def test_auth_with_aws_role_same_role_irsa():
    """Test that when IRSA role matches the requested role, we skip assumption"""
    base_llm = BaseAWSLLM()

    # Set IRSA environment variables
    with patch.dict(os.environ, {
        'AWS_ROLE_ARN': 'arn:aws:iam::111111111111:role/LitellmRole',
        'AWS_WEB_IDENTITY_TOKEN_FILE': '/var/run/secrets/eks.amazonaws.com/serviceaccount/token'
    }):
        # Mock the _auth_with_env_vars method
        mock_creds = MagicMock()
        mock_creds.access_key = 'irsa-access-key'
        mock_creds.secret_key = 'irsa-secret-key'
        mock_creds.token = 'irsa-session-token'

        with patch.object(base_llm, '_auth_with_env_vars', return_value=(mock_creds, None)) as mock_env_auth:
            # Call get_credentials instead of _auth_with_aws_role directly
            # This tests the full flow
            creds = base_llm.get_credentials(
                aws_access_key_id=None,
                aws_secret_access_key=None,
                aws_role_name='arn:aws:iam::111111111111:role/LitellmRole',  # Same as AWS_ROLE_ARN
                aws_session_name='test-session',
                aws_region_name='us-east-1'
            )

            # Verify it used the env vars auth (no role assumption)
            mock_env_auth.assert_called_once()

            # Verify the returned credentials
            assert creds.access_key == 'irsa-access-key'


def test_assume_role_with_external_id():
    """Test that assume_role STS call includes ExternalId parameter when provided"""
    base_aws_llm = BaseAWSLLM()

    # Mock the boto3 STS client
    mock_sts_client = MagicMock()
    mock_expiry = datetime.now(timezone.utc) + timedelta(hours=1)

    mock_sts_response = {
        "Credentials": {
            "AccessKeyId": "test-access-key",
            "SecretAccessKey": "test-secret-key",
            "SessionToken": "test-session-token",
            "Expiration": mock_expiry,
        }
    }
    mock_sts_client.assume_role.return_value = mock_sts_response

    with patch("boto3.client", return_value=mock_sts_client):
        # Call _auth_with_aws_role with external ID
        credentials, ttl = base_aws_llm._auth_with_aws_role(
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_session_token=None,
            aws_role_name="arn:aws:iam::123456789012:role/ExampleRole",
            aws_session_name="test-session",
            aws_external_id="UniqueExternalID123"
        )

        # Verify assume_role was called with ExternalId
        mock_sts_client.assume_role.assert_called_once_with(
            RoleArn="arn:aws:iam::123456789012:role/ExampleRole",
            RoleSessionName="test-session",
            ExternalId="UniqueExternalID123"
        )


def test_assume_role_without_external_id():
    """Test that assume_role STS call excludes ExternalId parameter when not provided"""
    base_aws_llm = BaseAWSLLM()

    # Mock the boto3 STS client
    mock_sts_client = MagicMock()
    mock_expiry = datetime.now(timezone.utc) + timedelta(hours=1)

    mock_sts_response = {
        "Credentials": {
            "AccessKeyId": "test-access-key",
            "SecretAccessKey": "test-secret-key",
            "SessionToken": "test-session-token",
            "Expiration": mock_expiry,
        }
    }
    mock_sts_client.assume_role.return_value = mock_sts_response

    with patch("boto3.client", return_value=mock_sts_client):
        # Call _auth_with_aws_role without external ID
        credentials, ttl = base_aws_llm._auth_with_aws_role(
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_session_token=None,
            aws_role_name="arn:aws:iam::123456789012:role/ExampleRole",
            aws_session_name="test-session"
        )

        # Verify assume_role was called without ExternalId
        mock_sts_client.assume_role.assert_called_once_with(
            RoleArn="arn:aws:iam::123456789012:role/ExampleRole",
            RoleSessionName="test-session"
        )


def test_converse_handler_external_id_extraction():
    """Test that BedrockConverseLLM properly extracts and passes aws_external_id parameter"""
    from litellm.llms.bedrock.chat.converse_handler import BedrockConverseLLM

    converse_llm = BedrockConverseLLM()

    # Mock get_credentials to capture parameters
    def mock_get_credentials(**kwargs):
        mock_get_credentials.called_kwargs = kwargs
        mock_credentials = MagicMock()
        mock_credentials.access_key = "test-access-key"
        mock_credentials.secret_key = "test-secret-key"
        mock_credentials.token = "test-session-token"
        return mock_credentials

    with patch.object(converse_llm, 'get_credentials', side_effect=mock_get_credentials):
        with patch.object(converse_llm, '_get_aws_region_name', return_value="us-west-2"):
            with patch.object(converse_llm, 'get_runtime_endpoint', return_value=("https://test", "https://test")):
                with patch('litellm.AmazonConverseConfig') as mock_config:
                    mock_config.return_value._transform_request.return_value = {"test": "data"}
                    with patch.object(converse_llm, 'get_request_headers') as mock_headers:
                        mock_headers.return_value = MagicMock()
                        mock_headers.return_value.headers = {"Authorization": "test"}
                        with patch('litellm.llms.custom_httpx.http_handler._get_httpx_client') as mock_client:
                            mock_http_client = MagicMock()
                            mock_response = MagicMock()
                            mock_response.raise_for_status.return_value = None
                            mock_http_client.post.return_value = mock_response
                            mock_client.return_value = mock_http_client

                            # Mock the transform_response method
                            mock_config.return_value._transform_response.return_value = MagicMock()

                            # Call completion with aws_external_id in optional_params
                            optional_params = {
                                "aws_role_name": "arn:aws:iam::123456789012:role/ExampleRole",
                                "aws_session_name": "test-session",
                                "aws_external_id": "TestExternalID123"
                            }

                            try:
                                converse_llm.completion(
                                    model="anthropic.claude-3-sonnet-20240229-v1:0",
                                    messages=[{"role": "user", "content": "Hello"}],
                                    api_base=None,
                                    custom_prompt_dict={},
                                    model_response=MagicMock(),
                                    encoding="utf-8",
                                    logging_obj=MagicMock(),
                                    optional_params=optional_params,
                                    acompletion=False,
                                    timeout=None,
                                    litellm_params={}
                                )
                            except Exception:
                                # We expect this to fail due to mocking, but that's OK
                                # We just want to verify the parameter extraction
                                pass

                            # Verify aws_external_id was extracted and passed to get_credentials
                            assert hasattr(mock_get_credentials, 'called_kwargs')
                            assert "aws_external_id" in mock_get_credentials.called_kwargs
                            assert mock_get_credentials.called_kwargs["aws_external_id"] == "TestExternalID123"
