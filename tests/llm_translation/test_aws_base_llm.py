import pytest
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from botocore.credentials import Credentials
from typing import Dict, Any
from litellm.llms.bedrock.base_aws_llm import (
    BaseAWSLLM,
    AwsAuthError,
    Boto3CredentialsInfo,
)


# Test fixtures
@pytest.fixture
def base_aws_llm():
    return BaseAWSLLM()


@pytest.fixture
def mock_credentials():
    return Credentials(
        access_key="test_access", secret_key="test_secret", token="test_token"
    )


# Test cache key generation
def test_get_cache_key(base_aws_llm):
    test_args = {
        "aws_access_key_id": "test_key",
        "aws_secret_access_key": "test_secret",
    }
    cache_key = base_aws_llm.get_cache_key(test_args)
    assert isinstance(cache_key, str)
    assert len(cache_key) == 64  # SHA-256 produces 64 character hex string


# Test web identity token authentication
@patch("boto3.client")
@patch("litellm.llms.bedrock.base_aws_llm.get_secret")  # Add this patch
def test_auth_with_web_identity_token(mock_get_secret, mock_boto3_client, base_aws_llm):
    # Mock get_secret to return a token
    mock_get_secret.return_value = "mocked_oidc_token"

    # Mock the STS client and response
    mock_sts = MagicMock()
    mock_sts.assume_role_with_web_identity.return_value = {
        "Credentials": {
            "AccessKeyId": "test_access",
            "SecretAccessKey": "test_secret",
            "SessionToken": "test_token",
        },
        "PackedPolicySize": 10,
    }
    mock_boto3_client.return_value = mock_sts

    credentials, ttl = base_aws_llm._auth_with_web_identity_token(
        aws_web_identity_token="test_token",
        aws_role_name="test_role",
        aws_session_name="test_session",
        aws_region_name="us-west-2",
        aws_sts_endpoint=None,
    )

    # Verify get_secret was called with the correct argument
    mock_get_secret.assert_called_once_with("test_token")

    assert isinstance(credentials, Credentials)
    assert ttl == 3540  # default TTL (3600 - 60)


# Test AWS role authentication
@patch("boto3.client")
def test_auth_with_aws_role(mock_boto3_client, base_aws_llm):
    # Mock the STS client and response
    mock_sts = MagicMock()
    expiry_time = datetime.now(timezone.utc)
    mock_sts.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "test_access",
            "SecretAccessKey": "test_secret",
            "SessionToken": "test_token",
            "Expiration": expiry_time,
        }
    }
    mock_boto3_client.return_value = mock_sts

    credentials, ttl = base_aws_llm._auth_with_aws_role(
        aws_access_key_id="test_access",
        aws_secret_access_key="test_secret",
        aws_session_token="test_token",
        aws_role_name="test_role",
        aws_session_name="test_session",
    )

    assert isinstance(credentials, Credentials)
    assert isinstance(ttl, float)


# Test AWS profile authentication
@patch("boto3.Session")
def test_auth_with_aws_profile(mock_session, base_aws_llm, mock_credentials):
    # Mock the session
    mock_session_instance = MagicMock()
    mock_session_instance.get_credentials.return_value = mock_credentials
    mock_session.return_value = mock_session_instance

    credentials, ttl = base_aws_llm._auth_with_aws_profile("test_profile")

    assert credentials == mock_credentials
    assert ttl is None


# Test session token authentication
def test_auth_with_aws_session_token(base_aws_llm):
    credentials, ttl = base_aws_llm._auth_with_aws_session_token(
        aws_access_key_id="test_access",
        aws_secret_access_key="test_secret",
        aws_session_token="test_token",
    )

    assert isinstance(credentials, Credentials)
    assert credentials.access_key == "test_access"
    assert credentials.secret_key == "test_secret"
    assert credentials.token == "test_token"
    assert ttl is None


# Test access key and secret key authentication
@patch("boto3.Session")
def test_auth_with_access_key_and_secret_key(
    mock_session, base_aws_llm, mock_credentials
):
    # Mock the session
    mock_session_instance = MagicMock()
    mock_session_instance.get_credentials.return_value = mock_credentials
    mock_session.return_value = mock_session_instance

    credentials, ttl = base_aws_llm._auth_with_access_key_and_secret_key(
        aws_access_key_id="test_access",
        aws_secret_access_key="test_secret",
        aws_region_name="us-west-2",
    )

    assert credentials == mock_credentials
    assert ttl == 3540  # default TTL (3600 - 60)


# Test environment variables authentication
@patch("boto3.Session")
def test_auth_with_env_vars(mock_session, base_aws_llm, mock_credentials):
    # Mock the session
    mock_session_instance = MagicMock()
    mock_session_instance.get_credentials.return_value = mock_credentials
    mock_session.return_value = mock_session_instance

    credentials, ttl = base_aws_llm._auth_with_env_vars()

    assert credentials == mock_credentials
    assert ttl is None


# Test runtime endpoint resolution
def test_get_runtime_endpoint(base_aws_llm):
    endpoint_url, proxy_endpoint_url = base_aws_llm.get_runtime_endpoint(
        api_base=None, aws_bedrock_runtime_endpoint=None, aws_region_name="us-west-2"
    )
    assert endpoint_url == "https://bedrock-runtime.us-west-2.amazonaws.com"
    assert proxy_endpoint_url == "https://bedrock-runtime.us-west-2.amazonaws.com"

    endpoint_url, proxy_endpoint_url = base_aws_llm.get_runtime_endpoint(
        aws_bedrock_runtime_endpoint=None, aws_region_name="us-east-1", api_base=None
    )
    assert endpoint_url == "https://bedrock-runtime.us-east-1.amazonaws.com"
    assert proxy_endpoint_url == "https://bedrock-runtime.us-east-1.amazonaws.com"


@pytest.fixture
def clear_cache(base_aws_llm):
    """Clear the cache before each test"""
    base_aws_llm.iam_cache.in_memory_cache.cache_dict = {}
    yield
