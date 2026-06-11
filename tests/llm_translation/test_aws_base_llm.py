import pytest
from botocore.credentials import Credentials
from litellm.llms.bedrock.base_aws_llm import (
    BaseAWSLLM,
)


# Test fixtures
@pytest.fixture
def base_aws_llm():
    return BaseAWSLLM()


# Test cache key generation
def test_get_cache_key(base_aws_llm):
    test_args = {
        "aws_access_key_id": "test_key",
        "aws_secret_access_key": "test_secret",
    }
    cache_key = base_aws_llm.get_cache_key(test_args)
    assert isinstance(cache_key, str)
    assert len(cache_key) == 64  # SHA-256 produces 64 character hex string


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
