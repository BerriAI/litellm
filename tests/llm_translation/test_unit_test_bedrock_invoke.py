import os
import sys
import traceback
from dotenv import load_dotenv
import litellm.types
import pytest
from litellm import AmazonInvokeConfig
import json

load_dotenv()
import io
import os

sys.path.insert(0, os.path.abspath("../.."))
from unittest.mock import AsyncMock, Mock, patch


# Initialize the transformer
@pytest.fixture
def bedrock_transformer():
    return AmazonInvokeConfig()


def test_get_complete_url_basic(bedrock_transformer):
    """Test basic URL construction for non-streaming request"""
    url = bedrock_transformer.get_complete_url(
        api_base="https://bedrock-runtime.us-east-1.amazonaws.com",
        api_key=None,
        model="anthropic.claude-v2",
        optional_params={},
        stream=False,
        litellm_params={},
    )

    assert (
        url
        == "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-v2/invoke"
    )


def test_get_complete_url_streaming(bedrock_transformer):
    """Test URL construction for streaming request"""
    url = bedrock_transformer.get_complete_url(
        api_base="https://bedrock-runtime.us-east-1.amazonaws.com",
        api_key=None,
        model="anthropic.claude-v2",
        optional_params={},
        stream=True,
        litellm_params={},
    )

    assert (
        url
        == "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-v2/invoke-with-response-stream"
    )


def test_transform_request_invalid_provider(bedrock_transformer):
    """Test request transformation with invalid provider"""
    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(Exception) as exc_info:
        bedrock_transformer.transform_request(
            model="invalid.model",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

    assert "Unknown provider" in str(exc_info.value)


@patch("botocore.auth.SigV4Auth")
@patch("botocore.awsrequest.AWSRequest")
def test_sign_request_basic(mock_aws_request, mock_sigv4_auth, bedrock_transformer):
    """Test basic request signing without extra headers"""
    # Mock credentials
    mock_credentials = Mock()
    bedrock_transformer.get_credentials = Mock(return_value=mock_credentials)

    # Setup mock SigV4Auth instance
    mock_auth_instance = Mock()
    mock_sigv4_auth.return_value = mock_auth_instance

    # Setup mock AWSRequest instance
    mock_request = Mock()
    mock_request.headers = {
        "Authorization": "AWS4-HMAC-SHA256 Credential=...",
        "X-Amz-Date": "20240101T000000Z",
        "Content-Type": "application/json",
    }
    mock_aws_request.return_value = mock_request

    # Test parameters
    headers = {}
    optional_params = {"aws_region_name": "us-east-1"}
    request_data = {"prompt": "Hello"}
    api_base = "https://bedrock-runtime.us-east-1.amazonaws.com"

    # Call the method
    result, _ = bedrock_transformer.sign_request(
        headers=headers,
        optional_params=optional_params,
        request_data=request_data,
        api_base=api_base,
    )

    # Verify the results
    mock_sigv4_auth.assert_called_once_with(mock_credentials, "bedrock", "us-east-1")
    mock_aws_request.assert_called_once_with(
        method="POST",
        url=api_base,
        data='{"prompt": "Hello"}',
        headers={"Content-Type": "application/json"},
    )
    mock_auth_instance.add_auth.assert_called_once_with(mock_request)
    assert result == mock_request.headers


def test_transform_request_cohere_command(bedrock_transformer):
    """Test request transformation for Cohere Command model"""
    messages = [{"role": "user", "content": "Hello"}]

    result = bedrock_transformer.transform_request(
        model="cohere.command-r",
        messages=messages,
        optional_params={"max_tokens": 2048},
        litellm_params={},
        headers={},
    )

    print(
        "transformed request for invoke cohere command=", json.dumps(result, indent=4)
    )
    expected_result = {"message": "Hello", "max_tokens": 2048, "chat_history": []}
    assert result == expected_result


def test_transform_request_ai21(bedrock_transformer):
    """Test request transformation for AI21"""
    messages = [{"role": "user", "content": "Hello"}]

    result = bedrock_transformer.transform_request(
        model="ai21.j2-ultra",
        messages=messages,
        optional_params={"max_tokens": 2048},
        litellm_params={},
        headers={},
    )

    print("transformed request for invoke ai21=", json.dumps(result, indent=4))

    expected_result = {
        "prompt": "Hello",
        "max_tokens": 2048,
    }
    assert result == expected_result


def test_transform_request_mistral(bedrock_transformer):
    """Test request transformation for Mistral"""
    messages = [{"role": "user", "content": "Hello"}]

    result = bedrock_transformer.transform_request(
        model="mistral.mistral-7b",
        messages=messages,
        optional_params={"max_tokens": 2048},
        litellm_params={},
        headers={},
    )

    print("transformed request for invoke mistral=", json.dumps(result, indent=4))

    expected_result = {
        "prompt": "<s>[INST] Hello [/INST]\n",
        "max_tokens": 2048,
    }
    assert result == expected_result


def test_transform_request_amazon_titan(bedrock_transformer):
    """Test request transformation for Amazon Titan"""
    messages = [{"role": "user", "content": "Hello"}]

    result = bedrock_transformer.transform_request(
        model="amazon.titan-text-express-v1",
        messages=messages,
        optional_params={"maxTokenCount": 2048},
        litellm_params={},
        headers={},
    )
    print("transformed request for invoke amazon titan=", json.dumps(result, indent=4))

    expected_result = {
        "inputText": "\n\nUser: Hello\n\nBot: ",
        "textGenerationConfig": {
            "maxTokenCount": 2048,
        },
    }
    assert result == expected_result


def test_transform_request_meta_llama(bedrock_transformer):
    """Test request transformation for Meta/Llama"""
    messages = [{"role": "user", "content": "Hello"}]

    result = bedrock_transformer.transform_request(
        model="meta.llama2-70b",
        messages=messages,
        optional_params={"max_gen_len": 2048},
        litellm_params={},
        headers={},
    )

    print("transformed request for invoke meta llama=", json.dumps(result, indent=4))
    expected_result = {"prompt": "Hello", "max_gen_len": 2048}
    assert result == expected_result


def test_filter_headers_for_aws_signature():
    """Test that header filtering works correctly for AWS signature calculation"""
    from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
    
    # Create a test instance
    aws_llm = BaseAWSLLM()
    
    # Test headers including both AWS and non-AWS headers
    test_headers = {
        "Content-Type": "application/json",
        "Host": "bedrock-runtime.us-east-1.amazonaws.com",
        "x-amz-date": "20240101T120000Z",
        "x-amz-security-token": "test-token",
        "x-custom-header": "custom-value",
        "x-litellm-user-id": "user123",
        "x-forwarded-for": "192.168.1.1",
        "authorization": "Bearer test-token",
        "user-agent": "test-agent",
        "x-envoy-expected-rq-timeout-ms": "300000",
        "x-envoy-external-address": "10.105.1.156"
    }
    
    # Filter headers for AWS signature
    filtered_headers = aws_llm._filter_headers_for_aws_signature(test_headers)
    
    # Verify that only AWS-related headers are included
    expected_aws_headers = {
        "Content-Type": "application/json",
        "Host": "bedrock-runtime.us-east-1.amazonaws.com",
        "x-amz-date": "20240101T120000Z",
        "x-amz-security-token": "test-token"
    }
    
    assert filtered_headers == expected_aws_headers, f"Expected {expected_aws_headers}, got {filtered_headers}"
    
    # Verify that non-AWS headers are excluded
    excluded_headers = ["x-custom-header", "x-litellm-user-id", "x-forwarded-for", "user-agent", 
                       "x-envoy-expected-rq-timeout-ms", "x-envoy-external-address"]
    for header in excluded_headers:
        assert header not in filtered_headers, f"Header {header} should not be in filtered headers"
    
    # Test with empty headers
    empty_filtered = aws_llm._filter_headers_for_aws_signature({})
    assert empty_filtered == {}
    
    # Test with only non-AWS headers
    non_aws_headers = {
        "x-custom-trace": "trace-123",
        "x-user-context": "premium",
        "x-request-source": "mobile-app"
    }
    filtered_non_aws = aws_llm._filter_headers_for_aws_signature(non_aws_headers)
    assert filtered_non_aws == {}
