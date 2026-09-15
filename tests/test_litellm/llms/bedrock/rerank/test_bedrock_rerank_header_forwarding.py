"""
Test to verify that custom headers are correctly forwarded to Bedrock rerank API calls.

This test verifies the fix for the issue where headers configured via
forward_client_headers_to_llm_api were not being passed to Bedrock rerank provider.
"""

import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

import litellm
from litellm.llms.bedrock.base_aws_llm import Boto3CredentialsInfo
from litellm.llms.bedrock.rerank.handler import BedrockRerankHandler
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

# Mock response for Bedrock rerank
# Format based on Bedrock rerank API response structure
bedrock_rerank_response = {
    "results": [
        {"index": 2, "relevanceScore": 0.95},
        {"index": 0, "relevanceScore": 0.1},
        {"index": 1, "relevanceScore": 0.05},
    ],
    "usage": {"search_units": 1},
}

# Test data
test_query = "What is the capital of the United States?"
test_documents = [
    "Carson City is the capital city of the American state of Nevada.",
    "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
    "Washington, D.C. is the capital of the United States.",
]


def create_mock_credentials():
    """Create mock AWS credentials for testing"""
    mock_credentials = MagicMock()
    mock_credentials.access_key = "test-access-key"
    mock_credentials.secret_key = "test-secret-key"
    mock_credentials.token = None
    return Boto3CredentialsInfo(
        credentials=mock_credentials,
        aws_region_name="us-east-1",
        aws_bedrock_runtime_endpoint="https://bedrock-runtime.us-east-1.amazonaws.com",
    )


@pytest.mark.parametrize(
    "model",
    [
        "bedrock/arn:aws:bedrock:us-east-1::foundation-model/cohere.rerank-v3-5:0",
        "bedrock/arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0",
    ],
)
def test_bedrock_rerank_header_forwarding_sync(model):
    """
    Test that custom headers are correctly forwarded to Bedrock rerank API calls (sync).

    This test verifies the fix for the issue where headers configured via
    forward_client_headers_to_llm_api were not being passed to Bedrock rerank provider.
    """
    client = HTTPHandler()
    test_api_key = "test-bearer-token-12345"

    # Headers that would be set by the proxy when forwarding client headers
    # Using x- prefix headers as those are the ones that get forwarded
    custom_headers = {
        "X-Custom-Header": "CustomValue",
        "X-BYOK-Token": "secret-token",
        "X-Test-Header": "test-value",
    }

    # Mock AWS credentials and SigV4 auth
    mock_credentials_info = create_mock_credentials()

    with (
        patch.object(client, "post") as mock_post,
        patch(  # test-quality-ok: boto credential lookup needs live AWS; the HTTP boundary is already a MockTransport
            "litellm.llms.bedrock.rerank.handler.BedrockRerankHandler._get_boto_credentials_from_optional_params",
            return_value=mock_credentials_info,
        ),
        patch("botocore.auth.SigV4Auth") as mock_sigv4,
    ):

        # Mock SigV4Auth to not actually sign the request
        mock_sigv4_instance = MagicMock()
        mock_sigv4.return_value = mock_sigv4_instance

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(bedrock_rerank_response)
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_response.raise_for_status = lambda: None
        mock_post.return_value = mock_response

        try:
            # Call rerank with custom headers via kwargs
            # This simulates what the proxy does when forward_client_headers_to_llm_api is set
            response = litellm.rerank(
                model=model,
                query=test_query,
                documents=test_documents,
                top_n=3,
                client=client,
                headers=custom_headers,  # This is how proxy passes forwarded headers
                aws_region_name="us-east-1",
                aws_bedrock_runtime_endpoint="https://bedrock-runtime.us-east-1.amazonaws.com",
                api_key=test_api_key,
            )

            assert isinstance(response, litellm.RerankResponse)

            # Verify that the request was made
            assert mock_post.called, "HTTP client post should be called"

            # Get the actual call arguments
            call_kwargs = mock_post.call_args.kwargs
            headers = call_kwargs.get("headers", {})

            # Verify our custom headers are present in the request headers
            # Note: AWS SigV4 signing may modify header names to lowercase
            for header_key, header_value in custom_headers.items():
                header_found = (
                    header_key in headers
                    or header_key.lower() in headers
                    or any(k.lower() == header_key.lower() for k in headers.keys())
                )
                assert header_found, (
                    f"Header {header_key} should be in request headers. "
                    f"Found headers: {list(headers.keys())}"
                )

            print(f"✓ Test passed for {model} (sync)")
            print(f"  Headers correctly forwarded: {list(headers.keys())}")

        except Exception as e:
            pytest.fail(f"Failed to forward headers to {model}: {str(e)}")


@pytest.mark.parametrize(
    "model",
    [
        "bedrock/arn:aws:bedrock:us-east-1::foundation-model/cohere.rerank-v3-5:0",
        "bedrock/arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0",
    ],
)
@pytest.mark.asyncio
async def test_bedrock_rerank_header_forwarding_async(model):
    """
    Test that custom headers are correctly forwarded to Bedrock rerank API calls (async).

    This test verifies the fix for the issue where headers configured via
    forward_client_headers_to_llm_api were not being passed to Bedrock rerank provider.
    """
    client = AsyncHTTPHandler()
    test_api_key = "test-bearer-token-12345"

    # Headers that would be set by the proxy when forwarding client headers
    # Using x- prefix headers as those are the ones that get forwarded
    custom_headers = {
        "X-Custom-Header": "CustomValue",
        "X-BYOK-Token": "secret-token",
        "X-Test-Header": "test-value",
    }

    # Mock AWS credentials and SigV4 auth
    mock_credentials_info = create_mock_credentials()

    with (
        patch.object(client, "post", new_callable=AsyncMock) as mock_post,
        patch(  # test-quality-ok: boto credential lookup needs live AWS; the HTTP boundary is already a MockTransport
            "litellm.llms.bedrock.rerank.handler.BedrockRerankHandler._get_boto_credentials_from_optional_params",
            return_value=mock_credentials_info,
        ),
        patch("botocore.auth.SigV4Auth") as mock_sigv4,
    ):

        # Mock SigV4Auth to not actually sign the request
        mock_sigv4_instance = MagicMock()
        mock_sigv4.return_value = mock_sigv4_instance

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(bedrock_rerank_response)
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_response.raise_for_status = lambda: None
        mock_post.return_value = mock_response

        try:
            # Call rerank with custom headers via kwargs
            response = await litellm.arerank(
                model=model,
                query=test_query,
                documents=test_documents,
                top_n=3,
                client=client,
                headers=custom_headers,  # This is how proxy passes forwarded headers
                aws_region_name="us-east-1",
                aws_bedrock_runtime_endpoint="https://bedrock-runtime.us-east-1.amazonaws.com",
                api_key=test_api_key,
            )

            assert isinstance(response, litellm.RerankResponse)

            # Verify that the request was made
            assert mock_post.called, "HTTP client post should be called"

            # Get the actual call arguments
            call_kwargs = mock_post.call_args.kwargs
            headers = call_kwargs.get("headers", {})

            # Verify our custom headers are present in the request headers
            # Note: AWS SigV4 signing may modify header names to lowercase
            for header_key, header_value in custom_headers.items():
                header_found = (
                    header_key in headers
                    or header_key.lower() in headers
                    or any(k.lower() == header_key.lower() for k in headers.keys())
                )
                assert header_found, (
                    f"Header {header_key} should be in request headers. "
                    f"Found headers: {list(headers.keys())}"
                )

            print(f"✓ Test passed for {model} (async)")
            print(f"  Headers correctly forwarded: {list(headers.keys())}")

        except Exception as e:
            pytest.fail(f"Failed to forward headers to {model}: {str(e)}")


def test_bedrock_rerank_timeout_sync():
    """
    Test that the timeout parameter is passed through to the HTTP client for Bedrock rerank (sync).
    """
    client = HTTPHandler()
    model = "bedrock/arn:aws:bedrock:us-east-1::foundation-model/cohere.rerank-v3-5:0"
    mock_credentials_info = create_mock_credentials()

    with (
        patch.object(client, "post") as mock_post,
        patch(  # test-quality-ok: boto credential lookup needs live AWS; the HTTP boundary is already a MockTransport
            "litellm.llms.bedrock.rerank.handler.BedrockRerankHandler._get_boto_credentials_from_optional_params",
            return_value=mock_credentials_info,
        ),
        patch("botocore.auth.SigV4Auth") as mock_sigv4,
    ):

        mock_sigv4.return_value = MagicMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(bedrock_rerank_response)
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_response.raise_for_status = lambda: None
        mock_post.return_value = mock_response

        litellm.rerank(
            model=model,
            query=test_query,
            documents=test_documents,
            top_n=3,
            client=client,
            timeout=0.001,
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint="https://bedrock-runtime.us-east-1.amazonaws.com",
        )

        assert mock_post.called
        call_kwargs = mock_post.call_args.kwargs
        assert (
            call_kwargs.get("timeout") == 0.001
        ), f"Expected timeout=0.001, got timeout={call_kwargs.get('timeout')}"


@pytest.mark.asyncio
async def test_bedrock_rerank_timeout_async():
    """
    Test that the timeout parameter is passed through to the HTTP client for Bedrock rerank (async).
    """
    client = AsyncHTTPHandler()
    model = "bedrock/arn:aws:bedrock:us-east-1::foundation-model/cohere.rerank-v3-5:0"
    mock_credentials_info = create_mock_credentials()

    with (
        patch.object(client, "post", new_callable=AsyncMock) as mock_post,
        patch(  # test-quality-ok: boto credential lookup needs live AWS; the HTTP boundary is already a MockTransport
            "litellm.llms.bedrock.rerank.handler.BedrockRerankHandler._get_boto_credentials_from_optional_params",
            return_value=mock_credentials_info,
        ),
        patch("botocore.auth.SigV4Auth") as mock_sigv4,
    ):

        mock_sigv4.return_value = MagicMock()
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(bedrock_rerank_response)
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_response.raise_for_status = lambda: None
        mock_post.return_value = mock_response

        await litellm.arerank(
            model=model,
            query=test_query,
            documents=test_documents,
            top_n=3,
            client=client,
            timeout=0.001,
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint="https://bedrock-runtime.us-east-1.amazonaws.com",
        )

        assert mock_post.called
        call_kwargs = mock_post.call_args.kwargs
        assert (
            call_kwargs.get("timeout") == 0.001
        ), f"Expected timeout=0.001, got timeout={call_kwargs.get('timeout')}"


def test_bedrock_rerank_extra_headers_and_headers_merge():
    """
    Test that both extra_headers and headers parameters are correctly merged for Bedrock rerank.

    This ensures that headers from kwargs (forwarded by proxy) and extra_headers
    (passed explicitly) are both included in the final headers sent to the provider.
    """
    client = HTTPHandler()
    test_api_key = "test-bearer-token-12345"
    model = "bedrock/arn:aws:bedrock:us-east-1::foundation-model/cohere.rerank-v3-5:0"

    # Headers from proxy (via kwargs["headers"])
    proxy_headers = {"X-Forwarded-Header": "ProxyValue"}

    # Explicit extra_headers
    explicit_headers = {"X-Explicit-Header": "ExplicitValue"}

    # Mock AWS credentials and SigV4 auth
    mock_credentials_info = create_mock_credentials()

    with (
        patch.object(client, "post") as mock_post,
        patch(  # test-quality-ok: boto credential lookup needs live AWS; the HTTP boundary is already a MockTransport
            "litellm.llms.bedrock.rerank.handler.BedrockRerankHandler._get_boto_credentials_from_optional_params",
            return_value=mock_credentials_info,
        ),
        patch("botocore.auth.SigV4Auth") as mock_sigv4,
    ):

        # Mock SigV4Auth to not actually sign the request
        mock_sigv4_instance = MagicMock()
        mock_sigv4.return_value = mock_sigv4_instance

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = json.dumps(bedrock_rerank_response)
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_response.raise_for_status = lambda: None
        mock_post.return_value = mock_response

        try:
            response = litellm.rerank(
                model=model,
                query=test_query,
                documents=test_documents,
                top_n=3,
                client=client,
                headers=proxy_headers,  # From proxy forwarding
                extra_headers=explicit_headers,  # Explicitly passed
                aws_region_name="us-east-1",
                aws_bedrock_runtime_endpoint="https://bedrock-runtime.us-east-1.amazonaws.com",
                api_key=test_api_key,
            )

            assert isinstance(response, litellm.RerankResponse)

            call_kwargs = mock_post.call_args.kwargs
            headers = call_kwargs.get("headers", {})

            # Both sets of headers should be present
            # Note: AWS SigV4 signing may modify header names to lowercase
            proxy_header_found = any(
                k.lower() == "x-forwarded-header" for k in headers.keys()
            )
            assert proxy_header_found, (
                "Proxy forwarded header should be present. "
                f"Found headers: {list(headers.keys())}"
            )

            explicit_header_found = any(
                k.lower() == "x-explicit-header" for k in headers.keys()
            )
            assert explicit_header_found, (
                "Explicitly passed header should be present. "
                f"Found headers: {list(headers.keys())}"
            )

            print("✓ Both header sources correctly merged and forwarded")
            print(f"  Final headers: {list(headers.keys())}")

        except Exception as e:
            pytest.fail(f"Failed to merge and forward headers: {str(e)}")


def test_bedrock_rerank_forwarded_headers_excluded_from_sigv4_signature():
    """
    A forwarded header like x-forwarded-for can be rewritten between LiteLLM
    signing the request and AWS receiving it (e.g. by an intermediate load
    balancer), which invalidates the signature if that header was part of
    the signed set. It must still reach Bedrock, just unsigned.
    """
    handler = BedrockRerankHandler()

    prepared_request = handler._prepare_request(
        model="cohere.rerank-v3-5:0",
        api_base=None,
        extra_headers={"x-forwarded-for": "203.0.113.5"},
        data={"query": test_query, "documents": test_documents},
        optional_params={
            "aws_access_key_id": "test-access-key",
            "aws_secret_access_key": "test-secret-key",
            "aws_region_name": "us-east-1",
        },
    )

    headers = prepared_request["prepped"].headers
    signed_headers = headers["Authorization"].split("SignedHeaders=")[1].split(",")[0].split(";")

    assert "x-forwarded-for" not in signed_headers, (
        f"x-forwarded-for must not be part of the SigV4 signature, got SignedHeaders={signed_headers}"
    )
    assert headers["x-forwarded-for"] == "203.0.113.5", "forwarded header must still reach Bedrock, unsigned"


def test_bedrock_rerank_signs_with_sigv4_even_when_bedrock_api_key_is_set(monkeypatch):
    """
    Bedrock API keys are only valid for Bedrock and Bedrock Runtime actions, not for
    Agents for Amazon Bedrock Runtime ones. Rerank is served by bedrock-agent-runtime,
    so it has to keep signing with SigV4 even when AWS_BEARER_TOKEN_BEDROCK is set.
    """
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-bedrock-api-key")

    handler = BedrockRerankHandler()

    prepared_request = handler._prepare_request(
        model="cohere.rerank-v3-5:0",
        api_base=None,
        extra_headers=None,
        data={"query": test_query, "documents": test_documents},
        optional_params={
            "aws_access_key_id": "test-access-key",
            "aws_secret_access_key": "test-secret-key",
            "aws_region_name": "us-east-1",
        },
    )

    assert prepared_request["endpoint_url"].startswith("https://bedrock-agent-runtime.")

    authorization = prepared_request["prepped"].headers["Authorization"]
    assert authorization.startswith("AWS4-HMAC-SHA256"), (
        f"rerank must sign with SigV4, got Authorization={authorization[:30]}"
    )


@pytest.mark.asyncio
async def test_bedrock_rerank_records_llm_api_duration():
    """The bedrock rerank handler must feed httpx timing into the logging obj, so the
    proxy can emit x-litellm-overhead-duration-ms / x-litellm-timing-* on /rerank."""
    import httpx

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=bedrock_rerank_response)

    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))

    with patch(  # test-quality-ok: boto credential lookup needs live AWS; the HTTP boundary is already a MockTransport
        "litellm.llms.bedrock.rerank.handler.BedrockRerankHandler._get_boto_credentials_from_optional_params",
        return_value=create_mock_credentials(),
    ):
        response = await litellm.arerank(
            model="bedrock/arn:aws:bedrock:us-east-1::foundation-model/cohere.rerank-v3-5:0",
            query=test_query,
            documents=test_documents,
            top_n=3,
            client=client,
            aws_region_name="us-east-1",
        )

    assert response._hidden_params["litellm_overhead_time_ms"] is not None
    assert response._hidden_params["_response_ms"] >= response._hidden_params["litellm_overhead_time_ms"]
