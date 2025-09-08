"""
Tests for HuggingFace rerank functionality.
Based on the test patterns from other rerank providers and the current HuggingFace implementation.
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm


def assert_response_shape(response, custom_llm_provider):
    """Helper function to validate response structure"""
    assert hasattr(response, "id")
    assert hasattr(response, "results")
    assert hasattr(response, "meta")
    assert isinstance(response.results, list)

    for result in response.results:
        assert "index" in result
        assert "relevance_score" in result
        assert isinstance(result["index"], int)
        assert isinstance(result["relevance_score"], (int, float))


@pytest.mark.parametrize("sync_mode", [True, False])
@patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post")
@patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
def test_basic_rerank_huggingface(mock_sync_post, mock_async_post, sync_mode):
    """Test basic HuggingFace rerank functionality."""
    # Mock response data that matches HuggingFace rerank API format
    mock_response_data = [{"index": 0, "score": 0.9}, {"index": 1, "score": 0.1}]

    def return_val():
        return mock_response_data

    api_key = "test_hf_api_key"

    if sync_mode:
        # Create mock response object for sync
        mock_response = MagicMock()
        mock_response.json = return_val
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_sync_post.return_value = mock_response

        response = litellm.rerank(
            model="huggingface/BAAI/bge-reranker-base",
            query="hello",
            documents=["hello", "world"],
            top_n=2,
            api_key=api_key,
        )
        mock_sync_post.assert_called_once()
    else:
        # Create mock response object for async
        mock_response = AsyncMock()

        def return_val():
            return mock_response_data

        mock_response.json = return_val
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_async_post.return_value = mock_response

        response = asyncio.run(
            litellm.arerank(
                model="huggingface/BAAI/bge-reranker-base",
                query="hello",
                documents=["hello", "world"],
                top_n=2,
                api_key=api_key,
            )
        )
        mock_async_post.assert_called_once()

    assert response.results is not None
    assert len(response.results) == 2
    assert response.results[0]["index"] == 0
    assert response.results[0]["relevance_score"] == 0.9

    assert_response_shape(response, custom_llm_provider="huggingface")


@pytest.mark.parametrize("sync_mode", [True, False])
@patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post")
@patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
def test_huggingface_rerank_custom_api_base(mock_sync_post, mock_async_post, sync_mode):
    """Test HuggingFace rerank with custom API base."""
    mock_response_data = [{"index": 0, "score": 0.9}, {"index": 1, "score": 0.1}]

    def return_val():
        return mock_response_data

    if sync_mode:
        mock_response = MagicMock()
        mock_response.json = return_val
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_sync_post.return_value = mock_response

        response = litellm.rerank(
            model="huggingface/BAAI/bge-reranker-base",
            query="hello",
            documents=["hello", "world"],
            top_n=2,
            api_base="https://my-custom-hf-endpoint.com",
            api_key="test_api_key",
        )

        mock_sync_post.assert_called_once()
        call_url = mock_sync_post.call_args.kwargs["url"]
        assert "my-custom-hf-endpoint.com" in call_url
        assert response.results is not None
        assert len(response.results) == 2
    else:
        mock_response = AsyncMock()

        def return_val():
            return mock_response_data

        mock_response.json = return_val
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_async_post.return_value = mock_response

        response = asyncio.run(
            litellm.arerank(
                model="huggingface/BAAI/bge-reranker-base",
                query="hello",
                documents=["hello", "world"],
                top_n=2,
                api_base="https://my-custom-hf-endpoint.com",
                api_key="test_api_key",
            )
        )

        mock_async_post.assert_called_once()
        call_url = mock_async_post.call_args.kwargs["url"]
        assert "my-custom-hf-endpoint.com" in call_url
        assert response.results is not None
        assert len(response.results) == 2


@patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
def test_huggingface_rerank_with_env_vars(mock_post, monkeypatch):
    """Test HuggingFace rerank with environment variable configuration."""
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "env_test_key")
    monkeypatch.setenv("HUGGINGFACE_API_BASE", "https://env-hf-endpoint.com")

    mock_response_data = [{"index": 0, "score": 0.9}, {"index": 1, "score": 0.1}]

    def return_val():
        return mock_response_data

    mock_response = MagicMock()
    mock_response.json = return_val
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_post.return_value = mock_response

    response = litellm.rerank(
        model="huggingface/BAAI/bge-reranker-base",
        query="hello",
        documents=["hello", "world"],
        top_n=2,
    )

    mock_post.assert_called_once()
    call_url = mock_post.call_args.kwargs["url"]
    assert "env-hf-endpoint.com" in call_url

    headers = mock_post.call_args.kwargs.get("headers", {})
    assert "env_test_key" in str(headers.get("Authorization", ""))

    assert response.results is not None
    assert len(response.results) == 2


@patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
def test_huggingface_rerank_return_documents(mock_post):
    """Test HuggingFace rerank with return_documents=True."""
    mock_response_data = [
        {"index": 0, "score": 0.9, "text": "hello"},
        {"index": 1, "score": 0.1, "text": "world"},
    ]

    def return_val():
        return mock_response_data

    mock_response = MagicMock()
    mock_response.json = return_val
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_post.return_value = mock_response

    response = litellm.rerank(
        model="huggingface/BAAI/bge-reranker-base",
        query="hello",
        documents=["hello", "world"],
        top_n=2,
        return_documents=True,
        api_key="test_api_key",
    )

    mock_post.assert_called_once()
    request_data = json.loads(mock_post.call_args.kwargs["data"])
    assert request_data.get("return_text") is True

    assert response.results is not None
    assert len(response.results) == 2
    # Check that documents are included in response
    for result in response.results:
        if "document" in result:
            assert "text" in result["document"]


@patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
def test_huggingface_rerank_error_handling(mock_post):
    """Test HuggingFace rerank error handling."""

    def return_val():
        return {"error": "Unauthorized"}

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.json = return_val
    mock_response.text = "Unauthorized"
    mock_post.return_value = mock_response

    with pytest.raises(Exception):
        litellm.rerank(
            model="huggingface/BAAI/bge-reranker-base",
            query="hello",
            documents=["hello", "world"],
            top_n=2,
            api_key="invalid_key",
        )


def test_huggingface_rerank_config():
    """Test HuggingFaceRerankConfig class functionality."""
    from litellm.llms.huggingface.rerank.transformation import HuggingFaceRerankConfig

    config = HuggingFaceRerankConfig()

    # Test complete URL generation
    assert (
        config.get_complete_url(None, "test")
        == "https://api-inference.huggingface.co/rerank"
    )

    # Test custom API base
    custom_url = config.get_complete_url("https://custom.huggingface.co", "test")
    assert custom_url == "https://custom.huggingface.co/rerank"

    # Test supported parameters
    supported_params = config.get_supported_cohere_rerank_params("test")
    assert "query" in supported_params
    assert "documents" in supported_params
    assert "top_n" in supported_params
    assert "return_documents" in supported_params

    # Test parameter mapping
    params = config.map_cohere_rerank_params(
        non_default_params={
            "query": "hello",
            "documents": ["hello", "world"],
            "top_n": 2,
            "return_documents": True,
        },
        model="test",
        drop_params=False,
        query="hello",
        documents=["hello", "world"],
    )
    print(f"params: {params}")
    assert params["query"] == "hello"
    assert params["texts"] == ["hello", "world"]
    assert params["top_n"] == 2
    assert params["return_text"] is True


def test_request_transformation():
    """Test request transformation logic."""
    from litellm.llms.huggingface.rerank.transformation import HuggingFaceRerankConfig
    from litellm.types.rerank import OptionalRerankParams

    config = HuggingFaceRerankConfig()

    optional_params = OptionalRerankParams(
        query="hello", texts=["hello", "world"], top_n=2, return_text=True
    )

    request_body = config.transform_rerank_request(
        model="test", optional_rerank_params=optional_params, headers={}
    )

    assert request_body["query"] == "hello"
    assert request_body["texts"] == ["hello", "world"]
    assert request_body["top_n"] == 2
    assert request_body["return_text"] is True
    assert request_body["raw_scores"] is False
    assert request_body["truncate"] is False
    assert request_body["truncation_direction"] == "Right"


def test_response_transformation():
    """Test response transformation logic."""
    from litellm.llms.huggingface.rerank.transformation import HuggingFaceRerankConfig
    from litellm.types.rerank import RerankResponse

    config = HuggingFaceRerankConfig()

    # Mock HuggingFace response
    hf_response_data = [
        {"index": 0, "score": 0.9, "text": "hello"},
        {"index": 1, "score": 0.1, "text": "world"},
    ]

    def return_val():
        return hf_response_data

    # Create mock httpx response
    mock_response = MagicMock()
    mock_response.json = return_val

    model_response = RerankResponse()

    transformed_response = config.transform_rerank_response(
        model="test",
        raw_response=mock_response,
        model_response=model_response,
        logging_obj=None,
        request_data={"return_text": True},
    )

    assert transformed_response.results is not None
    assert len(transformed_response.results) == 2
    assert transformed_response.results[0]["index"] == 0
    assert transformed_response.results[0]["relevance_score"] == 0.9
    assert transformed_response.results[1]["index"] == 1
    assert transformed_response.results[1]["relevance_score"] == 0.1

    # Check documents are included when return_text is True
    for result in transformed_response.results:
        if "document" in result:
            assert "text" in result["document"]


def test_validate_environment():
    """Test environment validation logic."""
    from litellm.llms.huggingface.rerank.transformation import HuggingFaceRerankConfig

    config = HuggingFaceRerankConfig()

    # Test with API key
    headers = config.validate_environment(headers={}, model="test", api_key="test_key")

    assert "Authorization" in headers
    assert "Bearer test_key" in headers["Authorization"]
    assert headers["accept"] == "application/json"
    assert headers["content-type"] == "application/json"

    # Test headers override
    custom_headers = {"custom": "header"}
    headers = config.validate_environment(
        headers=custom_headers, model="test", api_key="test_key"
    )

    assert "custom" in headers
    assert headers["custom"] == "header"


@patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
def test_huggingface_rerank_request_payload(mock_post):
    """Test that the request payload is correctly formatted for HuggingFace API."""
    mock_response_data = [{"index": 0, "score": 0.9}, {"index": 1, "score": 0.1}]

    def return_val():
        return mock_response_data

    mock_response = MagicMock()
    mock_response.json = return_val
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_post.return_value = mock_response

    response = litellm.rerank(
        model="huggingface/BAAI/bge-reranker-base",
        query="hello",
        documents=["hello", "world"],
        api_key="test_api_key",
        top_n=2,
        return_documents=True,
    )

    mock_post.assert_called_once()

    # Verify URL
    call_url = mock_post.call_args.kwargs["url"]
    assert call_url == "https://api-inference.huggingface.co/rerank"

    # Verify headers
    headers = mock_post.call_args.kwargs["headers"]
    assert "Bearer test_api_key" in headers["Authorization"]
    assert headers["content-type"] == "application/json"

    # Verify request body
    request_data = json.loads(mock_post.call_args.kwargs["data"])
    expected_request = {
        "query": "hello",
        "texts": ["hello", "world"],
        "raw_scores": False,
        "return_text": True,
        "truncate": False,
        "truncation_direction": "Right",
        "top_n": 2,
    }

    for key, value in expected_request.items():
        assert request_data[key] == value

    assert response.results is not None
    assert len(response.results) == 2
