"""
Unit tests for OpenRouter embedding transformation logic.
"""

import json
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.openrouter.embedding.transformation import (
    OpenrouterEmbeddingConfig,
)


def test_openrouter_embedding_supported_params():
    """Test that supported OpenAI params are correctly defined."""
    config = OpenrouterEmbeddingConfig()
    supported = config.get_supported_openai_params("test-model")

    assert "timeout" in supported
    assert "dimensions" in supported
    assert "encoding_format" in supported
    assert "user" in supported


def test_openrouter_embedding_transform_request():
    """Test request transformation logic."""
    config = OpenrouterEmbeddingConfig()

    # Test with string input
    result = config.transform_embedding_request(
        model="openrouter/google/text-embedding-004",
        input="Hello world",
        optional_params={},
        headers={},
    )

    assert result["model"] == "google/text-embedding-004"
    assert result["input"] == ["Hello world"]

    # Test with list input
    result = config.transform_embedding_request(
        model="google/text-embedding-004",
        input=["Hello", "World"],
        optional_params={"dimensions": 512},
        headers={},
    )

    assert result["model"] == "google/text-embedding-004"
    assert result["input"] == ["Hello", "World"]
    assert result["dimensions"] == 512


def test_openrouter_embedding_validate_environment():
    """Test environment validation and header setup."""
    config = OpenrouterEmbeddingConfig()

    # Test with API key
    headers = config.validate_environment(
        headers={"Custom-Header": "value"},
        model="test-model",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="test-api-key",
    )

    # Should include OpenRouter-specific headers
    assert "HTTP-Referer" in headers
    assert "X-Title" in headers
    # Should include Content-Type header
    assert "Content-Type" in headers
    assert headers["Content-Type"] == "application/json"
    # Should include Authorization header
    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer test-api-key"
    # Should preserve custom headers
    assert headers["Custom-Header"] == "value"

    # Test without API key
    headers_no_key = config.validate_environment(
        headers={},
        model="test-model",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key=None,
    )

    # Should still include OpenRouter headers but not Authorization
    assert "HTTP-Referer" in headers_no_key
    assert "X-Title" in headers_no_key
    assert "Content-Type" in headers_no_key
    assert "Authorization" not in headers_no_key


def test_openrouter_embedding_get_complete_url():
    """Test URL construction."""
    config = OpenrouterEmbeddingConfig()

    url = config.get_complete_url(
        api_base="https://openrouter.ai/api/v1",
        api_key="test-key",
        model="test-model",
        optional_params={},
        litellm_params={},
    )

    assert url == "https://openrouter.ai/api/v1/embeddings"

    # Test with trailing slash
    url = config.get_complete_url(
        api_base="https://openrouter.ai/api/v1/",
        api_key="test-key",
        model="test-model",
        optional_params={},
        litellm_params={},
    )

    assert url == "https://openrouter.ai/api/v1/embeddings"


def test_openrouter_embedding_map_params():
    """Test parameter mapping."""
    config = OpenrouterEmbeddingConfig()

    result = config.map_openai_params(
        non_default_params={"dimensions": 512, "timeout": 30, "unsupported": "value"},
        optional_params={},
        model="test-model",
        drop_params=False,
    )

    # Supported params should be included
    assert result["dimensions"] == 512
    assert result["timeout"] == 30
    # Unsupported params should not be included
    assert "unsupported" not in result


@pytest.mark.respx(assert_all_called=True)
def test_openrouter_embedding_request_and_response(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post("https://openrouter.ai/api/v1/embeddings")
    route.return_value = httpx.Response(
        200,
        json={
            "object": "list",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
            ],
            "model": "openai/text-embedding-3-small",
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        },
    )
    inputs: Final = ["Hello world", "How are you?"]

    response: Final = litellm.embedding(
        model="openrouter/openai/text-embedding-3-small",
        input=inputs,
        api_key="test-api-key",
        client=HTTPHandler(),
    )

    request_body: Final = json.loads(route.calls[0].request.content)
    assert request_body == {
        "model": "openai/text-embedding-3-small",
        "input": inputs,
    }
    assert route.calls[0].request.headers["Authorization"] == "Bearer test-api-key"
    assert [item["embedding"] for item in response.data] == [[0.1, 0.2], [0.3, 0.4]]
    assert response.usage.total_tokens == 4
