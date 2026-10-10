"""
Unit tests for OpenRouter embedding transformation logic.
"""

from unittest.mock import Mock

import httpx
from litellm.cost_calculator import response_cost_calculator
from litellm.types.utils import EmbeddingResponse

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


def test_openrouter_embedding_preserves_provider_reported_cost():
    config = OpenrouterEmbeddingConfig()
    raw_response = httpx.Response(
        status_code=200,
        json={
            "data": [{"embedding": [0.1, 0.2], "index": 0, "object": "embedding"}],
            "id": "gen-emb-test",
            "model": "text-embedding-3-small",
            "object": "list",
            "provider": "OpenAI",
            "usage": {
                "prompt_tokens": 3,
                "total_tokens": 3,
                "cost": 0.00000006,
                "is_byok": False,
                "cost_details": {
                    "upstream_inference_cost": 0.00000006,
                    "upstream_inference_prompt_cost": 0.00000006,
                    "upstream_inference_completions_cost": 0,
                },
            },
        },
    )

    response = config.transform_embedding_response(
        model="openrouter/openai/text-embedding-3-small",
        raw_response=raw_response,
        model_response=EmbeddingResponse(),
        logging_obj=Mock(),
        api_key="test-api-key",
        request_data={},
        optional_params={},
        litellm_params={},
    )

    assert response._hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"] == 0.00000006
    assert response._hidden_params["response_cost_details"] == {
        "upstream_inference_cost": 0.00000006,
        "upstream_inference_prompt_cost": 0.00000006,
        "upstream_inference_completions_cost": 0,
    }
    assert (
        response_cost_calculator(
            response_object=response,
            model="openai/text-embedding-3-small",
            custom_llm_provider="openrouter",
            call_type="embedding",
            optional_params={},
        )
        == 0.00000006
    )


def test_openrouter_embedding_accepts_usage_without_cost():
    config = OpenrouterEmbeddingConfig()
    raw_response = httpx.Response(
        status_code=200,
        json={
            "data": [{"embedding": [0.1, 0.2], "index": 0, "object": "embedding"}],
            "model": "text-embedding-3-small",
            "object": "list",
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        },
    )

    response = config.transform_embedding_response(
        model="openrouter/openai/text-embedding-3-small",
        raw_response=raw_response,
        model_response=EmbeddingResponse(),
        logging_obj=Mock(),
        api_key="test-api-key",
        request_data={},
        optional_params={},
        litellm_params={},
    )

    assert "llm_provider-x-litellm-response-cost" not in response._hidden_params.get("additional_headers", {})
    assert "response_cost_details" not in response._hidden_params


def test_openrouter_embedding_accepts_partial_cost_details():
    config = OpenrouterEmbeddingConfig()
    raw_response = httpx.Response(
        status_code=200,
        json={
            "data": [{"embedding": [0.1, 0.2], "index": 0, "object": "embedding"}],
            "model": "text-embedding-3-small",
            "object": "list",
            "usage": {
                "prompt_tokens": 3,
                "total_tokens": 3,
                "cost": 0.00000006,
                "cost_details": {"upstream_inference_cost": 0.00000005},
            },
        },
    )

    response = config.transform_embedding_response(
        model="openrouter/openai/text-embedding-3-small",
        raw_response=raw_response,
        model_response=EmbeddingResponse(),
        logging_obj=Mock(),
        api_key="test-api-key",
        request_data={},
        optional_params={},
        litellm_params={},
    )

    assert response._hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"] == 0.00000006
    assert response._hidden_params["response_cost_details"] == {"upstream_inference_cost": 0.00000005}
