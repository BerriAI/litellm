"""
Unit tests for OpenRouter embedding transformation logic.
"""

import json
from typing import Final
from unittest.mock import Mock, patch

import httpx
import pytest
import respx

import litellm
from litellm.cost_calculator import response_cost_calculator
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.openrouter.embedding.transformation import (
    OpenrouterEmbeddingConfig,
)
from litellm.types.utils import EmbeddingResponse


def _embedding_response(usage: dict[str, object], model: str = "qwen/qwen3-embedding-8b") -> EmbeddingResponse:
    raw_response: Final = httpx.Response(
        200,
        text=json.dumps(
            {
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "model": model,
                "usage": usage,
            }
        ),
    )
    return OpenrouterEmbeddingConfig().transform_embedding_response(
        model=f"openrouter/{model}",
        raw_response=raw_response,
        model_response=EmbeddingResponse(),
        logging_obj=Mock(),
        api_key="test-api-key",
        request_data={},
        optional_params={},
        litellm_params={},
    )


@pytest.mark.parametrize("cost", [2e-8, 0])
def test_openrouter_embedding_preserves_cost_without_price_map(cost: float) -> None:
    with patch.dict(litellm.model_cost, {}, clear=True):
        response: Final = _embedding_response({"prompt_tokens": 2, "total_tokens": 2, "cost": cost})
        assert response._hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"] == cost
        assert (
            response_cost_calculator(
                response_object=response,
                model="openrouter/qwen/qwen3-embedding-8b",
                custom_llm_provider="openrouter",
                call_type="embedding",
                optional_params={},
            )
            == cost
        )
    assert response.data[0]["embedding"] == [0.1, 0.2]
    assert response.usage.prompt_tokens == response.usage.total_tokens == 2


@pytest.mark.parametrize(
    "cost_fields",
    [
        pytest.param({}, id="missing"),
        pytest.param({"cost": None}, id="null"),
        pytest.param({"cost": "invalid"}, id="string"),
        pytest.param({"cost": True}, id="boolean"),
        pytest.param({"cost": -1}, id="negative"),
        pytest.param({"cost": {}}, id="object"),
        pytest.param({"cost": float("nan")}, id="nan"),
        pytest.param({"cost": float("inf")}, id="infinity"),
    ],
)
@patch.dict(
    litellm.model_cost,
    {
        "openrouter/openai/text-embedding-3-small": {
            "input_cost_per_token": 0.001,
            "output_cost_per_token": 0,
            "litellm_provider": "openrouter",
            "mode": "embedding",
        }
    },
)
def test_openrouter_embedding_missing_or_invalid_cost_uses_price_map(cost_fields: dict[str, object]) -> None:
    response: Final = _embedding_response(
        {"prompt_tokens": 2, "total_tokens": 2, **cost_fields}, model="openai/text-embedding-3-small"
    )
    assert "llm_provider-x-litellm-response-cost" not in response._hidden_params.get("additional_headers", {})
    assert response.data[0]["embedding"] == [0.1, 0.2]
    assert response.usage.prompt_tokens == response.usage.total_tokens == 2
    assert (
        response_cost_calculator(
            response_object=response,
            model="openai/text-embedding-3-small",
            custom_llm_provider="openrouter",
            call_type="embedding",
            optional_params={},
        )
        == 2 * 0.001
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
