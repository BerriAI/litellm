import os
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.vercel_ai_gateway.embedding.transformation import (
    VercelAIGatewayEmbeddingConfig,
)
from litellm.llms.vercel_ai_gateway.common_utils import VercelAIGatewayException
from litellm.types.utils import EmbeddingResponse


def test_vercel_ai_gateway_embedding_get_complete_url():
    """Test URL generation for embeddings endpoint"""
    config = VercelAIGatewayEmbeddingConfig()

    # Test with default API base
    url = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="openai/text-embedding-3-small",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://ai-gateway.vercel.sh/v1/embeddings"

    # Test with custom API base
    url = config.get_complete_url(
        api_base="https://custom.vercel.sh/v1",
        api_key=None,
        model="openai/text-embedding-3-small",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://custom.vercel.sh/v1/embeddings"

    # Test with trailing slash
    url = config.get_complete_url(
        api_base="https://custom.vercel.sh/v1/",
        api_key=None,
        model="openai/text-embedding-3-small",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://custom.vercel.sh/v1/embeddings"


def test_vercel_ai_gateway_embedding_transform_request():
    """Test request transformation for embeddings"""
    config = VercelAIGatewayEmbeddingConfig()

    # Test with string input
    request = config.transform_embedding_request(
        model="openai/text-embedding-3-small",
        input="Hello world",
        optional_params={},
        headers={},
    )
    assert request["model"] == "openai/text-embedding-3-small"
    assert request["input"] == ["Hello world"]

    # Test with list input
    request = config.transform_embedding_request(
        model="openai/text-embedding-3-small",
        input=["Hello", "World"],
        optional_params={},
        headers={},
    )
    assert request["model"] == "openai/text-embedding-3-small"
    assert request["input"] == ["Hello", "World"]

    # Test stripping vercel_ai_gateway/ prefix
    request = config.transform_embedding_request(
        model="vercel_ai_gateway/openai/text-embedding-3-small",
        input="Hello",
        optional_params={},
        headers={},
    )
    assert request["model"] == "openai/text-embedding-3-small"


def test_vercel_ai_gateway_embedding_transform_request_with_dimensions():
    """Test request transformation with dimensions parameter"""
    config = VercelAIGatewayEmbeddingConfig()

    request = config.transform_embedding_request(
        model="openai/text-embedding-3-small",
        input="Hello world",
        optional_params={"dimensions": 768},
        headers={},
    )
    assert request["model"] == "openai/text-embedding-3-small"
    assert request["input"] == ["Hello world"]
    assert request["dimensions"] == 768


def test_vercel_ai_gateway_embedding_validate_environment():
    """Test header validation and setup"""
    config = VercelAIGatewayEmbeddingConfig()

    headers = config.validate_environment(
        headers={},
        model="openai/text-embedding-3-small",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="test_key",
    )
    assert headers["Content-Type"] == "application/json"
    assert headers["Authorization"] == "Bearer test_key"

    # Test with existing headers (should merge)
    headers = config.validate_environment(
        headers={"X-Custom": "value"},
        model="openai/text-embedding-3-small",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="test_key",
    )
    assert headers["X-Custom"] == "value"
    assert headers["Authorization"] == "Bearer test_key"


def test_vercel_ai_gateway_embedding_get_supported_params():
    """Test supported OpenAI parameters"""
    config = VercelAIGatewayEmbeddingConfig()
    supported = config.get_supported_openai_params("openai/text-embedding-3-small")

    assert "dimensions" in supported
    assert "encoding_format" in supported
    assert "timeout" in supported
    assert "user" in supported


def test_vercel_ai_gateway_embedding_map_openai_params():
    """Test OpenAI parameter mapping"""
    config = VercelAIGatewayEmbeddingConfig()

    optional_params = config.map_openai_params(
        non_default_params={"dimensions": 768, "encoding_format": "float"},
        optional_params={},
        model="openai/text-embedding-3-small",
        drop_params=False,
    )
    assert optional_params["dimensions"] == 768
    assert optional_params["encoding_format"] == "float"


def test_vercel_ai_gateway_embedding_error_class():
    """Test error class creation"""
    config = VercelAIGatewayEmbeddingConfig()

    error = config.get_error_class(
        error_message="Test error",
        status_code=400,
        headers={"Content-Type": "application/json"},
    )

    assert isinstance(error, VercelAIGatewayException)
    assert error.message == "Test error"
    assert error.status_code == 400


def test_vercel_ai_gateway_embedding_transform_response():
    """Test response transformation"""
    config = VercelAIGatewayEmbeddingConfig()

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.text = '{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2,0.3]}],"model":"openai/text-embedding-3-small","usage":{"prompt_tokens":2,"total_tokens":2}}'
    mock_response.json.return_value = {
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
        "model": "openai/text-embedding-3-small",
        "usage": {"prompt_tokens": 2, "total_tokens": 2},
    }

    mock_logging = MagicMock()

    response = config.transform_embedding_response(
        model="openai/text-embedding-3-small",
        raw_response=mock_response,
        model_response=EmbeddingResponse(),
        logging_obj=mock_logging,
        api_key="test_key",
        request_data={},
        optional_params={},
        litellm_params={},
    )

    assert response is not None
    mock_logging.post_call.assert_called_once()


def test_vercel_ai_gateway_embedding_env_vars():
    """Test environment variable handling"""
    config = VercelAIGatewayEmbeddingConfig()

    with patch.dict(
        os.environ,
        {
            "VERCEL_AI_GATEWAY_API_BASE": "https://env.vercel.sh/v1",
        },
    ):
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="openai/text-embedding-3-small",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://env.vercel.sh/v1/embeddings"
