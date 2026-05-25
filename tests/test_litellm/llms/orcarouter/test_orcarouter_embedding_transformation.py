"""
Unit tests for OrcaRouter embedding transformation logic.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.llms.orcarouter.embedding.transformation import (
    OrcaRouterEmbeddingConfig,
)


def test_supported_params():
    supported = OrcaRouterEmbeddingConfig().get_supported_openai_params("test-model")
    assert "timeout" in supported
    assert "dimensions" in supported
    assert "encoding_format" in supported
    assert "user" in supported


def test_transform_request_strips_orcarouter_prefix():
    result = OrcaRouterEmbeddingConfig().transform_embedding_request(
        model="orcarouter/openai/text-embedding-3-small",
        input="Hello world",
        optional_params={},
        headers={},
    )
    assert result["model"] == "openai/text-embedding-3-small"
    assert result["input"] == ["Hello world"]


def test_transform_request_preserves_unprefixed_model():
    result = OrcaRouterEmbeddingConfig().transform_embedding_request(
        model="openai/text-embedding-3-small",
        input=["a", "b"],
        optional_params={"dimensions": 512},
        headers={},
    )
    assert result["model"] == "openai/text-embedding-3-small"
    assert result["input"] == ["a", "b"]
    assert result["dimensions"] == 512


def test_transform_request_string_input_wrapped_in_list():
    result = OrcaRouterEmbeddingConfig().transform_embedding_request(
        model="orcarouter/google/gemini-embedding-001",
        input="single string",
        optional_params={},
        headers={},
    )
    assert result["input"] == ["single string"]


def test_get_complete_url_default():
    url = OrcaRouterEmbeddingConfig().get_complete_url(
        api_base=None,
        api_key=None,
        model="orcarouter/openai/text-embedding-3-small",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://api.orcarouter.ai/v1/embeddings"


def test_get_complete_url_custom_base_strips_trailing_slash():
    url = OrcaRouterEmbeddingConfig().get_complete_url(
        api_base="https://custom.orcarouter.ai/v1/",
        api_key=None,
        model="x",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://custom.orcarouter.ai/v1/embeddings"


def test_validate_environment_sets_auth_and_attribution():
    headers = OrcaRouterEmbeddingConfig().validate_environment(
        headers={},
        model="x",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="sk-test",
    )
    assert headers["Authorization"] == "Bearer sk-test"
    assert headers["HTTP-Referer"] == "https://www.orcarouter.ai/"
    assert headers["X-Title"] == "liteLLM"


def test_validate_environment_raises_without_api_key(monkeypatch):
    """If neither the explicit api_key arg nor the ORCAROUTER_API_KEY env var
    is set, raise instead of sending `Authorization: Bearer None` to upstream."""
    monkeypatch.delenv("ORCAROUTER_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OrcaRouter API key is required"):
        OrcaRouterEmbeddingConfig().validate_environment(
            headers={},
            model="x",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
        )


def test_validate_environment_user_headers_take_priority():
    headers = OrcaRouterEmbeddingConfig().validate_environment(
        headers={"X-Title": "my-app", "Custom-Header": "v"},
        model="x",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="sk-test",
    )
    assert headers["X-Title"] == "my-app"
    assert headers["Custom-Header"] == "v"
    assert headers["Authorization"] == "Bearer sk-test"


def test_map_openai_params_filters_unsupported():
    params = OrcaRouterEmbeddingConfig().map_openai_params(
        non_default_params={"dimensions": 512, "unsupported_field": "x"},
        optional_params={},
        model="openai/text-embedding-3-small",
        drop_params=False,
    )
    assert params == {"dimensions": 512}


def test_transform_embedding_response_parses_openai_compatible_payload():
    """Smoke check that the inherited LiteLLM response converter handles
    OrcaRouter's OpenAI-compatible embedding payload."""
    from unittest.mock import Mock

    import httpx

    from litellm.types.utils import EmbeddingResponse

    raw = Mock(spec=httpx.Response)
    raw.status_code = 200
    raw.text = ""
    raw.headers = {}
    raw.json.return_value = {
        "object": "list",
        "data": [
            {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
        ],
        "model": "openai/text-embedding-3-small",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }

    result = OrcaRouterEmbeddingConfig().transform_embedding_response(
        model="orcarouter/openai/text-embedding-3-small",
        raw_response=raw,
        model_response=EmbeddingResponse(),
        logging_obj=Mock(),
        api_key="sk-test",
        request_data={"input": ["hi"]},
        optional_params={},
        litellm_params={},
    )
    assert result.data[0]["embedding"] == [0.1, 0.2, 0.3]
    assert result.usage.prompt_tokens == 5


def test_get_error_class_wraps_with_orcarouter_exception():
    """The error class hook should return our provider-specific exception
    so callers see a consistent error type from OrcaRouter."""
    from litellm.llms.orcarouter.common_utils import OrcaRouterException

    err = OrcaRouterEmbeddingConfig().get_error_class(
        error_message="upstream said no",
        status_code=503,
        headers={"x-trace": "abc"},
    )
    assert isinstance(err, OrcaRouterException)
    assert err.status_code == 503
