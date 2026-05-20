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
    """API key must be resolved from explicit arg, litellm globals, or env;
    if all four sources are empty, raise instead of sending Bearer None."""
    monkeypatch.delenv("ORCAROUTER_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "api_key", None, raising=False)
    monkeypatch.setattr(litellm, "orcarouter_key", None, raising=False)

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
