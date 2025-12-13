import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.gigachat.embedding.transformation import GigaChatEmbeddingConfig


@pytest.fixture()
def embedding_config(monkeypatch):
    monkeypatch.setenv("GIGACHAT_VERIFY_SSL_CERTS", "true")
    return GigaChatEmbeddingConfig()


def test_get_complete_url_appends_default_path(embedding_config):
    api_base = "https://gigachat.devices.sberbank.ru/api/"
    result = embedding_config.get_complete_url(
        api_base=api_base,
        api_key="key",
        model="gigachat-embed",
        optional_params={},
        litellm_params={},
    )
    assert result.endswith("/v1/embeddings")


def test_validate_environment_uses_direct_api_key(embedding_config, monkeypatch):
    monkeypatch.setattr(
        "litellm.llms.gigachat.embedding.transformation",
        MagicMock(get_secret_str=MagicMock(return_value=None)),
    )

    headers = embedding_config.validate_environment(
        headers={"Custom": "value"},
        model="gigachat-embed",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="direct-key",
    )

    assert headers["Authorization"] == "Bearer direct-key"
    assert headers["Content-Type"] == "application/json"
    assert headers["Custom"] == "value"


def test_validate_environment_falls_back_to_oauth(embedding_config, monkeypatch):
    mock_litellm = MagicMock()
    mock_litellm.get_secret_str.return_value = None
    monkeypatch.setattr(
        "litellm.llms.gigachat.embedding.transformation",
        mock_litellm,
    )
    embedding_config._get_oauth_token = MagicMock(return_value="oauth-token")

    headers = embedding_config.validate_environment(
        headers={},
        model="gigachat-embed",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key=None,
    )

    assert headers["Authorization"] == "Bearer oauth-token"
    embedding_config._get_oauth_token.assert_called_once()


def test_transform_embedding_request_rejects_numeric_arrays(embedding_config):
    with pytest.raises(ValueError):
        embedding_config.transform_embedding_request(
            model="gigachat-embed",
            input=[[1, 2, 3]],
            optional_params={},
            headers={},
        )


def test_transform_embedding_request_basic_payload(embedding_config):
    payload = embedding_config.transform_embedding_request(
        model="gigachat-embed",
        input=["hello", "world"],
        optional_params={},
        headers={},
    )

    assert payload == {"model": "gigachat-embed", "input": ["hello", "world"]}


def test_transform_embedding_response_returns_model(monkeypatch, embedding_config):
    response_json = {
        "object": "list",
        "data": [{"embedding": [0.1, 0.2], "index": 0}],
        "model": "gigachat-embed",
    }
    raw_response = MagicMock()
    raw_response.json.return_value = response_json

    mock_embedding_response = MagicMock(return_value="parsed-response")
    monkeypatch.setattr(
        "litellm.llms.gigachat.embedding.transformation.EmbeddingResponse",
        mock_embedding_response,
    )

    result = embedding_config.transform_embedding_response(
        model="gigachat-embed",
        raw_response=raw_response,
        model_response=None,
        logging_obj=MagicMock(),
        api_key=None,
        request_data={},
        optional_params={},
        litellm_params={},
    )

    assert result == "parsed-response"
    mock_embedding_response.assert_called_once_with(**response_json)

