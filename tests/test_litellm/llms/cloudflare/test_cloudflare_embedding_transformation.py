import json
from unittest.mock import Mock, patch

import pytest

import litellm
from litellm.llms.cloudflare.embedding.transformation import CloudflareEmbeddingConfig
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.utils import ProviderConfigManager


def test_provider_config_manager_returns_cloudflare_embedding_config():
    config = ProviderConfigManager.get_provider_embedding_config(
        model="@cf/baai/bge-large-en-v1.5",
        provider=litellm.LlmProviders.CLOUDFLARE,
    )

    assert isinstance(config, CloudflareEmbeddingConfig)


def test_get_complete_url_defaults_to_openai_compatible_endpoint(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    config = CloudflareEmbeddingConfig()

    url = config.get_complete_url(
        api_base=None,
        api_key="cf-key",
        model="@cf/baai/bge-large-en-v1.5",
        optional_params={},
        litellm_params={},
    )

    assert url == "https://api.cloudflare.com/client/v4/accounts/acct/ai/v1/embeddings"


def test_get_complete_url_is_idempotent_for_full_endpoint():
    config = CloudflareEmbeddingConfig()

    url = config.get_complete_url(
        api_base="https://api.cloudflare.com/client/v4/accounts/acct/ai/v1/embeddings",
        api_key="cf-key",
        model="@cf/baai/bge-large-en-v1.5",
        optional_params={},
        litellm_params={},
    )

    assert url == "https://api.cloudflare.com/client/v4/accounts/acct/ai/v1/embeddings"


def test_get_complete_url_migrates_legacy_ai_run_base():
    config = CloudflareEmbeddingConfig()

    url = config.get_complete_url(
        api_base="https://api.cloudflare.com/client/v4/accounts/acct/ai/run/",
        api_key="cf-key",
        model="@cf/baai/bge-large-en-v1.5",
        optional_params={},
        litellm_params={},
    )

    assert url == "https://api.cloudflare.com/client/v4/accounts/acct/ai/v1/embeddings"


def test_validate_environment_preserves_extra_headers():
    config = CloudflareEmbeddingConfig()

    headers = config.validate_environment(
        headers={"X-Test": "value"},
        model="@cf/baai/bge-large-en-v1.5",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="cf-key",
    )

    assert headers == {
        "Authorization": "Bearer cf-key",
        "Content-Type": "application/json",
        "X-Test": "value",
    }


def test_embedding_routes_to_cloudflare_openai_compatible_endpoint(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    client = HTTPHandler()
    response_json = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.1, 0.2, 0.3],
                "index": 0,
            }
        ],
        "model": "@cf/baai/bge-large-en-v1.5",
        "usage": {"prompt_tokens": 2, "total_tokens": 2},
    }
    raw_response = Mock()
    raw_response.status_code = 200
    raw_response.headers = {"content-type": "application/json"}
    raw_response.json.return_value = response_json
    raw_response.text = json.dumps(response_json)

    with patch.object(HTTPHandler, "post", return_value=raw_response) as mock_post:
        response = litellm.embedding(
            model="cloudflare/@cf/baai/bge-large-en-v1.5",
            input=["hello"],
            api_key="cf-key",
            client=client,
            caching=False,
        )

    request = mock_post.call_args.kwargs
    body = json.loads(request["data"])
    assert request["url"] == "https://api.cloudflare.com/client/v4/accounts/acct/ai/v1/embeddings"
    assert request["headers"]["Authorization"] == "Bearer cf-key"
    assert body == {
        "model": "@cf/baai/bge-large-en-v1.5",
        "input": ["hello"],
    }
    assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]


def test_embedding_requires_cloudflare_api_key(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.delenv("CLOUDFLARE_API_KEY", raising=False)

    with pytest.raises(litellm.APIConnectionError, match="Missing Cloudflare API Key"):
        litellm.embedding(
            model="cloudflare/@cf/baai/bge-large-en-v1.5",
            input=["hello"],
            caching=False,
        )
