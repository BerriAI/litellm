import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.voyage.common_utils import (
    MONGODB_API_BASE,
    VOYAGE_API_BASE,
    get_default_base_url,
    get_voyage_api_key,
)
from litellm.llms.voyage.embedding.transformation import VoyageEmbeddingConfig
from litellm.llms.voyage.embedding.transformation_contextual import (
    VoyageContextualEmbeddingConfig,
)
from litellm.llms.voyage.embedding.transformation_multimodal import (
    VoyageMultimodalEmbeddingConfig,
)
from litellm.llms.voyage.rerank.transformation import VoyageRerankConfig

VOYAGE_KEY_ENV_VARS = ("VOYAGE_API_KEY", "VOYAGE_AI_API_KEY", "VOYAGE_AI_TOKEN")


@pytest.fixture(autouse=True)
def clear_voyage_env(monkeypatch):
    for env_var in VOYAGE_KEY_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)


def test_mongodb_key_routes_to_mongodb_host():
    """MongoDB-issued keys carry the `al-` prefix and are only valid on ai.mongodb.com"""
    assert get_default_base_url("al-1234567890") == MONGODB_API_BASE


@pytest.mark.parametrize("api_key", ["pa-1234567890", "sk-1234567890", "al", "", None])
def test_non_mongodb_key_routes_to_voyage_host(api_key):
    assert get_default_base_url(api_key) == VOYAGE_API_BASE


@pytest.mark.parametrize("env_var", VOYAGE_KEY_ENV_VARS)
def test_mongodb_key_from_any_supported_env_var_routes_to_mongodb_host(monkeypatch, env_var):
    monkeypatch.setenv(env_var, "al-from-env")
    assert get_default_base_url() == MONGODB_API_BASE


def test_explicit_key_wins_over_env_for_routing(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "al-from-env")
    assert get_default_base_url("pa-explicit") == VOYAGE_API_BASE


@pytest.mark.parametrize(
    "config, endpoint",
    [
        (VoyageEmbeddingConfig(), "embeddings"),
        (VoyageContextualEmbeddingConfig(), "contextualizedembeddings"),
        (VoyageMultimodalEmbeddingConfig(), "multimodalembeddings"),
    ],
)
@pytest.mark.parametrize("api_key, expected_host", [("al-key", MONGODB_API_BASE), ("pa-key", VOYAGE_API_BASE)])
def test_embedding_configs_route_by_key_prefix(config, endpoint, api_key, expected_host):
    url = config.get_complete_url(None, api_key, "voyage-3", {}, {})
    assert url == f"{expected_host}/{endpoint}"


@pytest.mark.parametrize(
    "config, endpoint",
    [
        (VoyageEmbeddingConfig(), "embeddings"),
        (VoyageContextualEmbeddingConfig(), "contextualizedembeddings"),
        (VoyageMultimodalEmbeddingConfig(), "multimodalembeddings"),
    ],
)
def test_explicit_api_base_overrides_key_routing(config, endpoint):
    url = config.get_complete_url("https://gateway.internal/v1", "al-key", "voyage-3", {}, {})
    assert url == f"https://gateway.internal/v1/{endpoint}"


@pytest.mark.parametrize("api_key, expected_host", [("al-key", MONGODB_API_BASE), ("pa-key", VOYAGE_API_BASE)])
def test_rerank_routes_by_env_key_prefix(monkeypatch, api_key, expected_host):
    monkeypatch.setenv("VOYAGE_API_KEY", api_key)
    assert VoyageRerankConfig().get_complete_url(None, "rerank-2.5") == f"{expected_host}/rerank"


@pytest.mark.parametrize(
    "config",
    [VoyageEmbeddingConfig(), VoyageContextualEmbeddingConfig(), VoyageMultimodalEmbeddingConfig()],
)
def test_auth_header_uses_the_key_the_url_was_routed_on(monkeypatch, config):
    """The host is picked from a key, so the Authorization header has to carry that same key"""
    monkeypatch.setenv("VOYAGE_AI_TOKEN", "al-from-env")

    headers = config.validate_environment({}, "voyage-3", [], {}, {})
    url = config.get_complete_url(None, None, "voyage-3", {}, {})

    assert headers["Authorization"] == "Bearer al-from-env"
    assert url.startswith(MONGODB_API_BASE)


def test_rerank_auth_header_uses_the_key_the_url_was_routed_on(monkeypatch):
    monkeypatch.setenv("VOYAGE_AI_TOKEN", "al-from-env")

    headers = VoyageRerankConfig().validate_environment({}, "rerank-2.5")

    assert headers["Authorization"] == "Bearer al-from-env"
    assert VoyageRerankConfig().get_complete_url(None, "rerank-2.5").startswith(MONGODB_API_BASE)


def test_get_voyage_api_key_prefers_env_vars_in_documented_order(monkeypatch):
    monkeypatch.setenv("VOYAGE_AI_API_KEY", "second")
    monkeypatch.setenv("VOYAGE_AI_TOKEN", "third")
    assert get_voyage_api_key() == "second"

    monkeypatch.setenv("VOYAGE_API_KEY", "first")
    assert get_voyage_api_key() == "first"
    assert get_voyage_api_key("explicit") == "explicit"
