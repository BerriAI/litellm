import pytest

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
def test_rerank_routes_by_request_key_prefix(api_key, expected_host):
    config = VoyageRerankConfig()
    config.validate_environment({}, "rerank-2.5", api_key=api_key)

    assert config.get_complete_url(None, "rerank-2.5") == f"{expected_host}/rerank"


@pytest.mark.parametrize("api_key, expected_host", [("al-key", MONGODB_API_BASE), ("pa-key", VOYAGE_API_BASE)])
def test_rerank_routes_by_env_key_prefix(monkeypatch, api_key, expected_host):
    monkeypatch.setenv("VOYAGE_API_KEY", api_key)
    config = VoyageRerankConfig()
    config.validate_environment({}, "rerank-2.5")

    assert config.get_complete_url(None, "rerank-2.5") == f"{expected_host}/rerank"


def test_rerank_request_key_beats_env_key_for_routing(monkeypatch):
    """A MongoDB key on the request must not be posted to the Voyage host the env key names"""
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-from-env")
    config = VoyageRerankConfig()

    headers = config.validate_environment({}, "rerank-2.5", api_key="al-on-request")

    assert headers["Authorization"] == "Bearer al-on-request"
    assert config.get_complete_url(None, "rerank-2.5") == f"{MONGODB_API_BASE}/rerank"


def test_rerank_config_is_built_per_request_so_keys_cannot_leak(monkeypatch):
    """get_complete_url reads a key off the instance, so each request must get its own instance"""
    import litellm
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    first = ProviderConfigManager.get_provider_rerank_config(
        model="rerank-2.5", provider=LlmProviders.VOYAGE, api_base=None, present_version_params=[]
    )
    second = ProviderConfigManager.get_provider_rerank_config(
        model="rerank-2.5", provider=LlmProviders.VOYAGE, api_base=None, present_version_params=[]
    )
    assert isinstance(first, litellm.VoyageRerankConfig) and first is not second

    first.validate_environment({}, "rerank-2.5", api_key="al-first-request")
    second.validate_environment({}, "rerank-2.5", api_key="pa-second-request")

    assert first.get_complete_url(None, "rerank-2.5") == f"{MONGODB_API_BASE}/rerank"
    assert second.get_complete_url(None, "rerank-2.5") == f"{VOYAGE_API_BASE}/rerank"


def test_rerank_falls_back_to_env_when_validate_environment_did_not_run(monkeypatch):
    """A caller that skips validate_environment keeps the pre-existing env-only behaviour"""
    monkeypatch.setenv("VOYAGE_API_KEY", "al-from-env")

    assert VoyageRerankConfig().get_complete_url(None, "rerank-2.5") == f"{MONGODB_API_BASE}/rerank"


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
    config = VoyageRerankConfig()

    headers = config.validate_environment({}, "rerank-2.5")

    assert headers["Authorization"] == "Bearer al-from-env"
    assert config.get_complete_url(None, "rerank-2.5").startswith(MONGODB_API_BASE)


def test_get_voyage_api_key_prefers_env_vars_in_documented_order(monkeypatch):
    monkeypatch.setenv("VOYAGE_AI_API_KEY", "second")
    monkeypatch.setenv("VOYAGE_AI_TOKEN", "third")
    assert get_voyage_api_key() == "second"

    monkeypatch.setenv("VOYAGE_API_KEY", "first")
    assert get_voyage_api_key() == "first"
    assert get_voyage_api_key("explicit") == "explicit"
