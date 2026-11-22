"""Process-wide deployment_url caching for SAP GenAI Hub configs (issue #47096).

The provider map builds a fresh config instance on every LLM call, so an
instance-scoped cached_property re-ran the two orchestration-discovery GETs on
every call. The cache now lives on the class, keyed on (base_url, resource_group).
"""

import pytest

from litellm.llms.sap.chat.transformation import GenAIHubOrchestrationConfig
from litellm.llms.sap.embed.transformation import GenAIHubEmbeddingConfig


@pytest.fixture(autouse=True)
def clear_deployment_url_caches():
    GenAIHubOrchestrationConfig._deployment_url_cache.clear()
    GenAIHubEmbeddingConfig._deployment_url_cache.clear()
    yield
    GenAIHubOrchestrationConfig._deployment_url_cache.clear()
    GenAIHubEmbeddingConfig._deployment_url_cache.clear()


def _chat_config(base_url: str, resource_group: str) -> GenAIHubOrchestrationConfig:
    config = GenAIHubOrchestrationConfig()
    config._base_url = base_url
    config._resource_group = resource_group
    return config


def _embed_config(monkeypatch, base_url: str, resource_group: str) -> GenAIHubEmbeddingConfig:
    monkeypatch.setattr(
        "litellm.llms.sap.embed.transformation.get_token_creator",
        lambda service_key=None: (lambda: "token", base_url, resource_group),
    )
    return GenAIHubEmbeddingConfig()


def test_chat_deployment_url_fetched_once_across_instances(monkeypatch):
    fetches = []

    def fake_fetch(self):
        fetches.append(self)
        return "https://deploy.example/v2"

    monkeypatch.setattr(GenAIHubOrchestrationConfig, "_fetch_deployment_url", fake_fetch)

    # Two distinct instances, like the per-call lambda in the provider map
    first = _chat_config("https://api.example", "default")
    second = _chat_config("https://api.example", "default")

    assert first.deployment_url == "https://deploy.example/v2"
    assert second.deployment_url == "https://deploy.example/v2"
    assert len(fetches) == 1


def test_chat_deployment_url_cache_keyed_on_base_url_and_resource_group(monkeypatch):
    fetches = []

    def fake_fetch(self):
        fetches.append(self)
        return f"https://deploy.example/{self.resource_group}"

    monkeypatch.setattr(GenAIHubOrchestrationConfig, "_fetch_deployment_url", fake_fetch)

    default_rg = _chat_config("https://api.example", "default")
    other_rg = _chat_config("https://api.example", "other")

    assert default_rg.deployment_url == "https://deploy.example/default"
    assert other_rg.deployment_url == "https://deploy.example/other"
    assert len(fetches) == 2


def test_embed_deployment_url_fetched_once_across_instances(monkeypatch):
    fetches = []

    def fake_fetch(self):
        fetches.append(self)
        return "https://deploy.example/v2"

    monkeypatch.setattr(GenAIHubEmbeddingConfig, "_fetch_deployment_url", fake_fetch)

    first = _embed_config(monkeypatch, "https://api.example", "default")
    second = _embed_config(monkeypatch, "https://api.example", "default")

    assert first.deployment_url == "https://deploy.example/v2"
    assert second.deployment_url == "https://deploy.example/v2"
    assert len(fetches) == 1
