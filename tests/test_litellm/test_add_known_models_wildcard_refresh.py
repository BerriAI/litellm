"""Regression tests for LIT-4947.

`add_known_models` must refresh `litellm.models_by_provider` so that a model newly
added to the cost map (e.g. after "Reload Price Data") shows up everywhere wildcard
expansion is used, without requiring a proxy restart.
"""

import copy

import pytest

import litellm
from litellm.proxy.auth.model_checks import get_known_models_from_wildcard


@pytest.fixture
def restore_model_registry():
    """Snapshot and restore the global model registry mutated by add_known_models."""
    original_model_cost = litellm.model_cost
    original_models_by_provider = litellm.models_by_provider
    original_sets = {
        name: set(value)
        for name in dir(litellm)
        if name.endswith("_models") and isinstance((value := getattr(litellm, name)), set)
    }

    yield

    litellm.model_cost = original_model_cost
    litellm.models_by_provider = original_models_by_provider
    for name, snapshot in original_sets.items():
        live_set = getattr(litellm, name)
        live_set.clear()
        live_set.update(snapshot)


def _reload_with_new_model(model: str, litellm_provider: str) -> None:
    new_map = copy.deepcopy(litellm.model_cost)
    new_map[model] = {"litellm_provider": litellm_provider, "mode": "chat"}
    litellm.model_cost = new_map
    litellm.add_known_models(model_cost_map=new_map)


class TestAddKnownModelsRefreshesWildcardExpansion:
    def test_new_vertex_language_model_reaches_models_by_provider_and_wildcard(self, restore_model_registry):
        model = "vertex_ai/gemini-9.9-flash-lite"

        _reload_with_new_model(model, "vertex_ai-language-models")

        assert model in litellm.vertex_language_models
        assert model in litellm.models_by_provider["vertex_ai"]
        assert model in get_known_models_from_wildcard("vertex_ai/*")

    def test_bare_vertex_ai_provider_model_is_bucketed(self, restore_model_registry):
        model = "vertex_ai/some-new-native-thing"

        _reload_with_new_model(model, "vertex_ai")

        assert model in litellm.vertex_ai_models
        assert model in litellm.models_by_provider["vertex_ai"]
        assert model in get_known_models_from_wildcard("vertex_ai/*")

    def test_new_openai_model_reaches_wildcard(self, restore_model_registry):
        model = "gpt-99-turbo"

        _reload_with_new_model(model, "openai")

        assert model in litellm.models_by_provider["openai"]
        assert f"openai/{model}" in get_known_models_from_wildcard("openai/*")
