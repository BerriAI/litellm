"""
Apodex provider registration and model-family classification.
"""

import json
from pathlib import Path

import pytest

import litellm
from litellm.llms.apodex.common_utils import (
    APODEX_API_BASE_URL,
    get_apodex_api_base,
    get_apodex_api_key,
    is_deep_research_model,
)
from litellm.types.utils import LlmProviders

REPO_ROOT = Path(__file__).parents[4]

CORE_MODELS = ("apodex-1.1", "apodex-1.1-mini")
DEEP_RESEARCH_MODELS = (
    "apodex-1-1-deep-research",
    "apodex-1-1-deep-solve",
    "apodex-1-1-deep-discover",
)


class TestModelFamily:
    """The model id, not the provider, selects which Apodex contract applies."""

    @pytest.mark.parametrize("model", CORE_MODELS)
    def test_core_models_are_not_deep_research(self, model: str):
        assert is_deep_research_model(model) is False
        assert is_deep_research_model(f"apodex/{model}") is False

    @pytest.mark.parametrize("model", DEEP_RESEARCH_MODELS)
    def test_deep_research_models_are_detected(self, model: str):
        assert is_deep_research_model(model) is True
        assert is_deep_research_model(f"apodex/{model}") is True

    def test_prefix_does_not_leak_into_classification(self):
        """A provider prefix containing the marker must not flip a core model."""
        assert is_deep_research_model("some-deep-gateway/apodex-1.1") is False


class TestCredentialResolution:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("APODEX_API_BASE", raising=False)
        monkeypatch.setenv("APODEX_API_KEY", "sk-env")

        assert get_apodex_api_base(None) == APODEX_API_BASE_URL
        assert get_apodex_api_key(None) == "sk-env"

    def test_explicit_values_win_over_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("APODEX_API_BASE", "https://env.apodex.test/v1")
        monkeypatch.setenv("APODEX_API_KEY", "sk-env")

        assert get_apodex_api_base("https://explicit.apodex.test/v1") == "https://explicit.apodex.test/v1"
        assert get_apodex_api_key("sk-explicit") == "sk-explicit"

    def test_env_base_overrides_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("APODEX_API_BASE", "https://env.apodex.test/v1")
        assert get_apodex_api_base(None) == "https://env.apodex.test/v1"


class TestRegistration:
    def test_provider_enum_and_lists(self):
        assert LlmProviders.APODEX.value == "apodex"
        assert "apodex" in litellm.provider_list
        assert "apodex" in litellm.constants.openai_compatible_providers
        assert APODEX_API_BASE_URL in litellm.constants.openai_compatible_endpoints

    def test_not_registered_as_a_json_provider(self):
        """Apodex needs model-aware transformations, so it must not fall into the
        generic JSON path, which would shadow the Python configs in provider resolution."""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("apodex") is False

    def test_config_classes_resolve_from_the_lazy_registry(self):
        assert litellm.ApodexChatConfig().custom_llm_provider == "apodex"
        assert litellm.ApodexResponsesConfig().custom_llm_provider == LlmProviders.APODEX

    def test_packaged_endpoint_matrix_matches_the_source(self):
        source = json.loads((REPO_ROOT / "provider_endpoints_support.json").read_text())
        backup = json.loads((REPO_ROOT / "litellm" / "provider_endpoints_support_backup.json").read_text())

        assert backup["providers"]["apodex"] == source["providers"]["apodex"]


class TestModelMetadata:
    @pytest.fixture(scope="class")
    def model_cost(self) -> dict:
        with open(REPO_ROOT / "model_prices_and_context_window.json") as f:
            return json.load(f)

    def test_every_apodex_model_is_registered(self, model_cost: dict):
        assert {key for key in model_cost if key.startswith("apodex/")} == {
            f"apodex/{model}" for model in (*CORE_MODELS, *DEEP_RESEARCH_MODELS)
        }

    def test_core_model_pricing(self, model_cost: dict):
        info = model_cost["apodex/apodex-1.1"]
        assert info["litellm_provider"] == "apodex"
        assert info["mode"] == "chat"
        assert info["max_input_tokens"] == 262144
        # GET /v1/models reports max_completion_tokens 65536, well under the context window
        assert info["max_output_tokens"] == 65536
        assert info["max_tokens"] == info["max_output_tokens"]
        assert info["input_cost_per_token"] == 3e-07
        assert info["cache_read_input_token_cost"] == 3e-08
        assert info["output_cost_per_token"] == 3e-06
        # Requests over 200K input tokens are billed at 2x across every tier
        assert info["input_cost_per_token_above_200k_tokens"] == 6e-07
        assert info["cache_read_input_token_cost_above_200k_tokens"] == 6e-08
        assert info["output_cost_per_token_above_200k_tokens"] == 6e-06
        assert info["supports_prompt_caching"] is True
        assert info["supports_function_calling"] is True
        assert info["supported_endpoints"] == ["/v1/chat/completions", "/v1/responses", "/v1/messages"]

    def test_deep_research_model_pricing(self, model_cost: dict):
        info = model_cost["apodex/apodex-1-1-deep-research"]
        assert info["max_input_tokens"] == 131072
        assert info["max_output_tokens"] == 65536
        assert info["input_cost_per_token"] == 5e-06
        assert info["output_cost_per_token"] == 2e-05
        assert info["supports_function_calling"] is False
        assert info["supports_response_schema"] is False
        assert info["supports_prompt_caching"] is False
        assert info["supports_web_search"] is True

    @pytest.mark.parametrize("model", DEEP_RESEARCH_MODELS)
    def test_deep_research_models_are_not_on_the_native_messages_path(self, model_cost: dict, model: str):
        """Apodex serves /v1/messages for the core models only."""
        assert "/v1/messages" not in model_cost[f"apodex/{model}"]["supported_endpoints"]

    def test_discover_is_responses_only(self, model_cost: dict):
        """The Discover tiers answer 400 unsupported_api on /v1/chat/completions."""
        assert model_cost["apodex/apodex-1-1-deep-discover"]["supported_endpoints"] == ["/v1/responses"]

    @pytest.mark.parametrize("model", ("apodex-1-1-deep-research", "apodex-1-1-deep-solve"))
    def test_the_other_deep_tiers_keep_chat_completions(self, model_cost: dict, model: str):
        assert model_cost[f"apodex/{model}"]["supported_endpoints"] == [
            "/v1/chat/completions",
            "/v1/responses",
        ]

    def test_backup_cost_map_in_sync(self, model_cost: dict):
        with open(REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json") as f:
            backup = json.load(f)
        for key in (key for key in model_cost if key.startswith("apodex/")):
            assert backup[key] == model_cost[key], f"{key} differs between main and backup cost maps"
