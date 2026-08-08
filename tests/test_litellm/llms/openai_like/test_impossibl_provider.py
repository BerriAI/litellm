"""
Tests for Impossibl provider configuration and integration.
"""

import pytest

import litellm


@pytest.fixture
def local_model_cost_map(monkeypatch):
    """Assert against the in-repo cost map, not the copy fetched from GitHub main.

    Without this the cost-map assertions would depend on network state and could not
    pass until these entries had already been merged upstream.
    """
    original_model_cost = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original_model_cost
        litellm.get_model_info.cache_clear()


class TestImpossiblProviderConfig:
    """Test Impossibl provider configuration"""

    def test_impossibl_in_provider_list(self):
        """Test that impossibl is in the provider list"""
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "IMPOSSIBL")
        assert LlmProviders.IMPOSSIBL.value == "impossibl"
        assert "impossibl" in litellm.provider_list

    def test_impossibl_json_config_exists(self):
        """Test that impossibl is configured in providers.json"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("impossibl")

        impossibl = JSONProviderRegistry.get("impossibl")
        assert impossibl is not None
        assert impossibl.base_url == "https://api.impossibl.com/v1"
        assert impossibl.api_key_env == "IMPOSSIBL_API_KEY"
        assert impossibl.api_base_env == "IMPOSSIBL_API_BASE"

    def test_impossibl_provider_resolution(self):
        """Provider resolution splits on the first '/' only.

        Impossibl model ids are themselves `creator/model` (the OpenRouter convention),
        so the gateway must receive the full `anthropic/claude-opus-4-8` string.
        """
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="impossibl/anthropic/claude-opus-4-8",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "anthropic/claude-opus-4-8"
        assert provider == "impossibl"
        assert api_base == "https://api.impossibl.com/v1"

    def test_impossibl_api_base_override(self):
        """Test that an explicit api_base / api_key overrides the default"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="impossibl/anthropic/claude-opus-4-8",
            custom_llm_provider=None,
            api_base="https://custom.example.com/v1",
            api_key="sk-test",
        )

        assert provider == "impossibl"
        assert api_base == "https://custom.example.com/v1"
        assert api_key == "sk-test"

    def test_impossibl_model_cost_map(self, local_model_cost_map):
        """Test that impossibl models are present in the model cost map"""
        model_cost = litellm.model_cost

        assert "impossibl/anthropic/claude-opus-4-8" in model_cost
        info = model_cost["impossibl/anthropic/claude-opus-4-8"]
        assert info["litellm_provider"] == "impossibl"
        assert info["mode"] == "chat"
        assert info["max_input_tokens"] == 1000000
        assert info["max_output_tokens"] == 128000
        # priced per token, converted from the provider's per-1M list price
        assert info["input_cost_per_token"] == 5e-06
        assert info["output_cost_per_token"] == 2.5e-05
        assert info["cache_read_input_token_cost"] == 5e-07
        assert info["supports_prompt_caching"] is True
        assert info["supports_reasoning"] is True

    def test_impossibl_router_config(self):
        """Test that impossibl can be used in Router configuration"""
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "impossibl-chat",
                    "litellm_params": {
                        "model": "impossibl/anthropic/claude-opus-4-8",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "impossibl-chat"

    def test_impossibl_responses_api_supported(self):
        """/v1/responses is declared, so the generated responses config is available."""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.supports_responses_api("impossibl")

    def test_impossibl_native_anthropic_messages_passthrough(self):
        """Declaring /v1/messages must resolve the native passthrough config.

        Impossibl exposes the Anthropic Messages API natively, so the payload is
        forwarded untranslated rather than being converted to chat/completions --
        which is what preserves cache_control / thinking / tool blocks.
        """
        from litellm.llms.openai_like.messages.transformation import (
            JSONProviderAnthropicMessagesConfig,
        )
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_anthropic_messages_config(
            model="anthropic/claude-opus-4-8",
            provider=litellm.LlmProviders.IMPOSSIBL,
        )

        assert isinstance(config, JSONProviderAnthropicMessagesConfig)
        assert config.custom_llm_provider == "impossibl"

    def test_impossibl_cost_map_root_and_backup_agree(self):
        """The shipped backup must carry the same entries as the repo-root cost map.

        litellm serves pricing from the packaged backup when the remote map is
        unavailable, so an entry present in only one file would price differently
        depending on network state.
        """
        import json
        from pathlib import Path

        import litellm as _litellm

        root = json.loads(
            (Path(_litellm.__file__).parents[1] / "model_prices_and_context_window.json").read_text()
        )
        backup = json.loads(
            (Path(_litellm.__file__).parent / "model_prices_and_context_window_backup.json").read_text()
        )

        root_impossibl = {k: v for k, v in root.items() if k.startswith("impossibl/")}
        backup_impossibl = {k: v for k, v in backup.items() if k.startswith("impossibl/")}

        assert root_impossibl, "no impossibl entries in model_prices_and_context_window.json"
        assert root_impossibl == backup_impossibl

    def test_impossibl_supported_endpoints_matrix(self):
        """The runtime-served backup matrix (GET /public/supported_endpoints) lists impossibl."""
        import json
        from pathlib import Path

        import litellm as _litellm

        backup_path = (
            Path(_litellm.__file__).parent / "provider_endpoints_support_backup.json"
        )
        matrix = json.loads(backup_path.read_text())

        assert "impossibl" in matrix["providers"]
        endpoints = matrix["providers"]["impossibl"]["endpoints"]
        assert endpoints["chat_completions"] is True
        assert endpoints["responses"] is True
        # Impossibl exposes the Anthropic Messages API natively, so /v1/messages is
        # forwarded untranslated rather than being bridged from chat/completions.
        assert endpoints["messages"] is True
