"""
Tests for LibertAI provider configuration and integration.
"""

import litellm


class TestLibertAIProviderConfig:
    """Test LibertAI provider configuration"""

    def test_libertai_in_provider_list(self):
        """Test that libertai is in the provider list"""
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "LIBERTAI")
        assert LlmProviders.LIBERTAI.value == "libertai"
        assert "libertai" in litellm.provider_list

    def test_libertai_json_config_exists(self):
        """Test that libertai is configured in providers.json"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("libertai")

        libertai = JSONProviderRegistry.get("libertai")
        assert libertai is not None
        assert libertai.base_url == "https://api.libertai.io/v1"
        assert libertai.api_key_env == "LIBERTAI_API_KEY"
        assert libertai.api_base_env == "LIBERTAI_API_BASE"
        assert libertai.param_mappings.get("max_completion_tokens") == "max_tokens"

    def test_libertai_provider_resolution(self):
        """Test that provider resolution finds libertai and the default base URL"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="libertai/qwen3.6-27b",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "qwen3.6-27b"
        assert provider == "libertai"
        assert api_base == "https://api.libertai.io/v1"

    def test_libertai_api_base_override(self):
        """Test that an explicit api_base / api_key overrides the default"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="libertai/qwen3.6-27b",
            custom_llm_provider=None,
            api_base="https://custom.example.com/v1",
            api_key="sk-test",
        )

        assert provider == "libertai"
        assert api_base == "https://custom.example.com/v1"
        assert api_key == "sk-test"

    def test_libertai_model_cost_map(self):
        """Test that libertai models are present in the model cost map"""
        model_cost = litellm.model_cost

        assert "libertai/qwen3.6-27b" in model_cost
        info = model_cost["libertai/qwen3.6-27b"]
        assert info["litellm_provider"] == "libertai"
        assert info["mode"] == "chat"
        assert info["max_input_tokens"] == 262144
        assert info["max_output_tokens"] == 262144

        # thinking variants are marked as reasoning models
        assert (
            model_cost["libertai/qwen3.6-27b-thinking"].get("supports_reasoning")
            is True
        )

    def test_libertai_router_config(self):
        """Test that libertai can be used in Router configuration"""
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "libertai-chat",
                    "litellm_params": {
                        "model": "libertai/qwen3.6-27b",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "libertai-chat"

    def test_libertai_model_modes(self):
        """Chat models carry mode 'chat'; the embedding model carries mode 'embedding'."""
        model_cost = litellm.model_cost

        # chat model
        assert model_cost["libertai/qwen3.6-27b"]["mode"] == "chat"

        # embedding model (bge-m3) must be normalized to mode 'embedding' so
        # /embeddings routing and the supported-endpoints matrix stay consistent
        assert "libertai/bge-m3" in model_cost
        bge = model_cost["libertai/bge-m3"]
        assert bge["litellm_provider"] == "libertai"
        assert bge["mode"] == "embedding"

    def test_libertai_supported_endpoints_matrix(self):
        """The runtime-served backup matrix (GET /public/supported_endpoints) lists libertai."""
        import json
        from pathlib import Path

        import litellm as _litellm

        backup_path = (
            Path(_litellm.__file__).parent / "provider_endpoints_support_backup.json"
        )
        matrix = json.loads(backup_path.read_text())

        assert "libertai" in matrix["providers"]
        endpoints = matrix["providers"]["libertai"]["endpoints"]
        assert endpoints["chat_completions"] is True
        # embeddings is advertised false: the JSON-configured-provider path only
        # wires chat routing (the OpenAILike embedding handler is reached solely
        # for the literal openai_like/llamafile/lm_studio providers), matching
        # the llamagate precedent. bge-m3 stays in the cost map for metadata.
        assert endpoints["embeddings"] is False
