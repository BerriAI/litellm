import litellm


class TestSaladCloudProviderConfig:
    def test_provider_registration(self):
        from litellm import LlmProviders

        assert LlmProviders.SALAD_CLOUD.value == "salad_cloud"
        assert "salad_cloud" in litellm.provider_list

    def test_json_config(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("salad_cloud")
        provider = JSONProviderRegistry.get("salad_cloud")
        assert provider is not None
        assert provider.base_url == "https://ai.salad.cloud/v1"
        assert provider.api_key_env == "SALAD_CLOUD_API_KEY"
        assert provider.api_base_env == "SALAD_CLOUD_API_BASE"
        assert provider.param_mappings.get("max_completion_tokens") == "max_tokens"

    def test_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, _, api_base = get_llm_provider(
            model="salad_cloud/qwen3.6-35b-a3b",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "qwen3.6-35b-a3b"
        assert provider == "salad_cloud"
        assert api_base == "https://ai.salad.cloud/v1"

    def test_api_base_override(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        _, provider, api_key, api_base = get_llm_provider(
            model="salad_cloud/qwen3.6-35b-a3b",
            custom_llm_provider=None,
            api_base="https://custom.example.com/v1",
            api_key="test-key",
        )

        assert provider == "salad_cloud"
        assert api_base == "https://custom.example.com/v1"
        assert api_key == "test-key"

    def test_model_metadata(self):
        import json
        from pathlib import Path

        cost_map_path = (
            Path(litellm.__file__).parent
            / "model_prices_and_context_window_backup.json"
        )
        info = json.loads(cost_map_path.read_text())["salad_cloud/qwen3.6-35b-a3b"]

        assert info["litellm_provider"] == "salad_cloud"
        assert info["mode"] == "chat"
        assert info["max_input_tokens"] == 262144
        assert info["max_output_tokens"] == 262144
        assert info["input_cost_per_token"] == 0.09e-6
        assert info["output_cost_per_token"] == 0.60e-6
        assert info["supports_vision"] is True
        assert info["supports_reasoning"] is True
        assert info["supports_response_schema"] is True
        assert info["supports_native_streaming"] is True

    def test_router_config(self):
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "salad-chat",
                    "litellm_params": {
                        "model": "salad_cloud/qwen3.6-35b-a3b",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert router.model_list[0]["model_name"] == "salad-chat"

    def test_supported_endpoints_matrix(self):
        import json
        from pathlib import Path

        backup_path = (
            Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
        )
        matrix = json.loads(backup_path.read_text())

        endpoints = matrix["providers"]["salad_cloud"]["endpoints"]
        assert endpoints["chat_completions"] is True
        assert endpoints["messages"] is True
        assert endpoints["responses"] is False
