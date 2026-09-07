import litellm


JALAPENO_BASE_URL = "https://api.jalapeno-cloud.ai/v1"


class TestJalapenoProviderConfig:
    def test_jalapeno_in_provider_list(self):
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "JALAPENO")
        assert LlmProviders.JALAPENO.value == "jalapeno"
        assert "jalapeno" in litellm.provider_list

    def test_jalapeno_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("jalapeno")
        provider = JSONProviderRegistry.get("jalapeno")
        assert provider is not None
        assert provider.base_url == JALAPENO_BASE_URL
        assert provider.api_key_env == "JALAPENO_API_KEY"
        assert provider.api_base_env == "JALAPENO_API_BASE"
        assert provider.param_mappings.get("max_completion_tokens") == "max_tokens"
        assert provider.supported_endpoints == [
            "/v1/chat/completions",
            "/v1/responses",
            "/v1/messages",
        ]

    def test_jalapeno_in_openai_compatible_lists(self):
        from litellm.constants import (
            openai_compatible_endpoints,
            openai_compatible_providers,
        )

        assert "jalapeno" in openai_compatible_providers
        assert JALAPENO_BASE_URL in openai_compatible_endpoints

    def test_jalapeno_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="jalapeno/DeepSeek-V4-Flash",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )
        assert model == "DeepSeek-V4-Flash"
        assert provider == "jalapeno"
        assert api_base == JALAPENO_BASE_URL

    def test_jalapeno_api_base_inference(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="DeepSeek-V4-Flash",
            custom_llm_provider=None,
            api_base=JALAPENO_BASE_URL,
            api_key="sk-test",
        )
        assert model == "DeepSeek-V4-Flash"
        assert provider == "jalapeno"
        assert api_base == JALAPENO_BASE_URL
        assert api_key == "sk-test"

    def test_jalapeno_maps_max_completion_tokens(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("jalapeno")
        assert provider is not None
        config = create_config_class(provider)()
        params = config.map_openai_params(
            non_default_params={"max_completion_tokens": 256},
            optional_params={},
            model="jalapeno/DeepSeek-V4-Flash",
            drop_params=False,
        )
        assert params.get("max_tokens") == 256
        assert "max_completion_tokens" not in params

    def test_jalapeno_chat_complete_url(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("jalapeno")
        assert provider is not None
        config = create_config_class(provider)()
        url = config.get_complete_url(
            api_base=JALAPENO_BASE_URL,
            api_key="sk-test",
            model="jalapeno/DeepSeek-V4-Flash",
            optional_params={},
            litellm_params={},
            stream=False,
        )
        assert url == f"{JALAPENO_BASE_URL}/chat/completions"

    def test_jalapeno_model_cost_map(self):
        info = litellm.model_cost["jalapeno/DeepSeek-V4-Flash"]
        assert info["litellm_provider"] == "jalapeno"
        assert info["mode"] == "chat"
        assert info["input_cost_per_token"] == 1.4e-07
        assert info["output_cost_per_token"] == 2.8e-07
        assert info["max_input_tokens"] == 1048576
        assert info["supports_function_calling"] is True
        assert info["supports_reasoning"] is True
        assert "/v1/chat/completions" in info["supported_endpoints"]
        assert "/v1/responses" in info["supported_endpoints"]
        assert "/v1/messages" in info["supported_endpoints"]

    def test_jalapeno_supported_endpoints_matrix(self):
        import json
        from pathlib import Path

        backup_path = (
            Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
        )
        matrix = json.loads(backup_path.read_text())
        assert "jalapeno" in matrix["providers"]
        endpoints = matrix["providers"]["jalapeno"]["endpoints"]
        assert endpoints["chat_completions"] is True
        assert endpoints["messages"] is True
        assert endpoints["responses"] is True


class TestJalapenoResponsesAndMessages:
    def test_jalapeno_responses_config_url_and_auth(self, monkeypatch):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry
        from litellm.utils import ProviderConfigManager

        assert JSONProviderRegistry.supports_responses_api("jalapeno") is True
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider="jalapeno",
            model="jalapeno/DeepSeek-V4-Flash",
        )
        assert config is not None
        assert config.custom_llm_provider == "jalapeno"
        assert (
            config.get_complete_url(api_base=None, litellm_params={})
            == f"{JALAPENO_BASE_URL}/responses"
        )

        monkeypatch.setenv("JALAPENO_API_KEY", "sk-env-key")
        headers = config.validate_environment(
            headers={}, model="DeepSeek-V4-Flash", litellm_params=None
        )
        assert headers["Authorization"] == "Bearer sk-env-key"

    def test_jalapeno_messages_config_url_and_auth(self, monkeypatch):
        from litellm.llms.openai_like.messages.transformation import (
            JSONProviderAnthropicMessagesConfig,
        )

        cfg = litellm.ProviderConfigManager.get_provider_anthropic_messages_config(
            model="DeepSeek-V4-Flash",
            provider=litellm.LlmProviders.JALAPENO,
        )
        assert isinstance(cfg, JSONProviderAnthropicMessagesConfig)
        assert (
            cfg.get_complete_url(
                api_base=None,
                api_key="sk-test",
                model="DeepSeek-V4-Flash",
                optional_params={},
                litellm_params={},
            )
            == f"{JALAPENO_BASE_URL}/messages"
        )

        monkeypatch.setenv("JALAPENO_API_KEY", "sk-env-key")
        headers, _ = cfg.validate_anthropic_messages_environment(
            headers={},
            model="DeepSeek-V4-Flash",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base=None,
        )
        assert headers["authorization"] == "Bearer sk-env-key"
        assert headers["anthropic-version"] == "2023-06-01"
