"""
Tests for app.nz provider configuration and resolution.
"""

import litellm


class TestAppNZProviderConfig:
    """Test app.nz JSON-configured provider"""

    def test_appnz_json_config_exists(self):
        """app_nz is registered in providers.json with the expected shape"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("app_nz")

        app_nz = JSONProviderRegistry.get("app_nz")
        assert app_nz is not None
        assert app_nz.base_url == "https://app.nz/v1"
        assert app_nz.api_key_env == "APP_NZ_API_KEY"
        assert app_nz.api_base_env == "APP_NZ_API_BASE"
        assert app_nz.base_class == "openai_gpt"
        assert app_nz.extra_supported_params == ["reasoning_effort"]

    def test_appnz_provider_resolution(self):
        """Resolving an app_nz/* model returns the default app.nz base URL"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="app_nz/app/auto",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "app/auto"
        assert provider == "app_nz"
        assert api_base == "https://app.nz/v1"

    def test_appnz_api_base_override(self):
        """An explicit api_base / api_key overrides the app.nz defaults"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="app_nz/app/auto-code",
            custom_llm_provider=None,
            api_base="https://custom.example.com/v1",
            api_key="app_live_test",
        )

        assert provider == "app_nz"
        assert api_base == "https://custom.example.com/v1"
        assert api_key == "app_live_test"

    def test_appnz_dynamic_config(self):
        """The generated dynamic config resolves to the app.nz base URL"""
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("app_nz")
        config_class = create_config_class(provider)
        config = config_class()

        api_base, _ = config._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://app.nz/v1"

    def test_appnz_complete_url_appends_endpoint(self):
        """get_complete_url appends the chat completions path to the app.nz base"""
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("app_nz")
        config_class = create_config_class(provider)
        config = config_class()

        url = config.get_complete_url(
            api_base="https://app.nz/v1",
            api_key="app_live_test",
            model="app_nz/app/auto",
            optional_params={},
            litellm_params={},
            stream=False,
        )

        assert url == "https://app.nz/v1/chat/completions"

    def test_appnz_supports_reasoning_effort(self):
        """app.nz forwards its documented reasoning_effort parameter"""
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("app_nz")
        config_class = create_config_class(provider)
        config = config_class()

        supported_params = config.get_supported_openai_params(model="app/auto")
        assert "reasoning_effort" in supported_params

        optional_params = config.map_openai_params(
            non_default_params={"reasoning_effort": "auto"},
            optional_params={},
            model="app/auto",
            drop_params=False,
        )

        assert optional_params["reasoning_effort"] == "auto"
