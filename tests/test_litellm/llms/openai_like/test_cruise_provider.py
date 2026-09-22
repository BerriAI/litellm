import litellm


class TestCruiseProviderConfig:
    def test_cruise_in_provider_list(self):
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "CRUISE")
        assert LlmProviders.CRUISE.value == "cruise"
        assert "cruise" in litellm.provider_list

    def test_cruise_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("cruise")

        cruise = JSONProviderRegistry.get("cruise")
        assert cruise is not None
        assert cruise.base_url == "https://cruise.bytesbrains.net/v1"
        assert cruise.api_key_env == "CRUISE_API_KEY"
        assert cruise.api_base_env == "CRUISE_API_BASE"

    def test_cruise_not_routed_to_openai_audio_transcription(self):
        from litellm.constants import OPENAI_AUDIO_TRANSCRIPTION_PROVIDERS, openai_compatible_providers

        assert "cruise" not in openai_compatible_providers
        assert "cruise" not in OPENAI_AUDIO_TRANSCRIPTION_PROVIDERS

    def test_cruise_provider_resolution_keeps_upstream_prefix(self, monkeypatch):
        monkeypatch.delenv("CRUISE_API_BASE", raising=False)
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="cruise/anthropic/claude-sonnet-5",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "anthropic/claude-sonnet-5"
        assert provider == "cruise"
        assert api_base == "https://cruise.bytesbrains.net/v1"

    def test_cruise_api_base_override(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="cruise/deepseek/deepseek-flash",
            custom_llm_provider=None,
            api_base="https://cruise-demo.bytesbrains.net/v1",
            api_key="cru_test",
        )

        assert model == "deepseek/deepseek-flash"
        assert provider == "cruise"
        assert api_base == "https://cruise-demo.bytesbrains.net/v1"
        assert api_key == "cru_test"

    def test_cruise_api_base_env_override(self, monkeypatch):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.setenv("CRUISE_API_BASE", "https://cruise-demo.bytesbrains.net/v1")
        monkeypatch.setenv("CRUISE_API_KEY", "cru_test")

        model, provider, api_key, api_base = get_llm_provider(
            model="cruise/deepseek/deepseek-flash",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert provider == "cruise"
        assert api_base == "https://cruise-demo.bytesbrains.net/v1"
        assert api_key == "cru_test"

    def test_cruise_max_completion_tokens_passed_through(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("cruise")
        assert provider is not None
        config = create_config_class(provider)()

        optional_params = config.map_openai_params(
            non_default_params={"max_completion_tokens": 256},
            optional_params={},
            model="anthropic/claude-sonnet-5",
            drop_params=False,
        )
        assert optional_params["max_completion_tokens"] == 256

    def test_cruise_router_config(self):
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "cruise-chat",
                    "litellm_params": {
                        "model": "cruise/anthropic/claude-sonnet-5",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "cruise-chat"


class TestCruiseEndpointSupport:
    @staticmethod
    def _load(path_parts):
        import json
        from pathlib import Path

        json_path = Path(__file__).parents[4].joinpath(*path_parts)
        with open(json_path) as f:
            return json.load(f)

    def test_cruise_endpoint_support_synced_to_backup(self):
        root = self._load(("provider_endpoints_support.json",))["providers"]
        backup = self._load(("litellm", "provider_endpoints_support_backup.json"))["providers"]
        assert root["cruise"] == backup["cruise"]
        assert root["cruise"]["endpoints"]["chat_completions"] is True
        assert root["cruise"]["endpoints"]["responses"] is False


class TestCruiseDashboardRegistration:
    def test_cruise_is_selectable_in_the_add_model_form(self):
        import json
        from pathlib import Path

        path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
        with open(path) as f:
            entries = [e for e in json.load(f) if e["litellm_provider"] == "cruise"]

        assert len(entries) == 1
        entry = entries[0]
        assert entry["provider"] == "CRUISE"
        assert entry["default_model_placeholder"].startswith("cruise/")

        fields = {f["key"]: f for f in entry["credential_fields"]}
        assert fields["api_key"]["required"] is True
        assert fields["api_key"]["field_type"] == "password"
        assert fields["api_base"]["required"] is False
