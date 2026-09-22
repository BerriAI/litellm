"""
Tests for Yolo-Auto provider configuration and integration.
"""

import litellm


class TestYoloAutoProviderConfig:
    def test_yolo_auto_in_provider_list(self):
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "YOLO_AUTO")
        assert LlmProviders.YOLO_AUTO.value == "yolo-auto"
        assert "yolo-auto" in litellm.provider_list

    def test_yolo_auto_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("yolo-auto")

        yolo = JSONProviderRegistry.get("yolo-auto")
        assert yolo is not None
        assert yolo.base_url == "https://yolo-auto.com/v1"
        assert yolo.api_key_env == "YOLO_AUTO_API_KEY"
        assert yolo.param_mappings.get("max_completion_tokens") == "max_tokens"

    def test_yolo_auto_in_openai_compatible_providers(self):
        from litellm.constants import openai_compatible_providers

        assert "yolo-auto" in openai_compatible_providers

    def test_yolo_auto_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="yolo-auto/yolo",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "yolo"
        assert provider == "yolo-auto"
        assert api_base == "https://yolo-auto.com/v1"

    def test_yolo_auto_api_base_override(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="yolo-auto/yolo",
            custom_llm_provider=None,
            api_base="https://custom.yolo-auto.com/v1",
            api_key="sk-test",
        )

        assert provider == "yolo-auto"
        assert api_base == "https://custom.yolo-auto.com/v1"
        assert api_key == "sk-test"

    def test_yolo_auto_url_autodetection(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="yolo",
            custom_llm_provider=None,
            api_base="https://yolo-auto.com/v1",
            api_key=None,
        )
        assert provider == "yolo-auto"
        assert api_base == "https://yolo-auto.com/v1"

    def test_yolo_auto_max_completion_tokens_mapped(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("yolo-auto")
        assert provider is not None
        config = create_config_class(provider)()

        optional_params = config.map_openai_params(
            non_default_params={"max_completion_tokens": 256},
            optional_params={},
            model="yolo",
            drop_params=False,
        )
        assert optional_params["max_tokens"] == 256
        assert "max_completion_tokens" not in optional_params

    def test_yolo_auto_router_config(self):
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "yolo-chat",
                    "litellm_params": {
                        "model": "yolo-auto/yolo",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "yolo-chat"


class TestYoloAutoModelMetadata:
    YOLO_AUTO_MODELS = (
        "yolo-auto/yolo",
        "yolo-auto/yolo-small",
    )
    VISION_MODELS = ("yolo-auto/yolo",)

    @staticmethod
    def _load(path_parts):
        import json
        from pathlib import Path

        json_path = Path(__file__).parents[4].joinpath(*path_parts)
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)

    def test_yolo_auto_models_synced_to_backup(self):
        model_cost = self._load(("model_prices_and_context_window.json",))
        backup = self._load(("litellm", "model_prices_and_context_window_backup.json"))
        for model in self.YOLO_AUTO_MODELS:
            assert model in backup, f"{model} missing from backup json"
            assert backup[model] == model_cost[model], f"{model} differs between root and backup json"

    def test_yolo_auto_model_rows_are_chat_mode(self):
        model_cost = self._load(("model_prices_and_context_window.json",))
        for model in self.YOLO_AUTO_MODELS:
            row = model_cost[model]
            assert row["litellm_provider"] == "yolo-auto"
            assert row["mode"] == "chat"
            assert row["max_input_tokens"] == 262144
        for model in self.VISION_MODELS:
            assert model_cost[model]["supports_vision"] is True
        for model in set(self.YOLO_AUTO_MODELS) - set(self.VISION_MODELS):
            assert model_cost[model]["supports_vision"] is False


class TestYoloAutoDashboardRegistration:
    @staticmethod
    def _provider_create_fields():
        import json
        from pathlib import Path

        import litellm

        path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def test_yolo_auto_is_selectable_in_the_add_model_form(self):
        entries = [e for e in self._provider_create_fields() if e["litellm_provider"] == "yolo-auto"]
        assert len(entries) == 1, "yolo-auto must appear exactly once in provider_create_fields.json"

        entry = entries[0]
        assert entry["provider"] == "YOLO_AUTO"
        assert entry["provider_display_name"] == "Yolo-Auto"
        assert entry["default_model_placeholder"].startswith("yolo-auto/")

        fields = {f["key"]: f for f in entry["credential_fields"]}
        assert fields["api_key"]["required"] is True
        assert fields["api_key"]["field_type"] == "password"
        assert fields["api_base"]["required"] is False

    def test_yolo_auto_is_documented_in_provider_endpoints_support(self):
        import json
        from pathlib import Path

        json_path = Path(__file__).parents[4].joinpath("provider_endpoints_support.json")
        with open(json_path, encoding="utf-8") as f:
            support = json.load(f)

        entry = support["providers"]["yolo-auto"]
        assert entry["endpoints"]["chat_completions"] is True
        assert entry["endpoints"]["messages"] is False
