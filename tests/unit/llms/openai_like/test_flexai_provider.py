"""
Tests for FlexAI provider configuration and integration.
"""

import json
import os

import litellm

# a flagship model FlexAI pins always-on; used only for provider-resolution
# assertions, which do not require the model to be reachable
FLEXAI_MODEL = "gpt-oss-120b"
FLEXAI_BASE_URL = "https://api.flex.ai/v1"


class TestFlexAIProviderConfig:
    """Test FlexAI provider configuration"""

    def test_flexai_in_provider_list(self):
        """Test that flexai is in the provider list"""
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "FLEXAI")
        assert LlmProviders.FLEXAI.value == "flexai"
        assert "flexai" in litellm.provider_list

    def test_flexai_json_config_exists(self):
        """Test that flexai is configured in providers.json"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("flexai")

        flexai = JSONProviderRegistry.get("flexai")
        assert flexai is not None
        assert flexai.base_url == FLEXAI_BASE_URL
        assert flexai.api_key_env == "FLEXAI_API_KEY"
        assert flexai.api_base_env == "FLEXAI_API_BASE"

    def test_flexai_supports_responses_api(self):
        """Test that flexai declares Responses API support"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.supports_responses_api("flexai")

    def test_flexai_in_openai_compatible_providers(self):
        """Test that flexai is in the openai_compatible_providers list"""
        from litellm.constants import (
            openai_compatible_endpoints,
            openai_compatible_providers,
        )

        assert "flexai" in openai_compatible_providers
        assert FLEXAI_BASE_URL in openai_compatible_endpoints

    def test_flexai_provider_resolution(self):
        """Test that provider resolution finds flexai and returns the default base URL"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model=f"flexai/{FLEXAI_MODEL}",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == FLEXAI_MODEL
        assert provider == "flexai"
        assert api_base == FLEXAI_BASE_URL

    def test_flexai_api_base_override(self):
        """Test that an explicit api_base / api_key overrides the default"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model=f"flexai/{FLEXAI_MODEL}",
            custom_llm_provider=None,
            api_base="https://tokens.staging.flex.ai/v1",
            api_key="sk-test",
        )

        assert provider == "flexai"
        assert api_base == "https://tokens.staging.flex.ai/v1"
        assert api_key == "sk-test"

    def test_flexai_env_key_resolution(self, monkeypatch):
        """Test that the key is read from FLEXAI_API_KEY when not passed explicitly"""
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        monkeypatch.setenv("FLEXAI_API_KEY", "sk-flexai-env")

        config = create_config_class(JSONProviderRegistry.get("flexai"))()
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)

        assert api_base == FLEXAI_BASE_URL
        assert api_key == "sk-flexai-env"

    def test_flexai_url_autodetection(self):
        """Test that api_base=api.flex.ai/v1 auto-sets custom_llm_provider=flexai"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model=FLEXAI_MODEL,
            custom_llm_provider=None,
            api_base=FLEXAI_BASE_URL,
            api_key=None,
        )

        assert provider == "flexai"
        assert api_base == FLEXAI_BASE_URL

    def test_flexai_url_autodetection_prefers_caller_key(self, monkeypatch):
        """An explicit api_key must win over the server's FLEXAI_API_KEY.

        `completion()` overwrites api_key with dynamic_api_key whenever it is
        set, so reading the environment unconditionally here would spend the
        server's credential on a caller-supplied api_base.
        """
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.setenv("FLEXAI_API_KEY", "sk-server-env")

        _, provider, api_key, _ = get_llm_provider(
            model=FLEXAI_MODEL,
            custom_llm_provider=None,
            api_base=FLEXAI_BASE_URL,
            api_key="sk-caller",
        )

        assert provider == "flexai"
        assert api_key == "sk-caller"

    def test_flexai_url_autodetection_falls_back_to_env_key(self, monkeypatch):
        """With no explicit key, the server environment is still used"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.setenv("FLEXAI_API_KEY", "sk-server-env")

        _, provider, api_key, _ = get_llm_provider(
            model=FLEXAI_MODEL,
            custom_llm_provider=None,
            api_base=FLEXAI_BASE_URL,
            api_key=None,
        )

        assert provider == "flexai"
        assert api_key == "sk-server-env"

    def test_flexai_router_config(self):
        """Test that flexai can be used in Router configuration"""
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "flexai-chat",
                    "litellm_params": {
                        "model": f"flexai/{FLEXAI_MODEL}",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "flexai-chat"


class TestFlexAICostMap:
    """Test the FlexAI entries in the model cost map"""

    @staticmethod
    def _cost_map():
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../..")
        )
        with open(os.path.join(repo_root, "model_prices_and_context_window.json")) as f:
            return json.load(f)

    @staticmethod
    def _backup_cost_map():
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../..")
        )
        path = os.path.join(
            repo_root, "litellm", "model_prices_and_context_window_backup.json"
        )
        with open(path) as f:
            return json.load(f)

    def test_flexai_entries_present(self):
        """Test that flexai models are priced in the cost map"""
        entries = {
            k: v for k, v in self._cost_map().items() if k.startswith("flexai/")
        }

        assert entries, "no flexai/* entries in the cost map"
        for key, entry in entries.items():
            assert entry["litellm_provider"] == "flexai", key
            assert entry.get("mode"), key

    def test_flexai_chat_entries_are_priced(self):
        """Test that every flexai chat model carries per-token pricing + context"""
        entries = {
            k: v
            for k, v in self._cost_map().items()
            if k.startswith("flexai/") and v.get("mode") == "chat"
        }

        assert entries, "no flexai/* chat entries in the cost map"
        for key, entry in entries.items():
            assert entry["input_cost_per_token"] > 0, key
            assert entry["output_cost_per_token"] > 0, key
            assert entry["max_input_tokens"] > 0, key
            assert entry["max_output_tokens"] > 0, key

    def test_flexai_entries_match_packaged_backup(self):
        """Test that the packaged backup carries the same flexai entries.

        litellm loads the packaged backup when the remote cost map is
        unavailable (or when LITELLM_LOCAL_MODEL_COST_MAP=True), so the two
        files must not drift.
        """
        root = {k: v for k, v in self._cost_map().items() if k.startswith("flexai/")}
        backup = {
            k: v for k, v in self._backup_cost_map().items() if k.startswith("flexai/")
        }

        assert root == backup

    def test_flexai_non_chat_entries_use_calculator_readable_fields(self):
        """Non-chat entries must carry the cost fields the generic calculators read.

        `default_image_cost_calculator` reads `input_cost_per_image` /
        `input_cost_per_pixel`, and `select_cost_metric_for_model` requires
        `input_cost_per_character` (or `input_cost_per_token`) for speech. Using
        the `output_*` variants instead makes cost calculation raise rather than
        return the configured cost.
        """
        readable_fields = {
            "image_generation": ("input_cost_per_image", "input_cost_per_pixel"),
            "audio_speech": ("input_cost_per_character", "input_cost_per_token"),
            "audio_transcription": ("input_cost_per_second", "input_cost_per_token"),
            "embedding": ("input_cost_per_token",),
        }

        entries = {
            k: v
            for k, v in self._cost_map().items()
            if k.startswith("flexai/") and v.get("mode") != "chat"
        }
        assert entries, "no non-chat flexai/* entries in the cost map"

        for key, entry in entries.items():
            options = readable_fields.get(entry["mode"])
            assert options, f"{key}: unhandled mode {entry['mode']}"
            assert any(field in entry for field in options), (
                f"{key} (mode={entry['mode']}) has none of {options}"
            )

    def test_flexai_image_cost_calculation(self):
        """Test that image cost resolves through the generic image calculator"""
        from litellm.cost_calculator import default_image_cost_calculator

        cost_map = self._cost_map()
        key = next(
            k
            for k, v in cost_map.items()
            if k.startswith("flexai/") and v.get("mode") == "image_generation"
        )
        per_image = cost_map[key]["input_cost_per_image"]

        for n in (1, 3):
            cost = default_image_cost_calculator(
                model=key, custom_llm_provider="flexai", n=n, quality=None, size=None
            )
            assert cost == per_image * n

    def test_flexai_speech_cost_calculation(self):
        """Test that TTS cost resolves per input character"""
        from litellm.cost_calculator import cost_per_token
        from litellm.litellm_core_utils.llm_cost_calc.utils import (
            select_cost_metric_for_model,
        )

        cost_map = self._cost_map()
        key = next(
            k
            for k, v in cost_map.items()
            if k.startswith("flexai/") and v.get("mode") == "audio_speech"
        )
        model = key.split("/", 1)[1]
        per_character = cost_map[key]["input_cost_per_character"]

        model_info = litellm.get_model_info(model=model, custom_llm_provider="flexai")
        assert select_cost_metric_for_model(model_info) == "cost_per_character"

        prompt_cost, completion_cost = cost_per_token(
            model=model,
            custom_llm_provider="flexai",
            call_type="speech",
            prompt_characters=1000,
        )
        assert prompt_cost + completion_cost == per_character * 1000

    def test_flexai_cost_calculation(self):
        """Test that completion_cost resolves for a flexai chat model"""
        from litellm.types.utils import Choices, Message, ModelResponse, Usage

        cost_map = self._cost_map()
        key = next(
            k
            for k, v in cost_map.items()
            if k.startswith("flexai/") and v.get("mode") == "chat"
        )
        entry = cost_map[key]

        response = ModelResponse(
            model=key.split("/", 1)[1],
            choices=[Choices(message=Message(role="assistant", content="ok"))],
            usage=Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
        )
        response._hidden_params = {"custom_llm_provider": "flexai"}

        cost = litellm.completion_cost(
            completion_response=response, custom_llm_provider="flexai", model=key
        )

        expected = (
            1000 * entry["input_cost_per_token"] + 500 * entry["output_cost_per_token"]
        )
        assert cost == expected
