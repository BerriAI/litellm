import json
from pathlib import Path

import pytest

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

STREAMLAKE_BASE_URL = "https://vanchin.streamlake.ai/api/gateway/v1/endpoints"
STREAMLAKE_CHAT_URL = f"{STREAMLAKE_BASE_URL}/chat/completions"

STREAMLAKE_MODELS = (
    ("streamlake/KAT-Coder-Pro-V2.5", 7.4e-07, 2.96e-06, 1.5e-07, None, 80000, 80000, None, None),
    ("streamlake/GLM-5.3", 1.4e-06, 4.4e-06, 2.6e-07, 1048576, 128000, 128000, 10, 300000),
    ("streamlake/GLM-5.3-Flash", 1.5e-07, 5e-07, 3e-08, 1048576, 128000, 128000, 30, 300000),
    ("streamlake/KAT-Coder-Air-V2.5", 1.5e-07, 6e-07, 3e-08, None, 80000, 80000, None, None),
    ("streamlake/Qwen3-Coder-Next", 3e-07, 1.5e-06, 6e-08, None, 64000, 64000, None, None),
    ("streamlake/KAT-Coder-Pro-V2", 3e-07, 1.2e-06, 6e-08, None, 80000, 80000, None, None),
    ("streamlake/Qwen3-Coder-480B-A35B-Instruct-FP8", 9e-07, 4.5e-06, 1.8e-07, 200000, 64000, 64000, None, None),
    ("streamlake/Qwen3-235B-A22B-Instruct-2507", 3.5e-07, 1.4e-06, 7e-08, 126000, 32000, 32000, None, None),
    ("streamlake/Qwen3-30B-A3B-Instruct-2507", 1.07e-07, 4.29e-07, 2.2e-08, 126000, 32000, 32000, None, None),
    ("streamlake/Kimi-K2-Instruct-0905", 5.71e-07, 2.286e-06, 1.43e-07, 224000, 32000, 32000, None, None),
    ("streamlake/DeepSeek-V3", 2.86e-07, 1.143e-06, 1.14e-07, 56000, 8000, 8000, None, None),
    ("streamlake/DeepSeek-V4-Pro-0813", 1.32e-06, 3.96e-06, 4.4e-08, None, 384000, 384000, 30, 600000),
    ("streamlake/deepseek-v4-flash-0731", 4.4e-07, 1.32e-06, 1.4e-08, None, 384000, 384000, 60, 600000),
    ("streamlake/Qwen3-30B-A3B", 1.07e-07, 1.071e-06, 4.3e-08, 30000, 8000, 8000, None, None),
    ("streamlake/KAT-Coder-Pro-V1", 3e-07, 1.2e-06, 6e-08, 256000, 32000, 32000, None, None),
)

PUBLIC_MODEL_IDS = tuple(row[0].split("/", 1)[1] for row in STREAMLAKE_MODELS)

STREAMLAKE_CAPABILITIES = {
    "streamlake/GLM-5.3": {"reasoning", "function_calling", "response_schema", "prefill", "prompt_caching", "text"},
    "streamlake/GLM-5.3-Flash": {
        "reasoning",
        "function_calling",
        "response_schema",
        "prefill",
        "prompt_caching",
        "vision",
        "text",
    },
    "streamlake/DeepSeek-V4-Pro-0813": {
        "reasoning",
        "function_calling",
        "response_schema",
        "prompt_caching",
        "text",
    },
    "streamlake/deepseek-v4-flash-0731": {"reasoning", "function_calling", "response_schema", "text"},
}


class TestStreamlakeProviderConfig:
    def test_streamlake_in_provider_list(self):
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "STREAMLAKE")
        assert LlmProviders.STREAMLAKE.value == "streamlake"
        assert "streamlake" in litellm.provider_list

    def test_streamlake_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("streamlake")

        streamlake = JSONProviderRegistry.get("streamlake")
        assert streamlake is not None
        assert streamlake.base_url == STREAMLAKE_BASE_URL
        assert streamlake.api_key_env == "STREAMLAKE_API_KEY"
        assert streamlake.api_base_env == "STREAMLAKE_API_BASE"

    def test_streamlake_is_chat_completions_only(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        streamlake = JSONProviderRegistry.get("streamlake")
        assert streamlake is not None
        assert streamlake.supported_endpoints == ["/v1/chat/completions"]
        assert JSONProviderRegistry.supports_responses_api("streamlake") is False

    def test_streamlake_in_openai_compatible_providers(self):
        from litellm.constants import openai_compatible_providers

        assert "streamlake" in openai_compatible_providers

    def test_streamlake_provider_resolution(self):
        model, provider, api_key, api_base = get_llm_provider(
            model="streamlake/ep-test-endpoint",
            custom_llm_provider=None,
            api_base=None,
            api_key="sk-test",
        )

        assert model == "ep-test-endpoint"
        assert provider == "streamlake"
        assert api_base == STREAMLAKE_BASE_URL
        assert api_key == "sk-test"

    def test_streamlake_api_base_override(self):
        model, provider, api_key, api_base = get_llm_provider(
            model="streamlake/ep-test-endpoint",
            custom_llm_provider=None,
            api_base="https://custom.streamlake.example/v1",
            api_key="sk-test",
        )

        assert model == "ep-test-endpoint"
        assert provider == "streamlake"
        assert api_base == "https://custom.streamlake.example/v1"
        assert api_key == "sk-test"

    def test_streamlake_url_autodetection(self):
        model, provider, api_key, api_base = get_llm_provider(
            model="some-public-model",
            custom_llm_provider=None,
            api_base=STREAMLAKE_BASE_URL,
            api_key=None,
        )
        assert provider == "streamlake"
        assert api_base == STREAMLAKE_BASE_URL

    def test_streamlake_api_key_resolved_from_env(self, monkeypatch):
        monkeypatch.setenv("STREAMLAKE_API_KEY", "sk-env-key")

        _, provider, api_key, _ = get_llm_provider(
            model="streamlake/ep-test-endpoint",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )
        assert provider == "streamlake"
        assert api_key == "sk-env-key"

    def test_streamlake_validate_environment_sets_bearer(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("streamlake")
        assert provider is not None
        config = create_config_class(provider)()

        headers = config.validate_environment(
            headers={},
            model="ep-test-endpoint",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            api_key="sk-resolved-key",
            api_base=None,
        )
        assert headers["Authorization"] == "Bearer sk-resolved-key"

    def test_streamlake_complete_url_appends_chat_completions(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("streamlake")
        assert provider is not None
        config = create_config_class(provider)()

        url = config.get_complete_url(
            api_base=None,
            api_key="sk-test",
            model="ep-test-endpoint",
            optional_params={},
            litellm_params={},
        )
        assert url == STREAMLAKE_CHAT_URL

    def test_streamlake_max_completion_tokens_mapped_to_max_tokens(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("streamlake")
        assert provider is not None
        config = create_config_class(provider)()

        optional_params = config.map_openai_params(
            non_default_params={"max_completion_tokens": 1024},
            optional_params={},
            model="ep-test-endpoint",
            drop_params=False,
        )
        assert optional_params == {"max_tokens": 1024}
        assert "max_completion_tokens" not in optional_params

    def test_streamlake_chat_params_pass_through(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("streamlake")
        assert provider is not None
        config = create_config_class(provider)()

        optional_params = config.map_openai_params(
            non_default_params={"temperature": 0.2, "top_p": 0.9, "stream": True},
            optional_params={},
            model="ep-test-endpoint",
            drop_params=False,
        )
        assert optional_params == {"temperature": 0.2, "top_p": 0.9, "stream": True}

    def test_streamlake_does_not_advertise_tools_or_reasoning_params(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("streamlake")
        assert provider is not None
        config = create_config_class(provider)()

        for model in ("ep-test-endpoint", "KAT-Coder-Pro-V2"):
            supported = config.get_supported_openai_params(model=model)
            assert "tools" not in supported
            assert "tool_choice" not in supported
            assert "reasoning_effort" not in supported


class TestStreamlakeNoHardcodedEndpointIds:
    def test_provider_config_never_hardcodes_an_endpoint_id(self):
        path = Path(litellm.__file__).parent / "llms" / "openai_like" / "providers.json"
        with open(path) as f:
            providers = json.load(f)

        streamlake_config = providers.get("streamlake")
        assert streamlake_config is not None
        serialized = json.dumps(streamlake_config)
        assert "ep-" not in serialized, "provider config must not embed any Inference Endpoint ID"

    def test_registered_public_models_are_only_the_aligned_scope(self):
        path = Path(litellm.__file__).parent / "model_prices_and_context_window_backup.json"
        with open(path) as f:
            model_cost = json.load(f)

        streamlake_keys = {key for key in model_cost if key.startswith("streamlake/")}
        assert streamlake_keys == {f"streamlake/{model}" for model in PUBLIC_MODEL_IDS}

    def test_registry_keys_contain_no_whitespace(self):
        path = Path(litellm.__file__).parent / "model_prices_and_context_window_backup.json"
        with open(path) as f:
            model_cost = json.load(f)

        streamlake_keys = [key for key in model_cost if key.startswith("streamlake/")]
        assert streamlake_keys
        for key in streamlake_keys:
            assert " " not in key, f"model registry keys must not contain spaces: {key}"


class TestStreamlakeModelMetadata:
    @staticmethod
    def _load(*path_parts):
        path = Path(__file__).parents[4].joinpath(*path_parts)
        with open(path) as f:
            return json.load(f)

    @pytest.mark.parametrize(
        "model, input_cost, output_cost, cache_cost, max_input, max_output, max_tokens, rpm, tpm",
        STREAMLAKE_MODELS,
        ids=[model.split("/", 1)[1] for model, *_ in STREAMLAKE_MODELS],
    )
    def test_model_entry_pricing_and_limits(
        self, model, input_cost, output_cost, cache_cost, max_input, max_output, max_tokens, rpm, tpm
    ):
        entry = self._load("model_prices_and_context_window.json")[model]

        assert entry["litellm_provider"] == "streamlake"
        assert entry["mode"] == "chat"

        assert entry["input_cost_per_token"] == input_cost
        assert entry["output_cost_per_token"] == output_cost
        assert entry["cache_read_input_token_cost"] == cache_cost

        if max_input is None:
            assert "max_input_tokens" not in entry
        else:
            assert entry["max_input_tokens"] == max_input
        assert entry["max_output_tokens"] == max_output
        assert entry["max_tokens"] == max_tokens
        if model in STREAMLAKE_CAPABILITIES:
            assert entry["rpm"] == rpm
            assert entry["tpm"] == tpm
        else:
            assert rpm is None
            assert tpm is None

    @pytest.mark.parametrize("model, capabilities", STREAMLAKE_CAPABILITIES.items())
    def test_new_model_capabilities(self, model, capabilities):
        entry = self._load("model_prices_and_context_window.json")[model]
        assert entry.get("supports_reasoning") is ("reasoning" in capabilities)
        assert entry.get("supports_function_calling") is ("function_calling" in capabilities)
        assert entry.get("supports_response_schema") is ("response_schema" in capabilities)
        assert entry.get("supports_vision", False) is ("vision" in capabilities)
        assert entry.get("supports_prompt_caching", False) is ("prompt_caching" in capabilities)
        assert entry.get("supports_assistant_prefill", False) is ("prefill" in capabilities)

    @pytest.mark.parametrize(
        "model, max_output_tokens",
        (
            ("streamlake/GLM-5.3", 128000),
            ("streamlake/GLM-5.3-Flash", 128000),
            ("streamlake/DeepSeek-V4-Pro-0813", 384000),
            ("streamlake/deepseek-v4-flash-0731", 384000),
        ),
    )
    def test_get_model_info_resolves_new_models(self, local_model_cost_map, model, max_output_tokens):
        info = litellm.get_model_info(model=model)
        assert info["litellm_provider"] == "streamlake"
        assert info["mode"] == "chat"
        assert info["max_output_tokens"] == max_output_tokens
        assert info["max_tokens"] == max_output_tokens
        assert (
            info["rpm"]
            == {
                "streamlake/GLM-5.3": 10,
                "streamlake/GLM-5.3-Flash": 30,
                "streamlake/DeepSeek-V4-Pro-0813": 30,
                "streamlake/deepseek-v4-flash-0731": 60,
            }[model]
        )
        assert (
            info["tpm"]
            == {
                "streamlake/GLM-5.3": 300000,
                "streamlake/GLM-5.3-Flash": 300000,
                "streamlake/DeepSeek-V4-Pro-0813": 600000,
                "streamlake/deepseek-v4-flash-0731": 600000,
            }[model]
        )
        assert info["supports_function_calling"] is True
        assert info["supports_reasoning"] is True
        assert info["supports_response_schema"] is True
        assert (
            info.get("supports_prompt_caching") is None
            if model == "streamlake/deepseek-v4-flash-0731"
            else info["supports_prompt_caching"] is True
        )
        assert (
            info.get("supports_assistant_prefill", False) is True
            if model.startswith("streamlake/GLM")
            else info.get("supports_assistant_prefill") is None
        )

        model_cost = self._load("model_prices_and_context_window.json")
        backup = self._load("litellm", "model_prices_and_context_window_backup.json")
        for model, *_ in STREAMLAKE_MODELS:
            assert model in backup, f"{model} missing from backup json"
            assert backup[model] == model_cost[model], f"{model} differs between root and backup json"

    def test_minimal_scope_metadata_only(self):
        model_cost = self._load("model_prices_and_context_window.json")
        function_calling_models = {
            model
            for model, *_ in STREAMLAKE_MODELS
            if model not in {"streamlake/Qwen3-Coder-Next", "streamlake/KAT-Coder-Pro-V2"}
        }
        for model, *_ in STREAMLAKE_MODELS:
            entry = model_cost[model]
            if model in function_calling_models:
                assert entry.get("supports_function_calling") is True
            else:
                assert "supports_function_calling" not in entry
            for capability in (
                "supports_parallel_function_calling",
                "supports_tool_choice",
            ):
                assert capability not in entry, f"{model} must not advertise {capability}"
            if model not in STREAMLAKE_CAPABILITIES:
                for capability in (
                    "supports_vision",
                    "supports_reasoning",
                    "supports_response_schema",
                    "supports_prompt_caching",
                ):
                    assert capability not in entry, f"{model} must not advertise {capability}"

    def test_get_model_info_resolves_public_model_metadata(self, local_model_cost_map):
        info = litellm.get_model_info(model="streamlake/KAT-Coder-Pro-V2.5")
        assert info["litellm_provider"] == "streamlake"
        assert info["input_cost_per_token"] == 7.4e-07
        assert info["output_cost_per_token"] == 2.96e-06
        assert info["cache_read_input_token_cost"] == 1.5e-07
        assert info["max_output_tokens"] == 80000
        assert info["max_tokens"] == 80000
        assert info["mode"] == "chat"

    def test_completion_cost_uses_streamlake_pricing(self, local_model_cost_map):
        from litellm import completion_cost
        from litellm.types.utils import ModelResponse, Usage

        response = ModelResponse(
            model="KAT-Coder-Pro-V2.5",
            usage=Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
        )
        cost = completion_cost(
            completion_response=response,
            model="streamlake/KAT-Coder-Pro-V2.5",
            custom_llm_provider="streamlake",
        )
        expected = 1000 * 7.4e-07 + 500 * 2.96e-06
        assert abs(cost - expected) < 1e-12

    def test_completion_cost_uses_lowercase_provider_response_model(self, local_model_cost_map):
        from litellm import completion_cost
        from litellm.types.utils import ModelResponse, Usage

        response = ModelResponse(
            model="deepseek-v4-flash-0731",
            usage=Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
        )
        cost = completion_cost(
            completion_response=response,
            model="streamlake/ep-test-endpoint",
            custom_llm_provider="streamlake",
            base_model="streamlake/deepseek-v4-flash-0731",
        )
        expected = 1000 * 4.4e-07 + 500 * 1.32e-06
        assert abs(cost - expected) < 1e-12

    def test_qwen3_30b_a3b_uses_highest_published_output_price(self):
        entry = self._load("model_prices_and_context_window.json")["streamlake/Qwen3-30B-A3B"]
        assert entry["output_cost_per_token"] == 1.071e-06
