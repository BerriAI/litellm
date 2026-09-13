from types import MappingProxyType

import pytest

from litellm.llms.openai_like import dynamic_config
from litellm.llms.openai_like.dynamic_config import create_responses_config_class
from litellm.llms.openai_like.json_loader import SimpleProviderConfig
from litellm.types.router import GenericLiteLLMParams

_BASE = {"base_url": "https://api.example.com/v1", "api_key_env": "EXAMPLE_API_KEY"}


def _provider(slug, **overrides):
    return SimpleProviderConfig(slug=slug, data={**_BASE, **overrides})


@pytest.fixture(autouse=True)
def _isolate_generated_class_cache():
    dynamic_config._responses_config_cache.clear()
    yield
    dynamic_config._responses_config_cache.clear()


class TestBaseModelParamSupport:
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        }
    ]

    @pytest.mark.parametrize("base_class", ["openai_gpt", "openai_like"])
    def test_generated_chat_config_uses_base_model_for_supported_params(self, local_model_cost_map, base_class):
        config = dynamic_config.create_config_class(_provider("publicai", base_class=base_class))()

        endpoint_params = config.get_supported_openai_params(model="ep-publicai")
        assert "tools" not in endpoint_params
        assert "reasoning_effort" not in endpoint_params

        instruct_params = config.get_supported_openai_params(
            model="ep-publicai",
            base_model="publicai/allenai/Olmo-3-7B-Instruct",
        )
        assert "tools" in instruct_params
        assert "reasoning_effort" not in instruct_params

        thinking_params = config.get_supported_openai_params(
            model="ep-publicai",
            base_model="publicai/allenai/Olmo-3-7B-Think",
        )
        assert "tools" in thinking_params
        assert "reasoning_effort" in thinking_params

    @pytest.mark.parametrize("base_class", ["openai_gpt", "openai_like"])
    @pytest.mark.parametrize("base_model", [None, "ep-publicai", "publicai/allenai/Olmo-3-7B-Think"])
    def test_supported_params_allow_per_call_extensions_without_leaking(
        self, local_model_cost_map, base_class, base_model
    ):
        config = dynamic_config.create_config_class(_provider("publicai", base_class=base_class))()
        supported = config.get_supported_openai_params("ep-publicai", base_model=base_model)
        original = tuple(supported)

        supported.extend(["request_specific_param"])

        assert supported[-1] == "request_specific_param"
        assert tuple(config.get_supported_openai_params("ep-publicai", base_model=base_model)) == original

    @pytest.mark.parametrize("base_class", ["openai_gpt", "openai_like"])
    def test_mapping_preserves_caller_owned_output_and_accepts_readonly_input(self, local_model_cost_map, base_class):
        config = dynamic_config.create_config_class(_provider("publicai", base_class=base_class))()
        non_default_params = MappingProxyType({"tools": self.TOOLS, "reasoning_effort": "high"})
        optional_params = {"temperature": 0.4}

        mapped = config.map_openai_params_with_base_model(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="ep-publicai",
            drop_params=False,
            base_model="publicai/allenai/Olmo-3-7B-Think",
        )

        assert mapped is optional_params
        assert optional_params == {"temperature": 0.4, "tools": self.TOOLS, "reasoning_effort": "high"}
        assert non_default_params == {"tools": self.TOOLS, "reasoning_effort": "high"}

    @pytest.mark.parametrize("drop_params", [True, False])
    def test_get_optional_params_passes_base_model_to_json_provider(self, local_model_cost_map, drop_params):
        from litellm.utils import get_optional_params

        optional_params = get_optional_params(
            model="ep-publicai",
            custom_llm_provider="publicai",
            tools=self.TOOLS,
            reasoning_effort="high",
            base_model="publicai/allenai/Olmo-3-7B-Think",
            drop_params=drop_params,
        )

        assert optional_params["tools"] == self.TOOLS
        assert optional_params["reasoning_effort"] == "high"
        assert "base_model" not in optional_params

    def test_non_json_provider_does_not_receive_base_model_kwarg(self):
        from litellm.utils import get_optional_params

        optional_params = get_optional_params(
            model="a2a/test-agent",
            custom_llm_provider="a2a",
            tools=self.TOOLS,
            base_model="publicai/allenai/Olmo-3-7B-Think",
            drop_params=True,
        )

        assert "base_model" not in optional_params


class TestClassCaching:
    def test_same_slug_returns_the_identical_class_object(self):
        provider = _provider("cache_same_slug")
        assert create_responses_config_class(provider) is create_responses_config_class(provider)

    def test_cache_is_keyed_on_slug_not_on_the_provider_instance(self):
        first = create_responses_config_class(_provider("cache_by_slug"))
        second = create_responses_config_class(_provider("cache_by_slug"))
        assert first is second

    def test_different_slugs_get_different_classes(self):
        assert create_responses_config_class(_provider("cache_slug_a")) is not (
            create_responses_config_class(_provider("cache_slug_b"))
        )

    def test_returns_a_class_not_an_instance(self):
        assert isinstance(create_responses_config_class(_provider("returns_class")), type)


class TestCustomLlmProvider:
    def test_provider_property_reports_the_slug(self):
        config = create_responses_config_class(_provider("provider_prop"))()
        assert config.custom_llm_provider == "provider_prop"


class TestValidateEnvironment:
    def test_explicit_api_key_becomes_a_bearer_header(self):
        config = create_responses_config_class(_provider("ve_explicit"))()
        headers = config.validate_environment(
            headers={}, model="m", litellm_params=GenericLiteLLMParams(api_key="sk-explicit")
        )
        assert headers["Authorization"] == "Bearer sk-explicit"

    def test_api_key_falls_back_to_the_configured_env_var(self, monkeypatch):
        monkeypatch.setenv("VE_ENV_KEY", "sk-from-env")
        config = create_responses_config_class(_provider("ve_env", api_key_env="VE_ENV_KEY"))()
        headers = config.validate_environment(headers={}, model="m", litellm_params=None)
        assert headers["Authorization"] == "Bearer sk-from-env"

    def test_explicit_key_wins_over_the_env_var(self, monkeypatch):
        monkeypatch.setenv("VE_LOSER_KEY", "sk-from-env")
        config = create_responses_config_class(_provider("ve_precedence", api_key_env="VE_LOSER_KEY"))()
        headers = config.validate_environment(
            headers={}, model="m", litellm_params=GenericLiteLLMParams(api_key="sk-wins")
        )
        assert headers["Authorization"] == "Bearer sk-wins"

    def test_no_key_anywhere_leaves_the_header_unset(self, monkeypatch):
        monkeypatch.delenv("VE_MISSING_KEY", raising=False)
        config = create_responses_config_class(_provider("ve_missing", api_key_env="VE_MISSING_KEY"))()
        assert config.validate_environment(headers={}, model="m", litellm_params=None) == {}

    def test_existing_headers_are_preserved(self):
        config = create_responses_config_class(_provider("ve_preserve"))()
        headers = config.validate_environment(
            headers={"X-Trace": "abc"},
            model="m",
            litellm_params=GenericLiteLLMParams(api_key="sk-1"),
        )
        assert headers["X-Trace"] == "abc"


class TestGetCompleteUrl:
    def test_explicit_api_base_gets_the_responses_suffix(self):
        config = create_responses_config_class(_provider("url_explicit"))()
        assert config.get_complete_url(api_base="https://host/v1", litellm_params={}) == "https://host/v1/responses"

    def test_trailing_slash_is_stripped_before_appending(self):
        config = create_responses_config_class(_provider("url_slash"))()
        assert config.get_complete_url(api_base="https://host/v1/", litellm_params={}) == "https://host/v1/responses"

    def test_falls_back_to_the_api_base_env_var(self, monkeypatch):
        monkeypatch.setenv("URL_BASE_ENV", "https://from-env/v1")
        config = create_responses_config_class(_provider("url_env", api_base_env="URL_BASE_ENV"))()
        assert config.get_complete_url(api_base=None, litellm_params={}) == "https://from-env/v1/responses"

    def test_falls_back_to_the_configured_base_url_last(self, monkeypatch):
        monkeypatch.delenv("URL_UNSET_ENV", raising=False)
        config = create_responses_config_class(_provider("url_base_url", api_base_env="URL_UNSET_ENV"))()
        assert config.get_complete_url(api_base=None, litellm_params={}) == "https://api.example.com/v1/responses"

    def test_explicit_api_base_wins_over_the_env_var(self, monkeypatch):
        monkeypatch.setenv("URL_LOSER_ENV", "https://from-env/v1")
        config = create_responses_config_class(_provider("url_precedence", api_base_env="URL_LOSER_ENV"))()
        assert (
            config.get_complete_url(api_base="https://explicit/v1", litellm_params={})
            == "https://explicit/v1/responses"
        )

    def test_no_base_anywhere_raises_naming_the_provider(self):
        provider = _provider("url_none")
        provider.base_url = None
        config = create_responses_config_class(provider)()
        with pytest.raises(ValueError, match="url_none"):
            config.get_complete_url(api_base=None, litellm_params={})


class TestForceStoreFalse:
    def test_force_store_false_overrides_the_caller(self):
        config = create_responses_config_class(
            _provider("store_forced", special_handling={"force_store_false": True})
        )()
        params = {"store": True}
        config.transform_responses_api_request(
            model="m",
            input="hi",
            response_api_optional_request_params=params,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert params["store"] is False

    def test_without_the_flag_the_callers_store_value_is_left_alone(self):
        config = create_responses_config_class(_provider("store_untouched"))()
        params = {"store": True}
        config.transform_responses_api_request(
            model="m",
            input="hi",
            response_api_optional_request_params=params,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert params["store"] is True
