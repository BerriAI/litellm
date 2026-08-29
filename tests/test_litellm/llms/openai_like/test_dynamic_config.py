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
