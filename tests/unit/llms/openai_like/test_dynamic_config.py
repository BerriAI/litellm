from types import MappingProxyType
from typing import Final

import pytest
from pydantic import ValidationError

from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap
from litellm.llms.openai_like import dynamic_config
from litellm.llms.openai_like.dynamic_config import create_config_class, create_responses_config_class
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

    def test_a_provider_without_a_base_url_is_rejected_at_load(self) -> None:
        with pytest.raises(ValidationError):
            _provider("url_none", base_url=None)


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


class TestWebSearchRequestsFromToolUsage:
    def test_counts_only_paid_search_tools(self):
        payload = {
            "usage": {"tool_usage": {"web_search": 2, "image_search": 1, "code_interpreter": 3, "web_extractor": 1}}
        }
        assert dynamic_config.web_search_requests_from_tool_usage(payload) == 3

    @pytest.mark.parametrize(
        "payload",
        [
            {"usage": {"prompt_tokens": 1}},
            {"usage": {"tool_usage": {"code_interpreter": 2}}},
            {"usage": {"tool_usage": {"web_search": "2"}}},
            {"usage": {"tool_usage": {"web_search": True}}},
            {"usage": {"tool_usage": []}},
            {"usage": None},
            "not a mapping",
        ],
    )
    def test_anything_but_a_positive_search_count_is_none(self, payload):
        assert dynamic_config.web_search_requests_from_tool_usage(payload) is None


class TestModelBaseUrls:
    _LEGACY: Final = MappingProxyType({"legacy-model": "https://api.example.com/v0"})

    @pytest.mark.parametrize(
        ("model", "api_base", "expected"),
        [
            pytest.param("legacy-model", None, "https://api.example.com/v0", id="listed_model"),
            pytest.param("mb/legacy-model", None, "https://api.example.com/v0", id="listed_model_with_prefix"),
            pytest.param("new-model", None, "https://api.example.com/v1", id="unlisted_model"),
            pytest.param("legacy-model", "https://explicit/v1", "https://explicit/v1", id="explicit_api_base_wins"),
        ],
    )
    def test_model_base_url_resolution(
        self, monkeypatch: pytest.MonkeyPatch, model: str, api_base: str | None, expected: str
    ) -> None:
        monkeypatch.delenv("MB_UNSET_BASE", raising=False)
        config: Final = create_config_class(_provider("mb", model_base_urls=self._LEGACY, api_base_env="MB_UNSET_BASE"))()
        assert config.get_model_provider_info(model, api_base, "sk-test")[0] == expected

    def test_api_base_env_wins_over_the_model_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MB_ENV_BASE", "https://from-env/v1")
        config: Final = create_config_class(_provider("mb_env", model_base_urls=self._LEGACY, api_base_env="MB_ENV_BASE"))()
        assert config.get_model_provider_info("legacy-model", None, "sk-test")[0] == "https://from-env/v1"

    def test_chat_url_without_api_base_uses_the_model_base_url(self) -> None:
        config: Final = create_config_class(_provider("mb_url", model_base_urls=self._LEGACY))()
        assert (
            config.get_complete_url(
                api_base=None, api_key=None, model="legacy-model", optional_params={}, litellm_params={}
            )
            == "https://api.example.com/v0/chat/completions"
        )


class TestModelSpecialHandling:
    _TEXT_LIST: Final = ({"role": "user", "content": [{"type": "text", "text": "Say "}, {"type": "text", "text": "hi"}]},)

    @pytest.mark.parametrize(
        ("provider_flags", "model_flags", "model", "expected_content"),
        [
            pytest.param(
                {}, {"legacy-model": {"convert_content_list_to_string": True}}, "legacy-model", "Say hi",
                id="listed_model",
            ),
            pytest.param(
                {}, {"legacy-model": {"convert_content_list_to_string": True}}, "msh/legacy-model", "Say hi",
                id="listed_model_with_prefix",
            ),
            pytest.param(
                {}, {"legacy-model": {"convert_content_list_to_string": True}}, "new-model", _TEXT_LIST[0]["content"],
                id="unlisted_model_keeps_lists",
            ),
            pytest.param(
                {"convert_content_list_to_string": True}, {"list-model": {"convert_content_list_to_string": False}},
                "list-model", _TEXT_LIST[0]["content"], id="model_flag_overrides_provider_flag",
            ),
            pytest.param(
                {"convert_content_list_to_string": True}, {"list-model": {"convert_content_list_to_string": False}},
                "other-model", "Say hi", id="provider_flag_applies_to_unlisted_models",
            ),
            pytest.param(
                {}, {"legacy-model": {"convert_text_content_list_to_string": True}}, "legacy-model", "Say \nhi",
                id="text_only_flag_joins_parts_by_line",
            ),
        ],
    )
    def test_content_list_flattening_follows_the_model_flags(
        self,
        provider_flags: dict[str, bool],
        model_flags: dict[str, dict[str, bool]],
        model: str,
        expected_content: object,
    ) -> None:
        config: Final = create_config_class(
            _provider("msh", special_handling=provider_flags, model_special_handling=model_flags)
        )()
        body: Final = config.transform_request(
            model=model, messages=list(self._TEXT_LIST), optional_params={}, litellm_params={}, headers={}
        )
        assert body["messages"][0]["content"] == expected_content


class TestBackgroundRuns:
    @pytest.mark.parametrize(
        ("model", "starts_runs"),
        [("inline-model", False), ("queueing-model", True), ("background_runs/queueing-model", True)],
    )
    def test_one_model_can_start_runs_on_a_provider_that_answers_background_inline(
        self, model: str, starts_runs: bool
    ) -> None:
        config: Final = create_responses_config_class(
            _provider(
                "background_runs",
                special_handling={"answers_background_inline": True},
                model_special_handling={"queueing-model": {"answers_background_inline": False}},
            )
        )()

        assert config.starts_background_runs(model) is starts_runs


class TestReasoningEffortRounding:
    _CATALOG_KEY: Final = "refuseless/glm-5.3-abliterated"

    @pytest.mark.parametrize("rounds", [False, True], ids=["without_the_flag", "with_the_flag"])
    def test_only_a_flagged_provider_rounds_to_the_declared_levels(self, rounds: bool) -> None:
        declared: Final = GetModelCostMap.loaded_model_cost_map()[self._CATALOG_KEY]["reasoning_effort_levels"]
        assert "medium" not in declared
        config: Final = create_config_class(
            _provider("refuseless", special_handling={"round_reasoning_effort_to_declared_levels": rounds})
        )()

        sent: Final = config.map_openai_params(
            non_default_params={"reasoning_effort": "medium"},
            optional_params={},
            model="glm-5.3-abliterated",
            drop_params=False,
        )["reasoning_effort"]

        assert (sent in declared) if rounds else (sent == "medium")
