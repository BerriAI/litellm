"""
Tests for Parallel AI Responses API transformation.

Source: litellm/llms/parallel_ai/responses/transformation.py
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.parallel_ai.responses.transformation import ParallelAIResponsesConfig
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


class TestParallelAIResponsesConfig:
    def test_provider_config_manager_returns_parallel_config(self):
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.PARALLEL_AI, model="parallel"
        )
        assert isinstance(config, ParallelAIResponsesConfig)

    @pytest.mark.parametrize(
        "api_base,expected",
        [
            (None, "https://api.parallel.ai/v1/responses"),
            ("https://api.parallel.ai", "https://api.parallel.ai/v1/responses"),
            ("https://api.parallel.ai/", "https://api.parallel.ai/v1/responses"),
            ("https://api.parallel.ai/v1", "https://api.parallel.ai/v1/responses"),
            ("https://proxy.example.com/v1/responses", "https://proxy.example.com/v1/responses"),
        ],
    )
    def test_get_complete_url(self, api_base, expected, monkeypatch):
        monkeypatch.delenv("PARALLEL_AI_API_BASE", raising=False)
        config = ParallelAIResponsesConfig()
        assert config.get_complete_url(api_base=api_base, litellm_params={}) == expected

    def test_validate_environment_sets_bearer_from_env(self, monkeypatch):
        monkeypatch.delenv("PARALLEL_AI_API_KEY", raising=False)
        monkeypatch.setenv("PARALLEL_API_KEY", "pk-test")
        config = ParallelAIResponsesConfig()

        headers = config.validate_environment(headers={}, model="parallel", litellm_params=None)
        assert headers["Authorization"] == "Bearer pk-test"

    def test_validate_environment_prefers_explicit_key(self, monkeypatch):
        monkeypatch.setenv("PARALLEL_AI_API_KEY", "pk-env")
        config = ParallelAIResponsesConfig()

        headers = config.validate_environment(
            headers={},
            model="parallel",
            litellm_params=GenericLiteLLMParams(api_key="pk-explicit"),
        )
        assert headers["Authorization"] == "Bearer pk-explicit"

    def test_reasoning_effort_is_supported(self):
        config = ParallelAIResponsesConfig()
        supported = config.get_supported_openai_params(model="parallel")
        assert "reasoning" in supported
        assert "instructions" in supported
        assert "previous_response_id" in supported
        assert "tools" not in supported

    def test_unsupported_params_dropped_with_drop_params(self):
        from litellm.responses.utils import ResponsesAPIRequestUtils

        config = ParallelAIResponsesConfig()
        params = ResponsesAPIOptionalRequestParams(
            reasoning={"effort": "high"},
            tools=[{"type": "web_search"}],
            temperature=0.5,
        )

        mapped = ResponsesAPIRequestUtils.get_optional_params_responses_api(
            model="parallel",
            responses_api_provider_config=config,
            response_api_optional_params=params,
            drop_params=True,
        )
        assert mapped["reasoning"] == {"effort": "high"}
        assert "tools" not in mapped
        assert "temperature" not in mapped

    def test_unsupported_params_raise_without_drop_params(self):
        import litellm as litellm_module
        from litellm.responses.utils import ResponsesAPIRequestUtils

        config = ParallelAIResponsesConfig()
        params = ResponsesAPIOptionalRequestParams(tools=[{"type": "web_search"}])

        with pytest.raises(litellm_module.UnsupportedParamsError):
            ResponsesAPIRequestUtils.get_optional_params_responses_api(
                model="parallel",
                responses_api_provider_config=config,
                response_api_optional_params=params,
                drop_params=False,
            )

    def test_transform_request_sends_parallel_model(self):
        config = ParallelAIResponsesConfig()
        request = config.transform_responses_api_request(
            model="parallel",
            input="What is the latest AI news?",
            response_api_optional_request_params={"reasoning": {"effort": "low"}},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert request["model"] == "parallel"
        assert request["input"] == "What is the latest AI news?"
        assert request["reasoning"] == {"effort": "low"}

    def test_no_native_websocket(self):
        assert ParallelAIResponsesConfig().supports_native_websocket() is False
