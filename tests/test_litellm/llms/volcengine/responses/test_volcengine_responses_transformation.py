"""
Tests for Volcengine Responses API transformation.
"""
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm.llms.volcengine.responses.transformation import (
    VolcEngineResponsesAPIConfig,
)
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams
from litellm.types.responses.main import DeleteResponseResult
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


class TestVolcengineResponsesAPITransformation:
    """Test Volcengine Responses API configuration and transformations."""

    def test_provider_config_registration(self):
        """Provider registry should return VolcEngineResponsesAPIConfig."""
        config = ProviderConfigManager.get_provider_responses_api_config(
            model="volcengine/demo-model",
            provider=LlmProviders.VOLCENGINE,
        )

        assert config is not None, "Config should not be None for Volcengine provider"
        assert isinstance(
            config, VolcEngineResponsesAPIConfig
        ), f"Expected VolcEngineResponsesAPIConfig, got {type(config)}"
        assert (
            config.custom_llm_provider == LlmProviders.VOLCENGINE
        ), "custom_llm_provider should be VOLCENGINE"

    def test_parallel_tool_calls_dropped(self):
        """Volcengine does not list parallel_tool_calls; ensure it is removed."""
        config = VolcEngineResponsesAPIConfig()
        params = ResponsesAPIOptionalRequestParams(
            parallel_tool_calls=True,
            temperature=0.5,
            metadata={"k": "v"},
        )

        mapped = config.map_openai_params(
            response_api_optional_params=params,
            model="volcengine/demo-model",
            drop_params=False,
        )

        assert "parallel_tool_calls" not in mapped, "parallel_tool_calls must be dropped"
        assert mapped.get("temperature") == 0.5
        assert "metadata" not in mapped, "Undocumented params should not be included"

    def test_unsupported_params_are_dropped(self):
        """Unknown fields should be dropped before send, including nested extra_body."""
        config = VolcEngineResponsesAPIConfig()

        request = config.transform_responses_api_request(
            model="volcengine/demo-model",
            input="hi",
            response_api_optional_request_params={
                "unsupported_custom_param": 0.1,
                "temperature": 0.2,
                "metadata": {"k": "v"},
                "extra_body": {"unsupported_custom_param": 1, "temperature": 0.3},
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert "unsupported_custom_param" not in request
        assert request["temperature"] == 0.2
        assert "metadata" not in request
        assert "extra_body" in request
        assert "unsupported_custom_param" not in request["extra_body"]
        assert request["extra_body"]["temperature"] == 0.3

    def test_get_complete_url_variants(self):
        """Ensure Volcengine endpoint construction handles different bases."""
        config = VolcEngineResponsesAPIConfig()

        default_url = config.get_complete_url(api_base=None, litellm_params={})
        assert default_url == "https://ark.cn-beijing.volces.com/api/v3/responses"

        api_base_with_api = config.get_complete_url(
            api_base="https://custom.volc.com/api/v3", litellm_params={}
        )
        assert api_base_with_api == "https://custom.volc.com/api/v3/responses"

        api_base_full = config.get_complete_url(
            api_base="https://custom.volc.com/api/v3/responses", litellm_params={}
        )
        assert api_base_full == "https://custom.volc.com/api/v3/responses"

    @pytest.mark.parametrize(
        "litellm_params, expected_key",
        [
            ({"api_key": "dict-key"}, "dict-key"),
            (GenericLiteLLMParams(api_key="attr-key"), "attr-key"),
        ],
    )
    def test_validate_environment_uses_api_key(
        self, monkeypatch, litellm_params, expected_key
    ):
        """validate_environment should pull api key from params/env and attach headers."""
        config = VolcEngineResponsesAPIConfig()

        monkeypatch.setattr(litellm, "api_key", None)
        monkeypatch.delenv("ARK_API_KEY", raising=False)
        monkeypatch.delenv("VOLCENGINE_API_KEY", raising=False)

        headers = config.validate_environment(
            headers={}, model="volcengine/demo-model", litellm_params=litellm_params
        )

        assert headers.get("Authorization") == f"Bearer {expected_key}"
        assert headers.get("Content-Type") == "application/json"

    def test_validate_environment_raises_without_key(self, monkeypatch):
        """validate_environment should error when no key is available."""
        config = VolcEngineResponsesAPIConfig()

        monkeypatch.setattr(litellm, "api_key", None)
        monkeypatch.delenv("ARK_API_KEY", raising=False)
        monkeypatch.delenv("VOLCENGINE_API_KEY", raising=False)

        with pytest.raises(ValueError):
            config.validate_environment(
                headers={}, model="volcengine/demo", litellm_params={}
            )

    def test_unsupported_params_are_dropped_with_extra_body(self):
        """Unknown fields (including extra_body) should be dropped before send."""
        config = VolcEngineResponsesAPIConfig()

        request = config.transform_responses_api_request(
            model="volcengine/demo-model",
            input="hi",
            response_api_optional_request_params={
                "unsupported_custom_param": 0.1,
                "temperature": 0.2,
                "metadata": {"k": "v"},
                "extra_body": {"unsupported_custom_param": 1, "temperature": 0.3},
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert "unsupported_custom_param" not in request
        assert "metadata" not in request
        assert request["temperature"] == 0.2
        assert "extra_body" in request
        assert "unsupported_custom_param" not in request["extra_body"]
        assert request["extra_body"]["temperature"] == 0.3

    def test_valid_thinking_caching_and_expire_at_pass(self):
        """Documented params should pass through without validation errors."""
        config = VolcEngineResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model="volcengine/demo-model",
            input="hi",
            response_api_optional_request_params={
                "instructions": "do X",
                "thinking": {"type": "enabled"},
                "caching": {"type": "enabled"},
                "expire_at": 1234567890,
                "temperature": 0.5,
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert request["thinking"]["type"] == "enabled"
        assert request["caching"]["type"] == "enabled"
        assert request["expire_at"] == 1234567890
        assert request["instructions"] == "do X"

    def test_supported_params_limited_to_docs(self):
        """Supported params should match documented Volcengine surface."""
        config = VolcEngineResponsesAPIConfig()
        supported = set(config.get_supported_openai_params("volcengine/demo-model"))

        expected = {
            "input",
            "model",
            "instructions",
            "max_output_tokens",
            "previous_response_id",
            "store",
            "reasoning",
            "stream",
            "temperature",
            "top_p",
            "text",
            "tools",
            "tool_choice",
            "max_tool_calls",
            "thinking",
            "caching",
            "expire_at",
            "context_management",
            "extra_headers",
            "extra_query",
            "extra_body",
            "timeout",
        }

        assert supported == expected

    def test_error_class_returns_volcengine_error(self):
        """Errors should be wrapped with VolcEngineError for consistent handling."""
        config = VolcEngineResponsesAPIConfig()
        error = config.get_error_class("bad request", 400, headers={"x": "y"})

        # Use class name comparison instead of isinstance to avoid issues with
        # module reloading during parallel test execution (conftest reloads litellm)
        assert type(error).__name__ == "VolcEngineError", f"Expected VolcEngineError, got {type(error).__name__}"
        assert error.status_code == 400
        assert error.message == "bad request"
        assert error.headers.get("x") == "y"

    def test_transform_response_api_response_sets_headers_and_created_at(self):
        """Responses should include processed headers and keep created_at intact."""
        config = VolcEngineResponsesAPIConfig()
        response_payload = {
            "id": "resp_123",
            "object": "response",
            "created_at": 123,
            "status": "completed",
            "output": [],
            "model": "demo-model",
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }
        http_response = httpx.Response(
            status_code=200,
            json=response_payload,
            request=httpx.Request("POST", "https://example.com/responses"),
            headers={"x-test": "1"},
        )

        result = config.transform_response_api_response(
            model="volcengine/demo-model",
            raw_response=http_response,
            logging_obj=type(
                "Logger",
                (),
                {"post_call": staticmethod(lambda **kwargs: None)},
            ),
        )

        assert result.created_at == 123
        assert result._hidden_params["headers"].get("x-test") == "1"
        assert "additional_headers" in result._hidden_params

    def test_transform_delete_response_api_response_parses_json(self):
        """DELETE response parsing should return DeleteResponseResult."""
        config = VolcEngineResponsesAPIConfig()
        http_response = httpx.Response(
            status_code=200,
            json={"id": "resp_123", "deleted": True},
            request=httpx.Request("DELETE", "https://example.com/responses/resp_123"),
        )

        result = config.transform_delete_response_api_response(
            raw_response=http_response,
            logging_obj=None,
        )

        assert isinstance(result, DeleteResponseResult)
        assert result.deleted is True
