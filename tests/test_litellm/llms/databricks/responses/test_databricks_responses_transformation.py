import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
from unittest.mock import patch

import litellm
from litellm.llms.databricks.responses.transformation import (
    DatabricksOpenResponsesAPIConfig,
    DatabricksResponsesAPIConfig,
    DatabricksSupervisorResponsesAPIConfig,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


class TestDatabricksResponsesAPIConfig:
    """Tests for DatabricksResponsesAPIConfig"""

    def test_custom_llm_provider(self):
        config = DatabricksResponsesAPIConfig()
        assert config.custom_llm_provider == LlmProviders.DATABRICKS

    def test_get_complete_url(self):
        config = DatabricksResponsesAPIConfig()
        url = config.get_complete_url(
            api_base="https://my-workspace.cloud.databricks.com/serving-endpoints",
            litellm_params={},
        )
        assert (
            url
            == "https://my-workspace.cloud.databricks.com/serving-endpoints/responses"
        )

    def test_get_complete_url_strips_trailing_slash(self):
        config = DatabricksResponsesAPIConfig()
        url = config.get_complete_url(
            api_base="https://my-workspace.cloud.databricks.com/serving-endpoints/",
            litellm_params={},
        )
        assert (
            url
            == "https://my-workspace.cloud.databricks.com/serving-endpoints/responses"
        )

    def test_transform_request_strips_provider_prefix(self):
        config = DatabricksResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model="databricks/databricks-gpt-5-nano",
            input="Hello!",
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert request["model"] == "databricks-gpt-5-nano"

    def test_transform_request_no_prefix(self):
        config = DatabricksResponsesAPIConfig()
        request = config.transform_responses_api_request(
            model="databricks-gpt-5-nano",
            input="Hello!",
            response_api_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert request["model"] == "databricks-gpt-5-nano"

    def test_transform_request_preserves_text_param(self):
        """Verify that the text/format param (response schema) is passed through to the request."""
        config = DatabricksResponsesAPIConfig()
        text_param = {
            "format": {
                "type": "json_schema",
                "name": "Color",
                "schema": {
                    "type": "object",
                    "properties": {"color": {"type": "string"}},
                    "required": ["color"],
                    "additionalProperties": False,
                },
            }
        }
        request = config.transform_responses_api_request(
            model="databricks-gpt-5-nano",
            input="Hello!",
            response_api_optional_request_params={"text": text_param},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert request["text"] == text_param

    def test_validate_environment_with_api_key(self):
        config = DatabricksResponsesAPIConfig()
        headers = config.validate_environment(
            headers={},
            model="databricks-gpt-5-nano",
            litellm_params=GenericLiteLLMParams(
                api_key="dapi_test_key",
                api_base="https://my-workspace.cloud.databricks.com/serving-endpoints",
            ),
        )
        assert headers["Authorization"] == "Bearer dapi_test_key"
        assert headers["Content-Type"] == "application/json"


class TestProviderConfigManagerDatabricks:
    """Tests for Databricks registration in ProviderConfigManager"""

    def test_gpt_model_returns_responses_config(self):
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.DATABRICKS,
            model="databricks-gpt-5-nano",
        )
        assert config is not None
        assert isinstance(config, DatabricksResponsesAPIConfig)

    def test_gpt_model_case_insensitive(self):
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.DATABRICKS,
            model="databricks-GPT-5-nano",
        )
        assert config is not None
        assert isinstance(config, DatabricksResponsesAPIConfig)

    def test_claude_model_uses_supervisor(self):
        """Claude responses() routes to the Supervisor API on the AI Gateway."""
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.DATABRICKS,
            model="databricks-claude-sonnet-4-5",
        )
        assert isinstance(config, DatabricksSupervisorResponsesAPIConfig)

    def test_gemini_model_uses_open_responses(self):
        """Gemini responses() routes to the unified Open Responses API (serving)."""
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.DATABRICKS,
            model="databricks-gemini-2-5-pro",
        )
        assert isinstance(config, DatabricksOpenResponsesAPIConfig)

    def test_llama_model_uses_open_responses(self):
        """Long-tail open models route to the unified Open Responses API."""
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.DATABRICKS,
            model="databricks-meta-llama-3-1-70b-instruct",
        )
        assert isinstance(config, DatabricksOpenResponsesAPIConfig)

    def test_gpt_oss_uses_open_responses_not_native(self):
        """gpt-oss has no native Responses API -> Open Responses, not OpenAI Responses."""
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.DATABRICKS,
            model="databricks-gpt-oss-120b",
        )
        assert isinstance(config, DatabricksOpenResponsesAPIConfig)
        assert not isinstance(config, DatabricksResponsesAPIConfig)

    def test_qwen_uses_open_responses_default(self):
        """qwen catch-all default is Open Responses (Supervisor is the runtime fallback)."""
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.DATABRICKS,
            model="databricks-qwen35-122b-a10b",
        )
        assert isinstance(config, DatabricksOpenResponsesAPIConfig)

    def test_no_model_returns_none(self):
        config = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.DATABRICKS,
            model=None,
        )
        assert config is None


class TestGatewayUrls:
    """Native-first gateway URL routing for responses()."""

    HOST = "https://my-ws.cloud.databricks.com"

    def test_gpt_n_bare_host_uses_openai_responses(self):
        url = DatabricksResponsesAPIConfig().get_complete_url(
            api_base=self.HOST, litellm_params={}
        )
        assert url == f"{self.HOST}/ai-gateway/openai/v1/responses"

    def test_supervisor_bare_host_uses_mlflow_responses(self):
        url = DatabricksSupervisorResponsesAPIConfig().get_complete_url(
            api_base=self.HOST, litellm_params={}
        )
        assert url == f"{self.HOST}/ai-gateway/mlflow/v1/responses"

    def test_supervisor_pinned_to_gateway_even_with_flag_false(self):
        """Supervisor exists only on the AI Gateway, so the surface is pinned."""
        url = DatabricksSupervisorResponsesAPIConfig().get_complete_url(
            api_base=self.HOST,
            litellm_params={"databricks_use_ai_gateway": False},
        )
        assert url == f"{self.HOST}/ai-gateway/mlflow/v1/responses"

    def test_open_responses_uses_serving_open_responses(self):
        url = DatabricksOpenResponsesAPIConfig().get_complete_url(
            api_base=self.HOST, litellm_params={}
        )
        assert url == f"{self.HOST}/serving-endpoints/open-responses"

    def test_open_responses_pinned_to_serving_even_with_flag_true(self):
        """Open Responses exists only on serving-endpoints, so the surface is pinned."""
        url = DatabricksOpenResponsesAPIConfig().get_complete_url(
            api_base=f"{self.HOST}/ai-gateway",
            litellm_params={"databricks_use_ai_gateway": True},
        )
        assert url == f"{self.HOST}/serving-endpoints/open-responses"

    def test_gpt_n_explicit_gateway_base(self):
        url = DatabricksResponsesAPIConfig().get_complete_url(
            api_base=f"{self.HOST}/ai-gateway", litellm_params={}
        )
        assert url == f"{self.HOST}/ai-gateway/openai/v1/responses"

    def test_explicit_serving_endpoints_keeps_legacy_path(self):
        url = DatabricksResponsesAPIConfig().get_complete_url(
            api_base=f"{self.HOST}/serving-endpoints", litellm_params={}
        )
        assert url == f"{self.HOST}/serving-endpoints/responses"

    def test_flag_false_forces_serving_endpoints(self):
        url = DatabricksResponsesAPIConfig().get_complete_url(
            api_base=self.HOST,
            litellm_params={"databricks_use_ai_gateway": False},
        )
        assert url == f"{self.HOST}/serving-endpoints/responses"


class TestDatabricksResponsesFallbackChain:
    """Per-family responses() surface fallback chains + error classification."""

    def _names(self, model):
        from litellm.llms.databricks.responses.fallback import (
            databricks_responses_config_chain,
        )

        return [type(c).__name__ for c in databricks_responses_config_chain(model)]

    def test_gpt_n_chain(self):
        assert self._names("databricks-gpt-5") == [
            "DatabricksResponsesAPIConfig",
            "DatabricksSupervisorResponsesAPIConfig",
        ]

    def test_claude_chain(self):
        assert self._names("databricks-claude-sonnet-4-5") == [
            "DatabricksSupervisorResponsesAPIConfig",
            "DatabricksOpenResponsesAPIConfig",
        ]

    def test_gemini_chain(self):
        assert self._names("databricks-gemini-2-5-pro") == [
            "DatabricksOpenResponsesAPIConfig",
        ]

    def test_oss_catch_all_chain(self):
        # qwen35 lives on Supervisor (not open-responses), reached via the fallback.
        assert self._names("databricks-qwen35-122b-a10b") == [
            "DatabricksOpenResponsesAPIConfig",
            "DatabricksSupervisorResponsesAPIConfig",
        ]

    @pytest.mark.parametrize(
        "status,message,expected",
        [
            (400, "INVALID_PARAMETER_VALUE: model not supported", True),
            (404, "ENDPOINT_NOT_FOUND", True),
            (404, "anything", True),
            (400, '{"error_code":"BAD_REQUEST","message":"unknown field input"}', True),
            (500, "INTERNAL_ERROR while serving", True),
            (400, "you must provide a temperature", False),
            (401, "invalid token", False),
            (429, "rate limited", False),
        ],
    )
    def test_is_surface_unavailable_error(self, status, message, expected):
        from litellm.llms.databricks.responses.fallback import (
            is_surface_unavailable_error,
        )

        exc = Exception(message)
        exc.status_code = status  # type: ignore[attr-defined]
        assert is_surface_unavailable_error(exc) is expected


class TestDatabricksResponsesFallbackRetry:
    """responses() retries the next surface on a surface-unavailable error and
    finally emulates via chat completions."""

    def _ok_response(self):
        from litellm.types.llms.openai import ResponsesAPIResponse

        return ResponsesAPIResponse(
            id="resp_123", created_at=0, model="m", object="response", output=[]
        )

    def _surface_error(self):
        exc = Exception("INVALID_PARAMETER_VALUE: not supported by this surface")
        exc.status_code = 400  # type: ignore[attr-defined]
        return exc

    def test_qwen35_falls_back_to_supervisor(self):
        rmain = sys.modules["litellm.responses.main"]

        calls = []

        def fake_handler(*args, **kwargs):
            cfg = kwargs.get("responses_api_provider_config")
            calls.append(type(cfg).__name__)
            if len(calls) == 1:
                raise self._surface_error()
            return self._ok_response()

        with patch.object(
            rmain.base_llm_http_handler, "response_api_handler", side_effect=fake_handler
        ):
            resp = litellm.responses(
                model="databricks/databricks-qwen35-122b-a10b",
                input="hi",
                custom_llm_provider="databricks",
                api_key="dummy",
                api_base="https://test.cloud.databricks.com",
            )

        from litellm.types.llms.openai import ResponsesAPIResponse

        assert calls == [
            "DatabricksOpenResponsesAPIConfig",
            "DatabricksSupervisorResponsesAPIConfig",
        ]
        assert isinstance(resp, ResponsesAPIResponse)

    def test_all_surfaces_rejected_emulates_via_chat(self):
        rmain = sys.modules["litellm.responses.main"]

        config_calls = []

        def fake_handler(*args, **kwargs):
            config_calls.append(type(kwargs.get("responses_api_provider_config")).__name__)
            raise self._surface_error()

        with patch.object(
            rmain.base_llm_http_handler, "response_api_handler", side_effect=fake_handler
        ), patch.object(
            rmain.litellm_completion_transformation_handler,
            "response_api_handler",
            return_value=self._ok_response(),
        ) as mock_emulate:
            resp = litellm.responses(
                model="databricks/databricks-meta-llama-3-3-70b-instruct",
                input="hi",
                custom_llm_provider="databricks",
                api_key="dummy",
                api_base="https://test.cloud.databricks.com",
            )

        # llama catch-all chain = [OpenResponses, Supervisor]; both rejected -> emulate.
        assert config_calls == [
            "DatabricksOpenResponsesAPIConfig",
            "DatabricksSupervisorResponsesAPIConfig",
        ]
        from litellm.types.llms.openai import ResponsesAPIResponse

        mock_emulate.assert_called_once()
        assert isinstance(resp, ResponsesAPIResponse)

    def test_non_fallbackable_error_is_raised(self):
        rmain = sys.modules["litellm.responses.main"]

        def fake_handler(*args, **kwargs):
            exc = Exception("invalid token")
            exc.status_code = 401  # type: ignore[attr-defined]
            raise exc

        with patch.object(
            rmain.base_llm_http_handler, "response_api_handler", side_effect=fake_handler
        ):
            with pytest.raises(Exception):
                litellm.responses(
                    model="databricks/databricks-gemini-2-5-pro",
                    input="hi",
                    custom_llm_provider="databricks",
                    api_key="dummy",
                    api_base="https://test.cloud.databricks.com",
                )
