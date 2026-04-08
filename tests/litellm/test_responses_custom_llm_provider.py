"""
Test that Responses API endpoints pass custom_llm_provider inside litellm_params
to update_from_kwargs, ensuring OTEL spans show correct provider attributes.

Regression test for https://github.com/BerriAI/litellm/issues/25240
"""

from unittest.mock import MagicMock, patch

import pytest


def _make_litellm_logging_mock():
    mock = MagicMock()
    mock.update_from_kwargs = MagicMock()
    return mock


class TestResponsesApiCustomLlmProviderInLitellmParams:

    @pytest.mark.parametrize(
        "method_name",
        [
            "responses",
            "delete_responses",
            "get_responses",
            "list_input_items",
            "cancel_responses",
            "compact_responses",
        ],
    )
    def test_custom_llm_provider_in_litellm_params(self, method_name):
        from litellm.responses import main as responses_main

        logging_mock = _make_litellm_logging_mock()
        provider = "openai"

        with patch.object(
            responses_main, "LiteLLMLoggingObj", return_value=logging_mock,
        ), patch.object(
            responses_main, "ProviderConfigManager",
        ) as pcm_mock, patch.object(
            responses_main, "base_llm_http_handler",
        ), patch.object(
            responses_main, "litellm_completion_transformation_handler",
        ), patch.object(
            responses_main, "ResponsesAPIRequestUtils",
        ), patch.object(
            responses_main, "update_responses_input_with_model_file_ids",
            return_value=[],
        ), patch.object(
            responses_main, "update_responses_tools_with_model_file_ids",
            return_value=None,
        ), patch(
            "litellm.get_llm_provider",
            return_value=("gpt-4.1", provider, None, None),
        ), patch(
            "litellm.responses_api_models", new=["gpt-4.1"],
        ), patch(
            "litellm.model_alias_map", new={},
        ), patch(
            "litellm.model_list", new=None,
        ):
            config_mock = MagicMock()
            config_mock.get_complete_url.return_value = "https://api.openai.com"
            config_mock.get_supported_openai_params.return_value = []
            config_mock.map_openai_params.return_value = {}
            config_mock.transform_responses_api_request.return_value = {}
            pcm_mock.get_provider_responses_api_config.return_value = config_mock

            try:
                if method_name == "responses":
                    responses_main.responses(
                        model="gpt-4.1", input="test",
                        custom_llm_provider=provider,
                    )
                elif method_name == "delete_responses":
                    responses_main.delete_responses(
                        response_id="resp_test123",
                        custom_llm_provider=provider,
                    )
                elif method_name == "get_responses":
                    responses_main.get_responses(
                        response_id="resp_test123",
                        custom_llm_provider=provider,
                    )
                elif method_name == "list_input_items":
                    responses_main.list_input_items(
                        response_id="resp_test123",
                        custom_llm_provider=provider,
                    )
                elif method_name == "cancel_responses":
                    responses_main.cancel_responses(
                        response_id="resp_test123",
                        custom_llm_provider=provider,
                    )
                elif method_name == "compact_responses":
                    responses_main.compact_responses(
                        model="gpt-4.1", response_id="resp_test123",
                        custom_llm_provider=provider,
                    )
            except Exception:
                pass

            assert logging_mock.update_from_kwargs.called, (
                f"{method_name}: update_from_kwargs was never called"
            )

            call_kwargs = logging_mock.update_from_kwargs.call_args
            litellm_params = call_kwargs.kwargs.get(
                "litellm_params"
            ) or call_kwargs[1].get("litellm_params")

            assert litellm_params is not None, (
                f"{method_name}: litellm_params not passed to update_from_kwargs"
            )
            assert "custom_llm_provider" in litellm_params, (
                f"{method_name}: custom_llm_provider missing from litellm_params"
            )
            assert litellm_params["custom_llm_provider"] == provider, (
                f"{method_name}: expected custom_llm_provider=openai, "
                f"got {litellm_params.get('custom_llm_provider')}"
            )
