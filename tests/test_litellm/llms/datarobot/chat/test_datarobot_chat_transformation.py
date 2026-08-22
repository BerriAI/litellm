import os
from unittest.mock import patch

import pytest

import litellm
from litellm.llms.datarobot.chat.transformation import DataRobotConfig


@patch.dict(os.environ, {}, clear=True)
class TestDataRobotConfig:
    @pytest.fixture
    def handler(self):
        return DataRobotConfig()

    @pytest.mark.parametrize(
        "api_base, expected_url",
        [
            (None, "https://app.datarobot.com/api/v2/genai/llmgw/chat/completions/"),
            (
                "http://localhost:5001",
                "http://localhost:5001/api/v2/genai/llmgw/chat/completions/",
            ),
            (
                "https://app.datarobot.com",
                "https://app.datarobot.com/api/v2/genai/llmgw/chat/completions/",
            ),
            (
                "https://app.datarobot.com/api/v2/",
                "https://app.datarobot.com/api/v2/genai/llmgw/chat/completions/",
            ),
            (
                "https://app.datarobot.com/api/v2",
                "https://app.datarobot.com/api/v2/genai/llmgw/chat/completions/",
            ),
            (
                "https://app.datarobot.com/api/v2/genai/llmgw/chat/completions",
                "https://app.datarobot.com/api/v2/genai/llmgw/chat/completions/",
            ),
            (
                "https://app.datarobot.com/api/v2/genai/llmgw/chat/completions/",
                "https://app.datarobot.com/api/v2/genai/llmgw/chat/completions/",
            ),
            (
                "https://staging.datarobot.com",
                "https://staging.datarobot.com/api/v2/genai/llmgw/chat/completions/",
            ),
            (
                "https://app.datarobot.com/api/v2/deployments/deployment_id",
                "https://app.datarobot.com/api/v2/deployments/deployment_id/",
            ),
            (
                "https://app.datarobot.com/api/v2/deployments/deployment_id/",
                "https://app.datarobot.com/api/v2/deployments/deployment_id/",
            ),
        ],
    )
    def test_resolve_api_base(self, api_base, expected_url, handler):
        """Test that URLs properly resolve to the expected format."""
        assert handler._resolve_api_base(api_base) == expected_url

        # Check that the complete url with the resolution is expected
        assert (
            handler.get_complete_url(
                api_base=handler._resolve_api_base(api_base),
                api_key="PASSTHROUGH_KEY",
                model="datarobot/vertex_ai/gemini-1.5-flash-002",
                optional_params={},
                litellm_params={},
            )
            == expected_url
        )

        # Check that the complete url with the original api_base does not change the url
        if api_base is not None:
            assert (
                handler.get_complete_url(
                    api_base=api_base,
                    api_key="PASSTHROUGH_KEY",
                    model="datarobot/vertex_ai/gemini-1.5-flash-002",
                    optional_params={},
                    litellm_params={},
                )
                == api_base
            )

    def test_resolve_api_base_with_environment_variable(self, handler):
        os.environ["DATAROBOT_ENDPOINT"] = "https://env.datarobot.com"
        assert (
            handler._resolve_api_base(None)
            == "https://env.datarobot.com/api/v2/genai/llmgw/chat/completions/"
        )
        del os.environ["DATAROBOT_ENDPOINT"]

    @pytest.mark.parametrize(
        "api_key, expected_api_key",
        [
            (None, "fake-api-key"),
            ("PASSTHROUGH_KEY", "PASSTHROUGH_KEY"),
        ],
    )
    def test_resolve_api_key(self, api_key, expected_api_key, handler):
        assert handler._resolve_api_key(api_key) == expected_api_key

    def test_resolve_api_key_with_environment_variable(self, handler):
        os.environ["DATAROBOT_API_TOKEN"] = "env_key"
        assert handler._resolve_api_key(None) == "env_key"
        del os.environ["DATAROBOT_API_TOKEN"]

    def test_get_provider_info_resolves_inner_provider(self, handler):
        """Capabilities come from the provider the gateway routes to."""
        info = handler.get_provider_info("azure/gpt-4o")
        assert info is not None
        assert info["supports_function_calling"] is True

    def test_get_provider_info_resolves_vertex_ai(self, handler):
        """Non-azure inner providers (e.g. vertex_ai) resolve too."""
        info = handler.get_provider_info("vertex_ai/gemini-2.5-pro")
        assert info is not None
        assert info["supports_function_calling"] is True

    def test_get_provider_info_normalizes_azure_deployment_name(self, handler):
        captured = {}

        def fake_helper(model, *args, **kwargs):
            captured["model"] = model
            return {"supports_function_calling": True}

        with patch(
            "litellm.llms.datarobot.chat.transformation._get_model_info_helper",
            side_effect=fake_helper,
        ):
            info = handler.get_provider_info("azure/gpt-5-1-2025-11-13")

        # dashed version dotted, date kept
        assert captured["model"] == "azure/gpt-5.1-2025-11-13"
        assert info["supports_function_calling"] is True

    def test_get_provider_info_passes_unversioned_name_unchanged(self, handler):
        """The azure regex must not mangle a name with no dashed version."""
        captured = {}

        def fake_helper(model, *args, **kwargs):
            captured["model"] = model
            return {"supports_function_calling": True}

        with patch(
            "litellm.llms.datarobot.chat.transformation._get_model_info_helper",
            side_effect=fake_helper,
        ):
            handler.get_provider_info("azure/gpt-4o")

        assert captured["model"] == "azure/gpt-4o"  # untouched

    def test_get_provider_info_normalizes_multi_digit_minor(self, handler):
        captured = {}

        def fake_helper(model, *args, **kwargs):
            captured["model"] = model
            return {"supports_function_calling": True}

        with patch(
            "litellm.llms.datarobot.chat.transformation._get_model_info_helper",
            side_effect=fake_helper,
        ):
            handler.get_provider_info("azure/gpt-5-12")

        assert captured["model"] == "azure/gpt-5.12"  # two-digit minor

    def test_get_provider_info_dated_name_not_read_as_version(self, handler):
        """A 4-digit date suffix must not be dotted into a version."""
        captured = {}

        def fake_helper(model, *args, **kwargs):
            captured["model"] = model
            return {"supports_function_calling": True}

        with patch(
            "litellm.llms.datarobot.chat.transformation._get_model_info_helper",
            side_effect=fake_helper,
        ):
            handler.get_provider_info("azure/gpt-4-2024-08-06")

        assert captured["model"] == "azure/gpt-4-2024-08-06"  # untouched

    def test_get_provider_info_unknown_model_returns_none(self, handler):
        with patch(
            "litellm.llms.datarobot.chat.transformation._get_model_info_helper",
            side_effect=Exception("not mapped"),
        ):
            assert handler.get_provider_info("vertex_ai/unknown-xyz") is None

    def test_get_provider_info_no_inner_provider_returns_none(self, handler):
        assert handler.get_provider_info("datarobot-deployed-llm") is None


def test_supports_function_calling_for_datarobot_routed_model():
    """End-to-end: the `datarobot/` prefix resolves via the provider model-info."""
    assert litellm.supports_function_calling("datarobot/azure/gpt-4o") is True
