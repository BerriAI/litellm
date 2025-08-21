import os
from unittest.mock import patch

import pytest

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
            ("http://localhost:5001", "http://localhost:5001/api/v2/genai/llmgw/chat/completions/"),
            ("https://app.datarobot.com", "https://app.datarobot.com/api/v2/genai/llmgw/chat/completions/"),
            ("https://app.datarobot.com/api/v2/genai/llmgw/chat/completions", "https://app.datarobot.com/api/v2/genai/llmgw/chat/completions/"),
            ("https://app.datarobot.com/api/v2/genai/llmgw/chat/completions/", "https://app.datarobot.com/api/v2/genai/llmgw/chat/completions/"),
            ("https://staging.datarobot.com", "https://staging.datarobot.com/api/v2/genai/llmgw/chat/completions/"),
            ("https://app.datarobot.com/api/v2/deployments/deployment_id", "https://app.datarobot.com/api/v2/deployments/deployment_id/"),
            ("https://app.datarobot.com/api/v2/deployments/deployment_id/", "https://app.datarobot.com/api/v2/deployments/deployment_id/"),
        ]
    )
    def test_resolve_api_base(self, api_base, expected_url, handler):
        """Test that URLs properly resolve to the expected format."""
        assert handler._resolve_api_base(api_base) == expected_url

        # Check that the complete url with the resolution is expected
        assert handler.get_complete_url(
            api_base=handler._resolve_api_base(api_base),
            api_key="PASSTHROUGH_KEY",
            model="datarobot/vertex_ai/gemini-1.5-flash-002",
            optional_params={},
            litellm_params={},
        ) == expected_url

        # Check that the complete url with the original api_base does not change the url
        if api_base is not None:
            assert handler.get_complete_url(
                api_base=api_base,
                api_key="PASSTHROUGH_KEY",
                model="datarobot/vertex_ai/gemini-1.5-flash-002",
                optional_params={},
                litellm_params={},
            ) == api_base

    def test_resolve_api_base_with_environment_variable(self, handler):
        os.environ["DATAROBOT_ENDPOINT"] = "https://env.datarobot.com"
        assert handler._resolve_api_base(None) == "https://env.datarobot.com/api/v2/genai/llmgw/chat/completions/"
        del os.environ["DATAROBOT_ENDPOINT"]

    @pytest.mark.parametrize(
        "api_key, expected_api_key",
        [
            (None, "fake-api-key"),
            ("PASSTHROUGH_KEY", "PASSTHROUGH_KEY"),
        ]
    )
    def test_resolve_api_key(self, api_key, expected_api_key, handler):
        assert handler._resolve_api_key(api_key) == expected_api_key

    def test_resolve_api_key_with_environment_variable(self, handler):
        os.environ["DATAROBOT_API_TOKEN"] = "env_key"
        assert handler._resolve_api_key(None) == "env_key"
        del os.environ["DATAROBOT_API_TOKEN"]
