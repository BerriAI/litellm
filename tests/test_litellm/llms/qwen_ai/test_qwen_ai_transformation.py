from unittest.mock import patch

from litellm.llms.qwen_ai.chat.transformation import QwenAIConfig


def _build_messages():
    return [{"role": "user", "content": "Hello"}]


class TestQwenAIConfig:
    def test_oauth_only_ignores_api_key(self):
        config = QwenAIConfig()

        with patch.object(
            config.authenticator, "get_access_token", return_value="oauth-token"
        ):
            api_base, api_key = config._get_openai_compatible_provider_info(
                api_base=None,
                api_key="manual-key",
            )

        assert api_key == "oauth-token"
        assert api_base is not None

    def test_validate_environment_sets_qwen_headers(self):
        config = QwenAIConfig()

        with patch.object(
            config.authenticator, "get_access_token", return_value="oauth-token"
        ):
            config._get_openai_compatible_provider_info(
                api_base=None,
                api_key=None,
            )

        headers = config.validate_environment(
            headers={},
            model="qwen_ai/qwen3-coder-plus",
            messages=_build_messages(),
            optional_params={},
            litellm_params={},
            api_key="oauth-token",
            api_base=None,
        )

        assert "User-Agent" in headers
        assert headers.get("X-DashScope-UserAgent")
        assert headers.get("X-DashScope-CacheControl") == "enable"
        assert headers.get("X-DashScope-AuthType") == "qwen-oauth"
