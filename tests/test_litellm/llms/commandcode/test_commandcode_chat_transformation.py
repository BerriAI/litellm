import os
import pytest
import litellm
from litellm import completion
from litellm.llms.commandcode.chat.transformation import (
    CommandCodeOpenAIConfig,
    CommandCodeAnthropicConfig,
)

os.environ["COMMANDCODE_API_KEY"] = "fake-commandcode-key"


class TestCommandCodeChatConfig:
    def test_openai_config_validate_environment(self):
        """Test that headers are set correctly for OpenAI-compatible models"""
        config = CommandCodeOpenAIConfig()
        headers = {}
        api_key = "fake-commandcode-key"

        result = config.validate_environment(
            headers=headers,
            model="deepseek/deepseek-v4-flash",
            messages=[{"role": "user", "content": "Hey"}],
            optional_params={},
            litellm_params={},
            api_key=api_key,
            api_base=None,
        )

        assert result["Authorization"] == f"Bearer {api_key}"
        assert result["Content-Type"] == "application/json"

    def test_openai_config_get_complete_url(self):
        """Test that the correct URL is returned for OpenAI-compatible models"""
        config = CommandCodeOpenAIConfig()

        url = config.get_complete_url(
            api_base=None,
            api_key="fake-key",
            model="deepseek/deepseek-v4-flash",
            optional_params={},
            litellm_params={},
        )

        assert url == "https://api.commandcode.ai/provider/v1/chat/completions"

    def test_anthropic_config_validate_environment(self):
        """Test that headers are set correctly for Claude models"""
        config = CommandCodeAnthropicConfig()
        headers = {}
        api_key = "fake-commandcode-key"

        result = config.validate_environment(
            headers=headers,
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hey"}],
            optional_params={},
            litellm_params={},
            api_key=api_key,
            api_base=None,
        )

        assert result["Authorization"] == f"Bearer {api_key}"
        assert result["Content-Type"] == "application/json"
        assert result["anthropic-version"] == "2023-06-01"

    def test_anthropic_config_get_complete_url(self):
        """Test that the correct URL is returned for Claude models"""
        config = CommandCodeAnthropicConfig()

        url = config.get_complete_url(
            api_base=None,
            api_key="fake-key",
            model="claude-sonnet-4-6",
            optional_params={},
            litellm_params={},
        )

        assert url == "https://api.commandcode.ai/provider/v1/messages"

    def test_get_llm_provider_openai_model(self):
        """Test that provider detection works for OpenAI-compatible models"""
        model, provider, _, _ = litellm.get_llm_provider(
            model="commandcode/deepseek/deepseek-v4-flash"
        )
        assert provider == "commandcode"
        assert model == "deepseek/deepseek-v4-flash"

    def test_get_llm_provider_claude_model(self):
        """Test that provider detection works for Claude models"""
        model, provider, _, _ = litellm.get_llm_provider(
            model="commandcode/claude-sonnet-4-6"
        )
        assert provider == "commandcode"
        assert model == "claude-sonnet-4-6"

    @pytest.mark.respx()
    def test_commandcode_openai_mock(self, respx_mock):
        """Test OpenAI-compatible model call with mocked response"""
        litellm.disable_aiohttp_transport = True

        respx_mock.post(
            "https://api.commandcode.ai/provider/v1/chat/completions"
        ).respond(
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "deepseek/deepseek-v4-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello!",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 5,
                    "total_tokens": 14,
                },
            },
            status_code=200,
        )

        response = completion(
            model="commandcode/deepseek/deepseek-v4-flash",
            messages=[{"role": "user", "content": "Say hello."}],
        )

        assert len(respx_mock.calls) == 1
        request = respx_mock.calls[0].request
        assert request.headers["Authorization"] == "Bearer fake-commandcode-key"
        assert request.headers["Content-Type"] == "application/json"
        assert response.choices[0].message.content == "Hello!"

    @pytest.mark.respx()
    def test_commandcode_api_key_override(self, respx_mock):
        """Test that api_key parameter overrides environment variable"""
        litellm.disable_aiohttp_transport = True

        # Mock the endpoint
        respx_mock.post(
            "https://api.commandcode.ai/provider/v1/chat/completions"
        ).respond(
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "deepseek/deepseek-v4-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello!",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 5,
                    "total_tokens": 14,
                },
            },
            status_code=200,
        )

        # Call with custom api_key (should override env var)
        response = completion(
            model="commandcode/deepseek/deepseek-v4-flash",
            messages=[{"role": "user", "content": "Say hello."}],
            api_key="custom-key-from-param",  # ← This should be used, not env var
        )

        # Verify request was made
        assert len(respx_mock.calls) == 1
        request = respx_mock.calls[0].request

        # Verify the custom api_key was used in Authorization header
        assert request.headers["Authorization"] == "Bearer custom-key-from-param"
        assert response.choices[0].message.content == "Hello!"