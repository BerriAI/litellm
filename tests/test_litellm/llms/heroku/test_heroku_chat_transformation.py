import os
import pytest
import litellm
from litellm import completion
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from unittest.mock import patch
from litellm.llms.heroku.chat.transformation import HerokuChatConfig

os.environ["HEROKU_API_BASE"] = "https://us.inference.heroku.com"
os.environ["HEROKU_API_KEY"] = "fake-heroku-key"

class TestHerokuChatConfig:
    def test_default_api_base(self):
        """Test that default API base is used when none is provided"""
        config = HerokuChatConfig()
        headers = {}
        api_key = "fake-heroku-key"

        # Call validate_environment without specifying api_base
        result = config.validate_environment(
            headers=headers,
            model="claude-3-5-haiku",
            messages=[{"role": "user", "content": "Hey"}],
            optional_params={},
            litellm_params={},
            api_key=api_key,
            api_base=None,  # Not providing api_base
        )

        # Verify headers are still set correctly
        assert result["Authorization"] == f"Bearer {api_key}"
        assert result["Content-Type"] == "application/json"

    @pytest.mark.respx()
    def test_heroku_chat_mock(self, respx_mock):
        """Test that the Heroku chat API is called correctly"""
        
        litellm.disable_aiohttp_transport = True

        model = "heroku/claude-3-5-haiku"
        model_name = "claude-3-5-haiku"

        respx_mock.post("https://us.inference.heroku.com/v1/chat/completions").respond(
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "It's me, Mia! How are you?",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 12,
                    "total_tokens": 21,
                },
            },
            status_code=200,
        )

        response = completion(
            model=model,
            messages=[
                {"role": "user", "content": "write code for saying hey from LiteLLM"}
            ],
            extended_thinking={ "enabled": True, "include_reasoning":True }
        )

        # Verify the request was made with correct headers
        assert len(respx_mock.calls) == 1
        request = respx_mock.calls[0].request
        
        assert request.headers["Authorization"] == f"Bearer {os.environ['HEROKU_API_KEY']}"
        assert request.headers["Content-Type"] == "application/json"

        assert response.choices[0].message.content == "It's me, Mia! How are you?"