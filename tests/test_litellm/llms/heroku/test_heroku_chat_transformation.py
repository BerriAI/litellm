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

    @pytest.mark.respx()
    def test_heroku_tool_calling(self, respx_mock):
        """Test that the Heroku tool calling API is called correctly"""
        config = HerokuChatConfig()
        headers = {}
        api_key = "fake-heroku-key"

        litellm.disable_aiohttp_transport = True

        model = "heroku/claude-4-sonnet"

        respx_mock.post("https://us.inference.heroku.com/v1/chat/completions").respond(
            json={
                "id": "chatcmpl-1859428879fc791b17d73",
                "object": "chat.completion",
                "created": 1754506683,
                "model": "claude-4-sonnet",
                "system_fingerprint": "heroku-inf-cp42st",
                "choices": [
                    {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "refusal": None,
                        "tool_calls": [
                        {
                            "id": "tooluse_dV3Vtnb-S9-Z_YFicSv2Gw",
                            "type": "function",
                            "function": {
                            "name": "get_current_weather",
                            "arguments": "{\"location\":\"Portland, OR\"}"
                            }
                        }
                        ],
                        "content": "Let me check the current weather in Portland for you."
                    },
                    "finish_reason": "tool_calls"
                    }
                ],
                "usage": {
                    "prompt_tokens": 354,
                    "completion_tokens": 69,
                    "total_tokens": 423
                }
            },
            status_code=200,
        )

        response = completion(
            model=model,
            messages=[{"role": "user", "content": "What's the weather in Portland?"}],
            tools=[{
                "type": "function", 
                "function": {
                    "name": "get_current_weather", 
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. Portland, OR"
                            }
                        },
                        "required": [
                            "location"
                        ]
                    }
                }
            }],
            tool_choice="auto",
        )
        print(response)
        assert response.choices[0].message.content == "Let me check the current weather in Portland for you."
        assert response.choices[0].message.tool_calls[0].id == "tooluse_dV3Vtnb-S9-Z_YFicSv2Gw"
        assert response.choices[0].message.tool_calls[0].type == "function"
        assert response.choices[0].message.tool_calls[0].function.name == "get_current_weather"
        assert response.choices[0].message.tool_calls[0].function.arguments == "{\"location\":\"Portland, OR\"}"

        assert response.usage.prompt_tokens == 354
        assert response.usage.completion_tokens == 69
        assert response.usage.total_tokens == 423