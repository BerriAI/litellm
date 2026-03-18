import pytest
from unittest.mock import MagicMock, patch
import httpx
import json
from litellm.llms.google_code_assist.chat import GoogleCodeAssistChat
from litellm.types.utils import ModelResponse


class TestGoogleCodeAssist:
    @patch("litellm.llms.google_code_assist.chat._get_httpx_client")
    @patch("litellm.llms.gemini.common_utils.get_gemini_oauth_token")
    def test_completion_basic(self, mock_get_token, mock_get_client):
        """
        Test basic completion with mocked handshake and API call.
        """
        mock_get_token.return_value = {
            "token": "test-token",
            "project_id": "test-project",
        }

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Mock handshake response
        handshake_data = {"cloudaicompanionProject": "final-project"}
        mock_handshake_resp = httpx.Response(
            status_code=200,
            content=json.dumps(handshake_data).encode(),
            request=httpx.Request("POST", "https://handshake"),
        )

        # Mock completion response
        completion_data = {
            "response": {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": "Hello world"}],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 5,
                    "candidatesTokenCount": 2,
                    "totalTokenCount": 7,
                },
            }
        }
        mock_completion_resp = httpx.Response(
            status_code=200,
            content=json.dumps(completion_data).encode(),
            request=httpx.Request("POST", "https://completion"),
        )

        mock_client.post.side_effect = [mock_handshake_resp, mock_completion_resp]

        handler = GoogleCodeAssistChat()
        response = handler.completion(
            model="google_code_assist/gemini-1.5-flash",
            messages=[{"role": "user", "content": "hi"}],
            model_response=ModelResponse(),
            print_verbose=False,
            logging_obj=MagicMock(),
            optional_params={},
            litellm_params={},
        )

        assert response.choices[0].message.content == "Hello world"
        assert response.usage.total_tokens == 7

    @pytest.mark.asyncio
    @patch("litellm.llms.google_code_assist.chat.AsyncHTTPHandler.post")
    @patch("litellm.llms.gemini.common_utils.get_gemini_oauth_token")
    @patch("httpx.Client.post")
    async def test_acompletion_basic(
        self, mock_sync_post, mock_get_token, mock_async_post
    ):
        """
        Test async completion.
        """
        mock_get_token.return_value = {"token": "test-token"}

        # Handshake (sync)
        handshake_data = {"cloudaicompanionProject": "final-project"}
        mock_handshake_resp = httpx.Response(
            status_code=200,
            content=json.dumps(handshake_data).encode(),
            request=httpx.Request("POST", "https://handshake"),
        )
        mock_sync_post.return_value = mock_handshake_resp

        # Completion (async)
        completion_data = {
            "response": {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": "Async success"}],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 5,
                    "candidatesTokenCount": 2,
                    "totalTokenCount": 7,
                },
            }
        }
        mock_completion_resp = httpx.Response(
            status_code=200,
            content=json.dumps(completion_data).encode(),
            request=httpx.Request("POST", "https://completion"),
        )
        mock_async_post.return_value = mock_completion_resp

        handler = GoogleCodeAssistChat()
        response = await handler.acompletion(
            model="google_code_assist/gemini-1.5-flash",
            messages=[{"role": "user", "content": "hi"}],
            model_response=ModelResponse(),
            print_verbose=False,
            logging_obj=MagicMock(),
            optional_params={},
            litellm_params={},
        )

        assert response.choices[0].message.content == "Async success"
