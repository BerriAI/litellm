import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import json
from litellm.llms.google_code_assist.chat import (
    GoogleCodeAssistChat,
    get_google_code_assist_chat,
)
from litellm.types.utils import ModelResponse


class TestGoogleCodeAssist:
    def test_get_google_code_assist_chat_returns_singleton(self):
        assert get_google_code_assist_chat() is get_google_code_assist_chat()

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

    @patch("litellm.llms.gemini.common_utils.get_gemini_oauth_token")
    def test_completion_raises_when_token_missing(self, mock_get_token):
        mock_get_token.return_value = {"token": None}

        handler = GoogleCodeAssistChat()
        with pytest.raises(Exception, match="Missing Gemini OAuth token value"):
            handler.completion(
                model="google_code_assist/gemini-1.5-flash",
                messages=[{"role": "user", "content": "hi"}],
                model_response=ModelResponse(),
                print_verbose=False,
                logging_obj=MagicMock(),
                optional_params={},
                litellm_params={},
            )

    @pytest.mark.asyncio
    @patch("litellm.llms.google_code_assist.chat.AsyncHTTPHandler.post")
    @patch(
        "litellm.llms.google_code_assist.chat.AsyncHTTPHandler.close",
        new_callable=AsyncMock,
    )
    @patch("litellm.llms.gemini.common_utils.get_gemini_oauth_token")
    async def test_acompletion_basic(
        self, mock_get_token, mock_async_close, mock_async_post
    ):
        """
        Test async completion.
        """
        mock_get_token.return_value = {"token": "test-token"}

        # Handshake (async)
        handshake_data = {"cloudaicompanionProject": "final-project"}
        mock_handshake_resp = httpx.Response(
            status_code=200,
            content=json.dumps(handshake_data).encode(),
            request=httpx.Request("POST", "https://handshake"),
        )

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
        mock_async_post.side_effect = [mock_handshake_resp, mock_completion_resp]

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
        mock_async_close.assert_awaited_once()

    @pytest.mark.asyncio
    @patch(
        "litellm.llms.google_code_assist.chat.asyncio.to_thread", new_callable=AsyncMock
    )
    async def test_acompletion_uses_thread_for_oauth_lookup(self, mock_to_thread):
        mock_to_thread.return_value = {"token": "test-token"}

        with (
            patch(
                "litellm.llms.google_code_assist.chat.GoogleCodeAssistChat._ahandle_handshake",
                new_callable=AsyncMock,
                return_value="final-project",
            ),
            patch(
                "litellm.llms.google_code_assist.chat.AsyncHTTPHandler.post",
                new_callable=AsyncMock,
            ) as mock_async_post,
            patch(
                "litellm.llms.google_code_assist.chat.AsyncHTTPHandler.close",
                new_callable=AsyncMock,
            ),
        ):
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
            await handler.acompletion(
                model="google_code_assist/gemini-1.5-flash",
                messages=[{"role": "user", "content": "hi"}],
                model_response=ModelResponse(),
                print_verbose=False,
                logging_obj=MagicMock(),
                optional_params={},
                litellm_params={},
            )

        mock_to_thread.assert_awaited_once()
