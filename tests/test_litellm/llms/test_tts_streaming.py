"""
Tests for TTS streaming (stream=True) through the public litellm.speech()/aspeech() API.

Covers:
- OpenAI sync/async: mock with_streaming_response, assert HttpxStreamHandler returned
- httpx-based provider sync/async: mock HTTPHandler.post, assert stream=True forwarded
- Non-streaming provider: stream=True on supports_streaming=False returns HttpxBinaryResponseContent
- Default behavior: no stream param returns HttpxBinaryResponseContent
"""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.httpx_stream_handler import HttpxStreamHandler


def _make_fake_httpx_response(status_code: int = 200) -> httpx.Response:
    """Create a minimal httpx.Response that can be used in tests."""
    request = httpx.Request("POST", "https://fake.example.com/v1/audio/speech")
    response = httpx.Response(
        status_code=status_code,
        request=request,
        content=b"fake-audio-bytes",
    )
    return response


# ---------------------------------------------------------------------------
# OpenAI sync — stream=True
# ---------------------------------------------------------------------------
class TestOpenAISyncStreaming:
    def test_openai_speech_stream_returns_handler(self):
        """litellm.speech(stream=True) with OpenAI should return HttpxStreamHandler."""
        from openai import OpenAI

        openai_client = OpenAI(api_key="fake-key")

        fake_response = _make_fake_httpx_response()

        # Mock the with_streaming_response.create context manager
        fake_streamed = MagicMock()
        fake_streamed.http_response = fake_response

        @contextmanager
        def fake_ctx(*args, **kwargs):
            yield fake_streamed

        mock_create = MagicMock(side_effect=fake_ctx)

        with patch.object(
            openai_client.audio.speech.with_streaming_response,
            "create",
            mock_create,
        ):
            result = litellm.speech(
                model="openai/tts-1",
                input="Hello world",
                voice="alloy",
                client=openai_client,
                stream=True,
            )

        assert isinstance(result, HttpxStreamHandler)
        mock_create.assert_called_once()


# ---------------------------------------------------------------------------
# OpenAI async — stream=True
# ---------------------------------------------------------------------------
class TestOpenAIAsyncStreaming:
    @pytest.mark.asyncio
    async def test_openai_aspeech_stream_returns_handler(self):
        """litellm.aspeech(stream=True) with OpenAI should return HttpxStreamHandler."""
        from openai import AsyncOpenAI

        openai_client = AsyncOpenAI(api_key="fake-key")

        fake_response = _make_fake_httpx_response()

        fake_streamed = MagicMock()
        fake_streamed.http_response = fake_response

        # Build an async context manager mock
        async_ctx = AsyncMock()
        async_ctx.__aenter__ = AsyncMock(return_value=fake_streamed)
        async_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_create = MagicMock(return_value=async_ctx)

        with patch.object(
            openai_client.audio.speech.with_streaming_response,
            "create",
            mock_create,
        ):
            result = await litellm.aspeech(
                model="openai/tts-1",
                input="Hello world",
                voice="alloy",
                client=openai_client,
                stream=True,
            )

        assert isinstance(result, HttpxStreamHandler)
        mock_create.assert_called_once()


# ---------------------------------------------------------------------------
# httpx-based provider sync — stream=True (ElevenLabs)
# ---------------------------------------------------------------------------
class TestHttpxProviderSyncStreaming:
    @patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
    def test_elevenlabs_speech_stream_returns_handler(self, mock_post):
        """litellm.speech(stream=True) with ElevenLabs should pass stream=True
        to HTTPHandler.post and return HttpxStreamHandler."""
        fake_response = _make_fake_httpx_response()
        mock_post.return_value = fake_response

        result = litellm.speech(
            model="elevenlabs/eleven_multilingual_v2",
            input="Hello world",
            voice="test-voice-id",
            api_key="fake-elevenlabs-key",
            stream=True,
        )

        assert isinstance(result, HttpxStreamHandler)
        # Verify stream=True was passed to the HTTP client
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs.get("stream") is True or (
            len(call_kwargs.args) > 0 and call_kwargs.kwargs.get("stream") is True
        )


# ---------------------------------------------------------------------------
# httpx-based provider async — stream=True (ElevenLabs)
# ---------------------------------------------------------------------------
class TestHttpxProviderAsyncStreaming:
    @pytest.mark.asyncio
    @patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post")
    async def test_elevenlabs_aspeech_stream_returns_handler(self, mock_post):
        """litellm.aspeech(stream=True) with ElevenLabs should return HttpxStreamHandler."""
        fake_response = _make_fake_httpx_response()
        mock_post.return_value = fake_response

        result = await litellm.aspeech(
            model="elevenlabs/eleven_multilingual_v2",
            input="Hello world",
            voice="test-voice-id",
            api_key="fake-elevenlabs-key",
            stream=True,
        )

        assert isinstance(result, HttpxStreamHandler)
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs.get("stream") is True


# ---------------------------------------------------------------------------
# Non-streaming provider — stream=True on supports_streaming=False
# ---------------------------------------------------------------------------
class TestNonStreamingProvider:
    @patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
    def test_minimax_stream_true_returns_binary(self, mock_post):
        """When supports_streaming=False (MiniMax), stream=True should be
        ignored and return HttpxBinaryResponseContent."""
        import json

        from litellm.types.llms.openai import HttpxBinaryResponseContent

        # MiniMax returns JSON with hex-encoded audio
        minimax_response_body = json.dumps(
            {
                "status": 0,
                "data": {"audio": "48454c4c4f"},  # hex for "HELLO"
            }
        ).encode()
        request = httpx.Request("POST", "https://fake.example.com/v1/t2a_v2")
        fake_response = httpx.Response(
            status_code=200,
            request=request,
            content=minimax_response_body,
        )
        mock_post.return_value = fake_response

        result = litellm.speech(
            model="minimax/speech-02-hd",
            input="Hello world",
            voice="Wise_Woman",
            api_key="fake-minimax-key",
            stream=True,
        )

        assert isinstance(result, HttpxBinaryResponseContent)
        # Verify stream was NOT passed as True to HTTPHandler.post
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs.get("stream") is not True


# ---------------------------------------------------------------------------
# Default behavior — no stream param
# ---------------------------------------------------------------------------
class TestDefaultBehavior:
    @patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post")
    def test_elevenlabs_no_stream_returns_binary(self, mock_post):
        """Without stream param, litellm.speech() should return HttpxBinaryResponseContent."""
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        fake_response = _make_fake_httpx_response()
        mock_post.return_value = fake_response

        result = litellm.speech(
            model="elevenlabs/eleven_multilingual_v2",
            input="Hello world",
            voice="test-voice-id",
            api_key="fake-elevenlabs-key",
        )

        assert isinstance(result, HttpxBinaryResponseContent)

    def test_openai_no_stream_returns_binary(self):
        """Without stream param, litellm.speech() with OpenAI should return
        HttpxBinaryResponseContent."""
        from openai import OpenAI

        from litellm.types.llms.openai import HttpxBinaryResponseContent

        openai_client = OpenAI(api_key="fake-key")

        fake_response = _make_fake_httpx_response()

        # Mock the regular (non-streaming) create
        fake_speech_response = MagicMock()
        fake_speech_response.response = fake_response

        mock_create = MagicMock(return_value=fake_speech_response)

        with patch.object(
            openai_client.audio.speech,
            "create",
            mock_create,
        ):
            result = litellm.speech(
                model="openai/tts-1",
                input="Hello world",
                voice="alloy",
                client=openai_client,
            )

        assert isinstance(result, HttpxBinaryResponseContent)
        mock_create.assert_called_once()
