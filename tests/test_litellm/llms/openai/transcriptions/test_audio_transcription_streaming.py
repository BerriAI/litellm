"""
Tests for streaming audio transcription support.

Covers:
- whisper transformation skips verbose_json override on stream=True
- get_supported_openai_params includes "stream" for whisper + gpt
- get_optional_params_transcription threads `stream` through
- BaseAudioTranscriptionConfig.transform_audio_transcription_streaming_chunk
  is pass-through by default
- BaseLLMHTTPHandler.audio_transcriptions returns TranscriptionStreamingResponse
  when data["stream"] is truthy
"""

from unittest.mock import MagicMock, patch

import pytest

from litellm.llms.base_llm.audio_transcription.transformation import (
    BaseAudioTranscriptionConfig,
)
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.openai.transcriptions.gpt_transformation import (
    OpenAIGPTAudioTranscriptionConfig,
)
from litellm.llms.openai.transcriptions.whisper_transformation import (
    OpenAIWhisperAudioTranscriptionConfig,
)
from litellm.types.utils import (
    TranscriptionResponse,
    TranscriptionStreamingResponse,
)
from litellm.utils import get_optional_params_transcription


class TestWhisperTransformation:
    """The verbose_json override must NOT fire when stream=True."""

    def test_streaming_skips_verbose_json_override(self):
        config = OpenAIWhisperAudioTranscriptionConfig()
        result = config.transform_audio_transcription_request(
            model="whisper-1",
            audio_file=("audio.wav", b"fake-bytes", "audio/wav"),
            optional_params={"stream": True, "response_format": "json"},
            litellm_params={},
        )
        assert isinstance(result.data, dict)
        # bools are stringified for httpx multipart compatibility
        assert result.data["stream"] == "true"
        assert result.data["response_format"] == "json"

    def test_non_streaming_applies_verbose_json_override(self):
        """When stream is absent, override fires for json/text/missing format."""
        config = OpenAIWhisperAudioTranscriptionConfig()
        result = config.transform_audio_transcription_request(
            model="whisper-1",
            audio_file=("audio.wav", b"fake-bytes", "audio/wav"),
            optional_params={"response_format": "json"},
            litellm_params={},
        )
        assert isinstance(result.data, dict)
        assert result.data["response_format"] == "verbose_json"

    def test_streaming_preserves_explicit_format(self):
        """User-specified srt under stream=True must not be clobbered."""
        config = OpenAIWhisperAudioTranscriptionConfig()
        result = config.transform_audio_transcription_request(
            model="whisper-1",
            audio_file=("a.wav", b"x", "audio/wav"),
            optional_params={"stream": True, "response_format": "srt"},
            litellm_params={},
        )
        assert isinstance(result.data, dict)
        assert result.data["response_format"] == "srt"

    def test_supported_params_includes_stream(self):
        config = OpenAIWhisperAudioTranscriptionConfig()
        assert "stream" in config.get_supported_openai_params(model="whisper-1")


class TestGPTTranscriptionTransformation:
    def test_supported_params_includes_stream(self):
        config = OpenAIGPTAudioTranscriptionConfig()
        assert "stream" in config.get_supported_openai_params(model="gpt-4o-transcribe")


class TestGetOptionalParamsTranscription:
    """The `stream` kwarg must flow through into the optional_params dict."""

    def test_stream_threaded_through_for_openai(self):
        params = get_optional_params_transcription(
            model="whisper-1",
            custom_llm_provider="openai",
            stream=True,
        )
        assert params.get("stream") is True

    def test_stream_default_none_not_emitted(self):
        params = get_optional_params_transcription(
            model="whisper-1",
            custom_llm_provider="openai",
        )
        # default None means non-default-params filter drops it
        assert "stream" not in params or params.get("stream") is None


class TestBaseStreamingChunkHook:
    """Default hook is pass-through; subclass override path works."""

    def test_default_pass_through(self):
        # BaseAudioTranscriptionConfig is abstract — use a concrete impl
        config = OpenAIWhisperAudioTranscriptionConfig()
        chunk = b'data: {"text":"hi"}\n\n'
        assert config.transform_audio_transcription_streaming_chunk(chunk) == chunk

    def test_method_is_defined_on_base(self):
        # Defensive: the hook must exist on the base class so providers
        # that don't override still inherit pass-through.
        assert hasattr(
            BaseAudioTranscriptionConfig,
            "transform_audio_transcription_streaming_chunk",
        )


class TestHTTPHandlerStreamingBranch:
    """audio_transcriptions sync path returns TranscriptionStreamingResponse."""

    def test_sync_streaming_returns_streaming_response(self):
        handler = BaseLLMHTTPHandler()
        provider_config = OpenAIWhisperAudioTranscriptionConfig()

        # Mock httpx.Response with iter_bytes
        mock_response = MagicMock()
        mock_response.headers = {"content-type": "text/event-stream"}
        mock_response.iter_bytes = MagicMock(
            return_value=iter([b'data: {"text":"hi"}\n\n', b"data: [DONE]\n\n"])
        )
        mock_response.close = MagicMock()

        mock_client = MagicMock()
        mock_client.post = MagicMock(return_value=mock_response)

        with patch(
            "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
            return_value=mock_client,
        ):
            result = handler.audio_transcriptions(
                model="whisper-1",
                audio_file=("a.wav", b"x", "audio/wav"),
                optional_params={"stream": True},
                litellm_params={},
                model_response=TranscriptionResponse(),
                timeout=60.0,
                max_retries=0,
                logging_obj=MagicMock(),
                api_key="sk-test",
                api_base="https://api.openai.com",
                custom_llm_provider="openai",
                client=None,
                atranscription=False,
                provider_config=provider_config,
            )

        assert isinstance(result, TranscriptionStreamingResponse)
        # client was invoked with stream=True
        _, kwargs = mock_client.post.call_args
        assert kwargs.get("stream") is True

        # Iterator yields the upstream chunks unchanged (default pass-through)
        chunks = list(result)
        assert chunks == [b'data: {"text":"hi"}\n\n', b"data: [DONE]\n\n"]

    def test_sync_non_streaming_returns_transcription_response(self):
        handler = BaseLLMHTTPHandler()
        provider_config = OpenAIWhisperAudioTranscriptionConfig()

        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json = MagicMock(return_value={"text": "hello world"})

        mock_client = MagicMock()
        mock_client.post = MagicMock(return_value=mock_response)

        with patch(
            "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
            return_value=mock_client,
        ):
            result = handler.audio_transcriptions(
                model="whisper-1",
                audio_file=("a.wav", b"x", "audio/wav"),
                optional_params={},
                litellm_params={},
                model_response=TranscriptionResponse(),
                timeout=60.0,
                max_retries=0,
                logging_obj=MagicMock(),
                api_key="sk-test",
                api_base="https://api.openai.com",
                custom_llm_provider="openai",
                client=None,
                atranscription=False,
                provider_config=provider_config,
            )

        assert isinstance(result, TranscriptionResponse)
        _, kwargs = mock_client.post.call_args
        assert kwargs.get("stream") is False


class TestProxyStreamCoercion:
    """Proxy must coerce string 'true'/'false' from form data to bool."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("false", False),
            ("False", False),
            ("0", False),
        ],
    )
    def test_stream_coercion_logic(self, raw, expected):
        from litellm.proxy.proxy_server import _coerce_stream_form_field

        data = {"stream": raw}
        _coerce_stream_form_field(data)
        assert data["stream"] is expected
