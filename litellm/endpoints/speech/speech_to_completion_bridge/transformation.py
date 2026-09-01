from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast

from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.constants import OPENAI_CHAT_COMPLETION_PARAMS

if TYPE_CHECKING:
    from litellm import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import ChatCompletionUserMessage, HttpxBinaryResponseContent
    from litellm.types.utils import ModelResponse


def _completion_response_cost(model_response: "ModelResponse") -> float | None:
    hidden_params: Final = getattr(model_response, "_hidden_params", None)
    if not isinstance(hidden_params, dict):
        return None
    response_cost: Final = hidden_params.get("response_cost")
    return response_cost if isinstance(response_cost, float) else None


GEMINI_TTS_CHAT_AUDIO_FORMAT: Final = "pcm16"


class ChatAudioParam(TypedDict):
    voice: ReadOnly[str]
    format: ReadOnly[NotRequired[str]]


class SpeechToCompletionBridgeTransformationHandler:
    def _chat_completion_params(self, optional_params: Mapping[str, object]) -> Mapping[str, object]:
        return MappingProxyType(
            {
                param: value
                for param, value in optional_params.items()
                if param in OPENAI_CHAT_COMPLETION_PARAMS and param != "response_format"
            }
        )

    def _chat_audio_format(self, model: str, optional_params: Mapping[str, object]) -> str | None:
        if self._is_gemini_tts_model(model):
            return GEMINI_TTS_CHAT_AUDIO_FORMAT
        response_format: Final = optional_params.get("response_format")
        return response_format if isinstance(response_format, str) else None

    def _chat_audio_param(
        self, model: str, voice: str | Mapping[str, object] | None, optional_params: Mapping[str, object]
    ) -> ChatAudioParam | None:
        if not isinstance(voice, str):
            return None
        audio_format: Final = self._chat_audio_format(model, optional_params)
        if audio_format is None:
            voice_only: Final[ChatAudioParam] = {"voice": voice}
            return voice_only
        audio: Final[ChatAudioParam] = {"voice": voice, "format": audio_format}
        return audio

    def transform_request(
        self,
        model: str,
        input: str,
        voice: str | dict | None,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
        litellm_logging_obj: "LiteLLMLoggingObj",
        custom_llm_provider: str,
    ) -> dict:
        user_message: Final[ChatCompletionUserMessage] = {"role": "user", "content": input}
        return_kwargs: Final = {
            "model": model,
            "messages": [user_message],
            "modalities": ["audio"],
            **self._chat_completion_params(optional_params),
            "audio": self._chat_audio_param(model, voice, optional_params),
            **litellm_params,
            "headers": headers,
            "litellm_logging_obj": litellm_logging_obj,
            "custom_llm_provider": custom_llm_provider,
        }
        return {k: v for k, v in return_kwargs.items() if v is not None}

    def _convert_pcm16_to_wav(self, pcm_data: bytes, sample_rate: int = 24000, channels: int = 1) -> bytes:
        """
        Convert raw PCM16 data to WAV format.

        Args:
            pcm_data: Raw PCM16 audio data
            sample_rate: Sample rate in Hz (Gemini TTS typically uses 24000)
            channels: Number of audio channels (1 for mono)

        Returns:
            bytes: WAV formatted audio data
        """
        import struct

        # WAV header parameters
        byte_rate: Final = sample_rate * channels * 2  # 2 bytes per sample (16-bit)
        block_align: Final = channels * 2
        data_size: Final = len(pcm_data)
        file_size: Final = 36 + data_size

        # Create WAV header
        wav_header: Final = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",  # Chunk ID
            file_size,  # Chunk Size
            b"WAVE",  # Format
            b"fmt ",  # Subchunk1 ID
            16,  # Subchunk1 Size (PCM)
            1,  # Audio Format (PCM)
            channels,  # Number of Channels
            sample_rate,  # Sample Rate
            byte_rate,  # Byte Rate
            block_align,  # Block Align
            16,  # Bits per Sample
            b"data",  # Subchunk2 ID
            data_size,  # Subchunk2 Size
        )

        return wav_header + pcm_data

    def _is_gemini_tts_model(self, model: str) -> bool:
        """Check if the model is a Gemini TTS model that returns PCM16 data."""
        return "gemini" in model.lower() and ("tts" in model.lower() or "preview-tts" in model.lower())

    def transform_response(self, model_response: "ModelResponse") -> "HttpxBinaryResponseContent":
        import base64

        import httpx

        from litellm.types.llms.openai import HttpxBinaryResponseContent
        from litellm.types.utils import Choices

        audio_part: Final = cast(Choices, model_response.choices[0]).message.audio
        if audio_part is None:
            raise ValueError("No audio part found in the response")
        audio_content: Final = audio_part.data

        # Decode base64 to get binary content
        binary_data = base64.b64decode(audio_content)

        # Check if this is a Gemini TTS model that returns raw PCM16 data
        model: Final = getattr(model_response, "model", "")
        headers: Final = {}
        if self._is_gemini_tts_model(model):
            # Convert PCM16 to WAV format for proper audio file playback
            binary_data = self._convert_pcm16_to_wav(binary_data)
            headers["Content-Type"] = "audio/wav"
        else:
            headers["Content-Type"] = "audio/mpeg"

        # Create an httpx.Response object
        response: Final = httpx.Response(status_code=200, content=binary_data, headers=headers)
        binary_response: Final = HttpxBinaryResponseContent(response)
        binary_response.set_response_cost(_completion_response_cost(model_response))
        return binary_response
