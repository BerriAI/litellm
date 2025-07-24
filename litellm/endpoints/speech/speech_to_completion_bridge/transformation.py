from typing import TYPE_CHECKING, Optional, Union, cast

from litellm.constants import OPENAI_CHAT_COMPLETION_PARAMS

if TYPE_CHECKING:
    from litellm import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import HttpxBinaryResponseContent
    from litellm.types.utils import ModelResponse


class SpeechToCompletionBridgeTransformationHandler:
    def transform_request(
        self,
        model: str,
        input: str,
        voice: Optional[Union[str, dict]],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
        litellm_logging_obj: "LiteLLMLoggingObj",
        custom_llm_provider: str,
    ) -> dict:
        passed_optional_params = {}
        for op in optional_params:
            if op in OPENAI_CHAT_COMPLETION_PARAMS:
                passed_optional_params[op] = optional_params[op]

        if voice is not None:
            if isinstance(voice, str):
                passed_optional_params["audio"] = {"voice": voice}
                if "response_format" in optional_params:
                    passed_optional_params["audio"]["format"] = optional_params[
                        "response_format"
                    ]

        return_kwargs = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": input,
                }
            ],
            "modalities": ["audio"],
            **passed_optional_params,
            **litellm_params,
            "headers": headers,
            "litellm_logging_obj": litellm_logging_obj,
            "custom_llm_provider": custom_llm_provider,
        }

        # filter out None values
        return_kwargs = {k: v for k, v in return_kwargs.items() if v is not None}
        return return_kwargs

    def _convert_pcm16_to_wav(
        self, pcm_data: bytes, sample_rate: int = 24000, channels: int = 1
    ) -> bytes:
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
        byte_rate = sample_rate * channels * 2  # 2 bytes per sample (16-bit)
        block_align = channels * 2
        data_size = len(pcm_data)
        file_size = 36 + data_size

        # Create WAV header
        wav_header = struct.pack(
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
        return "gemini" in model.lower() and (
            "tts" in model.lower() or "preview-tts" in model.lower()
        )

    def transform_response(
        self, model_response: "ModelResponse"
    ) -> "HttpxBinaryResponseContent":
        import base64

        import httpx

        from litellm.types.llms.openai import HttpxBinaryResponseContent
        from litellm.types.utils import Choices

        audio_part = cast(Choices, model_response.choices[0]).message.audio
        if audio_part is None:
            raise ValueError("No audio part found in the response")
        audio_content = audio_part.data

        # Decode base64 to get binary content
        binary_data = base64.b64decode(audio_content)

        # Check if this is a Gemini TTS model that returns raw PCM16 data
        model = getattr(model_response, "model", "")
        headers = {}
        if self._is_gemini_tts_model(model):
            # Convert PCM16 to WAV format for proper audio file playback
            binary_data = self._convert_pcm16_to_wav(binary_data)
            headers["Content-Type"] = "audio/wav"
        else:
            headers["Content-Type"] = "audio/mpeg"

        # Create an httpx.Response object
        response = httpx.Response(status_code=200, content=binary_data, headers=headers)
        return HttpxBinaryResponseContent(response)
