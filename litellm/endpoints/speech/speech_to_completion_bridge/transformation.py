from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast

from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.constants import OPENAI_CHAT_COMPLETION_PARAMS
from litellm.llms.gemini.speech_audio import (
    gemini_tts_chat_audio_format,
    gemini_tts_speech_response,
    validate_gemini_tts_speech_format,
)

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


class ChatAudioParam(TypedDict):
    voice: ReadOnly[str]
    format: ReadOnly[NotRequired[str]]


class SpeechToCompletionBridgeTransformationHandler:
    def _validate_response_format(
        self, model: str, custom_llm_provider: str, optional_params: Mapping[str, object]
    ) -> None:
        validate_gemini_tts_speech_format(model, custom_llm_provider, optional_params)

    def _chat_completion_params(self, optional_params: Mapping[str, object]) -> Mapping[str, object]:
        return MappingProxyType(
            {
                param: value
                for param, value in optional_params.items()
                if param in OPENAI_CHAT_COMPLETION_PARAMS and param != "response_format"
            }
        )

    def _chat_audio_format(self, model: str, optional_params: Mapping[str, object]) -> str | None:
        gemini_format: Final = gemini_tts_chat_audio_format(model)
        if gemini_format is not None:
            return gemini_format
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
        self._validate_response_format(model, custom_llm_provider, optional_params)
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

    def transform_response(
        self, model_response: "ModelResponse", response_format: str | None
    ) -> "HttpxBinaryResponseContent":
        import base64

        import httpx

        from litellm.types.llms.openai import HttpxBinaryResponseContent
        from litellm.types.utils import Choices

        audio_part: Final = cast(Choices, model_response.choices[0]).message.audio
        if audio_part is None:
            raise ValueError("No audio part found in the response")
        decoded_audio: Final = base64.b64decode(audio_part.data)

        model: Final = getattr(model_response, "model", "")
        provider_body: Final = gemini_tts_speech_response(model, decoded_audio, response_format)
        content, content_type = provider_body if provider_body is not None else (decoded_audio, "audio/mpeg")
        response: Final = httpx.Response(
            status_code=200, content=content, headers=MappingProxyType({"Content-Type": content_type})
        )
        binary_response: Final = HttpxBinaryResponseContent(response)
        binary_response.set_response_cost(_completion_response_cost(model_response))
        return binary_response
