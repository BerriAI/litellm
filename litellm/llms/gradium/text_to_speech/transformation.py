"""
Gradium Text-to-Speech transformation

Maps OpenAI TTS spec to Gradium's REST TTS API.

Reference: https://docs.gradium.ai/api-reference/endpoint/tts-post
"""

from typing import TYPE_CHECKING, ClassVar, Final

import httpx
from httpx import Headers

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.text_to_speech.transformation import (
    BaseTextToSpeechConfig,
    TextToSpeechRequestData,
)
from litellm.secret_managers.main import get_secret_str

from ..common_utils import GradiumException

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import HttpxBinaryResponseContent
else:
    LiteLLMLoggingObj = object
    HttpxBinaryResponseContent = object


class GradiumTextToSpeechConfig(BaseTextToSpeechConfig):
    """
    Configuration for Gradium Text-to-Speech

    Reference: https://docs.gradium.ai/api-reference/endpoint/tts-post
    """

    TTS_BASE_URL = "https://api.gradium.ai"
    TTS_ENDPOINT_PATH = "/api/post/speech/tts"
    DEFAULT_OUTPUT_FORMAT = "wav"

    # OpenAI canonical voice names → Gradium library voice IDs. Anything not
    # listed is forwarded verbatim, so callers can pass raw Gradium voice IDs.
    VOICE_MAPPINGS: ClassVar[dict[str, str]] = {
        "alloy": "YTpq7expH9539ERJ",
        "echo": "s0PhgjzOTRD5wo5L",
        "fable": "1VAVLmmbQFDw7TMn",
        "onyx": "QkmUhBH4hIV2_BkY",
        "nova": "w9V1722uEmTkWqnR",
        "shimmer": "BbLb4TxdlrldgpHI",
    }

    # Gradium serves raw PCM and Ogg-wrapped Opus, plus WAV. It has no mp3,
    # aac or flac encoder, so those OpenAI formats have no mapping.
    FORMAT_MAPPINGS: ClassVar[dict[str, str]] = {
        "wav": "wav",
        "pcm": "pcm_24000",
        "opus": "opus",
    }

    def get_supported_openai_params(self, model: str) -> list:
        return ["voice", "response_format"]

    def _resolve_voice_id(self, voice: str | dict | None, params: dict) -> str:
        """Normalize the caller's `voice` into a Gradium voice_id."""
        candidate: str | None = None

        if isinstance(voice, str) and voice.strip():
            candidate = voice.strip()
        elif isinstance(voice, dict):
            for key in ("voice_id", "id", "name"):
                value = voice.get(key)
                if isinstance(value, str) and value.strip():
                    candidate = value.strip()
                    break

        if candidate is None:
            override = params.pop("voice_id", None)
            if isinstance(override, str) and override.strip():
                candidate = override.strip()

        if candidate is None:
            raise ValueError("Gradium voice_id is required. Pass `voice` when calling `litellm.speech()`.")

        return self.VOICE_MAPPINGS.get(candidate.lower(), candidate)

    def map_openai_params(
        self,
        model: str,
        optional_params: dict,
        voice: str | dict | None = None,
        drop_params: bool = False,
        kwargs: dict | None = None,
    ) -> tuple[str | None, dict]:
        params: Final = dict(optional_params) if optional_params else {}
        mapped_params: Final[dict[str, object]] = {}

        mapped_voice: Final = self._resolve_voice_id(voice, params)

        response_format: Final = params.pop("response_format", None)
        if isinstance(response_format, str):
            mapped_format = self.FORMAT_MAPPINGS.get(response_format)
            if mapped_format is None:
                if not (drop_params or litellm.drop_params):
                    raise litellm.UnsupportedParamsError(
                        status_code=400,
                        message=(
                            f"Gradium TTS does not support response_format={response_format!r}. "
                            f"Supported: {sorted(self.FORMAT_MAPPINGS)}. "
                            "Set `litellm.drop_params = True` to fall back to the default format."
                        ),
                    )
            else:
                mapped_params["output_format"] = mapped_format

        # Gradium synthesizes at the voice's native rate; there is no speed
        # control and no instructions field on the REST endpoint.
        params.pop("speed", None)
        params.pop("instructions", None)

        for key, value in params.items():
            if value is not None:
                mapped_params[key] = value

        return mapped_voice, mapped_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        api_key = api_key or litellm.api_key or get_secret_str("GRADIUM_API_KEY")

        if api_key is None:
            raise ValueError("Gradium API key is required. Set GRADIUM_API_KEY environment variable.")

        headers.update(
            {
                "x-api-key": api_key,
                "Content-Type": "application/json",
            }
        )

        return headers

    def get_error_class(self, error_message: str, status_code: int, headers: dict | Headers) -> BaseLLMException:
        return GradiumException(message=error_message, status_code=status_code, headers=headers)

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        base_url: Final = (api_base or get_secret_str("GRADIUM_API_BASE") or self.TTS_BASE_URL).rstrip("/")
        return f"{base_url}{self.TTS_ENDPOINT_PATH}"

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: str | None,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> TextToSpeechRequestData:
        params: Final = dict(optional_params) if optional_params else {}
        extra_body: Final = params.pop("extra_body", None)

        request_body: Final[dict[str, object]] = {
            "text": input,
            "voice_id": voice,
            "output_format": params.pop("output_format", self.DEFAULT_OUTPUT_FORMAT),
            # Without this Gradium answers a line-delimited JSON message
            # stream instead of audio bytes, which `/v1/audio/speech` callers
            # cannot read.
            "only_audio": True,
        }

        if model and model != "default":
            request_body["model_name"] = model

        for key, value in params.items():
            if value is not None:
                request_body[key] = value

        if isinstance(extra_body, dict):
            for key, value in extra_body.items():
                if value is not None:
                    request_body[key] = value

        return TextToSpeechRequestData(
            dict_body=request_body,
            headers={"Content-Type": "application/json"},
        )

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> "HttpxBinaryResponseContent":
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        # Gradium reports pre-stream failures as HTTP 500 with a plain-text
        # body ("error from server <code>: <reason>"), so the status alone
        # would surface as an opaque binary payload.
        if raw_response.status_code >= 400:
            raise GradiumException(
                message=raw_response.text,
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        return HttpxBinaryResponseContent(raw_response)
