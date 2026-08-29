"""
Gandr Text-to-Speech transformation

Maps OpenAI TTS spec to the Gandr /v1/audio/speech OpenAI-compatible API.

Gandr is a spoken-audio inference provider. Its HTTP lane accepts the OpenAI
Audio Speech request shape and returns raw audio bytes, so the same
`sample` that works against `tts.gandr.ai/v1/audio/speech` with the official
OpenAI SDK works here unchanged.
"""

from typing import TYPE_CHECKING, Any, ClassVar, Final  # noqa: TID251  # Any matches the base interface

import httpx
import litellm
from httpx import Headers
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.text_to_speech.transformation import (
    BaseTextToSpeechConfig,
    TextToSpeechRequestData,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import all_litellm_params

from ..common_utils import GandrException

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import HttpxBinaryResponseContent
else:
    LiteLLMLoggingObj = Any
    HttpxBinaryResponseContent = Any


class GandrTextToSpeechConfig(BaseTextToSpeechConfig):
    """
    Configuration for Gandr Text-to-Speech.

    The Gandr endpoint is OpenAI-compatible (`POST /v1/audio/speech`), so the
    request body is passed through with the OpenAI field names (`voice`,
    `response_format`, `speed`) intact. Only authentication differs: calls are
    authorized with an `Authorization: Bearer` header using a `gnd_` token.

    Reference: https://gandr.ai/docs
    """

    TTS_BASE_URL = "https://tts.gandr.ai/v1"
    TTS_ENDPOINT_PATH = "/audio/speech"
    DEFAULT_OUTPUT_FORMAT = "wav"
    SUPPORTED_MODES: ClassVar[list[str]] = ["audio_speech"]

    #: Gandr's speed range. The engine clamps speed to 0.6-1.5.
    SUPPORTED_SPEED_RANGE = (0.6, 1.5)

    def get_supported_openai_params(self, model: str) -> list:
        """
        Gandr TTS supports these OpenAI parameters
        """
        return ["voice", "response_format", "speed"]

    def map_openai_params(
        self,
        model: str,
        optional_params: dict,
        voice: str | dict | None = None,
        drop_params: bool = False,
        kwargs: dict[str, Any] | None = None,
    ) -> tuple[str | None, dict]:
        """
        Map OpenAI parameters to Gandr TTS parameters.

        Gandr accepts the OpenAI request body verbatim, so this is effectively
        a passthrough. `voice` is returned as-is and the remaining params are
        preserved on the body under their OpenAI names, which means an
        unmodified client always gets audio back.
        """
        params: Final = dict(optional_params) if optional_params else {}
        passthrough_kwargs: Final = dict(kwargs) if kwargs is not None else {}

        mapped_voice: str | None = None
        if isinstance(voice, str) and voice.strip():
            mapped_voice = voice
        elif isinstance(voice, dict):
            for key in ("voice_id", "id", "name"):
                candidate = voice.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    mapped_voice = candidate
                    break

        if mapped_voice is None:
            voice_override: Final = params.pop("voice_id", None)
            if isinstance(voice_override, str) and voice_override.strip():
                mapped_voice = voice_override

        if mapped_voice is None:
            raise ValueError("Gandr voice is required. Pass `voice` when calling `litellm.speech()`.")

        # `response_format` is passed through under its OpenAI name; the
        # endpoint serves mp3 (its default), wav and pcm.
        response_format: Final = params.get("response_format")
        if response_format is None:
            params["response_format"] = self.DEFAULT_OUTPUT_FORMAT

        # `speed` passes through too. Gandr clamps to its own audible range
        # server-side, so out-of-range values degrade gracefully.
        speed: Final = params.get("speed")
        if speed is not None:
            try:
                float(speed)
            except (TypeError, ValueError):
                params.pop("speed", None)

        mapped_params: Final[dict[str, Any]] = {k: v for k, v in params.items() if v is not None}

        reserved_kwarg_keys: Final = set(all_litellm_params) | {
            "voice",
            "model",
            "response_format",
            "output_format",
            "extra_body",
            "user",
        }
        for key in list(passthrough_kwargs.keys()):
            if key in reserved_kwarg_keys:
                continue
            value = passthrough_kwargs[key]
            if value is None:
                continue
            mapped_params[key] = value

        return mapped_voice, mapped_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        """
        Validate the Gandr environment and set up authentication headers.

        A caller-supplied api_base is honored when constructing the request
        URL, so falling back to a server-managed key while the caller controls
        the host would leak that key. The provider default and the operator's
        own GANDR_API_BASE override are the only trusted destinations for a
        server-managed key; anything else requires an explicit api_key.
        """
        server_api_key: Final = litellm.api_key or get_secret_str("GANDR_API_KEY")

        if api_key is None and api_base is not None and server_api_key is not None:
            trusted_base: Final = (get_secret_str("GANDR_API_BASE") or self.TTS_BASE_URL).rstrip("/")
            if api_base.rstrip("/") != trusted_base:
                raise ValueError(
                    "Refusing to send the server-configured GANDR_API_KEY to the "
                    f"caller-supplied api_base '{api_base}'. Pass an explicit "
                    "api_key when overriding api_base."
                )

        api_key = api_key or server_api_key

        if api_key is None:
            raise ValueError(
                "Gandr API key is required. Set GANDR_API_KEY environment variable "
                "or pass `api_key` to `litellm.speech()`."
            )

        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

        return headers

    def get_error_class(self, error_message: str, status_code: int, headers: dict | Headers) -> BaseLLMException:
        return GandrException(message=error_message, status_code=status_code, headers=headers)

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: str | None,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> TextToSpeechRequestData:
        """
        Build the Gandr TTS request payload. The body is OpenAI-shaped
        (`input`, `model`, `voice`, `response_format`, `speed`); the repo's
        HTTP handler attaches the required `Authorization: Bearer` header from
        `validate_environment`.
        """
        params: Final = dict(optional_params) if optional_params else {}
        extra_body: Final = params.pop("extra_body", None)

        request_body: Final[dict[str, Any]] = {
            "input": input,
            "model": model,
            "voice": voice,
            "response_format": params.get("response_format", self.DEFAULT_OUTPUT_FORMAT),
            "speed": params.get("speed", 1.0),
        }

        for key, value in params.items():
            if value is None or key in ("response_format", "speed", "voice"):
                continue
            request_body[key] = value

        if isinstance(extra_body, dict):
            for key, value in extra_body.items():
                if value is None:
                    continue
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
        """
        Wrap Gandr binary audio response.
        """
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        return HttpxBinaryResponseContent(raw_response)

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        """
        Construct the Gandr endpoint URL.
        """
        base_url = api_base or get_secret_str("GANDR_API_BASE") or self.TTS_BASE_URL
        base_url = base_url.rstrip("/")

        return f"{base_url}{self.TTS_ENDPOINT_PATH}"
