"""
Gradium Text-to-Speech transformation

Maps OpenAI TTS spec to Gradium's REST TTS API.

Reference: https://docs.gradium.ai/api-reference/endpoint/tts-post
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar, Final, TypeAlias

import httpx
from httpx import Headers

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.text_to_speech.transformation import (
    BaseTextToSpeechConfig,
    TextToSpeechRequestData,
)

from ..common_utils import GradiumException

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import HttpxBinaryResponseContent
else:
    LiteLLMLoggingObj: TypeAlias = object
    HttpxBinaryResponseContent: TypeAlias = object


class GradiumTextToSpeechConfig(BaseTextToSpeechConfig):
    """
    Configuration for Gradium Text-to-Speech

    Reference: https://docs.gradium.ai/api-reference/endpoint/tts-post
    """

    TTS_ENDPOINT_PATH = "/api/post/speech/tts"
    DEFAULT_OUTPUT_FORMAT = "wav"

    # OpenAI canonical voice names → Gradium library voice IDs. Anything not
    # listed is forwarded verbatim, so callers can pass raw Gradium voice IDs.
    VOICE_MAPPINGS: ClassVar[Mapping[str, str]] = {
        "alloy": "YTpq7expH9539ERJ",
        "echo": "s0PhgjzOTRD5wo5L",
        "fable": "1VAVLmmbQFDw7TMn",
        "onyx": "QkmUhBH4hIV2_BkY",
        "nova": "w9V1722uEmTkWqnR",
        "shimmer": "BbLb4TxdlrldgpHI",
    }

    # Gradium serves raw PCM and Ogg-wrapped Opus, plus WAV. It has no mp3,
    # aac or flac encoder, so those OpenAI formats have no mapping.
    FORMAT_MAPPINGS: ClassVar[Mapping[str, str]] = {
        "wav": "wav",
        "pcm": "pcm_24000",
        "opus": "opus",
    }

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: BaseTextToSpeechConfig declares list
        return ["voice", "response_format"]

    def _resolve_voice_id(
        self,
        voice: str | dict[str, object] | None,  # mutable-ok: mirrors the dict|str the base class passes through
        params: Mapping[str, object],
    ) -> str:
        """Normalize the caller's `voice` into a Gradium voice_id."""
        sources: Final = (
            (voice,)
            if isinstance(voice, str)
            else ((voice.get("voice_id"), voice.get("id"), voice.get("name")) if isinstance(voice, dict) else ())
        )
        candidate: Final = next(
            (value.strip() for value in (*sources, params.get("voice_id")) if isinstance(value, str) and value.strip()),
            None,
        )

        if candidate is None:
            raise ValueError("Gradium voice_id is required. Pass `voice` when calling `litellm.speech()`.")

        return self.VOICE_MAPPINGS.get(candidate.lower(), candidate)

    def map_openai_params(
        self,
        model: str,
        optional_params: dict[str, object],  # mutable-ok: signature fixed by BaseTextToSpeechConfig
        voice: str | dict[str, object] | None = None,  # mutable-ok: signature fixed by BaseTextToSpeechConfig
        drop_params: bool = False,
        kwargs: dict[str, object] | None = None,  # mutable-ok: signature fixed by BaseTextToSpeechConfig
    ) -> tuple[str | None, dict[str, object]]:  # mutable-ok: callers mutate the returned params downstream
        params: Final = dict(optional_params) if optional_params else {}
        mapped_voice: Final = self._resolve_voice_id(voice, params)

        response_format: Final = params.pop("response_format", None)
        output_format: Final = self.FORMAT_MAPPINGS.get(response_format) if isinstance(response_format, str) else None

        if isinstance(response_format, str) and output_format is None and not (drop_params or litellm.drop_params):
            raise litellm.UnsupportedParamsError(
                status_code=400,
                message=(
                    f"Gradium TTS does not support response_format={response_format!r}. "
                    f"Supported: {sorted(self.FORMAT_MAPPINGS)}. "
                    "Set `litellm.drop_params = True` to fall back to the default format."
                ),
            )

        # Gradium synthesizes at the voice's native rate; there is no speed
        # control and no instructions field on the REST endpoint.
        passthrough: Final = {
            key: value
            for key, value in params.items()
            if value is not None and key not in ("speed", "instructions", "voice_id")
        }
        mapped_params: Final[dict[str, object]] = (  # mutable-ok: the declared return type is dict
            {**passthrough, "output_format": output_format} if output_format is not None else dict(passthrough)
        )

        return mapped_voice, mapped_params

    def validate_environment(
        self,
        headers: dict[str, str],  # mutable-ok: signature fixed by BaseTextToSpeechConfig
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: signature fixed by BaseTextToSpeechConfig
        if not api_key:
            raise ValueError("Gradium API key is required. Set `api_key` on the model.")

        return {
            **headers,
            "x-api-key": api_key,
            "Content-Type": "application/json",
        }

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str] | Headers,  # mutable-ok: signature fixed by BaseTextToSpeechConfig
    ) -> BaseLLMException:
        return GradiumException(message=error_message, status_code=status_code, headers=headers)

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict[str, object],  # mutable-ok: signature fixed by BaseTextToSpeechConfig
    ) -> str:
        if not api_base:
            raise ValueError(
                "Gradium API base is required. Set `api_base` on the model, e.g. https://eu.api.gradium.ai."
            )

        base_url: Final = api_base.rstrip("/")
        # Gradium documents its regional bases with the /api suffix already on
        # them (https://eu.api.gradium.ai/api), so appending the full endpoint
        # path to one would request /api/api/post/speech/tts.
        root: Final = base_url.removesuffix("/api")
        return f"{root}{self.TTS_ENDPOINT_PATH}"

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: str | None,
        optional_params: dict[str, object],  # mutable-ok: signature fixed by BaseTextToSpeechConfig
        litellm_params: dict[str, object],  # mutable-ok: signature fixed by BaseTextToSpeechConfig
        headers: dict[str, str],  # mutable-ok: signature fixed by BaseTextToSpeechConfig
    ) -> TextToSpeechRequestData:
        params: Final = dict(optional_params) if optional_params else {}
        extra_body: Final = params.pop("extra_body", None)
        passthrough: Final = {key: value for key, value in params.items() if value is not None}
        model_name: Final = {"model_name": model} if model and model != "default" else {}

        request_body: Final[dict[str, object]] = {  # mutable-ok: TextToSpeechRequestData.dict_body is dict
            "text": input,
            "voice_id": voice,
            "output_format": params.get("output_format", self.DEFAULT_OUTPUT_FORMAT),
            # Without this Gradium answers a line-delimited JSON message
            # stream instead of audio bytes, which `/v1/audio/speech` callers
            # cannot read.
            "only_audio": True,
            **model_name,
            **passthrough,
        }

        # Written in a loop rather than a comprehension: `extra_body` is typed
        # `object` here, and isinstance only narrows it to dict[Unknown, Unknown].
        if isinstance(extra_body, dict):
            for key, value in extra_body.items():  # pyright: ignore[reportUnknownVariableType]  # isinstance narrows a bare dict only to dict[Unknown, Unknown]
                if value is not None:
                    request_body[str(key)] = value  # pyright: ignore[reportUnknownArgumentType]  # same bare-dict narrowing

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
