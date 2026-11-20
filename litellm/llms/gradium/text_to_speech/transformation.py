from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar, Final, TypeAlias

import httpx
from httpx import Headers
from pydantic import BaseModel, ConfigDict

import litellm
from litellm.litellm_core_utils.core_helpers import drops_unsupported_params
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


class _GradiumSpeechRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    output_format: str


class GradiumTextToSpeechConfig(BaseTextToSpeechConfig):
    TTS_ENDPOINT_PATH = "/api/speech/tts"
    DEFAULT_OUTPUT_FORMAT = "wav"

    VOICE_MAPPINGS: ClassVar[Mapping[str, str]] = {
        "alloy": "YTpq7expH9539ERJ",
        "echo": "s0PhgjzOTRD5wo5L",
        "fable": "1VAVLmmbQFDw7TMn",
        "onyx": "QkmUhBH4hIV2_BkY",
        "nova": "w9V1722uEmTkWqnR",
        "shimmer": "BbLb4TxdlrldgpHI",
        "ash": "_6Aslh2DxfmnRLmP",
        "ballad": "yk2FCQroF3f6O5Zg",
        "coral": "vjFyp2BcCyTlUmYb",
        "sage": "7Qu2QSs0IjinH2yU",
        "verse": "dME3IWyZBvmh1n1q",
        "marin": "NbpkqMVS3CJeq2j8",
        "cedar": "POBHtemksfWQbng0",
    }

    FORMAT_MAPPINGS: ClassVar[Mapping[str, str]] = {
        "wav": "wav",
        "pcm": "pcm_24000",
        "opus": "opus",
        **{
            native: native
            for native in (
                "ulaw_8000",
                "mulaw_8000",
                "alaw_8000",
                "pcm_8000",
                "pcm_16000",
                "pcm_22050",
                "pcm_24000",
                "pcm_44100",
                "pcm_48000",
            )
        },
    }

    RAW_AUDIO_MEDIA_TYPES: ClassVar[Mapping[str, str]] = {
        "ulaw_8000": "audio/basic",
        "mulaw_8000": "audio/basic",
        "alaw_8000": "audio/x-alaw-basic",
    }

    UNSUPPORTED_OPENAI_PARAMS: ClassVar[frozenset[str]] = frozenset({"speed", "instructions"})

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: BaseTextToSpeechConfig declares list
        return ["voice", "response_format"]

    def _resolve_voice_id(
        self,
        model: str,
        voice: str | dict[str, object] | None,  # mutable-ok: mirrors the dict|str the base class passes through
        params: Mapping[str, object],
    ) -> str:
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
            raise litellm.BadRequestError(
                message="Gradium voice_id is required. Pass `voice` when calling `litellm.speech()`.",
                model=model,
                llm_provider="gradium",
            )

        return self.VOICE_MAPPINGS.get(candidate.lower(), candidate)

    def map_openai_params(
        self,
        model: str,
        optional_params: dict[str, object],  # mutable-ok: signature fixed by BaseTextToSpeechConfig
        voice: str | dict[str, object] | None = None,  # mutable-ok: signature fixed by BaseTextToSpeechConfig
        drop_params: bool = False,
        kwargs: dict[str, object] | None = None,  # mutable-ok: signature fixed by BaseTextToSpeechConfig
    ) -> tuple[str | None, dict[str, object]]:  # mutable-ok: callers mutate the returned params downstream
        mapped_voice: Final = self._resolve_voice_id(model, voice, optional_params)
        response_format: Final = optional_params.get("response_format")
        output_format: Final = self.FORMAT_MAPPINGS.get(response_format) if isinstance(response_format, str) else None
        unsupported: Final = tuple(
            f"{key}={value!r}"
            for key, value in optional_params.items()
            if value is not None and key in self.UNSUPPORTED_OPENAI_PARAMS
        ) + (
            (f"response_format={response_format!r}",)
            if isinstance(response_format, str) and output_format is None
            else ()
        )

        if unsupported and not (drop_params or drops_unsupported_params((kwargs or {}).get("drop_params"))):
            raise litellm.UnsupportedParamsError(
                status_code=400,
                message=(
                    f"Gradium TTS does not support {', '.join(unsupported)}. "
                    f"Supported response_format values: {sorted(self.FORMAT_MAPPINGS)}. "
                    "Set drop_params to true to drop them."
                ),
            )

        return mapped_voice, ({"output_format": output_format} if output_format is not None else {})

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

        root: Final = api_base.rstrip("/").removesuffix("/api")
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
        model_name: Final = {"model_name": model} if model and model != "default" else {}

        request_body: Final[dict[str, object]] = {  # mutable-ok: TextToSpeechRequestData.dict_body is dict
            "text": input,
            "voice_id": voice,
            "output_format": optional_params.get("output_format", self.DEFAULT_OUTPUT_FORMAT),
            "only_audio": True,
            **model_name,
        }

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

        if raw_response.status_code >= 400:
            raise GradiumException(
                message=raw_response.text,
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        sent: Final = _GradiumSpeechRequest.model_validate_json(raw_response.request.content)
        if not any(value.startswith("audio/") for value in raw_response.headers.get_list("content-type")):
            media_type: Final = self.RAW_AUDIO_MEDIA_TYPES.get(sent.output_format, "audio/pcm")
            raw_response.headers["content-type"] = media_type  # rebind-ok: httpx exposes headers only as mutable

        speech: Final = HttpxBinaryResponseContent(raw_response)
        speech.hidden_params = {**speech.hidden_params, "billed_characters": len(sent.text)}
        return speech
