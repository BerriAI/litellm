import base64
import binascii
from typing import TYPE_CHECKING, Final

import httpx
from httpx import Headers
from pydantic import BaseModel

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.text_to_speech.transformation import (
    BaseTextToSpeechConfig,
    TextToSpeechRequestData,
)
from litellm.secret_managers.main import get_secret_str

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import HttpxBinaryResponseContent


class FlowSpeechException(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: dict[str, str] | Headers | None = None,  # mutable-ok: exception contract accepts dict headers
    ) -> None:
        super().__init__(  # pyright: ignore[reportUnknownMemberType]  # upstream exception keeps untyped dict fields
            status_code=status_code,
            message=message,
            headers=headers,
        )


class FlowSpeechAudioData(BaseModel):
    mimeType: str
    audioBase64: str


class FlowSpeechResponse(BaseModel):
    code: int
    message: str | None = None
    data: FlowSpeechAudioData | None = None


class FlowSpeechTextToSpeechConfig(BaseTextToSpeechConfig):
    TTS_BASE_URL = "https://flowspeech.io"
    TTS_ENDPOINT_PATH = "/api/ai/text-to-speech"
    DEFAULT_VOICE = "Kore"

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: base provider interface returns list
        return ["voice", "instructions"]  # mutable-ok: base provider interface returns list

    def map_openai_params(
        self,
        model: str,
        optional_params: dict[str, object],  # mutable-ok: base provider interface requires dict
        voice: str | dict[str, object] | None = None,  # mutable-ok: base provider interface accepts dict voices
        drop_params: bool = False,
        kwargs: dict[str, object] | None = None,  # mutable-ok: base provider interface requires dict
    ) -> tuple[str | None, dict[str, object]]:  # mutable-ok: base provider interface returns dict params
        params: Final[dict[str, object]] = (  # mutable-ok: isolated request copy
            dict(optional_params) if optional_params else {}  # mutable-ok: isolated request copy
        )
        mapped_voice: Final = self._resolve_voice(voice)
        instructions: Final = params.get("instructions")
        mapped_params: Final[dict[str, object]] = (  # mutable-ok: base provider interface returns dict params
            {"prompt": instructions}  # mutable-ok: base provider interface returns dict params
            if isinstance(instructions, str) and instructions.strip()
            else {}  # mutable-ok: base provider interface returns dict params
        )
        return mapped_voice, mapped_params

    def _resolve_voice(
        self,
        voice: str | dict[str, object] | None,  # mutable-ok: base provider interface accepts dict voices
    ) -> str:
        if isinstance(voice, str) and voice.strip():
            return voice.strip()
        if isinstance(voice, dict):
            candidate: Final = next(
                (
                    value
                    for key in ("voice_name", "voiceName", "id", "name")
                    if isinstance((value := voice.get(key)), str) and value.strip()
                ),
                None,
            )
            if isinstance(candidate, str):
                return candidate.strip()
        return self.DEFAULT_VOICE

    def validate_environment(
        self,
        headers: dict[str, str],  # mutable-ok: base provider interface requires dict headers
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: base provider interface returns dict headers
        resolved_api_key: Final = api_key or litellm.api_key or get_secret_str("FLOWSPEECH_API_KEY")
        if resolved_api_key is None:
            raise ValueError("FlowSpeech API key is required. Set FLOWSPEECH_API_KEY environment variable.")
        return {  # mutable-ok: base provider interface returns dict headers
            **headers,
            "Authorization": f"Bearer {resolved_api_key}",
            "Content-Type": "application/json",
        }

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str] | Headers,  # mutable-ok: exception contract accepts dict headers
    ) -> BaseLLMException:
        return FlowSpeechException(message=error_message, status_code=status_code, headers=headers)

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: str | None,
        optional_params: dict[str, object],  # mutable-ok: base provider interface requires dict
        litellm_params: dict[str, object],  # mutable-ok: base provider interface requires dict
        headers: dict[str, str],  # mutable-ok: base provider interface requires dict
    ) -> TextToSpeechRequestData:
        prompt: Final = optional_params.get("prompt")
        prompt_data: Final[dict[str, str]] = (  # mutable-ok: JSON request body requires a dict
            {"prompt": prompt}  # mutable-ok: JSON request body requires a dict
            if isinstance(prompt, str) and prompt.strip()
            else {}  # mutable-ok: JSON request body requires a dict
        )
        request_body: Final[dict[str, object]] = {  # mutable-ok: JSON request body requires a dict
            "text": input,
            "originalText": input,
            "speakers": [  # mutable-ok: FlowSpeech JSON schema requires a speakers array
                {"voiceName": voice or self.DEFAULT_VOICE}  # mutable-ok: FlowSpeech JSON schema requires an object
            ],
            **prompt_data,
        }
        return TextToSpeechRequestData(
            dict_body=request_body,
            headers={"Content-Type": "application/json"},  # mutable-ok: base request type requires dict headers
        )

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> "HttpxBinaryResponseContent":
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        try:
            payload: Final = FlowSpeechResponse.model_validate(raw_response.json())
        except ValueError as exc:
            raise FlowSpeechException(
                status_code=raw_response.status_code,
                message="FlowSpeech API returned an invalid JSON response",
                headers=raw_response.headers,
            ) from exc

        if payload.code != 0 or payload.data is None or not payload.data.audioBase64:
            raise FlowSpeechException(
                status_code=raw_response.status_code,
                message=payload.message or "FlowSpeech API response did not include audio data",
                headers=raw_response.headers,
            )

        try:
            audio_bytes: Final = base64.b64decode(payload.data.audioBase64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise FlowSpeechException(
                status_code=raw_response.status_code,
                message="FlowSpeech API returned invalid base64 audio data",
                headers=raw_response.headers,
            ) from exc

        preserved_headers: Final = {  # mutable-ok: httpx.Response requires materialized headers
            key: value
            for key, value in raw_response.headers.items()
            if key.lower() not in frozenset({"content-encoding", "content-length", "content-type", "transfer-encoding"})
        }
        response_headers: Final = {  # mutable-ok: httpx.Response requires materialized headers
            **preserved_headers,
            "content-length": str(len(audio_bytes)),
            "content-type": payload.data.mimeType,
        }
        binary_response: Final = httpx.Response(
            status_code=200,
            headers=response_headers,
            content=audio_bytes,
            request=raw_response.request,
        )
        return HttpxBinaryResponseContent(binary_response)

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict[str, object],  # mutable-ok: base provider interface requires dict
    ) -> str:
        base_url: Final = api_base or get_secret_str("FLOWSPEECH_API_BASE") or self.TTS_BASE_URL
        normalized_base_url: Final = base_url.rstrip("/")
        if normalized_base_url.endswith(self.TTS_ENDPOINT_PATH):
            return normalized_base_url
        return f"{normalized_base_url}{self.TTS_ENDPOINT_PATH}"
