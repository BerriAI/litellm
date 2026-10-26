from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal
from uuid import UUID, uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.text_to_speech.transformation import BaseTextToSpeechConfig, TextToSpeechRequestData
from litellm.secret_managers.main import get_secret_str

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import HttpxBinaryResponseContent


class FlowSpeechException(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: dict[str, str] | httpx.Headers | None = None,  # mutable-ok: upstream exception interface
    ) -> None:
        super().__init__(  # pyright: ignore[reportUnknownMemberType]  # upstream exception has untyped dictionary fields
            status_code=status_code, message=message, headers=headers
        )


class FlowSpeechSegment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    speaker: str = Field(min_length=1, max_length=100)
    text: str = Field(min_length=1, max_length=100_000)
    voiceName: str = Field(min_length=1, max_length=100)


class FlowSpeechResult(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True)
    url: str
    mimeType: Literal["audio/wav"]
    sampleRate: Literal[24000]
    channels: Literal[1]
    bitsPerSample: Literal[16]


class FlowSpeechError(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True)
    code: str
    message: str
    retryable: bool = False


class FlowSpeechUsage(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True)
    billableCharacters: int = Field(ge=0)
    reservedCredits: int = Field(ge=0)
    chargedCredits: int = Field(ge=0)


class FlowSpeechJob(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True)
    id: str
    requestId: str
    usage: FlowSpeechUsage
    status: Literal["queued", "running", "merging", "settling", "success", "failed", "cancelled"]
    result: FlowSpeechResult | None = None
    resultDeleted: bool = False
    error: FlowSpeechError | None = None

    @field_validator("id", "requestId")
    @classmethod
    def canonical_identifier(cls, value: str) -> str:
        return validate_uuid(value)


class FlowSpeechErrorResponse(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True)
    error: FlowSpeechError
    traceId: str | None = None


def validate_uuid(value: object) -> str:
    text: Final = TypeAdapter(str).validate_python(value, strict=True)
    try:
        parsed: Final = str(UUID(text))
    except ValueError as exc:
        raise ValueError("FlowSpeech requires a UUID") from exc
    if parsed != text.lower():
        raise ValueError("FlowSpeech requires a hyphenated UUID")
    return parsed


def validate_url(value: str) -> httpx.URL:
    try:
        url: Final = httpx.URL(value)
    except httpx.InvalidURL as exc:
        raise ValueError("FlowSpeech requires a valid HTTPS URL") from exc
    if url.scheme != "https" or not url.host or url.userinfo or url.fragment:
        raise ValueError("FlowSpeech requires an HTTPS URL without credentials or fragment")
    return url


class FlowSpeechTextToSpeechConfig(BaseTextToSpeechConfig):
    TTS_BASE_URL = "https://flowspeech.io"
    TTS_ENDPOINT_PATH = "/api/v1/tts/jobs"
    DEFAULT_VOICE = "Kore"

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: base provider interface
        return ["voice", "response_format"]

    def map_openai_params(
        self,
        model: str,
        optional_params: Mapping[str, object],
        voice: str | Mapping[str, object] | None = None,
        drop_params: bool = False,
        kwargs: Mapping[str, object] | None = None,
    ) -> tuple[str | None, dict[str, object]]:  # mutable-ok: base provider interface
        if model != "flowspeech-tts":
            raise ValueError("FlowSpeech only supports the flowspeech/flowspeech-tts route")
        if any(key != "response_format" for key in optional_params):
            raise ValueError("FlowSpeech does not support instructions or speed")
        if optional_params.get("response_format") not in (None, "wav"):
            raise ValueError("FlowSpeech only supports response_format='wav'")
        supplied: Final = kwargs or MappingProxyType({})
        if any(
            key in supplied
            for key in ("source", "format", "speakers", "originalText", "prompt", "stream", "extra_body")
        ):
            raise ValueError("Unsupported FlowSpeech option; use voice or segments")
        if "segments" in supplied and voice is not None:
            raise ValueError("Use segment voiceName fields instead of voice for multi-speaker input")
        return self._resolve_voice(voice), {key: supplied[key] for key in ("segments", "request_id") if key in supplied}

    def _resolve_voice(self, voice: str | Mapping[str, object] | None) -> str:
        if voice is None:
            return self.DEFAULT_VOICE
        if isinstance(voice, str) and voice.strip():
            return voice.strip()
        if isinstance(voice, Mapping):
            candidate: Final = next(
                (voice[key] for key in ("voice_name", "voiceName", "id", "name") if key in voice), None
            )
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        raise ValueError("FlowSpeech voice must be a nonempty voice name")

    def validate_environment(
        self, headers: Mapping[str, str], model: str, api_key: str | None = None, api_base: str | None = None
    ) -> dict[str, str]:  # mutable-ok: base provider interface
        key: Final = api_key if api_key is not None else (litellm.api_key or get_secret_str("FLOWSPEECH_API_KEY"))
        if not key or not key.isascii() or any(char.isspace() or ord(char) < 33 or ord(char) == 127 for char in key):
            raise ValueError("FlowSpeech API key is required and must be a nonempty token. Set FLOWSPEECH_API_KEY.")
        if any(name.lower() in ("cookie", "proxy-authorization", "host", "content-length") for name in headers):
            raise ValueError("FlowSpeech does not accept cookie, proxy credential, host or content-length overrides")
        return {
            **{name: value for name, value in headers.items() if name.lower() not in ("authorization", "content-type")},
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str] | httpx.Headers,  # mutable-ok: upstream exception interface
    ) -> BaseLLMException:
        return FlowSpeechException(status_code, error_message, headers)

    def _input_body(
        self,
        input: str,
        voice: str | None,
        optional_params: Mapping[str, object],
    ) -> dict[str, object]:  # mutable-ok: JSON body required by base interface
        if "segments" in optional_params:
            if input:
                raise ValueError("Use input='' with FlowSpeech segments")
            segments: Final = TypeAdapter(list[FlowSpeechSegment]).validate_python(optional_params["segments"])
            if not 1 <= len(segments) <= 1000 or any(
                not part.speaker.strip() or not part.text.strip() or not part.voiceName.strip() for part in segments
            ):
                raise ValueError("FlowSpeech requires 1 to 1000 nonempty speaker turns")
            return {
                "mode": "multi",
                "segments": [part.model_dump() for part in segments],
            }
        if not input.strip():
            raise ValueError("FlowSpeech input must contain text")
        return {
            "mode": "single",
            "text": input,
            "voiceName": self._resolve_voice(voice),
        }

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: str | None,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, str],
    ) -> TextToSpeechRequestData:
        if frozenset(optional_params) - frozenset(("segments", "request_id")):
            raise ValueError("Unsupported FlowSpeech request options")
        body: Final = self._input_body(input, voice, optional_params)
        header_id: Final = next((value for key, value in headers.items() if key.lower() == "idempotency-key"), None)
        supplied_id: Final = optional_params.get("request_id", header_id)
        request_id: Final = validate_uuid(supplied_id) if supplied_id is not None else str(uuid4())
        if header_id is not None and validate_uuid(header_id) != request_id:
            raise ValueError("FlowSpeech request_id and Idempotency-Key must agree")
        return TextToSpeechRequestData(
            dict_body={"input": body},
            headers={"Idempotency-Key": request_id},
        )

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> "HttpxBinaryResponseContent":
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        if raw_response.status_code != 200 or raw_response.headers.get("content-type") != "audio/wav":
            raise FlowSpeechException(502, "FlowSpeech requires completed, downloaded WAV audio")
        return HttpxBinaryResponseContent(raw_response)

    def get_complete_url(self, model: str, api_base: str | None, litellm_params: Mapping[str, object]) -> str:
        base: Final = api_base if api_base is not None else (get_secret_str("FLOWSPEECH_API_BASE") or self.TTS_BASE_URL)
        url: Final = validate_url(base)
        if url.query or url.path.rstrip("/") not in ("", self.TTS_ENDPOINT_PATH):
            raise ValueError("FlowSpeech api_base must be an origin or the /api/v1/tts/jobs endpoint")
        return str(url.copy_with(path=self.TTS_ENDPOINT_PATH))
