from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Final
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from typing_extensions import override

import litellm
from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.core_helpers import drops_unsupported_params
from litellm.litellm_core_utils.url_utils import SSRFError, async_safe_get, safe_get
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.text_to_speech.transformation import (
    BaseTextToSpeechConfig,
    TextToSpeechRequestData,
)
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # factory stub leaves params untyped
    get_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # factory stub leaves params untyped
)
from litellm.llms.dashscope.common_utils import (
    DASHSCOPE_MULTIMODAL_GENERATION_PATH,
    DashScopeError,
    dashscope_api_root,
    is_aliyun_host,
    missing_dashscope_family_key_message,
    resolve_dashscope_family_api_key,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import HttpxBinaryResponseContent
from litellm.types.utils import Usage

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

DASHSCOPE_TTS_DEFAULT_API_BASE: Final = "https://dashscope-intl.aliyuncs.com"
_SPEECH_SYNTHESIZER_MODEL_PREFIX: Final = "qwen-audio-"
_TIMEOUT_PHASES: Final = TypeAdapter(dict[str, float | None])


@dataclass(frozen=True, slots=True)
class _TextToSpeechFamily:
    path: str
    response_formats: frozenset[str]
    sends_format: bool
    takes_instructions: bool


MULTIMODAL_GENERATION_FAMILY: Final = _TextToSpeechFamily(
    path=DASHSCOPE_MULTIMODAL_GENERATION_PATH,
    response_formats=frozenset({"wav"}),
    sends_format=False,
    takes_instructions=False,
)
SPEECH_SYNTHESIZER_FAMILY: Final = _TextToSpeechFamily(
    path="/api/v1/services/audio/tts/SpeechSynthesizer",
    response_formats=frozenset({"mp3", "wav", "opus", "pcm"}),
    sends_format=True,
    takes_instructions=True,
)


class _Audio(BaseModel):
    model_config = ConfigDict(frozen=True)
    url: str


class _Output(BaseModel):
    model_config = ConfigDict(frozen=True)
    audio: _Audio


class _Usage(BaseModel):
    model_config = ConfigDict(frozen=True)
    input_tokens: int | None = None
    output_tokens: int | None = None
    characters: int | None = None


class _SpeechResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    output: _Output | None = None
    usage: _Usage | None = None
    code: str | None = None
    message: str | None = None


def text_to_speech_family(model: str) -> _TextToSpeechFamily:
    if model.startswith(_SPEECH_SYNTHESIZER_MODEL_PREFIX):
        return SPEECH_SYNTHESIZER_FAMILY
    return MULTIMODAL_GENERATION_FAMILY


def _mapped_field(family: _TextToSpeechFamily, key: str, value: object) -> Mapping[str, object] | None:
    match key:
        case "response_format" if value in family.response_formats:
            return {"format": value} if family.sends_format else {}
        case "instructions" if family.takes_instructions:
            return {"instruction": value}
        case _:
            return None


def _unsupported(model: str, unsupported: tuple[str, ...]) -> UnsupportedParamsError:
    return UnsupportedParamsError(
        message=f"DashScope {model} does not support {', '.join(unsupported)}, set drop_params=True to ignore it",
        model=model,
        llm_provider="dashscope",
    )


def _speech_response(raw_response: httpx.Response) -> _SpeechResponse:
    try:
        return _SpeechResponse.model_validate_json(raw_response.content)
    except ValidationError as e:
        raise DashScopeError(
            status_code=502,
            message=f"Unexpected DashScope text-to-speech response: {e}",
            headers=raw_response.headers,
        ) from e


def _https_on_aliyun(url: str) -> str:
    parsed: Final = urlparse(url)
    return (
        parsed._replace(scheme="https").geturl() if parsed.scheme == "http" and is_aliyun_host(parsed.hostname) else url
    )


def _audio_url(raw_response: httpx.Response) -> tuple[str, _Usage | None]:
    parsed: Final = _speech_response(raw_response)
    if parsed.output is None:
        raise DashScopeError(
            status_code=max(raw_response.status_code, 400),
            message=parsed.message or parsed.code or raw_response.text,
            headers=raw_response.headers,
        )
    return _https_on_aliyun(parsed.output.audio.url), parsed.usage


def _request_timeout(raw_response: httpx.Response) -> httpx.Timeout | None:
    extension: Final = raw_response.request.extensions.get("timeout")
    if extension is None:
        return None
    phases: Final = _TIMEOUT_PHASES.validate_python(extension)
    return httpx.Timeout(
        connect=phases.get("connect"), read=phases.get("read"), write=phases.get("write"), pool=phases.get("pool")
    )


def _token_usage(usage: _Usage | None) -> Usage | None:
    if usage is None or usage.input_tokens is None or usage.output_tokens is None:
        return None
    return Usage(
        prompt_tokens=usage.input_tokens,
        completion_tokens=usage.output_tokens,
        total_tokens=usage.input_tokens + usage.output_tokens,
    )


def _refused_audio_url(error: SSRFError) -> DashScopeError:
    return DashScopeError(status_code=502, message=f"DashScope returned an audio URL that was refused: {error}")


def _binary_audio(audio_response: httpx.Response, usage: _Usage | None) -> HttpxBinaryResponseContent:
    if audio_response.status_code != 200:
        raise DashScopeError(
            status_code=502,
            message=f"DashScope audio download failed with HTTP {audio_response.status_code}",
            headers=audio_response.headers,
        )
    binary: Final = HttpxBinaryResponseContent(audio_response)
    binary.usage = _token_usage(usage)
    if usage is not None and usage.characters is not None:
        binary.hidden_params = {**binary.hidden_params, "billed_characters": usage.characters}
    return binary


def _shared_async_client() -> AsyncHTTPHandler:
    return get_async_httpx_client(llm_provider=litellm.LlmProviders.DASHSCOPE)


class DashScopeTextToSpeechConfig(BaseTextToSpeechConfig):
    def __init__(
        self,
        sync_client: Callable[[], HTTPHandler] = get_httpx_client,
        async_client: Callable[[], AsyncHTTPHandler] = _shared_async_client,
    ) -> None:
        super().__init__()
        self._sync_client: Final = sync_client
        self._async_client: Final = async_client

    @override
    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: mirrors override contract
        family: Final = text_to_speech_family(model)
        return ["voice", "response_format", *(("instructions",) if family.takes_instructions else ())]

    @override
    def map_openai_params(
        self,
        model: str,
        optional_params: dict[str, object],  # mutable-ok: mirrors override contract
        voice: str | dict[str, object] | None = None,  # mutable-ok: mirrors override contract
        drop_params: bool = False,
        kwargs: Mapping[str, object] | None = None,
    ) -> tuple[str | None, dict[str, object]]:  # mutable-ok: mirrors override contract
        family: Final = text_to_speech_family(model)
        fields: Final = tuple((key, value, _mapped_field(family, key, value)) for key, value in optional_params.items())
        unsupported: Final = (
            *(f"{key}={value!r}" for key, value, field in fields if field is None),
            *(("voice as an object",) if isinstance(voice, dict) else ()),
        )
        if unsupported and not (drop_params or drops_unsupported_params((kwargs or {}).get("drop_params"))):
            raise _unsupported(model, unsupported)
        supported: Final = tuple(field for _, _, field in fields if field is not None)
        return voice if isinstance(voice, str) else None, dict(chain.from_iterable(f.items() for f in supported))

    @override
    def validate_environment(
        self,
        headers: dict[str, str],  # mutable-ok: mirrors override contract
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: mirrors override contract
        resolved_api_key: Final = resolve_dashscope_family_api_key("dashscope", api_key)
        if resolved_api_key is None:
            raise ValueError(missing_dashscope_family_key_message("dashscope"))
        return {**headers, "Authorization": f"Bearer {resolved_api_key}", "Content-Type": "application/json"}

    @override
    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict[str, object],  # mutable-ok: mirrors override contract
    ) -> str:
        configured: Final = api_base or get_secret_str("DASHSCOPE_API_BASE")
        root: Final = dashscope_api_root(configured) if configured else DASHSCOPE_TTS_DEFAULT_API_BASE
        return f"{root}{text_to_speech_family(model).path}"

    @override
    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: str | None,
        optional_params: dict[str, object],  # mutable-ok: mirrors override contract
        litellm_params: dict[str, object],  # mutable-ok: mirrors override contract
        headers: dict[str, str],  # mutable-ok: mirrors override contract
    ) -> TextToSpeechRequestData:
        voice_field: Final = {"voice": voice} if voice is not None else {}
        return {"dict_body": {"model": model, "input": {"text": input, **voice_field, **optional_params}}}

    @override
    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> HttpxBinaryResponseContent:
        audio_url, usage = _audio_url(raw_response)
        try:
            audio_response: Final = safe_get(self._sync_client(), audio_url, timeout=_request_timeout(raw_response))
        except SSRFError as e:
            raise _refused_audio_url(e) from e
        return _binary_audio(audio_response, usage)

    @override
    async def async_transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> HttpxBinaryResponseContent:
        audio_url, usage = _audio_url(raw_response)
        try:
            audio_response: Final = await async_safe_get(
                self._async_client(), audio_url, timeout=_request_timeout(raw_response)
            )
        except SSRFError as e:
            raise _refused_audio_url(e) from e
        return _binary_audio(audio_response, usage)

    @override
    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str] | httpx.Headers,  # mutable-ok: mirrors override contract
    ) -> BaseLLMException:
        return DashScopeError(status_code=status_code, message=error_message, headers=httpx.Headers(headers))
