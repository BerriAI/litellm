from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Final

import httpx

from litellm.exceptions import AuthenticationError, BadRequestError, UnsupportedParamsError
from litellm.litellm_core_utils.core_helpers import as_str_mapping, drops_unsupported_params
from litellm.litellm_core_utils.url_utils import ensure_api_route
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.text_to_speech.transformation import BaseTextToSpeechConfig, TextToSpeechRequestData
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import HttpxBinaryResponseContent

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging

_CONTAINERS: Final = MappingProxyType({"mp3": "mp3", "wav": "wav", "pcm": "raw", "opus": "ogg"})
_SPEEDS: Final = MappingProxyType({"slow": 0.85, "medium": 1.0, "fast": 1.15})


def _audio_config(model: str, response_format: object, native: object) -> Mapping[str, object]:
    config: Final = as_str_mapping(native)
    if native is not None and config is None:
        raise BadRequestError(message="audio_config must be an object.", model=model, llm_provider="gnani")
    if response_format is None:
        return config if config is not None else {"container": "mp3"}
    container: Final = _CONTAINERS.get(response_format) if isinstance(response_format, str) else None
    if container is None:
        raise UnsupportedParamsError(
            message="Gnani supports mp3, wav, pcm and opus. Use audio_config for native formats.",
            model=model,
            llm_provider="gnani",
        )
    fields: Final = config or {}
    native_container: Final = fields.get("container", container)
    encoding: Final = fields.get("encoding")
    compatible_ogg: Final = container == "ogg" and encoding == "oggopus" and native_container in ("ogg", "raw")
    if not compatible_ogg and (native_container != container or encoding in ("pcm_mulaw", "pcm_alaw", "oggopus")):
        raise BadRequestError(message="response_format conflicts with audio_config.", model=model, llm_provider="gnani")
    return {**({"sample_rate": 24000} if response_format == "pcm" else {}), "container": container, **fields}


def _speed(model: str, speed: object) -> object:
    if not isinstance(speed, str) or speed not in _SPEEDS:
        return speed
    from litellm.utils import catalog_with_runtime_registrations

    info: Final = catalog_with_runtime_registrations().get(f"gnani/{model}") or {}
    flags: Final = as_str_mapping(info.get("provider_specific_entry")) or {}
    return _SPEEDS[speed] if flags.get("gnani_speed_shortcuts") else speed


class GnaniTextToSpeechConfig(BaseTextToSpeechConfig):
    endpoint: ClassVar[str] = "inference"

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: base contract returns a list
        return ["voice", "response_format", "speed"]

    def map_openai_params(
        self,
        model: str,
        optional_params: Mapping[str, object],
        voice: object = None,
        drop_params: bool = False,
        kwargs: Mapping[str, object] | None = None,
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: speech dispatch expects a dict
        if not isinstance(voice, str) or not voice.strip():
            raise BadRequestError(
                message="Gnani requires a voice from the selected Timbre model.", model=model, llm_provider="gnani"
            )
        caller: Final = kwargs or {}
        if optional_params.get("instructions") is not None and not (
            drop_params or drops_unsupported_params(caller.get("drop_params"))
        ):
            raise UnsupportedParamsError(
                message="Gnani does not support instructions.", model=model, llm_provider="gnani"
            )
        extra: Final = as_str_mapping(caller.get("extra_body")) or {}
        if extra.get("stream") is True or extra.get("stream_format", caller.get("stream_format")) not in (
            None,
            "audio",
        ):
            raise UnsupportedParamsError(
                message="Gnani speech returns binary audio. Pass stream=True as a top-level speech parameter.",
                model=model,
                llm_provider="gnani",
            )
        speed: Final = optional_params.get("speed")
        return voice, {
            "language": extra.get("language", caller.get("language", "auto")),
            "audio_config": _audio_config(
                model, optional_params.get("response_format"), extra.get("audio_config", caller.get("audio_config"))
            ),
            **({"speed": _speed(model, speed)} if speed is not None else {}),
        }

    def validate_environment(
        self, headers: Mapping[str, str], model: str, api_key: str | None = None, api_base: str | None = None
    ) -> dict[str, str]:  # mutable-ok: HTTP handler updates the returned headers
        key: Final = api_key or get_secret_str("GNANI_API_KEY")
        if not key:
            raise AuthenticationError(
                message="Set GNANI_API_KEY or pass api_key for Gnani speech.", model=model, llm_provider="gnani"
            )
        if any(name.lower() in ("x-api-key-id", "authorization") for name in headers):
            raise BadRequestError(message="Pass Gnani credentials through api_key.", model=model, llm_provider="gnani")
        return {**headers, "X-API-Key-ID": key, "Content-Type": "application/json"}

    def get_complete_url(self, model: str, api_base: str | None, litellm_params: Mapping[str, object]) -> str:
        base: Final = (api_base or get_secret_str("GNANI_API_BASE") or "https://api.vachana.ai").rstrip("/")
        root: Final = base.removesuffix("/api/v1/tts/inference").removesuffix("/api/v1/tts/sse")
        return ensure_api_route(root, f"api/v1/tts/{self.endpoint}")

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: str | None,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, str],
    ) -> TextToSpeechRequestData:
        return {"dict_body": {**optional_params, "model": model, "text": input, "voice": voice}}

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "Logging",
    ) -> HttpxBinaryResponseContent:
        media_type: Final = next(iter(raw_response.headers.get_list("content-type")), "").partition(";")[0].lower()
        if not raw_response.content or not (
            media_type.startswith("audio/") or media_type == "application/octet-stream"
        ):
            raise BaseLLMException(502, "Gnani returned an empty or non-audio speech response.")
        if media_type == "application/octet-stream":
            raw_response.headers["content-type"] = "audio/pcm"  # rebind-ok: label Gnani's raw PCM for speech clients
        return HttpxBinaryResponseContent(raw_response)
