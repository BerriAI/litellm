import base64
import json
from typing import TYPE_CHECKING, ClassVar, Final

import httpx
from httpx import Headers

import litellm
from litellm.llms.base_llm.text_to_speech.transformation import (
    BaseTextToSpeechConfig,
    TextToSpeechRequestData,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import HttpxBinaryResponseContent

from ..common_utils import BytePlusError

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_BYTEPLUS_TTS_AUTH_HEADERS: Final = frozenset(
    {
        "x-api-resource-id",
        "x-api-key",
        "x-api-app-id",
        "x-api-access-key",
        "x-api-app-key",
    }
)

_FORMAT_CONTENT_TYPES: Final[dict[str, str]] = {
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
    "aac": "audio/aac",
    "opus": "audio/opus",
    "flac": "audio/flac",
}


def _infer_audio_content_type(audio_bytes: bytes, requested_format: str) -> str:
    if audio_bytes[:4] == b"OggS":
        return "audio/ogg"
    if audio_bytes[:4] == b"RIFF":
        return "audio/wav"
    if audio_bytes[:4] == b"fLaC":
        return "audio/flac"
    if audio_bytes[:3] == b"ID3" or (
        len(audio_bytes) >= 2 and audio_bytes[0] == 0xFF and audio_bytes[1] in (0xFB, 0xF3, 0xF2)
    ):
        return "audio/mpeg"
    return _FORMAT_CONTENT_TYPES.get(requested_format.lower(), f"audio/{requested_format.lower()}")


def _extract_requested_format(
    raw_response: httpx.Response,
    logging_obj: "LiteLLMLoggingObj | None",
    default_format: str,
) -> str:
    req: Final[httpx.Request | None] = getattr(raw_response, "_request", None)
    if req is not None:
        try:
            req_content: Final = getattr(req, "content", None)
            if isinstance(req_content, (bytes, bytearray, str)) and req_content:
                req_json: Final = json.loads(req_content)
                if isinstance(req_json, dict):
                    req_params: Final = req_json.get("req_params")
                    if isinstance(req_params, dict):
                        audio_params: Final = req_params.get("audio_params")
                        if isinstance(audio_params, dict):
                            fmt: Final = audio_params.get("format")
                            if isinstance(fmt, str) and fmt.strip():
                                return fmt.strip()
        except (json.JSONDecodeError, ValueError, AttributeError, TypeError):  # noqa: BLE001  # fallback if unparseable
            pass

    if logging_obj is not None:
        opts: Final = getattr(logging_obj, "optional_params", None)
        if isinstance(opts, dict):
            fmt_opts: Final = opts.get("response_format") or opts.get("format")
            if isinstance(fmt_opts, str) and fmt_opts.strip():
                return fmt_opts.strip()
        mcd: Final = getattr(logging_obj, "model_call_details", None)
        if isinstance(mcd, dict):
            mcd_opts: Final = mcd.get("optional_params")
            if isinstance(mcd_opts, dict):
                fmt_mcd_opts: Final = mcd_opts.get("response_format") or mcd_opts.get("format")
                if isinstance(fmt_mcd_opts, str) and fmt_mcd_opts.strip():
                    return fmt_mcd_opts.strip()

    return default_format


class BytePlusTextToSpeechConfig(BaseTextToSpeechConfig):
    DEFAULT_TTS_BASE_URL = "https://voice.ap-southeast-1.bytepluses.com/api/v3/tts/unidirectional"
    DEFAULT_APP_KEY = "aGjiRDfUWi"
    DEFAULT_SAMPLE_RATE = 24000
    DEFAULT_FORMAT = "mp3"

    def get_supported_openai_params(self, model: str) -> list:
        return ["voice", "response_format", "speed"]

    def map_openai_params(
        self,
        model: str,
        optional_params: dict,
        voice: str | dict | None = None,
        drop_params: bool = False,
        kwargs: dict | None = None,
    ) -> tuple[str | None, dict]:
        mapped_voice: str | None = None

        if isinstance(voice, str) and voice.strip():
            mapped_voice = voice.strip()
        elif isinstance(voice, dict):
            for key in ("speaker", "voice", "id", "name"):
                candidate = voice.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    mapped_voice = candidate.strip()
                    break

        if mapped_voice is None and kwargs:
            speaker_kwarg = kwargs.pop("speaker", None)
            if isinstance(speaker_kwarg, str) and speaker_kwarg.strip():
                mapped_voice = speaker_kwarg.strip()

        mapped_params = dict(optional_params) if optional_params else {}

        return mapped_voice, mapped_params

    VOICE_MAPPINGS: ClassVar[dict[str, str]] = {
        "alloy": "en_female_stokie_uranus_bigtts",
        "echo": "en_male_adam_mars_bigtts",
        "fable": "en_female_skye_emo_v2_mars_bigtts",
        "onyx": "id_male_han_uranus_bigtts",
        "nova": "en_female_stokie_uranus_bigtts",
        "shimmer": "en_female_skye_emo_v2_mars_bigtts",
    }

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        resolved_api_key: Final = (
            api_key or litellm.api_key or get_secret_str("BYTEPLUS_API_KEY") or get_secret_str("ARK_API_KEY")
        )
        app_id = get_secret_str("BYTEPLUS_TTS_APP_ID")
        access_key = get_secret_str("BYTEPLUS_TTS_ACCESS_KEY")
        app_key = get_secret_str("BYTEPLUS_TTS_APP_KEY") or self.DEFAULT_APP_KEY

        resource_id = model
        if resource_id.startswith("byteplus/"):
            resource_id = resource_id.replace("byteplus/", "", 1)

        req_headers: dict[str, str] = {
            "X-Api-Resource-Id": resource_id,
            "Content-Type": "application/json",
            "Connection": "keep-alive",
        }

        if resolved_api_key:
            req_headers["x-api-key"] = resolved_api_key.strip()
        elif app_id and access_key:
            req_headers["X-Api-App-Id"] = app_id.strip()
            req_headers["X-Api-Access-Key"] = access_key.strip()
            req_headers["X-Api-App-Key"] = app_key.strip()
        else:
            raise ValueError(
                "BytePlus TTS requires authentication. Provide BYTEPLUS_API_KEY (or ARK_API_KEY) or both BYTEPLUS_TTS_APP_ID and BYTEPLUS_TTS_ACCESS_KEY."
            )

        headers.update(req_headers)
        return headers

    def get_error_class(self, error_message: str, status_code: int, headers: dict | Headers) -> BytePlusError:
        typed_headers: httpx.Headers = headers if isinstance(headers, httpx.Headers) else httpx.Headers(headers or {})
        return BytePlusError(status_code=status_code, message=error_message, headers=typed_headers)

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        tts_base_env = get_secret_str("BYTEPLUS_TTS_API_BASE")
        if tts_base_env:
            base_url = tts_base_env
        elif api_base and "voice" in api_base:
            base_url = api_base
        else:
            base_url = self.DEFAULT_TTS_BASE_URL

        base_url = base_url.rstrip("/")
        if base_url.endswith("/api/v3/tts/unidirectional"):
            return base_url
        if base_url.endswith("/api/v3"):
            return f"{base_url}/tts/unidirectional"
        if base_url.endswith("bytepluses.com"):
            return f"{base_url}/api/v3/tts/unidirectional"

        return base_url

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: str | None,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> TextToSpeechRequestData:
        params = dict(optional_params) if optional_params else {}
        extra_body = params.pop("extra_body", {}) or {}

        raw_speaker = voice or extra_body.get("speaker") or "en_female_stokie_uranus_bigtts"
        speaker = self.VOICE_MAPPINGS.get(raw_speaker.lower(), raw_speaker)

        audio_format = params.get("response_format") or extra_body.get("format") or self.DEFAULT_FORMAT
        sample_rate = extra_body.get("sample_rate") or self.DEFAULT_SAMPLE_RATE
        speed = params.get("speed")

        additions_dict: dict = {
            "disable_markdown_filter": True,
            "enable_language_detector": True,
            "enable_latex_tn": True,
            "disable_default_bit_rate": True,
            "max_length_to_filter_parenthesis": 0,
            "cache_config": {"text_type": 1, "use_cache": True},
        }

        user_additions = extra_body.get("additions")
        if isinstance(user_additions, str):
            try:
                parsed_user_additions = json.loads(user_additions)
                if isinstance(parsed_user_additions, dict):
                    additions_dict.update(parsed_user_additions)
            except (json.JSONDecodeError, ValueError):  # noqa: BLE001  # ignore invalid user additions
                pass
        elif isinstance(user_additions, dict):
            additions_dict.update(user_additions)

        user_id = extra_body.get("uid") or litellm_params.get("user") or "litellm-user"

        audio_params: dict = {
            "format": audio_format,
            "sample_rate": sample_rate,
        }
        if speed is not None:
            audio_params["speed_ratio"] = float(speed)

        payload = {
            "user": {"uid": str(user_id)},
            "req_params": {
                "text": input,
                "speaker": speaker,
                "audio_params": audio_params,
                "additions": json.dumps(additions_dict),
            },
        }

        return TextToSpeechRequestData(
            dict_body=payload,
            headers={"Content-Type": "application/json"},
        )

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> HttpxBinaryResponseContent:
        if raw_response.status_code != 200:
            raise BytePlusError(
                status_code=raw_response.status_code,
                message=f"BytePlus TTS request failed: {raw_response.text}",
                headers=raw_response.headers,
            )

        audio_bytes = bytearray()

        lines = raw_response.text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except (json.JSONDecodeError, ValueError):  # noqa: BLE001  # skip invalid line
                continue

            code = data.get("code", 0)

            if code == 0 and data.get("data"):
                try:
                    chunk_audio = base64.b64decode(data["data"])
                    audio_bytes.extend(chunk_audio)
                except (TypeError, ValueError):  # noqa: BLE001  # skip invalid audio chunk
                    pass
            elif code == 20000000:
                break
            elif code > 0:
                msg = data.get("message") or f"BytePlus TTS error code {code}"
                raise BytePlusError(
                    status_code=400,
                    message=f"BytePlus TTS API Error: {msg}",
                    headers=raw_response.headers,
                )

        if not audio_bytes:
            raise BytePlusError(
                status_code=500,
                message="BytePlus TTS returned no audio data.",
                headers=raw_response.headers,
            )

        audio_bytes_final: Final = bytes(audio_bytes)
        requested_format: Final = _extract_requested_format(raw_response, logging_obj, self.DEFAULT_FORMAT)
        content_type: Final = _infer_audio_content_type(audio_bytes_final, requested_format)
        mock_http_response: Final = httpx.Response(
            status_code=200,
            content=audio_bytes_final,
            headers={"content-type": content_type},
        )
        return HttpxBinaryResponseContent(mock_http_response)

    def dispatch_text_to_speech(
        self,
        model: str,
        input: str,
        voice: str | dict | None,
        optional_params: dict,
        litellm_params_dict: dict,
        logging_obj: "LiteLLMLoggingObj",
        timeout: float | httpx.Timeout,
        extra_headers: dict | None,
        base_llm_http_handler: object,
        aspeech: bool,
        api_base: str | None,
        api_key: str | None,
        **kwargs: object,
    ) -> object:
        safe_extra_headers: Final[dict | None] = (
            {k: v for k, v in extra_headers.items() if k.lower() not in _BYTEPLUS_TTS_AUTH_HEADERS}
            if extra_headers
            else None
        )
        litellm_params_dict.update(
            {
                "api_key": api_key,
                "api_base": api_base,
            }
        )
        return base_llm_http_handler.text_to_speech_handler(
            model=model,
            input=input,
            voice=voice,
            text_to_speech_provider_config=self,
            text_to_speech_optional_params=optional_params,
            custom_llm_provider="byteplus",
            litellm_params=litellm_params_dict,
            logging_obj=logging_obj,
            timeout=timeout,
            extra_headers=safe_extra_headers,
            client=None,
            _is_async=aspeech,
        )
