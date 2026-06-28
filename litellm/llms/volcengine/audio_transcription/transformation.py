from typing import Any

from httpx import Headers, Response

from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.volcengine.common_utils import (
    VolcEngineError,
    get_volcengine_configured_ws_api_base,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIAudioTranscriptionOptionalParams,
)
from litellm.types.utils import FileTypes, TranscriptionResponse

VOLCENGINE_STT_DEFAULT_API_BASE = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel"
VOLCENGINE_STT_DEFAULT_RESOURCE_ID = "volc.bigasr.sauc.duration"
VOLCENGINE_STT_RESOURCE_IDS = {
    "volc.bigasr.sauc.duration",
    "volc.bigasr.sauc.concurrent",
    "volc.seedasr.sauc.duration",
    "volc.seedasr.sauc.concurrent",
}


class VolcEngineAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIAudioTranscriptionOptionalParams]:
        return ["language", "response_format"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for key in ("language", "response_format"):
            value = non_default_params.get(key)
            if value is not None:
                optional_params[key] = value
        return optional_params

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        return get_volcengine_configured_ws_api_base(
            litellm_params=litellm_params,
            default_api_base=VOLCENGINE_STT_DEFAULT_API_BASE,
        )

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
    ) -> AudioTranscriptionRequestData:
        request_extra = optional_params.get("request_extra")
        if request_extra is not None and not isinstance(request_extra, dict):
            raise VolcEngineError(
                status_code=400,
                message="Volcengine STT request_extra must be an object when provided.",
            )
        payload: dict[str, Any] = {
            "user": {"uid": optional_params.get("user_id", "litellm-proxy")},
            "audio": {
                "format": "pcm",
                "codec": "raw",
                "rate": 16000,
                "bits": 16,
                "channel": 1,
            },
            "request": {
                "model_name": optional_params.get("model_name", "bigmodel"),
                "enable_itn": optional_params.get("enable_itn", True),
                "enable_punc": optional_params.get("enable_punc", True),
                "show_utterances": optional_params.get("show_utterances", True),
                "result_type": optional_params.get("result_type", "single"),
                "end_window_size": optional_params.get("end_window_size", 300),
                **(request_extra or {}),
            },
        }
        if optional_params.get("language"):
            payload["audio"]["language"] = optional_params["language"]
        return AudioTranscriptionRequestData(data=payload, files=None)

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        return headers

    def transform_audio_transcription_response(
        self,
        raw_response: Response,
    ) -> TranscriptionResponse:
        raise NotImplementedError("Volcengine STT uses a WebSocket handler.")

    def get_error_class(
        self, error_message: str, status_code: int, headers: dict | Headers
    ) -> BaseLLMException:
        return VolcEngineError(status_code=status_code, message=error_message)


def pick_stt_resource_id(model_name: str | None) -> str:
    model = (model_name or "").lower().strip()
    if model.startswith("volcengine/"):
        model = model.split("/", 1)[1]
    if model in VOLCENGINE_STT_RESOURCE_IDS:
        return model
    return VOLCENGINE_STT_DEFAULT_RESOURCE_ID
