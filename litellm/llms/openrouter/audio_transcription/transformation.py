from typing import Any, Dict, List, Optional, Union

import httpx

from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openrouter.common_utils import (
    OpenRouterException,
    get_openrouter_endpoint,
    get_openrouter_headers,
    raise_openrouter_error,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIAudioTranscriptionOptionalParams,
)
from litellm.types.utils import FileTypes, TranscriptionResponse


class OpenRouterAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    def get_supported_openai_params(self, model: str) -> List[OpenAIAudioTranscriptionOptionalParams]:
        return [
            "language",
            "prompt",
            "response_format",
            "temperature",
            "timestamp_granularities",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = set(self.get_supported_openai_params(model))
        for key, value in non_default_params.items():
            if value is None:
                continue
            if key in supported_params or not drop_params:
                optional_params[key] = value
        return optional_params

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        return get_openrouter_endpoint(api_base, "audio/transcriptions")

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        merged_headers = get_openrouter_headers(api_key=api_key, headers=headers, content_type=None)
        merged_headers.pop("Content-Type", None)
        merged_headers.pop("content-type", None)
        return merged_headers

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
    ) -> AudioTranscriptionRequestData:
        processed_audio = process_audio_file(audio_file)

        form_fields: Dict[str, Any] = {"model": model}
        for key in self.get_supported_openai_params(model):
            value = optional_params.get(key)
            if value is not None:
                form_key = "timestamp_granularities[]" if key == "timestamp_granularities" else key
                form_fields[form_key] = value

        provider_specific_params = self.get_provider_specific_params(
            model=model,
            optional_params=optional_params,
            openai_params=self.get_supported_openai_params(model),
        )
        form_fields.update(provider_specific_params)

        files = {
            "file": (
                processed_audio.filename,
                processed_audio.file_content,
                processed_audio.content_type,
            )
        }

        return AudioTranscriptionRequestData(data=form_fields, files=files)

    def transform_audio_transcription_response(
        self,
        raw_response: httpx.Response,
    ) -> TranscriptionResponse:
        try:
            response_json = raw_response.json()
        except ValueError:
            raise OpenRouterException(
                message=raw_response.text,
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        raise_openrouter_error(raw_response)

        if not isinstance(response_json, dict):
            raise OpenRouterException(
                message="OpenRouter returned a non-object transcription response.",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        response = TranscriptionResponse(text=response_json.get("text") or "")
        for key, value in response_json.items():
            if key != "text":
                response[key] = value
        response._hidden_params = response_json
        return response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return OpenRouterException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )
