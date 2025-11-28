from typing import List, Optional, Union

from httpx import Headers, Response

from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIAudioTranscriptionOptionalParams,
)
from litellm.types.utils import FileTypes, TranscriptionResponse

from ..common_utils import BurnCloudError


class BurnCloudAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        if api_base is None:
            api_base = get_secret_str("BURNCLOUD_API_BASE")

        # Remove trailing slashes and ensure clean base URL
        api_base = api_base.rstrip("/") if api_base else api_base

        # if endswith "/v1"
        if api_base and api_base.endswith("/v1"):
            api_base = f"{api_base}/audio/transcriptions"
        else:
            api_base = f"{api_base}/v1/audio/transcriptions"

        return api_base

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIAudioTranscriptionOptionalParams]:
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
        supported_params = self.get_supported_openai_params(model)
        for k, v in non_default_params.items():
            if k in supported_params:
                optional_params[k] = v
        return optional_params

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
        api_key = api_key or get_secret_str("BURNCLOUD_API_KEY")

        default_headers = {
            "Authorization": f"Bearer {api_key}",
        }

        # If 'Authorization' is provided in headers, it overrides the default.
        if "Authorization" in headers:
            default_headers["Authorization"] = headers["Authorization"]

        # Merge other headers, overriding any default ones except Authorization
        return {**default_headers, **headers}

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
    ) -> AudioTranscriptionRequestData:
        """
        Transform the audio transcription request
        """
        data = {"model": model, "file": audio_file, **optional_params}
        return AudioTranscriptionRequestData(
            data=data,
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, Headers]
    ) -> BaseLLMException:
        return BurnCloudError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    def transform_audio_transcription_response(
        self,
        raw_response: Response,
    ) -> TranscriptionResponse:
        try:
            raw_response_json = raw_response.json()
        except Exception as e:
            raise ValueError(
                f"Error transforming response to json: {str(e)}\nResponse: {raw_response.text}"
            )

        if any(
            key in raw_response_json
            for key in TranscriptionResponse.model_fields.keys()
        ):
            return TranscriptionResponse(**raw_response_json)
        else:
            raise ValueError(
                "Invalid response format. Received response does not match the expected format. Got: ",
                raw_response_json,
            )
