"""
Translates from OpenAI's `/v1/audio/transcriptions` to Databricks Model
Serving's MLflow invoke route `/serving-endpoints/{endpoint}/invocations`.

The api_base (parameter or DATABRICKS_API_BASE) is accepted either as a
bare workspace host or as a base that already carries `/serving-endpoints`
(the litellm docs convention); a bare host is normalized by appending the
path. The OpenAI-compatible audio routes require a token with `all-apis`
scope; the invoke route accepts any workspace token, so it is the default
target.
"""

import base64
import os
import re
from collections.abc import Mapping, Sequence
from typing import Final

from httpx import Headers, Response
from pydantic import TypeAdapter, ValidationError

from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIAudioTranscriptionOptionalParams,
)
from litellm.types.utils import FileTypes, TranscriptionResponse

from ...base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from ..common_utils import DatabricksBase, DatabricksException

_OBJECT_LIST: Final = TypeAdapter(list[object])
_STRING_OBJECT_DICT: Final = TypeAdapter(dict[str, object])
# dots are excluded so path-shaping inputs such as ".." cannot reach the URL
_ENDPOINT_NAME: Final = re.compile(r"[A-Za-z0-9_-]+")


class DatabricksAudioTranscriptionConfig(BaseAudioTranscriptionConfig, DatabricksBase):
    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIAudioTranscriptionOptionalParams]:  # mutable-ok: base class signature returns list
        # accept-and-ignore: advertising these avoids UnsupportedParamsError; the invoke route has no parameter surface
        return [  # mutable-ok: fixed provider param set, never grown
            "language",
            "prompt",
            "response_format",
            "temperature",
            "timestamp_granularities",
        ]

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: base class signature returns dict
        # accept-and-ignore, see get_supported_openai_params
        return dict(optional_params)  # mutable-ok: base class signature returns dict

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, object] | Headers,  # mutable-ok: base class signature takes dict
    ) -> BaseLLMException:
        return DatabricksException(message=error_message, status_code=status_code, headers=headers)

    def validate_environment(
        self,
        headers: dict[str, object],  # mutable-ok: base class signature takes and returns dict
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, object]:  # mutable-ok: base class signature returns dict
        # endpoint_type="chat_completions" reuses the shared Databricks auth; the
        # /chat/completions suffix it appends is discarded (get_complete_url builds the URL)
        auth_headers: Final = _STRING_OBJECT_DICT.validate_python(
            self.databricks_validate_environment(  # pyright: ignore[reportUnknownMemberType]  # common_utils auth predates strict typing; re-annotating it is out of scope here
                api_key=api_key,
                api_base=api_base,
                endpoint_type="chat_completions",
                custom_endpoint=False,
                headers=headers,
            )[1]
        )
        auth_headers["Content-Type"] = "application/json"
        return auth_headers

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        endpoint: Final = model.removeprefix("databricks/")
        if _ENDPOINT_NAME.fullmatch(endpoint) is None:
            raise DatabricksException(
                status_code=400,
                message=(
                    f"Invalid Databricks endpoint name {endpoint!r}: names may only contain "
                    "letters, digits, underscores, and hyphens."
                ),
            )
        resolved_base: Final = self._get_api_base(api_base or os.getenv("DATABRICKS_API_BASE")).rstrip("/")
        # _get_api_base appends /serving-endpoints only in its SDK-fallback branch
        serving_base: Final = (
            resolved_base if resolved_base.endswith("/serving-endpoints") else f"{resolved_base}/serving-endpoints"
        )
        return f"{serving_base}/{endpoint}/invocations"

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
    ) -> AudioTranscriptionRequestData:
        processed_audio: Final = process_audio_file(audio_file)
        encoded_audio: Final = base64.b64encode(processed_audio.file_content).decode("ascii")
        return AudioTranscriptionRequestData(
            data={"inputs": [encoded_audio]},  # mutable-ok: AudioTranscriptionRequestData.data is a dict payload
            files=None,
            content_type="application/json",
        )

    def transform_audio_transcription_response(
        self,
        raw_response: Response,
    ) -> TranscriptionResponse:
        if raw_response.status_code >= 400:
            raise self.get_error_class(
                error_message=raw_response.text,
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )
        try:
            payload: Final = _STRING_OBJECT_DICT.validate_python(raw_response.json())
        except ValueError as e:
            raise DatabricksException(
                status_code=500,
                message=f"Error parsing Databricks transcription response: {e}\nResponse: {raw_response.text}",
            )
        text: Final = self._extract_prediction(payload)
        if text is None:  # "" is a valid transcript (e.g. silent audio); only a missing prediction is an error
            raise DatabricksException(
                status_code=500,
                message=(
                    f"Databricks transcription response carried no prediction text. Response: {raw_response.text}"
                ),
            )
        response: Final = TranscriptionResponse(text=text)
        response["task"] = "transcribe"
        response._hidden_params = payload  # pyright: ignore[reportPrivateUsage]  # TranscriptionResponse exposes no public hidden-params setter
        return response

    @staticmethod
    def _extract_prediction(payload: Mapping[str, object]) -> str | None:
        predictions: Final[object] = payload.get("predictions")
        if isinstance(predictions, str):
            return predictions
        try:
            prediction_items: Final = _OBJECT_LIST.validate_python(predictions)
        except ValidationError:
            return None
        if not prediction_items:
            return None
        first: Final[object] = prediction_items[0]
        if isinstance(first, str):
            return first
        try:
            first_dict: Final = _STRING_OBJECT_DICT.validate_python(first)
        except ValidationError:
            return None
        text: Final[object] = first_dict.get("text")
        return text if isinstance(text, str) else None
