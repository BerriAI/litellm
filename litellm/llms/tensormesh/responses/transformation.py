from typing import Any, Optional, Union

import httpx

from litellm.litellm_core_utils.core_helpers import process_response_headers
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    _safe_convert_created_field,
)
from litellm.llms.openai.common_utils import OpenAIError
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    ResponseAPIUsage,
    ResponsesAPIResponse,
    ResponsesAPIStreamingResponse,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

from litellm.llms.tensormesh.common_utils import TENSORMESH_API_BASE


class TensormeshResponsesConfig(OpenAIResponsesAPIConfig):
    @property
    def custom_llm_provider(self) -> Union[str, LlmProviders]:  # type: ignore[override]
        return "tensormesh"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: Optional[GenericLiteLLMParams],
    ) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key = litellm_params.api_key or get_secret_str("TENSORMESH_INFERENCE_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    @staticmethod
    def _normalize_response_usage(
        response_json: dict,
        convert_to_model: bool = True,
    ) -> None:
        usage = response_json.get("usage")
        if not isinstance(usage, dict):
            return

        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if "input_tokens" not in usage and prompt_tokens is not None:
            usage["input_tokens"] = prompt_tokens
        if "output_tokens" not in usage and completion_tokens is not None:
            usage["output_tokens"] = completion_tokens
        if (
            "total_tokens" not in usage
            and usage.get("input_tokens") is not None
            and usage.get("output_tokens") is not None
        ):
            usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]

        prompt_details = usage.get("prompt_tokens_details")
        if isinstance(prompt_details, dict) and "input_tokens_details" not in usage:
            usage["input_tokens_details"] = {
                "audio_tokens": prompt_details.get("audio_tokens"),
                "cached_tokens": prompt_details.get("cached_tokens") or 0,
                "text_tokens": prompt_details.get("text_tokens"),
            }

        completion_details = usage.get("completion_tokens_details")
        if isinstance(completion_details, dict) and "output_tokens_details" not in usage:
            usage["output_tokens_details"] = {
                "reasoning_tokens": completion_details.get("reasoning_tokens") or 0,
                "text_tokens": completion_details.get("text_tokens"),
            }
        if (
            convert_to_model
            and usage.get("input_tokens") is not None
            and usage.get("output_tokens") is not None
            and usage.get("total_tokens") is not None
        ):
            response_json["usage"] = ResponseAPIUsage(**usage)

    def transform_response_api_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: Any,
    ) -> ResponsesAPIResponse:
        try:
            logging_obj.post_call(
                original_response=raw_response.text,
                additional_args={"complete_input_dict": {}},
            )
            raw_response_json = raw_response.json()
            raw_response_json["created_at"] = _safe_convert_created_field(raw_response_json["created_at"])
        except Exception:
            raise OpenAIError(message=raw_response.text, status_code=raw_response.status_code)

        self._normalize_response_usage(raw_response_json)
        raw_response_headers = dict(raw_response.headers)
        try:
            response = ResponsesAPIResponse(**raw_response_json)
        except Exception:
            response = ResponsesAPIResponse.model_construct(**raw_response_json)
        response._hidden_params["additional_headers"] = process_response_headers(raw_response_headers)
        response._hidden_params["headers"] = raw_response_headers
        return response

    def transform_streaming_response(
        self,
        model: str,
        parsed_chunk: dict,
        logging_obj: Any,
    ) -> ResponsesAPIStreamingResponse:
        response = parsed_chunk.get("response")
        if isinstance(response, dict):
            self._normalize_response_usage(response, convert_to_model=False)
        self._normalize_response_usage(parsed_chunk, convert_to_model=False)
        return super().transform_streaming_response(
            model=model,
            parsed_chunk=parsed_chunk,
            logging_obj=logging_obj,
        )

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        api_base = api_base or get_secret_str("TENSORMESH_SERVERLESS_BASE_URL") or TENSORMESH_API_BASE
        return f"{api_base.rstrip('/')}/responses"

    def supports_native_websocket(self) -> bool:
        return False
