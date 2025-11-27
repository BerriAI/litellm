"""
Responses API transformation for Volcengine (Ark) models.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import httpx

import litellm
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.llms.openai.common_utils import OpenAIError
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
    ResponsesAPIStreamingResponse,
)
from litellm.types.responses.main import DeleteResponseResult
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

from ..common_utils import (
    VolcEngineError,
    get_volcengine_base_url,
    get_volcengine_headers,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class VolcEngineResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """
    Volcengine /responses implementation.

    Reference: https://www.volcengine.com/docs/82379/1569618
    """

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.VOLCENGINE

    def get_supported_openai_params(self, model: str) -> list:
        # Volcengine follows the OpenAI Responses API spec
        return super().get_supported_openai_params(model)

    def map_openai_params(
        self,
        response_api_optional_params: ResponsesAPIOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        # Reuse the OpenAI parameter mapping logic
        return super().map_openai_params(
            response_api_optional_params=response_api_optional_params,
            model=model,
            drop_params=drop_params,
        )

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: Optional[GenericLiteLLMParams],
    ) -> dict:
        """
        Ensure a Volcengine API key is available and attach the required headers.
        """
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key = (
            getattr(litellm_params, "api_key", None)
            or getattr(litellm, "api_key", None)
            or get_secret_str("ARK_API_KEY")
            or get_secret_str("VOLCENGINE_API_KEY")
        )

        if not api_key:
            raise VolcEngineError(
                status_code=401,
                message=(
                    "Volcengine API key is required. "
                    "Pass api_key or set VOLCENGINE_API_KEY."
                ),
            )

        # Merge existing headers with Volcengine defaults
        return get_volcengine_headers(api_key=api_key, extra_headers=headers)

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Construct the Volcengine /responses endpoint.
        """
        base_url = get_volcengine_base_url(api_base)
        base_url = base_url.rstrip("/")
        if base_url.endswith("/responses"):
            return base_url
        if base_url.endswith("/api/v3"):
            return f"{base_url}/responses"
        return f"{base_url}/api/v3/responses"

    def transform_responses_api_request(
        self,
        model: str,
        input,
        response_api_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        # Volcengine matches the OpenAI spec for request payloads
        return super().transform_responses_api_request(
            model=model,
            input=input,
            response_api_optional_request_params=response_api_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    def transform_response_api_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIResponse:
        return self._call_with_volcengine_error(
            super().transform_response_api_response,
            model=model,
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

    def transform_streaming_response(
        self,
        model: str,
        parsed_chunk: dict,
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIStreamingResponse:
        return self._call_with_volcengine_error(
            super().transform_streaming_response,
            model=model,
            parsed_chunk=parsed_chunk,
            logging_obj=logging_obj,
        )

    def transform_delete_response_api_request(
        self,
        response_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        return super().transform_delete_response_api_request(
            response_id=response_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )

    def transform_delete_response_api_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> DeleteResponseResult:
        return self._call_with_volcengine_error(
            super().transform_delete_response_api_response,
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

    def transform_get_response_api_request(
        self,
        response_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        return super().transform_get_response_api_request(
            response_id=response_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )

    def transform_get_response_api_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIResponse:
        return self._call_with_volcengine_error(
            super().transform_get_response_api_response,
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

    def transform_list_input_items_request(
        self,
        response_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        after: Optional[str] = None,
        before: Optional[str] = None,
        include: Optional[list] = None,
        limit: int = 20,
        order: Optional[str] = "desc",
    ) -> Tuple[str, Dict]:
        return super().transform_list_input_items_request(
            response_id=response_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
            after=after,
            before=before,
            include=include,
            limit=limit,
            order=order,
        )

    def transform_list_input_items_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> Dict:
        return self._call_with_volcengine_error(
            super().transform_list_input_items_response,
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

    def transform_cancel_response_api_request(
        self,
        response_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        return super().transform_cancel_response_api_request(
            response_id=response_id,
            api_base=api_base,
            litellm_params=litellm_params,
            headers=headers,
        )

    def transform_cancel_response_api_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIResponse:
        return self._call_with_volcengine_error(
            super().transform_cancel_response_api_response,
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

    def _call_with_volcengine_error(self, func, *args, **kwargs):
        """
        将 OpenAIError 统一转换成 VolcEngineError。
        """
        try:
            return func(*args, **kwargs)
        except OpenAIError as exc:
            raise VolcEngineError(
                status_code=exc.status_code,
                message=exc.message,
                headers=exc.headers,
            ) from exc
