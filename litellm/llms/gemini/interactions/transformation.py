"""
Google AI Studio Interactions API configuration.

Per OpenAPI spec (https://ai.google.dev/static/api/interactions.openapi.json):
- Create: POST https://generativelanguage.googleapis.com/{api_version}/interactions
- Get: GET https://generativelanguage.googleapis.com/{api_version}/interactions/{interaction_id}
- Delete: DELETE https://generativelanguage.googleapis.com/{api_version}/interactions/{interaction_id}

This is a thin wrapper - no transformation needed since we follow the spec directly.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import httpx

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.core_helpers import process_response_headers
from litellm.llms.base_llm.interactions.transformation import BaseInteractionsAPIConfig
from litellm.llms.gemini.common_utils import GeminiError, GeminiModelInfo
from litellm.types.interactions import (
    CancelInteractionResult,
    DeleteInteractionResult,
    InteractionInput,
    InteractionsAPIOptionalRequestParams,
    InteractionsAPIResponse,
    InteractionsAPIStreamingResponse,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class GoogleAIStudioInteractionsConfig(BaseInteractionsAPIConfig):
    """
    Configuration for Google AI Studio Interactions API.
    
    Minimal config - we follow the OpenAPI spec directly with no transformation.
    """

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.GEMINI

    @property
    def api_version(self) -> str:
        return "v1beta"

    def get_supported_params(self, model: str) -> List[str]:
        """Per OpenAPI spec CreateModelInteractionParams."""
        return [
            "model", "agent", "input", "tools", "system_instruction",
            "generation_config", "stream", "store", "background",
            "response_modalities", "response_format", "response_mime_type",
            "previous_interaction_id",
        ]

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: Optional[GenericLiteLLMParams],
    ) -> dict:
        """Google AI Studio uses API key in query params, not headers."""
        headers = headers or {}
        headers["Content-Type"] = "application/json"
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        model: Optional[str],
        agent: Optional[str] = None,
        litellm_params: Optional[dict] = None,
        stream: Optional[bool] = None,
    ) -> str:
        """POST /{api_version}/interactions"""
        litellm_params = litellm_params or {}
        api_base = GeminiModelInfo.get_api_base(api_base)
        api_key = GeminiModelInfo.get_api_key(litellm_params.get("api_key"))
        
        if not api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
            )
        
        query_params = f"key={api_key}"
        if stream:
            query_params += "&alt=sse"
        
        return f"{api_base}/{self.api_version}/interactions?{query_params}"

    def transform_request(
        self,
        model: Optional[str],
        agent: Optional[str],
        input: Optional[InteractionInput],
        optional_params: InteractionsAPIOptionalRequestParams,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        """
        Build request body per OpenAPI spec - minimal transformation.
        """
        request_body: Dict[str, Any] = {}
        
        # Model or Agent (one required)
        if model:
            request_body["model"] = GeminiModelInfo.get_base_model(model) or model
        elif agent:
            request_body["agent"] = agent
        else:
            raise ValueError("Either 'model' or 'agent' must be provided")
        
        # Input
        if input is not None:
            request_body["input"] = input
        
        # Pass through optional params directly (they match the spec)
        optional_keys = [
            "tools", "system_instruction", "generation_config", "stream", "store",
            "background", "response_modalities", "response_format",
            "response_mime_type", "previous_interaction_id",
        ]
        for key in optional_keys:
            if optional_params.get(key) is not None:
                request_body[key] = optional_params[key]
        
        return request_body

    def transform_response(
        self,
        model: Optional[str],
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> InteractionsAPIResponse:
        """Parse response - it already matches our response type."""
        try:
            logging_obj.post_call(
                original_response=raw_response.text,
                additional_args={"complete_input_dict": {}},
            )
            raw_json = raw_response.json()
        except Exception:
            raise GeminiError(
                message=raw_response.text,
                status_code=raw_response.status_code,
                headers=dict(raw_response.headers),
            )
        
        verbose_logger.debug("Google AI Interactions response: %s", raw_json)
        
        response = InteractionsAPIResponse(**raw_json)
        response._hidden_params["headers"] = dict(raw_response.headers)
        response._hidden_params["additional_headers"] = process_response_headers(dict(raw_response.headers))
        
        return response

    def transform_streaming_response(
        self,
        model: Optional[str],
        parsed_chunk: dict,
        logging_obj: LiteLLMLoggingObj,
    ) -> InteractionsAPIStreamingResponse:
        """Parse streaming chunk."""
        verbose_logger.debug("Google AI Interactions streaming chunk: %s", parsed_chunk)
        return InteractionsAPIStreamingResponse(**parsed_chunk)

    # GET / DELETE / CANCEL - just build URLs, responses match spec directly
    
    def transform_get_interaction_request(
        self,
        interaction_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """GET /{api_version}/interactions/{interaction_id}"""
        resolved_api_base = GeminiModelInfo.get_api_base(api_base)
        api_key = GeminiModelInfo.get_api_key(litellm_params.api_key)
        if not api_key:
            raise ValueError("Google API key is required")
        return f"{resolved_api_base}/{self.api_version}/interactions/{interaction_id}?key={api_key}", {}

    def transform_get_interaction_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> InteractionsAPIResponse:
        try:
            raw_json = raw_response.json()
        except Exception:
            raise GeminiError(
                message=raw_response.text,
                status_code=raw_response.status_code,
                headers=dict(raw_response.headers),
            )
        response = InteractionsAPIResponse(**raw_json)
        response._hidden_params["headers"] = dict(raw_response.headers)
        return response

    def transform_delete_interaction_request(
        self,
        interaction_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """DELETE /{api_version}/interactions/{interaction_id}"""
        resolved_api_base = GeminiModelInfo.get_api_base(api_base)
        api_key = GeminiModelInfo.get_api_key(litellm_params.api_key)
        if not api_key:
            raise ValueError("Google API key is required")
        return f"{resolved_api_base}/{self.api_version}/interactions/{interaction_id}?key={api_key}", {}

    def transform_delete_interaction_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        interaction_id: str,
    ) -> DeleteInteractionResult:
        if 200 <= raw_response.status_code < 300:
            return DeleteInteractionResult(success=True, id=interaction_id)
        raise GeminiError(
            message=raw_response.text,
            status_code=raw_response.status_code,
            headers=dict(raw_response.headers),
        )

    def transform_cancel_interaction_request(
        self,
        interaction_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """POST /{api_version}/interactions/{interaction_id}:cancel (if supported)"""
        resolved_api_base = GeminiModelInfo.get_api_base(api_base)
        api_key = GeminiModelInfo.get_api_key(litellm_params.api_key)
        if not api_key:
            raise ValueError("Google API key is required")
        return f"{resolved_api_base}/{self.api_version}/interactions/{interaction_id}:cancel?key={api_key}", {}

    def transform_cancel_interaction_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> CancelInteractionResult:
        try:
            raw_json = raw_response.json()
        except Exception:
            raise GeminiError(
                message=raw_response.text,
                status_code=raw_response.status_code,
                headers=dict(raw_response.headers),
            )
        return CancelInteractionResult(**raw_json)
