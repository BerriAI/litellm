import uuid
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.core_helpers import process_response_headers
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    _safe_convert_created_field,
)
from litellm.llms.openai.common_utils import OpenAIError
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    ResponseAPIUsage,
    ResponseInputParam,
    ResponsesAPIResponse,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

MANUS_API_BASE = "https://api.manus.im"


class ManusResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """
    Configuration for Manus API's Responses API.
    
    Manus API is OpenAI-compatible but has some differences:
    - API key passed via `API_KEY` header (not `Authorization: Bearer`)
    - Model format: `manus/{agent_profile}` (e.g., `manus/manus-1.6`)
    - Requires `extra_body` with `task_mode: "agent"` and `agent_profile`
    
    Reference: https://open.manus.im/docs/openai-compatibility
    """

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.MANUS

    def should_fake_stream(
        self,
        model: Optional[str],
        stream: Optional[bool],
        custom_llm_provider: Optional[str] = None,
    ) -> bool:
        """
        Manus API doesn't support real-time streaming.
        It returns a task that runs asynchronously.
        We fake streaming by converting the response into streaming events.
        """
        return stream is True

    def _extract_agent_profile(self, model: str) -> str:
        """
        Extract agent profile from model name.
        
        Model format: `manus/{agent_profile}`
        Examples: `manus/manus-1.6`, `manus/manus-1.6-lite`, `manus/manus-1.6-max`
        
        Returns:
            str: The agent profile (e.g., "manus-1.6")
        """
        if "/" in model:
            return model.split("/", 1)[1]
        # If no slash, assume the model name itself is the agent profile
        return model

    def validate_environment(
        self, headers: dict, model: str, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        """
        Validate environment and set up headers for Manus API.
        
        Manus uses `API_KEY` header instead of `Authorization: Bearer`.
        """
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key = (
            litellm_params.api_key
            or litellm.api_key
            or get_secret_str("MANUS_API_KEY")
        )
        
        if not api_key:
            raise ValueError(
                "Manus API key is required. Set MANUS_API_KEY environment variable or pass api_key parameter."
            )
        
        # Manus uses API_KEY header, not Authorization: Bearer
        # Content-Type is required for all requests (including GET)
        headers.update(
            {
                "API_KEY": api_key,
                "Content-Type": "application/json",
            }
        )
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Get the complete URL for Manus Responses API endpoint.
        
        Returns:
            str: The full URL for the Manus /v1/responses endpoint
        """
        api_base = (
            api_base
            or litellm.api_base
            or get_secret_str("MANUS_API_BASE")
            or MANUS_API_BASE
        )
        
        # Remove trailing slashes
        api_base = api_base.rstrip("/")
        
        # Manus API uses /v1/responses endpoint (OpenAI-compatible)
        if api_base.endswith("/v1"):
            return f"{api_base}/responses"
        return f"{api_base}/v1/responses"

    def transform_responses_api_request(
        self,
        model: str,
        input: Union[str, ResponseInputParam],
        response_api_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        """
        Transform the request for Manus API.
        
        Manus requires:
        - `task_mode: "agent"` in the request body
        - `agent_profile` extracted from model name in the request body
        """
        # First, get the base OpenAI request
        base_request = super().transform_responses_api_request(
            model=model,
            input=input,
            response_api_optional_request_params=response_api_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        
        # Extract agent profile from model name
        agent_profile = self._extract_agent_profile(model=model)
        
        # Add Manus-specific parameters directly to the request body
        # These will be sent as part of the request
        base_request["task_mode"] = "agent"
        base_request["agent_profile"] = agent_profile
        
        # Merge any existing extra_body into the request
        extra_body = response_api_optional_request_params.get("extra_body", {}) or {}
        if extra_body:
            base_request.update(extra_body)
        
        verbose_logger.debug(
            f"Manus: Using agent_profile={agent_profile}, task_mode=agent"
        )
        
        return base_request

    def transform_response_api_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIResponse:
        """
        Transform Manus API response to OpenAI-compatible format.
        
        Manus uses camelCase (createdAt) instead of snake_case (created_at).
        """
        try:
            logging_obj.post_call(
                original_response=raw_response.text,
                additional_args={"complete_input_dict": {}},
            )
            raw_response_json = raw_response.json()
            
            # Manus uses camelCase "createdAt" instead of snake_case "created_at"
            if "createdAt" in raw_response_json and "created_at" not in raw_response_json:
                raw_response_json["created_at"] = _safe_convert_created_field(
                    raw_response_json["createdAt"]
                )
            
            # Ensure created_at is set
            if "created_at" in raw_response_json:
                raw_response_json["created_at"] = _safe_convert_created_field(
                    raw_response_json["created_at"]
                )
        except Exception:
            raise OpenAIError(
                message=raw_response.text, status_code=raw_response.status_code
            )
        
        raw_response_headers = dict(raw_response.headers)
        processed_headers = process_response_headers(raw_response_headers)
        
        # Ensure reasoning is an empty dict if not present, OpenAI SDK does not allow None
        if "reasoning" not in raw_response_json or raw_response_json.get("reasoning") is None:
            raw_response_json["reasoning"] = {}
        
        if "text" not in raw_response_json or raw_response_json.get("text") is None:
            raw_response_json["text"] = {}
        
        if "output" not in raw_response_json or raw_response_json.get("output") is None:
            raw_response_json["output"] = []
        
        # Ensure usage is present with default values if not provided
        if "usage" not in raw_response_json or raw_response_json.get("usage") is None:
            raw_response_json["usage"] = ResponseAPIUsage(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
            )
        
        # Ensure id is present - failed responses may not include it
        if "id" not in raw_response_json or raw_response_json.get("id") is None:
            # Generate a placeholder id for failed responses
            # This allows the response object to be created even when the API doesn't return an id
            raw_response_json["id"] = f"unknown-{uuid.uuid4().hex[:8]}"
        
        try:
            response = ResponsesAPIResponse(**raw_response_json)
        except Exception:
            verbose_logger.debug(
                f"Error constructing ResponsesAPIResponse: {raw_response_json}, using model_construct"
            )
            response = ResponsesAPIResponse.model_construct(**raw_response_json)
        
        # Store processed headers in additional_headers so they get returned to the client
        response._hidden_params["additional_headers"] = processed_headers
        response._hidden_params["headers"] = raw_response_headers
        return response

    def transform_get_response_api_request(
        self,
        response_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform the get response API request into a URL and data.
        
        Manus API follows OpenAI-compatible format:
        - GET /v1/responses/{response_id}
        
        Reference: https://open.manus.im/docs/openai-compatibility
        """
        url = f"{api_base}/{response_id}"
        data: Dict = {}
        return url, data

    def transform_get_response_api_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIResponse:
        """
        Transform Manus API GET response to OpenAI-compatible format.
        
        Manus uses camelCase (createdAt) instead of snake_case (created_at).
        Same transformation as transform_response_api_response.
        """
        try:
            logging_obj.post_call(
                original_response=raw_response.text,
                additional_args={"complete_input_dict": {}},
            )
            raw_response_json = raw_response.json()
            
            # Manus uses camelCase "createdAt" instead of snake_case "created_at"
            if "createdAt" in raw_response_json and "created_at" not in raw_response_json:
                raw_response_json["created_at"] = _safe_convert_created_field(
                    raw_response_json["createdAt"]
                )
            
            # Ensure created_at is set
            if "created_at" in raw_response_json:
                raw_response_json["created_at"] = _safe_convert_created_field(
                    raw_response_json["created_at"]
                )
        except Exception:
            raise OpenAIError(
                message=raw_response.text, status_code=raw_response.status_code
            )
        
        raw_response_headers = dict(raw_response.headers)
        processed_headers = process_response_headers(raw_response_headers)
        
        # Ensure reasoning, text, output, and usage are present with defaults
        if "reasoning" not in raw_response_json or raw_response_json.get("reasoning") is None:
            raw_response_json["reasoning"] = {}
        
        if "text" not in raw_response_json or raw_response_json.get("text") is None:
            raw_response_json["text"] = {}
        
        if "output" not in raw_response_json or raw_response_json.get("output") is None:
            raw_response_json["output"] = []
        
        if "usage" not in raw_response_json or raw_response_json.get("usage") is None:
            raw_response_json["usage"] = ResponseAPIUsage(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
            )
        
        # Ensure id is present - failed responses may not include it
        if "id" not in raw_response_json or raw_response_json.get("id") is None:
            # Generate a placeholder id for failed responses
            raw_response_json["id"] = f"unknown-{uuid.uuid4().hex[:8]}"
        
        try:
            response = ResponsesAPIResponse(**raw_response_json)
        except Exception:
            verbose_logger.debug(
                f"Error constructing ResponsesAPIResponse: {raw_response_json}, using model_construct"
            )
            response = ResponsesAPIResponse.model_construct(**raw_response_json)
        
        # Store processed headers in additional_headers so they get returned to the client
        response._hidden_params["additional_headers"] = processed_headers
        response._hidden_params["headers"] = raw_response_headers
        return response

