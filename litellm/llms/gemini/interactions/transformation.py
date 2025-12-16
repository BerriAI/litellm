"""
Google AI Studio Interactions API transformation.

This implements the Google Interactions API for Google AI Studio (Gemini).

API Endpoints:
- Create: POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
- Get: GET https://generativelanguage.googleapis.com/v1beta/interactions/{interaction_id}
- Delete: DELETE https://generativelanguage.googleapis.com/v1beta/interactions/{interaction_id}
- Cancel: POST https://generativelanguage.googleapis.com/v1beta/interactions/{interaction_id}:cancel
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import httpx

from litellm._logging import verbose_logger
from litellm.llms.base_llm.interactions.transformation import BaseInteractionsAPIConfig
from litellm.llms.gemini.common_utils import GeminiError, GeminiModelInfo
from litellm.types.interactions.main import (
    CancelInteractionResult,
    DeleteInteractionResult,
    InteractionCandidate,
    InteractionContent,
    InteractionInput,
    InteractionPart,
    InteractionPromptFeedback,
    InteractionSafetyRating,
    InteractionsAPIOptionalRequestParams,
    InteractionsAPIResponse,
    InteractionsAPIStreamingResponse,
    InteractionUsageMetadata,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.litellm_core_utils.core_helpers import process_response_headers

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class GoogleAIStudioInteractionsConfig(BaseInteractionsAPIConfig):
    """
    Configuration for Google AI Studio Interactions API.
    
    This maps to the Google AI generativelanguage API endpoints.
    """

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.GEMINI

    @property
    def api_version(self) -> str:
        return "v1beta"

    def get_supported_params(self, model: str) -> List[str]:
        """
        Return the list of supported parameters for Google AI Studio.
        """
        return [
            "contents",
            "generation_config",
            "safety_settings",
            "tools",
            "tool_config",
            "system_instruction",
            "cached_content",
            "stream",
        ]

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: Optional[GenericLiteLLMParams],
    ) -> dict:
        """
        Validate and prepare headers for Google AI Studio.
        
        Note: Google AI Studio uses API key in query params, not headers.
        """
        headers = headers or {}
        headers["Content-Type"] = "application/json"
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        model: str,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Get the complete URL for the interaction request.
        
        For Google AI Studio:
        - Streaming: POST /v1beta/models/{model}:streamGenerateContent
        - Non-streaming: POST /v1beta/models/{model}:generateContent
        """
        api_base = GeminiModelInfo.get_api_base(api_base)
        api_key = GeminiModelInfo.get_api_key(litellm_params.get("api_key"))
        
        if not api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable, "
                "or pass api_key in the request."
            )
        
        # Strip gemini/ prefix if present
        base_model = GeminiModelInfo.get_base_model(model) or model
        
        if stream:
            endpoint = f"/{self.api_version}/models/{base_model}:streamGenerateContent"
        else:
            endpoint = f"/{self.api_version}/models/{base_model}:generateContent"
        
        return f"{api_base}{endpoint}?key={api_key}"

    def _convert_input_to_contents(
        self, input: InteractionInput
    ) -> List[Dict[str, Any]]:
        """
        Convert the input to Google's contents format.
        
        Args:
            input: Can be a string, single content dict, or list of content dicts
            
        Returns:
            List of content dicts in Google's format
        """
        if isinstance(input, str):
            # Simple string input - wrap in user content
            return [
                {
                    "role": "user",
                    "parts": [{"text": input}]
                }
            ]
        elif isinstance(input, dict):
            # Single content dict
            return [input]
        elif isinstance(input, list):
            # List of content dicts
            return input
        else:
            raise ValueError(f"Invalid input type: {type(input)}")

    def transform_request(
        self,
        model: str,
        input: InteractionInput,
        optional_params: InteractionsAPIOptionalRequestParams,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        """
        Transform the request to Google AI's expected format.
        
        Google AI expects:
        {
            "contents": [...],
            "generationConfig": {...},
            "safetySettings": [...],
            "tools": [...],
            "toolConfig": {...},
            "systemInstruction": {...},
            "cachedContent": "..."
        }
        """
        request_body: Dict[str, Any] = {}
        
        # Convert input to contents
        request_body["contents"] = self._convert_input_to_contents(input)
        
        # Map generation_config
        if optional_params.get("generation_config"):
            gen_config = optional_params["generation_config"]
            request_body["generationConfig"] = {
                k: v for k, v in gen_config.items() if v is not None
            }
        
        # Map safety_settings
        if optional_params.get("safety_settings"):
            request_body["safetySettings"] = optional_params["safety_settings"]
        
        # Map tools
        if optional_params.get("tools"):
            request_body["tools"] = optional_params["tools"]
        
        # Map tool_config
        if optional_params.get("tool_config"):
            request_body["toolConfig"] = optional_params["tool_config"]
        
        # Map system_instruction
        if optional_params.get("system_instruction"):
            request_body["systemInstruction"] = optional_params["system_instruction"]
        
        # Map cached_content
        if optional_params.get("cached_content"):
            request_body["cachedContent"] = optional_params["cached_content"]
        
        return request_body

    def _parse_candidates(
        self, raw_candidates: Optional[List[Dict]]
    ) -> Optional[List[InteractionCandidate]]:
        """Parse candidates from raw response."""
        if not raw_candidates:
            return None
        
        candidates = []
        for raw_candidate in raw_candidates:
            content = None
            if raw_candidate.get("content"):
                raw_content = raw_candidate["content"]
                parts = None
                if raw_content.get("parts"):
                    parts = [
                        InteractionPart(**part) for part in raw_content["parts"]
                    ]
                content = InteractionContent(
                    role=raw_content.get("role"),
                    parts=parts,
                )
            
            safety_ratings = None
            if raw_candidate.get("safetyRatings"):
                safety_ratings = [
                    InteractionSafetyRating(**rating)
                    for rating in raw_candidate["safetyRatings"]
                ]
            
            candidate = InteractionCandidate(
                content=content,
                finish_reason=raw_candidate.get("finishReason"),
                safety_ratings=safety_ratings,
                token_count=raw_candidate.get("tokenCount"),
                index=raw_candidate.get("index"),
                avg_logprobs=raw_candidate.get("avgLogprobs"),
            )
            candidates.append(candidate)
        
        return candidates

    def _parse_usage_metadata(
        self, raw_usage: Optional[Dict]
    ) -> Optional[InteractionUsageMetadata]:
        """Parse usage metadata from raw response."""
        if not raw_usage:
            return None
        
        return InteractionUsageMetadata(
            prompt_token_count=raw_usage.get("promptTokenCount"),
            candidates_token_count=raw_usage.get("candidatesTokenCount"),
            total_token_count=raw_usage.get("totalTokenCount"),
            cached_content_token_count=raw_usage.get("cachedContentTokenCount"),
        )

    def _parse_prompt_feedback(
        self, raw_feedback: Optional[Dict]
    ) -> Optional[InteractionPromptFeedback]:
        """Parse prompt feedback from raw response."""
        if not raw_feedback:
            return None
        
        safety_ratings = None
        if raw_feedback.get("safetyRatings"):
            safety_ratings = [
                InteractionSafetyRating(**rating)
                for rating in raw_feedback["safetyRatings"]
            ]
        
        return InteractionPromptFeedback(
            block_reason=raw_feedback.get("blockReason"),
            safety_ratings=safety_ratings,
        )

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> InteractionsAPIResponse:
        """
        Transform the raw HTTP response into an InteractionsAPIResponse.
        """
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
        
        # Parse candidates
        candidates = self._parse_candidates(raw_json.get("candidates"))
        
        # Parse usage metadata
        usage_metadata = self._parse_usage_metadata(raw_json.get("usageMetadata"))
        
        # Parse prompt feedback
        prompt_feedback = self._parse_prompt_feedback(raw_json.get("promptFeedback"))
        
        # Extract interaction name if present
        name = raw_json.get("name")
        interaction_id = None
        if name and name.startswith("interactions/"):
            interaction_id = name.replace("interactions/", "")
        
        response = InteractionsAPIResponse(
            name=name,
            interaction_id=interaction_id,
            model=model,
            candidates=candidates,
            prompt_feedback=prompt_feedback,
            usage_metadata=usage_metadata,
            model_version=raw_json.get("modelVersion"),
        )
        
        # Add response headers to hidden params
        raw_headers = dict(raw_response.headers)
        processed_headers = process_response_headers(raw_headers)
        response._hidden_params["additional_headers"] = processed_headers
        response._hidden_params["headers"] = raw_headers
        
        return response

    def transform_streaming_response(
        self,
        model: str,
        parsed_chunk: dict,
        logging_obj: LiteLLMLoggingObj,
    ) -> InteractionsAPIStreamingResponse:
        """
        Transform a parsed streaming response chunk.
        """
        verbose_logger.debug("Google AI Interactions streaming chunk: %s", parsed_chunk)
        
        # Parse candidates
        candidates = self._parse_candidates(parsed_chunk.get("candidates"))
        
        # Parse usage metadata
        usage_metadata = self._parse_usage_metadata(parsed_chunk.get("usageMetadata"))
        
        # Parse prompt feedback
        prompt_feedback = self._parse_prompt_feedback(parsed_chunk.get("promptFeedback"))
        
        # Extract interaction name if present
        name = parsed_chunk.get("name")
        interaction_id = None
        if name and name.startswith("interactions/"):
            interaction_id = name.replace("interactions/", "")
        
        return InteractionsAPIStreamingResponse(
            name=name,
            interaction_id=interaction_id,
            model=model,
            candidates=candidates,
            prompt_feedback=prompt_feedback,
            usage_metadata=usage_metadata,
        )

    # =========================================================
    # GET INTERACTION
    # =========================================================
    
    def transform_get_interaction_request(
        self,
        interaction_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform the get interaction request.
        
        GET https://generativelanguage.googleapis.com/v1beta/interactions/{interaction_id}
        """
        api_base = GeminiModelInfo.get_api_base(api_base)
        api_key = GeminiModelInfo.get_api_key(litellm_params.api_key)
        
        if not api_key:
            raise ValueError("Google API key is required")
        
        url = f"{api_base}/{self.api_version}/interactions/{interaction_id}?key={api_key}"
        return url, {}

    def transform_get_interaction_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> InteractionsAPIResponse:
        """
        Transform the get interaction response.
        """
        try:
            raw_json = raw_response.json()
        except Exception:
            raise GeminiError(
                message=raw_response.text,
                status_code=raw_response.status_code,
                headers=dict(raw_response.headers),
            )
        
        # Extract interaction name and ID
        name = raw_json.get("name")
        interaction_id = None
        if name and name.startswith("interactions/"):
            interaction_id = name.replace("interactions/", "")
        
        response = InteractionsAPIResponse(
            name=name,
            interaction_id=interaction_id,
            state=raw_json.get("state"),
            create_time=raw_json.get("createTime"),
            update_time=raw_json.get("updateTime"),
            model=raw_json.get("model"),
        )
        
        raw_headers = dict(raw_response.headers)
        response._hidden_params["headers"] = raw_headers
        
        return response

    # =========================================================
    # DELETE INTERACTION
    # =========================================================
    
    def transform_delete_interaction_request(
        self,
        interaction_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform the delete interaction request.
        
        DELETE https://generativelanguage.googleapis.com/v1beta/interactions/{interaction_id}
        """
        api_base = GeminiModelInfo.get_api_base(api_base)
        api_key = GeminiModelInfo.get_api_key(litellm_params.api_key)
        
        if not api_key:
            raise ValueError("Google API key is required")
        
        url = f"{api_base}/{self.api_version}/interactions/{interaction_id}?key={api_key}"
        return url, {}

    def transform_delete_interaction_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        interaction_id: str,
    ) -> DeleteInteractionResult:
        """
        Transform the delete interaction response.
        
        Google AI returns empty response on successful delete.
        """
        # Check if response indicates success (2xx status)
        if raw_response.status_code >= 200 and raw_response.status_code < 300:
            return DeleteInteractionResult(
                success=True,
                interaction_id=interaction_id,
            )
        
        # Handle error case
        try:
            error_json = raw_response.json()
            error_message = error_json.get("error", {}).get("message", raw_response.text)
        except Exception:
            error_message = raw_response.text
        
        raise GeminiError(
            message=error_message,
            status_code=raw_response.status_code,
            headers=dict(raw_response.headers),
        )

    # =========================================================
    # CANCEL INTERACTION
    # =========================================================
    
    def transform_cancel_interaction_request(
        self,
        interaction_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform the cancel interaction request.
        
        POST https://generativelanguage.googleapis.com/v1beta/interactions/{interaction_id}:cancel
        """
        api_base = GeminiModelInfo.get_api_base(api_base)
        api_key = GeminiModelInfo.get_api_key(litellm_params.api_key)
        
        if not api_key:
            raise ValueError("Google API key is required")
        
        url = f"{api_base}/{self.api_version}/interactions/{interaction_id}:cancel?key={api_key}"
        return url, {}

    def transform_cancel_interaction_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> CancelInteractionResult:
        """
        Transform the cancel interaction response.
        """
        try:
            raw_json = raw_response.json()
        except Exception:
            raise GeminiError(
                message=raw_response.text,
                status_code=raw_response.status_code,
                headers=dict(raw_response.headers),
            )
        
        # Extract interaction name and ID
        name = raw_json.get("name")
        interaction_id = None
        if name and name.startswith("interactions/"):
            interaction_id = name.replace("interactions/", "")
        
        return CancelInteractionResult(
            name=name,
            interaction_id=interaction_id,
            state=raw_json.get("state"),
        )
