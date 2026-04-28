"""
GitHub Copilot Responses API Configuration.

This module provides the configuration for GitHub Copilot's Responses API,
which is required for models like gpt-5.1-codex that only support the
/responses endpoint.

Implementation based on analysis of the copilot-api project by caozhiyuan:
https://github.com/caozhiyuan/copilot-api
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import os

from litellm._logging import verbose_logger
from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH
from litellm.exceptions import AuthenticationError
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.types.llms.openai import (
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

from ..authenticator import Authenticator
from ..common_utils import (
    DEFAULT_GITHUB_COPILOT_API_BASE,
    GetAPIKeyError,
    determine_x_initiator,
    get_copilot_default_headers,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as _LiteLLMLoggingObj,
    )

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class GithubCopilotResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """
    Configuration for GitHub Copilot's Responses API.

    Inherits from OpenAIResponsesAPIConfig since GitHub Copilot's Responses API
    is compatible with OpenAI's Responses API specification.

    Key differences from OpenAI:
    - Uses OAuth Device Flow authentication (handled by Authenticator)
    - Uses api.githubcopilot.com as the API base
    - Requires specific headers for VSCode/Copilot integration
    - Supports vision requests with special header
    - Requires X-Initiator header based on input analysis

    Reference: https://api.githubcopilot.com/
    """

    def __init__(self) -> None:
        super().__init__()
        self.authenticator = Authenticator()

    @property
    def custom_llm_provider(self) -> LlmProviders:
        """Return the GitHub Copilot provider identifier."""
        return LlmProviders.GITHUB_COPILOT

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get supported parameters for GitHub Copilot Responses API.

        GitHub Copilot supports all standard OpenAI Responses API parameters.
        """
        return super().get_supported_openai_params(model)

    def map_openai_params(
        self,
        response_api_optional_params: ResponsesAPIOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        """
        Map parameters for GitHub Copilot Responses API.

        GitHub Copilot uses the same parameter format as OpenAI,
        so no transformation is needed.
        """
        return dict(response_api_optional_params)

    def transform_responses_api_request(
        self,
        model: str,
        input: Union[str, ResponseInputParam],
        response_api_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        """
        Transform the Responses API request and set X-Initiator header.

        GitHub Copilot requires X-Initiator header based on input analysis.
        This is set here (not in validate_environment) because only this method
        has access to the input parameter.
        """
        # Add X-Initiator header based on input analysis
        initiator = self._get_initiator(input)
        headers["X-Initiator"] = initiator
        verbose_logger.debug(
            f"GitHub Copilot Responses API: Set X-Initiator={initiator}"
        )

        # Add vision header if input contains images
        if self._has_vision_input(input):
            headers["copilot-vision-request"] = "true"
            verbose_logger.debug(
                "GitHub Copilot Responses API: Enabled vision request"
            )

        # Call parent to get request body (validates input, handles reasoning items)
        return super().transform_responses_api_request(
            model=model,
            input=input,
            response_api_optional_request_params=response_api_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: Optional[GenericLiteLLMParams],
    ) -> dict:
        """
        Validate environment and set up headers for GitHub Copilot API.

        Uses the Authenticator to obtain GitHub Copilot API key via OAuth Device Flow,
        then configures all required headers for the Responses API.

        Headers include:
        - Authorization with API key
        - Standard GitHub Copilot headers (editor-version, user-agent, etc.)
        - X-Initiator based on input analysis
        - copilot-vision-request if vision content detected
        - User-provided extra_headers (merged with priority)
        """
        try:
            # Get GitHub Copilot API key via OAuth
            api_key = self.authenticator.get_api_key()

            if not api_key:
                raise AuthenticationError(
                    model=model,
                    llm_provider="github_copilot",
                    message=(
                        "GitHub Copilot API key is required. "
                        "Please authenticate via OAuth Device Flow."
                    ),
                )

            # Extract optional conversation key for session-scoped billing
            conversation_key: Optional[str] = None
            if litellm_params is not None:
                lp_metadata = (
                    litellm_params.get("metadata")
                    if isinstance(litellm_params, dict)
                    else getattr(litellm_params, "metadata", None)
                ) or {}
                conversation_key = lp_metadata.get("copilot_conversation_id")

            # Get default headers (from copilot-api configuration)
            default_headers = get_copilot_default_headers(
                api_key, conversation_key=conversation_key
            )

            # Merge with existing headers (user's extra_headers take priority)
            merged_headers = {**default_headers, **headers}

            # X-Initiator and vision headers are set in transform_responses_api_request
            # where we have access to the input parameter

            verbose_logger.debug(
                f"GitHub Copilot Responses API: Successfully configured headers"
                f" for model {model}"
            )

            return merged_headers

        except GetAPIKeyError as e:
            raise AuthenticationError(
                model=model,
                llm_provider="github_copilot",
                message=str(e),
            )

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Get the complete URL for GitHub Copilot Responses API endpoint.
        """
        # Use provided api_base or fall back to authenticator's base or default
        effective_api_base = (
            api_base
            or self.authenticator.get_api_base()
            or os.getenv("GITHUB_COPILOT_API_BASE")
            or DEFAULT_GITHUB_COPILOT_API_BASE
        )

        # Remove trailing slashes
        effective_api_base = effective_api_base.rstrip("/")

        # Return the responses endpoint
        return f"{effective_api_base}/responses"

    def _handle_reasoning_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle reasoning items for GitHub Copilot, preserving encrypted_content.

        GitHub Copilot uses encrypted_content in reasoning items to maintain
        conversation state across turns. The parent class strips this field
        when converting to OpenAI's ResponseReasoningItem model, which causes
        "encrypted content could not be verified" errors on multi-turn requests.

        This override preserves encrypted_content while still filtering out
        status=None which OpenAI's API rejects.
        """
        if item.get("type") == "reasoning":
            # Preserve encrypted_content before parent processing
            encrypted_content = item.get("encrypted_content")

            # Filter out None values for known problematic fields,
            # but preserve encrypted_content even if it exists
            filtered_item: Dict[str, Any] = {}
            for k, v in item.items():
                # Always include encrypted_content if present (even if None)
                if k == "encrypted_content":
                    if encrypted_content is not None:
                        filtered_item[k] = v
                    continue
                # Filter out status=None which OpenAI API rejects
                if k == "status" and v is None:
                    continue
                # Include all other non-None values
                if v is not None:
                    filtered_item[k] = v

            verbose_logger.debug(
                f"GitHub Copilot reasoning item processed, encrypted_content"
                f" preserved: {encrypted_content is not None}"
            )
            return filtered_item
        return item

    # ==================== Helper Methods ====================

    def _get_initiator(self, input_param: Union[str, ResponseInputParam]) -> str:
        """
        Determine X-Initiator header value based on input analysis.

        Delegates to the shared determine_x_initiator() helper in common_utils.py.
        See that function for full logic documentation (FIX-03).
        """
        return determine_x_initiator(input_param)

    def _has_vision_input(self, input_param: Union[str, ResponseInputParam]) -> bool:
        """
        Check if input contains vision content (images).

        Based on copilot-api's hasVisionInput and containsVisionContent logic.
        Recursively searches for input_image type in the input structure.

        Args:
            input_param: The input parameter to analyze

        Returns:
            True if input contains image content, False otherwise
        """
        return self._contains_vision_content(input_param)

    def _contains_vision_content(
        self, value: Any, depth: int = 0, max_depth: int = DEFAULT_MAX_RECURSE_DEPTH
    ) -> bool:
        """
        Recursively check if a value contains vision content.

        Looks for items with type="input_image" in the structure.
        """
        if depth > max_depth:
            verbose_logger.warning(
                f"[GitHub Copilot] Max recursion depth {max_depth} reached"
                f" while checking for vision content"
            )
            return False

        if value is None:
            return False

        # Check arrays
        if isinstance(value, list):
            return any(
                self._contains_vision_content(
                    item, depth=depth + 1, max_depth=max_depth
                )
                for item in value
            )

        # Only check dict/object types
        if not isinstance(value, dict):
            return False

        # Check if this item is an input_image
        item_type = value.get("type")
        if isinstance(item_type, str) and item_type.lower() == "input_image":
            return True

        # Check content field recursively
        if "content" in value and isinstance(value["content"], list):
            return any(
                self._contains_vision_content(
                    item, depth=depth + 1, max_depth=max_depth
                )
                for item in value["content"]
            )

        return False

    def supports_native_websocket(self) -> bool:
        """GitHub Copilot does not support native WebSocket for Responses API"""
        return False
