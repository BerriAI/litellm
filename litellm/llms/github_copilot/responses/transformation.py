"""
GitHub Copilot Responses API Configuration.

This module provides the configuration for GitHub Copilot's Responses API,
which is required for models like gpt-5.1-codex that only support the /responses endpoint.

Implementation based on analysis of the copilot-api project by caozhiyuan:
https://github.com/caozhiyuan/copilot-api
"""
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

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
    GetAPIKeyError,
    GITHUB_COPILOT_API_BASE,
    get_copilot_default_headers,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

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
                    message="GitHub Copilot API key is required. Please authenticate via OAuth Device Flow.",
                )

            # Get default headers (from copilot-api configuration)
            default_headers = get_copilot_default_headers(api_key)

            # Merge with existing headers (user's extra_headers take priority)
            merged_headers = {**default_headers, **headers}

            # Analyze input to determine additional headers
            input_param = self._get_input_from_params(litellm_params)

            # Add X-Initiator header based on input analysis
            if input_param is not None:
                initiator = self._get_initiator(input_param)
                merged_headers["X-Initiator"] = initiator
                verbose_logger.debug(
                    f"GitHub Copilot Responses API: Set X-Initiator={initiator}"
                )

                # Add vision header if input contains images
                if self._has_vision_input(input_param):
                    merged_headers["copilot-vision-request"] = "true"
                    verbose_logger.debug(
                        "GitHub Copilot Responses API: Enabled vision request"
                    )

            verbose_logger.debug(
                f"GitHub Copilot Responses API: Successfully configured headers for model {model}"
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

        Returns: https://api.githubcopilot.com/responses

        Note: Currently only supports individual accounts.
        Business/enterprise accounts (api.business.githubcopilot.com) can be
        added in the future by detecting account type.
        """
        # Use provided api_base or fall back to authenticator's base or default
        api_base = (
            api_base
            or self.authenticator.get_api_base()
            or GITHUB_COPILOT_API_BASE
        )

        # Remove trailing slashes
        api_base = api_base.rstrip("/")

        # Return the responses endpoint
        return f"{api_base}/responses"

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
                f"GitHub Copilot reasoning item processed, encrypted_content preserved: {encrypted_content is not None}"
            )
            return filtered_item
        return item

    # ==================== Helper Methods ====================

    def _get_input_from_params(
        self, litellm_params: Optional[GenericLiteLLMParams]
    ) -> Optional[Union[str, ResponseInputParam]]:
        """
        Extract input parameter from litellm_params.

        The input parameter contains the conversation history and is needed
        for vision detection and initiator determination.
        """
        if litellm_params is None:
            return None

        # Try to get input from litellm_params
        # This might be in different locations depending on how LiteLLM structures it
        if hasattr(litellm_params, "input"):
            return litellm_params.input

        # If not found, return None and let the API handle it
        return None

    def _get_initiator(self, input_param: Union[str, ResponseInputParam]) -> str:
        """
        Determine X-Initiator header value based on input analysis.

        Based on copilot-api's hasAgentInitiator logic:
        - Returns "agent" if input contains assistant role or items without role
        - Returns "user" otherwise

        Args:
            input_param: The input parameter (string or list of input items)

        Returns:
            "agent" or "user"
        """
        # If input is a string, it's user-initiated
        if isinstance(input_param, str):
            return "user"

        # If input is a list, analyze items
        if isinstance(input_param, list):
            for item in input_param:
                if not isinstance(item, dict):
                    continue

                # Check if item has no role (agent-initiated)
                if "role" not in item or not item.get("role"):
                    return "agent"

                # Check if role is assistant (agent-initiated)
                role = item.get("role")
                if isinstance(role, str) and role.lower() == "assistant":
                    return "agent"

        # Default to user-initiated
        return "user"

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
                f"[GitHub Copilot] Max recursion depth {max_depth} reached while checking for vision content"
            )
            return False

        if value is None:
            return False

        # Check arrays
        if isinstance(value, list):
            return any(
                self._contains_vision_content(item, depth=depth + 1, max_depth=max_depth)
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
                self._contains_vision_content(item, depth=depth + 1, max_depth=max_depth)
                for item in value["content"]
            )

        return False
