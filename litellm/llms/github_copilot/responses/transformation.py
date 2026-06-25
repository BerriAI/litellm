"""
GitHub Copilot Responses API Configuration.

This module provides the configuration for GitHub Copilot's Responses API,
which is required for models like gpt-5.3-codex that only support the /responses endpoint.

Implementation based on analysis of the copilot-api project by caozhiyuan:
https://github.com/caozhiyuan/copilot-api
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import os

import litellm
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
from litellm.utils import _cached_get_model_info_helper

from ..authenticator import Authenticator
from ..common_utils import (
    DEFAULT_GITHUB_COPILOT_API_BASE,
    GetAPIKeyError,
    get_copilot_default_headers,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


def github_copilot_supports_responses_api(model: str) -> bool:
    """
    Gate native /v1/responses dispatch per github_copilot model.

    Resolution (first match wins): mode "responses" -> True; mode "chat" ->
    False (opt-out wins for dual-endpoint models); "/v1/responses" in
    supported_endpoints -> True; else False. Unknown model -> False (the bridge
    always works since every Copilot model supports /chat/completions).

    Reads merged model info (per-deployment model_info applied via the router's
    register_model, which also clears the cache used here).
    """
    try:
        info = _cached_get_model_info_helper(
            model=model, custom_llm_provider="github_copilot"
        )
    except Exception as e:
        verbose_logger.debug(
            "github_copilot_supports_responses_api: get_model_info failed for %s: %s",
            model,
            e,
        )
        return False

    mode = info.get("mode")
    if mode == "responses":
        return True
    if mode == "chat":
        return False

    # supported_endpoints is dropped by ModelInfoBase; read it from the raw
    # model_cost entry via the resolved key.
    key = info.get("key")
    raw_info = litellm.model_cost.get(key) if isinstance(key, str) else None
    endpoints = (
        raw_info.get("supported_endpoints") if isinstance(raw_info, dict) else None
    )
    return isinstance(endpoints, list) and "/v1/responses" in endpoints


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
        self._stream_item_ids_by_output_index: Dict[int, str] = {}

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

    def transform_streaming_response(
        self,
        model: str,
        parsed_chunk: dict,
        logging_obj: LiteLLMLoggingObj,
    ) -> Any:
        parsed_chunk = self._normalize_stream_item_id(parsed_chunk)
        return super().transform_streaming_response(
            model=model,
            parsed_chunk=parsed_chunk,
            logging_obj=logging_obj,
        )

    def _normalize_stream_item_id(self, parsed_chunk: dict) -> dict:
        """Rewrite streamed item ids to one stable id per output_index.

        GitHub Copilot tags each event of a single output item with a different
        item id, so clients that key streaming state by item id (e.g. the Vercel
        AI SDK) crash with "reasoning part <id> not found" / "text part <id> not
        found". Every sub-event carries a top-level ``item_id`` (whatever the
        item type), so its presence is the rewrite signal; output_item.added /
        .done instead nest the id under ``item``. The anchor is keyed by
        output_index and taken from output_item.added, which the protocol always
        emits first, so it is written before any sub-event reads it. Copilot
        accepts that id paired with the final encrypted_content next turn, so
        multi-turn replay is unaffected.

        State is keyed by output_index on this config, which
        ProviderConfigManager builds fresh per request, so it is stream-scoped.
        """
        output_index = parsed_chunk.get("output_index")
        if not isinstance(output_index, int):
            return parsed_chunk

        if parsed_chunk.get("type") == "response.output_item.added":
            item = parsed_chunk.get("item")
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                self._stream_item_ids_by_output_index[output_index] = item["id"]
            return parsed_chunk

        stable_id = self._stream_item_ids_by_output_index.get(output_index)
        if stable_id is None:
            return parsed_chunk

        if isinstance(parsed_chunk.get("item_id"), str):
            parsed_chunk = dict(parsed_chunk)
            parsed_chunk["item_id"] = stable_id
        elif parsed_chunk.get("type") == "response.output_item.done":
            item = parsed_chunk.get("item")
            if isinstance(item, dict):
                parsed_chunk = dict(parsed_chunk)
                parsed_chunk["item"] = {**item, "id": stable_id}

        return parsed_chunk

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
