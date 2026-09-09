"""
GitHub Copilot Responses API Configuration.

This module provides the configuration for GitHub Copilot's Responses API,
which is required for models like gpt-5.3-codex that only support the /responses endpoint.

Implementation based on analysis of the copilot-api project by caozhiyuan:
https://github.com/caozhiyuan/copilot-api
"""

import copy
import os
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final

import litellm
from litellm._logging import verbose_logger
from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH
from litellm.exceptions import AuthenticationError
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.types.llms.openai import (
    ALL_RESPONSES_API_TOOL_PARAMS,
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
        info: Final = _cached_get_model_info_helper(model=model, custom_llm_provider="github_copilot")
    except Exception as e:
        verbose_logger.debug(
            "github_copilot_supports_responses_api: get_model_info failed for %s: %s",
            model,
            e,
        )
        return False

    mode: Final = info.get("mode")
    if mode == "responses":
        return True
    if mode == "chat":
        return False

    # supported_endpoints is dropped by ModelInfoBase; read it from the raw
    # model_cost entry via the resolved key.
    key: Final = info.get("key")
    raw_info: Final = litellm.model_cost.get(key) if isinstance(key, str) else None
    endpoints: Final = raw_info.get("supported_endpoints") if isinstance(raw_info, dict) else None
    return isinstance(endpoints, list) and "/v1/responses" in endpoints


def _is_valid_regex(pattern: object) -> bool:
    """Check whether a pattern compiles as a valid regular expression.

    GitHub Copilot validates JSON-Schema ``pattern`` fields strictly and rejects the
    request with HTTP 400 if a pattern uses unsupported regex dialect features (e.g.
    Unicode-property escapes such as \\p{Cc} in Claude Code's Artifact tool) or is
    invalid syntax. Non-string inputs are safely treated as invalid.
    """
    if not isinstance(pattern, str):
        return False
    try:
        re.compile(pattern)
    except (re.error, OverflowError, ValueError):
        return False
    else:
        return True


def _clean_schema_dict(
    current: dict[str, object],  # mutable-ok: schema dict mutated in-place
    stack: list[object],  # mutable-ok: iterative traversal worklist
) -> None:
    """Sanitize regexes in a single JSON schema dict and push child schemas to stack."""
    if "pattern" in current:
        pattern_val: Final = current["pattern"]
        if not _is_valid_regex(pattern_val):
            verbose_logger.debug(
                "GitHub Copilot Responses API: Stripping incompatible tool schema regex pattern: %r",
                pattern_val,
            )
            del current["pattern"]  # rebind-ok: in-place schema mutation

    if "patternProperties" in current and isinstance(current["patternProperties"], dict):
        pp: Final = current["patternProperties"]
        invalid_keys: Final = tuple(pk for pk in pp if not _is_valid_regex(pk))
        for ik in invalid_keys:
            verbose_logger.debug(
                "GitHub Copilot Responses API: Stripping incompatible patternProperties regex key: %r",
                ik,
            )
            del pp[ik]  # rebind-ok: in-place schema mutation
        stack.extend(sub for sub in pp.values() if isinstance(sub, (dict, list)))

    for kw in ("properties", "$defs", "definitions", "dependentSchemas"):
        mapping: Final = current.get(kw)
        if isinstance(mapping, dict):
            stack.extend(sub for sub in mapping.values() if isinstance(sub, (dict, list)))

    for key, val in current.items():
        if key not in (
            "properties",
            "$defs",
            "definitions",
            "dependentSchemas",
            "patternProperties",
            "pattern",
        ) and isinstance(val, (dict, list)):
            stack.append(val)


def _sanitize_json_schema_regex_patterns(schema: object) -> object:
    """Iteratively traverse a JSON schema and remove incompatible regex patterns.

    Differentiates between schema keywords (e.g., 'pattern', 'patternProperties')
    and user-defined identifier mappings (e.g., 'properties', '$defs', 'definitions',
    'dependentSchemas') so that parameters named 'pattern' are not stripped.
    """
    if not isinstance(schema, (dict, list)):
        return schema

    cleaned_schema: Final = copy.deepcopy(schema)
    stack: Final[list[object]] = [cleaned_schema]  # mutable-ok: worklist for iterative schema traversal
    seen_ids: Final[set[int]] = set()  # mutable-ok: cycle detection set for iterative traversal

    while stack:
        current = stack.pop()
        current_id: Final = id(current)
        if current_id in seen_ids:
            continue
        seen_ids.add(current_id)

        if isinstance(current, dict):
            _clean_schema_dict(current, stack)
        elif isinstance(current, list):
            stack.extend(item for item in current if isinstance(item, (dict, list)))

    return cleaned_schema


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
        self._stream_item_ids_by_output_index: dict[int, str] = {}

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
    ) -> dict:
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
        output_index: Final = parsed_chunk.get("output_index")
        if not isinstance(output_index, int):
            return parsed_chunk

        if parsed_chunk.get("type") == "response.output_item.added":
            item = parsed_chunk.get("item")
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                self._stream_item_ids_by_output_index[output_index] = item["id"]
            return parsed_chunk

        stable_id: Final = self._stream_item_ids_by_output_index.get(output_index)
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
        litellm_params: GenericLiteLLMParams | None,
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
            api_key: Final = self.authenticator.get_api_key()

            if not api_key:
                raise AuthenticationError(
                    model=model,
                    llm_provider="github_copilot",
                    message="GitHub Copilot API key is required. Please authenticate via OAuth Device Flow.",
                )

            # Get default headers (from copilot-api configuration)
            default_headers: Final = get_copilot_default_headers(api_key)

            # Merge with existing headers (user's extra_headers take priority)
            merged_headers: Final = {**default_headers, **headers}

            # Analyze input to determine additional headers
            input_param: Final = self._get_input_from_params(litellm_params)

            # Add X-Initiator header based on input analysis
            if input_param is not None:
                initiator: Final = self._get_initiator(input_param)
                merged_headers["X-Initiator"] = initiator
                verbose_logger.debug("GitHub Copilot Responses API: Set X-Initiator=%s", initiator)

                # Add vision header if input contains images
                if self._has_vision_input(input_param):
                    merged_headers["copilot-vision-request"] = "true"
                    verbose_logger.debug("GitHub Copilot Responses API: Enabled vision request")

            verbose_logger.debug("GitHub Copilot Responses API: Successfully configured headers for model %s", model)

            return merged_headers

        except GetAPIKeyError as e:
            raise AuthenticationError(
                model=model,
                llm_provider="github_copilot",
                message=str(e),
            )

    def get_complete_url(
        self,
        api_base: str | None,
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

    def _handle_reasoning_item(self, item: dict[str, Any]) -> dict[str, Any]:
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
            encrypted_content: Final = item.get("encrypted_content")

            # Filter out None values for known problematic fields,
            # but preserve encrypted_content even if it exists
            filtered_item: Final[dict[str, Any]] = {}
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
                "GitHub Copilot reasoning item processed, encrypted_content preserved: %s",
                encrypted_content is not None,
            )
            return filtered_item
        return item

    def _prepared_input_and_tools(
        self,
        model: str,
        input: str | ResponseInputParam,
        tools: Sequence[ALL_RESPONSES_API_TOOL_PARAMS] | None,
        litellm_params: GenericLiteLLMParams,
    ) -> tuple[str | ResponseInputParam, Sequence[ALL_RESPONSES_API_TOOL_PARAMS] | None]:
        """Prepare input and tools for GitHub Copilot Responses API.

        Overrides parent to sanitize regex patterns in tool schemas that GitHub Copilot's
        validator rejects with HTTP 400 invalid_request_body (e.g. Claude Code's Artifact tool
        using PCRE/Unicode-property escapes like \\p{Cc}).
        """
        replay_safe_input, sanitized_tools = super()._prepared_input_and_tools(
            model=model,
            input=input,
            tools=tools,
            litellm_params=litellm_params,
        )
        cleaned_tools: Final = self._sanitize_tool_regex_patterns(sanitized_tools)
        return replay_safe_input, cleaned_tools

    @staticmethod
    def _sanitize_tool_regex_patterns(
        tools: Sequence[ALL_RESPONSES_API_TOOL_PARAMS] | None,
    ) -> Sequence[ALL_RESPONSES_API_TOOL_PARAMS] | None:
        """Sanitize regex pattern fields across all tool parameters.

        Handles Responses API function tools ('parameters'), chat/completions format
        tools ('function.parameters'), Anthropic format tools ('input_schema'),
        and namespace tools ('tools').
        """
        if tools is None:
            return None
        cleaned_tools: Final[list[ALL_RESPONSES_API_TOOL_PARAMS]] = []  # mutable-ok: accumulating response tools
        for tool in tools:
            if isinstance(tool, dict):
                cleaned_tool: Final = dict(tool)  # mutable-ok: shallow copy of tool dict for mutation
                if "parameters" in cleaned_tool and isinstance(cleaned_tool["parameters"], dict):
                    cleaned_tool["parameters"] = _sanitize_json_schema_regex_patterns(cleaned_tool["parameters"])
                if "input_schema" in cleaned_tool and isinstance(cleaned_tool["input_schema"], dict):
                    cleaned_tool["input_schema"] = _sanitize_json_schema_regex_patterns(cleaned_tool["input_schema"])
                if "function" in cleaned_tool and isinstance(cleaned_tool["function"], dict):
                    cleaned_func: Final = dict(cleaned_tool["function"])  # mutable-ok: copy function dict
                    if "parameters" in cleaned_func and isinstance(cleaned_func["parameters"], dict):
                        cleaned_func["parameters"] = _sanitize_json_schema_regex_patterns(cleaned_func["parameters"])
                    cleaned_tool["function"] = cleaned_func
                if "tools" in cleaned_tool and isinstance(cleaned_tool["tools"], list):
                    cleaned_tool["tools"] = GithubCopilotResponsesAPIConfig._sanitize_tool_regex_patterns(
                        cleaned_tool["tools"]
                    )
                cleaned_tools.append(cleaned_tool)  # pyright: ignore[reportArgumentType]  # cloned dict conforms to ALL_RESPONSES_API_TOOL_PARAMS
            else:
                cleaned_tools.append(tool)
        return cleaned_tools

    # ==================== Helper Methods ====================

    def _get_input_from_params(self, litellm_params: GenericLiteLLMParams | None) -> str | ResponseInputParam | None:
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

    def _get_initiator(self, input_param: str | ResponseInputParam) -> str:
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

    def _has_vision_input(self, input_param: str | ResponseInputParam) -> bool:
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

    def _contains_vision_content(self, value: Any, depth: int = 0, max_depth: int = DEFAULT_MAX_RECURSE_DEPTH) -> bool:
        """
        Recursively check if a value contains vision content.

        Looks for items with type="input_image" in the structure.
        """
        if depth > max_depth:
            verbose_logger.warning(
                "[GitHub Copilot] Max recursion depth %s reached while checking for vision content", max_depth
            )
            return False

        if value is None:
            return False

        # Check arrays
        if isinstance(value, list):
            return any(self._contains_vision_content(item, depth=depth + 1, max_depth=max_depth) for item in value)

        # Only check dict/object types
        if not isinstance(value, dict):
            return False

        # Check if this item is an input_image
        item_type: Final = value.get("type")
        if isinstance(item_type, str) and item_type.lower() == "input_image":
            return True

        # Check content field recursively
        if "content" in value and isinstance(value["content"], list):
            return any(
                self._contains_vision_content(item, depth=depth + 1, max_depth=max_depth) for item in value["content"]
            )

        return False

    def supports_native_websocket(self) -> bool:
        """GitHub Copilot does not support native WebSocket for Responses API"""
        return False
