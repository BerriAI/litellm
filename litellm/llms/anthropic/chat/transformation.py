import json
import re
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    NoReturn,
    Optional,
    Tuple,
    Union,
    cast,
)

import httpx

import litellm
from litellm.constants import (
    ANTHROPIC_MIN_THINKING_BUDGET_TOKENS,
    ANTHROPIC_WEB_SEARCH_TOOL_MAX_USES,
    DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS,
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MAX_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET,
    RESPONSE_FORMAT_TOOL_NAME,
)
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.llms.base_llm.base_utils import type_to_response_format_param
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.types.llms.anthropic import (
    ANTHROPIC_ADVISOR_TOOL_TYPE,
    ANTHROPIC_BETA_HEADER_VALUES,
    ANTHROPIC_HOSTED_TOOLS,
    AllAnthropicMessageValues,
    AllAnthropicToolsValues,
    AnthropicCodeExecutionTool,
    AnthropicComputerTool,
    AnthropicHostedTools,
    AnthropicInputSchema,
    AnthropicMcpServerTool,
    AnthropicMessagesTool,
    AnthropicMessagesToolChoice,
    AnthropicOutputSchema,
    AnthropicSystemMessageContent,
    AnthropicThinkingParam,
    AnthropicWebSearchTool,
    AnthropicWebSearchUserLocation,
)
from litellm.types.llms.openai import (
    REASONING_EFFORT,
    AllMessageValues,
    ChatCompletionCachedContent,
    ChatCompletionRedactedThinkingBlock,
    ChatCompletionSystemMessage,
    ChatCompletionThinkingBlock,
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionToolParam,
    OpenAIChatCompletionFinishReason,
    OpenAIMcpServerTool,
    OpenAIWebSearchOptions,
)
from litellm.types.responses.main import (
    OutputCodeInterpreterCall,
    build_code_interpreter_log_outputs,
)
from litellm.types.utils import (
    CacheCreationTokenDetails,
    CompletionTokensDetailsWrapper,
)
from litellm.types.utils import Message as LitellmMessage
from litellm.types.utils import (
    PromptTokensDetailsWrapper,
    ServerToolUse,
)
from litellm.utils import (
    ModelResponse,
    Usage,
    _supports_factory,
    add_dummy_tool,
    any_assistant_message_has_thinking_blocks,
    get_max_tokens,
    has_tool_call_blocks,
    last_assistant_with_tool_calls_has_no_thinking_blocks,
    supports_reasoning,
    token_counter,
)

from ..common_utils import (
    AnthropicError,
    AnthropicModelInfo,
    process_anthropic_headers,
    strip_advisor_blocks_from_messages,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    LoggingClass = LiteLLMLoggingObj
else:
    LoggingClass = Any


# Anthropic requires tool names to match ^[a-zA-Z0-9_-]{1,128}$. Any other
# character (commonly '/' or '.' from OpenAPI-derived MCP tools, e.g.
# "actions/download-job-logs-for-workflow-run") must be replaced before
# the request is sent.
#
# A naive "replace [^a-zA-Z0-9_-] with _" is unsafe because it's lossy:
# `foo/bar` and `foo_bar` both collapse to `foo_bar`. Two tools with the
# same sanitized name would either 400 at Anthropic (duplicate) or, worse,
# cause the response side to mis-translate `foo_bar` (a name the caller
# really did register) back to `foo/bar`.
#
# Instead we build a *per-request* forward map (original -> sanitized)
# whose codomain is unique within the request: when two originals collapse
# to the same candidate, or when a sanitized name collides with an already-
# valid name elsewhere in the request, we append numeric suffixes
# (`_2`, `_3`, ...) until the result is free.
#
# The reverse map (sanitized -> original) only contains entries where the
# original was actually rewritten. So a tool whose name is already valid
# round-trips identically and is *never* mistakenly re-mapped on the
# response side.
_ANTHROPIC_TOOL_NAME_INVALID_CHARS = re.compile(r"[^a-zA-Z0-9_-]")
_ANTHROPIC_TOOL_NAME_MAX_LEN = 128
# Single, internal-only key on ``litellm_params`` used to thread the per-
# request reverse map (sanitized -> original) from request build to response
# parsing. ``litellm_params`` is never serialized to a provider; ``optional_
# params`` IS (it becomes the JSON body via ``data = {**optional_params}``).
# Keep these two channels strictly separate -- never stash internal
# coordination state in ``optional_params``.
ANTHROPIC_TOOL_NAME_REVERSE_MAP_KEY = "_anthropic_tool_name_map"


def _basic_sanitize_anthropic_tool_name(name: str) -> str:
    """Lossy: replace [^a-zA-Z0-9_-] with '_' and truncate to 128.

    Used as a candidate generator for the per-request forward map.
    Callers should NOT use this directly for translation -- always go
    through the forward map so collisions are resolved.
    """
    if not isinstance(name, str) or not name:
        return name
    return _ANTHROPIC_TOOL_NAME_INVALID_CHARS.sub("_", name)[
        :_ANTHROPIC_TOOL_NAME_MAX_LEN
    ]


def _build_anthropic_tool_name_maps(
    original_names: List[str],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build (forward, reverse) tool-name maps for a single request.

    forward[original] = sanitized   -- only present when name was rewritten
    reverse[sanitized] = original   -- inverse of `forward`

    Properties:
    - All sanitized names satisfy ^[a-zA-Z0-9_-]{1,128}$.
    - Sanitized names are unique within the request (no two originals
      collide on the wire).
    - A name that's already valid AND doesn't collide with another tool's
      sanitized form passes through untouched and is absent from the maps.
      That's the key correctness property: response-side translation only
      runs on entries we actually rewrote, so a tool legitimately named
      `foo_bar` is never incorrectly retyped to `foo/bar` just because
      some *other* request had that pair.
    - Order-dependent: when two originals would clash, the *second* one
      seen gets the disambiguating suffix. Callers should preserve the
      caller's tool order (we do).
    """
    forward: Dict[str, str] = {}
    used: set = set()

    # First pass: reserve slots for names that are already valid so they
    # always have priority regardless of input order.
    for original in original_names:
        if not isinstance(original, str) or not original:
            continue
        candidate = _basic_sanitize_anthropic_tool_name(original)
        if candidate == original:
            used.add(candidate)

    # Second pass: sanitize/disambiguate names that need rewriting.
    for original in original_names:
        if not isinstance(original, str) or not original:
            continue
        candidate = _basic_sanitize_anthropic_tool_name(original)
        if candidate == original:
            continue
        # Skip duplicates of the same original name. Without this guard the
        # second pass would assign a fresh suffix and overwrite the forward
        # map entry, causing every reference to map to the suffixed name and
        # leaving the original sanitized slot orphaned in `used` with no
        # reverse mapping.
        if original in forward:
            continue
        # Disambiguate against names already chosen this request.
        unique = candidate
        n = 1
        while unique in used:
            n += 1
            suffix = f"_{n}"
            # Keep within the 128-char cap.
            head = candidate[: _ANTHROPIC_TOOL_NAME_MAX_LEN - len(suffix)]
            unique = f"{head}{suffix}"
        forward[original] = unique
        used.add(unique)
    reverse = {v: k for k, v in forward.items()}
    return forward, reverse


REASONING_EFFORT_TO_OUTPUT_CONFIG_EFFORT: Dict[str, str] = {
    "low": "low",
    "minimal": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}

DROP_UNSUPPORTED_OUTPUT_CONFIG_WARNING = (
    "Dropping unsupported `output_config` for model=%s "
    "(drop_params=True). Effort is only supported on Opus 4.5+, "
    "Sonnet 4.6+, and Mythos Preview."
)


class AnthropicConfig(AnthropicModelInfo, BaseConfig):
    """
    Reference: https://docs.anthropic.com/claude/reference/messages_post

    to pass metadata to anthropic, it's {"user_id": "any-relevant-information"}
    """

    max_tokens: Optional[int] = None
    stop_sequences: Optional[list] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    top_k: Optional[int] = None
    metadata: Optional[dict] = None
    system: Optional[str] = None

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[list] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        top_k: Optional[int] = None,
        metadata: Optional[dict] = None,
        system: Optional[str] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "anthropic"

    @classmethod
    def get_config(cls, *, model: Optional[str] = None):
        config = super().get_config()

        # anthropic requires a default value for max_tokens
        if config.get("max_tokens") is None:
            config["max_tokens"] = cls.get_max_tokens_for_model(model)

        return config

    @staticmethod
    def get_max_tokens_for_model(model: Optional[str] = None) -> int:
        """
        Get the max output tokens for a given model.
        Falls back to DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS (configurable via env var) if model is not found.
        """
        if model is None:
            return DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS
        try:
            max_tokens = get_max_tokens(model)
            if max_tokens is None:
                return DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS
            return max_tokens
        except Exception:
            return DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS

    @staticmethod
    def convert_tool_use_to_openai_format(
        anthropic_tool_content: Dict[str, Any],
        index: int,
    ) -> ChatCompletionToolCallChunk:
        """
        Convert Anthropic tool_use format to OpenAI ChatCompletionToolCallChunk format.

        Args:
            anthropic_tool_content: Anthropic tool_use content block with format:
                {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
            index: The index of this tool call

        Returns:
            ChatCompletionToolCallChunk in OpenAI format
        """
        tool_call = ChatCompletionToolCallChunk(
            id=anthropic_tool_content["id"],
            type="function",
            function=ChatCompletionToolCallFunctionChunk(
                name=anthropic_tool_content["name"],
                arguments=json.dumps(anthropic_tool_content["input"]),
            ),
            index=index,
        )
        # Include caller information if present (for programmatic tool calling)
        if "caller" in anthropic_tool_content:
            tool_call["caller"] = cast(Dict[str, Any], anthropic_tool_content["caller"])  # type: ignore[typeddict-item]
        return tool_call

    @staticmethod
    def _is_opus_4_6_model(model: str) -> bool:
        """Check if the model is specifically Claude Opus 4.6."""
        model_lower = model.lower()
        return any(
            v in model_lower for v in ("opus-4-6", "opus_4_6", "opus-4.6", "opus_4.6")
        )

    @staticmethod
    def _is_opus_4_7_model(model: str) -> bool:
        """Check if the model is specifically Claude Opus 4.7."""
        model_lower = model.lower()
        return any(
            v in model_lower for v in ("opus-4-7", "opus_4_7", "opus-4.7", "opus_4.7")
        )

    @staticmethod
    def _supports_effort_level(model: str, level: str) -> bool:
        """Check ``supports_{level}_reasoning_effort`` in the model map.

        Strips bedrock/vertex prefixes so a provider-routed Claude still
        resolves to the Anthropic model-map entry.
        """
        key = f"supports_{level}_reasoning_effort"
        try:
            if _supports_factory(
                model=model,
                custom_llm_provider="anthropic",
                key=key,
            ):
                return True
        except Exception:
            pass
        candidates = [model]
        for prefix in (
            "bedrock/converse/",
            "bedrock/invoke/",
            "bedrock/",
            "vertex_ai/",
        ):
            if model.startswith(prefix):
                candidates.append(model[len(prefix) :])
        try:
            from litellm.llms.bedrock.common_utils import BedrockModelInfo

            base = BedrockModelInfo.get_base_model(model)
            if base:
                candidates.append(base)
                candidates.append(f"bedrock/{base}")
        except Exception:
            pass
        try:
            import litellm

            for cand in candidates:
                if cand in litellm.model_cost and (
                    litellm.model_cost[cand].get(key) is True
                ):
                    return True
        except Exception:
            pass
        return False

    @staticmethod
    def _validate_effort_for_model(model: str, effort: Optional[str]) -> Optional[str]:
        """Return ``None`` if ``effort`` is allowed on ``model``, else an error message."""
        if effort == "max" and not (
            AnthropicConfig._is_claude_4_6_model(model)
            or AnthropicConfig._is_claude_4_7_model(model)
            or AnthropicConfig._supports_effort_level(model, "max")
        ):
            return f"effort='max' is not supported by this model. Got model: {model}"
        if effort == "xhigh" and not AnthropicConfig._supports_effort_level(
            model, "xhigh"
        ):
            return f"effort='xhigh' is not supported by this model. Got model: {model}"
        return None

    @staticmethod
    def _model_supports_effort_param(model: str) -> bool:
        """Whether the model accepts ``output_config.effort`` at all."""
        return any(
            AnthropicConfig._supports_effort_level(model, level)
            for level in ("low", "minimal", "medium", "high", "xhigh", "max")
        )

    @staticmethod
    def _raise_invalid_reasoning_effort(
        model: str, value: Any, llm_provider: str
    ) -> NoReturn:
        """Raise a ``BadRequestError`` for an unrecognised ``reasoning_effort``.

        Args:
            model: The model id the request was routed to (surfaced in the error).
            value: The offending ``reasoning_effort`` value supplied by the caller.
            llm_provider: Provider tag for the raised exception (``"anthropic"``,
                ``"bedrock_converse"``, ``"databricks"``, ...).

        Raises:
            litellm.exceptions.BadRequestError: Always.
        """
        raise litellm.exceptions.BadRequestError(
            message=(
                f"Invalid reasoning_effort: {value!r}. "
                f"Must be one of: 'minimal', 'low', 'medium', "
                f"'high', 'xhigh', 'max', 'none'"
            ),
            model=model,
            llm_provider=llm_provider,
        )

    def get_supported_openai_params(self, model: str):
        params = [
            "stream",
            "stop",
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "tools",
            "tool_choice",
            "extra_headers",
            "parallel_tool_calls",
            "response_format",
            "user",
            "web_search_options",
            "speed",
            "context_management",
            "cache_control",
        ]

        if (
            "claude-3-7-sonnet" in model
            or AnthropicConfig._is_claude_4_6_model(model)
            or AnthropicConfig._is_claude_4_7_model(model)
            or supports_reasoning(
                model=model,
                custom_llm_provider=self.custom_llm_provider,
            )
        ):
            params.append("thinking")
            params.append("reasoning_effort")

        return params

    @staticmethod
    def filter_anthropic_output_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter out unsupported fields from JSON schema for Anthropic's output_format API.

        Anthropic's output_format doesn't support certain JSON schema properties:
        - maxItems/minItems: Not supported for array types
        - minimum/maximum: Not supported for numeric types
        - minLength/maxLength: Not supported for string types

        This mirrors the transformation done by the Anthropic Python SDK.
        See: https://platform.claude.com/docs/en/build-with-claude/structured-outputs#how-sdk-transformation-works

        The SDK approach:
        1. Remove unsupported constraints from schema
        2. Add constraint info to description (e.g., "Must be at least 100")
        3. Validate responses against original schema
        Args:
            schema: The JSON schema dictionary to filter

        Returns:
            A new dictionary with unsupported fields removed and descriptions updated

        Related issues:
        - https://github.com/BerriAI/litellm/issues/19444
        """
        if not isinstance(schema, dict):
            return schema

        # All numeric/string/array constraints not supported by Anthropic
        unsupported_fields = {
            "maxItems",
            "minItems",  # array constraints
            "minimum",
            "maximum",  # numeric constraints
            "exclusiveMinimum",
            "exclusiveMaximum",  # numeric constraints
            "minLength",
            "maxLength",  # string constraints
        }

        # Build description additions from removed constraints
        constraint_descriptions: list = []
        constraint_labels = {
            "minItems": "minimum number of items: {}",
            "maxItems": "maximum number of items: {}",
            "minimum": "minimum value: {}",
            "maximum": "maximum value: {}",
            "exclusiveMinimum": "exclusive minimum value: {}",
            "exclusiveMaximum": "exclusive maximum value: {}",
            "minLength": "minimum length: {}",
            "maxLength": "maximum length: {}",
        }
        for field in unsupported_fields:
            if field in schema:
                constraint_descriptions.append(
                    constraint_labels[field].format(schema[field])
                )

        result: Dict[str, Any] = {}

        # Update description with removed constraint info
        if constraint_descriptions:
            existing_desc = schema.get("description", "")
            constraint_note = "Note: " + ", ".join(constraint_descriptions) + "."
            if existing_desc:
                result["description"] = existing_desc + " " + constraint_note
            else:
                result["description"] = constraint_note

        for key, value in schema.items():
            if key in unsupported_fields:
                continue
            if key == "description" and "description" in result:
                # Already handled above
                continue

            if key == "properties" and isinstance(value, dict):
                result[key] = {
                    k: AnthropicConfig.filter_anthropic_output_schema(v)
                    for k, v in value.items()
                }
            elif key == "items" and isinstance(value, dict):
                result[key] = AnthropicConfig.filter_anthropic_output_schema(value)
            elif key == "$defs" and isinstance(value, dict):
                result[key] = {
                    k: AnthropicConfig.filter_anthropic_output_schema(v)
                    for k, v in value.items()
                }
            elif key == "anyOf" and isinstance(value, list):
                result[key] = [
                    AnthropicConfig.filter_anthropic_output_schema(item)
                    for item in value
                ]
            elif key == "allOf" and isinstance(value, list):
                result[key] = [
                    AnthropicConfig.filter_anthropic_output_schema(item)
                    for item in value
                ]
            elif key == "oneOf" and isinstance(value, list):
                result[key] = [
                    AnthropicConfig.filter_anthropic_output_schema(item)
                    for item in value
                ]
            else:
                result[key] = value

        # Anthropic requires additionalProperties=false for object schemas
        # See: https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs
        if result.get("type") == "object" and "additionalProperties" not in result:
            result["additionalProperties"] = False

        return result

    def get_json_schema_from_pydantic_object(
        self, response_format: Union[Any, Dict, None]
    ) -> Optional[dict]:
        return type_to_response_format_param(
            response_format, ref_template="/$defs/{model}"
        )  # Relevant issue: https://github.com/BerriAI/litellm/issues/7755

    def get_cache_control_headers(self) -> dict:
        # Anthropic no longer requires the prompt-caching beta header
        # Prompt caching now works automatically when cache_control is used in messages
        # Reference: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
        return {
            "anthropic-version": "2023-06-01",
        }

    def _map_tool_choice(
        self,
        tool_choice: Optional[str],
        parallel_tool_use: Optional[bool],
    ) -> Optional[AnthropicMessagesToolChoice]:
        _tool_choice: Optional[AnthropicMessagesToolChoice] = None
        if tool_choice == "auto":
            _tool_choice = AnthropicMessagesToolChoice(
                type="auto",
            )
        elif tool_choice == "required":
            _tool_choice = AnthropicMessagesToolChoice(type="any")
        elif tool_choice == "none":
            _tool_choice = AnthropicMessagesToolChoice(type="none")
        elif isinstance(tool_choice, dict):
            if "type" in tool_choice and "function" not in tool_choice:
                tool_type = tool_choice.get("type")
                if tool_type == "auto":
                    _tool_choice = AnthropicMessagesToolChoice(type="auto")
                elif tool_type == "required" or tool_type == "any":
                    _tool_choice = AnthropicMessagesToolChoice(type="any")
                elif tool_type == "none":
                    _tool_choice = AnthropicMessagesToolChoice(type="none")
            else:
                _tool_name = tool_choice.get("function", {}).get("name")
                if _tool_name is not None:
                    _tool_choice = AnthropicMessagesToolChoice(type="tool")
                    _tool_choice["name"] = _tool_name

        if parallel_tool_use is not None:
            # Anthropic uses 'disable_parallel_tool_use' flag to determine if parallel tool use is allowed
            # this is the inverse of the openai flag.
            if tool_choice == "none":
                pass
            elif _tool_choice is not None:
                _tool_choice["disable_parallel_tool_use"] = not parallel_tool_use
            else:  # use anthropic defaults and make sure to send the disable_parallel_tool_use flag
                _tool_choice = AnthropicMessagesToolChoice(
                    type="auto",
                    disable_parallel_tool_use=not parallel_tool_use,
                )
        return _tool_choice

    def _map_tool_helper(  # noqa: PLR0915
        self,
        tool: ChatCompletionToolParam,
    ) -> Tuple[Optional[AllAnthropicToolsValues], Optional[AnthropicMcpServerTool]]:
        returned_tool: Optional[AllAnthropicToolsValues] = None
        mcp_server: Optional[AnthropicMcpServerTool] = None

        if tool["type"] == "function" or tool["type"] == "custom":
            _input_schema: dict = tool["function"].get(
                "parameters",
                {
                    "type": "object",
                    "properties": {},
                },
            )

            # Anthropic requires input_schema.type to be "object". Normalize
            # schemas from external sources (MCP servers, OpenAI callers) that
            # may omit the type field or use a non-object type.
            if _input_schema.get("type") != "object":
                litellm.verbose_logger.debug(
                    "_map_tool_helper: coercing input_schema type from %r to "
                    "'object' for Anthropic compatibility (tool: %s)",
                    _input_schema.get("type"),
                    tool["function"].get("name"),
                )
                _input_schema = dict(_input_schema)  # avoid mutating caller's dict
                _input_schema["type"] = "object"
                if "properties" not in _input_schema:
                    _input_schema["properties"] = {}

            _allowed_properties = set(AnthropicInputSchema.__annotations__.keys())
            input_schema_filtered = {
                k: v for k, v in _input_schema.items() if k in _allowed_properties
            }
            input_anthropic_schema: AnthropicInputSchema = AnthropicInputSchema(
                **input_schema_filtered
            )

            _tool = AnthropicMessagesTool(
                name=tool["function"]["name"],
                input_schema=input_anthropic_schema,
                type="custom",
            )

            _description = tool["function"].get("description")
            if _description is not None:
                _tool["description"] = _description

            returned_tool = _tool

        elif tool["type"].startswith("computer_"):
            ## check if all required 'display_' params are given
            if "parameters" not in tool["function"]:
                raise ValueError("Missing required parameter: parameters")

            _display_width_px: Optional[int] = tool["function"]["parameters"].get(
                "display_width_px"
            )
            _display_height_px: Optional[int] = tool["function"]["parameters"].get(
                "display_height_px"
            )
            if _display_width_px is None or _display_height_px is None:
                raise ValueError(
                    "Missing required parameter: display_width_px or display_height_px"
                )

            _computer_tool = AnthropicComputerTool(
                type=tool["type"],
                name=tool["function"].get("name", "computer"),
                display_width_px=_display_width_px,
                display_height_px=_display_height_px,
            )

            _display_number = tool["function"]["parameters"].get("display_number")
            if _display_number is not None:
                _computer_tool["display_number"] = _display_number

            returned_tool = _computer_tool
        elif any(tool["type"].startswith(t) for t in ANTHROPIC_HOSTED_TOOLS):
            function_name_obj = tool.get("name", tool.get("function", {}).get("name"))
            if function_name_obj is None or not isinstance(function_name_obj, str):
                raise ValueError("Missing required parameter: name")
            function_name = function_name_obj

            additional_tool_params = {}
            for k, v in tool.items():
                if k != "type" and k != "name":
                    additional_tool_params[k] = v

            returned_tool = AnthropicHostedTools(
                type=tool["type"], name=function_name, **additional_tool_params  # type: ignore
            )
        elif tool["type"] == "url":  # mcp server tool
            mcp_server = AnthropicMcpServerTool(**tool)  # type: ignore
        elif tool["type"] == "mcp":
            mcp_server = self._map_openai_mcp_server_tool(
                cast(OpenAIMcpServerTool, tool)
            )
        elif tool["type"] == "tool_search_tool_regex_20251119":
            # Tool search tool using regex
            from litellm.types.llms.anthropic import AnthropicToolSearchToolRegex

            tool_name_obj = tool.get("name", "tool_search_tool_regex")
            if not isinstance(tool_name_obj, str):
                raise ValueError("Tool search tool must have a valid name")
            tool_name = tool_name_obj
            returned_tool = AnthropicToolSearchToolRegex(
                type="tool_search_tool_regex_20251119",
                name=tool_name,
            )
        elif tool["type"] == "tool_search_tool_bm25_20251119":
            # Tool search tool using BM25
            from litellm.types.llms.anthropic import AnthropicToolSearchToolBM25

            tool_name_obj = tool.get("name", "tool_search_tool_bm25")
            if not isinstance(tool_name_obj, str):
                raise ValueError("Tool search tool must have a valid name")
            tool_name = tool_name_obj
            returned_tool = AnthropicToolSearchToolBM25(
                type="tool_search_tool_bm25_20251119",
                name=tool_name,
            )
        elif tool["type"] == ANTHROPIC_ADVISOR_TOOL_TYPE:
            from litellm.types.llms.anthropic import AnthropicAdvisorTool

            _tool_dict = cast(dict, tool)
            advisor_model = _tool_dict.get("model")
            if not isinstance(advisor_model, str):
                raise ValueError("Advisor tool must have a valid model")
            _advisor_tool = AnthropicAdvisorTool(
                type=ANTHROPIC_ADVISOR_TOOL_TYPE,
                name="advisor",
                model=advisor_model,
            )
            if _tool_dict.get("max_uses") is not None:
                _advisor_tool["max_uses"] = _tool_dict["max_uses"]
            if _tool_dict.get("caching") is not None:
                _advisor_tool["caching"] = _tool_dict["caching"]
            returned_tool = _advisor_tool  # type: ignore[assignment]
        if returned_tool is None and mcp_server is None:
            raise ValueError(f"Unsupported tool type: {tool['type']}")

        ## check if cache_control is set in the tool
        _cache_control = tool.get("cache_control", None)
        _cache_control_function = tool.get("function", {}).get("cache_control", None)
        if returned_tool is not None:
            # Only set cache_control on tools that support it (not tool search tools)
            tool_type = returned_tool.get("type", "")
            if tool_type not in (
                "tool_search_tool_regex_20251119",
                "tool_search_tool_bm25_20251119",
            ):
                if _cache_control is not None:
                    returned_tool["cache_control"] = _cache_control  # type: ignore[typeddict-item]
                elif _cache_control_function is not None and isinstance(
                    _cache_control_function, dict
                ):
                    returned_tool["cache_control"] = ChatCompletionCachedContent(  # type: ignore[typeddict-item]
                        **_cache_control_function  # type: ignore
                    )

        ## check if defer_loading is set in the tool
        _defer_loading = tool.get("defer_loading", None)
        _defer_loading_function = tool.get("function", {}).get("defer_loading", None)
        if returned_tool is not None:
            # Only set defer_loading on tools that support it (not tool search tools or computer tools)
            tool_type = returned_tool.get("type", "")
            if tool_type not in (
                "tool_search_tool_regex_20251119",
                "tool_search_tool_bm25_20251119",
                "computer_20241022",
                "computer_20250124",
            ):
                if _defer_loading is not None:
                    if not isinstance(_defer_loading, bool):
                        raise ValueError("defer_loading must be a boolean")
                    returned_tool["defer_loading"] = _defer_loading  # type: ignore[typeddict-item]
                elif _defer_loading_function is not None:
                    if not isinstance(_defer_loading_function, bool):
                        raise ValueError("defer_loading must be a boolean")
                    returned_tool["defer_loading"] = _defer_loading_function  # type: ignore[typeddict-item]

        ## check if allowed_callers is set in the tool
        _allowed_callers = tool.get("allowed_callers", None)
        _allowed_callers_function = tool.get("function", {}).get(
            "allowed_callers", None
        )
        if returned_tool is not None:
            # Only set allowed_callers on tools that support it (not tool search tools or computer tools)
            tool_type = returned_tool.get("type", "")
            if tool_type not in (
                "tool_search_tool_regex_20251119",
                "tool_search_tool_bm25_20251119",
                "computer_20241022",
                "computer_20250124",
            ):
                if _allowed_callers is not None:
                    if not isinstance(_allowed_callers, list) or not all(
                        isinstance(item, str) for item in _allowed_callers
                    ):
                        raise ValueError("allowed_callers must be a list of strings")
                    returned_tool["allowed_callers"] = _allowed_callers  # type: ignore[typeddict-item]
                elif _allowed_callers_function is not None:
                    if not isinstance(_allowed_callers_function, list) or not all(
                        isinstance(item, str) for item in _allowed_callers_function
                    ):
                        raise ValueError("allowed_callers must be a list of strings")
                    returned_tool["allowed_callers"] = _allowed_callers_function  # type: ignore[typeddict-item]

        ## check if input_examples is set in the tool
        _input_examples = tool.get("input_examples", None)
        _input_examples_function = tool.get("function", {}).get("input_examples", None)
        if returned_tool is not None:
            # Only set input_examples on user-defined tools (type "custom" or no type)
            tool_type = returned_tool.get("type", "")
            if tool_type == "custom" or (tool_type == "" and "name" in returned_tool):
                if _input_examples is not None and isinstance(_input_examples, list):
                    returned_tool["input_examples"] = _input_examples  # type: ignore[typeddict-item]
                elif _input_examples_function is not None and isinstance(
                    _input_examples_function, list
                ):
                    returned_tool["input_examples"] = _input_examples_function  # type: ignore[typeddict-item]

        return returned_tool, mcp_server

    def _map_openai_mcp_server_tool(
        self, tool: OpenAIMcpServerTool
    ) -> AnthropicMcpServerTool:
        from litellm.types.llms.anthropic import AnthropicMcpServerToolConfiguration

        allowed_tools = tool.get("allowed_tools", None)
        tool_configuration: Optional[AnthropicMcpServerToolConfiguration] = None
        if allowed_tools is not None:
            tool_configuration = AnthropicMcpServerToolConfiguration(
                allowed_tools=tool.get("allowed_tools", None),
            )

        headers = tool.get("headers", {})
        authorization_token: Optional[str] = None
        if headers is not None:
            bearer_token = headers.get("Authorization", None)
            if bearer_token is not None:
                authorization_token = bearer_token.replace("Bearer ", "")

        initial_tool = AnthropicMcpServerTool(
            type="url",
            url=tool["server_url"],
            name=tool["server_label"],
        )

        if tool_configuration is not None:
            initial_tool["tool_configuration"] = tool_configuration
        if authorization_token is not None:
            initial_tool["authorization_token"] = authorization_token
        return initial_tool

    def _map_tools(
        self,
        tools: List,
    ) -> Tuple[List[AllAnthropicToolsValues], List[AnthropicMcpServerTool]]:
        anthropic_tools = []
        mcp_servers = []
        for tool in tools:
            if "input_schema" in tool:  # assume in anthropic format
                anthropic_tools.append(tool)
            else:  # assume openai tool call
                new_tool, mcp_server_tool = self._map_tool_helper(tool)

                if new_tool is not None:
                    anthropic_tools.append(new_tool)
                if mcp_server_tool is not None:
                    mcp_servers.append(mcp_server_tool)
        return anthropic_tools, mcp_servers

    @staticmethod
    def _rewrite_tool_names_in_messages(
        messages: List[AllMessageValues],
        name_forward_map: Dict[str, str],
    ) -> List[AllMessageValues]:
        """Return a copy of `messages` with tool_call/function_call names
        rewritten using the per-request forward map.

        Only mutates messages whose tool_call/function_call name is *in* the
        forward map. Names absent from the map (already valid, no collision)
        round-trip untouched. We only deep-copy the entries we actually
        change to keep this O(turns-with-rewritten-tools), not O(history).
        """
        if not name_forward_map:
            return messages
        new_messages: List[AllMessageValues] = []
        for msg in messages:
            if not isinstance(msg, dict):
                new_messages.append(msg)
                continue
            tool_calls = msg.get("tool_calls")
            function_call = msg.get("function_call")
            if not tool_calls and not function_call:
                new_messages.append(msg)
                continue
            new_msg = dict(msg)
            if isinstance(tool_calls, list):
                new_calls = []
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        new_calls.append(tc)
                        continue
                    fn = tc.get("function")
                    fn_name = fn.get("name") if isinstance(fn, dict) else None
                    if (
                        isinstance(fn, dict)
                        and isinstance(fn_name, str)
                        and fn_name in name_forward_map
                    ):
                        new_fn = dict(fn)
                        new_fn["name"] = name_forward_map[fn_name]
                        new_tc = dict(tc)
                        new_tc["function"] = new_fn
                        new_calls.append(new_tc)
                    else:
                        new_calls.append(tc)
                new_msg["tool_calls"] = new_calls
            fc_name = (
                function_call.get("name") if isinstance(function_call, dict) else None
            )
            if (
                isinstance(function_call, dict)
                and isinstance(fc_name, str)
                and fc_name in name_forward_map
            ):
                new_fc = dict(function_call)
                new_fc["name"] = name_forward_map[fc_name]
                new_msg["function_call"] = new_fc
            new_messages.append(cast(AllMessageValues, new_msg))
        return new_messages

    @staticmethod
    def _build_request_tool_name_maps(
        tools: List,
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Build the (forward, reverse) tool-name maps for an OpenAI tools list.

        Operates on **OpenAI-format** tool dicts (pre-``_map_tools``). The
        production sanitization path uses ``_sanitize_tool_names_in_request``
        instead, which operates on **Anthropic-format** tools (post-
        ``_map_tools``, where ``type == "custom"``). This helper exists for
        callers that need to compute the maps from the raw OpenAI shape --
        e.g. test setup or future pre-mapping consumers.

        See _build_anthropic_tool_name_maps for the collision rules. Pulls
        the original name out of either ``{"function": {"name": ...}}``
        (legacy OpenAI shape) or ``{"name": ...}`` (rare top-level shape).
        """
        original_names: List[str] = []
        for tool in tools or []:
            if not isinstance(tool, dict):
                continue
            original = (
                tool.get("function", {}).get("name")
                if isinstance(tool.get("function"), dict)
                else None
            )
            if original is None:
                original = tool.get("name")
            if isinstance(original, str) and original:
                original_names.append(original)
        return _build_anthropic_tool_name_maps(original_names)

    @staticmethod
    def _sanitize_tool_names_in_request(
        optional_params: Dict[str, Any],
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Sanitize ``optional_params['tools']`` and ``optional_params['tool_choice']``
        in place so every name matches Anthropic's ``^[a-zA-Z0-9_-]{1,128}$``.

        Returns ``(forward, reverse)`` for use by message-history rewriting
        and response translation. ``forward[original] = sanitized`` is only
        populated for names that were actually rewritten -- i.e. either
        contained an invalid character or collided with another tool's
        sanitized form. Names already valid AND unique pass through and are
        absent from both maps.

        Only ``type == "custom"`` tools (the OpenAI function-tool shape) are
        considered. Hosted tools (``web_search``, ``bash``, ``code_execution``,
        ``computer_*``, ``mcp``, ...) own reserved names defined by Anthropic
        and must not be touched.
        """
        tools = optional_params.get("tools")
        if not isinstance(tools, list) or not tools:
            return {}, {}

        # 1. Collect originals from the Anthropic-shaped custom-tool entries.
        #    Order matters: the first occurrence wins the canonical slot;
        #    later collisions get numeric suffixes (see
        #    ``_build_anthropic_tool_name_maps``).
        original_names: List[str] = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            if t.get("type") != "custom":
                continue
            name = t.get("name")
            if isinstance(name, str) and name:
                original_names.append(name)

        if not original_names:
            return {}, {}

        forward, reverse = _build_anthropic_tool_name_maps(original_names)
        if not forward:
            # Every name was already valid -- nothing to do.
            return forward, reverse

        # 2. Apply forward map. Build a new list with copy-on-change entries
        #    so a caller reusing the same tool list/dicts across requests
        #    doesn't see its inputs permanently rewritten (which would also
        #    drop the original key from `forward` on the next request).
        new_tools: List[Any] = []
        for t in tools:
            if (
                isinstance(t, dict)
                and t.get("type") == "custom"
                and isinstance(t.get("name"), str)
                and t["name"] in forward
            ):
                new_tools.append({**t, "name": forward[t["name"]]})
            else:
                new_tools.append(t)
        optional_params["tools"] = new_tools

        # 3. Same for ``tool_choice`` when it targets a named tool. Copy
        #    rather than mutate for the same reason as above.
        tool_choice = optional_params.get("tool_choice")
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "tool":
            tc_name = tool_choice.get("name")
            if isinstance(tc_name, str) and tc_name in forward:
                optional_params["tool_choice"] = {
                    **tool_choice,
                    "name": forward[tc_name],
                }

        return forward, reverse

    def _detect_tool_search_tools(self, tools: Optional[List]) -> bool:
        """Check if tool search tools are present in the tools list."""
        if not tools:
            return False

        for tool in tools:
            tool_type = tool.get("type", "")
            if tool_type in [
                "tool_search_tool_regex_20251119",
                "tool_search_tool_bm25_20251119",
            ]:
                return True
        return False

    def _separate_deferred_tools(self, tools: List) -> Tuple[List, List]:
        """
        Separate tools into deferred and non-deferred lists.

        Returns:
            Tuple of (non_deferred_tools, deferred_tools)
        """
        non_deferred = []
        deferred = []

        for tool in tools:
            if tool.get("defer_loading", False):
                deferred.append(tool)
            else:
                non_deferred.append(tool)

        return non_deferred, deferred

    def _expand_tool_references(
        self,
        content: List,
        deferred_tools: List,
    ) -> List:
        """
        Expand tool_reference blocks to full tool definitions.

        When Anthropic's tool search returns results, it includes tool_reference blocks
        that reference tools by name. This method expands those references to full
        tool definitions from the deferred_tools catalog.

        Args:
            content: Response content that may contain tool_reference blocks
            deferred_tools: List of deferred tools that can be referenced

        Returns:
            Content with tool_reference blocks expanded to full tool definitions
        """
        if not deferred_tools:
            return content

        # Create a mapping of tool names to tool definitions
        tool_map = {}
        for tool in deferred_tools:
            tool_name = tool.get("name") or tool.get("function", {}).get("name")
            if tool_name:
                tool_map[tool_name] = tool

        # Expand tool references in content
        expanded_content = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "tool_reference":
                tool_name = item.get("tool_name")
                if tool_name and tool_name in tool_map:
                    # Replace reference with full tool definition
                    expanded_content.append(tool_map[tool_name])
                else:
                    # Keep the reference if we can't find the tool
                    expanded_content.append(item)
            else:
                expanded_content.append(item)

        return expanded_content

    def _map_stop_sequences(
        self, stop: Optional[Union[str, List[str]]]
    ) -> Optional[List[str]]:
        new_stop: Optional[List[str]] = None
        if isinstance(stop, str):
            if (
                stop.isspace() and litellm.drop_params is True
            ):  # anthropic doesn't allow whitespace characters as stop-sequences
                return new_stop
            new_stop = [stop]
        elif isinstance(stop, list):
            new_v = []
            for v in stop:
                if (
                    v.isspace() and litellm.drop_params is True
                ):  # anthropic doesn't allow whitespace characters as stop-sequences
                    continue
                new_v.append(v)
            if len(new_v) > 0:
                new_stop = new_v
        return new_stop

    @staticmethod
    def _map_reasoning_effort(
        reasoning_effort: Optional[Union[REASONING_EFFORT, str]],
        model: str,
        llm_provider: str = "anthropic",
    ) -> Optional[AnthropicThinkingParam]:
        if reasoning_effort is None or reasoning_effort == "none":
            return None
        if AnthropicConfig._is_adaptive_thinking_model(model):
            return AnthropicThinkingParam(
                type="adaptive",
            )
        elif reasoning_effort == "low":
            return AnthropicThinkingParam(
                type="enabled",
                budget_tokens=DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
            )
        elif reasoning_effort == "medium":
            return AnthropicThinkingParam(
                type="enabled",
                budget_tokens=DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
            )
        elif reasoning_effort == "high":
            return AnthropicThinkingParam(
                type="enabled",
                budget_tokens=DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
            )
        elif reasoning_effort == "xhigh":
            return AnthropicThinkingParam(
                type="enabled",
                budget_tokens=DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET,
            )
        elif reasoning_effort == "max":
            return AnthropicThinkingParam(
                type="enabled",
                budget_tokens=DEFAULT_REASONING_EFFORT_MAX_THINKING_BUDGET,
            )
        elif reasoning_effort == "minimal":
            return AnthropicThinkingParam(
                type="enabled",
                budget_tokens=max(
                    DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET,
                    ANTHROPIC_MIN_THINKING_BUDGET_TOKENS,
                ),
            )
        else:
            raise litellm.exceptions.BadRequestError(
                message=(
                    f"Unmapped reasoning effort: {reasoning_effort!r}. "
                    f"Must be one of: 'minimal', 'low', 'medium', 'high', "
                    f"'xhigh', 'max', 'none'."
                ),
                model=model,
                llm_provider=llm_provider,
            )

    def _extract_json_schema_from_response_format(
        self, value: Optional[dict]
    ) -> Optional[dict]:
        if value is None:
            return None
        json_schema: Optional[dict] = None
        if "response_schema" in value:
            json_schema = value["response_schema"]
        elif "json_schema" in value:
            json_schema = value["json_schema"]["schema"]

        return json_schema

    def map_response_format_to_anthropic_output_format(
        self, value: Optional[dict]
    ) -> Optional[AnthropicOutputSchema]:
        json_schema: Optional[dict] = self._extract_json_schema_from_response_format(
            value
        )
        if json_schema is None:
            return None

        # Resolve $ref/$defs before filtering — Anthropic doesn't support
        # external schema references (e.g., /$defs/CalendarEvent).
        import copy

        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            unpack_defs,
        )

        json_schema = copy.deepcopy(json_schema)
        defs = json_schema.pop("$defs", json_schema.pop("definitions", {}))
        if defs:
            unpack_defs(json_schema, defs)

        # Filter out unsupported fields for Anthropic's output_format API
        filtered_schema = self.filter_anthropic_output_schema(json_schema)

        return AnthropicOutputSchema(
            type="json_schema",
            schema=filtered_schema,
        )

    def map_response_format_to_anthropic_tool(
        self, value: Optional[dict], optional_params: dict, is_thinking_enabled: bool
    ) -> Optional[AnthropicMessagesTool]:
        ignore_response_format_types = ["text"]
        if (
            value is None or value["type"] in ignore_response_format_types
        ):  # value is a no-op
            return None

        json_schema: Optional[dict] = self._extract_json_schema_from_response_format(
            value
        )
        if json_schema is None:
            return None
        """
        When using tools in this way: - https://docs.anthropic.com/en/docs/build-with-claude/tool-use#json-mode
        - You usually want to provide a single tool
        - You should set tool_choice (see Forcing tool use) to instruct the model to explicitly use that tool
        - Remember that the model will pass the input to the tool, so the name of the tool and description should be from the model’s perspective.
        """

        _tool = self._create_json_tool_call_for_response_format(
            json_schema=json_schema,
        )

        return _tool

    def map_web_search_tool(
        self,
        value: OpenAIWebSearchOptions,
    ) -> AnthropicWebSearchTool:
        value_typed = cast(OpenAIWebSearchOptions, value)
        hosted_web_search_tool = AnthropicWebSearchTool(
            type="web_search_20250305",
            name="web_search",
        )
        user_location = value_typed.get("user_location")
        if user_location is not None:
            anthropic_user_location = AnthropicWebSearchUserLocation(type="approximate")
            anthropic_user_location_keys = (
                AnthropicWebSearchUserLocation.__annotations__.keys()
            )
            user_location_approximate = user_location.get("approximate")
            if user_location_approximate is not None:
                for key, user_location_value in user_location_approximate.items():
                    if key in anthropic_user_location_keys and key != "type":
                        anthropic_user_location[key] = user_location_value  # type: ignore
                hosted_web_search_tool["user_location"] = anthropic_user_location

        ## MAP SEARCH CONTEXT SIZE
        search_context_size = value_typed.get("search_context_size")
        if search_context_size is not None:
            hosted_web_search_tool["max_uses"] = ANTHROPIC_WEB_SEARCH_TOOL_MAX_USES[
                search_context_size
            ]

        return hosted_web_search_tool

    @staticmethod
    def map_openai_context_management_to_anthropic(
        context_management: Union[List[Dict[str, Any]], Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        OpenAI format: [{"type": "compaction", "compact_threshold": 200000}]
        Anthropic format: {
            "edits": [
                {
                    "type": "compact_20260112",
                    "trigger": {"type": "input_tokens", "value": 150000}
                }
            ]
        }

        Args:
            context_management: OpenAI or Anthropic context_management parameter

        Returns:
            Anthropic-formatted context_management dict, or None if invalid
        """
        # If already in Anthropic format (dict with 'edits'), pass through
        if isinstance(context_management, dict) and "edits" in context_management:
            return context_management

        # If in OpenAI format (list), transform to Anthropic format
        if isinstance(context_management, list):
            anthropic_edits = []
            for entry in context_management:
                if not isinstance(entry, dict):
                    continue

                entry_type = entry.get("type")
                if entry_type == "compaction":
                    anthropic_edit: Dict[str, Any] = {"type": "compact_20260112"}
                    compact_threshold = entry.get("compact_threshold")
                    # Rewrite to 'trigger' with correct nesting if threshold exists
                    if compact_threshold is not None and isinstance(
                        compact_threshold, (int, float)
                    ):
                        anthropic_edit["trigger"] = {
                            "type": "input_tokens",
                            "value": int(compact_threshold),
                        }
                    # Map any other keys by passthrough except handled ones
                    for k in entry:
                        if k not in {
                            "type",
                            "compact_threshold",
                        }:  # only passthrough other keys
                            anthropic_edit[k] = entry[k]

                    anthropic_edits.append(anthropic_edit)

            if anthropic_edits:
                return {"edits": anthropic_edits}

        return None

    def map_openai_params(  # noqa: PLR0915
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        is_thinking_enabled = self.is_thinking_enabled(
            non_default_params=non_default_params
        )

        # NB: ``map_openai_params`` deliberately does NOT sanitize tool names
        # here. Names are the *original* OpenAI names at this stage, and must
        # remain so until ``transform_request`` -- which is the single
        # chokepoint where Anthropic, Bedrock-Anthropic, and Vertex-Anthropic
        # all pass through. Doing it there guarantees:
        #   1. one source of truth for the per-request forward/reverse maps,
        #   2. the maps land on ``litellm_params`` (internal), never on
        #      ``optional_params`` (which is serialized into the request body
        #      via ``data = {**optional_params}`` and would 400 with
        #      ``Extra inputs are not permitted``).

        for param, value in non_default_params.items():
            if param == "max_tokens":
                optional_params["max_tokens"] = (
                    value if isinstance(value, int) else max(1, int(round(value)))
                )
            elif param == "max_completion_tokens":
                optional_params["max_tokens"] = (
                    value if isinstance(value, int) else max(1, int(round(value)))
                )
            elif param == "tools":
                anthropic_tools, mcp_servers = self._map_tools(value)
                optional_params = self._add_tools_to_optional_params(
                    optional_params=optional_params, tools=anthropic_tools
                )
                if mcp_servers:
                    optional_params["mcp_servers"] = mcp_servers
            elif param == "tool_choice" or param == "parallel_tool_calls":
                _tool_choice: Optional[AnthropicMessagesToolChoice] = (
                    self._map_tool_choice(
                        tool_choice=non_default_params.get("tool_choice"),
                        parallel_tool_use=non_default_params.get("parallel_tool_calls"),
                    )
                )

                if _tool_choice is not None:
                    optional_params["tool_choice"] = _tool_choice
            elif param == "stream" and value is True:
                optional_params["stream"] = value
            elif param == "stop" and (
                isinstance(value, str) or isinstance(value, list)
            ):
                _value = self._map_stop_sequences(value)
                if _value is not None:
                    optional_params["stop_sequences"] = _value
            elif param == "temperature":
                optional_params["temperature"] = value
            elif param == "top_p":
                optional_params["top_p"] = value
            elif param == "response_format" and isinstance(value, dict):
                if any(
                    substring in model
                    for substring in {
                        "sonnet-4.5",
                        "sonnet-4-5",
                        "opus-4.1",
                        "opus-4-1",
                        "opus-4.5",
                        "opus-4-5",
                        "opus-4.6",
                        "opus-4-6",
                        "opus-4.7",
                        "opus-4-7",
                        "sonnet-4.6",
                        "sonnet-4-6",
                        "sonnet_4.6",
                        "sonnet_4_6",
                    }
                ):
                    _output_format = (
                        self.map_response_format_to_anthropic_output_format(value)
                    )
                    if _output_format is not None:
                        optional_params["output_format"] = _output_format
                else:
                    _tool = self.map_response_format_to_anthropic_tool(
                        value, optional_params, is_thinking_enabled
                    )
                    if _tool is None:
                        continue
                    if not is_thinking_enabled:
                        _tool_choice = {
                            "name": RESPONSE_FORMAT_TOOL_NAME,
                            "type": "tool",
                        }
                        optional_params["tool_choice"] = _tool_choice

                    optional_params = self._add_tools_to_optional_params(
                        optional_params=optional_params, tools=[_tool]
                    )
                optional_params["json_mode"] = True
            elif (
                param == "user"
                and value is not None
                and isinstance(value, str)
                and _valid_user_id(value)  # anthropic fails on emails
            ):
                optional_params["metadata"] = {"user_id": value}
            elif param == "thinking":
                optional_params["thinking"] = value
            elif param == "reasoning_effort" and isinstance(value, str):
                mapped_thinking = AnthropicConfig._map_reasoning_effort(
                    reasoning_effort=value,
                    model=model,
                    llm_provider=self.custom_llm_provider or "anthropic",
                )
                if mapped_thinking is None:
                    optional_params.pop("thinking", None)
                    optional_params.pop("output_config", None)
                else:
                    optional_params["thinking"] = mapped_thinking
                    if AnthropicConfig._is_adaptive_thinking_model(model):
                        mapped_effort = REASONING_EFFORT_TO_OUTPUT_CONFIG_EFFORT.get(
                            value
                        )
                        if mapped_effort is None:
                            AnthropicConfig._raise_invalid_reasoning_effort(
                                model=model,
                                value=value,
                                llm_provider=self.custom_llm_provider or "anthropic",
                            )
                        optional_params["output_config"] = {"effort": mapped_effort}
            elif param == "web_search_options" and isinstance(value, dict):
                hosted_web_search_tool = self.map_web_search_tool(
                    cast(OpenAIWebSearchOptions, value)
                )
                self._add_tools_to_optional_params(
                    optional_params=optional_params, tools=[hosted_web_search_tool]
                )
            elif param == "extra_headers":
                optional_params["extra_headers"] = value
            elif param == "context_management":
                # Supports both OpenAI list format and Anthropic dict format
                if isinstance(value, (list, dict)):
                    anthropic_context_management = (
                        self.map_openai_context_management_to_anthropic(value)
                    )
                    if anthropic_context_management is not None:
                        optional_params["context_management"] = (
                            anthropic_context_management
                        )
            elif param == "speed" and isinstance(value, str):
                # Pass through Anthropic-specific speed parameter for fast mode
                optional_params["speed"] = value
            elif param == "cache_control" and isinstance(value, dict):
                # Pass through top-level cache_control for automatic prompt caching
                optional_params["cache_control"] = value

        ## handle thinking tokens
        self.update_optional_params_with_thinking_tokens(
            non_default_params=non_default_params, optional_params=optional_params
        )

        return optional_params

    def _create_json_tool_call_for_response_format(
        self,
        json_schema: Optional[dict] = None,
    ) -> AnthropicMessagesTool:
        """
        Handles creating a tool call for getting responses in JSON format.

        Args:
            json_schema (Optional[dict]): The JSON schema the response should be in

        Returns:
            AnthropicMessagesTool: The tool call to send to Anthropic API to get responses in JSON format
        """
        _input_schema: AnthropicInputSchema = AnthropicInputSchema(
            type="object",
        )

        if json_schema is None:
            # Anthropic raises a 400 BadRequest error if properties is passed as None
            # see usage with additionalProperties (Example 5) https://github.com/anthropics/anthropic-cookbook/blob/main/tool_use/extracting_structured_json.ipynb
            _input_schema["additionalProperties"] = True
            _input_schema["properties"] = {}
        else:
            _input_schema.update(cast(AnthropicInputSchema, json_schema))

        _tool = AnthropicMessagesTool(
            name=RESPONSE_FORMAT_TOOL_NAME, input_schema=_input_schema
        )
        return _tool

    def translate_system_message(
        self, messages: List[AllMessageValues]
    ) -> List[AnthropicSystemMessageContent]:
        """
        Translate system message to anthropic format.

        Removes system message from the original list and returns a new list of anthropic system message content.
        Filters out system messages containing x-anthropic-billing-header metadata.
        """
        system_prompt_indices = []
        anthropic_system_message_list: List[AnthropicSystemMessageContent] = []
        for idx, message in enumerate(messages):
            if message["role"] == "system":
                system_prompt_indices.append(idx)
                system_message_block = ChatCompletionSystemMessage(**message)
                if isinstance(system_message_block["content"], str):
                    # Skip empty text blocks - Anthropic API raises errors for empty text
                    if not system_message_block["content"]:
                        continue
                    # Skip system messages containing x-anthropic-billing-header metadata
                    if system_message_block["content"].startswith(
                        "x-anthropic-billing-header:"
                    ):
                        continue
                    anthropic_system_message_content = AnthropicSystemMessageContent(
                        type="text",
                        text=system_message_block["content"],
                    )
                    if "cache_control" in system_message_block:
                        anthropic_system_message_content["cache_control"] = (
                            system_message_block["cache_control"]
                        )
                    anthropic_system_message_list.append(
                        anthropic_system_message_content
                    )
                elif isinstance(message["content"], list):
                    for _content in message["content"]:
                        # Skip empty text blocks - Anthropic API raises errors for empty text
                        text_value = _content.get("text")
                        if _content.get("type") == "text" and not text_value:
                            continue
                        # Skip system messages containing x-anthropic-billing-header metadata
                        if (
                            _content.get("type") == "text"
                            and text_value
                            and text_value.startswith("x-anthropic-billing-header:")
                        ):
                            continue
                        anthropic_system_message_content = (
                            AnthropicSystemMessageContent(
                                type=_content.get("type"),
                                text=text_value,
                            )
                        )
                        if "cache_control" in _content:
                            anthropic_system_message_content["cache_control"] = (
                                _content["cache_control"]
                            )

                        anthropic_system_message_list.append(
                            anthropic_system_message_content
                        )

        if len(system_prompt_indices) > 0:
            for idx in reversed(system_prompt_indices):
                messages.pop(idx)

        return anthropic_system_message_list

    def add_code_execution_tool(
        self,
        messages: List[AllAnthropicMessageValues],
        tools: List[Union[AllAnthropicToolsValues, Dict]],
    ) -> List[Union[AllAnthropicToolsValues, Dict]]:
        """if 'container_upload' in messages, add code_execution tool"""
        add_code_execution_tool = False
        for message in messages:
            message_content = message.get("content", None)
            if message_content and isinstance(message_content, list):
                for content in message_content:
                    content_type = content.get("type", None)
                    if content_type == "container_upload":
                        add_code_execution_tool = True
                        break

        if add_code_execution_tool:
            ## check if code_execution tool is already in tools
            for tool in tools:
                tool_type = tool.get("type", None)
                if (
                    tool_type
                    and isinstance(tool_type, str)
                    and tool_type.startswith("code_execution")
                ):
                    return tools
            tools.append(
                AnthropicCodeExecutionTool(
                    name="code_execution",
                    type="code_execution_20250522",
                )
            )
        return tools

    def _ensure_beta_header(self, headers: dict, beta_value: str) -> None:
        """
        Ensure a beta header value is present in the anthropic-beta header.
        Merges with existing values instead of overriding them.

        Args:
            headers: Dictionary of headers to update
            beta_value: The beta header value to add
        """
        existing_beta = headers.get("anthropic-beta")
        if existing_beta is None:
            headers["anthropic-beta"] = beta_value
            return
        existing_values = [beta.strip() for beta in existing_beta.split(",")]
        if beta_value not in existing_values:
            headers["anthropic-beta"] = f"{existing_beta}, {beta_value}"

    def _ensure_context_management_beta_header(
        self, headers: dict, context_management: object
    ) -> None:
        """
        Add appropriate beta headers based on context_management edits.
        """
        edits = []
        # If anthropic format (dict with "edits" key)
        if isinstance(context_management, dict) and "edits" in context_management:
            edits = context_management.get("edits", [])
        # If OpenAI format: list of context management entries
        elif isinstance(context_management, list):
            edits = context_management
        # Defensive: ignore/fallback if context_management not valid
        else:
            return

        has_compact = False
        has_other = False

        for edit in edits:
            edit_type = edit.get("type", "")
            if edit_type == "compact_20260112" or edit_type == "compaction":
                has_compact = True
            else:
                has_other = True

        # Add compact header if any compact edits/entries exist
        if has_compact:
            self._ensure_beta_header(
                headers, ANTHROPIC_BETA_HEADER_VALUES.COMPACT_2026_01_12.value
            )

        # Add context management header if any other edits/entries exist
        if has_other:
            self._ensure_beta_header(
                headers,
                ANTHROPIC_BETA_HEADER_VALUES.CONTEXT_MANAGEMENT_2025_06_27.value,
            )

    def update_headers_with_optional_anthropic_beta(
        self, headers: dict, optional_params: dict
    ) -> dict:
        """Update headers with optional anthropic beta."""

        # Skip adding beta headers for Vertex requests
        # Vertex AI handles these headers differently
        is_vertex_request = optional_params.get("is_vertex_request", False)
        if is_vertex_request:
            return headers

        _tools = optional_params.get("tools", [])
        for tool in _tools:
            if tool.get("type", None) and tool.get("type").startswith(
                ANTHROPIC_HOSTED_TOOLS.WEB_FETCH.value
            ):
                self._ensure_beta_header(
                    headers, ANTHROPIC_BETA_HEADER_VALUES.WEB_FETCH_2025_09_10.value
                )
            elif tool.get("type", None) and tool.get("type").startswith(
                ANTHROPIC_HOSTED_TOOLS.MEMORY.value
            ):
                self._ensure_beta_header(
                    headers,
                    ANTHROPIC_BETA_HEADER_VALUES.CONTEXT_MANAGEMENT_2025_06_27.value,
                )
        if optional_params.get("context_management") is not None:
            self._ensure_context_management_beta_header(
                headers, optional_params["context_management"]
            )
        if optional_params.get("output_format") is not None:
            self._ensure_beta_header(
                headers, ANTHROPIC_BETA_HEADER_VALUES.STRUCTURED_OUTPUT_2025_09_25.value
            )
        if optional_params.get("speed") == "fast":
            self._ensure_beta_header(
                headers, ANTHROPIC_BETA_HEADER_VALUES.FAST_MODE_2026_02_01.value
            )
        for tool in _tools:
            if tool.get("type") == ANTHROPIC_ADVISOR_TOOL_TYPE:
                self._ensure_beta_header(
                    headers, ANTHROPIC_BETA_HEADER_VALUES.ADVISOR_TOOL_2026_03_01.value
                )
                break
        return headers

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Translate messages to anthropic format.
        """
        ## VALIDATE REQUEST
        """Anthropic requires ``tools`` when messages include tool blocks; LiteLLM injects a dummy tool if omitted (no ``modify_params`` needed)."""
        from litellm.litellm_core_utils.prompt_templates.factory import (
            anthropic_messages_pt,
        )

        if (
            "tools" not in optional_params
            and messages is not None
            and has_tool_call_blocks(messages)
        ):
            optional_params["tools"], _ = self._map_tools(
                add_dummy_tool(custom_llm_provider="anthropic")
            )

        # Drop thinking param if thinking is enabled but thinking_blocks are missing
        # This prevents the error: "Expected thinking or redacted_thinking, but found tool_use"
        #
        # IMPORTANT: Only drop thinking if NO assistant messages have thinking_blocks.
        # If any message has thinking_blocks, we must keep thinking enabled, otherwise
        # Anthropic errors with: "When thinking is disabled, an assistant message cannot contain thinking"
        # Related issue: https://github.com/BerriAI/litellm/issues/18926
        if (
            optional_params.get("thinking") is not None
            and messages is not None
            and last_assistant_with_tool_calls_has_no_thinking_blocks(messages)
            and not any_assistant_message_has_thinking_blocks(messages)
        ):
            if litellm.modify_params:
                optional_params.pop("thinking", None)
                litellm.verbose_logger.warning(
                    "Dropping 'thinking' param because the last assistant message with tool_calls "
                    "has no thinking_blocks. The model won't use extended thinking for this turn."
                )

        headers = self.update_headers_with_optional_anthropic_beta(
            headers=headers, optional_params=optional_params
        )

        # === Tool-name sanitization (single chokepoint) ===
        # Anthropic enforces ^[a-zA-Z0-9_-]{1,128}$ on every tool name. We
        # sanitize *here* -- not in map_openai_params -- because:
        #
        #   - This function is the single boundary shared by AnthropicConfig,
        #     AmazonAnthropicConfig (Bedrock invoke), VertexAIAnthropicConfig,
        #     and AzureAnthropicConfig (all call ``super().transform_request``
        #     or ``AnthropicConfig.transform_request(self, ...)``). Sanitizing
        #     once here covers every Anthropic-shaped request.
        #   - The forward/reverse maps are coordination state; they belong on
        #     ``litellm_params`` (internal-only), never on ``optional_params``
        #     (which becomes the JSON body via ``{**optional_params}``).
        #   - It keeps ``map_openai_params`` a pure param translator with no
        #     side-channel state.
        #
        # The reverse map only contains entries for names that were actually
        # rewritten -- so a tool legitimately named ``foo_bar`` is never
        # incorrectly retyped to ``foo/bar`` on the response side.
        # See _build_anthropic_tool_name_maps for the collision-handling
        # rules and rationale.
        _name_forward_map, _name_reverse_map = self._sanitize_tool_names_in_request(
            optional_params=optional_params,
        )
        if _name_forward_map:
            messages = self._rewrite_tool_names_in_messages(messages, _name_forward_map)
        if _name_reverse_map and isinstance(litellm_params, dict):
            litellm_params[ANTHROPIC_TOOL_NAME_REVERSE_MAP_KEY] = _name_reverse_map

        # Separate system prompt from rest of message
        anthropic_system_message_list = self.translate_system_message(messages=messages)
        # Handling anthropic API Prompt Caching
        if len(anthropic_system_message_list) > 0:
            optional_params["system"] = anthropic_system_message_list
        # Format rest of message according to anthropic guidelines
        try:
            anthropic_messages = anthropic_messages_pt(
                model=model,
                messages=messages,
                llm_provider=self.custom_llm_provider or "anthropic",
            )
        except Exception as e:
            raise AnthropicError(
                status_code=400,
                message="{}\nReceived Messages={}".format(str(e), messages),
            )  # don't use verbose_logger.exception, if exception is raised

        ## Auto-strip advisor blocks from history if advisor tool is absent.
        ## Prevents Anthropic 400: advisor_tool_result in history requires advisor tool.
        _all_tools = optional_params.get("tools") or []
        _has_advisor = any(
            isinstance(t, dict) and t.get("type") == ANTHROPIC_ADVISOR_TOOL_TYPE
            for t in _all_tools
        )
        if not _has_advisor:
            anthropic_messages = strip_advisor_blocks_from_messages(anthropic_messages)

        ## Add code_execution tool if container_upload is in messages
        _tools = (
            cast(
                Optional[List[Union[AllAnthropicToolsValues, Dict]]],
                optional_params.get("tools"),
            )
            or []
        )
        tools = self.add_code_execution_tool(messages=anthropic_messages, tools=_tools)
        if len(tools) > 1:
            optional_params["tools"] = tools

        ## Load Config
        config = litellm.AnthropicConfig.get_config(model=model)
        for k, v in config.items():
            if (
                k not in optional_params
            ):  # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
                optional_params[k] = v

        ## Handle user_id in metadata
        _litellm_metadata = litellm_params.get("metadata", None)
        if (
            _litellm_metadata
            and isinstance(_litellm_metadata, dict)
            and "user_id" in _litellm_metadata
            and _litellm_metadata["user_id"] is not None
            and _valid_user_id(_litellm_metadata["user_id"])
        ):
            optional_params["metadata"] = {"user_id": _litellm_metadata["user_id"]}

        ## Ensure metadata only contains user_id (only documented field in Anthropic Messages API)
        if "metadata" in optional_params and isinstance(
            optional_params["metadata"], dict
        ):
            _user_id = optional_params["metadata"].get("user_id")
            if _user_id is not None:
                optional_params["metadata"] = {"user_id": _user_id}
            else:
                optional_params.pop("metadata")

        # Remove internal LiteLLM parameters that should not be sent to Anthropic API
        optional_params.pop("is_vertex_request", None)

        data = {
            "model": model,
            "messages": anthropic_messages,
            **optional_params,
        }

        self._apply_output_config(
            data=data, model=model, optional_params=optional_params
        )

        return data

    def _apply_output_config(
        self, data: dict, model: str, optional_params: dict
    ) -> None:
        """Validate and apply output_config to the request data."""
        if "output_config" not in optional_params:
            return
        output_config = optional_params.get("output_config")
        if not output_config or not isinstance(output_config, dict):
            return
        if litellm.drop_params is True and not self._model_supports_effort_param(model):
            litellm.verbose_logger.warning(
                DROP_UNSUPPORTED_OUTPUT_CONFIG_WARNING,
                model,
            )
            optional_params.pop("output_config", None)
            data.pop("output_config", None)
            return
        effort = output_config.get("effort")
        valid_efforts = ["high", "medium", "low", "xhigh", "max"]
        if effort is not None and effort not in valid_efforts:
            raise litellm.exceptions.BadRequestError(
                message=(
                    f"Invalid effort value: {effort!r}. Must be one of: "
                    f"'high', 'medium', 'low', 'xhigh', 'max'"
                ),
                model=model,
                llm_provider=self.custom_llm_provider or "anthropic",
            )
        gate_error = self._validate_effort_for_model(model, effort)
        if gate_error is not None:
            raise litellm.exceptions.BadRequestError(
                message=gate_error,
                model=model,
                llm_provider=self.custom_llm_provider or "anthropic",
            )
        data["output_config"] = output_config

    def _resolve_json_mode_non_streaming(
        self,
        json_mode: Optional[bool],
        tool_calls: List[ChatCompletionToolCallChunk],
    ) -> Tuple[
        Optional[LitellmMessage],
        List[ChatCompletionToolCallChunk],
        Optional[str],
    ]:
        """Strip internal response_format tool calls; merge payload into content when mixed with user tools."""
        if json_mode is not True or not tool_calls:
            return None, tool_calls, None

        json_indices = [
            i
            for i, t in enumerate(tool_calls)
            if t.get("function", {}).get("name") == RESPONSE_FORMAT_TOOL_NAME
        ]
        if not json_indices:
            return None, tool_calls, None

        if len(json_indices) == len(tool_calls):
            json_tool = tool_calls[json_indices[0]]
            if json_tool.get("function", {}).get("arguments") is None:
                return None, tool_calls, None
            _message = AnthropicConfig._convert_tool_response_to_message(
                tool_calls=[json_tool]
            )
            return _message, [], None

        first_json = tool_calls[json_indices[0]]
        json_msg = AnthropicConfig._convert_tool_response_to_message([first_json])
        extra_content: Optional[str] = (
            json_msg.content if json_msg is not None else None
        )
        filtered_tools = [t for i, t in enumerate(tool_calls) if i not in json_indices]
        return None, filtered_tools, extra_content

    def extract_response_content(self, completion_response: dict) -> Tuple[
        str,
        Optional[List[Any]],
        Optional[
            List[
                Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]
            ]
        ],
        Optional[str],
        List[ChatCompletionToolCallChunk],
        Optional[List[Any]],
        Optional[List[Any]],
        Optional[List[Any]],
    ]:
        text_content = ""
        citations: Optional[List[Any]] = None
        thinking_blocks: Optional[
            List[
                Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]
            ]
        ] = None
        reasoning_content: Optional[str] = None
        tool_calls: List[ChatCompletionToolCallChunk] = []
        web_search_results: Optional[List[Any]] = None
        tool_results: Optional[List[Any]] = None
        compaction_blocks: Optional[List[Any]] = None
        for idx, content in enumerate(completion_response["content"]):
            if content["type"] == "text":
                text_content += content["text"]
            ## TOOL CALLING
            elif content["type"] == "tool_use" or content["type"] == "server_tool_use":
                tool_call = AnthropicConfig.convert_tool_use_to_openai_format(
                    anthropic_tool_content=content,
                    index=idx,
                )
                tool_calls.append(tool_call)

            ## TOOL RESULTS - handle all tool result types (code execution, etc.)
            elif content["type"].endswith("_tool_result"):
                # Skip tool_search_tool_result as it's internal metadata
                if content["type"] == "tool_search_tool_result":
                    continue
                # Handle web_search_tool_result separately for backwards compatibility
                if content["type"] == "web_search_tool_result":
                    if web_search_results is None:
                        web_search_results = []
                    web_search_results.append(content)
                elif content["type"] == "web_fetch_tool_result":
                    if web_search_results is None:
                        web_search_results = []
                    web_search_results.append(content)
                else:
                    # All other tool results (bash_code_execution_tool_result, text_editor_code_execution_tool_result, etc.)
                    if tool_results is None:
                        tool_results = []
                    tool_results.append(content)

            elif content.get("type") == "thinking":
                if thinking_blocks is None:
                    thinking_blocks = []
                thinking_blocks.append(cast(ChatCompletionThinkingBlock, content))
            elif content["type"] == "redacted_thinking":
                if thinking_blocks is None:
                    thinking_blocks = []
                thinking_blocks.append(
                    cast(ChatCompletionRedactedThinkingBlock, content)
                )

            ## COMPACTION
            elif content["type"] == "compaction":
                if compaction_blocks is None:
                    compaction_blocks = []
                compaction_blocks.append(content)

            ## CITATIONS
            if content.get("citations") is not None:
                if citations is None:
                    citations = []
                citations.append(
                    [
                        {
                            **citation,
                            "supported_text": content.get("text", ""),
                        }
                        for citation in content["citations"]
                    ]
                )
        if thinking_blocks is not None:
            reasoning_content = ""
            for block in thinking_blocks:
                thinking_content = cast(Optional[str], block.get("thinking"))
                if thinking_content is not None:
                    reasoning_content += thinking_content

        return (
            text_content,
            citations,
            thinking_blocks,
            reasoning_content,
            tool_calls,
            web_search_results,
            tool_results,
            compaction_blocks,
        )

    def calculate_usage(
        self,
        usage_object: dict,
        reasoning_content: Optional[str],
        completion_response: Optional[dict] = None,
        speed: Optional[str] = None,
    ) -> Usage:
        # NOTE: Sometimes the usage object has None set explicitly for token counts, meaning .get() & key access returns None, and we need to account for this
        raw_prompt_tokens = usage_object.get("input_tokens", 0) or 0
        prompt_tokens: int = (
            int(raw_prompt_tokens) if isinstance(raw_prompt_tokens, (int, float)) else 0
        )
        raw_completion_tokens = usage_object.get("output_tokens", 0) or 0
        completion_tokens: int = (
            int(raw_completion_tokens)
            if isinstance(raw_completion_tokens, (int, float))
            else 0
        )
        _usage = usage_object
        cache_creation_input_tokens: int = 0
        cache_read_input_tokens: int = 0
        cache_creation_token_details: Optional[CacheCreationTokenDetails] = None
        web_search_requests: Optional[int] = None
        tool_search_requests: Optional[int] = None
        inference_geo: Optional[str] = None
        if "inference_geo" in _usage and _usage["inference_geo"] is not None:
            inference_geo = _usage["inference_geo"]

        if (
            "cache_creation_input_tokens" in _usage
            and _usage["cache_creation_input_tokens"] is not None
        ):
            cache_creation_input_tokens = _usage["cache_creation_input_tokens"]
            prompt_tokens += cache_creation_input_tokens
        if (
            "cache_read_input_tokens" in _usage
            and _usage["cache_read_input_tokens"] is not None
        ):
            cache_read_input_tokens = _usage["cache_read_input_tokens"]
            prompt_tokens += cache_read_input_tokens
        if "server_tool_use" in _usage and _usage["server_tool_use"] is not None:
            if (
                "web_search_requests" in _usage["server_tool_use"]
                and _usage["server_tool_use"]["web_search_requests"] is not None
            ):
                web_search_requests = cast(
                    int, _usage["server_tool_use"]["web_search_requests"]
                )
            if (
                "tool_search_requests" in _usage["server_tool_use"]
                and _usage["server_tool_use"]["tool_search_requests"] is not None
            ):
                tool_search_requests = cast(
                    int, _usage["server_tool_use"]["tool_search_requests"]
                )

        # Count tool_search_requests from content blocks if not in usage
        # Anthropic doesn't always include tool_search_requests in the usage object
        if tool_search_requests is None and completion_response is not None:
            tool_search_count = 0
            for content in completion_response.get("content", []):
                if content.get("type") == "server_tool_use":
                    tool_name = content.get("name", "")
                    if "tool_search" in tool_name:
                        tool_search_count += 1
            if tool_search_count > 0:
                tool_search_requests = tool_search_count

        if "cache_creation" in _usage and _usage["cache_creation"] is not None:
            cache_creation_token_details = CacheCreationTokenDetails(
                ephemeral_5m_input_tokens=_usage["cache_creation"].get(
                    "ephemeral_5m_input_tokens"
                ),
                ephemeral_1h_input_tokens=_usage["cache_creation"].get(
                    "ephemeral_1h_input_tokens"
                ),
            )

        raw_input_tokens = usage_object.get("input_tokens", 0) or 0
        prompt_tokens_details = PromptTokensDetailsWrapper(
            cached_tokens=cache_read_input_tokens,
            cache_creation_tokens=cache_creation_input_tokens,
            cache_creation_token_details=cache_creation_token_details,
            text_tokens=raw_input_tokens,
        )
        # Always populate completion_token_details, not just when there's reasoning_content
        estimated_reasoning_tokens = (
            token_counter(text=reasoning_content, count_response_tokens=True)
            if reasoning_content
            else 0
        )
        reasoning_tokens = min(estimated_reasoning_tokens, completion_tokens)
        completion_token_details = CompletionTokensDetailsWrapper(
            reasoning_tokens=reasoning_tokens if reasoning_tokens > 0 else 0,
            text_tokens=(
                completion_tokens - reasoning_tokens
                if reasoning_tokens > 0
                else completion_tokens
            ),
        )
        total_tokens = prompt_tokens + completion_tokens

        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_tokens_details=prompt_tokens_details,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            completion_tokens_details=completion_token_details,
            server_tool_use=(
                ServerToolUse(
                    web_search_requests=web_search_requests,
                    tool_search_requests=tool_search_requests,
                )
                if (web_search_requests is not None or tool_search_requests is not None)
                else None
            ),
            inference_geo=inference_geo,
            speed=speed,
        )
        return usage

    def _build_code_by_id_map(
        self, tool_calls: List[ChatCompletionToolCallChunk]
    ) -> Dict[str, str]:
        code_by_id: Dict[str, str] = {}
        for tc in tool_calls:
            try:
                args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                call_id = tc.get("id")
                command = args.get("command", "")
                if isinstance(call_id, str):
                    code_by_id[call_id] = command if isinstance(command, str) else ""
            except Exception:
                pass
        return code_by_id

    def _build_code_interpreter_results(
        self,
        tool_results: List[Any],
        code_by_id: Dict[str, str],
        container_id: Optional[str],
    ) -> List[OutputCodeInterpreterCall]:
        code_interpreter_results = []
        for tr in tool_results:
            if tr.get("type") != "bash_code_execution_tool_result":
                continue
            call_id = tr.get("tool_use_id", "")
            content = tr.get("content", {})
            log_outputs = build_code_interpreter_log_outputs(content)
            code_interpreter_results.append(
                OutputCodeInterpreterCall(
                    type="code_interpreter_call",
                    id=call_id,
                    code=code_by_id.get(call_id, ""),
                    container_id=container_id,
                    status="completed",
                    outputs=log_outputs,
                )
            )
        return code_interpreter_results

    def _build_provider_specific_fields(
        self,
        completion_response: dict,
        citations: Optional[List[Any]],
        thinking_blocks: Optional[
            List[
                Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]
            ]
        ],
        web_search_results: Optional[List[Any]],
        tool_results: Optional[List[Any]],
        compaction_blocks: Optional[List[Any]],
        tool_calls: List[ChatCompletionToolCallChunk],
    ) -> Dict[str, Any]:
        provider_specific_fields: Dict[str, Any] = {
            "citations": citations,
            "thinking_blocks": thinking_blocks,
        }

        context_management = completion_response.get("context_management")
        if context_management is not None:
            provider_specific_fields["context_management"] = context_management

        if web_search_results is not None:
            provider_specific_fields["web_search_results"] = web_search_results

        if tool_results is not None:
            provider_specific_fields["tool_results"] = tool_results
            container_id = (
                completion_response.get("container", {}).get("id")
                if isinstance(completion_response.get("container"), dict)
                else None
            )
            code_by_id = self._build_code_by_id_map(tool_calls)
            code_interpreter_results = self._build_code_interpreter_results(
                tool_results, code_by_id, container_id
            )
            provider_specific_fields["code_interpreter_results"] = (
                code_interpreter_results
            )

        container = completion_response.get("container")
        if container is not None:
            provider_specific_fields["container"] = container

        if compaction_blocks is not None:
            provider_specific_fields["compaction_blocks"] = compaction_blocks

        return provider_specific_fields

    def transform_parsed_response(
        self,
        completion_response: dict,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        json_mode: Optional[bool] = None,
        prefix_prompt: Optional[str] = None,
        speed: Optional[str] = None,
        tool_name_reverse_map: Optional[Dict[str, str]] = None,
    ):
        _hidden_params: Dict = {}
        _hidden_params["additional_headers"] = process_anthropic_headers(
            dict(raw_response.headers)
        )
        if "error" in completion_response:
            response_headers = getattr(raw_response, "headers", None)
            raise AnthropicError(
                message=str(completion_response["error"]),
                status_code=raw_response.status_code,
                headers=response_headers,
            )

        (
            text_content,
            citations,
            thinking_blocks,
            reasoning_content,
            tool_calls,
            web_search_results,
            tool_results,
            compaction_blocks,
        ) = self.extract_response_content(completion_response=completion_response)

        # Reverse-map rewritten tool names back to caller's originals so a
        # downstream OpenAI-style dispatcher can match on the registered name.
        # See _build_anthropic_tool_name_maps for why this is keyed on the
        # per-request reverse map (so a tool legitimately named `foo_bar` is
        # never incorrectly retyped to `foo/bar`). No-op when the map is
        # empty (the common case).
        if tool_name_reverse_map and tool_calls:
            for tc in tool_calls:
                fn = tc.get("function") if isinstance(tc, dict) else None
                if fn is None:
                    continue
                _name = fn.get("name")
                if isinstance(_name, str) and _name in tool_name_reverse_map:
                    fn["name"] = tool_name_reverse_map[_name]

        if (
            prefix_prompt is not None
            and not text_content.startswith(prefix_prompt)
            and not litellm.disable_add_prefix_to_prompt
        ):
            text_content = prefix_prompt + text_content

        provider_specific_fields = self._build_provider_specific_fields(
            completion_response,
            citations,
            thinking_blocks,
            web_search_results,
            tool_results,
            compaction_blocks,
            tool_calls,
        )

        json_mode_message, tool_calls_for_message, json_extra_content = (
            self._resolve_json_mode_non_streaming(
                json_mode=json_mode,
                tool_calls=tool_calls,
            )
        )
        merged_text = text_content or ""
        if json_extra_content:
            merged_text = (
                merged_text + json_extra_content if merged_text else json_extra_content
            )

        _message = litellm.Message(
            tool_calls=tool_calls_for_message,
            content=merged_text or None,
            provider_specific_fields=provider_specific_fields,
            thinking_blocks=thinking_blocks,
            reasoning_content=reasoning_content,
        )
        _message.provider_specific_fields = provider_specific_fields

        if json_mode_message is not None:
            completion_response["stop_reason"] = "stop"
            _message = json_mode_message

        model_response.choices[0].message = _message
        model_response._hidden_params["original_response"] = completion_response[
            "content"
        ]
        model_response.choices[0].finish_reason = cast(
            OpenAIChatCompletionFinishReason,
            map_finish_reason(completion_response["stop_reason"]),
        )

        usage = self.calculate_usage(
            usage_object=completion_response["usage"],
            reasoning_content=reasoning_content,
            completion_response=completion_response,
            speed=speed,
        )
        setattr(model_response, "usage", usage)

        model_response.created = int(time.time())
        model_response.model = completion_response["model"]

        _hidden_params["provider_specific_fields"] = provider_specific_fields
        model_response._hidden_params = _hidden_params
        return model_response

    def get_prefix_prompt(self, messages: List[AllMessageValues]) -> Optional[str]:
        """
        Get the prefix prompt from the messages.

        Check last message
        - if it's assistant message, with 'prefix': true, return the content

        E.g. :    {"role": "assistant", "content": "Argentina", "prefix": True}
        """
        if len(messages) == 0:
            return None

        message = messages[-1]
        message_content = message.get("content")
        if (
            message["role"] == "assistant"
            and message.get("prefix", False)
            and isinstance(message_content, str)
        ):
            return message_content

        return None

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LoggingClass,
        request_data: Dict,
        messages: List[AllMessageValues],
        optional_params: Dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        ## LOGGING
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=raw_response.text,
            additional_args={"complete_input_dict": request_data},
        )

        ## RESPONSE OBJECT
        try:
            completion_response = raw_response.json()
        except Exception as e:
            response_headers = getattr(raw_response, "headers", None)
            raise AnthropicError(
                message="Unable to get json response - {}, Original Response: {}".format(
                    str(e), raw_response.text
                ),
                status_code=raw_response.status_code,
                headers=response_headers,
            )

        prefix_prompt = self.get_prefix_prompt(messages=messages)
        speed = optional_params.get("speed")
        tool_name_reverse_map: Optional[Dict[str, str]] = None
        if isinstance(litellm_params, dict):
            _candidate = litellm_params.get(ANTHROPIC_TOOL_NAME_REVERSE_MAP_KEY)
            if isinstance(_candidate, dict):
                tool_name_reverse_map = _candidate

        model_response = self.transform_parsed_response(
            completion_response=completion_response,
            raw_response=raw_response,
            model_response=model_response,
            json_mode=json_mode,
            prefix_prompt=prefix_prompt,
            speed=speed,
            tool_name_reverse_map=tool_name_reverse_map,
        )
        return model_response

    @staticmethod
    def _convert_tool_response_to_message(
        tool_calls: List[ChatCompletionToolCallChunk],
    ) -> Optional[LitellmMessage]:
        """
        In JSON mode, Anthropic API returns JSON schema as a tool call, we need to convert it to a message to follow the OpenAI format

        """
        ## HANDLE JSON MODE - anthropic returns single function call
        json_mode_content_str: Optional[str] = tool_calls[0]["function"].get(
            "arguments"
        )
        try:
            if json_mode_content_str is not None:
                args = json.loads(json_mode_content_str)
                if (
                    isinstance(args, dict)
                    and (values := args.get("values")) is not None
                ):
                    _message = litellm.Message(content=json.dumps(values))
                    return _message
                else:
                    # a lot of the times the `values` key is not present in the tool response
                    # relevant issue: https://github.com/BerriAI/litellm/issues/6741
                    _message = litellm.Message(content=json.dumps(args))
                    return _message
        except json.JSONDecodeError:
            # json decode error does occur, return the original tool response str
            return litellm.Message(content=json_mode_content_str)
        return None

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[Dict, httpx.Headers]
    ) -> BaseLLMException:
        return AnthropicError(
            status_code=status_code,
            message=error_message,
            headers=cast(httpx.Headers, headers),
        )


def _valid_user_id(user_id: str) -> bool:
    """
    Validate that user_id is not an email or phone number.
    Returns: bool: True if valid (not email or phone), False otherwise
    """
    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    phone_pattern = r"^\+?[\d\s\(\)-]{7,}$"

    if re.match(email_pattern, user_id):
        return False
    if re.match(phone_pattern, user_id):
        return False

    return True
