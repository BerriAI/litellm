"""
MCP Sampling Handler
Handles `sampling/createMessage` requests from upstream MCP servers by
routing them through LiteLLM's internal completion infrastructure.
This allows MCP servers to perform agentic reasoning (e.g., multi-step
tool calling, chain-of-thought) without needing their own LLM API keys —
LiteLLM acts as the LLM provider using its existing 100+ provider support,
cost tracking, rate limiting, and model routing.
MCP Spec Reference:
    https://modelcontextprotocol.io/specification/2025-11-25/client/sampling
"""

import typing
from collections.abc import Mapping, Sequence
from typing import Any, Final, NamedTuple, Optional, Protocol, Union, runtime_checkable

if typing.TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fastapi import Request
    from mcp.client.session import ClientSession
    from mcp.shared.context import RequestContext
    from mcp.types import (
        ContentBlock,
        CreateMessageResult,
        CreateMessageResultWithTools,
        ErrorData,
        SamplingMessageContentBlock,
        TextContent,
        ToolUseContent,
    )

    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.utils import ModelResponse

from fastapi import HTTPException
from pydantic import TypeAdapter

from litellm._logging import verbose_logger

# Guard imports that require the mcp package
try:
    from mcp.types import (
        CreateMessageRequestParams,
        CreateMessageResult,
        CreateMessageResultWithTools,
        ErrorData,
        ModelPreferences,
        SamplingMessage,
        TextContent,
        Tool,
        ToolChoice,
        ToolUseContent,
    )

    MCP_SAMPLING_AVAILABLE = True
except ImportError as _sampling_import_err:
    MCP_SAMPLING_AVAILABLE = False
    verbose_logger.warning(
        "MCP sampling disabled: failed to import required types from mcp.types — %s. "
        "This usually means the 'mcp' package is not installed or is an older version "
        "that does not support sampling. Install/upgrade with: pip install 'mcp>=1.1'",
        _sampling_import_err,
    )


def _resolve_model_from_preferences(
    model_preferences: Optional["ModelPreferences"],
    default_model: str | None = None,
) -> str:
    """
    Resolve an LLM model name from MCP ModelPreferences.
    Strategy:
    1. Check hints for substring matches against known model names.
    2. Fall back to priority-based selection (cost/speed/intelligence).
    3. Fall back to the configured default model.
    Args:
        model_preferences: MCP ModelPreferences with hints and priorities.
        default_model: Fallback model if no hint matches.
    Returns:
        A model string suitable for litellm.acompletion().
    """
    import litellm

    # Build list of available model names from proxy Router or litellm.model_list
    available_model_names: list[str] = []
    try:
        from litellm.proxy.proxy_server import llm_router

        if llm_router is not None:
            available_model_names = llm_router.get_model_names()
    except Exception:
        pass
    if not available_model_names and litellm.model_list:
        for entry in litellm.model_list:
            if isinstance(entry, dict):
                name = entry.get("model_name")
                if name:
                    available_model_names.append(name)
            elif isinstance(entry, str):
                available_model_names.append(entry)
    if model_preferences and model_preferences.hints:
        for hint in model_preferences.hints:
            hint_name: str | None = getattr(hint, "name", None)
            if not hint_name:
                continue
            # Try direct match first
            if hint_name in available_model_names:
                verbose_logger.debug(
                    "MCP sampling model resolution: direct hint match '%s'",
                    hint_name,
                )
                return hint_name
            # Try substring match against known models
            for model_name in available_model_names:
                if hint_name.lower() in model_name.lower():
                    verbose_logger.debug(
                        "MCP sampling model resolution: substring hint match '%s' -> '%s'",
                        hint_name,
                        model_name,
                    )
                    return model_name
        verbose_logger.debug(
            "MCP sampling model resolution: no hint matched from %s against %d available models",
            [getattr(h, "name", None) for h in model_preferences.hints],
            len(available_model_names),
        )

    # 2. Priority-based selection (cost/speed/intelligence)
    if model_preferences and available_model_names and _has_priorities(model_preferences):
        best: Final = _select_model_by_priority(available_model_names, model_preferences)
        if best is not None:
            verbose_logger.debug(
                "MCP sampling model resolution: priority-based selection chose '%s'",
                best,
            )
            return best

    # 3. Use default model from caller
    if default_model:
        verbose_logger.debug(
            "MCP sampling model resolution: using caller-provided default '%s'",
            default_model,
        )
        return default_model
    # Fall back to first available model
    if available_model_names:
        verbose_logger.debug(
            "MCP sampling model resolution: no default configured, falling back to first available model '%s'",
            available_model_names[0],
        )
        return available_model_names[0]
    # Last resort - use LiteLLM default or raise error
    default_sampling_model: Final[str | None] = getattr(litellm, "default_mcp_sampling_model", None)
    if default_sampling_model:
        verbose_logger.debug(
            "MCP sampling model resolution: using litellm.default_mcp_sampling_model='%s'",
            default_sampling_model,
        )
        return default_sampling_model
    raise ValueError(
        "No model could be resolved for MCP sampling. Please configure 'default_mcp_sampling_model' in your LiteLLM configuration."
    )


def _has_priorities(model_preferences: "ModelPreferences") -> bool:
    """Return True if any priority weight is set (non-None and > 0)."""
    return any(
        (getattr(model_preferences, attr, None) or 0) > 0
        for attr in ("costPriority", "speedPriority", "intelligencePriority")
    )


class _ScoredModel(NamedTuple):
    name: str
    cost: float
    max_output: float
    output_tps: float


def _select_model_by_priority(
    model_names: list[str],
    model_preferences: "ModelPreferences",
) -> str | None:
    """Score available models by MCP priority weights and return the best.

    Scoring strategy (per the MCP spec, priorities are 0-1 floats):

    * **costPriority** — higher means "prefer cheaper models".
      Metric: combined (input + output) cost per token from
      ``model_prices_and_context_window.json``.  Lower cost → higher score.

    * **speedPriority** — higher means "prefer faster models".
      Metric: ``output_tokens_per_second`` from model info when available;
      otherwise a neutral score for every candidate, since no reliable
      latency proxy exists (context-window size does not track speed).

    * **intelligencePriority** — higher means "prefer smarter models".
      Metric: ``max_output_tokens`` is used as a rough capability proxy
      (frontier models expose larger context windows).

    Each metric is min-max normalised across the candidate set so that
    every model gets a 0-1 score per dimension.  The final score is the
    weighted sum of the three normalised dimensions.

    Returns the highest-scoring model name, or None if scoring fails for
    all candidates (e.g. no model_info available).
    """
    import litellm as _litellm

    cost_weight: Final[float] = getattr(model_preferences, "costPriority", None) or 0.0
    speed_weight: Final[float] = getattr(model_preferences, "speedPriority", None) or 0.0
    intel_weight: Final[float] = getattr(model_preferences, "intelligencePriority", None) or 0.0

    # Gather raw metrics for each model
    scored: Final[list[_ScoredModel]] = []
    for name in model_names:
        try:
            info = _litellm.get_model_info(name)
        except Exception:
            continue
        input_cost = info.get("input_cost_per_token") or 0.0
        output_cost = info.get("output_cost_per_token") or 0.0
        total_cost = input_cost + output_cost
        max_output = info.get("max_output_tokens") or info.get("max_tokens") or 0
        output_tps = info.get("output_tokens_per_second") or 0.0
        scored.append(
            _ScoredModel(
                name=name,
                cost=total_cost,
                max_output=max_output,
                output_tps=output_tps,
            )
        )

    if not scored:
        return None

    # Min-max normalisation helpers
    def _normalise(values: list[float], invert: bool = False) -> list[float]:
        """Normalise to [0, 1].  If *invert*, lower raw → higher score."""
        lo, hi = min(values), max(values)
        if hi == lo:
            return [0.5] * len(values)  # all equal → neutral score
        normed = [(v - lo) / (hi - lo) for v in values]
        if invert:
            normed = [1.0 - n for n in normed]
        return normed

    costs: Final = [s.cost for s in scored]
    max_outputs: Final = [float(s.max_output) for s in scored]
    output_tps_values: Final = [s.output_tps for s in scored]

    # costPriority: lower cost → higher score  (invert)
    cost_scores: Final = _normalise(costs, invert=True)
    # speedPriority: use output_tokens_per_second if any model has it,
    # otherwise a neutral score (no reliable latency proxy is available).
    if any(v > 0 for v in output_tps_values):
        speed_scores = _normalise(output_tps_values, invert=False)
    else:
        speed_scores = [0.5] * len(scored)
    # intelligencePriority: higher max_output → smarter
    intel_scores: Final = _normalise(max_outputs, invert=False)

    best_name = None
    best_score = -1.0
    for i, entry in enumerate(scored):
        score = cost_weight * cost_scores[i] + speed_weight * speed_scores[i] + intel_weight * intel_scores[i]
        verbose_logger.debug(
            "MCP priority scoring: model=%s cost_score=%.3f speed_score=%.3f intel_score=%.3f → weighted=%.3f",
            entry.name,
            cost_scores[i],
            speed_scores[i],
            intel_scores[i],
            score,
        )
        if score > best_score:
            best_score = score
            best_name = entry.name

    return best_name


def _convert_mcp_content_to_openai(
    content: "SamplingMessageContentBlock | Sequence[SamplingMessageContentBlock]",
) -> "str | dict[str, object] | list[dict[str, object]]":
    """
    Convert MCP SamplingMessage content to OpenAI message content format.
    Handles:
    - TextContent → string or {"type": "text", "text": ...}
    - ImageContent → {"type": "image_url", "image_url": {"url": "data:..."}}
    - AudioContent → {"type": "input_audio", "input_audio": {...}}
    - ToolUseContent → function call representation
    - ToolResultContent → tool result representation
    - List of mixed content → list of content parts
    """
    if isinstance(content, list):
        parts: Final = []
        for item in content:
            converted = _convert_single_content(item)
            if isinstance(converted, list):
                parts.extend(converted)
            else:
                parts.append(converted)
        return parts
    return _convert_single_content(content)


@runtime_checkable
class _TextContentLike(Protocol):
    @property
    def text(self) -> object: ...


def _convert_single_content(
    content: object,
) -> "dict[str, object] | list[dict[str, object]]":
    """Convert a single MCP content item to OpenAI format.

    For text/image/audio content, returns a single content-part dict.
    For tool_use/tool_result, returns a dict with a ``_marker_type`` key
    so the caller (``_convert_mcp_messages_to_openai``) can hoist it to
    the correct message-level position (``tool_calls`` array or a
    separate ``role: "tool"`` message).
    """
    import json

    content_type: Final[str | None] = getattr(content, "type", None)
    if content_type == "text":
        if not isinstance(content, _TextContentLike):
            raise AttributeError(f"{type(content).__name__!r} object has no attribute 'text'")
        return {"type": "text", "text": content.text}
    elif content_type == "image":
        image_data: Final[str] = getattr(content, "data", "")
        image_mime_type: Final[str] = getattr(content, "mimeType", "image/png")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{image_mime_type};base64,{image_data}"},
        }
    elif content_type == "audio":
        audio_data: Final[str] = getattr(content, "data", "")
        audio_mime_type: Final[str] = getattr(content, "mimeType", "audio/wav")
        # Map MIME type to OpenAI audio format
        format_map: Final = {
            "audio/wav": "wav",
            "audio/mp3": "mp3",
            "audio/mpeg": "mp3",
            "audio/flac": "flac",
            "audio/ogg": "ogg",
        }
        audio_format: Final = format_map.get(audio_mime_type, "wav")
        return {
            "type": "input_audio",
            "input_audio": {"data": audio_data, "format": audio_format},
        }
    elif content_type == "tool_use":
        # ToolUseContent → proper OpenAI function-call representation.
        # The ``_marker_type`` key lets the message-level converter
        # hoist this into the ``tool_calls`` array on the assistant
        # message instead of embedding it inline as a content part.
        tool_use_id: Final[str] = getattr(content, "id", f"call_{id(content)}")
        tool_name: Final[str] = getattr(content, "name", "")
        tool_input: Final[dict[str, object]] = getattr(content, "input", {})
        return {
            "_marker_type": "tool_use",
            "id": tool_use_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(tool_input, default=str),
            },
        }
    elif content_type == "tool_result":
        # ToolResultContent → proper OpenAI tool-role message.
        # Marked so the message-level converter can emit it as a
        # separate ``{"role": "tool", ...}`` message.
        tool_result_use_id: Final = getattr(content, "toolUseId", "")
        nested_content: Final[Sequence[ContentBlock]] = getattr(content, "content", [])
        if isinstance(nested_content, list):
            text_parts = [getattr(c, "text", str(c)) for c in nested_content if getattr(c, "type", None) == "text"]
            result_text = "\n".join(text_parts) if text_parts else ""
        else:
            result_text = str(nested_content)
        return {
            "_marker_type": "tool_result",
            "role": "tool",
            "tool_call_id": tool_result_use_id,
            "content": result_text,
        }
    # Fallback: treat as text
    return {"type": "text", "text": str(content)}


def _convert_mcp_messages_to_openai(
    messages: list["SamplingMessage"],
    system_prompt: str | None = None,
) -> "Sequence[Mapping[str, object]]":
    """
    Convert MCP SamplingMessage list to OpenAI messages format.
    MCP messages use:
    - role: "user" | "assistant"
    - content: TextContent | ImageContent | AudioContent | ToolUseContent
              | ToolResultContent | list[...]
    OpenAI messages use:
    - role: "system" | "user" | "assistant" | "tool"
    - content: str | list[content_part]
    """
    openai_messages: Final[list[Mapping[str, object]]] = []
    # Add system prompt if provided
    if system_prompt:
        openai_messages.append({"role": "system", "content": system_prompt})
    for msg in messages:
        role = msg.role
        content = msg.content
        # Handle tool use content from assistant
        if role == "assistant" and _has_tool_use(content):
            tool_calls = _extract_tool_calls(content)
            if tool_calls:
                openai_msg: dict[str, object] = {
                    "role": "assistant",
                    "tool_calls": tool_calls,
                }
                # Also include any text content alongside tool calls
                text_parts = _extract_text_parts(content)
                if text_parts:
                    openai_msg["content"] = text_parts
                openai_messages.append(openai_msg)
                continue
        # Handle tool result content from user
        if role == "user" and _has_tool_result(content):
            tool_results = _extract_tool_results(content)
            for tool_result in tool_results:
                openai_messages.append(tool_result)
            continue
        # Standard text/image/audio message — also handles any stray
        # tool_use / tool_result that slipped past the fast-path checks
        # above (e.g. unexpected role, single non-list content).
        converted = _convert_mcp_content_to_openai(content)
        converted_parts: Sequence[Mapping[str, object]] = (
            converted if isinstance(converted, list) else ([converted] if isinstance(converted, dict) else [])
        )

        # Separate marker items from regular content parts
        tool_call_markers = []
        tool_result_markers = []
        regular_parts = []
        for part in converted_parts:
            marker = part.get("_marker_type") if isinstance(part, dict) else None
            if marker == "tool_use":
                # Strip the internal marker before emitting
                tc = {k: v for k, v in part.items() if k != "_marker_type"}
                tool_call_markers.append(tc)
            elif marker == "tool_result":
                tr = {k: v for k, v in part.items() if k != "_marker_type"}
                tool_result_markers.append(tr)
            else:
                regular_parts.append(part)

        # Emit assistant message with tool_calls if any were found
        if tool_call_markers:
            openai_msg_tc: dict[str, object] = {
                "role": "assistant",
                "tool_calls": tool_call_markers,
            }
            if regular_parts:
                openai_msg_tc["content"] = regular_parts
            openai_messages.append(openai_msg_tc)
        elif regular_parts:
            if isinstance(converted, str):
                openai_messages.append({"role": role, "content": converted})
            else:
                openai_messages.append({"role": role, "content": regular_parts})

        # Emit separate tool-result messages
        for tr in tool_result_markers:
            openai_messages.append(tr)

    return openai_messages


def _has_tool_use(content: "SamplingMessageContentBlock | Sequence[SamplingMessageContentBlock]") -> bool:
    """Check if content contains ToolUseContent."""
    if isinstance(content, list):
        return any(getattr(c, "type", None) == "tool_use" for c in content)
    content_type: Final[str | None] = getattr(content, "type", None)
    return content_type == "tool_use"


def _has_tool_result(content: "SamplingMessageContentBlock | Sequence[SamplingMessageContentBlock]") -> bool:
    """Check if content contains ToolResultContent."""
    if isinstance(content, list):
        return any(getattr(c, "type", None) == "tool_result" for c in content)
    content_type: Final[str | None] = getattr(content, "type", None)
    return content_type == "tool_result"


def _extract_tool_calls(
    content: "SamplingMessageContentBlock | Sequence[SamplingMessageContentBlock]",
) -> "Sequence[Mapping[str, object]]":
    """Extract OpenAI-format tool_calls from MCP ToolUseContent."""
    import json

    items: Final = content if isinstance(content, list) else [content]
    tool_calls: Final = []
    for item in items:
        if getattr(item, "type", None) == "tool_use":
            tool_calls.append(
                {
                    "id": getattr(item, "id", f"call_{id(item)}"),
                    "type": "function",
                    "function": {
                        "name": getattr(item, "name", ""),
                        "arguments": json.dumps(getattr(item, "input", {}), default=str),
                    },
                }
            )
    return tool_calls


def _extract_text_parts(
    content: "SamplingMessageContentBlock | Sequence[SamplingMessageContentBlock]",
) -> str | None:
    """Extract text parts from mixed content."""
    items: Final = content if isinstance(content, list) else [content]
    texts: Final = []
    for item in items:
        if getattr(item, "type", None) == "text":
            texts.append(getattr(item, "text", ""))
    return "\n".join(texts) if texts else None


def _extract_tool_results(
    content: "SamplingMessageContentBlock | Sequence[SamplingMessageContentBlock]",
) -> "Sequence[Mapping[str, object]]":
    """Extract OpenAI-format tool messages from MCP ToolResultContent."""
    items: Final = content if isinstance(content, list) else [content]
    results: Final = []
    for item in items:
        if getattr(item, "type", None) == "tool_result":
            tool_use_id = getattr(item, "toolUseId", "")
            # Extract text from nested content
            nested_content: Sequence[ContentBlock] = getattr(item, "content", [])
            if isinstance(nested_content, list):
                text_parts = [getattr(c, "text", str(c)) for c in nested_content if getattr(c, "type", None) == "text"]
                result_text = "\n".join(text_parts) if text_parts else ""
            else:
                result_text = str(nested_content)
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": result_text,
                }
            )
    return results


def _convert_mcp_tools_to_openai(
    tools: list["Tool"] | None,
) -> "Sequence[Mapping[str, object]] | None":
    """
    Convert MCP Tool definitions to OpenAI function calling format.
    MCP Tool: {name, description, inputSchema}
    OpenAI Tool: {type: "function", function: {name, description, parameters}}
    """
    if not tools:
        return None
    openai_tools: Final = []
    for tool in tools:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema
                or {
                    "type": "object",
                    "properties": {},
                },
            },
        }
        openai_tools.append(openai_tool)
    return openai_tools


def _convert_mcp_tool_choice_to_openai(
    tool_choice: Optional["ToolChoice"],
) -> "str | None":
    """
    Convert MCP ToolChoice to OpenAI tool_choice format.
    MCP: {mode: "auto"} | {mode: "required"} | {mode: "none"}
    OpenAI: "auto" | "required" | "none"
    """
    if not tool_choice:
        return None
    mode: Final = getattr(tool_choice, "mode", "auto")
    if mode == "auto":
        return "auto"
    elif mode == "required":
        return "required"
    elif mode == "none":
        return "none"
    return "auto"


class _SamplingToolCallFunction(Protocol):
    @property
    def name(self) -> str | None: ...

    @property
    def arguments(self) -> object: ...


class _SamplingToolCall(Protocol):
    @property
    def id(self) -> str | None: ...

    @property
    def function(self) -> _SamplingToolCallFunction: ...


class _SamplingResponseMessage(Protocol):
    @property
    def content(self) -> str | None: ...

    @property
    def tool_calls(self) -> Sequence[_SamplingToolCall] | None: ...


class _SamplingResponseChoice(Protocol):
    @property
    def message(self) -> _SamplingResponseMessage: ...

    @property
    def finish_reason(self) -> str | None: ...


class _SamplingCompletionResponse(Protocol):
    @property
    def choices(self) -> Sequence[_SamplingResponseChoice]: ...

    @property
    def model(self) -> str | None: ...


_TOOL_ARGUMENTS_ADAPTER: Final = TypeAdapter(dict[str, object])


def _parse_tool_arguments(arguments: object) -> "dict[str, object]":
    """Decode OpenAI tool-call arguments into the MCP ``input`` mapping."""
    import json

    if not isinstance(arguments, str):
        return _TOOL_ARGUMENTS_ADAPTER.validate_python(arguments)
    try:
        return _TOOL_ARGUMENTS_ADAPTER.validate_python(json.loads(arguments))
    except (json.JSONDecodeError, TypeError):
        return {"raw": arguments}


def _convert_openai_response_to_mcp_result(
    response: _SamplingCompletionResponse,
    model_name: str,
) -> Union["CreateMessageResult", "CreateMessageResultWithTools", "ErrorData"]:
    """
    Convert a litellm completion response to MCP CreateMessageResult.
    Args:
        response: The litellm ModelResponse.
        model_name: The model that was used.
    Returns:
        MCP CreateMessageResult or CreateMessageResultWithTools.
    """
    if not response.choices:
        verbose_logger.warning(
            "MCP sampling: LLM returned empty choices list for model=%s (possible content filter or provider error)",
            model_name,
        )
        return ErrorData(
            code=-1,
            message=(
                f"LLM returned no choices for model '{model_name}'. "
                "This may indicate content filtering or a provider-side error."
            ),
        )
    choice: Final = response.choices[0]
    message: Final = choice.message
    # Determine stop reason
    finish_reason: Final = getattr(choice, "finish_reason", "stop")
    if finish_reason == "tool_calls":
        stop_reason = "toolUse"
    elif finish_reason == "length":
        stop_reason = "maxTokens"
    else:
        stop_reason = "endTurn"
    actual_model: Final[str] = getattr(response, "model", model_name) or model_name
    # Check if response has tool calls
    tool_calls: Final = message.tool_calls if hasattr(message, "tool_calls") else None
    if tool_calls:
        # Build ToolUseContent items
        content_parts: Final[list[SamplingMessageContentBlock]] = []
        # Include text content if present
        if message.content:
            content_parts.append(TextContent(type="text", text=message.content))
        # Convert tool calls to MCP ToolUseContent
        for tc in tool_calls:
            content_parts.append(
                ToolUseContent.model_validate(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": _parse_tool_arguments(tc.function.arguments),
                    }
                )
            )
        return CreateMessageResultWithTools(
            role="assistant",
            content=content_parts,
            model=actual_model,
            stopReason=stop_reason,
        )
    # Simple text response
    text: Final = message.content or ""
    return CreateMessageResult(
        role="assistant",
        content=TextContent(type="text", text=text),
        model=actual_model,
        stopReason=stop_reason,
    )


async def _check_model_access(model: str, user_api_key_auth: "UserAPIKeyAuth | None") -> Optional["ErrorData"]:
    """Enforce model-permission checks for MCP sampling requests.

    Runs the same authorization checks as ``/chat/completions``:
    key-level, team-level, per-member, user-level, and project-level
    model restrictions. The model name comes from the upstream MCP
    server (untrusted input).

    Returns None if authorized, or an ErrorData describing the denial.
    """
    if user_api_key_auth is None:
        return None

    _api_key: Final = getattr(user_api_key_auth, "api_key", None)
    _token: Final = getattr(user_api_key_auth, "token", None)
    _user_role: Final = getattr(user_api_key_auth, "user_role", None)

    _has_real_credential: Final = bool(_api_key) or bool(_token)
    _is_admin: Final = _user_role in ("proxy_admin", "proxy_admin_viewer") if _user_role else False

    if not _has_real_credential and not _is_admin:
        verbose_logger.warning(
            "MCP sampling: denying model access for model=%s — "
            "auth context has no real LiteLLM credential (possible "
            "OAuth passthrough placeholder). api_key=%s, token=%s, role=%s",
            model,
            bool(_api_key),
            bool(_token),
            _user_role,
        )
        return ErrorData(
            code=-1,
            message=(
                "Model access denied: sampling requires a valid LiteLLM "
                "API key or admin credential. OAuth-only sessions cannot "
                "trigger proxy model calls without explicit authorization."
            ),
        )

    try:
        import litellm
        from litellm.proxy.auth.auth_checks import (
            _check_team_member_model_access,
            can_key_call_model,
            can_project_access_model,
            can_team_access_model,
            can_user_call_model,
            get_project_object,
            get_team_object,
            get_user_object,
        )

        try:
            from litellm.proxy.proxy_server import llm_router as _llm_router
        except ImportError:
            _llm_router = None

        await can_key_call_model(
            model=model,
            llm_model_list=getattr(litellm, "model_list", None),
            valid_token=user_api_key_auth,
            llm_router=_llm_router,
        )

        _team_id: Final[str | None] = getattr(user_api_key_auth, "team_id", None)
        _user_id: Final[str | None] = getattr(user_api_key_auth, "user_id", None)
        _project_id: Final[str | None] = getattr(user_api_key_auth, "project_id", None)

        try:
            from litellm.proxy.proxy_server import (
                prisma_client as _prisma_client,
            )
            from litellm.proxy.proxy_server import (
                proxy_logging_obj as _proxy_logging_obj,
            )
            from litellm.proxy.proxy_server import (
                user_api_key_cache as _user_api_key_cache,
            )
        except ImportError:
            _prisma_client = None
            _user_api_key_cache = None
            _proxy_logging_obj = None

        if _team_id and _prisma_client and _user_api_key_cache:
            try:
                team_obj = await get_team_object(
                    team_id=_team_id,
                    prisma_client=_prisma_client,
                    user_api_key_cache=_user_api_key_cache,
                    proxy_logging_obj=_proxy_logging_obj,
                )
            except Exception:
                team_obj = None

            if team_obj:
                await can_team_access_model(
                    model=model,
                    team_object=team_obj,
                    llm_router=_llm_router,
                    team_model_aliases=getattr(user_api_key_auth, "team_model_aliases", None),
                )
                if _user_id and _proxy_logging_obj:
                    await _check_team_member_model_access(
                        model=model,
                        team_object=team_obj,
                        valid_token=user_api_key_auth,
                        llm_router=_llm_router,
                        prisma_client=_prisma_client,
                        user_api_key_cache=_user_api_key_cache,
                        proxy_logging_obj=_proxy_logging_obj,
                    )
        elif not _team_id and _user_id and _prisma_client and _user_api_key_cache:
            try:
                user_obj = await get_user_object(
                    user_id=_user_id,
                    prisma_client=_prisma_client,
                    user_api_key_cache=_user_api_key_cache,
                    user_id_upsert=False,
                    proxy_logging_obj=_proxy_logging_obj,
                )
            except Exception:
                user_obj = None

            if user_obj:
                await can_user_call_model(
                    model=model,
                    llm_router=_llm_router,
                    user_object=user_obj,
                )

        if _project_id and _prisma_client and _user_api_key_cache:
            try:
                project_obj = await get_project_object(
                    project_id=_project_id,
                    prisma_client=_prisma_client,
                    user_api_key_cache=_user_api_key_cache,
                    proxy_logging_obj=_proxy_logging_obj,
                )
            except Exception:
                project_obj = None

            if project_obj:
                can_project_access_model(
                    model=model,
                    project_object=project_obj,
                    llm_router=_llm_router,
                )

        verbose_logger.debug(
            "MCP sampling: model access check passed for model=%s",
            model,
        )
        return None
    except Exception as access_err:
        verbose_logger.warning(
            "MCP sampling: model access denied for model=%s: %s",
            model,
            access_err,
        )
        return ErrorData(
            code=-1,
            message=(f"Model access denied: the API key is not authorized to use model '{model}'. {access_err}"),
        )


async def _run_budget_checks(
    model: str,
    user_api_key_auth: "UserAPIKeyAuth",
    raw_headers: dict[str, str] | None = None,
    client_ip: str | None = None,
) -> Optional["ErrorData"]:
    """Enforce key/team/user/org/global budget checks for sampling requests.

    Runs the same ``common_checks`` path that ``/chat/completions`` uses,
    so sampling cannot bypass budget limits.

    Returns None if all checks pass, or an ErrorData describing the denial.
    """
    try:
        import litellm
        from litellm.proxy.auth.auth_checks import (
            common_checks,
            get_team_object,
            get_user_object,
        )
        from litellm.proxy.proxy_server import (
            general_settings,
        )
        from litellm.proxy.proxy_server import (
            llm_router as _llm_router,
        )
        from litellm.proxy.proxy_server import (
            prisma_client as _prisma_client,
        )
        from litellm.proxy.proxy_server import (
            proxy_logging_obj as _proxy_logging_obj,
        )
        from litellm.proxy.proxy_server import (
            user_api_key_cache as _user_api_key_cache,
        )
    except ImportError as import_err:
        verbose_logger.warning("MCP sampling: budget check imports unavailable: %s", import_err)
        return None  # Can't enforce budgets without the modules

    _team_id: Final[str | None] = getattr(user_api_key_auth, "team_id", None)
    _user_id: Final[str | None] = getattr(user_api_key_auth, "user_id", None)

    team_obj = None
    if _team_id and _prisma_client and _user_api_key_cache:
        try:
            team_obj = await get_team_object(
                team_id=_team_id,
                prisma_client=_prisma_client,
                user_api_key_cache=_user_api_key_cache,
                proxy_logging_obj=_proxy_logging_obj,
            )
        except Exception:
            pass

    user_obj = None
    if _user_id and _prisma_client and _user_api_key_cache:
        try:
            user_obj = await get_user_object(
                user_id=_user_id,
                prisma_client=_prisma_client,
                user_api_key_cache=_user_api_key_cache,
                user_id_upsert=False,
                proxy_logging_obj=_proxy_logging_obj,
            )
        except Exception:
            pass

    dummy_request: Final = _build_sampling_request(
        raw_headers=raw_headers,
        client_ip=client_ip,
    )

    # Enforce virtual-key route restrictions: a key limited to MCP routes
    # must not be able to trigger a /chat/completions call via sampling.
    # This mirrors the RouteChecks.should_call_route gate that runs in
    # user_api_key_auth before common_checks for regular requests.
    try:
        from litellm.proxy.auth.route_checks import RouteChecks

        RouteChecks.should_call_route(
            route="/chat/completions",
            valid_token=user_api_key_auth,
            request=dummy_request,
        )
    except HTTPException as route_err:
        verbose_logger.warning(
            "MCP sampling: route check denied /chat/completions for key: %s",
            route_err.detail,
        )
        return ErrorData(
            code=-1,
            message=f"Sampling denied: virtual key is not allowed to call /chat/completions. {route_err.detail}",
        )

    global_proxy_spend: Final = getattr(litellm, "_global_proxy_spend", None)

    # Build request body and merge x-litellm-tags from MCP headers BEFORE
    # common_checks runs. _tag_max_budget_check inside common_checks only
    # inspects request_body; without this pre-merge, header-supplied tags
    # bypass per-tag budget enforcement (mirroring the regular auth path).
    request_body: Final[dict[str, object]] = {"model": model}
    try:
        from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup

        LiteLLMProxyRequestSetup.apply_client_tag_policy_pre_auth(
            request=dummy_request,
            request_data=request_body,
            user_api_key_dict=user_api_key_auth,
        )
    except Exception:
        # Non-fatal: tag merge is defense-in-depth; don't block sampling
        # if the merge utility is unavailable or fails.
        pass

    try:
        await common_checks(
            request_body=request_body,
            team_object=team_obj,
            user_object=user_obj,
            end_user_object=None,
            global_proxy_spend=global_proxy_spend,
            general_settings=general_settings or {},
            route="/chat/completions",
            llm_router=_llm_router,
            proxy_logging_obj=_proxy_logging_obj,
            valid_token=user_api_key_auth,
            request=dummy_request,
        )
    except Exception as budget_err:
        verbose_logger.warning(
            "MCP sampling: budget check failed for model=%s: %s",
            model,
            budget_err,
        )
        return ErrorData(
            code=-1,
            message=f"Sampling denied: {budget_err}",
        )

    verbose_logger.debug("MCP sampling: budget checks passed for model=%s", model)
    return None


def _build_sampling_request(
    raw_headers: dict[str, str] | None = None,
    client_ip: str | None = None,
) -> "Request":
    """The synthetic FastAPI Request for sampling sub-calls, carrying the original
    MCP connection's headers and client IP."""
    from litellm.proxy._experimental.mcp_server.utils import build_synthetic_mcp_request

    return build_synthetic_mcp_request(
        path="/mcp/sampling/createMessage",
        raw_headers=raw_headers,
        client_ip=client_ip,
    )


async def _build_completion_kwargs(
    params: "CreateMessageRequestParams",
    model: str,
    user_api_key_auth: "UserAPIKeyAuth",
    raw_headers: dict[str, str] | None,
    client_ip: str | None,
) -> dict[str, Any]:
    openai_messages: Final = _convert_mcp_messages_to_openai(
        messages=params.messages,
        system_prompt=params.systemPrompt,
    )
    completion_kwargs: Final[dict[str, object]] = {
        "model": model,
        "messages": openai_messages,
        "max_tokens": params.maxTokens,
    }
    if params.temperature is not None:
        completion_kwargs["temperature"] = params.temperature
    if params.stopSequences:
        completion_kwargs["stop"] = params.stopSequences
    openai_tools: Final = _convert_mcp_tools_to_openai(params.tools)
    if openai_tools:
        completion_kwargs["tools"] = openai_tools
    openai_tool_choice: Final = _convert_mcp_tool_choice_to_openai(params.toolChoice)
    if openai_tool_choice is not None:
        completion_kwargs["tool_choice"] = openai_tool_choice
    completion_kwargs["metadata"] = {"mcp_metadata": params.metadata} if params.metadata else {}

    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
    from litellm.proxy.proxy_server import proxy_config

    completion_kwargs["user"] = getattr(user_api_key_auth, "user_id", None)
    _dummy_request: Final = _build_sampling_request(raw_headers=raw_headers, client_ip=client_ip)
    return await add_litellm_data_to_request(
        data=completion_kwargs,
        request=_dummy_request,
        user_api_key_dict=user_api_key_auth,
        proxy_config=proxy_config,
    )


class _AcompletionCall(NamedTuple):
    fn: "Callable[..., Awaitable[ModelResponse | CustomStreamWrapper]]"


async def _run_guardrails_and_call_llm(
    completion_kwargs: dict[str, object],
    user_api_key_auth: "UserAPIKeyAuth",
) -> Any:
    try:
        from litellm.proxy.proxy_server import proxy_logging_obj as _plo

        if _plo is not None:
            completion_kwargs = await _plo.pre_call_hook(
                user_api_key_dict=user_api_key_auth,
                data=completion_kwargs,
                call_type="acompletion",
            )
    except ImportError:
        pass
    except Exception as guardrail_err:
        verbose_logger.warning(
            "MCP sampling: pre-call guardrail rejected request: %s",
            guardrail_err,
        )
        raise

    import litellm

    try:
        from litellm.proxy.proxy_server import llm_router

        if llm_router is not None:
            return await _AcompletionCall(fn=llm_router.acompletion).fn(**completion_kwargs)
        return await _AcompletionCall(fn=litellm.acompletion).fn(**completion_kwargs)
    except ImportError:
        return await _AcompletionCall(fn=litellm.acompletion).fn(**completion_kwargs)


async def handle_sampling_create_message(
    context: "RequestContext[ClientSession, object]",
    params: "CreateMessageRequestParams",
    default_model: str | None = None,
    user_api_key_auth: "UserAPIKeyAuth | None" = None,
    raw_headers: dict[str, str] | None = None,
    client_ip: str | None = None,
) -> Union["CreateMessageResult", "CreateMessageResultWithTools", "ErrorData"]:
    """
    Handle an MCP sampling/createMessage request by routing through LiteLLM.
    This is the main entry point called by the MCP client session when an
    upstream MCP server requests LLM inference.
    Args:
        context: MCP RequestContext (contains session info).
        params: The CreateMessageRequestParams from the MCP server.
        default_model: Default model to use if no preferences match.
        user_api_key_auth: Auth context for the requesting user.
        raw_headers: Original HTTP headers from the MCP connection.
            Forwarded into the internal acompletion call so that
            header-dependent guardrails, IP-routing, trace-id
            correlation, and forward_llm_provider_auth_headers
            work correctly for sampling sub-calls.
        client_ip: Original client IP address for IP-based guardrails.
    Returns:
        CreateMessageResult with the LLM's response, or ErrorData on failure.
    """
    if not MCP_SAMPLING_AVAILABLE:
        return ErrorData(
            code=-1,
            message="MCP sampling is not available (mcp package not installed)",
        )

    if user_api_key_auth is None:
        return ErrorData(
            code=-1,
            message=(
                "Sampling requires an authenticated user context. "
                "Internal or unauthenticated sessions cannot trigger "
                "upstream-initiated model calls."
            ),
        )

    try:
        model: Final = _resolve_model_from_preferences(
            model_preferences=params.modelPreferences,
            default_model=default_model,
        )
        verbose_logger.info(
            "MCP sampling: resolved model=%s from preferences=%s",
            model,
            params.modelPreferences,
        )

        access_denial: Final = await _check_model_access(model, user_api_key_auth)
        if access_denial is not None:
            return access_denial

        budget_denial: Final = await _run_budget_checks(
            model=model,
            user_api_key_auth=user_api_key_auth,
            raw_headers=raw_headers,
            client_ip=client_ip,
        )
        if budget_denial is not None:
            return budget_denial

        completion_kwargs: Final = await _build_completion_kwargs(
            params=params,
            model=model,
            user_api_key_auth=user_api_key_auth,
            raw_headers=raw_headers,
            client_ip=client_ip,
        )

        openai_messages: Final[Sequence[Mapping[str, object]]] = completion_kwargs["messages"]
        openai_tools: Final = completion_kwargs.get("tools")
        verbose_logger.debug(
            "MCP sampling: calling litellm.acompletion with model=%s, num_messages=%d, has_tools=%s",
            model,
            len(openai_messages),
            bool(openai_tools),
        )

        response: Final[_SamplingCompletionResponse] = await _run_guardrails_and_call_llm(
            completion_kwargs=completion_kwargs,
            user_api_key_auth=user_api_key_auth,
        )

        result: Final = _convert_openai_response_to_mcp_result(response=response, model_name=model)
        verbose_logger.info(
            "MCP sampling: completed successfully, model=%s, stopReason=%s",
            getattr(result, "model", "unknown"),
            getattr(result, "stopReason", "unknown"),
        )
        return result
    except Exception as e:
        from litellm.exceptions import (
            AuthenticationError,
            BudgetExceededError,
            ContextWindowExceededError,
            PermissionDeniedError,
            RateLimitError,
            ServiceUnavailableError,
        )
        from litellm.proxy._types import ProxyException

        if isinstance(
            e,
            (
                HTTPException,
                BudgetExceededError,
                RateLimitError,
                AuthenticationError,
                PermissionDeniedError,
                ContextWindowExceededError,
                ServiceUnavailableError,
                ProxyException,
            ),
        ):
            raise

        verbose_logger.exception("MCP sampling handler failed: %s", e)
        return ErrorData(
            code=-1,
            message=f"Sampling failed: {e}",
        )
