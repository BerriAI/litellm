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

from typing import Any, Dict, List, Optional, Union
import typing

if typing.TYPE_CHECKING:
    from litellm.proxy.utils import ProxyLogging

from litellm._logging import verbose_logger

from fastapi import HTTPException

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
    default_model: Optional[str] = None,
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
    available_model_names: list = []
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
            hint_name = getattr(hint, "name", None)
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
        best = _select_model_by_priority(available_model_names, model_preferences)
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
    default_sampling_model = getattr(litellm, "default_mcp_sampling_model", None)
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


def _select_model_by_priority(
    model_names: List[str],
    model_preferences: "ModelPreferences",
) -> Optional[str]:
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

    cost_weight = getattr(model_preferences, "costPriority", None) or 0.0
    speed_weight = getattr(model_preferences, "speedPriority", None) or 0.0
    intel_weight = getattr(model_preferences, "intelligencePriority", None) or 0.0

    # Gather raw metrics for each model
    scored: List[Dict[str, Any]] = []
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
            {
                "name": name,
                "cost": total_cost,
                "max_output": max_output,
                "output_tps": output_tps,
            }
        )

    if not scored:
        return None

    # Min-max normalisation helpers
    def _normalise(values: List[float], invert: bool = False) -> List[float]:
        """Normalise to [0, 1].  If *invert*, lower raw → higher score."""
        lo, hi = min(values), max(values)
        if hi == lo:
            return [0.5] * len(values)  # all equal → neutral score
        normed = [(v - lo) / (hi - lo) for v in values]
        if invert:
            normed = [1.0 - n for n in normed]
        return normed

    costs = [s["cost"] for s in scored]
    max_outputs = [float(s["max_output"]) for s in scored]
    output_tps_values = [s["output_tps"] for s in scored]

    # costPriority: lower cost → higher score  (invert)
    cost_scores = _normalise(costs, invert=True)
    # speedPriority: use output_tokens_per_second if any model has it,
    # otherwise a neutral score (no reliable latency proxy is available).
    if any(v > 0 for v in output_tps_values):
        speed_scores = _normalise(output_tps_values, invert=False)
    else:
        speed_scores = [0.5] * len(scored)
    # intelligencePriority: higher max_output → smarter
    intel_scores = _normalise(max_outputs, invert=False)

    best_name = None
    best_score = -1.0
    for i, entry in enumerate(scored):
        score = cost_weight * cost_scores[i] + speed_weight * speed_scores[i] + intel_weight * intel_scores[i]
        verbose_logger.debug(
            "MCP priority scoring: model=%s cost_score=%.3f speed_score=%.3f intel_score=%.3f → weighted=%.3f",
            entry["name"],
            cost_scores[i],
            speed_scores[i],
            intel_scores[i],
            score,
        )
        if score > best_score:
            best_score = score
            best_name = entry["name"]

    return best_name


def _convert_mcp_content_to_openai(
    content: Any,
) -> Union[str, Dict[str, Any], List[Dict[str, Any]]]:
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
        parts = []
        for item in content:
            converted = _convert_single_content(item)
            if isinstance(converted, list):
                parts.extend(converted)
            else:
                parts.append(converted)
        return parts
    return _convert_single_content(content)


def _convert_single_content(
    content: Any,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Convert a single MCP content item to OpenAI format.

    For text/image/audio content, returns a single content-part dict.
    For tool_use/tool_result, returns a dict with a ``_marker_type`` key
    so the caller (``_convert_mcp_messages_to_openai``) can hoist it to
    the correct message-level position (``tool_calls`` array or a
    separate ``role: "tool"`` message).
    """
    import json

    content_type = getattr(content, "type", None)
    if content_type == "text":
        return {"type": "text", "text": content.text}
    elif content_type == "image":
        data = getattr(content, "data", "")
        mime_type = getattr(content, "mimeType", "image/png")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{data}"},
        }
    elif content_type == "audio":
        data = getattr(content, "data", "")
        mime_type = getattr(content, "mimeType", "audio/wav")
        # Map MIME type to OpenAI audio format
        format_map = {
            "audio/wav": "wav",
            "audio/mp3": "mp3",
            "audio/mpeg": "mp3",
            "audio/flac": "flac",
            "audio/ogg": "ogg",
        }
        audio_format = format_map.get(mime_type, "wav")
        return {
            "type": "input_audio",
            "input_audio": {"data": data, "format": audio_format},
        }
    elif content_type == "tool_use":
        # ToolUseContent → proper OpenAI function-call representation.
        # The ``_marker_type`` key lets the message-level converter
        # hoist this into the ``tool_calls`` array on the assistant
        # message instead of embedding it inline as a content part.
        return {
            "_marker_type": "tool_use",
            "id": getattr(content, "id", f"call_{id(content)}"),
            "type": "function",
            "function": {
                "name": getattr(content, "name", ""),
                "arguments": json.dumps(getattr(content, "input", {}), default=str),
            },
        }
    elif content_type == "tool_result":
        # ToolResultContent → proper OpenAI tool-role message.
        # Marked so the message-level converter can emit it as a
        # separate ``{"role": "tool", ...}`` message.
        tool_use_id = getattr(content, "toolUseId", "")
        nested_content = getattr(content, "content", [])
        if isinstance(nested_content, list):
            text_parts = [getattr(c, "text", str(c)) for c in nested_content if getattr(c, "type", None) == "text"]
            result_text = "\n".join(text_parts) if text_parts else ""
        else:
            result_text = str(nested_content)
        return {
            "_marker_type": "tool_result",
            "role": "tool",
            "tool_call_id": tool_use_id,
            "content": result_text,
        }
    # Fallback: treat as text
    return {"type": "text", "text": str(content)}


def _convert_mcp_messages_to_openai(
    messages: List["SamplingMessage"],
    system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
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
    openai_messages: List[Dict[str, Any]] = []
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
                openai_msg: Dict[str, Any] = {
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
        converted_parts = (
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
            openai_msg_tc: Dict[str, Any] = {
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


def _has_tool_use(content: Any) -> bool:
    """Check if content contains ToolUseContent."""
    if isinstance(content, list):
        return any(getattr(c, "type", None) == "tool_use" for c in content)
    return getattr(content, "type", None) == "tool_use"


def _has_tool_result(content: Any) -> bool:
    """Check if content contains ToolResultContent."""
    if isinstance(content, list):
        return any(getattr(c, "type", None) == "tool_result" for c in content)
    return getattr(content, "type", None) == "tool_result"


def _extract_tool_calls(content: Any) -> List[Dict[str, Any]]:
    """Extract OpenAI-format tool_calls from MCP ToolUseContent."""
    import json

    items = content if isinstance(content, list) else [content]
    tool_calls = []
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


def _extract_text_parts(content: Any) -> Optional[str]:
    """Extract text parts from mixed content."""
    items = content if isinstance(content, list) else [content]
    texts = []
    for item in items:
        if getattr(item, "type", None) == "text":
            texts.append(getattr(item, "text", ""))
    return "\n".join(texts) if texts else None


def _extract_tool_results(content: Any) -> List[Dict[str, Any]]:
    """Extract OpenAI-format tool messages from MCP ToolResultContent."""
    items = content if isinstance(content, list) else [content]
    results = []
    for item in items:
        if getattr(item, "type", None) == "tool_result":
            tool_use_id = getattr(item, "toolUseId", "")
            # Extract text from nested content
            nested_content = getattr(item, "content", [])
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
    tools: Optional[List["Tool"]],
) -> Optional[List[Dict[str, Any]]]:
    """
    Convert MCP Tool definitions to OpenAI function calling format.
    MCP Tool: {name, description, inputSchema}
    OpenAI Tool: {type: "function", function: {name, description, parameters}}
    """
    if not tools:
        return None
    openai_tools = []
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
) -> Optional[Union[str, Dict[str, Any]]]:
    """
    Convert MCP ToolChoice to OpenAI tool_choice format.
    MCP: {mode: "auto"} | {mode: "required"} | {mode: "none"}
    OpenAI: "auto" | "required" | "none"
    """
    if not tool_choice:
        return None
    mode = getattr(tool_choice, "mode", "auto")
    if mode == "auto":
        return "auto"
    elif mode == "required":
        return "required"
    elif mode == "none":
        return "none"
    return "auto"


def _convert_openai_response_to_mcp_result(
    response: Any,
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
    choice = response.choices[0]
    message = choice.message
    # Determine stop reason
    finish_reason = getattr(choice, "finish_reason", "stop")
    if finish_reason == "tool_calls":
        stop_reason = "toolUse"
    elif finish_reason == "length":
        stop_reason = "maxTokens"
    else:
        stop_reason = "endTurn"
    actual_model = getattr(response, "model", model_name) or model_name
    # Check if response has tool calls
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        # Build ToolUseContent items
        content_parts: "List[Any]" = []
        # Include text content if present
        if message.content:
            content_parts.append(TextContent(type="text", text=message.content))
        # Convert tool calls to MCP ToolUseContent
        for tc in tool_calls:
            import json

            tool_input = tc.function.arguments
            if isinstance(tool_input, str):
                try:
                    tool_input = json.loads(tool_input)
                except (json.JSONDecodeError, TypeError):
                    tool_input = {"raw": tool_input}
            content_parts.append(
                ToolUseContent(
                    type="tool_use",
                    id=tc.id,
                    name=tc.function.name,
                    input=tool_input,
                )
            )
        return CreateMessageResultWithTools(
            role="assistant",
            content=content_parts,
            model=actual_model,
            stopReason=stop_reason,
        )
    # Simple text response
    text = message.content or ""
    return CreateMessageResult(
        role="assistant",
        content=TextContent(type="text", text=text),
        model=actual_model,
        stopReason=stop_reason,
    )


async def _check_model_access(model: str, user_api_key_auth: Any) -> Optional["ErrorData"]:
    """Enforce model-permission checks for MCP sampling requests.

    Runs the same authorization checks as ``/chat/completions``:
    key-level, team-level, per-member, user-level, and project-level
    model restrictions. The model name comes from the upstream MCP
    server (untrusted input).

    Returns None if authorized, or an ErrorData describing the denial.
    """
    if user_api_key_auth is None:
        return None

    _api_key = getattr(user_api_key_auth, "api_key", None)
    _token = getattr(user_api_key_auth, "token", None)
    _user_role = getattr(user_api_key_auth, "user_role", None)

    _has_real_credential = bool(_api_key) or bool(_token)
    _is_admin = _user_role in ("proxy_admin", "proxy_admin_viewer") if _user_role else False

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
            can_key_call_model,
            can_team_access_model,
            can_user_call_model,
            can_project_access_model,
            _check_team_member_model_access,
            get_team_object,
            get_user_object,
            get_project_object,
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

        _team_id = getattr(user_api_key_auth, "team_id", None)
        _user_id = getattr(user_api_key_auth, "user_id", None)
        _project_id = getattr(user_api_key_auth, "project_id", None)

        try:
            from litellm.proxy.proxy_server import (
                prisma_client as _prisma_client,
                user_api_key_cache as _user_api_key_cache,
                proxy_logging_obj as _proxy_logging_obj,
            )
        except ImportError:
            _prisma_client = None
            _user_api_key_cache = None  # type: ignore[assignment]
            _proxy_logging_obj = None  # type: ignore[assignment]

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
    user_api_key_auth: Any,
    raw_headers: Optional[Dict[str, str]] = None,
    client_ip: Optional[str] = None,
) -> Optional["ErrorData"]:
    """Enforce key/team/user/org/global budget checks for sampling requests.

    Runs the same ``common_checks`` path that ``/chat/completions`` uses,
    so sampling cannot bypass budget limits.

    Returns None if all checks pass, or an ErrorData describing the denial.
    """
    try:
        from litellm.proxy.auth.auth_checks import common_checks
        from litellm.proxy.proxy_server import (
            general_settings,
            llm_router as _llm_router,
            prisma_client as _prisma_client,
            proxy_logging_obj as _proxy_logging_obj,
            user_api_key_cache as _user_api_key_cache,
        )
        from litellm.proxy.auth.auth_checks import (
            get_team_object,
            get_user_object,
        )
        import litellm
    except ImportError as import_err:
        verbose_logger.warning("MCP sampling: budget check imports unavailable: %s", import_err)
        return None  # Can't enforce budgets without the modules

    _team_id = getattr(user_api_key_auth, "team_id", None)
    _user_id = getattr(user_api_key_auth, "user_id", None)

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

    dummy_request = _build_sampling_request(
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

    global_proxy_spend = getattr(litellm, "_global_proxy_spend", None)

    # Build request body and merge x-litellm-tags from MCP headers BEFORE
    # common_checks runs. _tag_max_budget_check inside common_checks only
    # inspects request_body; without this pre-merge, header-supplied tags
    # bypass per-tag budget enforcement (mirroring the regular auth path).
    request_body: Dict[str, Any] = {"model": model}
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
            proxy_logging_obj=typing.cast("ProxyLogging", _proxy_logging_obj),
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
    raw_headers: Optional[Dict[str, str]] = None,
    client_ip: Optional[str] = None,
) -> Any:
    """Build a synthetic FastAPI Request for sampling sub-calls.

    Converts the original MCP connection's HTTP headers into ASGI
    scope format so that ``add_litellm_data_to_request`` can apply
    header-dependent guardrails, tag-based routing, trace correlation,
    and ``forward_llm_provider_auth_headers``.

    Key fields populated:
    - **headers**: All original HTTP headers are forwarded (except
      hop-by-hop: content-length, transfer-encoding).  This ensures
      ``traceparent``, ``authorization``, ``user-agent``, and
      ``x-litellm-api-key`` are visible to pre-call utils.
    - **client**: The ASGI ``(host, port)`` tuple so that
      ``request.client.host`` returns the real client IP for
      IP-based routing and guardrails.
    - **server**: Derived from the running proxy's ``server_host``
      / ``server_port`` when available, avoiding the misleading
      ``127.0.0.1:0`` placeholder.
    - **x-forwarded-for**: Injected from ``client_ip`` if the
      original headers don't already carry it, as a fallback for
      IP attribution.
    """
    from fastapi import Request

    # --- Build ASGI headers ---
    _scope_headers: list = [(b"content-type", b"application/json")]
    # Hop-by-hop headers that must NOT be forwarded into the
    # synthetic request (they describe the original HTTP framing,
    # not the logical request).
    _HOP_BY_HOP = frozenset(
        {
            "content-length",
            "transfer-encoding",
            "connection",
            "keep-alive",
            "upgrade",
            "te",
            "trailer",
        }
    )
    if raw_headers:
        for hdr_name, hdr_value in raw_headers.items():
            _key = hdr_name.lower()
            # Skip content-type (already set), x-forwarded-for (use resolved
            # client_ip instead to prevent spoofing), and hop-by-hop headers
            if _key in {"content-type", "x-forwarded-for"} or _key in _HOP_BY_HOP:
                continue
            _scope_headers.append(
                (
                    _key.encode("latin-1", errors="replace"),
                    hdr_value.encode("utf-8"),
                )
            )

    # Inject x-forwarded-for from captured client_ip if the
    # original headers don't already carry it
    if client_ip and not any(h[0] == b"x-forwarded-for" for h in _scope_headers):
        _scope_headers.append((b"x-forwarded-for", client_ip.encode("utf-8")))

    # --- Derive server (host, port) from the running proxy ---
    _server_host = "127.0.0.1"
    _server_port = 4000  # LiteLLM default
    try:
        import litellm.proxy.proxy_server as proxy_server

        _proxy_host = getattr(proxy_server, "server_host", None)
        _proxy_port = getattr(proxy_server, "server_port", None)

        if _proxy_host:
            _server_host = str(_proxy_host)
        if _proxy_port:
            _server_port = int(_proxy_port)
    except (ImportError, AttributeError, TypeError, ValueError):
        pass

    # --- Build ASGI client tuple for request.client.host ---
    _client_tuple = None
    if client_ip:
        _client_tuple = (client_ip, 0)

    scope: Dict[str, Any] = {
        "type": "http",
        "method": "POST",
        "path": "/mcp/sampling/createMessage",
        "scheme": "http",
        "server": (_server_host, _server_port),
        "query_string": b"",
        "root_path": "",
        "headers": _scope_headers,
    }
    if _client_tuple is not None:
        scope["client"] = _client_tuple

    return Request(scope=scope)


async def _build_completion_kwargs(
    params: "CreateMessageRequestParams",
    model: str,
    user_api_key_auth: Any,
    raw_headers: Optional[Dict[str, str]],
    client_ip: Optional[str],
) -> Dict[str, Any]:
    openai_messages = _convert_mcp_messages_to_openai(
        messages=params.messages,
        system_prompt=params.systemPrompt,
    )
    completion_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": openai_messages,
        "max_tokens": params.maxTokens,
    }
    if params.temperature is not None:
        completion_kwargs["temperature"] = params.temperature
    if params.stopSequences:
        completion_kwargs["stop"] = params.stopSequences
    openai_tools = _convert_mcp_tools_to_openai(params.tools)
    if openai_tools:
        completion_kwargs["tools"] = openai_tools
    openai_tool_choice = _convert_mcp_tool_choice_to_openai(params.toolChoice)
    if openai_tool_choice is not None:
        completion_kwargs["tool_choice"] = openai_tool_choice
    completion_kwargs["metadata"] = {}
    if params.metadata:
        completion_kwargs["metadata"]["mcp_metadata"] = params.metadata

    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
    from litellm.proxy.proxy_server import proxy_config

    completion_kwargs["user"] = getattr(user_api_key_auth, "user_id", None)
    _dummy_request = _build_sampling_request(raw_headers=raw_headers, client_ip=client_ip)
    completion_kwargs = await add_litellm_data_to_request(
        data=completion_kwargs,
        request=_dummy_request,
        user_api_key_dict=user_api_key_auth,
        proxy_config=proxy_config,
    )
    return completion_kwargs


async def _run_guardrails_and_call_llm(
    completion_kwargs: Dict[str, Any],
    user_api_key_auth: Any,
) -> Any:
    try:
        from litellm.proxy.proxy_server import proxy_logging_obj as _plo

        if _plo is not None:
            completion_kwargs = await typing.cast("ProxyLogging", _plo).pre_call_hook(
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
            return await llm_router.acompletion(**completion_kwargs)
        return await litellm.acompletion(**completion_kwargs)
    except ImportError:
        return await litellm.acompletion(**completion_kwargs)


async def handle_sampling_create_message(
    context: Any,
    params: "CreateMessageRequestParams",
    default_model: Optional[str] = None,
    user_api_key_auth: Optional[Any] = None,
    raw_headers: Optional[Dict[str, str]] = None,
    client_ip: Optional[str] = None,
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
        model = _resolve_model_from_preferences(
            model_preferences=params.modelPreferences,
            default_model=default_model,
        )
        verbose_logger.info(
            "MCP sampling: resolved model=%s from preferences=%s",
            model,
            params.modelPreferences,
        )

        access_denial = await _check_model_access(model, user_api_key_auth)
        if access_denial is not None:
            return access_denial

        budget_denial = await _run_budget_checks(
            model=model,
            user_api_key_auth=user_api_key_auth,
            raw_headers=raw_headers,
            client_ip=client_ip,
        )
        if budget_denial is not None:
            return budget_denial

        completion_kwargs = await _build_completion_kwargs(
            params=params,
            model=model,
            user_api_key_auth=user_api_key_auth,
            raw_headers=raw_headers,
            client_ip=client_ip,
        )

        openai_messages = completion_kwargs["messages"]
        openai_tools = completion_kwargs.get("tools")
        verbose_logger.debug(
            "MCP sampling: calling litellm.acompletion with model=%s, num_messages=%d, has_tools=%s",
            model,
            len(openai_messages),
            bool(openai_tools),
        )

        response = await _run_guardrails_and_call_llm(
            completion_kwargs=completion_kwargs,
            user_api_key_auth=user_api_key_auth,
        )

        result = _convert_openai_response_to_mcp_result(response=response, model_name=model)
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
            message=f"Sampling failed: {str(e)}",
        )
