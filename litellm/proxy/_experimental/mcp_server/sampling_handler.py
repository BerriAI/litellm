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
except ImportError:
    MCP_SAMPLING_AVAILABLE = False
# Maximum number of sampling iterations to prevent infinite loops
DEFAULT_MAX_SAMPLING_ITERATIONS = 10


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
                return hint_name
            # Try substring match against known models
            for model_name in available_model_names:
                if hint_name.lower() in model_name.lower():
                    return model_name
    # Use default model from caller
    if default_model:
        return default_model
    # Fall back to first available model
    if available_model_names:
        return available_model_names[0]
    # Last resort - use LiteLLM default or raise error
    default_sampling_model = getattr(litellm, "default_mcp_sampling_model", None)
    if default_sampling_model:
        return default_sampling_model
    raise ValueError(
        "No model could be resolved for MCP sampling. Please configure 'default_mcp_sampling_model' in your LiteLLM configuration."
    )


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


def _convert_single_content(content: Any) -> Dict[str, Any]:
    """Convert a single MCP content item to OpenAI format."""
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
        # ToolUseContent → represents the assistant calling a tool
        return {
            "type": "text",
            "text": f"[Tool call: {getattr(content, 'name', 'unknown')}]",
        }
    elif content_type == "tool_result":
        # ToolResultContent → represents tool results
        tool_content = getattr(content, "content", [])
        if isinstance(tool_content, list) and tool_content:
            texts = [
                getattr(c, "text", str(c))
                for c in tool_content
                if getattr(c, "type", None) == "text"
            ]
            return {"type": "text", "text": "\n".join(texts) if texts else ""}
        return {"type": "text", "text": str(tool_content)}
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
        # Standard text/image/audio message
        converted = _convert_mcp_content_to_openai(content)
        if isinstance(converted, str):
            openai_messages.append({"role": role, "content": converted})
        elif isinstance(converted, dict):
            openai_messages.append({"role": role, "content": [converted]})
        elif isinstance(converted, list):
            openai_messages.append({"role": role, "content": converted})
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
                        "arguments": json.dumps(
                            getattr(item, "input", {}), default=str
                        ),
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
                text_parts = [
                    getattr(c, "text", str(c))
                    for c in nested_content
                    if getattr(c, "type", None) == "text"
                ]
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
) -> Union["CreateMessageResult", "CreateMessageResultWithTools"]:
    """
    Convert a litellm completion response to MCP CreateMessageResult.
    Args:
        response: The litellm ModelResponse.
        model_name: The model that was used.
    Returns:
        MCP CreateMessageResult or CreateMessageResultWithTools.
    """
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


async def handle_sampling_create_message(
    context: Any,
    params: "CreateMessageRequestParams",
    default_model: Optional[str] = None,
    user_api_key_auth: Optional[Any] = None,
    max_iterations: int = DEFAULT_MAX_SAMPLING_ITERATIONS,
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
        max_iterations: Maximum tool-calling iterations to prevent loops.
    Returns:
        CreateMessageResult with the LLM's response, or ErrorData on failure.
    """
    if not MCP_SAMPLING_AVAILABLE:
        return ErrorData(
            code=-1,
            message="MCP sampling is not available (mcp package not installed)",
        )
    try:
        import litellm

        # 1. Resolve model from preferences
        model = _resolve_model_from_preferences(
            model_preferences=params.modelPreferences,
            default_model=default_model,
        )
        verbose_logger.info(
            "MCP sampling: resolved model=%s from preferences=%s",
            model,
            params.modelPreferences,
        )
        # 2. Convert MCP messages to OpenAI format
        openai_messages = _convert_mcp_messages_to_openai(
            messages=params.messages,
            system_prompt=params.systemPrompt,
        )
        # 3. Build completion kwargs
        completion_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": openai_messages,
            "max_tokens": params.maxTokens,
        }
        if params.temperature is not None:
            completion_kwargs["temperature"] = params.temperature
        if params.stopSequences:
            completion_kwargs["stop"] = params.stopSequences
        # 4. Convert tools and tool_choice if provided
        openai_tools = _convert_mcp_tools_to_openai(params.tools)
        if openai_tools:
            completion_kwargs["tools"] = openai_tools
        openai_tool_choice = _convert_mcp_tool_choice_to_openai(params.toolChoice)
        if openai_tool_choice is not None:
            completion_kwargs["tool_choice"] = openai_tool_choice
        # 5. Add metadata for tracking
        if params.metadata:
            completion_kwargs["metadata"] = params.metadata

        # 6. Inject auth context for cost tracking
        if user_api_key_auth:
            from fastapi import Request

            from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
            from litellm.proxy.proxy_server import proxy_config

            completion_kwargs["user"] = getattr(user_api_key_auth, "user_id", None)

            # We need a dummy FastAPI request object because add_litellm_data_to_request expects it
            _dummy_request = Request(
                scope={
                    "type": "http",
                    "method": "POST",
                    "path": "/mcp/sampling/createMessage",
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            completion_kwargs = await add_litellm_data_to_request(
                data=completion_kwargs,
                request=_dummy_request,
                user_api_key_dict=user_api_key_auth,
                proxy_config=proxy_config,
            )

        verbose_logger.debug(
            "MCP sampling: calling litellm.acompletion with model=%s, num_messages=%d, has_tools=%s",
            model,
            len(openai_messages),
            bool(openai_tools),
        )

        # 7. Call LiteLLM
        # Use proxy's llm_router if available, else fallback to litellm.acompletion
        try:
            from litellm.proxy.proxy_server import llm_router

            if llm_router is not None:
                response = await llm_router.acompletion(**completion_kwargs)
            else:
                response = await litellm.acompletion(**completion_kwargs)
        except ImportError:
            response = await litellm.acompletion(**completion_kwargs)
        # 7. Convert response to MCP format
        result = _convert_openai_response_to_mcp_result(
            response=response,
            model_name=model,
        )
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
            PermissionDeniedError,
            RateLimitError,
        )

        if isinstance(
            e,
            (
                BudgetExceededError,
                RateLimitError,
                AuthenticationError,
                PermissionDeniedError,
            ),
        ):
            raise

        verbose_logger.exception("MCP sampling handler failed: %s", e)
        return ErrorData(
            code=-1,
            message=f"Sampling failed: {str(e)}",
        )
