"""
OCI Generative AI — Cohere-specific chat transformation helpers.

Handles message history building, tool definition adaptation, non-streaming
response parsing, and streaming chunk parsing for models served with
``apiFormat="COHERE"`` (e.g. ``cohere.command-*``).
"""

import datetime
import json
import uuid
from typing import Any, Dict, List, Optional

from litellm.llms.oci.common_utils import (
    OCI_JSON_TO_PYTHON_TYPES,
    OCIError,
    enrich_cohere_param_description,
    resolve_oci_schema_anyof,
    resolve_oci_schema_refs,
    sanitize_oci_schema,
)
from litellm.types.llms.oci import (
    CohereChatResult,
    CohereMessage,
    CohereParameterDefinition,
    CohereStreamChunk,
    CohereTool,
    CohereToolCall,
    CohereToolMessage,
    CohereToolResult,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import (
    Choices,
    Delta,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)
from litellm.types.utils import Usage


def _extract_text_content(content: Any) -> str:
    """Return the plain-text representation of a message content value."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content)


def adapt_messages_to_cohere_standard(
    messages: List[AllMessageValues],
) -> List[CohereMessage]:
    """Build a Cohere ``chatHistory`` list from an OpenAI-format message array.

    - All messages except the *last user message* are included. The caller pulls
      the last user message into the request's top-level ``message`` field, so
      trailing tool results (the standard agentic continuation pattern) still
      appear in ``chatHistory`` and reach the model.
    - If no user message exists, every message is included (no slice).
    - Tool results are expressed as OCI ``CohereToolMessage.toolResults`` entries,
      with the originating call's name and parameters resolved from the preceding
      assistant message via a ``tool_call_id`` lookup.
    """
    # First pass: build tool_call_id → CohereToolCall so tool-result messages can
    # reference the originating call by name and parameters.
    tool_call_lookup: Dict[str, CohereToolCall] = {}
    for msg in messages:
        if msg.get("role") == "assistant":
            tool_calls_raw: Any = msg.get("tool_calls") or []
            for tc in tool_calls_raw:
                tc_id = tc.get("id", "")
                raw_args: Any = tc.get("function", {}).get("arguments", "{}")
                try:
                    params: Dict[str, Any] = (
                        json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    )
                except json.JSONDecodeError:
                    params = {}
                tool_call_lookup[tc_id] = CohereToolCall(
                    name=str(tc.get("function", {}).get("name", "")),
                    parameters=params,
                )

    last_user_index = next(
        (
            i
            for i in range(len(messages) - 1, -1, -1)
            if messages[i].get("role") == "user"
        ),
        None,
    )
    history_source = (
        messages
        if last_user_index is None
        else [m for i, m in enumerate(messages) if i != last_user_index]
    )

    chat_history: List[CohereMessage] = []
    for msg in history_source:
        role = msg.get("role")
        content = _extract_text_content(msg.get("content"))

        tool_calls: Optional[List[CohereToolCall]] = None
        if role == "assistant" and msg.get("tool_calls"):  # type: ignore[union-attr,typeddict-item]
            tool_calls = []
            for tc in msg["tool_calls"]:  # type: ignore[union-attr,typeddict-item]
                raw_arguments: Any = tc.get("function", {}).get("arguments", {})
                if isinstance(raw_arguments, str):
                    try:
                        arguments: Dict[str, Any] = json.loads(raw_arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                else:
                    arguments = raw_arguments
                tool_calls.append(
                    CohereToolCall(
                        name=str(tc.get("function", {}).get("name", "")),
                        parameters=arguments,
                    )
                )

        if role == "user":
            chat_history.append(CohereMessage(role="USER", message=content))
        elif role == "assistant":
            chat_history.append(
                CohereMessage(role="CHATBOT", message=content, toolCalls=tool_calls)
            )
        elif role == "tool":
            tool_call_id = str(msg.get("tool_call_id", "") or "")
            cohere_call = tool_call_lookup.get(
                tool_call_id, CohereToolCall(name="", parameters={})
            )
            chat_history.append(
                CohereToolMessage(
                    toolResults=[
                        CohereToolResult(
                            call=cohere_call,
                            outputs=[{"output": content}],
                        )
                    ]
                )
            )

    return chat_history


def adapt_tool_definitions_to_cohere_standard(
    tools: List[Dict[str, Any]],
) -> List[CohereTool]:
    """Adapt OpenAI-format tool definitions to the OCI Cohere format.

    - Resolves ``$ref``/``$defs`` and ``anyOf`` patterns that OCI rejects.
    - Maps JSON Schema type names to Python type names (``"string"`` → ``"str"``).
    - Embeds unsupported constraints (enum, format, range, pattern) into the
      parameter description so the model can still see them.
    """
    cohere_tools = []
    for tool in tools:
        function_def = tool.get("function", {})
        raw_params = function_def.get("parameters", {})

        resolved = sanitize_oci_schema(
            resolve_oci_schema_anyof(resolve_oci_schema_refs(raw_params))
        )
        properties = resolved.get("properties", {})
        required = resolved.get("required", [])

        parameter_definitions = {}
        for param_name, param_schema in properties.items():
            json_type = param_schema.get("type", "string")
            python_type = OCI_JSON_TO_PYTHON_TYPES.get(json_type, json_type)
            parameter_definitions[param_name] = CohereParameterDefinition(
                description=enrich_cohere_param_description(
                    param_schema.get("description", ""), param_schema
                ),
                type=python_type,
                isRequired=param_name in required,
            )

        cohere_tools.append(
            CohereTool(
                name=function_def.get("name", ""),
                description=function_def.get("description", ""),
                parameterDefinitions=parameter_definitions,
            )
        )

    return cohere_tools


def handle_cohere_response(
    json_response: dict,
    model: str,
    model_response: ModelResponse,
) -> ModelResponse:
    """Parse a non-streaming Cohere OCI response into a LiteLLM ModelResponse."""
    cohere_response = CohereChatResult(**json_response)

    model_response.model = model
    model_response.created = int(datetime.datetime.now().timestamp())

    response_text = cohere_response.chatResponse.text
    oci_finish_reason = cohere_response.chatResponse.finishReason

    if oci_finish_reason == "COMPLETE":
        finish_reason = "stop"
    elif oci_finish_reason == "MAX_TOKENS":
        finish_reason = "length"
    elif oci_finish_reason == "TOOL_CALL":
        finish_reason = "tool_calls"
    elif oci_finish_reason is not None:
        # OCI Cohere can emit error/cancel finish reasons (e.g. ``ERROR``,
        # ``ERROR_TOXIC``, ``ERROR_LIMIT``, ``USER_CANCEL``) that aren't part
        # of OpenAI's standard set. Normalize them to ``"stop"`` so downstream
        # consumers switching on ``finish_reason`` keep working — matches the
        # streaming handler below.
        finish_reason = "stop"
    else:
        finish_reason = None

    tool_calls: Optional[List[Dict[str, Any]]] = None
    if cohere_response.chatResponse.toolCalls:
        tool_calls = [
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.parameters),
                },
            }
            for tc in cohere_response.chatResponse.toolCalls
        ]

    content: Optional[str] = response_text if response_text else None

    model_response.choices = [
        Choices(
            index=0,
            message={
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            },
            finish_reason=finish_reason,
        )
    ]

    usage_info = cohere_response.chatResponse.usage
    if usage_info is not None:
        model_response.usage = Usage(  # type: ignore[attr-defined]
            prompt_tokens=usage_info.promptTokens,
            completion_tokens=usage_info.completionTokens,
            total_tokens=usage_info.totalTokens,
        )
    else:
        model_response.usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)  # type: ignore[attr-defined]

    return model_response


def handle_cohere_stream_chunk(dict_chunk: dict) -> ModelResponseStream:
    """Parse a single Cohere SSE chunk into a LiteLLM ModelResponseStream."""
    try:
        typed_chunk = CohereStreamChunk(**dict_chunk)
    except TypeError as e:
        raise OCIError(
            status_code=500,
            message=f"Chunk cannot be parsed as CohereStreamChunk: {str(e)}",
        )

    if typed_chunk.index is None:
        typed_chunk.index = 0

    # OCI Cohere's terminal SSE event re-sends the full assembled response in
    # `text` alongside a populated `chatHistory` and a non-null `finishReason`.
    # Emitting that text would concatenate the whole response onto the
    # already-streamed deltas. We require both signals to be present so that a
    # future API change which adds `chatHistory` to intermediate chunks (or a
    # rare early-populated case) doesn't silently drop legitimate token deltas.
    is_terminal_consolidation = (
        typed_chunk.chatHistory is not None and typed_chunk.finishReason is not None
    )
    text = "" if is_terminal_consolidation else (typed_chunk.text or "")

    # Tool calls on the terminal consolidation chunk (whether from
    # `typed_chunk.toolCalls` or from `chatHistory`) restate what was already
    # streamed in intermediate chunks. Re-emitting them here would mint fresh
    # `uuid4` IDs and cause downstream consumers to execute each tool call
    # twice. Suppress them on the terminal chunk for the same reason `text`
    # is suppressed above.
    cohere_tool_calls = None if is_terminal_consolidation else typed_chunk.toolCalls

    tool_calls: Optional[List[Dict[str, Any]]] = None
    if cohere_tool_calls:
        tool_calls = [
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.parameters),
                },
            }
            for tc in cohere_tool_calls
        ]

    finish_reason = typed_chunk.finishReason
    if finish_reason == "COMPLETE":
        finish_reason = "stop"
    elif finish_reason == "MAX_TOKENS":
        finish_reason = "length"
    elif finish_reason == "TOOL_CALL":
        finish_reason = "tool_calls"
    elif finish_reason is not None:
        # OCI Cohere can emit error/cancel finish reasons (e.g. ``ERROR``,
        # ``ERROR_TOXIC``, ``ERROR_LIMIT``, ``USER_CANCEL``) that aren't part
        # of OpenAI's standard set. Normalize them to ``"stop"`` so downstream
        # consumers switching on ``finish_reason`` keep working.
        finish_reason = "stop"

    return ModelResponseStream(
        choices=[
            StreamingChoices(
                index=typed_chunk.index if typed_chunk.index else 0,
                delta=Delta(
                    content=text,
                    tool_calls=tool_calls,
                    provider_specific_fields=None,
                    thinking_blocks=None,
                    reasoning_content=None,
                ),
                finish_reason=finish_reason,
            )
        ]
    )
