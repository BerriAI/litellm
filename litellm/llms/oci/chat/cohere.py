"""
OCI Generative AI — Cohere-specific chat transformation helpers.

Handles message history building, tool definition adaptation, non-streaming
response parsing, and streaming chunk parsing for models served with
``apiFormat="COHERE"`` (e.g. ``cohere.command-*``).
"""

import datetime
import json
from typing import Any, Dict, List, Optional

import httpx
from pydantic import ValidationError

from litellm.llms.oci.chat.generic import (
    _normalize_oci_finish_reason,
    _synthesize_oci_tool_call_id,
)
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
    CohereToolResult,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantToolCall,
)
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
            item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content)


def _message_text(msg: AllMessageValues) -> str:
    """Plain-text content of a message (OpenAI's content field is loosely typed)."""
    return _extract_text_content(msg.get("content"))  # any-ok: OpenAI content union


def _assistant_tool_calls(
    msg: AllMessageValues,
) -> list[ChatCompletionAssistantToolCall]:
    """Tool calls on an assistant message, empty for any other role."""
    if msg.get("role") != "assistant":
        return []
    return msg.get("tool_calls") or []  # any-ok: union .get() is loosely typed


def _tool_call_to_cohere(tc: ChatCompletionAssistantToolCall) -> CohereToolCall:
    """Convert one OpenAI tool call into a Cohere ``CohereToolCall``."""
    fn = tc["function"]
    name = fn.get("name") or ""
    raw = fn.get("arguments") or "{}"
    try:
        args = json.loads(raw)  # any-ok: tool-call args are arbitrary JSON
    except json.JSONDecodeError:
        args = {}  # any-ok: arbitrary JSON
    return CohereToolCall(name=name, parameters=args)  # any-ok: arbitrary JSON


def adapt_messages_to_cohere_standard(
    messages: List[AllMessageValues],
) -> List[CohereMessage]:
    """Build a Cohere ``chatHistory`` (USER and CHATBOT turns) from an OpenAI array.

    The final message is omitted: the caller sends it as the top-level ``message``
    (a normal user turn) or via ``toolResults`` (a tool-result continuation).
    Tool-result messages are never represented in ``chatHistory`` — OCI carries
    the current turn's results in a separate top-level ``toolResults`` field, and
    rejects a request whose last history entry is a tool result. System messages
    must be filtered out by the caller (they are routed into ``preambleOverride``).
    """
    chat_history: list[CohereMessage] = []
    for msg in messages[:-1]:
        role = msg.get("role")
        if role == "user":
            chat_history.append(CohereMessage(role="USER", message=_message_text(msg)))
        elif role == "assistant":
            calls = [_tool_call_to_cohere(tc) for tc in _assistant_tool_calls(msg)]
            chat_history.append(
                CohereMessage(
                    role="CHATBOT",
                    message=_message_text(msg),
                    toolCalls=calls or None,
                )
            )
    return chat_history


def extract_cohere_tool_results(
    messages: list[AllMessageValues],
) -> Optional[list[CohereToolResult]]:
    """Return the current turn's tool results for OCI's top-level ``toolResults``.

    The current turn spans from the last user message to the end. Each tool
    message is matched to its originating call by ``tool_call_id`` so OCI sees the
    call name and parameters alongside the output. Returns ``None`` when there are
    no tool results so the field is omitted.
    """
    lookup: dict[str, CohereToolCall] = {
        tc.get("id") or "": _tool_call_to_cohere(tc) for msg in messages for tc in _assistant_tool_calls(msg)
    }

    last_user_index = next(
        (i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"),
        None,
    )
    current_turn = messages if last_user_index is None else messages[last_user_index:]

    unknown_call = CohereToolCall(name="", parameters={})  # any-ok: arbitrary JSON
    results: list[CohereToolResult] = []
    for msg in current_turn:
        if msg.get("role") != "tool":
            continue
        call = lookup.get(str(msg.get("tool_call_id") or ""), unknown_call)
        outputs = [{"output": _message_text(msg)}]  # any-ok: outputs are arbitrary
        results.append(CohereToolResult(call=call, outputs=outputs))  # any-ok: outputs
    return results or None


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

        resolved = sanitize_oci_schema(resolve_oci_schema_anyof(resolve_oci_schema_refs(raw_params)))
        properties = resolved.get("properties", {})
        required = resolved.get("required", [])

        parameter_definitions = {}
        for param_name, param_schema in properties.items():
            json_type = param_schema.get("type", "string")
            python_type = OCI_JSON_TO_PYTHON_TYPES.get(json_type, json_type)
            parameter_definitions[param_name] = CohereParameterDefinition(
                description=enrich_cohere_param_description(param_schema.get("description", ""), param_schema),
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
    raw_response: httpx.Response,
) -> ModelResponse:
    """Parse a non-streaming Cohere OCI response into a LiteLLM ModelResponse."""
    try:
        cohere_response = CohereChatResult(**json_response)
    except (TypeError, ValidationError) as e:
        raise OCIError(
            message=f"Response cannot be casted to CohereChatResult: {str(e)}",
            status_code=raw_response.status_code,
        )

    model_response.model = model
    model_response.created = int(datetime.datetime.now().timestamp())

    response_text = cohere_response.chatResponse.text
    finish_reason = _normalize_oci_finish_reason(cohere_response.chatResponse.finishReason)

    tool_calls: Optional[List[Dict[str, Any]]] = None
    if cohere_response.chatResponse.toolCalls:
        tool_calls = [
            {
                "id": _synthesize_oci_tool_call_id(i, tc.name, json.dumps(tc.parameters, sort_keys=True)),
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.parameters),
                },
            }
            for i, tc in enumerate(cohere_response.chatResponse.toolCalls)
        ]

    content: Optional[str] = response_text if response_text else None

    # Only include ``tool_calls`` in the message dict when actually present.
    # Passing an explicit ``None`` would let downstream consumers that key off
    # ``"tool_calls" in message`` (rather than truthiness) incorrectly conclude
    # that tool calls were attempted. Matches the generic handler's behaviour,
    # which only sets ``message.tool_calls`` when tool calls are present.
    message: Dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls

    model_response.choices = [
        Choices(
            index=0,
            message=message,
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


def handle_cohere_stream_chunk(
    dict_chunk: dict,
    prior_tool_calls_emitted: bool = False,
    prior_text_emitted: bool = False,
) -> ModelResponseStream:
    """Parse a single Cohere SSE chunk into a LiteLLM ModelResponseStream.

    ``prior_tool_calls_emitted`` lets the caller signal whether tool calls
    were already emitted in earlier chunks of the same stream. When set, the
    terminal consolidation chunk's tool calls are suppressed (they would
    duplicate prior deltas); otherwise they are passed through so a stream
    that delivers tool calls only on the terminal chunk doesn't silently
    drop them.

    ``prior_text_emitted`` plays the analogous role for the ``text`` field:
    when set, the terminal consolidation chunk's ``text`` is suppressed
    (it would re-emit the full assembled response on top of prior deltas);
    when unset (e.g. a degenerate stream that delivers the entire response
    in a single SSE event carrying both ``chatHistory`` and ``finishReason``),
    the text is passed through so the response content isn't silently lost.
    """
    try:
        typed_chunk = CohereStreamChunk(**dict_chunk)
    except (TypeError, ValidationError) as e:
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
    is_terminal_consolidation = typed_chunk.chatHistory is not None and typed_chunk.finishReason is not None
    # On non-terminal text-free chunks (e.g. tool-call-only or keep-alive
    # chunks) emit ``content=None`` rather than ``content=""`` so downstream
    # stream-mergers that distinguish "no text in this delta" from "an
    # explicitly empty text delta" behave correctly.
    #
    # We only suppress the terminal chunk's ``text`` when the caller has
    # confirmed that text deltas were already emitted earlier — otherwise
    # (e.g. a degenerate stream that delivers the whole response in a
    # single SSE event), passing it through is the only chance to surface it.
    text: Optional[str] = None if (is_terminal_consolidation and prior_text_emitted) else typed_chunk.text

    # Tool calls on the terminal consolidation chunk (whether from
    # `typed_chunk.toolCalls` or from `chatHistory`) typically restate what
    # was already streamed in intermediate chunks. Re-emitting them would
    # mint fresh `uuid4` IDs and cause downstream consumers to execute each
    # tool call twice. We only suppress when the caller has confirmed that
    # tool calls were already emitted earlier — otherwise (e.g. a short
    # response that delivers tool calls exclusively on the terminal chunk),
    # passing them through is the only chance to surface them.
    cohere_tool_calls = None if (is_terminal_consolidation and prior_tool_calls_emitted) else typed_chunk.toolCalls

    tool_calls: Optional[List[Dict[str, Any]]] = None
    if cohere_tool_calls:
        tool_calls = [
            {
                # Cohere protocol has no tool-call id, so we synthesize one
                # deterministically from the call's content/position. A random
                # uuid4 per chunk would cause downstream stream-mergers to
                # treat each chunk as a distinct tool call.
                "id": _synthesize_oci_tool_call_id(i, tc.name, json.dumps(tc.parameters, sort_keys=True)),
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.parameters),
                },
            }
            for i, tc in enumerate(cohere_tool_calls)
        ]

    finish_reason = _normalize_oci_finish_reason(typed_chunk.finishReason)

    return ModelResponseStream(
        choices=[
            StreamingChoices(
                index=typed_chunk.index,
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
