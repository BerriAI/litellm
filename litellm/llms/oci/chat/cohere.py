"""
OCI Generative AI — Cohere-specific chat transformation helpers.

Handles message history building, tool definition adaptation, non-streaming
response parsing, and streaming chunk parsing for models served with
``apiFormat="COHERE"`` (e.g. ``cohere.command-*``).
"""

import datetime
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Final

import httpx
from pydantic import JsonValue, TypeAdapter, ValidationError

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
from litellm.types.llms.openai import AllMessageValues, ChatCompletionAssistantToolCall
from litellm.types.utils import (
    Choices,
    Delta,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)


def _json_dict(value: JsonValue) -> dict[str, JsonValue]:
    return value if isinstance(value, dict) else {}


def _json_list(value: JsonValue) -> list[JsonValue]:
    return value if isinstance(value, list) else []


def _json_str(value: JsonValue) -> str:
    return value if isinstance(value, str) else ""


def _content_block_text(block: Mapping[str, object]) -> str:
    if not isinstance(block, dict) or block.get("type") != "text":
        return ""
    text: Final = block.get("text", "")
    return text if isinstance(text, str) else ""


def _content_text(content: str | Iterable[Mapping[str, object]] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_content_block_text(block) for block in content)
    return str(content)


def _extract_text_content(content: str | Iterable[Mapping[str, object]] | None) -> str:
    """Return the plain-text representation of a message content value."""
    return _content_text(content)


_TOOL_ARGUMENTS_ADAPTER: Final = TypeAdapter(dict[str, object])


def _parsed_tool_arguments(raw_arguments: str | dict[str, object]) -> dict[str, object]:
    if not isinstance(raw_arguments, str):
        return raw_arguments
    try:
        return _TOOL_ARGUMENTS_ADAPTER.validate_json(raw_arguments)
    except ValidationError:
        return {}


def _to_cohere_tool_call(tool_call: ChatCompletionAssistantToolCall) -> CohereToolCall:
    function_fields: Final = tool_call.get("function", {})
    return CohereToolCall(
        name=str(function_fields.get("name", "")),
        parameters=_parsed_tool_arguments(function_fields.get("arguments", "{}")),
    )


def adapt_messages_to_cohere_standard(
    messages: list[AllMessageValues],
) -> list[CohereMessage]:
    """Build a Cohere ``chatHistory`` (USER and CHATBOT turns) from an OpenAI array.

    The final message is omitted: the caller sends it as the top-level ``message``
    (a normal user turn) or via ``toolResults`` (a tool-result continuation).
    Tool-result messages are never represented in ``chatHistory``: OCI carries the
    current turn's results in the separate top-level ``toolResults`` field and
    rejects a request whose last history entry is a tool result. System messages
    must be filtered out by the caller (they are routed into ``preambleOverride``).
    """
    return [entry for msg in messages[:-1] if (entry := _cohere_history_entry(msg)) is not None]


def cohere_message_text(msg: AllMessageValues) -> str:
    """Plain-text content of a message."""
    return _extract_text_content(msg.get("content"))


def _assistant_tool_calls(msg: AllMessageValues) -> list[ChatCompletionAssistantToolCall]:
    """Tool calls carried by an assistant message; empty for other roles or when absent."""
    if msg.get("role") != "assistant" or "tool_calls" not in msg:
        return []
    return list(msg["tool_calls"] or [])


def _cohere_history_entry(msg: AllMessageValues) -> CohereMessage | None:
    """One ``chatHistory`` entry for a message, ``None`` for non-history roles."""
    role: Final = msg.get("role")
    if role == "user":
        return CohereMessage(role="USER", message=cohere_message_text(msg))
    if role != "assistant":
        return None
    tool_calls: Final = [_to_cohere_tool_call(tool_call) for tool_call in _assistant_tool_calls(msg)]
    return CohereMessage(role="CHATBOT", message=cohere_message_text(msg), toolCalls=tool_calls or None)


def extract_cohere_tool_results(
    messages: list[AllMessageValues],
) -> list[CohereToolResult] | None:
    """Return the current turn's tool results for OCI's top-level ``toolResults``.

    The current turn spans from the last user message to the end. Each tool
    message is matched to its originating call by ``tool_call_id`` so OCI sees the
    call name and parameters alongside the output. Returns ``None`` when there are
    no tool results so the field is omitted.
    """
    tool_call_lookup: Final = {
        tool_call.get("id", ""): _to_cohere_tool_call(tool_call)
        for msg in messages
        for tool_call in _assistant_tool_calls(msg)
    }

    last_user_index: Final = next(
        (i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"),
        None,
    )
    current_turn: Final = messages if last_user_index is None else messages[last_user_index:]

    unknown_call: Final = CohereToolCall(name="", parameters={})
    tool_results: Final = [
        CohereToolResult(
            call=tool_call_lookup.get(str(msg.get("tool_call_id", "") or ""), unknown_call),
            outputs=[{"output": cohere_message_text(msg)}],
        )
        for msg in current_turn
        if msg.get("role") == "tool"
    ]
    return tool_results or None


def _resolved_oci_parameter_schema(raw_parameters: dict[str, JsonValue]) -> JsonValue:
    return sanitize_oci_schema(resolve_oci_schema_anyof(resolve_oci_schema_refs(raw_parameters)))


def _cohere_parameter_definition(param_schema: dict[str, JsonValue], is_required: bool) -> CohereParameterDefinition:
    json_type: Final = _json_str(param_schema.get("type")) or "string"
    return CohereParameterDefinition(
        description=enrich_cohere_param_description(_json_str(param_schema.get("description")), param_schema),
        type=OCI_JSON_TO_PYTHON_TYPES.get(json_type, json_type),
        isRequired=is_required,
    )


def _cohere_parameter_definitions(resolved_schema: JsonValue) -> dict[str, CohereParameterDefinition]:
    schema_fields: Final = _json_dict(resolved_schema)
    required: Final = _json_list(schema_fields.get("required"))
    return {
        param_name: _cohere_parameter_definition(_json_dict(param_schema), param_name in required)
        for param_name, param_schema in _json_dict(schema_fields.get("properties")).items()
    }


def _to_cohere_tool(tool: Mapping[str, JsonValue]) -> CohereTool:
    function_def: Final = _json_dict(tool.get("function"))
    return CohereTool(
        name=_json_str(function_def.get("name")),
        description=_json_str(function_def.get("description")),
        parameterDefinitions=_cohere_parameter_definitions(
            _resolved_oci_parameter_schema(_json_dict(function_def.get("parameters")))
        ),
    )


def adapt_tool_definitions_to_cohere_standard(
    tools: Sequence[Mapping[str, JsonValue]],
) -> list[CohereTool]:
    """Adapt OpenAI-format tool definitions to the OCI Cohere format.

    - Resolves ``$ref``/``$defs`` and ``anyOf`` patterns that OCI rejects.
    - Maps JSON Schema type names to Python type names (``"string"`` → ``"str"``).
    - Embeds unsupported constraints (enum, format, range, pattern) into the
      parameter description so the model can still see them.
    """
    return [_to_cohere_tool(tool) for tool in tools]


def handle_cohere_response(
    json_response: Mapping[str, JsonValue],
    model: str,
    model_response: ModelResponse,
    raw_response: httpx.Response,
) -> ModelResponse:
    """Parse a non-streaming Cohere OCI response into a LiteLLM ModelResponse."""
    try:
        cohere_response: Final = CohereChatResult.model_validate(json_response)
    except (TypeError, ValidationError) as e:
        raise OCIError(
            message=f"Response cannot be casted to CohereChatResult: {e}",
            status_code=raw_response.status_code,
        )

    model_response.model = model
    model_response.created = int(datetime.datetime.now().timestamp())

    response_text: Final = cohere_response.chatResponse.text
    finish_reason: Final = _normalize_oci_finish_reason(cohere_response.chatResponse.finishReason)

    tool_calls: list[dict[str, object]] | None = None
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

    content: Final[str | None] = response_text if response_text else None

    # Only include ``tool_calls`` in the message dict when actually present.
    # Passing an explicit ``None`` would let downstream consumers that key off
    # ``"tool_calls" in message`` (rather than truthiness) incorrectly conclude
    # that tool calls were attempted. Matches the generic handler's behaviour,
    # which only sets ``message.tool_calls`` when tool calls are present.
    message: Final[dict[str, object]] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls

    model_response.choices = [
        Choices(
            index=0,
            message=message,
            finish_reason=finish_reason,
        )
    ]

    usage_info: Final = cohere_response.chatResponse.usage
    if usage_info is not None:
        model_response.usage = Usage(
            prompt_tokens=usage_info.promptTokens,
            completion_tokens=usage_info.completionTokens,
            total_tokens=usage_info.totalTokens,
        )
    else:
        model_response.usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    return model_response


def handle_cohere_stream_chunk(
    dict_chunk: Mapping[str, JsonValue],
    prior_tool_calls_emitted: bool = False,
    prior_text_emitted: bool = False,
) -> ModelResponseStream:
    """Parse a single Cohere SSE chunk into a LiteLLM ModelResponseStream.

    OCI Cohere streams the answer as single-token ``text`` deltas, then restates
    the whole assembled ``text`` on every chunk that carries ``toolCalls`` or
    ``chatHistory`` (the tool-calls event and the terminal event). Once the
    caller reports that earlier chunks already emitted text
    (``prior_text_emitted``), those restatements are dropped so the client does
    not see the answer twice; a stream whose only text lives on such a chunk
    keeps it. ``prior_tool_calls_emitted`` plays the same role for the tool
    calls the terminal ``chatHistory`` chunk repeats.
    """
    try:
        typed_chunk: Final = CohereStreamChunk.model_validate(dict_chunk)
    except (TypeError, ValidationError) as e:
        raise OCIError(
            status_code=500,
            message=f"Chunk cannot be parsed as CohereStreamChunk: {e}",
        )

    if typed_chunk.index is None:
        typed_chunk.index = 0

    restates_text: Final = typed_chunk.chatHistory is not None or typed_chunk.toolCalls is not None
    restates_tool_calls: Final = typed_chunk.chatHistory is not None
    text: Final[str | None] = None if (restates_text and prior_text_emitted) else typed_chunk.text
    cohere_tool_calls: Final = None if (restates_tool_calls and prior_tool_calls_emitted) else typed_chunk.toolCalls

    tool_calls: list[dict[str, object]] | None = None
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

    finish_reason: Final = _normalize_oci_finish_reason(typed_chunk.finishReason)

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
