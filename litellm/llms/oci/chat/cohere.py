"""
OCI Generative AI — Cohere-specific chat transformation helpers.

Handles message history building, tool definition adaptation, non-streaming
response parsing, and streaming chunk parsing for models served with
``apiFormat="COHERE"`` (e.g. ``cohere.command-*``).
"""

import datetime
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

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
    CohereToolMessage,
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


def _extract_text_content(content: Any) -> str:
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
    """Build a Cohere ``chatHistory`` list from an OpenAI-format message array.

    - All messages except the *last user message* are included. The caller pulls
      the last user message into the request's top-level ``message`` field, so
      trailing tool results (the standard agentic continuation pattern) still
      appear in ``chatHistory`` and reach the model.
    - If no user message exists, every message is included (no slice).
    - System messages must be filtered out by the caller (they are routed into
      ``preambleOverride`` separately) — they are not represented in
      ``chatHistory``.
    - Tool results are expressed as OCI ``CohereToolMessage.toolResults`` entries,
      with the originating call's name and parameters resolved from the preceding
      assistant message via a ``tool_call_id`` lookup.
    """
    # First pass: build tool_call_id → CohereToolCall so tool-result messages can
    # reference the originating call by name and parameters.
    tool_call_lookup: Final = {
        tool_call.get("id", ""): _to_cohere_tool_call(tool_call)
        for msg in messages
        if msg.get("role") == "assistant" and "tool_calls" in msg
        for tool_call in msg["tool_calls"] or []
    }

    last_user_index: Final = next(
        (i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"),
        None,
    )
    history_source: Final = (
        messages if last_user_index is None else [m for i, m in enumerate(messages) if i != last_user_index]
    )

    chat_history: Final[list[CohereMessage]] = []
    for msg in history_source:
        role = msg.get("role")
        content = _extract_text_content(msg.get("content"))

        tool_calls = (
            [_to_cohere_tool_call(tool_call) for tool_call in msg["tool_calls"]]
            if role == "assistant" and "tool_calls" in msg and msg["tool_calls"]
            else None
        )

        if role == "user":
            chat_history.append(CohereMessage(role="USER", message=content))
        elif role == "assistant":
            chat_history.append(CohereMessage(role="CHATBOT", message=content, toolCalls=tool_calls))
        elif role == "tool":
            tool_call_id = str(msg.get("tool_call_id", "") or "")
            cohere_call = tool_call_lookup.get(tool_call_id, CohereToolCall(name="", parameters={}))
            tool_result = CohereToolResult(
                call=cohere_call,
                outputs=[{"output": content}],
            )
            # OpenAI emits one tool-role message per parallel tool call, but
            # the OCI Cohere API expects all results from a single assistant
            # turn to share one TOOL history entry with multiple toolResults.
            # Merge consecutive tool messages so the model sees the parallel
            # call/result pairing correctly during agentic loops.
            if chat_history and isinstance(chat_history[-1], CohereToolMessage):
                chat_history[-1].toolResults.append(tool_result)
            else:
                chat_history.append(CohereToolMessage(toolResults=[tool_result]))

    return chat_history


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
        typed_chunk: Final = CohereStreamChunk.model_validate(dict_chunk)
    except (TypeError, ValidationError) as e:
        raise OCIError(
            status_code=500,
            message=f"Chunk cannot be parsed as CohereStreamChunk: {e}",
        )

    if typed_chunk.index is None:
        typed_chunk.index = 0

    # OCI Cohere's terminal SSE event re-sends the full assembled response in
    # `text` alongside a populated `chatHistory` and a non-null `finishReason`.
    # Emitting that text would concatenate the whole response onto the
    # already-streamed deltas. We require both signals to be present so that a
    # future API change which adds `chatHistory` to intermediate chunks (or a
    # rare early-populated case) doesn't silently drop legitimate token deltas.
    is_terminal_consolidation: Final = typed_chunk.chatHistory is not None and typed_chunk.finishReason is not None
    # On non-terminal text-free chunks (e.g. tool-call-only or keep-alive
    # chunks) emit ``content=None`` rather than ``content=""`` so downstream
    # stream-mergers that distinguish "no text in this delta" from "an
    # explicitly empty text delta" behave correctly.
    #
    # We only suppress the terminal chunk's ``text`` when the caller has
    # confirmed that text deltas were already emitted earlier — otherwise
    # (e.g. a degenerate stream that delivers the whole response in a
    # single SSE event), passing it through is the only chance to surface it.
    text: Final[str | None] = None if (is_terminal_consolidation and prior_text_emitted) else typed_chunk.text

    # Tool calls on the terminal consolidation chunk (whether from
    # `typed_chunk.toolCalls` or from `chatHistory`) typically restate what
    # was already streamed in intermediate chunks. Re-emitting them would
    # mint fresh `uuid4` IDs and cause downstream consumers to execute each
    # tool call twice. We only suppress when the caller has confirmed that
    # tool calls were already emitted earlier — otherwise (e.g. a short
    # response that delivers tool calls exclusively on the terminal chunk),
    # passing them through is the only chance to surface them.
    cohere_tool_calls = None if (is_terminal_consolidation and prior_tool_calls_emitted) else typed_chunk.toolCalls

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
