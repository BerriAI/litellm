"""
OCI Generative AI — Generic-format chat transformation helpers.

Handles message building, tool definition adaptation, non-streaming response
parsing, and streaming chunk parsing for models served with
``apiFormat="GENERIC"`` (e.g. Meta Llama, xAI Grok, Google Gemini).
"""

import datetime
import hashlib
from typing import Any, Dict, List, Optional, Union

import httpx
from pydantic import ValidationError

from litellm.llms.oci.common_utils import (
    OCIError,
    resolve_oci_schema_anyof,
    resolve_oci_schema_refs,
    sanitize_oci_schema,
)
from litellm.types.llms.oci import (
    OCICompletionResponse,
    OCIContentPartUnion,
    OCIImageContentPart,
    OCIImageUrl,
    OCIMessage,
    OCIRoles,
    OCIStreamChunk,
    OCITextContentPart,
    OCIToolCall,
    OCIToolDefinition,
    OCIVendors,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import (
    Delta,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)
from litellm.types.utils import ChatCompletionMessageToolCall, Usage

# Maps OpenAI role names to OCI GENERIC role names.
open_ai_to_generic_oci_role_map: Dict[str, OCIRoles] = {
    "system": "SYSTEM",
    "user": "USER",
    "assistant": "ASSISTANT",
    "tool": "TOOL",
}


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------


def adapt_messages_to_generic_oci_standard_content_message(role: str, content: Union[str, list]) -> OCIMessage:
    """Convert a plain-text or multipart content message to OCI format."""
    new_content: List[OCIContentPartUnion] = []
    if isinstance(content, str):
        return OCIMessage(
            role=open_ai_to_generic_oci_role_map[role],
            content=[OCITextContentPart(text=content)],
            toolCalls=None,
            toolCallId=None,
        )

    for content_item in content:
        if not isinstance(content_item, dict):
            raise OCIError(status_code=400, message="Each content item must be a dictionary")

        item_type = content_item.get("type")
        if not isinstance(item_type, str):
            raise OCIError(
                status_code=400,
                message="Each content item must have a string `type` field",
            )
        if item_type not in ["text", "image_url"]:
            raise OCIError(
                status_code=400,
                message=f"Content type `{item_type}` is not supported by OCI",
            )

        if item_type == "text":
            text = content_item.get("text")
            if not isinstance(text, str):
                raise OCIError(
                    status_code=400,
                    message="Content item of type `text` must have a string `text` field",
                )
            new_content.append(OCITextContentPart(text=text))

        elif item_type == "image_url":
            image_url = content_item.get("image_url")
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            if not isinstance(image_url, str):
                raise OCIError(
                    status_code=400,
                    message="Prop `image_url` must be a string or an object with a `url` property",
                )
            new_content.append(OCIImageContentPart(imageUrl=OCIImageUrl(url=image_url)))

    return OCIMessage(
        role=open_ai_to_generic_oci_role_map[role],
        content=new_content,
        toolCalls=None,
        toolCallId=None,
    )


def adapt_messages_to_generic_oci_standard_tool_call(role: str, tool_calls: list) -> OCIMessage:
    """Convert an assistant tool-call message to OCI format."""
    tool_calls_formatted = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            raise OCIError(status_code=400, message="Each tool call must be a dictionary")
        if tool_call.get("type") != "function":
            raise OCIError(status_code=400, message="OCI only supports function tool calls")

        tool_call_id = tool_call.get("id")
        if not isinstance(tool_call_id, str):
            raise OCIError(status_code=400, message="Tool call `id` must be a string")

        tool_function = tool_call.get("function")
        if not isinstance(tool_function, dict):
            raise OCIError(status_code=400, message="Tool call `function` must be a dictionary")

        function_name = tool_function.get("name")
        if not isinstance(function_name, str):
            raise OCIError(status_code=400, message="Tool call `function.name` must be a string")

        arguments = tool_call["function"].get("arguments", "{}")
        if not isinstance(arguments, str):
            raise OCIError(
                status_code=400,
                message="Tool call `function.arguments` must be a JSON string",
            )

        tool_calls_formatted.append(
            OCIToolCall(
                id=tool_call_id,
                type="FUNCTION",
                name=function_name,
                arguments=arguments,
            )
        )

    return OCIMessage(
        role=open_ai_to_generic_oci_role_map[role],
        content=None,
        toolCalls=tool_calls_formatted,
        toolCallId=None,
    )


def adapt_messages_to_generic_oci_standard_tool_response(role: str, tool_call_id: str, content: str) -> OCIMessage:
    """Convert a tool-result message to OCI format."""
    return OCIMessage(
        role=open_ai_to_generic_oci_role_map[role],
        content=[OCITextContentPart(text=content)],
        toolCalls=None,
        toolCallId=tool_call_id,
    )


def adapt_messages_to_generic_oci_standard(
    messages: List[AllMessageValues],
) -> List[OCIMessage]:
    """Convert an OpenAI-format message array to OCI GENERIC format."""
    new_messages = []
    for message in messages:
        role = message["role"]
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        tool_call_id = message.get("tool_call_id")

        if role == "assistant" and tool_calls is not None:
            if not isinstance(tool_calls, list):
                raise OCIError(status_code=400, message="Message `tool_calls` must be a list")
            new_messages.append(adapt_messages_to_generic_oci_standard_tool_call(role, tool_calls))

        elif role in ["system", "user", "assistant"] and content is not None:
            if not isinstance(content, (str, list)):
                raise OCIError(
                    status_code=400,
                    message="Message `content` must be a string or list of content parts",
                )
            new_messages.append(adapt_messages_to_generic_oci_standard_content_message(role, content))

        elif role == "tool":
            if not isinstance(tool_call_id, str):
                raise OCIError(
                    status_code=400,
                    message="Tool result message must have a string `tool_call_id`",
                )
            if not isinstance(content, str):
                raise OCIError(
                    status_code=400,
                    message="Tool result message `content` must be a string",
                )
            new_messages.append(adapt_messages_to_generic_oci_standard_tool_response(role, tool_call_id, content))

    return new_messages


# ---------------------------------------------------------------------------
# Tool definition adaptation
# ---------------------------------------------------------------------------


def adapt_tool_definition_to_oci_standard(tools: List[Dict], vendor: OCIVendors) -> List[OCIToolDefinition]:
    """Convert OpenAI-format tool definitions to OCI GENERIC format.

    Resolves ``$ref``/``$defs`` and ``anyOf`` that the OCI endpoint rejects.
    """
    new_tools = []
    for tool in tools:
        if tool["type"] != "function":
            raise OCIError(status_code=400, message="OCI only supports function tools")

        tool_function = tool.get("function")
        if not isinstance(tool_function, dict):
            raise OCIError(status_code=400, message="Tool `function` must be a dictionary")

        raw_params = tool_function.get("parameters", {})
        resolved_params = sanitize_oci_schema(resolve_oci_schema_anyof(resolve_oci_schema_refs(raw_params)))

        new_tools.append(
            OCIToolDefinition(
                type="FUNCTION",
                name=tool_function.get("name"),
                description=tool_function.get("description", ""),
                parameters=resolved_params,
            )
        )

    return new_tools


def _normalize_oci_finish_reason(raw: Optional[str]) -> Optional[str]:
    """Map an OCI-specific finish reason to its OpenAI-standard equivalent.

    OCI emits ``COMPLETE`` / ``MAX_TOKENS`` / ``TOOL_CALL(S)`` plus a long tail
    of error/cancel reasons (``ERROR``, ``ERROR_TOXIC``, ``ERROR_LIMIT``,
    ``USER_CANCEL``, ``CONTENT_FILTERED``, ``CANCELLED``, ...). The OpenAI
    spec only defines ``stop`` / ``length`` / ``tool_calls`` / ... — anything
    else is collapsed to ``"stop"`` so downstream consumers switching on
    ``finish_reason`` keep working. A ``None`` input passes through unchanged.
    """
    if raw is None:
        return None
    if raw == "COMPLETE":
        return "stop"
    if raw == "MAX_TOKENS":
        return "length"
    if raw in ("TOOL_CALL", "TOOL_CALLS"):
        return "tool_calls"
    return "stop"


def _synthesize_oci_tool_call_id(position: int, name: str, arguments: str) -> str:
    """Deterministic synthetic tool-call id derived from chunk content.

    Used as a fallback when OCI omits ``id`` (always the case for the OCI
    Cohere protocol, occasionally the case for OCI GENERIC streaming chunks).
    A random ``uuid4`` per chunk would cause downstream stream-merging
    consumers — which key off the tool-call ``id`` — to treat re-emissions of
    the same logical call (e.g. terminal consolidation chunks, retries) as
    distinct calls. A content-derived digest stays stable across identical
    re-emissions while differing across truly distinct calls.
    """
    digest = hashlib.sha256(
        f"{position}|{name}|{arguments}".encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()[:24]
    return f"call_{digest}"


def adapt_tools_to_openai_standard(
    tools: List[OCIToolCall],
) -> List[ChatCompletionMessageToolCall]:
    """Convert OCI tool-call objects in a response to the OpenAI format."""
    return [
        ChatCompletionMessageToolCall(
            id=tool.id or _synthesize_oci_tool_call_id(i, tool.name, tool.arguments),
            type="function",
            function={"name": tool.name, "arguments": tool.arguments},
        )
        for i, tool in enumerate(tools)
    ]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def handle_generic_response(
    json_data: dict,
    model: str,
    model_response: ModelResponse,
    raw_response: httpx.Response,
) -> ModelResponse:
    """Parse a non-streaming GENERIC OCI response into a LiteLLM ModelResponse."""
    try:
        completion_response = OCICompletionResponse(**json_data)
    except (TypeError, ValidationError) as e:
        raise OCIError(
            message=f"Response cannot be casted to OCICompletionResponse: {str(e)}",
            status_code=raw_response.status_code,
        )

    iso_str = completion_response.chatResponse.timeCreated
    dt = datetime.datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    model_response.created = int(dt.timestamp())
    model_response.model = completion_response.modelId

    if not completion_response.chatResponse.choices:
        raise OCIError(
            message="OCI response contained no choices",
            status_code=raw_response.status_code,
        )

    response_choice = completion_response.chatResponse.choices[0]
    message = model_response.choices[0].message  # type: ignore
    response_message = response_choice.message
    if response_message is not None:
        if response_message.content:
            # Concatenate all text parts — matches the streaming handler, which
            # iterates the full content array. Skips non-text parts (e.g. image
            # parts) so a leading non-text part doesn't suppress trailing text.
            text: Optional[str] = None
            for item in response_message.content:
                if isinstance(item, OCITextContentPart):
                    text = (text or "") + item.text
            if text is not None:
                message.content = text
        if response_message.toolCalls:
            message.tool_calls = adapt_tools_to_openai_standard(response_message.toolCalls)

    model_response.choices[0].finish_reason = _normalize_oci_finish_reason(  # type: ignore[union-attr,assignment]
        response_choice.finishReason
    )

    oci_usage = completion_response.chatResponse.usage
    reasoning_tokens: Optional[int] = None
    if oci_usage.completionTokensDetails and oci_usage.completionTokensDetails.reasoningTokens is not None:
        reasoning_tokens = oci_usage.completionTokensDetails.reasoningTokens
    model_response.usage = Usage(  # type: ignore[attr-defined]
        prompt_tokens=oci_usage.promptTokens,
        completion_tokens=oci_usage.completionTokens or 0,
        total_tokens=oci_usage.totalTokens,
        reasoning_tokens=reasoning_tokens,
    )

    return model_response


def handle_generic_stream_chunk(dict_chunk: dict) -> ModelResponseStream:
    """Parse a single GENERIC SSE chunk into a LiteLLM ModelResponseStream."""
    # OCI streams tool calls progressively — early chunks may omit required fields.
    if dict_chunk.get("message") and dict_chunk["message"].get("toolCalls"):
        for tool_call in dict_chunk["message"]["toolCalls"]:
            tool_call.setdefault("arguments", "")
            tool_call.setdefault("id", "")
            tool_call.setdefault("name", "")

    try:
        typed_chunk = OCIStreamChunk(**dict_chunk)
    except (TypeError, ValidationError) as e:
        raise OCIError(
            status_code=500,
            message=f"Chunk cannot be parsed as OCIStreamChunk: {str(e)}",
        )

    if typed_chunk.index is None:
        typed_chunk.index = 0

    # Emit ``content=None`` rather than ``content=""`` on chunks with no text
    # parts (e.g. tool-call-only or keep-alive chunks) so downstream
    # stream-mergers that distinguish "no text in this delta" from "an
    # explicitly empty text delta" behave correctly.
    text: Optional[str] = None
    if typed_chunk.message and typed_chunk.message.content:
        for item in typed_chunk.message.content:
            if isinstance(item, OCITextContentPart):
                text = (text or "") + item.text
            elif isinstance(item, OCIImageContentPart):
                raise OCIError(
                    status_code=500,
                    message="OCI returned image content in a streaming response — not supported",
                )
            else:
                raise OCIError(
                    status_code=500,
                    message=f"Unsupported content type in OCI streaming response: {item.type}",
                )

    # Build plain tool-call dicts inline (matching the shape produced by
    # ``handle_cohere_stream_chunk``) rather than calling
    # ``adapt_tools_to_openai_standard`` and ``model_dump``-ing the typed
    # objects. Both code paths feed ``Delta.tool_calls``, so emitting the
    # same minimal ``{"id", "type", "function": {"name", "arguments"}}``
    # shape keeps downstream stream-mergers behaving identically across
    # GENERIC and Cohere chunks.
    tool_calls: Optional[List[Dict[str, Any]]] = None
    if typed_chunk.message and typed_chunk.message.toolCalls:
        tool_calls = [
            {
                "id": tc.id or _synthesize_oci_tool_call_id(i, tc.name, tc.arguments),
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments,
                },
            }
            for i, tc in enumerate(typed_chunk.message.toolCalls)
        ]

    finish_reason: Optional[str] = _normalize_oci_finish_reason(typed_chunk.finishReason)

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
