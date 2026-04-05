"""
OCI Generative AI — Generic-format chat transformation helpers.

Handles message building, tool definition adaptation, non-streaming response
parsing, and streaming chunk parsing for models served with
``apiFormat="GENERIC"`` (e.g. Meta Llama, xAI Grok, Google Gemini).
"""

import datetime
import uuid
from typing import Dict, List, Optional, Union

import httpx

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
from litellm.utils import ChatCompletionMessageToolCall, Usage

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


def adapt_messages_to_generic_oci_standard_content_message(
    role: str, content: Union[str, list]
) -> OCIMessage:
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
            raise OCIError(
                status_code=400, message="Each content item must be a dictionary"
            )

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


def adapt_messages_to_generic_oci_standard_tool_call(
    role: str, tool_calls: list
) -> OCIMessage:
    """Convert an assistant tool-call message to OCI format."""
    tool_calls_formatted = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            raise OCIError(
                status_code=400, message="Each tool call must be a dictionary"
            )
        if tool_call.get("type") != "function":
            raise OCIError(
                status_code=400, message="OCI only supports function tool calls"
            )

        tool_call_id = tool_call.get("id")
        if not isinstance(tool_call_id, str):
            raise OCIError(status_code=400, message="Tool call `id` must be a string")

        tool_function = tool_call.get("function")
        if not isinstance(tool_function, dict):
            raise OCIError(
                status_code=400, message="Tool call `function` must be a dictionary"
            )

        function_name = tool_function.get("name")
        if not isinstance(function_name, str):
            raise OCIError(
                status_code=400, message="Tool call `function.name` must be a string"
            )

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


def adapt_messages_to_generic_oci_standard_tool_response(
    role: str, tool_call_id: str, content: str
) -> OCIMessage:
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
                raise OCIError(
                    status_code=400, message="Message `tool_calls` must be a list"
                )
            new_messages.append(
                adapt_messages_to_generic_oci_standard_tool_call(role, tool_calls)
            )

        elif role in ["system", "user", "assistant"] and content is not None:
            if not isinstance(content, (str, list)):
                raise OCIError(
                    status_code=400,
                    message="Message `content` must be a string or list of content parts",
                )
            new_messages.append(
                adapt_messages_to_generic_oci_standard_content_message(role, content)
            )

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
            new_messages.append(
                adapt_messages_to_generic_oci_standard_tool_response(
                    role, tool_call_id, content
                )
            )

    return new_messages


# ---------------------------------------------------------------------------
# Tool definition adaptation
# ---------------------------------------------------------------------------


def adapt_tool_definition_to_oci_standard(
    tools: List[Dict], vendor: OCIVendors
) -> List[OCIToolDefinition]:
    """Convert OpenAI-format tool definitions to OCI GENERIC format.

    Resolves ``$ref``/``$defs`` and ``anyOf`` that the OCI endpoint rejects.
    """
    new_tools = []
    for tool in tools:
        if tool["type"] != "function":
            raise OCIError(status_code=400, message="OCI only supports function tools")

        tool_function = tool.get("function")
        if not isinstance(tool_function, dict):
            raise OCIError(
                status_code=400, message="Tool `function` must be a dictionary"
            )

        raw_params = tool_function.get("parameters", {})
        resolved_params = sanitize_oci_schema(
            resolve_oci_schema_anyof(resolve_oci_schema_refs(raw_params))
        )

        new_tools.append(
            OCIToolDefinition(
                type="FUNCTION",
                name=tool_function.get("name"),
                description=tool_function.get("description", ""),
                parameters=resolved_params,
            )
        )

    return new_tools


def adapt_tools_to_openai_standard(
    tools: List[OCIToolCall],
) -> List[ChatCompletionMessageToolCall]:
    """Convert OCI tool-call objects in a response to the OpenAI format."""
    return [
        ChatCompletionMessageToolCall(
            id=tool.id or f"call_{uuid.uuid4().hex[:24]}",
            type="function",
            function={"name": tool.name, "arguments": tool.arguments},
        )
        for tool in tools
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
    except TypeError as e:
        raise OCIError(
            message=f"Response cannot be casted to OCICompletionResponse: {str(e)}",
            status_code=raw_response.status_code,
        )

    iso_str = completion_response.chatResponse.timeCreated
    dt = datetime.datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    model_response.created = int(dt.timestamp())
    model_response.model = completion_response.modelId

    message = model_response.choices[0].message  # type: ignore
    response_message = completion_response.chatResponse.choices[0].message
    if response_message is not None:
        if (
            response_message.content
            and len(response_message.content) > 0
            and response_message.content[0].type == "TEXT"
        ):
            message.content = response_message.content[0].text
        if response_message.toolCalls:
            message.tool_calls = adapt_tools_to_openai_standard(
                response_message.toolCalls
            )

    oci_usage = completion_response.chatResponse.usage
    model_response.usage = Usage(  # type: ignore[attr-defined]
        prompt_tokens=oci_usage.promptTokens,
        completion_tokens=oci_usage.completionTokens or 0,
        total_tokens=oci_usage.totalTokens,
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
    except TypeError as e:
        raise OCIError(
            status_code=500,
            message=f"Chunk cannot be parsed as OCIStreamChunk: {str(e)}",
        )

    if typed_chunk.index is None:
        typed_chunk.index = 0

    text = ""
    if typed_chunk.message and typed_chunk.message.content:
        for item in typed_chunk.message.content:
            if isinstance(item, OCITextContentPart):
                text += item.text
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

    tool_calls = None
    if typed_chunk.message and typed_chunk.message.toolCalls:
        tool_calls = adapt_tools_to_openai_standard(typed_chunk.message.toolCalls)

    oci_finish_reason = typed_chunk.finishReason
    if oci_finish_reason == "COMPLETE":
        finish_reason: Optional[str] = "stop"
    elif oci_finish_reason == "MAX_TOKENS":
        finish_reason = "length"
    elif oci_finish_reason == "TOOL_CALLS":
        finish_reason = "tool_calls"
    else:
        finish_reason = oci_finish_reason

    return ModelResponseStream(
        choices=[
            StreamingChoices(
                index=typed_chunk.index if typed_chunk.index else 0,
                delta=Delta(
                    content=text,
                    tool_calls=(
                        [tool.model_dump() for tool in tool_calls]
                        if tool_calls
                        else None
                    ),
                    provider_specific_fields=None,
                    thinking_blocks=None,
                    reasoning_content=None,
                ),
                finish_reason=finish_reason,
            )
        ]
    )
