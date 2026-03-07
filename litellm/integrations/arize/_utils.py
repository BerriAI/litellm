import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

from typing_extensions import override

from litellm._logging import verbose_logger
from litellm.integrations.opentelemetry_utils.base_otel_llm_obs_attributes import (
    BaseLLMObsOTELAttributes,
    safe_set_attribute,
)
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.types.utils import StandardLoggingPayload

if TYPE_CHECKING:
    from opentelemetry.trace import Span
from litellm.integrations._types.open_inference import (
        MessageAttributes,
        MessageContentAttributes,
        ImageAttributes,
        SpanAttributes,
        AudioAttributes,
        EmbeddingAttributes,
        ToolCallAttributes,
        OpenInferenceSpanKindValues,
        OpenInferenceMimeTypeValues,
)


class ArizeOTELAttributes(BaseLLMObsOTELAttributes):
    @staticmethod
    @override
    def set_messages(span: "Span", kwargs: Dict[str, Any]):
        messages = kwargs.get("messages")

        if not messages:
            return

        # Set input.value from the last user message for display
        last_user_content = _extract_last_user_input(messages)
        if last_user_content:
            safe_set_attribute(span, SpanAttributes.INPUT_VALUE, last_user_content)
            safe_set_attribute(
                span, SpanAttributes.INPUT_MIME_TYPE, OpenInferenceMimeTypeValues.TEXT.value
            )

        # Set per-message attributes (input_messages tab in Phoenix)
        for idx, msg in enumerate(messages):
            prefix = f"{SpanAttributes.LLM_INPUT_MESSAGES}.{idx}"
            role = msg.get("role", "")
            safe_set_attribute(span, f"{prefix}.{MessageAttributes.MESSAGE_ROLE}", role)

            # Handle content — could be string, list of content blocks, or None
            content = msg.get("content")
            content_str = _content_to_string(content)
            if content_str:
                safe_set_attribute(
                    span, f"{prefix}.{MessageAttributes.MESSAGE_CONTENT}", content_str
                )

            # Handle multipart content blocks (text, image, tool_result, etc.)
            if isinstance(content, list):
                _set_message_contents(span, prefix, content)

            # Tool role messages: set tool_call_id and function name
            if role == "tool":
                tool_call_id = msg.get("tool_call_id")
                if tool_call_id:
                    safe_set_attribute(
                        span,
                        f"{prefix}.{MessageAttributes.MESSAGE_TOOL_CALL_ID}",
                        tool_call_id,
                    )
                func_name = msg.get("name")
                if func_name:
                    safe_set_attribute(
                        span,
                        f"{prefix}.{MessageAttributes.MESSAGE_NAME}",
                        func_name,
                    )

            # Assistant messages with tool_calls
            tool_calls = msg.get("tool_calls")
            if tool_calls and isinstance(tool_calls, list):
                _set_tool_calls_on_message(span, prefix, tool_calls)

    @staticmethod
    @override
    def set_response_output_messages(span: "Span", response_obj):
        """
        Sets output message attributes on the span from the LLM response.
        Handles text content, tool_calls, and structured output.
        """
        if not hasattr(response_obj, "get"):
            return

        for idx, choice in enumerate(response_obj.get("choices", [])):
            response_message = choice.get("message", {})
            content = response_message.get("content", "")

            # Build output.value — prefer text content, fall back to tool_calls summary
            tool_calls = response_message.get("tool_calls")
            if content:
                safe_set_attribute(span, SpanAttributes.OUTPUT_VALUE, content)
            elif tool_calls:
                # Summarize tool calls as the output value for display
                tool_summary = _summarize_tool_calls(tool_calls)
                safe_set_attribute(span, SpanAttributes.OUTPUT_VALUE, tool_summary)
                safe_set_attribute(
                    span,
                    SpanAttributes.OUTPUT_MIME_TYPE,
                    OpenInferenceMimeTypeValues.JSON.value,
                )

            prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{idx}"
            safe_set_attribute(
                span,
                f"{prefix}.{MessageAttributes.MESSAGE_ROLE}",
                response_message.get("role"),
            )
            if content:
                safe_set_attribute(
                    span,
                    f"{prefix}.{MessageAttributes.MESSAGE_CONTENT}",
                    content,
                )

            # Set tool_calls on output messages
            if tool_calls and isinstance(tool_calls, list):
                _set_tool_calls_on_message(span, prefix, tool_calls)


def _set_response_attributes(span: "Span", response_obj):
    """Helper to set response output and token usage attributes on span."""

    if not hasattr(response_obj, "get"):
        return

    _set_choice_outputs(span, response_obj, MessageAttributes, SpanAttributes)
    _set_image_outputs(span, response_obj, ImageAttributes, SpanAttributes)
    _set_audio_outputs(span, response_obj, AudioAttributes, SpanAttributes)
    _set_embedding_outputs(span, response_obj, EmbeddingAttributes, SpanAttributes)
    _set_structured_outputs(span, response_obj, MessageAttributes, SpanAttributes)
    _set_usage_outputs(span, response_obj, SpanAttributes)


def _set_choice_outputs(span: "Span", response_obj, msg_attrs, span_attrs):
    for idx, choice in enumerate(response_obj.get("choices", [])):
        response_message = choice.get("message", {})
        content = response_message.get("content", "")
        tool_calls = response_message.get("tool_calls")

        if content:
            safe_set_attribute(span, span_attrs.OUTPUT_VALUE, content)
        elif tool_calls:
            tool_summary = _summarize_tool_calls(tool_calls)
            safe_set_attribute(span, span_attrs.OUTPUT_VALUE, tool_summary)

        prefix = f"{span_attrs.LLM_OUTPUT_MESSAGES}.{idx}"
        safe_set_attribute(
            span,
            f"{prefix}.{msg_attrs.MESSAGE_ROLE}",
            response_message.get("role"),
        )
        if content:
            safe_set_attribute(
                span,
                f"{prefix}.{msg_attrs.MESSAGE_CONTENT}",
                content,
            )

        # Set tool_calls on output messages
        if tool_calls and isinstance(tool_calls, list):
            _set_tool_calls_on_message(span, prefix, tool_calls)


def _set_image_outputs(span: "Span", response_obj, image_attrs, span_attrs):
    images = response_obj.get("data", [])
    for i, image in enumerate(images):
        img_url = image.get("url")
        if img_url is None and image.get("b64_json"):
            img_url = f"data:image/png;base64,{image.get('b64_json')}"

        if not img_url:
            continue

        if i == 0:
            safe_set_attribute(span, span_attrs.OUTPUT_VALUE, img_url)

        safe_set_attribute(span, f"{image_attrs.IMAGE_URL}.{i}", img_url)


def _set_audio_outputs(span: "Span", response_obj, audio_attrs, span_attrs):
    audio = response_obj.get("audio", [])
    for i, audio_item in enumerate(audio):
        audio_url = audio_item.get("url")
        if audio_url is None and audio_item.get("b64_json"):
            audio_url = f"data:audio/wav;base64,{audio_item.get('b64_json')}"

        if audio_url:
            if i == 0:
                safe_set_attribute(span, span_attrs.OUTPUT_VALUE, audio_url)
            safe_set_attribute(span, f"{audio_attrs.AUDIO_URL}.{i}", audio_url)

        audio_mime = audio_item.get("mime_type")
        if audio_mime:
            safe_set_attribute(span, f"{audio_attrs.AUDIO_MIME_TYPE}.{i}", audio_mime)

        audio_transcript = audio_item.get("transcript")
        if audio_transcript:
            safe_set_attribute(span, f"{audio_attrs.AUDIO_TRANSCRIPT}.{i}", audio_transcript)


def _set_embedding_outputs(span: "Span", response_obj, embedding_attrs, span_attrs):
    embeddings = response_obj.get("data", [])
    for i, embedding_item in enumerate(embeddings):
        embedding_vector = embedding_item.get("embedding")
        if embedding_vector:
            if i == 0:
                safe_set_attribute(
                    span,
                    span_attrs.OUTPUT_VALUE,
                    str(embedding_vector),
                )

            safe_set_attribute(
                span,
                f"{embedding_attrs.EMBEDDING_VECTOR}.{i}",
                str(embedding_vector),
            )

        embedding_text = embedding_item.get("text")
        if embedding_text:
            safe_set_attribute(
                span,
                f"{embedding_attrs.EMBEDDING_TEXT}.{i}",
                str(embedding_text),
            )


def _set_structured_outputs(span: "Span", response_obj, msg_attrs, span_attrs):
    output_items = response_obj.get("output", [])
    for i, item in enumerate(output_items):
        prefix = f"{span_attrs.LLM_OUTPUT_MESSAGES}.{i}"
        if not hasattr(item, "type"):
            continue

        item_type = item.type
        if item_type == "reasoning" and hasattr(item, "summary"):
            for summary in item.summary:
                if hasattr(summary, "text"):
                    safe_set_attribute(
                        span,
                        f"{prefix}.{msg_attrs.MESSAGE_REASONING_SUMMARY}",
                        summary.text,
                    )
        elif item_type == "message" and hasattr(item, "content"):
            message_content = ""
            content_list = item.content
            if content_list and len(content_list) > 0:
                first_content = content_list[0]
                message_content = getattr(first_content, "text", "")
            message_role = getattr(item, "role", "assistant")
            safe_set_attribute(span, span_attrs.OUTPUT_VALUE, message_content)
            safe_set_attribute(span, f"{prefix}.{msg_attrs.MESSAGE_CONTENT}", message_content)
            safe_set_attribute(span, f"{prefix}.{msg_attrs.MESSAGE_ROLE}", message_role)


def _set_usage_outputs(span: "Span", response_obj, span_attrs):
    usage = response_obj and response_obj.get("usage")
    if not usage:
        return

    safe_set_attribute(span, span_attrs.LLM_TOKEN_COUNT_TOTAL, usage.get("total_tokens"))
    completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
    if completion_tokens:
        safe_set_attribute(span, span_attrs.LLM_TOKEN_COUNT_COMPLETION, completion_tokens)
    prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
    if prompt_tokens:
        safe_set_attribute(span, span_attrs.LLM_TOKEN_COUNT_PROMPT, prompt_tokens)
    reasoning_tokens = usage.get("output_tokens_details", {}).get("reasoning_tokens")
    if reasoning_tokens:
        safe_set_attribute(span, span_attrs.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING, reasoning_tokens)


def _extract_last_user_input(messages: List[Dict[str, Any]]) -> Optional[str]:
    """Extract the last user message content as the input value for display."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            return _content_to_string(content)
    # Fallback: last message content regardless of role
    if messages:
        return _content_to_string(messages[-1].get("content"))
    return None


def _content_to_string(content: Any) -> str:
    """Convert message content (string, list of blocks, or None) to a string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type == "text":
                    parts.append(block.get("text", ""))
                elif block_type == "tool_use":
                    name = block.get("name", "unknown_tool")
                    parts.append(f"[tool_use: {name}]")
                elif block_type == "tool_result":
                    tool_id = block.get("tool_use_id", "")
                    result_content = block.get("content", "")
                    if isinstance(result_content, list):
                        result_content = " ".join(
                            b.get("text", "") for b in result_content if isinstance(b, dict)
                        )
                    parts.append(f"[tool_result: {tool_id}] {result_content}")
                elif block_type == "image_url":
                    parts.append("[image]")
                else:
                    parts.append(f"[{block_type}]")
        return "\n".join(parts) if parts else ""
    return str(content)


def _set_message_contents(span: "Span", prefix: str, content_blocks: list):
    """Set message_contents attributes for multipart content blocks."""
    for cidx, block in enumerate(content_blocks):
        if not isinstance(block, dict):
            continue
        cprefix = f"{prefix}.{MessageAttributes.MESSAGE_CONTENTS}.{cidx}"
        block_type = block.get("type", "text")
        safe_set_attribute(
            span, f"{cprefix}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", block_type
        )
        if block_type == "text":
            safe_set_attribute(
                span, f"{cprefix}.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}", block.get("text", "")
            )
        elif block_type == "image_url":
            url = block.get("image_url", {}).get("url", "")
            safe_set_attribute(
                span, f"{cprefix}.{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}", url
            )


def _set_tool_calls_on_message(span: "Span", prefix: str, tool_calls: list):
    """Set tool_call attributes on a message (input or output)."""
    for tc_idx, tc in enumerate(tool_calls):
        if not isinstance(tc, dict):
            # Handle pydantic model objects
            tc = tc.__dict__ if hasattr(tc, "__dict__") else {}
        tc_prefix = f"{prefix}.{MessageAttributes.MESSAGE_TOOL_CALLS}.{tc_idx}"

        tc_id = tc.get("id")
        if tc_id:
            safe_set_attribute(span, f"{tc_prefix}.{ToolCallAttributes.TOOL_CALL_ID}", tc_id)

        function = tc.get("function")
        if isinstance(function, dict):
            fn_name = function.get("name")
            fn_args = function.get("arguments")
        elif hasattr(function, "name"):
            fn_name = getattr(function, "name", None)
            fn_args = getattr(function, "arguments", None)
        else:
            fn_name = None
            fn_args = None

        if fn_name:
            safe_set_attribute(
                span, f"{tc_prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}", fn_name
            )
        if fn_args:
            args_str = fn_args if isinstance(fn_args, str) else json.dumps(fn_args)
            safe_set_attribute(
                span,
                f"{tc_prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                args_str,
            )


def _summarize_tool_calls(tool_calls: list) -> str:
    """Create a JSON summary of tool_calls for output.value display."""
    summaries = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            function = tc.get("function", {})
            if isinstance(function, dict):
                name = function.get("name", "unknown")
                args = function.get("arguments", "{}")
            else:
                name = getattr(function, "name", "unknown")
                args = getattr(function, "arguments", "{}")
        elif hasattr(tc, "function"):
            name = getattr(tc.function, "name", "unknown")
            args = getattr(tc.function, "arguments", "{}")
        else:
            continue
        summaries.append({"name": name, "arguments": args})
    return json.dumps(summaries)


def _detect_tool_use_span_kind(
    messages: Optional[list], response_obj: Any
) -> Optional[str]:
    """
    Detect if the request/response involves tool use patterns that should
    override the default LLM span kind.

    Returns a more specific span kind if tool use is detected, or None.
    """
    if not response_obj or not hasattr(response_obj, "get"):
        return None

    # Check response tool_calls
    tool_names = []
    for choice in response_obj.get("choices", []):
        msg = choice.get("message", {})
        for tc in msg.get("tool_calls") or []:
            if isinstance(tc, dict):
                fn = tc.get("function", {})
                name = fn.get("name", "") if isinstance(fn, dict) else getattr(fn, "name", "")
            elif hasattr(tc, "function"):
                name = getattr(tc.function, "name", "")
            else:
                name = ""
            if name:
                tool_names.append(name.lower())

    if not tool_names:
        return None

    # Detect retriever patterns (file read, search, etc.)
    retriever_patterns = {"read", "glob", "grep", "file_search", "retrieval", "search"}
    web_search_patterns = {"web_search", "websearch", "brave_search", "web_fetch"}

    for name in tool_names:
        if any(pat in name for pat in web_search_patterns):
            return OpenInferenceSpanKindValues.RETRIEVER.value
        if any(pat in name for pat in retriever_patterns):
            return OpenInferenceSpanKindValues.RETRIEVER.value

    # If any tool calls are present, it's a TOOL span kind
    return None  # Let the default LLM kind apply; tools are visible via attributes


def _infer_open_inference_span_kind(call_type: Optional[str]) -> str:
    """
    Map LiteLLM call types to OpenInference span kinds.
    """

    if not call_type:
        return OpenInferenceSpanKindValues.UNKNOWN.value

    lowered = str(call_type).lower()

    if "embed" in lowered:
        return OpenInferenceSpanKindValues.EMBEDDING.value

    if "rerank" in lowered:
        return OpenInferenceSpanKindValues.RERANKER.value

    if "search" in lowered:
        return OpenInferenceSpanKindValues.RETRIEVER.value

    if "moderation" in lowered or "guardrail" in lowered:
        return OpenInferenceSpanKindValues.GUARDRAIL.value

    if lowered == "call_mcp_tool" or lowered == "mcp" or lowered.endswith("tool"):
        return OpenInferenceSpanKindValues.TOOL.value

    if "asend_message" in lowered or "a2a" in lowered or "assistant" in lowered:
        return OpenInferenceSpanKindValues.AGENT.value

    if any(
        keyword in lowered
        for keyword in (
            "completion",
            "chat",
            "image",
            "audio",
            "speech",
            "transcription",
            "generate_content",
            "response",
            "videos",
            "realtime",
            "pass_through",
            "anthropic_messages",
            "ocr",
        )
    ):
        return OpenInferenceSpanKindValues.LLM.value

    if any(keyword in lowered for keyword in ("file", "batch", "container", "fine_tuning_job")):
        return OpenInferenceSpanKindValues.CHAIN.value

    return OpenInferenceSpanKindValues.UNKNOWN.value

def _set_tool_attributes(
    span: "Span", optional_tools: Optional[list], metadata_tools: Optional[list]
):
    """set tool attributes on span from optional_params or tool call metadata"""
    if optional_tools:
        for idx, tool in enumerate(optional_tools):
            if not isinstance(tool, dict):
                continue
            function = tool.get("function") if isinstance(tool.get("function"), dict) else None
            if not function:
                continue
            tool_name = function.get("name")
            if tool_name:
                safe_set_attribute(span, f"{SpanAttributes.LLM_TOOLS}.{idx}.name", tool_name)
            tool_description = function.get("description")
            if tool_description:
                safe_set_attribute(span, f"{SpanAttributes.LLM_TOOLS}.{idx}.description", tool_description)
            params = function.get("parameters")
            if params is not None:
                safe_set_attribute(span, f"{SpanAttributes.LLM_TOOLS}.{idx}.parameters", json.dumps(params))

    if metadata_tools and isinstance(metadata_tools, list):
        for idx, tool in enumerate(metadata_tools):
            if not isinstance(tool, dict):
                continue
            tool_name = tool.get("name")
            if tool_name:
                safe_set_attribute(
                    span,
                    f"{SpanAttributes.LLM_INVOCATION_PARAMETERS}.tools.{idx}.name",
                    tool_name,
                )

            tool_description = tool.get("description")
            if tool_description:
                safe_set_attribute(
                    span,
                    f"{SpanAttributes.LLM_INVOCATION_PARAMETERS}.tools.{idx}.description",
                    tool_description,
                )


def set_attributes(
    span: "Span", kwargs, response_obj, attributes: Type[BaseLLMObsOTELAttributes]
):
    """
    Populates span with OpenInference-compliant LLM attributes for Arize and Phoenix tracing.
    """
    try:
        optional_params = _sanitize_optional_params(kwargs.get("optional_params"))
        litellm_params = kwargs.get("litellm_params", {}) or {}
        standard_logging_payload: Optional[StandardLoggingPayload] = kwargs.get(
            "standard_logging_object"
        )
        if standard_logging_payload is None:
            raise ValueError("standard_logging_object not found in kwargs")

        metadata = standard_logging_payload.get("metadata") if standard_logging_payload else None
        _set_metadata_attributes(span, metadata, SpanAttributes)

        metadata_tools = _extract_metadata_tools(metadata)
        optional_tools = _extract_optional_tools(optional_params)

        call_type = standard_logging_payload.get("call_type")
        _set_request_attributes(
            span=span,
            kwargs=kwargs,
            standard_logging_payload=standard_logging_payload,
            optional_params=optional_params,
            litellm_params=litellm_params,
            response_obj=response_obj,
            span_attrs=SpanAttributes,
        )

        span_kind = _infer_open_inference_span_kind(call_type=call_type)
        _set_tool_attributes(span, optional_tools, metadata_tools)
        if (optional_tools or metadata_tools) and span_kind != OpenInferenceSpanKindValues.TOOL.value:
            span_kind = OpenInferenceSpanKindValues.TOOL.value

        # Detect retriever/web_search patterns from response tool_calls
        messages = kwargs.get("messages")
        tool_use_kind = _detect_tool_use_span_kind(messages, response_obj)
        if tool_use_kind is not None:
            span_kind = tool_use_kind

        safe_set_attribute(span, SpanAttributes.OPENINFERENCE_SPAN_KIND, span_kind)
        attributes.set_messages(span, kwargs)

        model_params = standard_logging_payload.get("model_parameters") if standard_logging_payload else None
        _set_model_params(span, model_params, SpanAttributes)

        _set_response_attributes(span=span, response_obj=response_obj)

    except Exception as e:
        verbose_logger.error(
            f"[Arize/Phoenix] Failed to set OpenInference span attributes: {e}"
        )
        if hasattr(span, "record_exception"):
            span.record_exception(e)


def _sanitize_optional_params(optional_params: Optional[dict]) -> dict:
    if not isinstance(optional_params, dict):
        return {}
    optional_params.pop("secret_fields", None)
    return optional_params


def _set_metadata_attributes(span: "Span", metadata: Optional[Any], span_attrs) -> None:
    if metadata is not None:
        safe_set_attribute(span, span_attrs.METADATA, safe_dumps(metadata))


def _extract_metadata_tools(metadata: Optional[Any]) -> Optional[list]:
    if not isinstance(metadata, dict):
        return None
    llm_obj = metadata.get("llm")
    if isinstance(llm_obj, dict):
        return llm_obj.get("tools")
    return None


def _extract_optional_tools(optional_params: dict) -> Optional[list]:
    return optional_params.get("tools") if isinstance(optional_params, dict) else None


def _set_request_attributes(
    span: "Span",
    kwargs,
    standard_logging_payload: StandardLoggingPayload,
    optional_params: dict,
    litellm_params: dict,
    response_obj,
    span_attrs,
):
    if kwargs.get("model"):
        safe_set_attribute(span, span_attrs.LLM_MODEL_NAME, kwargs.get("model"))

    safe_set_attribute(span, "llm.request.type", standard_logging_payload.get("call_type"))
    safe_set_attribute(span, span_attrs.LLM_PROVIDER, litellm_params.get("custom_llm_provider", "Unknown"))

    if optional_params.get("max_tokens"):
        safe_set_attribute(span, "llm.request.max_tokens", optional_params.get("max_tokens"))
    if optional_params.get("temperature"):
        safe_set_attribute(span, "llm.request.temperature", optional_params.get("temperature"))
    if optional_params.get("top_p"):
        safe_set_attribute(span, "llm.request.top_p", optional_params.get("top_p"))

    safe_set_attribute(span, "llm.is_streaming", str(optional_params.get("stream", False)))

    if optional_params.get("user"):
        safe_set_attribute(span, "llm.user", optional_params.get("user"))

    if response_obj and response_obj.get("id"):
        safe_set_attribute(span, "llm.response.id", response_obj.get("id"))
    if response_obj and response_obj.get("model"):
        safe_set_attribute(span, "llm.response.model", response_obj.get("model"))


def _set_model_params(span: "Span", model_params: Optional[dict], span_attrs) -> None:
    if not model_params:
        return

    safe_set_attribute(span, span_attrs.LLM_INVOCATION_PARAMETERS, safe_dumps(model_params))
    if model_params.get("user"):
        user_id = model_params.get("user")
        if user_id is not None:
            safe_set_attribute(span, span_attrs.USER_ID, user_id)
