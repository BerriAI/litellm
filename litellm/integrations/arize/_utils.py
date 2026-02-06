import json
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

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
        ImageAttributes,
        SpanAttributes,
        AudioAttributes,
        EmbeddingAttributes,
        OpenInferenceSpanKindValues
)


class ArizeOTELAttributes(BaseLLMObsOTELAttributes):
    @staticmethod
    @override
    def set_messages(span: "Span", kwargs: Dict[str, Any]):
        messages = kwargs.get("messages")

        # for /chat/completions
        # https://docs.arize.com/arize/large-language-models/tracing/semantic-conventions
        if messages:
            last_message = messages[-1]
            safe_set_attribute(
                span,
                SpanAttributes.INPUT_VALUE,
                last_message.get("content", ""),
            )

            # LLM_INPUT_MESSAGES shows up under `input_messages` tab on the span page.
            for idx, msg in enumerate(messages):
                prefix = f"{SpanAttributes.LLM_INPUT_MESSAGES}.{idx}"
                # Set the role per message.
                safe_set_attribute(
                    span, f"{prefix}.{MessageAttributes.MESSAGE_ROLE}", msg.get("role")
                )
                # Set the content per message.
                safe_set_attribute(
                    span,
                    f"{prefix}.{MessageAttributes.MESSAGE_CONTENT}",
                    msg.get("content", ""),
                )

    @staticmethod
    @override
    def set_response_output_messages(span: "Span", response_obj):
        """
        Sets output message attributes on the span from the LLM response.
        Args:
            span: The OpenTelemetry span to set attributes on
            response_obj: The response object containing choices with messages
        """
        from litellm.integrations._types.open_inference import (
            MessageAttributes,
            SpanAttributes,
        )

        for idx, choice in enumerate(response_obj.get("choices", [])):
            response_message = choice.get("message", {})
            safe_set_attribute(
                span,
                SpanAttributes.OUTPUT_VALUE,
                response_message.get("content", ""),
            )

            # This shows up under `output_messages` tab on the span page.
            prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{idx}"
            safe_set_attribute(
                span,
                f"{prefix}.{MessageAttributes.MESSAGE_ROLE}",
                response_message.get("role"),
            )
            safe_set_attribute(
                span,
                f"{prefix}.{MessageAttributes.MESSAGE_CONTENT}",
                response_message.get("content", ""),
            )


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
        safe_set_attribute(
            span,
            span_attrs.OUTPUT_VALUE,
            response_message.get("content", ""),
        )
        prefix = f"{span_attrs.LLM_OUTPUT_MESSAGES}.{idx}"
        safe_set_attribute(
            span,
            f"{prefix}.{msg_attrs.MESSAGE_ROLE}",
            response_message.get("role"),
        )
        safe_set_attribute(
            span,
            f"{prefix}.{msg_attrs.MESSAGE_CONTENT}",
            response_message.get("content", ""),
        )


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
