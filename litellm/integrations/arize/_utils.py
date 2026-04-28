import json
import hashlib
import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Type
from pydantic import BaseModel
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
    AudioAttributes,
    EmbeddingAttributes,
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)


DEFAULT_MAX_INLINE_IMAGE_BYTES = 32 * 1024


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

    # Pydantic responses (ImageResponse, ResponsesAPIResponse, ...) need to be
    # dict-coerced so the dict-keyed setters below see the data.
    if isinstance(response_obj, BaseModel):
        try:
            response_obj = response_obj.model_dump()
        except Exception:
            return

    if not hasattr(response_obj, "get"):
        return

    _set_choice_outputs(span, response_obj, MessageAttributes, SpanAttributes)
    _set_audio_outputs(span, response_obj, AudioAttributes, SpanAttributes)
    _set_embedding_outputs(span, response_obj, EmbeddingAttributes, SpanAttributes)
    _set_image_outputs(
        span,
        response_obj,
        MessageAttributes,
        MessageContentAttributes,
        ImageAttributes,
        SpanAttributes,
    )
    _set_structured_outputs(span, response_obj, MessageAttributes, SpanAttributes)
    _set_usage_outputs(span, response_obj, SpanAttributes)


def _set_image_outputs(
    span: "Span",
    response_obj,
    msg_attrs,
    content_attrs,
    image_attrs,
    span_attrs,
):
    """
    Render generated images on the LLM span.

    Phoenix renders an image inline when the assistant message has an
    ``image``-type content entry whose ``image.url`` is either a public URL or
    a ``data:image/<type>;base64,<...>`` URI. Mirrors the structure produced
    by ``openinference.instrumentation.ImageMessageContent``.

    Iterates ``response_obj["data"]`` (image-gen / image-edit shape). Skips
    items that lack both ``url`` and ``b64_json`` so embedding payloads (which
    also live under ``data``) aren't matched.
    """
    data = response_obj.get("data")
    if not isinstance(data, list) or not data:
        return

    role_set = False
    content_idx = 0
    for image_item in data:
        if isinstance(image_item, BaseModel):
            image_item = image_item.model_dump()
        if not hasattr(image_item, "get"):
            continue

        if not image_item.get("url") and not image_item.get("b64_json"):
            continue  # not an image item (could be embedding)

        image_url, mime, omission_notice = _get_image_trace_payload(
            image_item, response_obj
        )
        if not image_url and not omission_notice:
            continue

        if not role_set:
            safe_set_attribute(
                span,
                f"{span_attrs.LLM_OUTPUT_MESSAGES}.0.{msg_attrs.MESSAGE_ROLE}",
                "assistant",
            )
            # First image also drives the span's top-level Output preview.
            if image_url:
                safe_set_attribute(span, span_attrs.OUTPUT_VALUE, image_url)
                safe_set_attribute(span, span_attrs.OUTPUT_MIME_TYPE, mime)
            else:
                safe_set_attribute(span, span_attrs.OUTPUT_VALUE, omission_notice)
                safe_set_attribute(span, span_attrs.OUTPUT_MIME_TYPE, "text/plain")
            role_set = True

        prefix = (
            f"{span_attrs.LLM_OUTPUT_MESSAGES}.0."
            f"{msg_attrs.MESSAGE_CONTENTS}.{content_idx}"
        )
        if image_url:
            safe_set_attribute(
                span, f"{prefix}.{content_attrs.MESSAGE_CONTENT_TYPE}", "image"
            )
            safe_set_attribute(
                span,
                f"{prefix}.{content_attrs.MESSAGE_CONTENT_IMAGE}.{image_attrs.IMAGE_URL}",
                image_url,
            )
        elif omission_notice:
            safe_set_attribute(
                span, f"{prefix}.{content_attrs.MESSAGE_CONTENT_TYPE}", "text"
            )
            safe_set_attribute(
                span,
                f"{prefix}.{content_attrs.MESSAGE_CONTENT_TEXT}",
                omission_notice,
            )
        content_idx += 1


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
            safe_set_attribute(
                span, f"{audio_attrs.AUDIO_TRANSCRIPT}.{i}", audio_transcript
            )


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


# Items can arrive as Pydantic models (SDK path) or dicts (proxy path).
# Read both via a uniform accessor so dict-shaped output[] arrays from
# Responses API don't get silently skipped.
def _get(obj, key, default=None):
    if hasattr(obj, "get"):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _set_structured_outputs(span: "Span", response_obj, msg_attrs, span_attrs):
    output_items = response_obj.get("output", [])
    for i, item in enumerate(output_items):
        prefix = f"{span_attrs.LLM_OUTPUT_MESSAGES}.{i}"

        item_type = _get(item, "type")
        if item_type is None:
            continue

        if item_type == "reasoning":
            summary = _get(item, "summary")
            if isinstance(summary, list):
                for s in summary:
                    text = _get(s, "text")
                    if text:
                        safe_set_attribute(
                            span,
                            f"{prefix}.{msg_attrs.MESSAGE_REASONING_SUMMARY}",
                            text,
                        )
        elif item_type == "message":
            content_list = _get(item, "content") or []
            message_content = ""
            if content_list:
                first_content = content_list[0]
                message_content = _get(first_content, "text", "") or ""
            message_role = _get(item, "role", "assistant") or "assistant"
            safe_set_attribute(span, span_attrs.OUTPUT_VALUE, message_content)
            safe_set_attribute(
                span, f"{prefix}.{msg_attrs.MESSAGE_CONTENT}", message_content
            )
            safe_set_attribute(span, f"{prefix}.{msg_attrs.MESSAGE_ROLE}", message_role)


def _set_usage_outputs(span: "Span", response_obj, span_attrs):
    usage = response_obj and response_obj.get("usage")
    if not usage:
        return

    safe_set_attribute(
        span, span_attrs.LLM_TOKEN_COUNT_TOTAL, usage.get("total_tokens")
    )
    completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
    if completion_tokens:
        safe_set_attribute(
            span, span_attrs.LLM_TOKEN_COUNT_COMPLETION, completion_tokens
        )
    prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
    if prompt_tokens:
        safe_set_attribute(span, span_attrs.LLM_TOKEN_COUNT_PROMPT, prompt_tokens)
    reasoning_tokens = usage.get("output_tokens_details", {}).get("reasoning_tokens")
    if reasoning_tokens:
        safe_set_attribute(
            span,
            span_attrs.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING,
            reasoning_tokens,
        )


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

    if any(
        keyword in lowered
        for keyword in ("file", "batch", "container", "fine_tuning_job")
    ):
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
            function = (
                tool.get("function") if isinstance(tool.get("function"), dict) else None
            )
            if not function:
                continue
            tool_name = function.get("name")
            if tool_name:
                safe_set_attribute(
                    span, f"{SpanAttributes.LLM_TOOLS}.{idx}.name", tool_name
                )
            tool_description = function.get("description")
            if tool_description:
                safe_set_attribute(
                    span,
                    f"{SpanAttributes.LLM_TOOLS}.{idx}.description",
                    tool_description,
                )
            params = function.get("parameters")
            if params is not None:
                safe_set_attribute(
                    span,
                    f"{SpanAttributes.LLM_TOOLS}.{idx}.parameters",
                    json.dumps(params),
                )

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

        metadata = (
            standard_logging_payload.get("metadata")
            if standard_logging_payload
            else None
        )
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
        if (
            optional_tools or metadata_tools
        ) and span_kind != OpenInferenceSpanKindValues.TOOL.value:
            span_kind = OpenInferenceSpanKindValues.TOOL.value

        safe_set_attribute(span, SpanAttributes.OPENINFERENCE_SPAN_KIND, span_kind)
        attributes.set_messages(span, kwargs)

        model_params = (
            standard_logging_payload.get("model_parameters")
            if standard_logging_payload
            else None
        )
        _set_model_params(span, model_params, SpanAttributes)

        _set_response_attributes(span=span, response_obj=response_obj)

    except Exception as e:
        verbose_logger.error(
            f"[Arize/Phoenix] Failed to set OpenInference span attributes: {e}"
        )
        if hasattr(span, "record_exception"):
            span.record_exception(e)


def _resolve_image_mime_type(image, response_obj) -> str:
    """
    Pick the MIME type for an image in a generation/edit response.

    Phoenix's image renderer keys off the data-URI mime, so a wrong value
    (e.g. ``image/png`` for a JPEG) breaks inline preview. Resolve in order:
    per-image ``mime_type`` → per-image ``output_format`` → response
    ``output_format`` → ``image/png``.
    """
    mime: Optional[str] = None
    if hasattr(image, "get"):
        mime = image.get("mime_type") or image.get("output_format")
    if not mime and hasattr(response_obj, "get"):
        mime = response_obj.get("output_format")
    if not mime:
        return "image/png"
    mime = str(mime).lower()
    if mime == "jpg":
        mime = "jpeg"
    if "/" not in mime:
        mime = f"image/{mime}"
    return mime


def _estimate_b64_decoded_bytes(b64_payload: str) -> int:
    """Estimate decoded byte size from base64 payload without decoding."""
    payload = b64_payload.strip()
    padding = payload.count("=")
    return max(0, (len(payload) * 3) // 4 - padding)


def _get_max_inline_image_bytes() -> Optional[int]:
    """
    Resolve max inline bytes for image payloads.

    - Default is 32KB to keep Phoenix spans exportable.
    - Set LITELLM_ARIZE_MAX_INLINE_IMAGE_BYTES<=0 to disable the cap.
    """
    raw_value = os.getenv("LITELLM_ARIZE_MAX_INLINE_IMAGE_BYTES")
    if raw_value is None or raw_value == "":
        return DEFAULT_MAX_INLINE_IMAGE_BYTES
    try:
        parsed = int(raw_value)
        if parsed <= 0:
            return None
        return parsed
    except (TypeError, ValueError):
        return DEFAULT_MAX_INLINE_IMAGE_BYTES


def _get_image_trace_payload(
    image_item, response_obj
) -> tuple[Optional[str], str, Optional[str]]:
    """
    Build image payload tuple:
    (image_url_or_data_uri, mime_type, omission_notice).
    """
    url = image_item.get("url")
    mime = _resolve_image_mime_type(image_item, response_obj)
    if url:
        return url, mime, None

    b64 = image_item.get("b64_json")
    if not b64:
        return None, mime, None

    max_inline_bytes = _get_max_inline_image_bytes()
    if max_inline_bytes is not None:
        decoded_bytes = _estimate_b64_decoded_bytes(str(b64))
        if decoded_bytes > max_inline_bytes:
            digest = hashlib.sha256(str(b64).encode("utf-8")).hexdigest()[:12]
            omission_notice = (
                f"[image omitted from trace: {decoded_bytes} bytes exceeds "
                f"{max_inline_bytes} byte inline limit, sha256={digest}]"
            )
            return None, mime, omission_notice

    return f"data:{mime};base64,{b64}", mime, None


def _extract_responses_api_text(output_items) -> Optional[str]:
    """Concatenate ``output[*].content[*].text`` for Responses API message items."""
    if not isinstance(output_items, list) or not output_items:
        return None

    def _g(obj, key, default=None):
        if hasattr(obj, "get"):
            return obj.get(key, default)
        return getattr(obj, key, default)

    texts = []
    for item in output_items:
        if _g(item, "type") != "message":
            continue
        content_list = _g(item, "content")
        if not isinstance(content_list, list):
            continue
        for c in content_list:
            text = _g(c, "text")
            if text:
                texts.append(text)
    return "\n\n".join(texts) if texts else None


def _coerce_text(value: Any) -> Optional[str]:
    """Reduce a heterogeneous prompt/response value to a renderable string."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        # Embedding `input` is often List[str]; show the first entry plus a
        # count rather than serialising every vector input.
        text_items = [item for item in value if isinstance(item, str)]
        if text_items:
            head = text_items[0]
            if len(text_items) == 1:
                return head
            return f"{head}\n\n[+ {len(text_items) - 1} more]"
        return safe_dumps(value)
    if isinstance(value, dict):
        return safe_dumps(value)
    return str(value)


def _extract_chain_input(
    kwargs, standard_logging_payload: Optional[StandardLoggingPayload]
) -> tuple[Optional[str], Optional[str]]:
    """
    - LLM chat history → last user message content
    - LLM completion / image gen → ``prompt``
    - Embedding → ``input`` (string or list of strings)
    - Reranker / Retriever → ``query`` (+ document count)
    - Tool / MCP → ``{name, arguments}`` JSON
    - Agent (a2a / responses) → ``input`` or last message
    - Guardrail → ``text``
    - Anything else → ``standard_logging_payload["messages"]`` as JSON
    """
    # 1. Tool calls — distinct shape, render as JSON arguments
    tool_name = kwargs.get("name") or kwargs.get("tool_name")
    tool_args = kwargs.get("arguments") or kwargs.get("tool_arguments")
    if tool_name and tool_args is not None:
        return (
            safe_dumps({"name": tool_name, "arguments": tool_args}),
            "application/json",
        )

    # 2. Chat-style messages (LLM, agents that use messages)
    messages = kwargs.get("messages")
    if isinstance(messages, list) and messages:
        last = messages[-1]
        if isinstance(last, dict):
            content = last.get("content")
            text = _coerce_text(content)
            if text:
                return text, None

    # 3. Reranker / Retriever — query plus optional document count summary
    query = kwargs.get("query")
    if query:
        text = _coerce_text(query)
        documents = kwargs.get("documents")
        if isinstance(documents, list) and documents:
            text = f"{text}\n\n[+ {len(documents)} documents]"
        return text, None

    # 4. Single-text inputs (embedding/image/completion/guardrail)
    for field in ("input", "prompt", "text"):
        text = _coerce_text(kwargs.get(field))
        if text:
            return text, None

    # 5. Universal fallback — every call type populates this
    if isinstance(standard_logging_payload, dict):
        msgs = standard_logging_payload.get("messages")
        text = _coerce_text(msgs)
        if text:
            mime = "application/json" if isinstance(msgs, (list, dict)) else None
            return text, mime

    return None, None


def _extract_chain_output(
    response_obj, standard_logging_payload: Optional[StandardLoggingPayload] = None
) -> tuple[Optional[str], Optional[str]]:
    """
    Build ``(output.value, output.mime_type)`` for the parent CHAIN span.

    Tries the rich response_obj shapes first, then falls back to
    ``standard_logging_payload["response"]`` so TOOL / AGENT / RETRIEVER /
    RERANKER / GUARDRAIL spans still get a meaningful output.

    - Chat → first choice message content (or tool_calls JSON if no content)
    - Image gen → first image URL or ``data:`` URI (matches Phoenix's renderer)
    - Embedding → ``"<n> embeddings"`` summary
    - Tool result → ``content`` text or full result JSON
    - Reranker → ``results`` JSON
    - Anything else → ``standard_logging_payload["response"]``
    """

    if isinstance(response_obj, BaseModel):
        try:
            response_obj = response_obj.model_dump()
        except Exception:
            pass

    response_obj_dict = response_obj if isinstance(response_obj, dict) else None
    if response_obj_dict is not None:
        # 1a. Responses API — output[*].content[*].text on message items.
        # Matches what _set_structured_outputs writes on the LLM child span,
        # so the parent CHAIN renders the same readable assistant text.
        responses_text = _extract_responses_api_text(response_obj_dict.get("output"))
        if responses_text:
            return responses_text, None

        # 1. Chat completion choices
        choices = response_obj_dict.get("choices") or []
        if isinstance(choices, list) and choices:
            first = choices[0]
            message = first.get("message") if hasattr(first, "get") else None
            if isinstance(message, dict):
                content = _coerce_text(message.get("content"))
                if content:
                    return content, None
                # Tool-calling assistant turn — no content but tool_calls present
                tool_calls = message.get("tool_calls")
                if tool_calls:
                    return safe_dumps(tool_calls), "application/json"

        # 2. Image / Embedding ``data`` array
        data = response_obj_dict.get("data")
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, BaseModel):
                first = first.model_dump()
            if isinstance(first, dict):
                url = first.get("url")
                if url:
                    return url, _resolve_image_mime_type(first, response_obj_dict)

                b64 = first.get("b64_json")
                if b64:
                    image_url, mime, omission_notice = _get_image_trace_payload(
                        first, response_obj_dict
                    )
                    if image_url:
                        return image_url, mime
                    if omission_notice:
                        return omission_notice, "text/plain"

                if first.get("embedding") is not None:
                    return f"{len(data)} embeddings", "text/plain"

        # 3. Tool / MCP result shapes — content array, output_text, result
        for field in ("output_text", "result"):
            text = _coerce_text(response_obj_dict.get(field))
            if text:
                mime = "application/json" or None
                return text, mime

        content = response_obj_dict.get("content")
        if content:
            return _coerce_text(content), ("application/json" or None)

        # 4. Reranker results
        results = response_obj_dict.get("results")
        if results:
            return safe_dumps(results), "application/json"

    # 5. Universal fallback for any call type
    if isinstance(standard_logging_payload, dict):
        resp = standard_logging_payload.get("response")
        text = _coerce_text(resp)
        if text:
            mime = "application/json" if isinstance(resp, (list, dict)) else None
            return text, mime

    return None, None


def set_parent_span_attributes(span: "Span", kwargs, response_obj):
    """
    set parent level span attr (model, call type) to prevent token / cost duplication from child spans
    """
    try:
        litellm_params = kwargs.get("litellm_params", {}) or {}
        standard_logging_payload: Optional[StandardLoggingPayload] = kwargs.get(
            "standard_logging_object"
        )

        safe_set_attribute(
            span,
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKindValues.CHAIN.value,
        )

        if kwargs.get("model"):
            safe_set_attribute(span, SpanAttributes.LLM_MODEL_NAME, kwargs.get("model"))

        safe_set_attribute(
            span,
            SpanAttributes.LLM_PROVIDER,
            litellm_params.get("custom_llm_provider", "Unknown"),
        )

        if standard_logging_payload is not None:
            call_type = standard_logging_payload.get("call_type")
            if call_type:
                safe_set_attribute(span, "llm.request.type", call_type)

            metadata = standard_logging_payload.get("metadata")
            _set_metadata_attributes(span, metadata, SpanAttributes)

            model_params = standard_logging_payload.get("model_parameters") or {}
            user_id = (
                model_params.get("user") if isinstance(model_params, dict) else None
            )
            if user_id is not None:
                safe_set_attribute(span, SpanAttributes.USER_ID, user_id)

        optional_params = _sanitize_optional_params(kwargs.get("optional_params"))
        safe_set_attribute(
            span, "llm.is_streaming", str(optional_params.get("stream", False))
        )

        # pass provider / litellm call id
        response_id = _resolve_response_id(response_obj, standard_logging_payload)
        if response_id is not None:
            safe_set_attribute(span, "llm.response.id", response_id)

        # sanitize input / output for parent span
        input_value, input_mime = _extract_chain_input(kwargs, standard_logging_payload)
        if input_value:
            safe_set_attribute(span, SpanAttributes.INPUT_VALUE, input_value)
            if input_mime:
                safe_set_attribute(span, SpanAttributes.INPUT_MIME_TYPE, input_mime)

        output_value, output_mime = _extract_chain_output(
            response_obj, standard_logging_payload
        )
        if output_value:
            safe_set_attribute(span, SpanAttributes.OUTPUT_VALUE, output_value)
            if output_mime:
                safe_set_attribute(span, SpanAttributes.OUTPUT_MIME_TYPE, output_mime)

    except Exception as e:
        verbose_logger.error(
            f"[Arize/Phoenix] Failed to set parent span attributes: {e}"
        )
        if hasattr(span, "record_exception"):
            span.record_exception(e)


def _resolve_response_id(
    response_obj, standard_logging_payload: Optional[StandardLoggingPayload]
) -> Optional[str]:
    """
    for completions / responses endpoint, pass the response id issued by provider.
    (chatcmpl-*)
    other endpoints (embeddings, images) pass the standard x-litellm-call-id
    to match litellm UI logs with phoenix
    """
    if isinstance(response_obj, dict):
        provider_id = response_obj.get("id")
        if provider_id:
            return str(provider_id)

    if isinstance(standard_logging_payload, dict):
        litellm_call_id = standard_logging_payload.get("litellm_call_id")
        if litellm_call_id:
            return str(litellm_call_id)

    return None


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

    # Track fallback context: which attempt number in the fallback chain
    fallback_attempt = kwargs.get("fallback_depth")
    if fallback_attempt is not None:
        safe_set_attribute(span, "llm.fallback.attempt_number", fallback_attempt)
        # Original model that was attempted before fallback
        original_model = kwargs.get("original_model")
        if original_model:
            safe_set_attribute(span, "llm.fallback.original_model", original_model)
        # Error that triggered fallback
        original_exception = kwargs.get("original_exception")
        if original_exception is not None:
            status_code = getattr(original_exception, "status_code", None)
            if status_code is not None:
                safe_set_attribute(span, "llm.fallback.error_status_code", status_code)
            exception_class = getattr(original_exception, "__class__.__name__", None)
            if exception_class:
                safe_set_attribute(span, "llm.fallback.error_class", exception_class)

    safe_set_attribute(
        span, "llm.request.type", standard_logging_payload.get("call_type")
    )
    safe_set_attribute(
        span,
        span_attrs.LLM_PROVIDER,
        litellm_params.get("custom_llm_provider", "Unknown"),
    )

    if optional_params.get("max_tokens"):
        safe_set_attribute(
            span, "llm.request.max_tokens", optional_params.get("max_tokens")
        )
    if optional_params.get("temperature"):
        safe_set_attribute(
            span, "llm.request.temperature", optional_params.get("temperature")
        )
    if optional_params.get("top_p"):
        safe_set_attribute(span, "llm.request.top_p", optional_params.get("top_p"))

    safe_set_attribute(
        span, "llm.is_streaming", str(optional_params.get("stream", False))
    )

    if optional_params.get("user"):
        safe_set_attribute(span, "llm.user", optional_params.get("user"))

    response_id = _resolve_response_id(response_obj, standard_logging_payload)
    if response_id is not None:
        safe_set_attribute(span, "llm.response.id", response_id)
    if response_obj and response_obj.get("model"):
        safe_set_attribute(span, "llm.response.model", response_obj.get("model"))


def _set_model_params(span: "Span", model_params: Optional[dict], span_attrs) -> None:
    if not model_params:
        return

    safe_set_attribute(
        span, span_attrs.LLM_INVOCATION_PARAMETERS, safe_dumps(model_params)
    )
    if model_params.get("user"):
        user_id = model_params.get("user")
        if user_id is not None:
            safe_set_attribute(span, span_attrs.USER_ID, user_id)
