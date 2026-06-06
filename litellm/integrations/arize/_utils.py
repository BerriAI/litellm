import json
from typing import TYPE_CHECKING, Any, Dict, Optional, Type

from typing_extensions import override

from litellm._logging import verbose_logger
from litellm.integrations.opentelemetry_utils.base_otel_llm_obs_attributes import (
    BaseLLMObsOTELAttributes,
    safe_set_attribute,
)
from litellm.litellm_core_utils.redact_messages import (
    should_redact_message_logging,
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
    ToolCallAttributes,
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

                # Additive: emit structured tool_calls / multimodal content
                # so Arize/Phoenix can render tool-using and image-bearing
                # turns. These set NEW attribute keys (MESSAGE_TOOL_CALLS /
                # MESSAGE_NAME / MESSAGE_TOOL_CALL_ID / MESSAGE_CONTENTS.*) —
                # never replace the MESSAGE_CONTENT write above.
                _safe_emit(
                    f"input message extras (idx={idx})",
                    _emit_input_message_extras,
                    span,
                    prefix,
                    msg,
                )

    # Note: `BaseLLMObsOTELAttributes.set_response_output_messages` is not
    # overridden here. The live code path uses `_set_choice_outputs` (called
    # via `_set_response_attributes` from `set_attributes`) which handles
    # tool_calls, multimodal output, embeddings, audio, images, and structured
    # outputs in a single place.


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

        # Tool-only assistant responses have empty content; serialize the
        # tool_calls into OUTPUT_VALUE so Arize's "Output" pane isn't blank.
        output_value = content
        if not output_value:
            tool_calls = _get_tool_calls(response_message)
            if tool_calls:
                output_value = _summarize_tool_calls_for_output(tool_calls)

        safe_set_attribute(span, span_attrs.OUTPUT_VALUE, output_value)
        prefix = f"{span_attrs.LLM_OUTPUT_MESSAGES}.{idx}"
        safe_set_attribute(
            span,
            f"{prefix}.{msg_attrs.MESSAGE_ROLE}",
            response_message.get("role"),
        )
        safe_set_attribute(
            span,
            f"{prefix}.{msg_attrs.MESSAGE_CONTENT}",
            content,
        )

        # Additive: emit assistant tool_calls so tool-using turns render in
        # Arize/Phoenix. Sets new MESSAGE_TOOL_CALLS keys only — does not
        # change MESSAGE_CONTENT/MESSAGE_ROLE writes above.
        _safe_emit(
            f"output tool_calls (idx={idx})",
            _emit_message_tool_calls,
            span,
            prefix,
            response_message,
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
            safe_set_attribute(
                span, f"{prefix}.{msg_attrs.MESSAGE_CONTENT}", message_content
            )
            safe_set_attribute(span, f"{prefix}.{msg_attrs.MESSAGE_ROLE}", message_role)


def _safe_get(obj, key, default=None):
    """Read ``key`` from a dict-like or Pydantic-model-like object.

    The arize/langfuse_otel logger receives ``usage`` objects from many sources:
    plain dicts, litellm ``Usage`` (which exposes ``.get``), and raw OpenAI
    Pydantic models (e.g. ``openai.types.completion_usage.CompletionUsage`` and
    nested ``CompletionTokensDetails`` / ``OutputTokensDetails``) which do NOT
    expose ``.get``. Calling ``.get`` on the latter raised ``AttributeError`` —
    see https://github.com/BerriAI/litellm/issues/13672.
    """
    if obj is None:
        return default
    getter = getattr(obj, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            # Some objects expose `.get` with a different signature
            pass
    return getattr(obj, key, default)


def _set_usage_outputs(span: "Span", response_obj, span_attrs):
    usage = response_obj and response_obj.get("usage")
    if not usage:
        return

    safe_set_attribute(
        span, span_attrs.LLM_TOKEN_COUNT_TOTAL, _safe_get(usage, "total_tokens")
    )
    completion_tokens = _safe_get(usage, "completion_tokens") or _safe_get(
        usage, "output_tokens"
    )
    if completion_tokens:
        safe_set_attribute(
            span, span_attrs.LLM_TOKEN_COUNT_COMPLETION, completion_tokens
        )
    prompt_tokens = _safe_get(usage, "prompt_tokens") or _safe_get(
        usage, "input_tokens"
    )
    if prompt_tokens:
        safe_set_attribute(span, span_attrs.LLM_TOKEN_COUNT_PROMPT, prompt_tokens)

    # Reasoning tokens live in `completion_tokens_details` for Chat Completions
    # API (Usage) and in `output_tokens_details` for Responses API
    # (ResponseAPIUsage). Both nested objects may be plain Pydantic models
    # without `.get`.
    token_details = _safe_get(usage, "completion_tokens_details") or _safe_get(
        usage, "output_tokens_details"
    )
    reasoning_tokens = _safe_get(token_details, "reasoning_tokens")
    if reasoning_tokens:
        safe_set_attribute(
            span,
            span_attrs.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING,
            reasoning_tokens,
        )

    # Additive: cache token breakdown so prompt-caching savings render in
    # Arize. Sources covered:
    #   - OpenAI Chat Completions: `prompt_tokens_details.cached_tokens`
    #   - Anthropic / Bedrock-Anthropic: `cache_read_input_tokens`,
    #     `cache_creation_input_tokens`
    # All emits are conditional, so when none of these fields exist (the
    # situation in the existing test fixtures) no extra attributes are set.
    prompt_token_details = _safe_get(usage, "prompt_tokens_details") or _safe_get(
        usage, "input_tokens_details"
    )
    cache_read = _safe_get(prompt_token_details, "cached_tokens") or _safe_get(
        usage, "cache_read_input_tokens"
    )
    if cache_read:
        safe_set_attribute(
            span,
            span_attrs.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
            cache_read,
        )
    # Anthropic / Bedrock-Anthropic only — OpenAI's `prompt_tokens_details`
    # does not expose a cache-write count, so we read straight off `usage`.
    cache_write = _safe_get(usage, "cache_creation_input_tokens")
    if cache_write:
        safe_set_attribute(
            span,
            span_attrs.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE,
            cache_write,
        )

    audio_prompt_tokens = _safe_get(prompt_token_details, "audio_tokens")
    if audio_prompt_tokens:
        safe_set_attribute(
            span,
            span_attrs.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO,
            audio_prompt_tokens,
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
            # `passthrough` (no underscore) is what real call_types use:
            # `allm_passthrough_route`, `llm_passthrough_route`. Without
            # this they fell through to UNKNOWN, blanking span.kind.
            "passthrough",
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
    # Coerce non-dict response objects (e.g. httpx.Response from passthrough
    # routes) into a dict so downstream `.get()` calls don't crash. Existing
    # dict / `.get()`-bearing objects (incl. Pydantic OpenAI Responses API
    # models) are returned unchanged, preserving the existing test behavior.
    response_obj_for_attrs = _coerce_response_obj_for_attrs(response_obj)

    # Set span.kind defensively before anything else. If a downstream step
    # throws, the span still has a kind so Arize can render it correctly
    # (an LLM call instead of UNKNOWN). This is the single source of truth
    # for span.kind — no late re-write happens below.
    _safe_emit("early span kind", _set_early_span_kind, span, kwargs)

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

        _set_request_attributes(
            span=span,
            kwargs=kwargs,
            standard_logging_payload=standard_logging_payload,
            optional_params=optional_params,
            litellm_params=litellm_params,
            response_obj=response_obj_for_attrs,
            span_attrs=SpanAttributes,
        )

        # span.kind was already set above by `_set_early_span_kind`. We do
        # NOT re-write it here based on tool presence: a chat completion
        # that passes `tools=[...]` (or returns `tool_calls`) is still an
        # LLM call per the OpenInference spec — TOOL is reserved for actual
        # tool execution spans, not LLM calls that request tools.
        _set_tool_attributes(span, optional_tools, metadata_tools)
        attributes.set_messages(span, kwargs)

        model_params = (
            standard_logging_payload.get("model_parameters")
            if standard_logging_payload
            else None
        )
        _set_model_params(span, model_params, SpanAttributes)

        _set_response_attributes(span=span, response_obj=response_obj_for_attrs)

    except Exception as e:
        verbose_logger.error(
            f"[Arize/Phoenix] Failed to set OpenInference span attributes: {e}"
        )
        if hasattr(span, "record_exception"):
            span.record_exception(e)

    # Additive emitters. Each is independently guarded so a failure can never
    # blank the attributes set by the main try-block above. New attributes are
    # written under new keys; existing attributes are not overwritten.
    slp = kwargs.get("standard_logging_object")
    _safe_emit("session/user attrs", _set_session_and_user_attrs, span, kwargs, slp)
    _safe_emit("response cost", _set_response_cost_attr, span, slp)
    _safe_emit(
        "passthrough normalization",
        _maybe_normalize_passthrough,
        span,
        kwargs,
        response_obj,
        response_obj_for_attrs,
        slp,
    )


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

    if response_obj and response_obj.get("id"):
        safe_set_attribute(span, "llm.response.id", response_obj.get("id"))
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


# ---------------------------------------------------------------------------
# Additive rendering helpers (introduced to enhance Arize/Phoenix rendering
# without changing any previously-emitted attribute keys or values).
# ---------------------------------------------------------------------------


def _safe_emit(label: str, fn, *args, **kwargs) -> None:
    """Run an additive attribute emitter, swallowing any error so it cannot
    blank attributes set elsewhere on the span. Failures are logged at debug.
    """
    try:
        fn(*args, **kwargs)
    except Exception as e:
        verbose_logger.debug("[Arize] %s skipped: %s", label, e)


def _set_early_span_kind(span: "Span", kwargs: dict) -> None:
    """Defensively set OPENINFERENCE_SPAN_KIND before any other logic runs."""
    slp = kwargs.get("standard_logging_object")
    call_type = slp.get("call_type") if isinstance(slp, dict) else None
    safe_set_attribute(
        span,
        SpanAttributes.OPENINFERENCE_SPAN_KIND,
        _infer_open_inference_span_kind(call_type=call_type),
    )


def _coerce_response_obj_for_attrs(response_obj):
    """Return a `.get`-compatible view of `response_obj` when possible.

    - dicts and Pydantic models that already expose `.get` are returned
      unchanged (preserves all current behavior, including the Responses API
      flow which relies on Pydantic attribute access).
    - `httpx.Response` and other text-only responses (passthrough routes)
      are JSON-decoded so the standard extraction paths can read fields like
      `id`, `model`, and `usage`. On failure the original object is returned
      so behavior is no worse than today.
    """
    if response_obj is None or hasattr(response_obj, "get"):
        return response_obj
    text = getattr(response_obj, "text", None)
    if isinstance(text, str) and text:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return response_obj


def _coerce_text(value) -> Optional[str]:
    """Best-effort text extraction from a message-content value.

    Returns None when no textual portion can be derived. Handles:
      - plain strings
      - lists of OpenAI-style content parts (`{"type": "text", "text": ...}`)
      - lists of Anthropic-style content parts (`{"type": "text", "text": ...}`
        or `{"type": "input_text", "text": ...}`)
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for part in value:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text") or part.get("input_text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "\n".join(parts)
    return None


def _to_plain_dict(value):
    """Best-effort: coerce a value (Pydantic model / dict / None) to a dict.

    Returns the original value when no safe conversion exists. Used to bridge
    OpenAI Pydantic message/tool_call objects into the dict-based helpers.
    """
    if value is None or isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump()
        except Exception:
            pass
    return value


def _get_tool_calls(message) -> Optional[list]:
    """Return ``message.tool_calls`` only when it's a non-empty list.

    Works for dicts and Pydantic message objects via ``_safe_get``.
    """
    tool_calls = _safe_get(message, "tool_calls")
    return tool_calls if isinstance(tool_calls, list) and tool_calls else None


def _normalize_tool_call(raw_tc) -> Optional[Dict[str, Any]]:
    """Normalize a single tool_call (dict or Pydantic) into a stable shape:

        {"id": str|None, "type": str, "function": {"name": str|None, "arguments": str|None}}

    Arguments are coerced to a JSON string per OpenInference convention.
    Returns ``None`` when ``raw_tc`` cannot be coerced to a dict.
    """
    tc = _to_plain_dict(raw_tc)
    if not isinstance(tc, dict):
        return None
    function = _to_plain_dict(tc.get("function"))
    name = function.get("name") if isinstance(function, dict) else None
    args = function.get("arguments") if isinstance(function, dict) else None
    if args is not None and not isinstance(args, str):
        try:
            args = json.dumps(args)
        except Exception:
            args = str(args)
    return {
        "id": tc.get("id"),
        "type": tc.get("type", "function"),
        "function": {"name": name, "arguments": args},
    }


def _summarize_tool_calls_for_output(tool_calls) -> str:
    """Render a tool_calls list as a compact JSON string for OUTPUT_VALUE.

    Best-effort: returns ``str(tool_calls)`` if anything unexpected happens
    so OUTPUT_VALUE is never blanked on a malformed payload.
    """
    try:
        normalized = [n for n in (_normalize_tool_call(tc) for tc in tool_calls) if n]
        return json.dumps({"tool_calls": normalized})
    except Exception:
        return str(tool_calls)


def _emit_message_tool_calls(span: "Span", prefix: str, message) -> None:
    """Emit ``MESSAGE_TOOL_CALLS.*`` for an assistant message that requested
    tool calls. Pure addition: only writes when ``tool_calls`` is non-empty.

    Accepts dicts or Pydantic message objects (e.g. ``litellm.Message``); the
    same applies to each tool_call entry.
    """
    tool_calls = _get_tool_calls(message)
    if not tool_calls:
        return
    for tc_idx, raw_tc in enumerate(tool_calls):
        tc = _normalize_tool_call(raw_tc)
        if tc is None:
            continue
        tc_prefix = f"{prefix}.{MessageAttributes.MESSAGE_TOOL_CALLS}.{tc_idx}"
        if tc["id"]:
            safe_set_attribute(
                span, f"{tc_prefix}.{ToolCallAttributes.TOOL_CALL_ID}", tc["id"]
            )
        fn = tc["function"]
        if fn["name"]:
            safe_set_attribute(
                span,
                f"{tc_prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}",
                fn["name"],
            )
        if fn["arguments"] is not None:
            safe_set_attribute(
                span,
                f"{tc_prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                fn["arguments"],
            )


def _emit_input_message_extras(span: "Span", prefix: str, message: dict) -> None:
    """Emit additive attributes for an input message:

    - `MESSAGE_NAME` and `MESSAGE_TOOL_CALL_ID` (commonly set on tool-result
      messages so traces show which tool produced which result).
    - `MESSAGE_TOOL_CALLS.*` when an assistant message requested tools.
    - `MESSAGE_CONTENTS.*` structured content for list-shaped content
      (multimodal text + image parts). The plain `MESSAGE_CONTENT` write is
      still performed by the caller, so renderers that only read the legacy
      key continue to work.
    """
    if not isinstance(message, dict):
        return

    name = message.get("name")
    if name:
        safe_set_attribute(span, f"{prefix}.{MessageAttributes.MESSAGE_NAME}", name)

    tool_call_id = message.get("tool_call_id")
    if tool_call_id:
        safe_set_attribute(
            span,
            f"{prefix}.{MessageAttributes.MESSAGE_TOOL_CALL_ID}",
            tool_call_id,
        )

    _emit_message_tool_calls(span, prefix, message)

    content = message.get("content")
    if isinstance(content, list):
        contents_prefix = f"{prefix}.{MessageAttributes.MESSAGE_CONTENTS}"
        for part_idx, part in enumerate(content):
            if not isinstance(part, dict):
                continue
            part_prefix = f"{contents_prefix}.{part_idx}"
            part_type = part.get("type")
            if part_type in ("text", "input_text"):
                text = part.get("text")
                if isinstance(text, str):
                    safe_set_attribute(
                        span,
                        f"{part_prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}",
                        "text",
                    )
                    safe_set_attribute(
                        span,
                        f"{part_prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}",
                        text,
                    )
            elif part_type in ("image_url", "image", "input_image"):
                url = None
                image = part.get("image_url")
                if isinstance(image, dict):
                    url = image.get("url")
                elif isinstance(image, str):
                    url = image
                if not url:
                    # Anthropic-style source.{type=base64,media_type,data}
                    source = part.get("source")
                    if isinstance(source, dict) and source.get("data"):
                        media_type = source.get("media_type", "image/jpeg")
                        url = f"data:{media_type};base64,{source['data']}"
                    elif isinstance(part.get("url"), str):
                        url = part["url"]
                if url:
                    safe_set_attribute(
                        span,
                        f"{part_prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}",
                        "image",
                    )
                    safe_set_attribute(
                        span,
                        f"{part_prefix}.message_content.image.image.url",
                        url,
                    )


def _set_session_and_user_attrs(
    span: "Span", kwargs: dict, standard_logging_payload
) -> None:
    """Emit `SESSION_ID` / `USER_ID` / team metadata when source data exists.

    `SESSION_ID` is emitted only when an explicit end-user identifier exists
    (`metadata.user_api_key_end_user_id`). We deliberately do NOT fall back
    to `trace_id`, because that would create a distinct "session" for every
    single request and distort Arize's Session-grouping analytics. The
    `trace_id` is still emitted under its own `litellm.trace_id` key so
    spans remain filterable by trace.

    USER_ID is *only* emitted when no upstream path (model_params.user or
    optional_params.user) has already set it, to avoid overwriting an
    existing value with a possibly-different one from API-key metadata.
    """
    if not isinstance(standard_logging_payload, dict):
        return
    metadata = standard_logging_payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        return

    session_id = metadata.get("user_api_key_end_user_id")
    if session_id:
        safe_set_attribute(span, SpanAttributes.SESSION_ID, str(session_id))

    trace_id = standard_logging_payload.get("trace_id")
    if trace_id:
        safe_set_attribute(span, "litellm.trace_id", str(trace_id))

    optional_params = kwargs.get("optional_params") or {}
    model_params = standard_logging_payload.get("model_parameters") or {}
    has_user_already = bool(
        (isinstance(optional_params, dict) and optional_params.get("user"))
        or (isinstance(model_params, dict) and model_params.get("user"))
    )
    if not has_user_already:
        user_id = metadata.get("user_api_key_user_id")
        if user_id:
            safe_set_attribute(span, SpanAttributes.USER_ID, str(user_id))

    team_id = metadata.get("user_api_key_team_id")
    if team_id:
        safe_set_attribute(span, "litellm.team_id", str(team_id))
    team_alias = metadata.get("user_api_key_team_alias")
    if team_alias:
        safe_set_attribute(span, "litellm.team_alias", str(team_alias))
    key_alias = metadata.get("user_api_key_alias")
    if key_alias:
        safe_set_attribute(span, "litellm.key_alias", str(key_alias))


def _set_response_cost_attr(span: "Span", standard_logging_payload) -> None:
    """Emit cost attributes from the StandardLoggingPayload when present.

    Uses the OpenInference `llm.cost.total` key so Arize / Phoenix can
    surface the cost in their "Total Cost" column. LiteLLM only tracks a
    single total in `StandardLoggingPayload.response_cost`, so we cannot
    split it into prompt/completion. We also keep the legacy
    `llm.response.cost` key for back-compat with any consumer querying it.
    """
    if not isinstance(standard_logging_payload, dict):
        return
    cost = standard_logging_payload.get("response_cost")
    if cost is None:
        return
    try:
        cost_value = float(cost)
    except (TypeError, ValueError):
        return
    safe_set_attribute(span, "llm.cost.total", cost_value)
    safe_set_attribute(span, "llm.response.cost", cost_value)


def _is_passthrough_call_type(call_type: Optional[str]) -> bool:
    if not call_type:
        return False
    lowered = str(call_type).lower()
    return "passthrough" in lowered or "pass_through" in lowered


def _maybe_normalize_passthrough(
    span: "Span",
    kwargs: dict,
    raw_response_obj,
    coerced_response_obj,
    standard_logging_payload,
) -> None:
    """Surface input/output text for passthrough routes (e.g. Bedrock
    InvokeModel) so the parent span renders as more than `usage` numbers.

    Only runs when `call_type` is a passthrough variant. Reads from:
      - `kwargs["additional_args"]["complete_input_dict"]` for input
      - the coerced response (or `kwargs["original_response"]`) for output

    All emits are best-effort: if the provider shape isn't recognized the
    helper exits silently. Existing chat/completion paths never enter this
    helper because their call_type doesn't contain "passthrough".

    TEMPORARY BRIDGE: passthrough handlers don't populate the
    StandardLoggingPayload `messages` field today (they call
    `transform_response(messages=[])`), so the input is only available via
    `additional_args.complete_input_dict`. The proper fix is upstream in
    `base_passthrough_logging_handler._create_response_logging_payload()`:
    once that populates SLP `messages`/`response`, every callback gets
    passthrough I/O (with central redaction) for free and this helper's
    `complete_input_dict` fallback can be deleted. See follow-up issue.
    """
    call_type = (
        standard_logging_payload.get("call_type")
        if isinstance(standard_logging_payload, dict)
        else None
    )
    if not _is_passthrough_call_type(call_type):
        return

    # Respect LiteLLM's central message-redaction contract. The normal
    # chat/completion path is redacted by `perform_redaction` before
    # callbacks run, but `complete_input_dict` (read below) is NOT covered by
    # that layer — so without this gate, an operator who enabled redaction
    # would still see raw passthrough prompts in Arize. Skip entirely when
    # redaction is on so neither input nor output leaks through this bridge.
    if should_redact_message_logging(kwargs):
        return

    # --- INPUT --------------------------------------------------------------
    additional_args = kwargs.get("additional_args") or {}
    complete_input_dict = (
        additional_args.get("complete_input_dict")
        if isinstance(additional_args, dict)
        else None
    )
    if isinstance(complete_input_dict, dict):
        _set_passthrough_input_attributes(span, complete_input_dict.get("messages"))

    # --- OUTPUT -------------------------------------------------------------
    parsed_response = _parse_passthrough_response(
        raw_response_obj, coerced_response_obj, kwargs
    )
    if not isinstance(parsed_response, dict):
        return

    _set_passthrough_output_attributes(span, parsed_response)


def _set_passthrough_input_attributes(span: "Span", messages) -> None:
    """Render passthrough request messages into INPUT_VALUE + LLM_INPUT_MESSAGES."""
    if not (isinstance(messages, list) and messages):
        return
    # Set INPUT_VALUE from the last user message text if discoverable.
    last_text = None
    for msg in reversed(messages):
        if isinstance(msg, dict):
            last_text = _coerce_text(msg.get("content"))
            if last_text:
                break
    if last_text:
        safe_set_attribute(span, SpanAttributes.INPUT_VALUE, last_text)
    # Mirror messages into LLM_INPUT_MESSAGES so the input pane renders.
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        prefix = f"{SpanAttributes.LLM_INPUT_MESSAGES}.{idx}"
        role = msg.get("role")
        if role:
            safe_set_attribute(
                span,
                f"{prefix}.{MessageAttributes.MESSAGE_ROLE}",
                role,
            )
        text = _coerce_text(msg.get("content"))
        if text is not None:
            safe_set_attribute(
                span,
                f"{prefix}.{MessageAttributes.MESSAGE_CONTENT}",
                text,
            )


def _set_passthrough_output_attributes(span: "Span", parsed_response: dict) -> None:
    """Render passthrough response into OUTPUT_VALUE + LLM_OUTPUT_MESSAGES."""
    # Anthropic / Bedrock-Anthropic: `content` is a list of typed parts.
    content_list = parsed_response.get("content")
    if isinstance(content_list, list) and content_list:
        texts = []
        for part in content_list:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
        joined = "\n\n".join(t for t in texts if t)
        if joined:
            safe_set_attribute(span, SpanAttributes.OUTPUT_VALUE, joined)
            prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"
            safe_set_attribute(
                span,
                f"{prefix}.{MessageAttributes.MESSAGE_ROLE}",
                parsed_response.get("role", "assistant"),
            )
            safe_set_attribute(
                span,
                f"{prefix}.{MessageAttributes.MESSAGE_CONTENT}",
                joined,
            )

    # OpenAI-style passthrough: `choices[0].message.content`
    choices = parsed_response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            msg = first.get("message")
            if isinstance(msg, dict):
                text = _coerce_text(msg.get("content"))
                if text:
                    safe_set_attribute(span, SpanAttributes.OUTPUT_VALUE, text)
                    prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"
                    safe_set_attribute(
                        span,
                        f"{prefix}.{MessageAttributes.MESSAGE_ROLE}",
                        msg.get("role", "assistant"),
                    )
                    safe_set_attribute(
                        span,
                        f"{prefix}.{MessageAttributes.MESSAGE_CONTENT}",
                        text,
                    )


def _parse_passthrough_response(raw_response_obj, coerced_response_obj, kwargs):
    """Return a dict view of the provider response for passthrough routes."""
    # Prefer the coerced view (already JSON-parsed for httpx.Response).
    candidates = []
    if isinstance(coerced_response_obj, dict):
        candidates.append(coerced_response_obj)
    if (
        isinstance(raw_response_obj, dict)
        and raw_response_obj is not coerced_response_obj
    ):
        candidates.append(raw_response_obj)

    for candidate in candidates:
        # StandardPassThroughResponseObject wrapper: {"response": "..."}.
        if (
            "response" in candidate
            and "content" not in candidate
            and "choices" not in candidate
        ):
            inner = candidate.get("response")
            if isinstance(inner, str):
                try:
                    parsed = json.loads(inner)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    continue
            if isinstance(inner, dict):
                return inner
        else:
            return candidate

    # Fallback: kwargs["original_response"] from the OTel base path.
    original = kwargs.get("original_response") if isinstance(kwargs, dict) else None
    if isinstance(original, dict):
        return original
    if isinstance(original, str):
        try:
            parsed = json.loads(original)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
    return None
