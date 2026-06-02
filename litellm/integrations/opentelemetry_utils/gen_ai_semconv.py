"""OTEL GenAI ``gen_ai_latest_experimental`` semantic conventions.

Setting ``OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`` switches the
emitted traces to the experimental OTEL GenAI conventions
(https://opentelemetry.io/docs/specs/semconv/gen-ai/). Concretely, versus the
default LiteLLM output:

Request span:

- name is ``{operation} {model}`` (e.g. ``chat gpt-4``) instead of
  ``litellm_request``; span kind is ``CLIENT``.
- ``gen_ai.operation.name`` is the actual operation (``chat`` /
  ``text_completion`` / ``embeddings``) instead of always ``chat``.
- the provider is reported as ``gen_ai.provider.name``; the superseded
  ``gen_ai.system`` and the legacy ``llm.is_streaming`` are dropped.
- adds ``gen_ai.request.{frequency_penalty,presence_penalty,top_k,seed}``,
  ``gen_ai.request.stop_sequences`` (a string array),
  ``gen_ai.request.stream`` (only when streaming),
  ``gen_ai.request.choice.count`` (only when n > 1), and
  ``gen_ai.usage.cache_{creation,read}.input_tokens``.
- the non-standard ``raw_gen_ai_request`` child span is no longer created.

Events:

- the per-message ``gen_ai.content.prompt`` / per-choice
  ``gen_ai.content.completion`` log events are replaced by a single
  ``gen_ai.client.inference.operation.details`` log event carrying
  ``gen_ai.input.messages`` / ``gen_ai.output.messages`` (message content
  included only when content capture is enabled).
"""

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.integrations.opentelemetry import OpenTelemetryConfig

    Span = Union[_Span, Any]
else:
    Span = Any


# OTEL_SEMCONV_STABILITY_OPT_IN is a comma-separated list of category-specific
# opt-in values. See https://opentelemetry.io/docs/specs/semconv/gen-ai/
OTEL_SEMCONV_STABILITY_OPT_IN_ENV = "OTEL_SEMCONV_STABILITY_OPT_IN"


class OTELSemconvCategory(Enum):
    GEN_AI_LATEST_EXPERIMENTAL = "gen_ai_latest_experimental"


# Reverse lookup: opt-in token string -> OTELSemconvCategory.
_SEMCONV_CATEGORY_BY_VALUE = {
    category.value: category for category in OTELSemconvCategory
}


# LiteLLM optional_params key -> OTEL gen_ai semconv span attribute.
_SEMCONV_REQUEST_ATTRIBUTES = {
    "frequency_penalty": "gen_ai.request.frequency_penalty",
    "presence_penalty": "gen_ai.request.presence_penalty",
    "top_k": "gen_ai.request.top_k",
    "seed": "gen_ai.request.seed",
}

# usage_object key -> OTEL gen_ai semconv cache-token span attribute.
_SEMCONV_CACHE_TOKEN_ATTRIBUTES = {
    "cache_creation_input_tokens": "gen_ai.usage.cache_creation.input_tokens",
    "cache_read_input_tokens": "gen_ai.usage.cache_read.input_tokens",
}

# Name of the consolidated GenAI inference event (replaces the legacy
# per-message gen_ai.content.prompt / per-choice gen_ai.content.completion).
_INFERENCE_DETAILS_EVENT_NAME = "gen_ai.client.inference.operation.details"


def parse_semconv_opt_in(raw: Optional[str]) -> Set[OTELSemconvCategory]:
    """Parse the comma-separated OTEL_SEMCONV_STABILITY_OPT_IN value into the
    set of recognized categories. Unknown tokens are ignored per the spec."""
    if not raw:
        return set()
    return {
        _SEMCONV_CATEGORY_BY_VALUE[token]
        for token in (part.strip() for part in raw.split(","))
        if token in _SEMCONV_CATEGORY_BY_VALUE
    }


class OTELGenAISemconvMixin:
    """OTEL GenAI ``gen_ai_latest_experimental`` semantic-convention behavior.

    Mixed into ``OpenTelemetry`` (its only host). Every member is internal to
    the OTEL integration; the leading underscore marks "subsystem-internal",
    not "class-private" (the host lives in a sibling module).

    Members the host calls (the mixin -> host contract):

    - ``_gen_ai_semconv_latest_experimental`` -- opt-in gate; guards every
      semconv code path in ``opentelemetry.py``.
    - ``_gen_ai_operation_name`` -- LiteLLM ``call_type`` -> spec
      ``gen_ai.operation.name``.
    - ``_set_semconv_request_attributes`` /
      ``_set_semconv_cache_token_attributes`` -- add the ``gen_ai.request.*``
      / ``gen_ai.usage.cache_*`` span attributes.
    - ``_emit_inference_details_event`` -- emit the consolidated event.

    Helpers the host must provide (declared under ``TYPE_CHECKING`` below):
    ``config``, ``safe_set_attribute``, ``_capture_in_event``,
    ``_transform_messages_to_otel_semantic_conventions``,
    ``_transform_choices_to_otel_semantic_conventions``, ``_to_ns``,
    ``_otel_log_types``.
    """

    if TYPE_CHECKING:
        config: "OpenTelemetryConfig"

        def safe_set_attribute(self, span: Span, key: str, value: Any) -> None: ...

        def _capture_in_event(self) -> bool: ...

        def _transform_messages_to_otel_semantic_conventions(
            self, messages: Union[List[dict], str]
        ) -> List[dict]: ...

        def _transform_choices_to_otel_semantic_conventions(
            self, choices: List[dict]
        ) -> List[dict]: ...

        def _to_ns(self, dt: datetime) -> int: ...

        def _otel_log_types(self) -> Tuple[Any, Any]: ...

    @property
    def _gen_ai_semconv_latest_experimental(self) -> bool:
        """Whether the ``gen_ai_latest_experimental`` opt-in is active.

        Every semconv behavior is gated on this; ``False`` => legacy output.
        """
        return (
            OTELSemconvCategory.GEN_AI_LATEST_EXPERIMENTAL
            in self.config.semconv_stability_opt_in
        )

    @staticmethod
    def _gen_ai_operation_name(kwargs: dict) -> str:
        """Map a LiteLLM ``call_type`` to spec ``gen_ai.operation.name``.

        Substring match (e.g. ``aembedding`` -> ``embeddings``); defaults to
        ``chat``.
        """
        call_type = kwargs.get("call_type", "") or ""
        match call_type:
            case s if "embedding" in s:
                return "embeddings"
            case s if "text_completion" in s:
                return "text_completion"
            case _:
                return "chat"

    def _set_semconv_request_attributes(
        self, span: Span, optional_params: dict
    ) -> None:
        """Add ``gen_ai.request.*`` span attributes from ``optional_params``.

        Covers the sampling params plus the conditionally-required
        ``stop_sequences`` / ``stream`` / ``choice.count`` per the spec.
        """
        for source_key, semconv_key in _SEMCONV_REQUEST_ATTRIBUTES.items():
            value = optional_params.get(source_key)
            if value is not None:
                self.safe_set_attribute(span=span, key=semconv_key, value=value)

        stop = optional_params.get("stop")
        if stop is not None:
            # Spec types this as string[]. safe_set_attribute coerces to a
            # primitive, so set the array directly via the span API.
            stop_list = stop if isinstance(stop, list) else [stop]
            span.set_attribute(
                "gen_ai.request.stop_sequences", [str(s) for s in stop_list]
            )

        # Conditionally required: set only when the request is streaming.
        if optional_params.get("stream"):
            self.safe_set_attribute(span=span, key="gen_ai.request.stream", value=True)

        # Conditionally required per spec ("if available and != 1"). Valid n is
        # an int >= 1, so n > 1 is equivalent for conformant input while
        # suppressing nonsensical values (0, negative, non-int).
        n = optional_params.get("n")
        if isinstance(n, int) and n > 1:
            self.safe_set_attribute(
                span=span, key="gen_ai.request.choice.count", value=n
            )

    def _set_semconv_cache_token_attributes(
        self, span: Span, standard_logging_payload
    ) -> None:
        """Add ``gen_ai.usage.cache_*.input_tokens`` from the usage object.

        No-op when the payload or the usage values are missing/zero.
        """
        if not standard_logging_payload:
            return
        usage = (standard_logging_payload.get("metadata") or {}).get(
            "usage_object"
        ) or {}
        for source_key, semconv_key in _SEMCONV_CACHE_TOKEN_ATTRIBUTES.items():
            value = usage.get(source_key)
            if value:
                self.safe_set_attribute(span=span, key=semconv_key, value=value)

    def _build_inference_details_attrs(
        self, kwargs: dict, response_obj: dict, provider: str
    ) -> Dict[str, Any]:
        """Build the attribute payload for the inference-details event.

        Always includes provider/operation; input/output messages are added
        only when content capture is enabled and non-empty. Mixin-internal.
        """
        attrs: Dict[str, Any] = {
            "event_name": _INFERENCE_DETAILS_EVENT_NAME,
            "gen_ai.provider.name": provider,
            "gen_ai.operation.name": self._gen_ai_operation_name(kwargs),
        }
        if not self._capture_in_event():
            return attrs

        input_messages = self._transform_messages_to_otel_semantic_conventions(
            kwargs.get("messages") or []
        )
        output_messages = self._transform_choices_to_otel_semantic_conventions(
            response_obj.get("choices", [])
        )
        if input_messages:
            attrs["gen_ai.input.messages"] = safe_dumps(input_messages)
        if output_messages:
            attrs["gen_ai.output.messages"] = safe_dumps(output_messages)
        return attrs

    def _emit_inference_details_event(
        self,
        kwargs: dict,
        response_obj: dict,
        provider: str,
        otel_logger,
        parent_ctx,
    ) -> None:
        """Emit the consolidated ``gen_ai.client.inference.operation.details``
        log event, correlated to the request span via ``parent_ctx``.

        Replaces the legacy per-message / per-choice content events.
        """
        LogRecord, SeverityNumber = self._otel_log_types()
        log_record = LogRecord(
            timestamp=self._to_ns(datetime.now()),
            trace_id=parent_ctx.trace_id,
            span_id=parent_ctx.span_id,
            trace_flags=parent_ctx.trace_flags,
            severity_number=SeverityNumber.INFO,
            severity_text="INFO",
            body=None,
            attributes=self._build_inference_details_attrs(
                kwargs, response_obj, provider
            ),
        )
        otel_logger.emit(log_record)
