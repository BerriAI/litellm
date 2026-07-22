import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

import litellm
from litellm._logging import verbose_logger
from litellm.integrations._types.open_inference import (
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.opentelemetry_utils.gen_ai_semconv import (
    OTEL_SEMCONV_STABILITY_OPT_IN_ENV,
    OTELGenAISemconvMixin,
    OTELSemconvCategory,
    parse_semconv_opt_in,
)
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.litellm_core_utils.secret_redaction import redact_string
from litellm.secret_managers.main import get_secret_bool, str_to_bool
from litellm.types.services import ServiceLoggerPayload
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    CostBreakdown,
    Function,
    LLMResponseTypes,
    StandardCallbackDynamicParams,
    StandardLoggingPayload,
)

# OpenTelemetry imports moved to individual functions to avoid import errors when not installed

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export import SpanExporter as _SpanExporter
    from opentelemetry.trace import Context as _Context
    from opentelemetry.trace import Span as _Span
    from opentelemetry.trace import Tracer as _Tracer

    from litellm.proxy._types import (
        ManagementEndpointLoggingPayload as _ManagementEndpointLoggingPayload,
    )
    from litellm.proxy.proxy_server import UserAPIKeyAuth as _UserAPIKeyAuth

    Span = Union[_Span, Any]
    Tracer = Union[_Tracer, Any]
    Context = Union[_Context, Any]
    SpanExporter = Union[_SpanExporter, Any]
    UserAPIKeyAuth = Union[_UserAPIKeyAuth, Any]
    ManagementEndpointLoggingPayload = Union[_ManagementEndpointLoggingPayload, Any]
else:
    Span = Any
    Tracer = Any
    SpanExporter = Any
    UserAPIKeyAuth = Any
    ManagementEndpointLoggingPayload = Any
    Context = Any

LITELLM_TRACER_NAME = os.getenv("OTEL_TRACER_NAME", "litellm")
LITELLM_METER_NAME = os.getenv("LITELLM_METER_NAME", "litellm")
LITELLM_LOGGER_NAME = os.getenv("LITELLM_LOGGER_NAME", "litellm")
LITELLM_PROXY_REQUEST_SPAN_NAME = "Received Proxy Server Request"
# OTel-standard names. status is also kept under error.code for back compat.
HTTP_RESPONSE_STATUS_CODE_ATTRIBUTE = "http.response.status_code"
HTTP_ROUTE_ATTRIBUTE = "http.route"
URL_PATH_ATTRIBUTE = "url.path"
PREPROCESSING_DURATION_MS_ATTRIBUTE = "litellm.preprocessing.duration_ms"
TEAM_METADATA_ATTRIBUTE = "litellm.team.metadata"
MODEL_GROUP_ATTRIBUTE = "litellm.model_group"
PROVIDER_MODEL_ATTRIBUTE = "litellm.provider.model"
# Remove the hardcoded LITELLM_RESOURCE dictionary - we'll create it properly later
RAW_REQUEST_SPAN_NAME = "raw_gen_ai_request"
LITELLM_REQUEST_SPAN_NAME = "litellm_request"

CAPTURE_MODE_NO_CONTENT = "NO_CONTENT"
CAPTURE_MODE_SPAN_ONLY = "SPAN_ONLY"
CAPTURE_MODE_EVENT_ONLY = "EVENT_ONLY"
CAPTURE_MODE_SPAN_AND_EVENT = "SPAN_AND_EVENT"
_VALID_CAPTURE_MODES = {
    CAPTURE_MODE_NO_CONTENT,
    CAPTURE_MODE_SPAN_ONLY,
    CAPTURE_MODE_EVENT_ONLY,
    CAPTURE_MODE_SPAN_AND_EVENT,
}

METRIC_METADATA_KEYS: Tuple[str, ...] = (
    "user_api_key_hash",
    "user_api_key_alias",
    "user_api_key_team_id",
    "user_api_key_org_id",
    "user_api_key_user_id",
    "user_api_key_team_alias",
    "user_api_key_user_email",
    "spend_logs_metadata",
    "requester_ip_address",
    "requester_metadata",
    "user_api_key_end_user_id",
    "prompt_management_metadata",
    "applied_guardrails",
    "mcp_tool_call_metadata",
    "vector_store_request_metadata",
)

TOKEN_TYPE_ATTRIBUTE: str = "gen_ai.token.type"

VALID_METRIC_ATTRIBUTE_NAMES: FrozenSet[str] = frozenset(
    (
        "gen_ai.operation.name",
        "gen_ai.system",
        "gen_ai.request.model",
        "gen_ai.framework",
        "hidden_params",
    )
    + tuple(f"metadata.{key}" for key in METRIC_METADATA_KEYS)
)


@dataclass(frozen=True)
class OTELMetricAttributeFilter:
    include_list: Optional[List[str]] = None
    exclude_list: Optional[List[str]] = None


def _build_metric_attribute_filter(value: Any) -> OTELMetricAttributeFilter:
    if isinstance(value, OTELMetricAttributeFilter):
        return value
    if not isinstance(value, dict):
        raise ValueError(
            "otel.attributes must be a mapping with optional 'include_list' / "
            f"'exclude_list', got {type(value).__name__}"
        )
    return OTELMetricAttributeFilter(
        include_list=value.get("include_list"),
        exclude_list=value.get("exclude_list"),
    )


def _resolve_metric_attribute_filter(
    attributes: Optional[OTELMetricAttributeFilter],
) -> Tuple[Optional[FrozenSet[str]], Optional[FrozenSet[str]]]:
    if attributes is None:
        return None, None
    include = attributes.include_list or None
    exclude = attributes.exclude_list or None
    if include and exclude:
        raise ValueError("otel.attributes: include_list and exclude_list are mutually exclusive")
    requested = include or exclude or []
    if TOKEN_TYPE_ATTRIBUTE in requested:
        raise ValueError(
            f"otel.attributes: {TOKEN_TYPE_ATTRIBUTE} is a structural token-usage discriminator and cannot be filtered"
        )
    unknown = sorted(name for name in requested if name not in VALID_METRIC_ATTRIBUTE_NAMES)
    if unknown:
        raise ValueError(
            f"otel.attributes: unknown attribute name(s) {unknown}. Valid names: {sorted(VALID_METRIC_ATTRIBUTE_NAMES)}"
        )
    return (
        frozenset(include) if include else None,
        frozenset(exclude) if exclude else None,
    )


def _normalize_team_metadata_keys(value: Any) -> List[str]:
    """Coerce a team-metadata allowlist from a list or comma-separated string.

    config.yaml passes a YAML list; an env var passes a comma-separated string.
    Both collapse to a list of stripped, non-empty keys.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


_FREEZE_MAX_DEPTH = 16

HashableScope = Union[
    str,
    int,
    float,
    bool,
    bytes,
    None,
    tuple["HashableScope", ...],
    frozenset["HashableScope"],
]


def _freeze_for_dedupe(value: object, _depth: int = 0) -> HashableScope:
    if _depth >= _FREEZE_MAX_DEPTH:
        return repr(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_for_dedupe(item, _depth + 1) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_for_dedupe(item, _depth + 1) for item in value)
    if isinstance(value, dict):
        return frozenset(
            (_freeze_for_dedupe(key, _depth + 1), _freeze_for_dedupe(item, _depth + 1)) for key, item in value.items()
        )
    if isinstance(value, (str, int, float, bytes)) or value is None:
        return value
    return repr(value)


@dataclass
class OpenTelemetryConfig:
    exporter: Union[str, SpanExporter] = "console"
    endpoint: Optional[str] = None
    headers: Optional[str] = None
    enable_metrics: bool = False
    enable_events: bool = False
    service_name: Optional[str] = None
    deployment_environment: Optional[str] = None
    model_id: Optional[str] = None
    ignore_context_propagation: Optional[bool] = None
    # When True, create a private TracerProvider instead of reusing or setting the global one.
    skip_set_global: bool = False
    # Programmatic override for OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT.
    # One of NO_CONTENT, SPAN_ONLY, EVENT_ONLY, SPAN_AND_EVENT (or "true" as legacy alias).
    capture_message_content: Optional[str] = None
    semconv_stability_opt_in: Set[OTELSemconvCategory] = field(default_factory=set)
    # Sub-keys of the team's free-form metadata stamped onto the inference span
    # under ``litellm.team.metadata``. Empty by default so none of a team's
    # metadata leaves the process until explicitly allowlisted.
    baggage_team_metadata_keys: List[str] = field(default_factory=list)
    # Prometheus-style include/exclude control over which attributes are stamped
    # on emitted metrics, to cap metric cardinality.
    attributes: Optional[OTELMetricAttributeFilter] = None

    def __post_init__(self) -> None:
        # If endpoint is specified but exporter is still the default "console",
        # automatically infer "otlp_http" to send traces to the endpoint.
        # This fixes an issue where UI-configured OTEL settings would default
        # to console output instead of sending traces to the configured endpoint.
        if self.endpoint and isinstance(self.exporter, str) and self.exporter == "console":
            self.exporter = "otlp_http"

        if not self.service_name:
            self.service_name = os.getenv("OTEL_SERVICE_NAME", "litellm")
        if not self.deployment_environment:
            self.deployment_environment = os.getenv("OTEL_ENVIRONMENT_NAME", "production")
        if not self.model_id:
            self.model_id = os.getenv("OTEL_MODEL_ID", self.service_name)
        if self.ignore_context_propagation is None:
            self.ignore_context_propagation = str_to_bool(os.getenv("OTEL_IGNORE_CONTEXT_PROPAGATION"))
        # Resolve the env opt-in once here so self.semconv_stability_opt_in is the
        # single source of truth: the union of programmatic and env categories.
        self.semconv_stability_opt_in |= parse_semconv_opt_in(os.getenv(OTEL_SEMCONV_STABILITY_OPT_IN_ENV))
        self.baggage_team_metadata_keys = _normalize_team_metadata_keys(
            self.baggage_team_metadata_keys
        ) or _normalize_team_metadata_keys(os.getenv("LITELLM_OTEL_BAGGAGE_TEAM_METADATA_KEYS"))

    @classmethod
    def from_env(cls):
        """
        OTEL_HEADERS=x-honeycomb-team=B85YgLm9****
        OTEL_EXPORTER="otlp_http"
        OTEL_ENDPOINT="https://api.honeycomb.io/v1/traces"

        OTEL_HEADERS gets sent as headers = {"x-honeycomb-team": "B85YgLm96******"}
        """
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        exporter = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", os.getenv("OTEL_EXPORTER", "console"))
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", os.getenv("OTEL_ENDPOINT"))
        headers = os.getenv(
            "OTEL_EXPORTER_OTLP_HEADERS", os.getenv("OTEL_HEADERS")
        )  # example: OTEL_HEADERS=x-honeycomb-team=B85YgLm96***"
        enable_metrics: bool = os.getenv("LITELLM_OTEL_INTEGRATION_ENABLE_METRICS", "false").lower() == "true"
        enable_events: bool = os.getenv("LITELLM_OTEL_INTEGRATION_ENABLE_EVENTS", "false").lower() == "true"
        service_name = os.getenv("OTEL_SERVICE_NAME", "litellm")
        deployment_environment = os.getenv("OTEL_ENVIRONMENT_NAME", "production")
        model_id = os.getenv("OTEL_MODEL_ID", service_name)

        if exporter == "in_memory":
            return cls(exporter=InMemorySpanExporter())
        return cls(
            exporter=exporter,
            endpoint=endpoint,
            headers=headers,  # example: OTEL_HEADERS=x-honeycomb-team=B85YgLm96***"
            enable_metrics=enable_metrics,
            enable_events=enable_events,
            service_name=service_name,
            deployment_environment=deployment_environment,
            model_id=model_id,
        )


class OpenTelemetry(OTELGenAISemconvMixin, CustomLogger):
    def __init__(
        self,
        config: Optional[OpenTelemetryConfig] = None,
        callback_name: Optional[str] = None,
        # injection points for testing
        tracer_provider: Optional[Any] = None,
        logger_provider: Optional[Any] = None,
        meter_provider: Optional[Any] = None,
        **kwargs,
    ):
        team_metadata_keys_override = kwargs.pop("baggage_team_metadata_keys", None)
        metric_attributes_override = kwargs.pop("attributes", None)
        if config is None:
            config = OpenTelemetryConfig.from_env()
        if team_metadata_keys_override is not None:
            config.baggage_team_metadata_keys = _normalize_team_metadata_keys(team_metadata_keys_override)
        if metric_attributes_override is not None:
            config.attributes = _build_metric_attribute_filter(metric_attributes_override)

        self.config = config
        self.callback_name = callback_name
        # Resolved on first metric record, not here: the proxy populates
        # callback_settings.otel.attributes after this logger is constructed, so
        # reading it now would miss it. An explicit config is validated eagerly so
        # a bad config still fails at startup.
        self._metric_attr_include: Optional[FrozenSet[str]] = None
        self._metric_attr_exclude: Optional[FrozenSet[str]] = None
        self._metric_attr_filter_resolved = False
        if config.attributes is not None:
            self._ensure_metric_attribute_filter()
        self.OTEL_EXPORTER = self.config.exporter
        self.OTEL_ENDPOINT = self.config.endpoint
        self.OTEL_HEADERS = self.config.headers
        self._tracer_provider_cache: Dict[str, Any] = {}
        self._init_tracing(tracer_provider)

        _debug_otel = str(os.getenv("DEBUG_OTEL", "False")).lower()

        if _debug_otel == "true":
            # Set up logging
            import logging

            logging.basicConfig(level=logging.DEBUG)
            logging.getLogger(__name__)

            # Enable OpenTelemetry logging
            otel_exporter_logger = logging.getLogger("opentelemetry.sdk.trace.export")
            otel_exporter_logger.setLevel(logging.DEBUG)

        # init CustomLogger params
        super().__init__(**kwargs)
        self._init_metrics(meter_provider)
        self._init_logs(logger_provider)
        # Sample env-var / config / message_logging at init so subsequent
        # _capture_in_span / _capture_in_event calls are deterministic.
        self._capture_mode_cached = self._compute_capture_mode_from_init_state()
        self._init_otel_logger_on_litellm_proxy()

    @staticmethod
    def _get_litellm_resource(config: OpenTelemetryConfig):
        """Create an OpenTelemetry Resource using config-driven defaults."""
        from opentelemetry.sdk.resources import OTELResourceDetector, Resource

        base_attributes: Dict[str, Optional[str]] = {
            "service.name": config.service_name,
            "deployment.environment": config.deployment_environment,
            "model_id": config.model_id or config.service_name,
        }

        base_resource = Resource.create(base_attributes)  # type: ignore[arg-type]
        otel_resource_detector = OTELResourceDetector()
        env_resource = otel_resource_detector.detect()
        return base_resource.merge(env_resource)

    def _init_otel_logger_on_litellm_proxy(self):
        """
        Initializes OpenTelemetry for litellm proxy server

        - Adds Otel as a service callback
        - Sets `proxy_server.open_telemetry_logger` to self
        """
        try:
            from litellm.proxy import proxy_server
        except ImportError:
            verbose_logger.warning("Proxy Server is not installed. Skipping OpenTelemetry initialization.")
            return

        # Add self as a service callback
        if "otel" not in litellm.service_callback and all(
            not isinstance(cb, OpenTelemetry) for cb in litellm.service_callback
        ):
            litellm.service_callback.append(self)
        # avoid proxy logger ownership being overwritten by later
        # handlers. Multiple integrations (default OTEL, Langfuse OTEL,
        # Arize OTEL, etc.) may initialize in sequence; without this guard,
        # the last one silently replaces the first and breaks expected
        # routing for proxy_server.open_telemetry_logger consumers.
        # Behavior: first-registered wins.
        if getattr(proxy_server, "open_telemetry_logger", None) is None:
            setattr(proxy_server, "open_telemetry_logger", self)

    def _get_or_create_provider(
        self,
        provider,
        provider_name: str,
        get_existing_provider_fn,
        sdk_provider_class,
        create_new_provider_fn,
        set_provider_fn,
        skip_set_global: bool = False,
    ):
        """
        Generic helper to get or create an OpenTelemetry provider (Tracer, Meter, or Logger).

        Args:
            provider: The provider instance passed to the init function (can be None)
            provider_name: Name for logging (e.g., "TracerProvider")
            get_existing_provider_fn: Function to get the existing global provider
            sdk_provider_class: The SDK provider class to check for (e.g., TracerProvider from SDK)
            create_new_provider_fn: Function to create a new provider instance
            set_provider_fn: Function to set the provider globally
            skip_set_global: If True, don't set the provider globally (for dynamic-only providers)

        Returns:
            The provider to use (either existing, new, or explicitly provided)
        """
        if provider is not None:
            # Provider explicitly provided (e.g., for testing)
            # Do NOT call set_provider_fn - the caller is responsible for managing global state
            # If they want it to be global, they've already set it before passing it to us
            verbose_logger.debug(
                "OpenTelemetry: Using provided TracerProvider: %s",
                type(provider).__name__,
            )
            return provider

        # Check if a provider is already set globally
        try:
            existing_provider = get_existing_provider_fn()

            if isinstance(existing_provider, sdk_provider_class):
                if skip_set_global:
                    verbose_logger.debug(
                        "OpenTelemetry: existing %s found but skip_set_global=True; creating private %s for isolation",
                        provider_name,
                        provider_name,
                    )
                    provider = create_new_provider_fn()
                else:
                    verbose_logger.debug(
                        "OpenTelemetry: Using existing %s: %s",
                        provider_name,
                        type(existing_provider).__name__,
                    )
                    provider = existing_provider
            else:
                # Default proxy provider or unknown type, create our own
                verbose_logger.debug("OpenTelemetry: Creating new %s", provider_name)
                provider = create_new_provider_fn()
                if not skip_set_global:
                    set_provider_fn(provider)
                else:
                    verbose_logger.info(
                        "OpenTelemetry: Created %s but NOT setting it globally (will use dynamic providers per-request)",
                        provider_name,
                    )
        except Exception as e:
            # Fallback: create a new provider if something goes wrong
            verbose_logger.debug(
                "OpenTelemetry: Exception checking existing %s, creating new one: %s",
                provider_name,
                str(e),
            )
            provider = create_new_provider_fn()
            if not skip_set_global:
                set_provider_fn(provider)

        return provider

    def _skip_set_global(self) -> bool:
        # langfuse_otel relies on the Langfuse SDK's providers; don't overwrite them.
        return self.config.skip_set_global or (hasattr(self, "callback_name") and self.callback_name == "langfuse_otel")

    def _compute_capture_mode_from_init_state(self) -> Optional[str]:
        """Sample explicit settings at init. Returns the resolved mode or
        None if nothing explicit is set (in which case the legacy
        ``self.message_logging`` flag is consulted dynamically per request).

        ``"true"``/``"1"`` map to ``EVENT_ONLY`` per the contrib convention.
        ``"false"``/``"0"`` map to ``NO_CONTENT``.
        Unknown values are ignored.
        """
        explicit = self.config.capture_message_content or os.getenv(
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
        )
        if not explicit:
            return None
        normalized = explicit.upper()
        if normalized in ("TRUE", "1"):
            return CAPTURE_MODE_EVENT_ONLY
        if normalized in ("FALSE", "0"):
            return CAPTURE_MODE_NO_CONTENT
        if normalized in _VALID_CAPTURE_MODES:
            return normalized
        return None

    def _resolve_capture_mode(self) -> str:
        """Return the active capture mode for this request.

        Precedence:
          1. ``litellm.turn_off_message_logging=True`` forces ``NO_CONTENT``
             (kill-switch checked dynamically).
          2. Explicit setting sampled at init from
             ``OpenTelemetryConfig.capture_message_content`` or
             ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``.
          3. Legacy ``self.message_logging`` (checked dynamically).
        """
        if litellm.turn_off_message_logging:
            return CAPTURE_MODE_NO_CONTENT
        if self._capture_mode_cached is not None:
            return self._capture_mode_cached
        return CAPTURE_MODE_SPAN_AND_EVENT if self.message_logging else CAPTURE_MODE_NO_CONTENT

    def _capture_in_span(self) -> bool:
        return self._resolve_capture_mode() in (
            CAPTURE_MODE_SPAN_ONLY,
            CAPTURE_MODE_SPAN_AND_EVENT,
        )

    def _capture_in_event(self) -> bool:
        return self._resolve_capture_mode() in (
            CAPTURE_MODE_EVENT_ONLY,
            CAPTURE_MODE_SPAN_AND_EVENT,
        )

    def _init_tracing(self, tracer_provider):
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.trace import SpanKind

        def create_tracer_provider():
            provider = TracerProvider(resource=self._get_litellm_resource(self.config))
            provider.add_span_processor(self._get_span_processor())
            return provider

        tracer_provider = self._get_or_create_provider(
            provider=tracer_provider,
            provider_name="TracerProvider",
            get_existing_provider_fn=trace.get_tracer_provider,
            sdk_provider_class=TracerProvider,
            create_new_provider_fn=create_tracer_provider,
            set_provider_fn=trace.set_tracer_provider,
            skip_set_global=self._skip_set_global(),
        )

        # Grab our tracer from the TracerProvider (not from global context)
        # This ensures we use the provided TracerProvider (e.g., for testing)
        self.tracer = tracer_provider.get_tracer(LITELLM_TRACER_NAME)
        self._tracer_provider = tracer_provider
        self.span_kind = SpanKind

    def _init_metrics(self, meter_provider):
        if not self.config.enable_metrics:
            self._meter_provider = None
            self._operation_duration_histogram = None
            self._token_usage_histogram = None
            self._cost_histogram = None
            self._time_to_first_token_histogram = None
            self._time_per_output_token_histogram = None
            self._response_duration_histogram = None
            return

        from opentelemetry import metrics
        from opentelemetry.sdk.metrics import MeterProvider

        def create_meter_provider():
            metric_reader = self._get_metric_reader()
            return MeterProvider(
                metric_readers=[metric_reader],
                resource=self._get_litellm_resource(self.config),
            )

        meter_provider = self._get_or_create_provider(
            provider=meter_provider,
            provider_name="MeterProvider",
            get_existing_provider_fn=metrics.get_meter_provider,
            sdk_provider_class=MeterProvider,
            create_new_provider_fn=create_meter_provider,
            set_provider_fn=metrics.set_meter_provider,
            skip_set_global=self._skip_set_global(),
        )
        self._meter_provider = meter_provider

        meter = meter_provider.get_meter(__name__)

        self._operation_duration_histogram = meter.create_histogram(
            name="gen_ai.client.operation.duration",  # Replace with semconv constant in otel 1.38
            description="GenAI operation duration",
            unit="s",
        )
        self._token_usage_histogram = meter.create_histogram(
            name="gen_ai.client.token.usage",  # Replace with semconv constant in otel 1.38
            description="GenAI token usage",
            unit="{token}",
        )
        self._cost_histogram = meter.create_histogram(
            name="gen_ai.client.token.cost",
            description="GenAI request cost",
            unit="USD",
        )
        self._time_to_first_token_histogram = meter.create_histogram(
            name="gen_ai.client.response.time_to_first_token",
            description="Time to first token for streaming requests",
            unit="s",
        )
        self._time_per_output_token_histogram = meter.create_histogram(
            name="gen_ai.client.response.time_per_output_token",
            description="Average time per output token (generation time / completion tokens)",
            unit="s",
        )
        self._response_duration_histogram = meter.create_histogram(
            name="gen_ai.client.response.duration",
            description="Total LLM API generation time (excludes LiteLLM overhead)",
            unit="s",
        )

    def _init_logs(self, logger_provider):
        # nothing to do if events disabled
        if not self.config.enable_events:
            self._logger_provider = None
            return

        from opentelemetry._logs import get_logger_provider, set_logger_provider
        from opentelemetry.sdk._logs import LoggerProvider as OTLoggerProvider
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

        def create_logger_provider():
            provider = OTLoggerProvider(resource=self._get_litellm_resource(self.config))
            log_exporter = self._get_log_exporter()
            provider.add_log_record_processor(
                BatchLogRecordProcessor(log_exporter)  # type: ignore[arg-type]
            )
            return provider

        self._logger_provider = self._get_or_create_provider(
            provider=logger_provider,
            provider_name="LoggerProvider",
            get_existing_provider_fn=get_logger_provider,
            sdk_provider_class=OTLoggerProvider,
            create_new_provider_fn=create_logger_provider,
            set_provider_fn=set_logger_provider,
            skip_set_global=self._skip_set_global(),
        )

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._handle_success(kwargs, response_obj, start_time, end_time)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._handle_failure(kwargs, response_obj, start_time, end_time)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._handle_success(kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._handle_failure(kwargs, response_obj, start_time, end_time)

    async def async_service_success_hook(
        self,
        payload: ServiceLoggerPayload,
        parent_otel_span: Optional[Span] = None,
        start_time: Optional[Union[datetime, float]] = None,
        end_time: Optional[Union[datetime, float]] = None,
        event_metadata: Optional[dict] = None,
    ):
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode

        _start_time_ns = 0
        _end_time_ns = 0

        if isinstance(start_time, float):
            _start_time_ns = int(start_time * 1e9)
        else:
            _start_time_ns = self._to_ns(start_time)

        if isinstance(end_time, float):
            _end_time_ns = int(end_time * 1e9)
        else:
            _end_time_ns = self._to_ns(end_time)

        if parent_otel_span is not None:
            _span_name = payload.service
            service_logging_span = self.tracer.start_span(
                name=_span_name,
                context=trace.set_span_in_context(parent_otel_span),
                start_time=_start_time_ns,
            )
            self.safe_set_attribute(
                span=service_logging_span,
                key="call_type",
                value=payload.call_type,
            )
            self.safe_set_attribute(
                span=service_logging_span,
                key="service",
                value=payload.service.value,
            )

            if event_metadata:
                for key, value in event_metadata.items():
                    if value is None:
                        value = "None"
                    if isinstance(value, dict):
                        try:
                            value = str(value)
                        except Exception:
                            value = "litellm logging error - could_not_json_serialize"
                    self.safe_set_attribute(
                        span=service_logging_span,
                        key=key,
                        value=value,
                    )
            service_logging_span.set_status(Status(StatusCode.OK))
            service_logging_span.end(end_time=_end_time_ns)

    async def async_service_failure_hook(
        self,
        payload: ServiceLoggerPayload,
        error: Optional[str] = "",
        parent_otel_span: Optional[Span] = None,
        start_time: Optional[Union[datetime, float]] = None,
        end_time: Optional[Union[float, datetime]] = None,
        event_metadata: Optional[dict] = None,
    ):
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode

        _start_time_ns = 0
        _end_time_ns = 0

        if isinstance(start_time, float):
            _start_time_ns = int(int(start_time) * 1e9)
        else:
            _start_time_ns = self._to_ns(start_time)

        if isinstance(end_time, float):
            _end_time_ns = int(int(end_time) * 1e9)
        else:
            _end_time_ns = self._to_ns(end_time)

        if parent_otel_span is not None:
            _span_name = payload.service
            service_logging_span = self.tracer.start_span(
                name=_span_name,
                context=trace.set_span_in_context(parent_otel_span),
                start_time=_start_time_ns,
            )
            self.safe_set_attribute(
                span=service_logging_span,
                key="call_type",
                value=payload.call_type,
            )
            self.safe_set_attribute(
                span=service_logging_span,
                key="service",
                value=payload.service.value,
            )
            if error:
                self.safe_set_attribute(
                    span=service_logging_span,
                    key="error",
                    value=error,
                )
            if event_metadata:
                for key, value in event_metadata.items():
                    if isinstance(value, dict):
                        try:
                            value = str(value)
                        except Exception:
                            value = "litllm logging error - could_not_json_serialize"
                    self.safe_set_attribute(
                        span=service_logging_span,
                        key=key,
                        value=value,
                    )

            service_logging_span.set_status(Status(StatusCode.ERROR))
            service_logging_span.end(end_time=_end_time_ns)

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: Optional[str] = None,
    ):
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode

        parent_otel_span = user_api_key_dict.parent_otel_span
        if parent_otel_span is not None:
            parent_otel_span.set_status(Status(StatusCode.ERROR))

            # Stamp team attributes onto the SERVER (root) span too, so the
            # trace root is team-filterable on the failure path like the
            # child exception span below.
            self._set_team_attributes_on_span(
                span=parent_otel_span,
                team_id=user_api_key_dict.team_id,
                team_alias=user_api_key_dict.team_alias,
            )

            # Stamp structured error attrs on the SERVER span itself; the
            # failure path otherwise only sets its status (_handle_failure
            # records on the litellm_request child span). Inline import:
            # litellm_logging <-> integrations is circular.
            from litellm.litellm_core_utils.litellm_logging import (
                StandardLoggingPayloadSetup,
            )

            error_information = StandardLoggingPayloadSetup.get_error_information(
                original_exception=original_exception,
                traceback_str=traceback_str,
            )
            self._record_exception_on_span(
                span=parent_otel_span,
                kwargs={
                    "exception": original_exception,
                    "standard_logging_object": {"error_information": error_information},
                },
            )

            # _record_exception_on_span only stamps when error_code is set;
            # bare TypeError etc. has none, and the span is about to be ended.
            error_code = error_information.get("error_code") if error_information else None
            if not error_code:
                self.set_response_status_code_attribute(parent_otel_span, 500)

            # Pre-request latency (request_data carries the propagated
            # metadata on the failure path; omitted if it failed before handoff).
            self.set_preprocessing_duration_attribute(parent_otel_span, request_data)

            _span_name = "Failed Proxy Server Request"

            # Exception Logging Child Span
            exception_logging_span = self.tracer.start_span(
                name=_span_name,
                context=trace.set_span_in_context(parent_otel_span),
            )
            self.safe_set_attribute(
                span=exception_logging_span,
                key="exception",
                value=str(original_exception),
            )
            self._set_team_attributes_on_span(
                span=exception_logging_span,
                team_id=user_api_key_dict.team_id,
                team_alias=user_api_key_dict.team_alias,
            )
            exception_logging_span.set_status(Status(StatusCode.ERROR))
            exception_logging_span.end(end_time=self._to_ns(datetime.now()))

            # Emit guardrail spans for any guardrail invocations that
            # ran during this request. _handle_failure typically does this,
            # but for pre-call guardrail blocks the standard_logging_object
            # may not carry guardrail_information by the time _handle_failure
            # fires (the data lives only in request_data["metadata"]). Pull
            # directly from request_data so the span is recorded either way;
            # _emit_once dedupes if _handle_failure already emitted it.
            self._emit_guardrail_spans_from_request_data(
                request_data=request_data,
                parent_span=parent_otel_span,
            )

            # End Parent OTEL Sspan
            parent_otel_span.end(end_time=self._to_ns(datetime.now()))

    def _emit_guardrail_spans_from_request_data(
        self,
        request_data: dict,
        parent_span: Optional[Any],
    ) -> None:
        """Emit ``guardrail`` spans from ``request_data["metadata"]
        ["standard_logging_guardrail_information"]``.

        Routed through ``_create_guardrail_span`` so the dedupe state in
        ``_otel_internal`` is honoured — if ``_handle_failure`` already
        emitted these spans for the same kwargs, this is a no-op.
        """
        from opentelemetry import trace as _trace

        metadata = (request_data or {}).get("metadata") or {}
        guardrail_information = metadata.get("standard_logging_guardrail_information")
        if not guardrail_information:
            return

        # _create_guardrail_span reads guardrail_information from
        # kwargs["standard_logging_object"] and shares its dedupe state via
        # kwargs["litellm_params"]["metadata"]["_otel_internal"]. Pass the
        # SAME metadata dict the proxy populated so _handle_failure and
        # this hook see the same dedupe markers.
        kwargs: Dict[str, Any] = {
            "litellm_params": {"metadata": metadata},
            "standard_logging_object": {
                "guardrail_information": guardrail_information,
                "metadata": metadata,
            },
        }
        context = _trace.set_span_in_context(parent_span) if parent_span is not None else None
        self._create_guardrail_span(kwargs=kwargs, context=context)

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: LLMResponseTypes,
    ):
        from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging

        litellm_logging_obj = data.get("litellm_logging_obj")

        if litellm_logging_obj is not None and isinstance(litellm_logging_obj, LiteLLMLogging):
            kwargs = litellm_logging_obj.model_call_details
            parent_span = user_api_key_dict.parent_otel_span

            ctx, _ = self._get_span_context(kwargs, default_span=parent_span)

            # Pre-request latency on the SERVER span (success path).
            self.set_preprocessing_duration_attribute(parent_span, kwargs)

            # 3. Guardrail span
            self._create_guardrail_span(kwargs=kwargs, context=ctx)

        return response

    #########################################################
    # Team/Key Based Logging Control Flow
    #########################################################
    def get_tracer_to_use_for_request(self, kwargs: dict) -> Tracer:
        """
        Get the tracer to use for this request

        If dynamic headers are present, a temporary tracer is created with the dynamic headers.
        Otherwise, the default tracer is used.

        Returns:
            Tracer: The tracer to use for this request
        """
        dynamic_config = self._get_dynamic_otel_config_from_kwargs(kwargs)
        if dynamic_config is not None:
            verbose_logger.debug(
                "[OTEL DEBUG] Using DYNAMIC config tracer with endpoint: %s",
                dynamic_config.endpoint,
            )
            return self._get_tracer_with_dynamic_config(dynamic_config)

        dynamic_headers = self._get_dynamic_otel_headers_from_kwargs(kwargs)

        if dynamic_headers is not None:
            # Create spans using a temporary tracer with dynamic headers
            tracer_to_use = self._get_tracer_with_dynamic_headers(dynamic_headers)
            verbose_logger.debug(
                "[OTEL DEBUG] Using DYNAMIC tracer with headers: %s", redact_string(str(dynamic_headers))
            )
        else:
            # For langfuse_otel without dynamic headers, create a provider with env var credentials
            if hasattr(self, "callback_name") and self.callback_name == "langfuse_otel":
                # Use the headers from config (which were set from env vars during init)
                env_var_headers = self._get_headers_dictionary(self.OTEL_HEADERS) if self.OTEL_HEADERS else {}
                if env_var_headers:
                    tracer_to_use = self._get_tracer_with_dynamic_headers(env_var_headers)
                    verbose_logger.debug(
                        "[OTEL DEBUG] Using env var credentials for langfuse_otel (master key request)"
                    )
                else:
                    # No env vars set, use global tracer (will be NoOp)
                    tracer_to_use = self.tracer
                    verbose_logger.debug("[OTEL DEBUG] No credentials available for langfuse_otel")
            else:
                tracer_to_use = self.tracer
                verbose_logger.debug("[OTEL DEBUG] Using GLOBAL tracer (no dynamic headers)")

        return tracer_to_use

    def _get_dynamic_otel_headers_from_kwargs(self, kwargs) -> Optional[dict]:
        """Extract dynamic headers from kwargs if available."""
        standard_callback_dynamic_params: Optional[StandardCallbackDynamicParams] = kwargs.get(
            "standard_callback_dynamic_params"
        )

        if not standard_callback_dynamic_params:
            return None

        dynamic_headers = self.construct_dynamic_otel_headers(
            standard_callback_dynamic_params=standard_callback_dynamic_params
        )

        return dynamic_headers if dynamic_headers else None

    def _get_dynamic_otel_config_from_kwargs(self, kwargs: dict) -> Optional[OpenTelemetryConfig]:
        """Extract a full dynamic exporter config from kwargs if available."""
        standard_callback_dynamic_params: Optional[StandardCallbackDynamicParams] = kwargs.get(
            "standard_callback_dynamic_params"
        )

        if not standard_callback_dynamic_params:
            return None

        return self.construct_dynamic_otel_config(standard_callback_dynamic_params=standard_callback_dynamic_params)

    def _get_tracer_with_dynamic_config(self, dynamic_config: OpenTelemetryConfig):
        """Create (or reuse) a tracer whose exporter target comes from a per-request config."""
        from opentelemetry.sdk.trace import TracerProvider

        cache_key = f"dynamic_config:{dynamic_config.exporter}:{dynamic_config.endpoint}:{dynamic_config.headers}"
        if cache_key in self._tracer_provider_cache:
            return self._tracer_provider_cache[cache_key].get_tracer(LITELLM_TRACER_NAME)

        temp_provider = TracerProvider(resource=self._get_litellm_resource(self.config))
        temp_provider.add_span_processor(self._get_span_processor(config_override=dynamic_config))

        self._tracer_provider_cache[cache_key] = temp_provider

        return temp_provider.get_tracer(LITELLM_TRACER_NAME)

    def _get_tracer_with_dynamic_headers(self, dynamic_headers: dict):
        """Create a temporary tracer with dynamic headers for this request only."""
        from opentelemetry.sdk.trace import TracerProvider

        # Prevents thread exhaustion by reusing providers for the same credential sets (e.g. per-team keys)
        cache_key = str(sorted(dynamic_headers.items()))
        if cache_key in self._tracer_provider_cache:
            return self._tracer_provider_cache[cache_key].get_tracer(LITELLM_TRACER_NAME)

        # Create a temporary tracer provider with dynamic headers
        temp_provider = TracerProvider(resource=self._get_litellm_resource(self.config))
        temp_provider.add_span_processor(self._get_span_processor(dynamic_headers=dynamic_headers))

        # Store in cache for reuse
        self._tracer_provider_cache[cache_key] = temp_provider

        return temp_provider.get_tracer(LITELLM_TRACER_NAME)

    def construct_dynamic_otel_headers(
        self, standard_callback_dynamic_params: StandardCallbackDynamicParams
    ) -> Optional[dict]:
        """
        Construct dynamic headers from standard callback dynamic params

        Note: You just need to override this method in Arize, Langfuse Otel if you want to allow team/key based logging.

        Returns:
            dict: A dictionary of dynamic headers
        """
        return None

    def construct_dynamic_otel_config(
        self, standard_callback_dynamic_params: StandardCallbackDynamicParams
    ) -> Optional[OpenTelemetryConfig]:
        """
        Construct a full exporter config from standard callback dynamic params.

        Override this when team/key dynamic params must control the export
        target (exporter kind + endpoint), not just the request headers. When
        this returns a config, it takes precedence over
        construct_dynamic_otel_headers for the request.
        """
        return None

    #########################################################
    # End of Team/Key Based Logging Control Flow
    #########################################################

    def _emit_once(self, kwargs: dict, *scope: object) -> bool:
        """Return True the first time this handler is asked to emit a span
        for the given (handler, scope) on this kwargs; False on repeats.

        Used to suppress duplicate span emission for two distinct patterns:

        1. **Handler-level dual-fire**: streaming code paths trigger both
           the sync and async callback for one request, so ``_handle_success``
           / ``_handle_failure`` would otherwise produce two
           ``litellm_request`` spans. Scope: ``("success",)`` / ``("failure",)``.
        2. **Payload-driven multi-entrypoint emission**: a span loop that
           reads entries from ``standard_logging_payload`` (currently only
           guardrails) is invoked from multiple lifecycle points
           (post-call hooks, success callback, failure callback). The list
           can be re-read with mutated entries between calls, so dedupe
           must be at entry granularity. Scope: the entry's stable identity.

        ``scope`` parts may include unhashable containers (list, dict, set);
        they are normalized into a hashable shape via ``_freeze_for_dedupe``
        before keying the marker dict. The marker is stored in
        ``kwargs["litellm_params"]["metadata"]["_otel_internal"]`` so it is
        request-local (kwargs is shared across the sync/async callbacks and
        lifecycle hooks for one request).
        """
        litellm_params = kwargs.get("litellm_params")
        if not isinstance(litellm_params, dict):
            litellm_params = {}
            kwargs["litellm_params"] = litellm_params

        _metadata = litellm_params.get("metadata")
        if not isinstance(_metadata, dict):
            _metadata = {}
            litellm_params["metadata"] = _metadata

        _otel_internal = _metadata.get("_otel_internal")
        if not isinstance(_otel_internal, dict):
            _otel_internal = {}
            _metadata["_otel_internal"] = _otel_internal

        spans_logged = _otel_internal.get("spans_logged")
        if not isinstance(spans_logged, dict):
            spans_logged = {}
            _otel_internal["spans_logged"] = spans_logged

        dedupe_key = (
            self.__class__.__name__,
            id(self),
            *(_freeze_for_dedupe(part) for part in scope),
        )
        if spans_logged.get(dedupe_key) is True:
            return False

        spans_logged[dedupe_key] = True
        return True

    def _end_proxy_span_from_kwargs(self, kwargs: dict, end_time) -> None:
        """Close the proxy-level parent span if it is still recording.

        This helper retrieves the proxy span directly from kwargs metadata
        and closes it after all child spans have been recorded.

        Only called from the success path. The failure path deliberately
        leaves the proxy span open so ``async_post_call_failure_hook`` can
        append the ``"Failed Proxy Server Request"`` child span before
        closing it.

        Only spans named ``LITELLM_PROXY_REQUEST_SPAN_NAME`` are closed —
        externally provided spans must not be closed by LiteLLM.
        """
        litellm_params = kwargs.get("litellm_params", {}) or {}
        _metadata = litellm_params.get("metadata", {}) or {}
        proxy_span = _metadata.get("litellm_parent_otel_span", None)

        # Fallback: check litellm_metadata (used by /v1/messages and other
        # LITELLM_METADATA_ROUTES).
        if proxy_span is None:
            _litellm_metadata = litellm_params.get("litellm_metadata", {}) or {}
            proxy_span = _litellm_metadata.get("litellm_parent_otel_span", None)

        if (
            proxy_span is not None
            and getattr(proxy_span, "name", None) == LITELLM_PROXY_REQUEST_SPAN_NAME
            and hasattr(proxy_span, "is_recording")
            and proxy_span.is_recording()
        ):
            self._close_proxy_span_ok(proxy_span, end_time)

    def _close_proxy_span_ok(self, span: Span, end_time) -> None:
        """Stamp http.response.status_code=200 + status=OK, then end the span."""
        from opentelemetry.trace import Status, StatusCode

        self.set_response_status_code_attribute(span, 200)
        span.set_status(Status(StatusCode.OK))
        span.end(end_time=self._to_ns(end_time))

    def _handle_success(self, kwargs, response_obj, start_time, end_time):
        """Create the litellm_request span then close the proxy span."""
        verbose_logger.debug(
            "OpenTelemetry Logger: Logging kwargs: %s, OTEL config settings=%s",
            kwargs,
            self.config,
        )

        # sync + async success handlers can both fire for one
        # request (notably in streaming code paths). Guard against duplicate
        # span writes — but still close the proxy span on the skip path so
        # the trace doesn't leak an open root span.
        if not self._emit_once(kwargs, "success"):
            verbose_logger.debug(
                "OpenTelemetry: skipping duplicate success span for handler=%s",
                self.__class__.__name__,
            )
            self._end_proxy_span_from_kwargs(kwargs, end_time)
            return

        ctx, parent_span = self._get_span_context(kwargs)

        if self.config.ignore_context_propagation:
            parent_span = None  # Ignore parent spans from other providers
            ctx = None

        # Decide whether to create a primary span
        # Always create if no parent span exists (backward compatibility)
        # OR if USE_OTEL_LITELLM_REQUEST_SPAN is explicitly enabled
        should_create_primary_span = parent_span is None or get_secret_bool("USE_OTEL_LITELLM_REQUEST_SPAN")

        if should_create_primary_span:
            # Create a new litellm_request span
            span = self._start_primary_span(kwargs, response_obj, start_time, end_time, ctx)
            # Raw-request sub-span (if enabled) - child of litellm_request span
            self._maybe_log_raw_request(kwargs, response_obj, start_time, end_time, span)
            # Do NOT duplicate attributes onto the parent proxy-request span.
            # The child litellm_request span already carries all attributes;
            # copying them to the parent doubles storage and complicates
            # search (Issue #4).
        else:
            # Do not create primary span (keep hierarchy shallow when parent exists)
            from opentelemetry.trace import Status, StatusCode

            span = None
            # Only set attributes if the span is still recording (not closed)
            # Note: parent_span is guaranteed to be not None here
            if hasattr(parent_span, "set_status"):
                parent_span.set_status(Status(StatusCode.OK))
                self.set_attributes(parent_span, kwargs, response_obj)
            # Raw-request as direct child of parent_span
            self._maybe_log_raw_request(kwargs, response_obj, start_time, end_time, parent_span)

        # 3. Guardrail span — ensure guardrails are always parented to an
        #    existing span so they never become orphaned root spans (Issue #5).
        guardrail_ctx = self._resolve_guardrail_context(span=span, parent_span=parent_span, fallback_ctx=ctx)
        self._create_guardrail_span(kwargs=kwargs, context=guardrail_ctx)

        # 4. Metrics & cost recording
        self._record_metrics(kwargs, response_obj, start_time, end_time)

        # 5. Semantic logs.
        if self.config.enable_events:
            log_span = span if span is not None else parent_span
            if log_span is not None:
                self._emit_semantic_logs(kwargs, response_obj, log_span)

        # 6. Do NOT end parent span - it should be managed by its creator
        # External spans (from Langfuse, user code, HTTP headers, global context) must not be closed by LiteLLM
        # However, proxy-created spans should be closed here.
        if (
            parent_span is not None
            and hasattr(parent_span, "name")
            and parent_span.name == LITELLM_PROXY_REQUEST_SPAN_NAME
            and hasattr(parent_span, "is_recording")
            and parent_span.is_recording()
        ):
            self._close_proxy_span_ok(parent_span, end_time)

        # Stamp team attributes onto the SERVER (root) span before it is
        # closed, so the trace root carries them like every child span.
        self._set_team_attributes_on_proxy_span_from_kwargs(kwargs)

        # close the proxy span explicitly from kwargs metadata
        # after all child spans (litellm_request, guardrail, raw_request)
        # have been fully recorded and exported.
        self._end_proxy_span_from_kwargs(kwargs, end_time)

    def _start_primary_span(
        self,
        kwargs,
        response_obj,
        start_time,
        end_time,
        context,
    ):
        from opentelemetry.trace import Status, StatusCode

        otel_tracer: Tracer = self.get_tracer_to_use_for_request(kwargs)

        span_kwargs: Dict[str, Any] = {
            "name": self._get_span_name(kwargs),
            "start_time": self._to_ns(start_time),
            "context": context,
        }
        if self._gen_ai_semconv_latest_experimental:
            span_kwargs["kind"] = self.span_kind.CLIENT
        span = otel_tracer.start_span(**span_kwargs)

        span.set_status(Status(StatusCode.OK))
        self.set_attributes(span, kwargs, response_obj)
        span.end(end_time=self._to_ns(end_time))
        return span

    def _maybe_log_raw_request(self, kwargs, response_obj, start_time, end_time, parent_span):
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode

        # raw_gen_ai_request is non-standard in semconv mode.
        if self._gen_ai_semconv_latest_experimental:
            return

        if not self._capture_in_span():
            return

        litellm_params = kwargs.get("litellm_params", {})
        metadata = litellm_params.get("metadata") or {}
        generation_name = metadata.get("generation_name")

        raw_span_name = generation_name if generation_name else RAW_REQUEST_SPAN_NAME

        otel_tracer: Tracer = self.get_tracer_to_use_for_request(kwargs)
        raw_span = otel_tracer.start_span(
            name=raw_span_name,
            start_time=self._to_ns(start_time),
            context=trace.set_span_in_context(parent_span),
        )
        raw_span.set_status(Status(StatusCode.OK))
        self.set_raw_request_attributes(raw_span, kwargs, response_obj)
        self._set_team_attributes_from_kwargs(raw_span, kwargs)
        raw_span.end(end_time=self._to_ns(end_time))

    def _set_team_attributes_on_span(
        self,
        span: Span,
        team_id: Optional[str],
        team_alias: Optional[str],
    ) -> None:
        """Stamp team_id / team_alias onto a span so every child span of a
        litellm_request trace carries them, not just the root span.

        Empty strings are treated as absent: a request made with the master
        key or a team-less virtual key carries ``user_api_key_team_id=""``
        in ``standard_logging_object.metadata``; propagating that to every
        span only adds noise that makes traces look mis-instrumented.
        """
        if team_id:
            self.safe_set_attribute(
                span=span,
                key="metadata.user_api_key_team_id",
                value=team_id,
            )
        if team_alias:
            self.safe_set_attribute(
                span=span,
                key="metadata.user_api_key_team_alias",
                value=team_alias,
            )

    def _set_team_attributes_from_kwargs(self, span: Span, kwargs: dict) -> None:
        """Pull team_id / team_alias from the standard logging metadata in kwargs and stamp them onto span."""
        std_log = kwargs.get("standard_logging_object")
        md: dict = {}
        if isinstance(std_log, dict):
            md = std_log.get("metadata") or {}
        elif std_log is not None:
            md = getattr(std_log, "metadata", None) or {}
        self._set_team_attributes_on_span(
            span=span,
            team_id=md.get("user_api_key_team_id"),
            team_alias=md.get("user_api_key_team_alias"),
        )

    def _set_team_attributes_on_proxy_span_from_kwargs(self, kwargs: dict) -> None:
        """Stamp team attributes onto the proxy SERVER (root) span so the
        trace root is filterable by team, not just its children. The root
        span is created in auth before the team is resolved and is
        otherwise only closed (never re-attributed) on the success path.

        Guarded to the LiteLLM-created proxy span (by name + recording) so
        externally provided parent spans are never mutated.
        """
        litellm_params = kwargs.get("litellm_params") or {}
        metadata = litellm_params.get("metadata") or {}
        proxy_span = metadata.get("litellm_parent_otel_span")
        if (
            proxy_span is not None
            and getattr(proxy_span, "name", None) == LITELLM_PROXY_REQUEST_SPAN_NAME
            and hasattr(proxy_span, "is_recording")
            and proxy_span.is_recording()
        ):
            self._set_team_attributes_from_kwargs(proxy_span, kwargs)

    def _set_inference_identity_attributes(
        self,
        span: Span,
        standard_logging_payload: StandardLoggingPayload,
        litellm_params: dict,
    ) -> None:
        """Stamp request-identity attributes onto an inference span so every
        LLM-call span is filterable by the route it came in on, the team's
        metadata, and both the user-facing (model_group alias) and the
        dispatched (provider) model names. Empty/absent values are skipped.
        """
        metadata = standard_logging_payload.get("metadata") or {}

        http_route = metadata.get("user_api_key_request_route")
        if http_route:
            self.safe_set_attribute(span=span, key=HTTP_ROUTE_ATTRIBUTE, value=http_route)

        # ``user_api_key_team_metadata`` is dropped from the standard logging
        # payload metadata, so read it from the raw request metadata in kwargs.
        # ``metadata`` and ``litellm_metadata`` are alternate names for the same
        # full metadata dict (the name varies by endpoint), so first-truthy wins.
        raw_metadata = litellm_params.get("metadata") or litellm_params.get("litellm_metadata") or {}
        team_metadata = self._team_metadata_json(
            raw_metadata.get("user_api_key_team_metadata"),
            self.config.baggage_team_metadata_keys,
        )
        if team_metadata:
            self.safe_set_attribute(span=span, key=TEAM_METADATA_ATTRIBUTE, value=team_metadata)

        model_group = standard_logging_payload.get("model_group")
        if model_group:
            self.safe_set_attribute(span=span, key=MODEL_GROUP_ATTRIBUTE, value=model_group)

        hidden_params = standard_logging_payload.get("hidden_params") or {}
        provider_model = hidden_params.get("litellm_model_name") or standard_logging_payload.get("model")
        if provider_model:
            self.safe_set_attribute(span=span, key=PROVIDER_MODEL_ATTRIBUTE, value=provider_model)

    @staticmethod
    def _team_metadata_json(value: Any, allowed_keys: List[str]) -> Optional[str]:
        """JSON-serialize only the allowlisted sub-keys of a team's metadata.

        Returns ``None`` when nothing is allowlisted or no allowlisted key is
        present, so the empty case is dropped rather than stamping a useless
        ``"{}"`` (and so a team's metadata never leaves the process until an
        operator opts each sub-key in via ``baggage_team_metadata_keys``).
        """
        if not isinstance(value, dict) or not value or not allowed_keys:
            return None
        filtered = {key: value[key] for key in allowed_keys if key in value}
        if not filtered:
            return None
        return safe_dumps(filtered)

    def _ensure_metric_attribute_filter(self) -> None:
        """Resolve the include/exclude filter once, falling back to the proxy's
        callback_settings.otel.attributes when no explicit config was passed."""
        if self._metric_attr_filter_resolved:
            return
        attributes = self.config.attributes
        if attributes is None and self.callback_name in (None, "otel"):
            otel_settings = (litellm.callback_settings or {}).get("otel") or {}
            raw = otel_settings.get("attributes") if isinstance(otel_settings, dict) else None
            if raw is not None:
                attributes = _build_metric_attribute_filter(raw)
        (
            self._metric_attr_include,
            self._metric_attr_exclude,
        ) = _resolve_metric_attribute_filter(attributes)
        self._metric_attr_filter_resolved = True

    def _filter_metric_attributes(self, attrs: Dict[str, Any]) -> Dict[str, Any]:
        if not self._metric_attr_filter_resolved:
            self._ensure_metric_attribute_filter()
        if self._metric_attr_include is not None:
            return {k: v for k, v in attrs.items() if k in self._metric_attr_include}
        if self._metric_attr_exclude is not None:
            return {k: v for k, v in attrs.items() if k not in self._metric_attr_exclude}
        return attrs

    def _record_metrics(self, kwargs, response_obj, start_time, end_time):
        duration_s = (end_time - start_time).total_seconds()
        params = kwargs.get("litellm_params") or {}
        provider = params.get("custom_llm_provider", "Unknown")

        common_attrs = {
            "gen_ai.operation.name": (
                self._gen_ai_operation_name(kwargs) if self._gen_ai_semconv_latest_experimental else "chat"
            ),
            "gen_ai.system": provider,
            "gen_ai.request.model": kwargs.get("model"),
            "gen_ai.framework": "litellm",
        }

        std_log = kwargs.get("standard_logging_object")
        md = getattr(std_log, "metadata", None) or (std_log or {}).get("metadata", {})
        for key in METRIC_METADATA_KEYS:
            value = md.get(key)
            if value is None:
                continue
            if isinstance(value, (dict, list)):
                common_attrs[f"metadata.{key}"] = safe_dumps(value)
            else:
                common_attrs[f"metadata.{key}"] = str(value)

        # get hidden params
        hidden_params = getattr(std_log, "hidden_params", None) or (std_log or {}).get("hidden_params", {})
        if hidden_params:
            common_attrs["hidden_params"] = safe_dumps(hidden_params)

        common_attrs = self._filter_metric_attributes(common_attrs)

        if self._operation_duration_histogram:
            self._operation_duration_histogram.record(duration_s, attributes=common_attrs)
            if response_obj and (usage := response_obj.get("usage")) and self._token_usage_histogram:
                in_attrs = {**common_attrs, TOKEN_TYPE_ATTRIBUTE: "input"}
                out_attrs = {**common_attrs, TOKEN_TYPE_ATTRIBUTE: "output"}
                self._token_usage_histogram.record(usage.get("prompt_tokens", 0), attributes=in_attrs)
                self._token_usage_histogram.record(usage.get("completion_tokens", 0), attributes=out_attrs)

        cost = kwargs.get("response_cost")
        if self._cost_histogram and cost:
            self._cost_histogram.record(cost, attributes=common_attrs)

        # Record latency metrics (TTFT, TPOT, and Total Generation Time)
        self._record_time_to_first_token_metric(kwargs, common_attrs)
        self._record_time_per_output_token_metric(kwargs, response_obj, end_time, duration_s, common_attrs)
        self._record_response_duration_metric(kwargs, end_time, common_attrs)

    @staticmethod
    def _to_timestamp(
        val: Optional[Union[datetime, float, str]],
    ) -> Optional[float]:
        """Convert datetime/float/string to timestamp."""
        if val is None:
            return None
        if isinstance(val, datetime):
            return val.timestamp()
        if isinstance(val, (int, float)):
            return float(val)
        # isinstance(val, str) - parse datetime string (with or without microseconds)
        try:
            return datetime.strptime(val, "%Y-%m-%d %H:%M:%S.%f").timestamp()
        except ValueError:
            try:
                return datetime.strptime(val, "%Y-%m-%d %H:%M:%S").timestamp()
            except ValueError:
                return None

    def _record_time_to_first_token_metric(self, kwargs: dict, common_attrs: dict):
        """Record Time to First Token (TTFT) metric for streaming requests."""
        optional_params = kwargs.get("optional_params", {})
        is_streaming = optional_params.get("stream", False)

        if not (self._time_to_first_token_histogram and is_streaming):
            return

        # Use api_call_start_time for precision (matches Prometheus implementation)
        # This excludes LiteLLM overhead and measures pure LLM API latency
        api_call_start_time = kwargs.get("api_call_start_time", None)
        completion_start_time = kwargs.get("completion_start_time", None)

        if api_call_start_time is not None and completion_start_time is not None:
            # Convert to timestamps if needed (handles datetime, float, and string)
            api_call_start_ts = self._to_timestamp(api_call_start_time)
            completion_start_ts = self._to_timestamp(completion_start_time)

            if api_call_start_ts is None or completion_start_ts is None:
                return  # Skip recording if conversion failed

            time_to_first_token_seconds = completion_start_ts - api_call_start_ts
            self._time_to_first_token_histogram.record(time_to_first_token_seconds, attributes=common_attrs)

    def _record_time_per_output_token_metric(
        self,
        kwargs: dict,
        response_obj: Optional[Any],
        end_time: datetime,
        duration_s: float,
        common_attrs: dict,
    ):
        """Record Time Per Output Token (TPOT) metric.

        Calculated as: generation_time / completion_tokens
        - For streaming: uses end_time - completion_start_time (time to generate all tokens after first)
        - For non-streaming: uses end_time - api_call_start_time (total generation time)
        """
        if not self._time_per_output_token_histogram:
            return

        # Get completion tokens from response_obj
        completion_tokens = None
        if response_obj and (usage := response_obj.get("usage")):
            completion_tokens = usage.get("completion_tokens")

        if completion_tokens is None or completion_tokens <= 0:
            return

        # Calculate generation time
        completion_start_time = kwargs.get("completion_start_time", None)
        api_call_start_time = kwargs.get("api_call_start_time", None)

        # Convert end_time to timestamp (handles datetime, float, and string)
        end_time_ts = self._to_timestamp(end_time)
        if end_time_ts is None:
            # Fallback to duration_s if conversion failed
            generation_time_seconds = duration_s
            if generation_time_seconds > 0:
                time_per_output_token_seconds = generation_time_seconds / completion_tokens
                self._time_per_output_token_histogram.record(time_per_output_token_seconds, attributes=common_attrs)
            return

        if completion_start_time is not None:
            # Streaming: use completion_start_time (when first token arrived)
            # This measures time to generate all tokens after the first one
            completion_start_ts = self._to_timestamp(completion_start_time)
            if completion_start_ts is None:
                # Fallback to duration_s if conversion failed
                generation_time_seconds = duration_s
            else:
                generation_time_seconds = end_time_ts - completion_start_ts
        elif api_call_start_time is not None:
            # Non-streaming: use api_call_start_time (total generation time)
            api_call_start_ts = self._to_timestamp(api_call_start_time)
            if api_call_start_ts is None:
                # Fallback to duration_s if conversion failed
                generation_time_seconds = duration_s
            else:
                generation_time_seconds = end_time_ts - api_call_start_ts
        else:
            # Fallback: use duration_s (already calculated as (end_time - start_time).total_seconds())
            generation_time_seconds = duration_s

        if generation_time_seconds > 0:
            time_per_output_token_seconds = generation_time_seconds / completion_tokens
            self._time_per_output_token_histogram.record(time_per_output_token_seconds, attributes=common_attrs)

    def _record_response_duration_metric(
        self,
        kwargs: dict,
        end_time: Union[datetime, float],
        common_attrs: dict,
    ):
        """Record Total Generation Time (response duration) metric.

        Measures pure LLM API generation time: end_time - api_call_start_time
        This excludes LiteLLM overhead and measures only the LLM provider's response time.
        Works for both streaming and non-streaming requests.

        Mirrors Prometheus's litellm_llm_api_latency_metric.
        Uses kwargs.get("end_time") with fallback to parameter for consistency with Prometheus.
        """
        if not self._response_duration_histogram:
            return

        api_call_start_time = kwargs.get("api_call_start_time", None)
        if api_call_start_time is None:
            return

        # Use end_time from kwargs if available (matches Prometheus), otherwise use parameter
        # For streaming: end_time is when the stream completes (final chunk received)
        # For non-streaming: end_time is when the response is received
        _end_time = kwargs.get("end_time") or end_time
        if _end_time is None:
            _end_time = datetime.now()

        # Convert to timestamps if needed (handles datetime, float, and string)
        api_call_start_ts = self._to_timestamp(api_call_start_time)
        end_time_ts = self._to_timestamp(_end_time)

        if api_call_start_ts is None or end_time_ts is None:
            return  # Skip recording if conversion failed

        response_duration_seconds = end_time_ts - api_call_start_ts

        if response_duration_seconds > 0:
            self._response_duration_histogram.record(response_duration_seconds, attributes=common_attrs)

    @staticmethod
    def _otel_log_types():
        """Resolve ``(LogRecord, SeverityNumber)`` across OTEL SDK versions.

        ``LogRecord`` moved out of ``opentelemetry.sdk._logs`` in OTEL >= 1.39.0
        (open-telemetry/opentelemetry-python#4676). Imports stay function-local
        because the SDK is an optional dependency.
        """
        from opentelemetry._logs import SeverityNumber

        try:
            from opentelemetry.sdk._logs import LogRecord  # OTEL < 1.39.0
        except ImportError:
            from opentelemetry.sdk._logs._internal import (  # OTEL >= 1.39.0
                LogRecord,
            )
        return LogRecord, SeverityNumber

    def _emit_semantic_logs(self, kwargs, response_obj, span: Span):
        if not self.config.enable_events:
            return

        # NOTE: Semantic logs (gen_ai.content.prompt/completion events) have compatibility issues
        # with OTEL SDK >= 1.39.0 due to breaking changes in PR #4676:
        # - LogRecord moved from opentelemetry.sdk._logs to opentelemetry.sdk._logs._internal
        # - LogRecord constructor no longer accepts 'resource' parameter (now inherited from LoggerProvider)
        # - LogData class was removed entirely
        # These logs work correctly in OTEL SDK < 1.39.0 but may fail in >= 1.39.0.
        # See: https://github.com/open-telemetry/opentelemetry-python/pull/4676
        # TODO: Refactor to use the proper OTEL Logs API instead of directly creating SDK LogRecords

        SdkLogRecord, SeverityNumber = self._otel_log_types()

        # Resolve through the handler's own LoggerProvider (which may be a
        # private one when skip_set_global=True) rather than the module-level
        # get_logger() which always goes through the global provider.
        otel_logger = self._logger_provider.get_logger(LITELLM_LOGGER_NAME)

        parent_ctx = span.get_span_context()
        provider = (kwargs.get("litellm_params") or {}).get("custom_llm_provider", "Unknown")

        if self._gen_ai_semconv_latest_experimental:
            self._emit_inference_details_event(
                kwargs=kwargs,
                response_obj=response_obj,
                provider=provider,
                otel_logger=otel_logger,
                parent_ctx=parent_ctx,
            )
            return

        # per-message events
        for msg in kwargs.get("messages", []):
            role = msg.get("role", "user")
            attrs = {
                "event_name": "gen_ai.content.prompt",
                "gen_ai.system": provider,
            }
            if role == "tool" and msg.get("id"):
                attrs["id"] = msg["id"]
            capture_event_content = self._capture_in_event()
            if capture_event_content and msg.get("content"):
                attrs["gen_ai.prompt"] = msg["content"]

            body = msg.copy()
            if not capture_event_content:
                body.pop("content", None)

            log_record = SdkLogRecord(
                timestamp=self._to_ns(datetime.now()),
                trace_id=parent_ctx.trace_id,
                span_id=parent_ctx.span_id,
                trace_flags=parent_ctx.trace_flags,
                severity_number=SeverityNumber.INFO,
                severity_text="INFO",
                body=body,
                attributes=attrs,
            )
            otel_logger.emit(log_record)

        # per-choice events
        for idx, choice in enumerate(response_obj.get("choices", [])):
            attrs = {
                "event_name": "gen_ai.content.completion",
                "gen_ai.system": provider,
                "index": idx,
                "finish_reason": choice.get("finish_reason"),
            }
            body_msg = choice.get("message", {})
            capture_event_content = self._capture_in_event()
            if capture_event_content and body_msg.get("content"):
                attrs["message.content"] = body_msg["content"]
            body = {
                "index": idx,
                "finish_reason": choice.get("finish_reason"),
                "message": {"role": body_msg.get("role", "assistant")},
            }
            if capture_event_content and body_msg.get("content"):
                body["message"]["content"] = body_msg["content"]

            log_record = SdkLogRecord(
                timestamp=self._to_ns(datetime.now()),
                trace_id=parent_ctx.trace_id,
                span_id=parent_ctx.span_id,
                trace_flags=parent_ctx.trace_flags,
                severity_number=SeverityNumber.INFO,
                severity_text="INFO",
                body=body,
                attributes=attrs,
            )
            otel_logger.emit(log_record)

    @staticmethod
    def _resolve_guardrail_context(
        span: Optional[Any],
        parent_span: Optional[Any],
        fallback_ctx: Optional[Any],
    ) -> Optional[Any]:
        """
        Return a valid OTEL context for guardrail child spans so they are
        never orphaned (Issue #5).  Priority:
          1. The litellm_request span that was just created
          2. The parent proxy-request span
          3. The original fallback context (may be None — last resort)
        """
        from opentelemetry import trace as _trace

        if span is not None:
            return _trace.set_span_in_context(span)
        if parent_span is not None:
            return _trace.set_span_in_context(parent_span)
        return fallback_ctx

    def _create_guardrail_span(self, kwargs: Optional[dict], context: Optional[Context]):
        """
        Creates a span for Guardrail, if any guardrail information is present in standard_logging_object
        """
        # Create span for guardrail information
        kwargs = kwargs or {}
        standard_logging_payload: Optional[StandardLoggingPayload] = kwargs.get("standard_logging_object")
        if standard_logging_payload is None:
            return

        guardrail_information_data = standard_logging_payload.get("guardrail_information")

        if not guardrail_information_data:
            return

        guardrail_information_list = [
            information for information in guardrail_information_data if isinstance(information, dict)
        ]

        if not guardrail_information_list:
            return

        otel_tracer: Tracer = self.get_tracer_to_use_for_request(kwargs)
        for guardrail_information in guardrail_information_list:
            start_time_float = guardrail_information.get("start_time")
            end_time_float = guardrail_information.get("end_time")

            # ``_create_guardrail_span`` is called from three lifecycle
            # points (``async_post_call_success_hook``, ``_handle_success``,
            # ``_handle_failure``) and re-reads the (mutating) entry list
            # each time. Dedupe at entry granularity so a single real
            # guardrail invocation produces exactly one span per handler.
            if not self._emit_once(
                kwargs,
                "guardrail",
                guardrail_information.get("guardrail_name"),
                start_time_float,
                guardrail_information.get("guardrail_mode"),
            ):
                continue

            start_time_datetime = datetime.now()
            if start_time_float is not None:
                start_time_datetime = datetime.fromtimestamp(start_time_float)
            end_time_datetime = datetime.now()
            if end_time_float is not None:
                end_time_datetime = datetime.fromtimestamp(end_time_float)

            guardrail_span = otel_tracer.start_span(
                name="guardrail",
                start_time=self._to_ns(start_time_datetime),
                context=context,
            )

            self.safe_set_attribute(
                span=guardrail_span,
                key=SpanAttributes.OPENINFERENCE_SPAN_KIND,
                value=OpenInferenceSpanKindValues.GUARDRAIL.value,
            )

            self.safe_set_attribute(
                span=guardrail_span,
                key="guardrail_name",
                value=guardrail_information.get("guardrail_name"),
            )

            self.safe_set_attribute(
                span=guardrail_span,
                key="guardrail_mode",
                value=guardrail_information.get("guardrail_mode"),
            )

            masked_entity_count = guardrail_information.get("masked_entity_count")
            if masked_entity_count is not None:
                guardrail_span.set_attribute("masked_entity_count", safe_dumps(masked_entity_count))

            guardrail_response = guardrail_information.get("guardrail_response")
            if guardrail_response is not None:
                guardrail_span.set_attribute("guardrail_response", safe_dumps(guardrail_response))

            # Surface guardrail_status (success / guardrail_intervened /
            # guardrail_failed_to_respond / not_run) as a top-level span
            # attribute so trace backends can filter on it without parsing
            # guardrail_response.
            self.safe_set_attribute(
                span=guardrail_span,
                key="guardrail_status",
                value=guardrail_information.get("guardrail_status"),
            )

            # Provider's raw top-level action (e.g. Bedrock's
            # ``GUARDRAIL_INTERVENED`` / ``NONE``). Populated by the provider
            # hook onto StandardLoggingGuardrailInformation so this integration
            # stays provider-agnostic — we only read a normalised string.
            guardrail_action = guardrail_information.get("guardrail_action")
            if guardrail_action:
                guardrail_span.set_attribute("guardrail_action", guardrail_action)

            # The provider hook (e.g. Bedrock) extracts violation_categories
            # from the raw response BEFORE redaction and stamps them onto
            # StandardLoggingGuardrailInformation. Surfacing them here as a
            # queryable attribute lets dashboards group by violation category
            # without parsing the redacted guardrail_response blob.
            violation_categories = guardrail_information.get("violation_categories")
            if violation_categories:
                # OTel sequence attributes must be homogeneous primitives;
                # serialise to JSON once so set_attribute never coerces.
                guardrail_span.set_attribute("guardrail_violation_categories", safe_dumps(violation_categories))

            self._set_team_attributes_from_kwargs(guardrail_span, kwargs)

            guardrail_span.end(end_time=self._to_ns(end_time_datetime))

    def _handle_failure(self, kwargs, response_obj, start_time, end_time):
        from opentelemetry.trace import Status, StatusCode

        verbose_logger.debug(
            "OpenTelemetry Logger: Failure HandlerLogging kwargs: %s, OTEL config settings=%s",
            kwargs,
            self.config,
        )

        # sync + async failure handlers can both fire for one
        # request (notably in streaming code paths), producing two
        # semantically identical ERROR spans. Unlike the success path, the
        # proxy span is intentionally left open here so that
        # ``async_post_call_failure_hook`` can append the
        # "Failed Proxy Server Request" child span before closing it —
        # there is no proxy-span side-effect to preserve on the skip path.
        if not self._emit_once(kwargs, "failure"):
            verbose_logger.debug(
                "OpenTelemetry: skipping duplicate failure span for handler=%s",
                self.__class__.__name__,
            )
            return

        _parent_context, parent_otel_span = self._get_span_context(kwargs)

        if self.config.ignore_context_propagation:
            parent_otel_span = None  # Ignore parent spans from other providers
            _parent_context = None

        # Decide whether to create a primary span
        # Always create if no parent span exists (backward compatibility)
        # OR if USE_OTEL_LITELLM_REQUEST_SPAN is explicitly enabled
        should_create_primary_span = parent_otel_span is None or get_secret_bool("USE_OTEL_LITELLM_REQUEST_SPAN")

        span = None
        if should_create_primary_span:
            # Span 1: Request sent to litellm SDK
            otel_tracer: Tracer = self.get_tracer_to_use_for_request(kwargs)
            span_kwargs: Dict[str, Any] = {
                "name": self._get_span_name(kwargs),
                "start_time": self._to_ns(start_time),
                "context": _parent_context,
            }
            if self._gen_ai_semconv_latest_experimental:
                span_kwargs["kind"] = self.span_kind.CLIENT
            span = otel_tracer.start_span(**span_kwargs)
            span.set_status(Status(StatusCode.ERROR))
            self.set_attributes(span, kwargs, response_obj)

            # Record exception information using OTEL standard method
            self._record_exception_on_span(span=span, kwargs=kwargs)

            span.end(end_time=self._to_ns(end_time))
        else:
            # When parent span exists and USE_OTEL_LITELLM_REQUEST_SPAN=false,
            # record error on parent span (keeps hierarchy shallow)
            # Only set attributes if the span is still recording (not closed)
            # Note: parent_otel_span is guaranteed to be not None here
            if parent_otel_span.is_recording():
                parent_otel_span.set_status(Status(StatusCode.ERROR))
                self.set_attributes(parent_otel_span, kwargs, response_obj)
                self._record_exception_on_span(span=parent_otel_span, kwargs=kwargs)

        # Create span for guardrail information — ensure proper parenting (Issue #5)
        guardrail_ctx = self._resolve_guardrail_context(
            span=span, parent_span=parent_otel_span, fallback_ctx=_parent_context
        )
        self._create_guardrail_span(kwargs=kwargs, context=guardrail_ctx)

        # Do NOT end parent span - it should be managed by its creator
        # External spans (from Langfuse, user code, HTTP headers, global context) must not be closed by LiteLLM
        # However, proxy-created spans should be closed here
        if (
            parent_otel_span is not None
            and hasattr(parent_otel_span, "name")
            and parent_otel_span.name == LITELLM_PROXY_REQUEST_SPAN_NAME
        ):
            parent_otel_span.end(end_time=self._to_ns(end_time))

    def _record_exception_on_span(self, span: Span, kwargs: dict):
        """
        Record exception information on the span using OTEL standard methods.

        This extracts error information from StandardLoggingPayload and:
        1. Uses span.record_exception() for the actual exception object (OTEL standard)
        2. Sets structured error attributes from StandardLoggingPayloadErrorInformation
        """
        try:
            from litellm.integrations._types.open_inference import (
                ErrorAttributes,
            )

            # Get the exception object if available
            exception = kwargs.get("exception")

            # Record the exception using OTEL's standard method
            if exception is not None:
                span.record_exception(exception)

            # Get StandardLoggingPayload for structured error information
            standard_logging_payload: Optional[StandardLoggingPayload] = kwargs.get("standard_logging_object")

            if standard_logging_payload is None:
                return

            # Extract error_information from StandardLoggingPayload
            error_information = standard_logging_payload.get("error_information")

            if error_information is None:
                # Fallback to error_str if error_information is not available
                error_str = standard_logging_payload.get("error_str")
                if error_str:
                    self.safe_set_attribute(
                        span=span,
                        key=ErrorAttributes.ERROR_MESSAGE,
                        value=error_str,
                    )
                return

            # Set structured error attributes from StandardLoggingPayloadErrorInformation
            if error_information.get("error_code"):
                self.safe_set_attribute(
                    span=span,
                    key=ErrorAttributes.ERROR_CODE,
                    value=error_information["error_code"],
                )

                # Also expose under the OTel-standard name as an int
                # (error_code is a str, may be non-numeric).
                _error_code_val = error_information["error_code"]
                if _error_code_val is not None:
                    try:
                        self.safe_set_attribute(
                            span=span,
                            key=HTTP_RESPONSE_STATUS_CODE_ATTRIBUTE,
                            value=int(_error_code_val),
                        )
                    except (ValueError, TypeError):
                        pass

            if error_information.get("error_class"):
                self.safe_set_attribute(
                    span=span,
                    key=ErrorAttributes.ERROR_TYPE,
                    value=error_information["error_class"],
                )

            if error_information.get("error_message"):
                self.safe_set_attribute(
                    span=span,
                    key=ErrorAttributes.ERROR_MESSAGE,
                    value=error_information["error_message"],
                )

            if error_information.get("llm_provider"):
                self.safe_set_attribute(
                    span=span,
                    key=ErrorAttributes.ERROR_LLM_PROVIDER,
                    value=error_information["llm_provider"],
                )

            if error_information.get("traceback"):
                self.safe_set_attribute(
                    span=span,
                    key=ErrorAttributes.ERROR_STACK_TRACE,
                    value=error_information["traceback"],
                )

        except Exception as e:
            verbose_logger.exception("OpenTelemetry: Error recording exception on span: %s", str(e))

    def set_tools_attributes(self, span: Span, tools):
        import json

        from litellm.proxy._types import SpanAttributes

        if not tools:
            return

        try:
            for i, tool in enumerate(tools):
                function = tool.get("function")
                if not function:
                    continue

                prefix = f"{SpanAttributes.LLM_REQUEST_FUNCTIONS.value}.{i}"
                self.safe_set_attribute(
                    span=span,
                    key=f"{prefix}.name",
                    value=function.get("name"),
                )
                self.safe_set_attribute(
                    span=span,
                    key=f"{prefix}.description",
                    value=function.get("description"),
                )
                self.safe_set_attribute(
                    span=span,
                    key=f"{prefix}.parameters",
                    value=json.dumps(function.get("parameters")),
                )
        except Exception as e:
            verbose_logger.error("OpenTelemetry: Error setting tools attributes: %s", str(e))
            pass

    def cast_as_primitive_value_type(self, value) -> Union[str, bool, int, float]:
        """
        Casts the value to a primitive OTEL type if it is not already a primitive type.

        OTEL supports - str, bool, int, float

        If it's not a primitive type, then it's converted to a string
        """
        if value is None:
            return ""
        if isinstance(value, (str, bool, int, float)):
            return value
        try:
            return str(value)
        except Exception:
            return ""

    @staticmethod
    def _tool_calls_kv_pair(
        tool_calls: List[ChatCompletionMessageToolCall],
    ) -> Dict[str, Any]:
        from litellm.proxy._types import SpanAttributes

        kv_pairs: Dict[str, Any] = {}
        for idx, tool_call in enumerate(tool_calls):
            _function = tool_call.get("function")
            if not _function:
                continue

            keys = Function.__annotations__.keys()
            for key in keys:
                _value = _function.get(key)
                if _value:
                    kv_pairs[f"{SpanAttributes.LLM_COMPLETIONS.value}.{idx}.function_call.{key}"] = _value

        return kv_pairs

    def set_attributes(self, span: Span, kwargs, response_obj: Optional[Any]):
        try:
            if self.callback_name == "langtrace":
                from litellm.integrations.langtrace import LangtraceAttributes

                LangtraceAttributes().set_langtrace_attributes(span, kwargs, response_obj)
                return
            elif self.callback_name == "langfuse_otel":
                from litellm.integrations.langfuse.langfuse_otel import (
                    LangfuseOtelLogger,
                )

                LangfuseOtelLogger.set_langfuse_otel_attributes(span, kwargs, response_obj)
                return
            elif self.callback_name == "weave_otel":
                from litellm.integrations.weave.weave_otel import (
                    set_weave_otel_attributes,
                )

                set_weave_otel_attributes(span, kwargs, response_obj)
                return
            from litellm.proxy._types import SpanAttributes

            optional_params = kwargs.get("optional_params", {})
            litellm_params = kwargs.get("litellm_params", {}) or {}
            standard_logging_payload: Optional[StandardLoggingPayload] = kwargs.get("standard_logging_object")
            if standard_logging_payload is None:
                raise ValueError("standard_logging_object not found in kwargs")

            # https://github.com/open-telemetry/semantic-conventions/blob/main/model/registry/gen-ai.yaml
            # Following Conventions here: https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/llm-spans.md
            #############################################
            ############ LLM CALL METADATA ##############
            #############################################
            metadata = standard_logging_payload["metadata"]
            for key, value in metadata.items():
                self.safe_set_attribute(span=span, key="metadata.{}".format(key), value=value)

            # get hidden params
            hidden_params = getattr(standard_logging_payload, "hidden_params", None) or (
                standard_logging_payload or {}
            ).get("hidden_params", {})
            if hidden_params:
                self.safe_set_attribute(
                    span=span,
                    key="hidden_params",
                    value=safe_dumps(hidden_params),
                )

            self._set_inference_identity_attributes(
                span=span,
                standard_logging_payload=standard_logging_payload,
                litellm_params=litellm_params,
            )
            # Cost breakdown tracking
            cost_breakdown: Optional[CostBreakdown] = standard_logging_payload.get("cost_breakdown")
            if cost_breakdown:
                for key, value in cost_breakdown.items():
                    if value is not None:
                        self.safe_set_attribute(
                            span=span,
                            key=f"gen_ai.cost.{key}",
                            value=value,
                        )
            #############################################
            ########## LLM Request Attributes ###########
            #############################################

            # The name of the LLM a request is being made to
            if kwargs.get("model"):
                self.safe_set_attribute(
                    span=span,
                    key=SpanAttributes.LLM_REQUEST_MODEL.value,
                    value=kwargs.get("model"),
                )

            # The LLM request type
            self.safe_set_attribute(
                span=span,
                key=SpanAttributes.LLM_REQUEST_TYPE.value,
                value=standard_logging_payload["call_type"],
            )

            # The Generative AI Provider: Azure, OpenAI, etc.
            provider_name = litellm_params.get("custom_llm_provider", "Unknown")
            # Latest-experimental semconv replaced gen_ai.system with
            # gen_ai.provider.name; emit only the conformant key in that mode.
            if self._gen_ai_semconv_latest_experimental:
                self.safe_set_attribute(
                    span=span,
                    key="gen_ai.provider.name",
                    value=provider_name,
                )
            else:
                self.safe_set_attribute(
                    span=span,
                    key=SpanAttributes.LLM_SYSTEM.value,
                    value=provider_name,
                )

            # The maximum number of tokens the LLM generates for a request.
            if optional_params.get("max_tokens"):
                self.safe_set_attribute(
                    span=span,
                    key=SpanAttributes.LLM_REQUEST_MAX_TOKENS.value,
                    value=optional_params.get("max_tokens"),
                )

            # The temperature setting for the LLM request.
            if optional_params.get("temperature"):
                self.safe_set_attribute(
                    span=span,
                    key=SpanAttributes.LLM_REQUEST_TEMPERATURE.value,
                    value=optional_params.get("temperature"),
                )

            # The top_p sampling setting for the LLM request.
            if optional_params.get("top_p"):
                self.safe_set_attribute(
                    span=span,
                    key=SpanAttributes.LLM_REQUEST_TOP_P.value,
                    value=optional_params.get("top_p"),
                )

            if self._gen_ai_semconv_latest_experimental:
                # Semconv emits gen_ai.request.stream (only when streaming) via
                # _set_semconv_request_attributes; skip the legacy llm.is_streaming.
                self._set_semconv_request_attributes(span, optional_params)
                self._set_semconv_cache_token_attributes(span, standard_logging_payload)
            else:
                self.safe_set_attribute(
                    span=span,
                    key=SpanAttributes.LLM_IS_STREAMING.value,
                    value=str(optional_params.get("stream", False)),
                )

            if optional_params.get("user"):
                self.safe_set_attribute(
                    span=span,
                    key=SpanAttributes.LLM_USER.value,
                    value=optional_params.get("user"),
                )

            # The unique identifier for the LLM call.
            # Completions have a provider response ID (e.g. "chatcmpl-xxx"),
            # but Embeddings and Image-gen responses do not.  Fall back to
            # the litellm call ID so every call type can be correlated
            # across LiteLLM UI, Phoenix traces, and provider logs (Issue #8).
            response_id = (response_obj.get("id") if response_obj else None) or standard_logging_payload.get("id")
            if response_id:
                self.safe_set_attribute(
                    span=span,
                    key="gen_ai.response.id",
                    value=response_id,
                )

            litellm_call_id = standard_logging_payload.get("litellm_call_id")
            if litellm_call_id:
                self.safe_set_attribute(
                    span=span,
                    key="litellm.call_id",
                    value=litellm_call_id,
                )

            # The model used to generate the response.
            if response_obj and response_obj.get("model"):
                self.safe_set_attribute(
                    span=span,
                    key=SpanAttributes.LLM_RESPONSE_MODEL.value,
                    value=response_obj.get("model"),
                )

            usage = response_obj and response_obj.get("usage")
            if usage:
                self.safe_set_attribute(
                    span=span,
                    key=SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS.value,
                    value=usage.get("total_tokens"),
                )

                # The number of tokens used in the LLM response (completion).
                self.safe_set_attribute(
                    span=span,
                    key=SpanAttributes.GEN_AI_USAGE_OUTPUT_TOKENS.value,
                    value=usage.get("completion_tokens"),
                )

                # The number of tokens used in the LLM prompt.
                self.safe_set_attribute(
                    span=span,
                    key=SpanAttributes.GEN_AI_USAGE_INPUT_TOKENS.value,
                    value=usage.get("prompt_tokens"),
                )

                ########################################################################
            ########## LLM Request Medssages / tools / content Attributes ###########
            #########################################################################

            if not self._capture_in_span():
                return

            if optional_params.get("tools"):
                tools = optional_params["tools"]
                self.set_tools_attributes(span, tools)

            if kwargs.get("messages"):
                transformed_messages = self._transform_messages_to_otel_semantic_conventions(kwargs.get("messages"))
                self.safe_set_attribute(
                    span=span,
                    key=SpanAttributes.GEN_AI_INPUT_MESSAGES.value,
                    value=safe_dumps(transformed_messages),
                )

            # Coalesce the different kwarg names that carry the system
            # prompt depending on the call path:
            #   - "system_instructions" — Vertex AI Gemini chat-completion
            #   - "instructions"        — OpenAI Responses API
            #   - "system"              — Anthropic Messages API
            # Use `is not None` rather than truthiness to avoid falsy
            # values (e.g. []) falling through to the wrong kwarg.
            system_instructions = (
                kwargs.get("system_instructions")
                if kwargs.get("system_instructions") is not None
                else (kwargs.get("instructions") if kwargs.get("instructions") is not None else kwargs.get("system"))
            )
            if system_instructions:
                if isinstance(system_instructions, str):
                    # Plain text system prompt — no transformation needed
                    self.safe_set_attribute(
                        span=span,
                        key=SpanAttributes.GEN_AI_SYSTEM_INSTRUCTIONS.value,
                        value=system_instructions,
                    )
                elif isinstance(system_instructions, list) and all(
                    isinstance(b, dict) and "role" not in b for b in system_instructions
                ):
                    # Anthropic Messages-style: `system` is a list of content
                    # blocks like [{"type": "text", "text": "..."}] — these
                    # are NOT messages and have no `role`. Treating each as
                    # a message drops the text into msg.get("content","") =
                    # "" and emits empty content under role=user (#29756).
                    # Hoist them all under a single system message.
                    parts: List[Dict[str, Any]] = []
                    for block in system_instructions:
                        text = block.get("text") or block.get("content") or ""
                        if not text:
                            continue
                        parts.append({"type": "text", "content": text})
                    if parts:
                        wrapped = [{"role": "system", "parts": parts}]
                        self.safe_set_attribute(
                            span=span,
                            key=SpanAttributes.GEN_AI_SYSTEM_INSTRUCTIONS.value,
                            value=safe_dumps(wrapped),
                        )
                else:
                    transformed_system_instructions = self._transform_messages_to_otel_semantic_conventions(
                        system_instructions
                    )
                    self.safe_set_attribute(
                        span=span,
                        key=SpanAttributes.GEN_AI_SYSTEM_INSTRUCTIONS.value,
                        value=safe_dumps(transformed_system_instructions),
                    )

            if self._gen_ai_semconv_latest_experimental:
                operation_name = self._gen_ai_operation_name(kwargs)
            else:
                operation_name = (
                    "chat"
                    if standard_logging_payload.get("call_type") == "completion"
                    else standard_logging_payload.get("call_type") or "chat"
                )
            self.safe_set_attribute(
                span=span,
                key=SpanAttributes.GEN_AI_OPERATION_NAME.value,
                value=operation_name,
            )

            if standard_logging_payload.get("request_id"):
                self.safe_set_attribute(
                    span=span,
                    key=SpanAttributes.GEN_AI_REQUEST_ID.value,
                    value=standard_logging_payload.get("request_id"),
                )
            #############################################
            ########## LLM Response Attributes ##########
            #############################################
            if response_obj is not None:
                if response_obj.get("choices"):
                    transformed_choices = self._transform_choices_to_otel_semantic_conventions(
                        response_obj.get("choices")
                    )
                    self.safe_set_attribute(
                        span=span,
                        key=SpanAttributes.GEN_AI_OUTPUT_MESSAGES.value,
                        value=safe_dumps(transformed_choices),
                    )

                    finish_reasons = []
                    for idx, choice in enumerate(response_obj.get("choices")):
                        if choice.get("finish_reason"):
                            finish_reasons.append(choice.get("finish_reason"))

                    if finish_reasons:
                        self.safe_set_attribute(
                            span=span,
                            key=SpanAttributes.GEN_AI_RESPONSE_FINISH_REASONS.value,
                            value=safe_dumps(finish_reasons),
                        )

                    for idx, choice in enumerate(response_obj.get("choices")):
                        if choice.get("finish_reason"):
                            message = choice.get("message")
                            tool_calls = message.get("tool_calls")
                            if tool_calls:
                                kv_pairs = OpenTelemetry._tool_calls_kv_pair(tool_calls)  # type: ignore
                                for key, value in kv_pairs.items():
                                    self.safe_set_attribute(
                                        span=span,
                                        key=key,
                                        value=value,
                                    )

                elif response_obj.get("output"):
                    # Responses API: ResponsesAPIResponse has an "output"
                    # list instead of "choices".  Each item with
                    # type="message" contains a "content" list of
                    # OutputText objects (type="output_text").
                    output_items = response_obj.get("output")
                    output_messages = self._transform_responses_api_output_to_otel(output_items)
                    if output_messages:
                        self.safe_set_attribute(
                            span=span,
                            key=SpanAttributes.GEN_AI_OUTPUT_MESSAGES.value,
                            value=safe_dumps(output_messages),
                        )

                    # Emit per-tool-call span attributes (parity with
                    # the choices branch that calls _tool_calls_kv_pair).
                    # Convert Responses API function_call items to the
                    # ChatCompletionMessageToolCall format expected by
                    # _tool_calls_kv_pair.
                    tool_calls = []
                    for out_item in output_items:
                        item_d = self._to_dict(out_item)
                        if item_d and item_d.get("type") == "function_call":
                            tool_calls.append(
                                {
                                    "function": {
                                        "name": item_d.get("name", ""),
                                        "arguments": item_d.get("arguments", ""),
                                    }
                                }
                            )
                    if tool_calls:
                        kv_pairs = OpenTelemetry._tool_calls_kv_pair(tool_calls)  # type: ignore
                        for key, value in kv_pairs.items():
                            self.safe_set_attribute(
                                span=span,
                                key=key,
                                value=value,
                            )

                    # Extract finish reason from ResponsesAPIResponse.status
                    status = response_obj.get("status")
                    if status:
                        self.safe_set_attribute(
                            span=span,
                            key=SpanAttributes.GEN_AI_RESPONSE_FINISH_REASONS.value,
                            value=safe_dumps([status]),
                        )

        except Exception as e:
            self.handle_callback_failure(callback_name=self.callback_name or "opentelemetry")
            verbose_logger.exception("OpenTelemetry logging error in set_attributes %s", str(e))

    def _cast_as_primitive_value_type(self, value) -> Union[str, bool, int, float]:
        """
        Casts the value to a primitive OTEL type if it is not already a primitive type.

        OTEL supports - str, bool, int, float

        If it's not a primitive type, then it's converted to a string
        """
        if value is None:
            return ""
        if isinstance(value, (str, bool, int, float)):
            return value
        try:
            return str(value)
        except Exception:
            return ""

    def safe_set_attribute(self, span: Span, key: str, value: Any):
        """
        Safely sets an attribute on the span, ensuring the value is a primitive type.
        """
        primitive_value = self._cast_as_primitive_value_type(value)
        span.set_attribute(key, primitive_value)

    def _transform_messages_to_otel_semantic_conventions(self, messages: Union[List[dict], str]) -> List[dict]:
        """
        Transforms LiteLLM/OpenAI style messages into OTEL GenAI 1.38 compliant format.
        OTEL expects a 'parts' array instead of a single 'content' string.
        """
        if isinstance(messages, str):
            # Handle system_instructions passed as a string
            return [
                {
                    "role": "system",
                    "parts": [{"type": "text", "content": messages}],
                }
            ]

        transformed = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts = []

            if isinstance(content, str):
                parts.append({"type": "text", "content": content})
            elif isinstance(content, list):
                # Handle multi-modal content if necessary.
                # Anthropic content blocks carry the text under "text" but
                # the OTel GenAI semconv 1.38 part shape uses "content" — keep
                # the part-key consistent with the string-content path so
                # downstream consumers don't have to look up two keys for the
                # same field (#29756).
                for part in content:
                    if isinstance(part, dict):
                        if "text" in part and "content" not in part:
                            normalized = {**part, "content": part["text"]}
                            normalized.pop("text", None)
                            parts.append(normalized)
                        else:
                            parts.append(part)
                    else:
                        parts.append({"type": "text", "content": str(part)})

            transformed_msg = {"role": role, "parts": parts}
            if "id" in msg:
                transformed_msg["id"] = msg["id"]
            if "tool_calls" in msg:
                transformed_msg["tool_calls"] = msg["tool_calls"]
            if "tool_call_id" in msg:
                transformed_msg["tool_call_id"] = msg["tool_call_id"]
            transformed.append(transformed_msg)

        return transformed

    def _transform_choices_to_otel_semantic_conventions(self, choices: List[dict]) -> List[dict]:
        """
        Transforms choices into OTEL GenAI 1.38 compliant format for output.messages.
        """
        transformed = []
        for choice in choices:
            message = choice.get("message") or {}
            finish_reason = choice.get("finish_reason")

            transformed_msg = self._transform_messages_to_otel_semantic_conventions([message])[0]
            if finish_reason:
                transformed_msg["finish_reason"] = finish_reason

            transformed.append(transformed_msg)
        return transformed

    @staticmethod
    def _to_dict(obj) -> Optional[dict]:
        """Normalize an object to a plain dict.

        Handles three forms that appear in practice:

        1. Plain ``dict`` — returned as-is.
        2. LiteLLM's ``BaseLiteLLMOpenAIResponseObject`` — exposes a
           ``.get()`` method that delegates to ``__dict__``.
        3. Raw Pydantic v2 models from the ``openai`` SDK (e.g.
           ``ResponseOutputMessage``, ``ResponseOutputText``) — these do
           **not** have ``.get()`` but do have ``.model_dump()``.

        Returns ``None`` for anything else so callers can skip it.
        """
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "get"):
            # BaseLiteLLMOpenAIResponseObject duck-type
            return obj  # type: ignore[return-value]
        if hasattr(obj, "model_dump"):
            # Raw Pydantic v2 model (e.g. openai SDK types)
            return obj.model_dump()  # type: ignore[union-attr]
        return None

    def _transform_responses_api_output_to_otel(self, output: List) -> List[dict]:
        """
        Transform Responses API output items into OTEL GenAI 1.38 format.

        The Responses API returns output as a list of items, each with a
        ``type`` field.  Message items (``type="message"``) contain a
        ``content`` list of ``OutputText`` objects with ``type="output_text"``
        and ``text`` fields.

        Items may be plain dicts, LiteLLM wrapper objects (with ``.get()``),
        or raw Pydantic v2 models from the ``openai`` SDK (with
        ``.model_dump()``).  We normalize each item to a dict via
        ``_to_dict`` before processing.

        This method converts them to the same ``{"role": ..., "parts": [...]}``
        format used by ``_transform_choices_to_otel_semantic_conventions``.
        """
        transformed = []
        for raw_item in output:
            item = self._to_dict(raw_item)
            if item is None:
                continue
            if item.get("type") == "message":
                role = item.get("role", "assistant")
                parts = []
                for raw_content in item.get("content", []):
                    content = self._to_dict(raw_content)
                    if content is None:
                        continue
                    if content.get("type") == "output_text":
                        text = content.get("text", "")
                        if text:
                            parts.append({"type": "text", "content": text})
                if parts:
                    transformed.append({"role": role, "parts": parts})
            elif item.get("type") == "function_call":
                # Surface tool calls from Responses API output
                part: dict = {
                    "type": "tool_call",
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", ""),
                }
                if item.get("call_id"):
                    part["id"] = item["call_id"]
                transformed.append({"role": "assistant", "parts": [part]})
        return transformed

    def set_raw_request_attributes(self, span: Span, kwargs, response_obj):
        try:
            # Only set provider-specific raw payload attributes on this span.
            # The parent litellm_request span already carries the standard
            # gen_ai.* / metadata.* attributes — duplicating them here doubles
            # storage and adds noise (Issue #3).
            litellm_params = kwargs.get("litellm_params", {}) or {}
            custom_llm_provider = litellm_params.get("custom_llm_provider", "Unknown")

            _raw_response = kwargs.get("original_response")
            _additional_args = kwargs.get("additional_args", {}) or {}
            complete_input_dict = _additional_args.get("complete_input_dict")
            #############################################
            ########## LLM Request Attributes ###########
            #############################################

            # OTEL Attributes for the RAW Request to https://docs.anthropic.com/en/api/messages
            if complete_input_dict and isinstance(complete_input_dict, dict):
                for param, val in complete_input_dict.items():
                    self.safe_set_attribute(
                        span=span,
                        key=f"llm.{custom_llm_provider}.{param}",
                        value=val,
                    )

            #############################################
            ########## LLM Response Attributes ##########
            #############################################
            if _raw_response and isinstance(_raw_response, str):
                # cast sr -> dict
                import json

                try:
                    _raw_response = json.loads(_raw_response)
                    for param, val in _raw_response.items():
                        self.safe_set_attribute(
                            span=span,
                            key=f"llm.{custom_llm_provider}.{param}",
                            value=val,
                        )
                except json.JSONDecodeError:
                    verbose_logger.debug(
                        "litellm.integrations.opentelemetry.py::set_raw_request_attributes() - raw_response not json string - {}".format(
                            _raw_response
                        )
                    )

                    self.safe_set_attribute(
                        span=span,
                        key=f"llm.{custom_llm_provider}.stringified_raw_response",
                        value=_raw_response,
                    )
        except Exception as e:
            verbose_logger.exception(
                "OpenTelemetry logging error in set_raw_request_attributes %s",
                str(e),
            )

    def _to_ns(self, dt):
        if dt is None:
            return int(datetime.now().timestamp() * 1e9)
        if isinstance(dt, (int, float)):
            return int(dt * 1e9)
        return int(dt.timestamp() * 1e9)

    def _get_span_name(self, kwargs):
        litellm_params = kwargs.get("litellm_params", {})
        metadata = litellm_params.get("metadata") or {}
        generation_name = metadata.get("generation_name")

        if generation_name:
            return generation_name

        if self._gen_ai_semconv_latest_experimental:
            model = kwargs.get("model") or "unknown"
            return f"{self._gen_ai_operation_name(kwargs)} {model}"

        return LITELLM_REQUEST_SPAN_NAME

    def get_traceparent_from_header(self, headers):
        if headers is None:
            return None
        _traceparent = headers.get("traceparent", None)
        if _traceparent is None:
            return None

        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator,
        )

        propagator = TraceContextTextMapPropagator()
        carrier = {"traceparent": _traceparent}
        _parent_context = propagator.extract(carrier=carrier)

        return _parent_context

    def _get_span_context(self, kwargs, default_span: Optional[Span] = None):
        from opentelemetry import context, trace
        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator,
        )

        litellm_params = kwargs.get("litellm_params", {}) or {}
        proxy_server_request = litellm_params.get("proxy_server_request", {}) or {}
        headers = proxy_server_request.get("headers", {}) or {}
        traceparent = headers.get("traceparent", None)
        _metadata = litellm_params.get("metadata", {}) or {}
        parent_otel_span = _metadata.get("litellm_parent_otel_span", None)

        # Fallback: check litellm_metadata (used by /v1/messages and other
        # LITELLM_METADATA_ROUTES that store proxy-internal metadata
        # separately from the provider's native "metadata" field).
        if parent_otel_span is None:
            _litellm_metadata = litellm_params.get("litellm_metadata", {}) or {}
            parent_otel_span = _litellm_metadata.get("litellm_parent_otel_span", None)

        # Priority 1: Explicit parent span from metadata
        if parent_otel_span is not None:
            verbose_logger.debug("OpenTelemetry: Using explicit parent span from metadata")
            return trace.set_span_in_context(parent_otel_span), None

        # Priority 2: HTTP traceparent header
        if traceparent is not None:
            verbose_logger.debug("OpenTelemetry: Using traceparent header for context propagation")
            carrier = {"traceparent": traceparent}
            return (
                TraceContextTextMapPropagator().extract(carrier=carrier),
                None,
            )

        # Priority 3: Active span from global context (auto-detection)
        try:
            current_span = trace.get_current_span()
            if current_span is not None:
                span_context = current_span.get_span_context()
                if span_context.is_valid:
                    verbose_logger.debug(
                        "OpenTelemetry: Using active span from global context: %s (trace_id=%s, span_id=%s, is_recording=%s)",
                        current_span,
                        format(span_context.trace_id, "032x"),
                        format(span_context.span_id, "016x"),
                        current_span.is_recording(),
                    )
                    return context.get_current(), current_span
        except Exception as e:
            verbose_logger.debug("OpenTelemetry: Error getting current span: %s", str(e))

        # Priority 4: No parent context
        verbose_logger.debug("OpenTelemetry: No parent context found, creating root span")
        return None, None

    def _get_span_processor(
        self,
        dynamic_headers: Optional[dict] = None,
        config_override: Optional[OpenTelemetryConfig] = None,
    ):
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
            SimpleSpanProcessor,
            SpanExporter,
        )

        otel_exporter = config_override.exporter if config_override else self.OTEL_EXPORTER
        otel_endpoint = config_override.endpoint if config_override else self.OTEL_ENDPOINT
        otel_headers = config_override.headers if config_override else self.OTEL_HEADERS

        verbose_logger.debug(
            "OpenTelemetry Logger, initializing span processor \nexporter: %s\nendpoint: %s\nheaders: %s",
            otel_exporter,
            otel_endpoint,
            redact_string(str(otel_headers)),
        )
        _split_otel_headers = OpenTelemetry._get_headers_dictionary(headers=dynamic_headers or otel_headers)

        if dynamic_headers:
            verbose_logger.debug(
                "[OTEL DEBUG] Creating span processor with DYNAMIC headers: %s",
                redact_string(str(_split_otel_headers)),
            )
        elif config_override:
            verbose_logger.debug(
                "[OTEL DEBUG] Creating span processor with DYNAMIC config, endpoint: %s",
                otel_endpoint,
            )
        else:
            verbose_logger.debug("[OTEL DEBUG] Creating span processor with GLOBAL headers")

        if hasattr(otel_exporter, "export"):  # Check if it has the export method that SpanExporter requires
            verbose_logger.debug(
                "OpenTelemetry: intiializing SpanExporter. Value of OTEL_EXPORTER: %s",
                otel_exporter,
            )
            return SimpleSpanProcessor(cast(SpanExporter, otel_exporter))

        if otel_exporter == "console":
            verbose_logger.debug(
                "OpenTelemetry: intiializing console exporter. Value of OTEL_EXPORTER: %s",
                otel_exporter,
            )
            return BatchSpanProcessor(ConsoleSpanExporter())
        elif otel_exporter == "otlp_http" or otel_exporter == "http/protobuf" or otel_exporter == "http/json":
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter as OTLPSpanExporterHTTP,
                )
            except ImportError as exc:
                raise ImportError(
                    "OpenTelemetry OTLP HTTP exporter is not available. Install "
                    "`opentelemetry-exporter-otlp` to enable OTLP HTTP."
                ) from exc

            verbose_logger.debug(
                "OpenTelemetry: intiializing http exporter. Value of OTEL_EXPORTER: %s",
                otel_exporter,
            )
            normalized_endpoint = self._normalize_otel_endpoint(otel_endpoint, "traces")
            return BatchSpanProcessor(
                OTLPSpanExporterHTTP(endpoint=normalized_endpoint, headers=_split_otel_headers),
            )
        elif otel_exporter == "otlp_grpc" or otel_exporter == "grpc":
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter as OTLPSpanExporterGRPC,
                )
            except ImportError as exc:
                raise ImportError(
                    "OpenTelemetry OTLP gRPC exporter is not available. Install "
                    "`opentelemetry-exporter-otlp` and `grpcio` (or `litellm[grpc]`)."
                ) from exc

            verbose_logger.debug(
                "OpenTelemetry: intiializing grpc exporter. Value of OTEL_EXPORTER: %s",
                otel_exporter,
            )
            normalized_endpoint = self._normalize_otel_endpoint(otel_endpoint, "traces")
            return BatchSpanProcessor(
                OTLPSpanExporterGRPC(endpoint=normalized_endpoint, headers=_split_otel_headers),
            )
        else:
            verbose_logger.debug(
                "OpenTelemetry: intiializing console exporter. Value of OTEL_EXPORTER: %s",
                otel_exporter,
            )
            return BatchSpanProcessor(ConsoleSpanExporter())

    def _get_log_exporter(self):
        """
        Get the appropriate log exporter based on the configuration.
        """
        verbose_logger.debug(
            "OpenTelemetry Logger, initializing log exporter \nself.OTEL_EXPORTER: %s\nself.OTEL_ENDPOINT: %s\nself.OTEL_HEADERS: %s",
            self.OTEL_EXPORTER,
            self.OTEL_ENDPOINT,
            redact_string(str(self.OTEL_HEADERS)),
        )

        _split_otel_headers = OpenTelemetry._get_headers_dictionary(self.OTEL_HEADERS)

        # Normalize endpoint for logs - ensure it points to /v1/logs instead of /v1/traces
        normalized_endpoint = self._normalize_otel_endpoint(self.OTEL_ENDPOINT, "logs")

        verbose_logger.debug(
            "OpenTelemetry: Log endpoint normalized from %s to %s",
            self.OTEL_ENDPOINT,
            normalized_endpoint,
        )

        if hasattr(self.OTEL_EXPORTER, "export"):
            # Custom exporter provided
            verbose_logger.debug(
                "OpenTelemetry: Using custom log exporter. Value of OTEL_EXPORTER: %s",
                self.OTEL_EXPORTER,
            )
            return self.OTEL_EXPORTER

        otel_logs_exporter = os.getenv("OTEL_LOGS_EXPORTER")
        if self.OTEL_EXPORTER == "console" or otel_logs_exporter == "console":
            from opentelemetry.sdk._logs.export import ConsoleLogExporter

            verbose_logger.debug(
                "OpenTelemetry: Using console log exporter. Value of OTEL_EXPORTER: %s",
                self.OTEL_EXPORTER,
            )
            return ConsoleLogExporter()
        elif (
            self.OTEL_EXPORTER == "otlp_http"
            or self.OTEL_EXPORTER == "http/protobuf"
            or self.OTEL_EXPORTER == "http/json"
        ):
            from opentelemetry.exporter.otlp.proto.http._log_exporter import (
                OTLPLogExporter,
            )

            verbose_logger.debug(
                "OpenTelemetry: Using HTTP log exporter. Value of OTEL_EXPORTER: %s, endpoint: %s",
                self.OTEL_EXPORTER,
                normalized_endpoint,
            )
            return OTLPLogExporter(endpoint=normalized_endpoint, headers=_split_otel_headers)
        elif self.OTEL_EXPORTER == "otlp_grpc" or self.OTEL_EXPORTER == "grpc":
            try:
                from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
                    OTLPLogExporter,
                )
            except ImportError as exc:
                raise ImportError(
                    "OpenTelemetry OTLP gRPC log exporter is not available. Install "
                    "`opentelemetry-exporter-otlp` and `grpcio` (or `litellm[grpc]`)."
                ) from exc

            verbose_logger.debug(
                "OpenTelemetry: Using gRPC log exporter. Value of OTEL_EXPORTER: %s, endpoint: %s",
                self.OTEL_EXPORTER,
                normalized_endpoint,
            )
            return OTLPLogExporter(endpoint=normalized_endpoint, headers=_split_otel_headers)
        else:
            verbose_logger.warning(
                "OpenTelemetry: Unknown log exporter '%s', defaulting to console. Supported: console, otlp_http, otlp_grpc",
                self.OTEL_EXPORTER,
            )
            from opentelemetry.sdk._logs.export import ConsoleLogExporter

            return ConsoleLogExporter()

    def _get_metric_reader(self):
        """
        Get the appropriate metric reader based on the configuration.
        """
        from opentelemetry.sdk.metrics import Histogram
        from opentelemetry.sdk.metrics.export import (
            AggregationTemporality,
            ConsoleMetricExporter,
            PeriodicExportingMetricReader,
        )

        verbose_logger.debug(
            "OpenTelemetry Logger, initializing metric reader\nself.OTEL_EXPORTER: %s\nself.OTEL_ENDPOINT: %s\nself.OTEL_HEADERS: %s",
            self.OTEL_EXPORTER,
            self.OTEL_ENDPOINT,
            redact_string(str(self.OTEL_HEADERS)),
        )

        _split_otel_headers = OpenTelemetry._get_headers_dictionary(self.OTEL_HEADERS)
        normalized_endpoint = self._normalize_otel_endpoint(self.OTEL_ENDPOINT, "metrics")

        if self.OTEL_EXPORTER == "console":
            exporter = ConsoleMetricExporter()
            return PeriodicExportingMetricReader(exporter, export_interval_millis=5000)

        elif (
            self.OTEL_EXPORTER == "otlp_http"
            or self.OTEL_EXPORTER == "http/protobuf"
            or self.OTEL_EXPORTER == "http/json"
        ):
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                OTLPMetricExporter,
            )

            exporter = OTLPMetricExporter(
                endpoint=normalized_endpoint,
                headers=_split_otel_headers,
                preferred_temporality={Histogram: AggregationTemporality.DELTA},
            )
            return PeriodicExportingMetricReader(exporter, export_interval_millis=5000)

        elif self.OTEL_EXPORTER == "otlp_grpc" or self.OTEL_EXPORTER == "grpc":
            try:
                from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                    OTLPMetricExporter,
                )
            except ImportError as exc:
                raise ImportError(
                    "OpenTelemetry OTLP gRPC metric exporter is not available. Install "
                    "`opentelemetry-exporter-otlp` and `grpcio` (or `litellm[grpc]`)."
                ) from exc

            exporter = OTLPMetricExporter(
                endpoint=normalized_endpoint,
                headers=_split_otel_headers,
                preferred_temporality={Histogram: AggregationTemporality.DELTA},
            )
            return PeriodicExportingMetricReader(exporter, export_interval_millis=5000)

        else:
            verbose_logger.warning(
                "OpenTelemetry: Unknown metric exporter '%s', defaulting to console. Supported: console, otlp_http, otlp_grpc",
                self.OTEL_EXPORTER,
            )
            exporter = ConsoleMetricExporter()
            return PeriodicExportingMetricReader(exporter, export_interval_millis=5000)

    def _normalize_otel_endpoint(self, endpoint: Optional[str], signal_type: str) -> Optional[str]:
        """
        Normalize the endpoint URL for a specific OpenTelemetry signal type.

        The OTLP exporters expect endpoints to use signal-specific paths:
        - traces: /v1/traces
        - metrics: /v1/metrics
        - logs: /v1/logs

        This method ensures the endpoint has the correct path for the given signal type.

        Args:
            endpoint: The endpoint URL to normalize
            signal_type: The telemetry signal type ('traces', 'metrics', or 'logs')

        Returns:
            Normalized endpoint URL with the correct signal path

        Examples:
            _normalize_otel_endpoint("http://collector:4318/v1/traces", "logs")
            -> "http://collector:4318/v1/logs"

            _normalize_otel_endpoint("http://collector:4318", "traces")
            -> "http://collector:4318/v1/traces"

            _normalize_otel_endpoint("http://collector:4318/v1/logs", "metrics")
            -> "http://collector:4318/v1/metrics"
        """
        if not endpoint:
            return endpoint

        # Validate signal_type
        valid_signals = {"traces", "metrics", "logs"}
        if signal_type not in valid_signals:
            verbose_logger.warning(
                "Invalid signal_type '%s' provided to _normalize_otel_endpoint. "
                "Valid values: %s. Returning endpoint unchanged.",
                signal_type,
                valid_signals,
            )
            return endpoint

        # Remove trailing slash
        endpoint = endpoint.rstrip("/")

        # Splunk Observability Cloud OTLP/HTTP uses /v2/trace/otlp (not /v1/traces). Do not rewrite.
        if signal_type == "traces" and "/v2/trace/otlp" in endpoint:
            return endpoint

        # Check if endpoint already ends with the correct signal path
        target_path = f"/v1/{signal_type}"
        if endpoint.endswith(target_path):
            return endpoint

        # Replace existing signal path with the target signal path
        other_signals = valid_signals - {signal_type}
        for other_signal in other_signals:
            other_path = f"/v1/{other_signal}"
            if endpoint.endswith(other_path):
                endpoint = endpoint.rsplit("/", 1)[0] + f"/{signal_type}"
                return endpoint

        # No existing signal path found, append the target path
        if not endpoint.endswith("/v1"):
            endpoint = endpoint + target_path
        else:
            endpoint = endpoint + f"/{signal_type}"

        return endpoint

    @staticmethod
    def _get_headers_dictionary(
        headers: Optional[Union[str, dict]],
    ) -> Dict[str, str]:
        """
        Convert a string or dictionary of headers into a dictionary of headers.
        """
        _split_otel_headers: Dict[str, str] = {}
        if headers:
            if isinstance(headers, str):
                # when passed HEADERS="x-honeycomb-team=B85YgLm96******"
                # Split only on first '=' occurrence
                parts = headers.split(",")
                for part in parts:
                    key, value = part.split("=", 1)
                    _split_otel_headers[key] = value
            elif isinstance(headers, dict):
                _split_otel_headers = headers
        return _split_otel_headers

    async def async_management_endpoint_success_hook(
        self,
        logging_payload: ManagementEndpointLoggingPayload,
        parent_otel_span: Optional[Span] = None,
    ):
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode

        _start_time_ns = 0
        _end_time_ns = 0

        start_time = logging_payload.start_time
        end_time = logging_payload.end_time

        if isinstance(start_time, float):
            _start_time_ns = int(start_time * 1e9)
        else:
            _start_time_ns = self._to_ns(start_time)

        if isinstance(end_time, float):
            _end_time_ns = int(end_time * 1e9)
        else:
            _end_time_ns = self._to_ns(end_time)

        if parent_otel_span is not None:
            _span_name = logging_payload.route
            management_endpoint_span = self.tracer.start_span(
                name=_span_name,
                context=trace.set_span_in_context(parent_otel_span),
                start_time=_start_time_ns,
            )

            _request_data = logging_payload.request_data
            if _request_data is not None:
                for key, value in _request_data.items():
                    self.safe_set_attribute(
                        span=management_endpoint_span,
                        key=f"request.{key}",
                        value=value,
                    )

            _response = logging_payload.response
            if _response is not None:
                for key, value in _response.items():
                    self.safe_set_attribute(
                        span=management_endpoint_span,
                        key=f"response.{key}",
                        value=value,
                    )

            management_endpoint_span.set_status(Status(StatusCode.OK))
            management_endpoint_span.end(end_time=_end_time_ns)

            # The management wrapper has no other hook that closes the SERVER span.
            self.set_response_status_code_attribute(parent_otel_span, 200)
            parent_otel_span.set_status(Status(StatusCode.OK))
            parent_otel_span.end(end_time=_end_time_ns)

    async def async_management_endpoint_failure_hook(
        self,
        logging_payload: ManagementEndpointLoggingPayload,
        parent_otel_span: Optional[Span] = None,
    ):
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode

        _start_time_ns = 0
        _end_time_ns = 0

        start_time = logging_payload.start_time
        end_time = logging_payload.end_time

        if isinstance(start_time, float):
            _start_time_ns = int(int(start_time) * 1e9)
        else:
            _start_time_ns = self._to_ns(start_time)

        if isinstance(end_time, float):
            _end_time_ns = int(int(end_time) * 1e9)
        else:
            _end_time_ns = self._to_ns(end_time)

        if parent_otel_span is not None:
            _span_name = logging_payload.route
            management_endpoint_span = self.tracer.start_span(
                name=_span_name,
                context=trace.set_span_in_context(parent_otel_span),
                start_time=_start_time_ns,
            )

            _request_data = logging_payload.request_data
            if _request_data is not None:
                for key, value in _request_data.items():
                    self.safe_set_attribute(
                        span=management_endpoint_span,
                        key=f"request.{key}",
                        value=value,
                    )

            _exception = logging_payload.exception
            self.safe_set_attribute(
                span=management_endpoint_span,
                key="exception",
                value=str(_exception),
            )
            management_endpoint_span.set_status(Status(StatusCode.ERROR))
            management_endpoint_span.end(end_time=_end_time_ns)

            # The management wrapper has no other hook that closes the SERVER span.
            from litellm.litellm_core_utils.litellm_logging import (
                StandardLoggingPayloadSetup,
            )

            error_information = StandardLoggingPayloadSetup.get_error_information(
                original_exception=_exception,
            )
            parent_otel_span.set_status(Status(StatusCode.ERROR))
            self._record_exception_on_span(
                span=parent_otel_span,
                kwargs={
                    "exception": _exception,
                    "standard_logging_object": {"error_information": error_information},
                },
            )
            parent_otel_span.end(end_time=_end_time_ns)

    def create_litellm_proxy_request_started_span(
        self,
        start_time: datetime,
        headers: dict,
    ) -> Optional[Span]:
        """
        Create a span for the received proxy server request.
        """

        return self.tracer.start_span(
            name=LITELLM_PROXY_REQUEST_SPAN_NAME,
            start_time=self._to_ns(start_time),
            context=self.get_traceparent_from_header(headers=headers),
            kind=self.span_kind.SERVER,
        )

    def set_proxy_request_route_attributes(
        self,
        span: Optional[Span],
        *,
        url_path: Optional[str] = None,
        http_route: Optional[str] = None,
    ) -> None:
        """
        Set OTel-standard ``http.route`` / ``url.path`` on the proxy SERVER
        span. Called from the auth path, the only point where both the
        SERVER span and the request are in hand. No-op if span/value missing.
        """
        if span is None:
            return
        if url_path:
            self.safe_set_attribute(span=span, key=URL_PATH_ATTRIBUTE, value=url_path)
        if http_route:
            self.safe_set_attribute(span=span, key=HTTP_ROUTE_ATTRIBUTE, value=http_route)

    def set_response_status_code_attribute(self, span: Optional[Span], status_code: Optional[int]) -> None:
        """
        Set OTel-standard ``http.response.status_code`` (int) on the proxy
        SERVER span. The failure path sets this from the error code in
        ``_record_exception_on_span``; this is the success-path counterpart
        so the attribute is present on every SERVER span regardless of
        outcome (required by the HTTP semconv, and needed for error-ratio /
        status-breakdown dashboards). No-op if span/value missing.
        """
        if span is None or status_code is None:
            return
        self.safe_set_attribute(
            span=span,
            key=HTTP_RESPONSE_STATUS_CODE_ATTRIBUTE,
            value=int(status_code),
        )

    def record_error_attributes_on_span(
        self,
        span: Optional[Span],
        exception: Optional[Exception],
        status_code: int,
    ) -> None:
        """Stamp structured ``error.*`` attributes on the SERVER span from the
        exception returned to the client, with ``error.code`` pinned to the real
        response status. Idempotent (overwrites); emits no exception event."""
        if span is None or exception is None:
            return
        from litellm.litellm_core_utils.litellm_logging import (
            StandardLoggingPayloadSetup,
        )

        error_information = StandardLoggingPayloadSetup.get_error_information(original_exception=exception)
        error_information["error_code"] = str(status_code)
        self._record_exception_on_span(
            span=span,
            kwargs={"standard_logging_object": {"error_information": error_information}},
        )

    def set_preprocessing_duration_attribute(self, span: Optional[Span], container: Any) -> None:
        """
        Set ``litellm.preprocessing.duration_ms`` (proxy-receive -> first
        provider handoff) on the proxy SERVER span. ``litellm_received_at``
        rides request metadata; ``first_api_call_start_time`` is the
        set-once first-handoff instant (retries/backoff excluded). Works
        uniformly for the success (model_call_details) and failure
        (request_data) containers. No-op if span/either anchor is missing.
        """
        if span is None or not isinstance(container, dict):
            return
        received_at = None
        # first_api_call_start_time is top-level (never in user metadata).
        first_handoff = container.get("first_api_call_start_time")
        _lp = container.get("litellm_params")
        for _md in (
            (_lp or {}).get("metadata") if isinstance(_lp, dict) else None,
            container.get("metadata"),
            container.get("litellm_metadata"),
        ):
            if isinstance(_md, dict):
                received_at = received_at or _md.get("litellm_received_at")
        if received_at is None or first_handoff is None:
            return
        try:
            start_ts = self._to_timestamp(received_at)
            end_ts = self._to_timestamp(first_handoff)
        except Exception:
            return
        if start_ts is None or end_ts is None:
            return
        duration_ms = (end_ts - start_ts) * 1000.0
        # Clock skew → omit rather than emit a negative latency.
        if duration_ms < 0:
            return
        self.safe_set_attribute(
            span=span,
            key=PREPROCESSING_DURATION_MS_ATTRIBUTE,
            value=duration_ms,
        )
