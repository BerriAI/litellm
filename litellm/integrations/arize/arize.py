"""
arize AI is OTEL compatible

this file has Arize ai specific helper functions
"""

import math
import os
import random
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final

from litellm._logging import verbose_logger
from litellm.integrations.arize import _utils
from litellm.integrations.arize._utils import ArizeOTELAttributes
from litellm.integrations.opentelemetry import _MAX_DYNAMIC_TRACER_PROVIDERS, OpenTelemetry, OpenTelemetryConfig
from litellm.types.integrations.arize import ArizeConfig
from litellm.types.services import ServiceLoggerPayload
from litellm.types.utils import StandardCallbackDynamicParams

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.types.integrations.arize import Protocol as _Protocol

    Protocol = _Protocol
    Span = _Span | Any
else:
    Protocol = Any
    Span = Any

_SUCCESS_SAMPLING_RATE_VAR: Final = "arize_success_sampling_rate"
_ERROR_SAMPLING_RATE_VAR: Final = "arize_error_sampling_rate"


class ArizeLogger(OpenTelemetry):
    """
    Arize logger that sends traces to an Arize endpoint.

    Creates its own dedicated TracerProvider so it can coexist with the
    generic ``otel`` callback (or any other OTEL-based integration) without
    fighting over the global ``opentelemetry.trace`` TracerProvider singleton.
    """

    def __init__(
        self,
        config: OpenTelemetryConfig | None = None,
        callback_name: str | None = None,
        tracer_provider: object | None = None,
        logger_provider: object | None = None,
        meter_provider: object | None = None,
        max_dynamic_tracer_providers: int = _MAX_DYNAMIC_TRACER_PROVIDERS,
        random_draw: Callable[[], float] | None = None,
    ) -> None:
        super().__init__(
            config=config,
            callback_name=callback_name,
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            max_dynamic_tracer_providers=max_dynamic_tracer_providers,
        )
        self._random_draw: Final[Callable[[], float]] = random_draw if random_draw is not None else random.random

    def _init_tracing(self, tracer_provider):
        """
        Override to always create a *private* TracerProvider for Arize.

        See ArizePhoenixLogger._init_tracing for full rationale.
        """
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.trace import SpanKind

        if tracer_provider is not None:
            self.tracer = tracer_provider.get_tracer("litellm")
            self.span_kind = SpanKind
            return

        provider: Final = TracerProvider(resource=self._get_litellm_resource(self.config))
        provider.add_span_processor(self._get_span_processor())
        self.tracer = provider.get_tracer("litellm")
        self.span_kind = SpanKind

    def _handle_success(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        if not self._should_export(kwargs, _SUCCESS_SAMPLING_RATE_VAR):
            return
        super()._handle_success(kwargs, response_obj, start_time, end_time)

    def _handle_failure(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        if not self._should_export(kwargs, _ERROR_SAMPLING_RATE_VAR):
            return
        super()._handle_failure(kwargs, response_obj, start_time, end_time)

    def _sampling_rate_for_request(self, kwargs: Mapping[str, object], var: str) -> float | None:
        dynamic_params: Final = kwargs.get("standard_callback_dynamic_params")
        if not isinstance(dynamic_params, Mapping):
            return None
        value: Final = dynamic_params.get(var)
        if value is None or value in ("", "None"):
            return None
        try:
            rate: Final = float(value)
        except (TypeError, ValueError):
            verbose_logger.warning(
                "ArizeLogger: %s value %r is not a number; exporting the request",
                var,
                value,
            )
            return None
        if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
            verbose_logger.warning(
                "ArizeLogger: %s value %r is outside 0.0..1.0; exporting the request",
                var,
                value,
            )
            return None
        return rate

    def _should_export(self, kwargs: dict[str, object], var: str) -> bool:
        rate: Final = self._sampling_rate_for_request(kwargs, var)
        if rate is None:
            return True
        otel_internal: Final = self._otel_internal_state(kwargs)
        key: Final = f"arize_sampled:{var}"
        cached: Final = otel_internal.get(key)
        if isinstance(cached, bool):
            return cached
        sampled: Final = rate > 0.0 and self._random_draw() <= rate
        otel_internal[key] = sampled
        if not sampled:
            verbose_logger.debug(
                "ArizeLogger: dropping request, %s rate %r rejected the draw",
                var,
                rate,
            )
        return sampled

    def _init_otel_logger_on_litellm_proxy(self):
        """
        Override: Arize should NOT overwrite the proxy's
        ``open_telemetry_logger``.  That attribute is reserved for the
        primary ``otel`` callback which handles proxy-level parent spans.
        """

    def set_attributes(self, span: Span, kwargs, response_obj: Any | None):
        ArizeLogger.set_arize_attributes(span, kwargs, response_obj)

    @staticmethod
    def set_arize_attributes(span: Span, kwargs, response_obj):
        _utils.set_attributes(span, kwargs, response_obj, ArizeOTELAttributes)

    @staticmethod
    def get_arize_config() -> ArizeConfig:
        """
        Helper function to get Arize configuration.

        Returns:
            ArizeConfig: A Pydantic model containing Arize configuration.

        Raises:
            ValueError: If required environment variables are not set.
        """
        space_id: Final = os.environ.get("ARIZE_SPACE_ID")
        space_key: Final = os.environ.get("ARIZE_SPACE_KEY")
        api_key: Final = os.environ.get("ARIZE_API_KEY")
        project_name: Final = os.environ.get("ARIZE_PROJECT_NAME")

        grpc_endpoint: Final = os.environ.get("ARIZE_ENDPOINT")
        http_endpoint: Final = os.environ.get("ARIZE_HTTP_ENDPOINT")

        endpoint = None
        protocol: Protocol = "otlp_grpc"

        if grpc_endpoint:
            protocol = "otlp_grpc"
            endpoint = grpc_endpoint
        elif http_endpoint:
            protocol = "otlp_http"
            endpoint = http_endpoint
        else:
            protocol = "otlp_grpc"
            endpoint = "https://otlp.arize.com/v1"

        return ArizeConfig(
            space_id=space_id,
            space_key=space_key,
            api_key=api_key,
            protocol=protocol,
            endpoint=endpoint,
            project_name=project_name,
        )

    async def async_service_success_hook(
        self,
        payload: ServiceLoggerPayload,
        parent_otel_span: Span | None = None,
        start_time: datetime | float | None = None,
        end_time: datetime | float | None = None,
        event_metadata: dict | None = None,
    ):
        """Arize is used mainly for LLM I/O tracing, sending router+caching metrics adds bloat to arize logs"""

    async def async_service_failure_hook(
        self,
        payload: ServiceLoggerPayload,
        error: str | None = "",
        parent_otel_span: Span | None = None,
        start_time: datetime | float | None = None,
        end_time: float | datetime | None = None,
        event_metadata: dict | None = None,
    ):
        """Arize is used mainly for LLM I/O tracing, sending router+caching metrics adds bloat to arize logs"""

    # def create_litellm_proxy_request_started_span(
    #     self,
    #     start_time: datetime,
    #     headers: dict,
    # ):
    #     """Arize is used mainly for LLM I/O tracing, sending Proxy Server Request adds bloat to arize logs"""
    #     pass

    async def async_health_check(self):
        """
        Performs a health check for Arize integration.

        Returns:
            dict: Health check result with status and message
        """
        try:
            config: Final = self.get_arize_config()

            if not config.space_id and not config.space_key:
                return {
                    "status": "unhealthy",
                    "error_message": "ARIZE_SPACE_ID or ARIZE_SPACE_KEY environment variable not set",
                }

            if not config.api_key:
                return {
                    "status": "unhealthy",
                    "error_message": "ARIZE_API_KEY environment variable not set",
                }

            return {
                "status": "healthy",
                "message": "Arize credentials are configured properly",
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error_message": f"Arize health check failed: {e}",
            }

    def construct_dynamic_otel_headers(
        self, standard_callback_dynamic_params: StandardCallbackDynamicParams
    ) -> dict | None:
        """
        Construct dynamic Arize headers from standard callback dynamic params

        This is used for team/key based logging.

        Returns:
            dict: A dictionary of dynamic Arize headers
        """
        dynamic_headers: Final = {}

        #########################################################
        # `arize-space-id` handling
        # the suggested param is `arize_space_key`
        #########################################################
        if standard_callback_dynamic_params.get("arize_space_id"):
            dynamic_headers["arize-space-id"] = standard_callback_dynamic_params.get("arize_space_id")
        if standard_callback_dynamic_params.get("arize_space_key"):
            dynamic_headers["arize-space-id"] = standard_callback_dynamic_params.get("arize_space_key")

        #########################################################
        # `api_key` handling
        #########################################################
        if standard_callback_dynamic_params.get("arize_api_key"):
            dynamic_headers["api_key"] = standard_callback_dynamic_params.get("arize_api_key")

        return dynamic_headers
