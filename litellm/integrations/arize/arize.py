"""
arize AI is OTEL compatible

this file has Arize ai specific helper functions
"""

import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final

from litellm.integrations.arize import _utils
from litellm.integrations.arize._utils import ArizeOTELAttributes
from litellm.integrations.opentelemetry import OpenTelemetry
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


class ArizeLogger(OpenTelemetry):
    """
    Arize logger that sends traces to an Arize endpoint.

    Creates its own dedicated TracerProvider so it can coexist with the
    generic ``otel`` callback (or any other OTEL-based integration) without
    fighting over the global ``opentelemetry.trace`` TracerProvider singleton.
    """

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
