import os
from typing import TYPE_CHECKING, Any
from litellm.integrations.arize import _utils
from litellm._logging import verbose_logger
from litellm.types.integrations.arize_phoenix import ArizePhoenixConfig
from opentelemetry.trace import Span as Span

if TYPE_CHECKING:
    from .opentelemetry import OpenTelemetryConfig as _OpenTelemetryConfig

    OpenTelemetryConfig = _OpenTelemetryConfig
else:
    OpenTelemetryConfig = Any

import os

class ArizePhoenixLogger:
    @staticmethod
    def set_arize_phoenix_attributes(span: Span, kwargs, response_obj):
        _utils.set_attributes(span, kwargs, response_obj)
        return

    @staticmethod
    def get_arize_phoenix_config() -> ArizePhoenixConfig:
        """
        Retrieves the Arize configuration based on environment variables.

        Returns:
            ArizePhoenixConfig: A Pydantic model containing Arize configuration.
        """
        api_key = os.environ.get("PHOENIX_API_KEY")
        grpc_endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
        http_endpoint = os.environ.get("PHOENIX_COLLECTOR_HTTP_ENDPOINT")

        endpoint = None
        if grpc_endpoint is not None:
            endpoint = grpc_endpoint
            protocol = "grpc"
        elif http_endpoint is not None:
            endpoint = http_endpoint
            protocol = "http"
        else:
            endpoint = "https://app.phoenix.arize.com/v1/traces"
            protocol = "grpc"       
            verbose_logger.debug(
                "No PHOENIX_COLLECTOR_ENDPOINT or PHOENIX_COLLECTOR_HTTP_ENDPOINT found, using default endpoint: https://app.phoenix.arize.com/v1/traces"
            )

        otlp_auth_headers = None
        if endpoint.startswith("https://app.phoenix.arize.com/v1/traces"):
            otlp_auth_headers = f"api_key={api_key}"
        else:
            otlp_auth_headers = f"Authorization=Bearer {api_key}"
            
        return ArizePhoenixConfig(
            otlp_auth_headers=otlp_auth_headers,
            protocol=protocol,
            endpoint=endpoint
        )

