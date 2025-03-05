import os
from typing import TYPE_CHECKING, Any
from litellm.integrations.arize import _utils
from litellm._logging import verbose_logger
from litellm.types.integrations.arize_phoenix import ArizePhoenixConfig

if TYPE_CHECKING:
    from .opentelemetry import OpenTelemetryConfig as _OpenTelemetryConfig
    from litellm.types.integrations.arize import Protocol as _Protocol
    from opentelemetry.trace import Span as _Span

    Protocol = _Protocol
    OpenTelemetryConfig = _OpenTelemetryConfig
    Span = _Span
else:
    Protocol = Any
    OpenTelemetryConfig = Any
    Span = Any


ARIZE_HOSTED_PHOENIX_ENDPOINT = "https://app.phoenix.arize.com/v1/traces"

class ArizePhoenixLogger:
    @staticmethod
    def set_arize_phoenix_attributes(span: Span, kwargs, response_obj):
        _utils.set_attributes(span, kwargs, response_obj)
        return

    @staticmethod
    def get_arize_phoenix_config() -> ArizePhoenixConfig:
        """
        Retrieves the Arize Phoenix configuration based on environment variables.

        Returns:
            ArizePhoenixConfig: A Pydantic model containing Arize Phoenix configuration.
        """
        api_key = os.environ.get("PHOENIX_API_KEY", None)
        grpc_endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", None)
        http_endpoint = os.environ.get("PHOENIX_COLLECTOR_HTTP_ENDPOINT", None)

        endpoint = None
        protocol: Protocol = "otlp_http"

        if http_endpoint:
            endpoint = http_endpoint
            protocol = "otlp_http"
        elif grpc_endpoint:
            endpoint = grpc_endpoint
            protocol = "otlp_grpc"
        else:
            endpoint = ARIZE_HOSTED_PHOENIX_ENDPOINT
            protocol = "otlp_http"       
            verbose_logger.debug(
                f"No PHOENIX_COLLECTOR_ENDPOINT or PHOENIX_COLLECTOR_HTTP_ENDPOINT found, using default endpoint with http: {ARIZE_HOSTED_PHOENIX_ENDPOINT}"
            )

        otlp_auth_headers = None
        # If the endpoint is the Arize hosted Phoenix endpoint, use the api_key as the auth header as currently it is uses
        # a slightly different auth header format than self hosted phoenix
        if endpoint == ARIZE_HOSTED_PHOENIX_ENDPOINT: 
            if api_key is None:
                raise ValueError("PHOENIX_API_KEY must be set when the Arize hosted Phoenix endpoint is used.")
            otlp_auth_headers = f"api_key={api_key}"
        elif api_key is not None:
            # api_key/auth is optional for self hosted phoenix
            otlp_auth_headers = f"Authorization=Bearer {api_key}"

        return ArizePhoenixConfig(
            otlp_auth_headers=otlp_auth_headers,
            protocol=protocol,
            endpoint=endpoint
        )

