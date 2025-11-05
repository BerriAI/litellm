import os
import urllib.parse
from typing import TYPE_CHECKING, Any, Union

from litellm._logging import verbose_logger
from litellm.integrations.arize import _utils
from litellm.integrations.arize._utils import ArizeOTELAttributes
from litellm.types.integrations.arize_phoenix import ArizePhoenixConfig

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.types.integrations.arize import Protocol as _Protocol

    from .opentelemetry import OpenTelemetryConfig as _OpenTelemetryConfig

    Protocol = _Protocol
    OpenTelemetryConfig = _OpenTelemetryConfig
    Span = Union[_Span, Any]
else:
    Protocol = Any
    OpenTelemetryConfig = Any
    Span = Any


ARIZE_HOSTED_PHOENIX_ENDPOINT = "https://app.phoenix.arize.com/v1/traces"


class ArizePhoenixLogger:
    @staticmethod
    def set_arize_phoenix_attributes(span: Span, kwargs, response_obj):
        _utils.set_attributes(span, kwargs, response_obj, ArizeOTELAttributes)
        return

    @staticmethod
    def get_arize_phoenix_config() -> ArizePhoenixConfig:
        """
        Retrieves the Arize Phoenix configuration based on environment variables.

        Returns:
            ArizePhoenixConfig: A Pydantic model containing Arize Phoenix configuration.
        """
        api_key = os.environ.get("PHOENIX_API_KEY", None)
        
        # Check for Phoenix collector endpoint (updated to match Phoenix's environment variable names)
        # Phoenix uses PHOENIX_COLLECTOR_ENDPOINT as the primary variable
        collector_endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", None)
        
        # Fallback to legacy LiteLLM environment variables for backward compatibility
        if not collector_endpoint:
            grpc_endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", None)
            http_endpoint = os.environ.get("PHOENIX_COLLECTOR_HTTP_ENDPOINT", None)
            collector_endpoint = http_endpoint or grpc_endpoint

        endpoint = None
        protocol: Protocol = "otlp_http"

        if collector_endpoint:
            # Parse the endpoint to determine protocol
            if collector_endpoint.startswith("grpc://") or (":4317" in collector_endpoint and not "/v1/traces" in collector_endpoint):
                endpoint = collector_endpoint
                protocol = "otlp_grpc"
            else:
                # Ensure HTTP endpoints have the correct path
                if not collector_endpoint.endswith("/v1/traces"):
                    if collector_endpoint.endswith("/"):
                        endpoint = f"{collector_endpoint}v1/traces"
                    else:
                        endpoint = f"{collector_endpoint}/v1/traces"
                else:
                    endpoint = collector_endpoint
                protocol = "otlp_http"
        else:
            endpoint = ARIZE_HOSTED_PHOENIX_ENDPOINT
            protocol = "otlp_http"
            verbose_logger.debug(
                f"No PHOENIX_COLLECTOR_ENDPOINT found, using default Arize hosted Phoenix endpoint: {ARIZE_HOSTED_PHOENIX_ENDPOINT}"
            )

        otlp_auth_headers = None
        # Phoenix now uses standard Authorization: Bearer <token> format for both hosted and self-hosted
        if api_key is not None:
            # Use standard Authorization header format that matches Phoenix's current API
            otlp_auth_headers = f"Authorization=Bearer {api_key}"
        elif endpoint == ARIZE_HOSTED_PHOENIX_ENDPOINT:
            # Arize hosted Phoenix requires API key
            raise ValueError(
                "PHOENIX_API_KEY must be set when using the Arize hosted Phoenix endpoint."
            )

        return ArizePhoenixConfig(
            otlp_auth_headers=otlp_auth_headers, protocol=protocol, endpoint=endpoint
        )
