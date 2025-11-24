import os
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


ARIZE_HOSTED_PHOENIX_ENDPOINT = "https://otlp.arize.com/v1/traces"


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

        collector_endpoint = os.environ.get("PHOENIX_COLLECTOR_HTTP_ENDPOINT", None)

        if not collector_endpoint:
            grpc_endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", None)
            http_endpoint = os.environ.get("PHOENIX_COLLECTOR_HTTP_ENDPOINT", None)
            collector_endpoint = http_endpoint or grpc_endpoint

        endpoint = None
        protocol: Protocol = "otlp_http"

        if collector_endpoint:
            # Parse the endpoint to determine protocol
            if collector_endpoint.startswith("grpc://") or (":4317" in collector_endpoint and "/v1/traces" not in collector_endpoint):
                endpoint = collector_endpoint
                protocol = "otlp_grpc"
            else:
                # Phoenix Cloud endpoints (app.phoenix.arize.com) include the space in the URL
                if "app.phoenix.arize.com" in collector_endpoint:
                    endpoint = collector_endpoint
                    protocol = "otlp_http"
                # For other HTTP endpoints, ensure they have the correct path
                elif "/v1/traces" not in collector_endpoint:
                    if collector_endpoint.endswith("/v1"):
                        endpoint = collector_endpoint + "/traces"
                    elif collector_endpoint.endswith("/"):
                        endpoint = f"{collector_endpoint}v1/traces"
                    else:
                        endpoint = f"{collector_endpoint}/v1/traces"
                else:
                    endpoint = collector_endpoint
                protocol = "otlp_http"
        else:
            # If no endpoint specified, self hosted phoenix
            endpoint = "http://localhost:6006/v1/traces"
            protocol = "otlp_http"
            verbose_logger.debug(
                f"No PHOENIX_COLLECTOR_ENDPOINT found, using default local Phoenix endpoint: {endpoint}"
            )

        otlp_auth_headers = None
        if api_key is not None:
            otlp_auth_headers = f"Authorization=Bearer {api_key}"
        elif "app.phoenix.arize.com" in endpoint:
            # Phoenix Cloud requires an API key
            raise ValueError(
                "PHOENIX_API_KEY must be set when using Phoenix Cloud (app.phoenix.arize.com)."
            )

        project_name = os.environ.get("PHOENIX_PROJECT_NAME", "litellm-project")

        return ArizePhoenixConfig(
            otlp_auth_headers=otlp_auth_headers,
            protocol=protocol,
            endpoint=endpoint,
            project_name=project_name,
        )
