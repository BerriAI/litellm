"""
arize AI is OTEL compatible

this file has Arize ai specific helper functions
"""
import os

from typing import TYPE_CHECKING, Any
from litellm.integrations.arize import _utils
from litellm.types.integrations.arize import ArizeConfig

if TYPE_CHECKING:
    from litellm.types.integrations.arize import Protocol as _Protocol
    from opentelemetry.trace import Span as _Span

    Protocol = _Protocol
    Span = _Span
else:
    Protocol = Any
    Span = Any
    


class ArizeLogger:

    @staticmethod
    def set_arize_attributes(span: Span, kwargs, response_obj):
        _utils.set_attributes(span, kwargs, response_obj)
        return
    

    @staticmethod
    def get_arize_config() -> ArizeConfig:
        """
        Helper function to get Arize configuration.

        Returns:
            ArizeConfig: A Pydantic model containing Arize configuration.

        Raises:
            ValueError: If required environment variables are not set.
        """
        space_key = os.environ.get("ARIZE_SPACE_KEY")
        api_key = os.environ.get("ARIZE_API_KEY")

        if not space_key:
            raise ValueError("ARIZE_SPACE_KEY not found in environment variables")
        if not api_key:
            raise ValueError("ARIZE_API_KEY not found in environment variables")

        grpc_endpoint = os.environ.get("ARIZE_ENDPOINT")
        http_endpoint = os.environ.get("ARIZE_HTTP_ENDPOINT")

        endpoint = None
        protocol: Protocol = "otlp_grpc"

        if grpc_endpoint:
            protocol="otlp_grpc"
            endpoint=grpc_endpoint
        elif http_endpoint:
            protocol="otlp_http"
            endpoint=http_endpoint
        else:
            protocol="otlp_grpc"
            endpoint = "https://otlp.arize.com/v1"

        return ArizeConfig(
            space_key=space_key,
            api_key=api_key,
            protocol=protocol,
            endpoint=endpoint,
        )


