import base64
import os
from typing import TYPE_CHECKING, Any, Union
from urllib.parse import quote

from litellm._logging import verbose_logger
from litellm.integrations.arize import _utils
from litellm.types.integrations.langfuse_otel import (
    LangfuseOtelConfig,
    LangfuseSpanAttributes,
)

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.integrations.opentelemetry import (
        OpenTelemetryConfig as _OpenTelemetryConfig,
    )
    from litellm.types.integrations.arize import Protocol as _Protocol

    Protocol = _Protocol
    OpenTelemetryConfig = _OpenTelemetryConfig
    Span = Union[_Span, Any]
else:
    Protocol = Any
    OpenTelemetryConfig = Any
    Span = Any


LANGFUSE_CLOUD_EU_ENDPOINT = "https://cloud.langfuse.com/api/public/otel"
LANGFUSE_CLOUD_US_ENDPOINT = "https://us.cloud.langfuse.com/api/public/otel"



class LangfuseOtelLogger:
    @staticmethod
    def set_langfuse_otel_attributes(span: Span, kwargs, response_obj):
        """
        Sets OpenTelemetry span attributes for Langfuse observability.
        Uses the same attribute setting logic as Arize Phoenix for consistency.
        """
        _utils.set_attributes(span, kwargs, response_obj)

        #########################################################
        # Set Langfuse specific attributes eg Langfuse Environment
        #########################################################
        LangfuseOtelLogger._set_langfuse_specific_attributes(
            span=span,
            kwargs=kwargs
        )
        return
    
    @staticmethod
    def _set_langfuse_specific_attributes(span: Span, kwargs):
        """
        Sets Langfuse specific attributes to the span.
        """
        from litellm.integrations.arize._utils import safe_set_attribute
        langfuse_environment = os.environ.get("LANGFUSE_TRACING_ENVIRONMENT", None)
        if langfuse_environment:
            safe_set_attribute(
                span=span,
                key=LangfuseSpanAttributes.LANGFUSE_ENVIRONMENT.value, 
                value=langfuse_environment
            )

    @staticmethod
    def get_langfuse_otel_config() -> LangfuseOtelConfig:
        """
        Retrieves the Langfuse OpenTelemetry configuration based on environment variables.

        Environment Variables:
            LANGFUSE_PUBLIC_KEY: Required. Langfuse public key for authentication.
            LANGFUSE_SECRET_KEY: Required. Langfuse secret key for authentication.
            LANGFUSE_HOST: Optional. Custom Langfuse host URL. Defaults to US cloud.

        Returns:
            LangfuseOtelConfig: A Pydantic model containing Langfuse OTEL configuration.

        Raises:
            ValueError: If required keys are missing.
        """
        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", None)
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY", None)

        if not public_key or not secret_key:
            raise ValueError(
                "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set for Langfuse OpenTelemetry integration."
            )

        # Determine endpoint - default to US cloud
        langfuse_host = os.environ.get("LANGFUSE_HOST", None)

        if langfuse_host:
            # If LANGFUSE_HOST is provided, construct OTEL endpoint from it
            if not langfuse_host.startswith("http"):
                langfuse_host = "https://" + langfuse_host
            endpoint = f"{langfuse_host.rstrip('/')}/api/public/otel"
            verbose_logger.debug(f"Using Langfuse OTEL endpoint from host: {endpoint}")
        else:
            # Default to US cloud endpoint
            endpoint = LANGFUSE_CLOUD_US_ENDPOINT
            verbose_logger.debug(f"Using Langfuse US cloud endpoint: {endpoint}")

        # Create Basic Auth header
        auth_string = f"{public_key}:{secret_key}"
        auth_header = base64.b64encode(auth_string.encode()).decode()
        # URL encode the entire header value as required by OpenTelemetry specification
        otlp_auth_headers = f"Authorization={quote(f'Basic {auth_header}')}"

        # Set standard OTEL environment variables
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = endpoint
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = otlp_auth_headers

        return LangfuseOtelConfig(
            otlp_auth_headers=otlp_auth_headers, protocol="otlp_http"
        )
