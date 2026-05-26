"""Typed configuration for the LiteLLM OpenTelemetry instrumentation.

This is the only module that reads OTel-related environment variables. It has no
``opentelemetry`` import so it is safe to load without the SDK installed.
"""

import os
from typing import List, Optional

from pydantic import BaseModel, Field

from litellm.integrations.otel.semconv import (
    BAGGAGE_PROMOTED_KEYS,
    DEFAULT_BAGGAGE_METADATA_KEYS,
)

#: Master feature flag. The new engine is inert until this is truthy, so the
#: package can ship fully built without touching the existing OTel code path.
OTEL_V2_ENV = "LITELLM_OTEL_V2"

_TRUE = {"1", "true", "yes", "on", "y", "t"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE


def is_otel_v2_enabled() -> bool:
    """Whether the new instrumentation should be active. Off unless opted in."""
    return _env_bool(OTEL_V2_ENV, default=False)


class CaptureMessageContent(str):
    """Sentinel namespace for ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``."""

    NO_CONTENT = "no_content"
    SPAN_ONLY = "span_only"
    EVENT_ONLY = "event_only"
    SPAN_AND_EVENT = "span_and_event"


class OpenTelemetryV2Config(BaseModel):
    """Fully typed OTel config. Construct via :meth:`from_env` or explicitly."""

    exporter: str = "console"
    endpoint: Optional[str] = None
    headers: Optional[str] = None
    service_name: str = "litellm"
    deployment_environment: Optional[str] = None

    enable_metrics: bool = False
    capture_message_content: str = CaptureMessageContent.NO_CONTENT
    ignore_context_propagation: bool = False

    #: Emit legacy attribute keys / span names alongside canonical ones during the
    #: deprecation window. Defaults on so existing dashboards keep working.
    legacy_compat: bool = True

    #: Bounded allowlists for Baggage-based promotion of request-scoped identity
    #: onto every span (see ``providers.LiteLLMBaggageSpanProcessor``).
    baggage_promoted_keys: List[str] = Field(
        default_factory=lambda: list(BAGGAGE_PROMOTED_KEYS)
    )
    baggage_metadata_keys: List[str] = Field(
        default_factory=lambda: list(DEFAULT_BAGGAGE_METADATA_KEYS)
    )

    @classmethod
    def from_env(cls) -> "OpenTelemetryV2Config":
        exporter = (
            os.getenv("OTEL_EXPORTER")
            or os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL")
            or "console"
        )
        endpoint = os.getenv("OTEL_ENDPOINT") or os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT"
        )
        # An endpoint with no explicit exporter implies OTLP/HTTP, matching the
        # behavior of the legacy integration.
        if endpoint and exporter == "console":
            exporter = "otlp_http"
        return cls(
            exporter=exporter,
            endpoint=endpoint,
            headers=os.getenv("OTEL_HEADERS")
            or os.getenv("OTEL_EXPORTER_OTLP_HEADERS"),
            service_name=os.getenv("OTEL_SERVICE_NAME") or "litellm",
            deployment_environment=os.getenv("OTEL_ENVIRONMENT_NAME"),
            enable_metrics=_env_bool(
                "LITELLM_OTEL_INTEGRATION_ENABLE_METRICS", default=False
            ),
            capture_message_content=os.getenv(
                "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
            )
            or CaptureMessageContent.NO_CONTENT,
            ignore_context_propagation=_env_bool(
                "OTEL_IGNORE_CONTEXT_PROPAGATION", default=False
            ),
            legacy_compat=_env_bool("LITELLM_OTEL_LEGACY_COMPAT", default=True),
        )
