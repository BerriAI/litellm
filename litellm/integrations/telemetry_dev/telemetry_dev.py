import os
from typing import TYPE_CHECKING, Any

from litellm.integrations.opentelemetry import OpenTelemetry

if TYPE_CHECKING:
    from litellm.integrations.opentelemetry import OpenTelemetryConfig
    from litellm.types.integrations.arize import Protocol
else:
    Protocol = Any
    OpenTelemetryConfig = Any


class TelemetryDevConfig:
    """Configuration for the telemetry.dev OTLP integration."""

    def __init__(
        self,
        otlp_auth_headers: str,
        protocol: Protocol,
        endpoint: str,
    ):
        self.otlp_auth_headers = otlp_auth_headers
        self.protocol = protocol
        self.endpoint = endpoint


class TelemetryDevLogger(OpenTelemetry):
    """telemetry.dev logger that extends OpenTelemetry for OTLP export."""

    @staticmethod
    def get_telemetry_dev_config() -> TelemetryDevConfig:
        api_key = os.environ.get("TELEMETRY_DEV_API_KEY")
        if not api_key:
            raise ValueError(
                "TELEMETRY_DEV_API_KEY environment variable is required for the telemetry.dev integration."
            )

        base_url = os.environ.get(
            "TELEMETRY_DEV_BASE_URL", "https://ingest.telemetry.dev"
        ).rstrip("/")

        return TelemetryDevConfig(
            otlp_auth_headers=f"Authorization=Bearer {api_key}",
            protocol="otlp_http",
            endpoint=f"{base_url}/v1/traces",
        )
