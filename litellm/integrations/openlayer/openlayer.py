import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from litellm.integrations.opentelemetry import OpenTelemetry
from litellm.types.integrations.base_health_check import IntegrationHealthCheckStatus

if TYPE_CHECKING:
    from litellm.types.integrations.arize import Protocol

OPENLAYER_DEFAULT_ENDPOINT: Final = "https://api.openlayer.com/v1/otel"


@dataclass(frozen=True, slots=True)
class OpenlayerConfig:
    """Resolved Openlayer OTLP settings."""

    otlp_auth_headers: str
    protocol: "Protocol"
    endpoint: str


class OpenlayerLogger(OpenTelemetry):
    """Openlayer logger that extends OpenTelemetry for OTLP integration."""

    @staticmethod
    def get_openlayer_config() -> OpenlayerConfig:
        """Build the Openlayer OTLP configuration from the environment.

        ``OPENLAYER_OTEL_ENDPOINT`` overrides the managed endpoint for
        self-hosted deployments. The endpoint is a base URL: the ``/v1/traces``
        signal path is appended, idempotently, by the exporter setup on both the
        v1 and v2 paths.

        Returns:
            OpenlayerConfig: endpoint, protocol and OTLP auth headers.

        Raises:
            ValueError: If a required environment variable is missing.
        """
        api_key: Final = os.environ.get("OPENLAYER_API_KEY", None)
        pipeline_id: Final = os.environ.get("OPENLAYER_INFERENCE_PIPELINE_ID", None)
        endpoint: Final = os.environ.get("OPENLAYER_OTEL_ENDPOINT") or OPENLAYER_DEFAULT_ENDPOINT

        if not api_key:
            raise ValueError(
                "OPENLAYER_API_KEY environment variable is required for the "
                "Openlayer integration. Find it under Workspace settings, API keys."
            )
        if not pipeline_id:
            raise ValueError(
                "OPENLAYER_INFERENCE_PIPELINE_ID environment variable is required "
                "for the Openlayer integration. It selects the inference pipeline "
                "traces are published to."
            )

        protocol: Final[Protocol] = "otlp_http"
        otlp_auth_headers: Final = ",".join(
            (
                f"Authorization=Bearer {api_key}",
                f"x-bt-parent=pipeline_id:{pipeline_id}",
            )
        )

        return OpenlayerConfig(
            otlp_auth_headers=otlp_auth_headers,
            protocol=protocol,
            endpoint=endpoint,
        )

    async def async_health_check(self) -> IntegrationHealthCheckStatus:
        """Report whether the Openlayer credentials are configured."""
        try:
            self.get_openlayer_config()
        except ValueError as e:
            return IntegrationHealthCheckStatus(status="unhealthy", error_message=str(e))

        return IntegrationHealthCheckStatus(status="healthy", error_message=None)
