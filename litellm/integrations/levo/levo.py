import os
from typing import TYPE_CHECKING, Any, Final

from litellm.integrations.opentelemetry import OpenTelemetry

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.integrations.opentelemetry import (
        OpenTelemetryConfig as _OpenTelemetryConfig,
    )
    from litellm.types.integrations.arize import Protocol as _Protocol

    Protocol = _Protocol
    OpenTelemetryConfig = _OpenTelemetryConfig
    Span = _Span | Any
else:
    Protocol = Any
    OpenTelemetryConfig = Any
    Span = Any


class LevoConfig:
    """Configuration for Levo OTLP integration."""

    def __init__(
        self,
        otlp_auth_headers: str | None,
        protocol: Protocol,
        endpoint: str,
    ):
        self.otlp_auth_headers = otlp_auth_headers
        self.protocol = protocol
        self.endpoint = endpoint


class LevoLogger(OpenTelemetry):
    """Levo Logger that extends OpenTelemetry for OTLP integration."""

    @staticmethod
    def get_levo_config() -> LevoConfig:
        """
        Retrieves the Levo configuration based on environment variables.

        Returns:
            LevoConfig: Configuration object containing Levo OTLP settings.

        Raises:
            ValueError: If required environment variables are missing.
        """
        # Required environment variables
        api_key: Final = os.environ.get("LEVOAI_API_KEY", None)
        org_id: Final = os.environ.get("LEVOAI_ORG_ID", None)
        workspace_id: Final = os.environ.get("LEVOAI_WORKSPACE_ID", None)
        collector_url: Final = os.environ.get("LEVOAI_COLLECTOR_URL", None)

        # Validate required env vars
        if not api_key:
            raise ValueError("LEVOAI_API_KEY environment variable is required for Levo integration.")
        if not org_id:
            raise ValueError("LEVOAI_ORG_ID environment variable is required for Levo integration.")
        if not workspace_id:
            raise ValueError("LEVOAI_WORKSPACE_ID environment variable is required for Levo integration.")
        if not collector_url:
            raise ValueError(
                "LEVOAI_COLLECTOR_URL environment variable is required for Levo integration. "
                "Please contact Levo support to get your collector URL."
            )

        # Use collector URL exactly as provided by the user
        endpoint: Final = collector_url
        protocol: Final[Protocol] = "otlp_http"

        # Build OTLP headers string
        # Format: Authorization=Bearer {api_key},x-levo-organization-id={org_id},x-levo-workspace-id={workspace_id}
        headers_parts: Final = [f"Authorization=Bearer {api_key}"]
        headers_parts.append(f"x-levo-organization-id={org_id}")
        headers_parts.append(f"x-levo-workspace-id={workspace_id}")

        otlp_auth_headers: Final = ",".join(headers_parts)

        return LevoConfig(
            otlp_auth_headers=otlp_auth_headers,
            protocol=protocol,
            endpoint=endpoint,
        )

    async def async_health_check(self):
        """
        Health check for Levo integration.

        Returns:
            dict: Health status with status and message/error_message keys.
        """
        try:
            config: Final = self.get_levo_config()

            if not config.otlp_auth_headers:
                return {
                    "status": "unhealthy",
                    "error_message": "LEVOAI_API_KEY environment variable not set",
                }

            return {
                "status": "healthy",
                "message": "Levo credentials are configured properly",
            }
        except ValueError as e:
            return {
                "status": "unhealthy",
                "error_message": str(e),
            }
