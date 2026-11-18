from typing import Final, Literal

from pydantic import Field

from .base import GuardrailConfigModel

AGENT_365_PROD_API_BASE: Final = "https://agent365.svc.cloud.microsoft"
AGENT_365_PROD_RESOURCE_APP_ID: Final = "ea9ffc3e-8a23-4a7d-836d-234d7c7565c1"
AGENT_365_SCOPE_NAME: Final = "ThreatProtection.Evaluate.All"


class Agent365GuardrailConfigModel(GuardrailConfigModel):
    tenant_id: str | None = Field(
        default=None,
        description=(
            "Entra tenant id used for the On-Behalf-Of token exchange. "
            "Falls back to the AGENT365_TENANT_ID environment variable."
        ),
    )

    client_id: str | None = Field(
        default=None,
        description=(
            "Client id of the gateway's Entra app registration (a confidential client). "
            "Falls back to the AGENT365_CLIENT_ID environment variable."
        ),
    )

    client_secret: str | None = Field(
        default=None,
        description=(
            "Client secret of the gateway's Entra app registration, used to perform the "
            "On-Behalf-Of exchange. Falls back to the AGENT365_CLIENT_SECRET environment variable."
        ),
    )

    unreachable_fallback: Literal["fail_closed", "fail_open"] = Field(
        default="fail_closed",
        description=(
            "Behavior when Agent 365 or Entra is unreachable, times out, returns 5xx, skips the evaluation, or "
            "rejects the gateway's own client credentials. 'fail_closed' (default) blocks the tool call with HTTP 503. "
            "'fail_open' allows it, logs an error and records it as Unscanned in the logs and OpenTelemetry. "
            "Policy blocks, 4xx rejections, throttling and a rejected caller token always block."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Microsoft Agent 365"
