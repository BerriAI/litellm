from typing import Final

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

    api_base: str | None = Field(
        default=None,
        description=(
            "Base URL of the Microsoft Agent 365 tool-evaluation endpoint. "
            f"Defaults to the production endpoint {AGENT_365_PROD_API_BASE}. "
            "Falls back to the AGENT365_API_BASE environment variable."
        ),
    )

    resource_app_id: str | None = Field(
        default=None,
        description=(
            "Application id of the Agent 365 resource the OBO token is minted for. "
            f"Defaults to the production resource {AGENT_365_PROD_RESOURCE_APP_ID}; "
            "the Test and PreProd environments use a different id. "
            "Falls back to the AGENT365_RESOURCE_APP_ID environment variable."
        ),
    )

    agent_id: str | None = Field(
        default=None,
        description=(
            "Agent identity reported to Agent 365 with every tool evaluation. "
            "When unset, the caller's key alias is used."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Microsoft Agent 365"
