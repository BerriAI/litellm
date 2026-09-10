from typing import Final, Literal, TypeAlias

from pydantic import Field

from .base import GuardrailConfigModel

AGENT_365_PROD_API_BASE: Final = "https://agent365.svc.cloud.microsoft"
AGENT_365_PROD_RESOURCE_APP_ID: Final = "ea9ffc3e-8a23-4a7d-836d-234d7c7565c1"
AGENT_365_SCOPE_NAME: Final = "ThreatProtection.Evaluate.All"


Agent365AuthMode: TypeAlias = Literal["on_behalf_of", "agent_identity"]


class Agent365GuardrailConfigModel(GuardrailConfigModel):
    auth_mode: Agent365AuthMode | None = Field(
        default=None,
        description=(
            "How the guardrail authenticates to Agent 365. 'on_behalf_of' (default) exchanges the caller's "
            "incoming Entra bearer token, so every MCP caller must present one. 'agent_identity' uses a "
            "Microsoft Entra Agent ID: client_id/client_secret are the agent identity blueprint's, and the "
            "guardrail mints the agent user's token itself, so callers need no Entra token."
        ),
    )

    agent_identity_client_id: str | None = Field(
        default=None,
        description=(
            "Client id of the Entra agent identity created from the blueprint. Required when "
            "auth_mode is 'agent_identity'. Falls back to the AGENT365_AGENT_IDENTITY_CLIENT_ID environment variable."
        ),
    )

    agent_user_upn: str | None = Field(
        default=None,
        description=(
            "User principal name of the agent user account parented by the agent identity; Defender evaluates "
            "and audits as this account. Required when auth_mode is 'agent_identity'. "
            "Falls back to the AGENT365_AGENT_USER_UPN environment variable."
        ),
    )

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
            "Client id of the gateway's Entra app registration (a confidential client), or of the agent "
            "identity blueprint when auth_mode is 'agent_identity'. "
            "Falls back to the AGENT365_CLIENT_ID environment variable."
        ),
    )

    client_secret: str | None = Field(
        default=None,
        description=(
            "Client secret of the gateway's Entra app registration (or of the agent identity blueprint), "
            "used to perform the token exchange. Falls back to the AGENT365_CLIENT_SECRET environment variable."
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
