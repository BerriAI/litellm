from pydantic import BaseModel, Field

from .base import GuardrailConfigModel


class AgentGuardsGuardrailConfigModelOptionalParams(BaseModel):
    use_case: str | None = Field(
        default="check",
        description="The `use_case` sent to AgentGuards `/v1/guardrails/evaluate-input`. Defaults to `check`.",
    )
    tenant_id: str | None = Field(
        default=None,
        description="Sent as the `X-Tenant-ID` header when no API key is configured (local / self-hosted dev AgentGuards).",
    )
    fail_closed: bool | None = Field(
        default=False,
        description="If true, reject the request when AgentGuards is unreachable or returns an error. Defaults to false (fail open).",
    )


class AgentGuardsGuardrailConfigModel(GuardrailConfigModel[AgentGuardsGuardrailConfigModelOptionalParams]):
    api_key: str | None = Field(
        default=None,
        description="The AgentGuards API key (sent as `X-API-Key: ag_...`). If not provided, the `AGENTGUARDS_API_KEY` environment variable is checked.",
    )
    api_base: str | None = Field(
        default=None,
        description="The AgentGuards API base URL. Default is https://prod.agentguards.co. Also checks the `AGENTGUARDS_API_BASE` environment variable.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "AgentGuards"
