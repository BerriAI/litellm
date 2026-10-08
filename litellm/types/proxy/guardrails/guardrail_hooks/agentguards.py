from pydantic import BaseModel, ConfigDict, Field

from .base import GuardrailConfigModel


class AgentGuardsGuardrailConfigModelOptionalParams(BaseModel):
    model_config = ConfigDict(frozen=True)


class AgentGuardsGuardrailConfigModel(GuardrailConfigModel[AgentGuardsGuardrailConfigModelOptionalParams]):
    """
    Config for the AgentGuards guardrail:
      pre_call  -> the request is screened by POST /v1/guardrails/evaluate-input
      post_call -> the response is validated by POST /v1/outputs/validate
    Unreachable-service behaviour follows `unreachable_fallback` (default fail_closed).
    """

    api_key: str | None = Field(
        default=None,
        description="AgentGuards API key, sent as `X-API-Key`. Env: AGENTGUARDS_API_KEY.",
    )
    api_base: str | None = Field(
        default=None,
        description="AgentGuards API base URL. Default: https://prod.agentguards.co. Env: AGENTGUARDS_API_BASE.",
    )
    agentguards_use_case: str | None = Field(
        default=None,
        description="The `use_case` sent with each input check (selects the AgentGuards policy). Default: `check`.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "AgentGuards"
