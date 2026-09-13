from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .base import GuardrailConfigModel


class ConductGuardrailConfigModelOptionalParams(BaseModel):
    workspace_id: str | None = Field(
        default=None,
        description="Conduct workspace id, sent as the X-Workspace-Id header. Env: CONDUCT_WORKSPACE_ID.",
    )
    tool_name: str | None = Field(
        default="llm_call",
        description="Conduct tool name the prompt is evaluated under. Match the tool your rules target.",
    )
    timeout: float | None = Field(
        default=8.0,
        gt=0.0,
        description="Timeout in seconds for the Conduct check.",
    )
    unreachable_fallback: Literal["fail_open", "fail_closed"] | None = Field(
        default="fail_closed",
        description="Behavior when Conduct is unreachable, times out, or rejects the token.",
    )


class ConductGuardrailConfigModel(GuardrailConfigModel[ConductGuardrailConfigModelOptionalParams]):
    api_key: str = Field(
        min_length=1,
        description="Conduct agent token. Env: CONDUCT_AGENT_TOKEN.",
    )
    api_base: str | None = Field(
        default="https://api.conductai.ai",
        description="Conduct API base URL. The MCP endpoint is derived as <api_base>/mcp.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Conduct Guard"
