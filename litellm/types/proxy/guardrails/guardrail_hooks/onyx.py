from typing import Optional

from pydantic import Field

from .base import GuardrailConfigModel


class OnyxGuardrailConfigModel(GuardrailConfigModel):
    api_base: Optional[str] = Field(
        default=None,
        description="The URL of the Onyx Guard server. If not provided, the `ONYX_API_BASE` environment variable is checked.",
    )

    api_key: Optional[str] = Field(
        default=None,
        description="The API key for the Onyx Guard server. If not provided, the `ONYX_API_KEY` environment variable is checked.",
    )

    mcp_api_key: Optional[str] = Field(
        default=None,
        description="The API key for MCP Guard policies. If not provided, the `ONYX_MCP_API_KEY` environment variable is checked. MCP calls are skipped when not set.",
    )

    timeout: Optional[float] = Field(
        default=None,
        description="The timeout for the Onyx Guard server in seconds. If not provided, the `ONYX_TIMEOUT` environment variable is checked.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Onyx Guardrail"
