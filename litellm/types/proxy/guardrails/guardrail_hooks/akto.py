from typing import Literal

from pydantic import BaseModel, Field

from .base import GuardrailConfigModel


class AktoGuardrailConfigModelOptionalParams(BaseModel):
    streaming_sampling_rate: int | None = Field(
        default=None,
        description=(
            "Check the streamed response every Nth chunk; the stream pauses at that chunk until Akto replies. "
            "1 checks every chunk. Default: 5."
        ),
    )


class AktoConfigModel(GuardrailConfigModel[AktoGuardrailConfigModelOptionalParams]):
    """
    Config for the Akto guardrail. Each mode checks the traffic with Akto, then blocks or masks it:
      pre_call      -> LLM request
      post_call     -> LLM response
      pre_mcp_call  -> MCP tool call
      post_mcp_call -> MCP tool result
    """

    akto_base_url: str | None = Field(
        default=None,
        description="Akto Guardrail API Base URL. Env: AKTO_GUARDRAIL_API_BASE.",
        json_schema_extra={
            "examples": [
                "http://localhost:9090",
                "https://akto-ingestion.example.com",
            ]
        },
    )

    akto_api_key: str | None = Field(
        default=None,
        description="API key for Akto. Env: AKTO_API_KEY.",
    )

    akto_account_id: str | None = Field(
        default=None,
        description="Akto account ID for multi-tenant deployments. Env: AKTO_ACCOUNT_ID. Default: '1000000'.",
    )

    akto_vxlan_id: str | None = Field(
        default=None,
        description="Akto VXLAN ID. Env: AKTO_VXLAN_ID. Default: '0'.",
    )

    context_source: Literal["ENDPOINT", "AGENTIC"] | None = Field(
        default=None,
        description="Akto context the traffic belongs to: 'ENDPOINT' (Atlas) or 'AGENTIC' (Argus). Default: AGENTIC.",
    )

    akto_metadata: dict | None = Field(
        default=None,
        description=(
            "JSON object sent to Akto. 'policy_name': comma-separated Akto policies to enforce (empty enforces all). "
            'Example: {"policy_name": "PII Strict, Secrets"}.'
        ),
    )

    guardrail_timeout: int | None = Field(
        default=None,
        description="HTTP timeout in seconds. Default: 5.",
    )

    file_guardrail_timeout: int | None = Field(
        default=None,
        description="HTTP timeout in seconds for checking attached files. Default: 10.",
    )

    unreachable_fallback: Literal["fail_closed", "fail_open"] = Field(
        default="fail_closed",
        description=(
            "What to do when Akto is unreachable, times out or errors. 'fail_closed' = block (default), "
            "'fail_open' = allow."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Akto"
