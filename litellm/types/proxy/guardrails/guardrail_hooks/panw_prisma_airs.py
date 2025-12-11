from typing import Literal, Optional

from pydantic import Field

from .base import GuardrailConfigModel


class PanwPrismaAirsGuardrailConfigModel(GuardrailConfigModel):
    api_key: Optional[str] = Field(
        default=None,
        description="The API key for the PANW Prisma AIRS guardrail. If not provided, the `PANW_PRISMA_AIRS_API_KEY` environment variable is checked.",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="The API base for the PANW Prisma AIRS guardrail. Defaults to https://service.api.aisecurity.paloaltonetworks.com. If not provided, the `PANW_PRISMA_AIRS_API_BASE` environment variable is checked.",
    )

    profile_name: Optional[str] = Field(
        default=None,
        description="PANW Prisma AIRS security profile name configured in Strata Cloud Manager. Optional if API key has a linked profile.",
    )

    app_name: Optional[str] = Field(
        default=None,
        description="Application name for tracking this LiteLLM instance in Prisma AIRS analytics and dashboards. Defaults to 'LiteLLM' if not specified.",
    )

    mask_on_block: bool = Field(
        default=False,
        description="Backwards compatible flag that enables both request and response masking. When True, enables both mask_request_content and mask_response_content.",
    )

    mask_request_content: bool = Field(
        default=False,
        description="Apply masking to prompts that would be blocked. When True, masked content is sent to the LLM instead of blocking the request.",
    )

    mask_response_content: bool = Field(
        default=False,
        description="Apply masking to responses that would be blocked. When True, masked content is returned to the user instead of blocking the response.",
    )

    fallback_on_error: Literal["block", "allow"] = Field(
        default="block",
        description="Action when PANW API is unavailable (timeout, rate limit, network error): 'block' (default, maximum security) rejects requests; 'allow' (high availability) proceeds without scanning. Authentication and configuration errors always block.",
    )

    timeout: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="PANW API call timeout in seconds (1-60).",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "PANW Prisma AIRS"
