from typing import Optional

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

    profile_name: str = Field(
        description="PANW Prisma AIRS security profile name configured in Strata Cloud Manager. Required.",
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

    @staticmethod
    def ui_friendly_name() -> str:
        return "PANW Prisma AIRS"
