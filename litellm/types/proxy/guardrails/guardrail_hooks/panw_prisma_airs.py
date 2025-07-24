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
        default="default",
        description="PANW Prisma AIRS security profile name. Required.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "PANW Prisma AIRS"
