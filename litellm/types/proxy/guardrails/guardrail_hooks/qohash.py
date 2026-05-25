from typing import Optional

from pydantic import Field

from .base import GuardrailConfigModel


class QostodianNexusConfigModel(GuardrailConfigModel):
    api_base: Optional[str] = Field(
        default=None,
        description="The API base URL for Qostodian Nexus. If not provided, the `QOSTODIAN_NEXUS_API_BASE` environment variable is checked. Defaults to http://nexus:8800.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Qostodian Nexus"
