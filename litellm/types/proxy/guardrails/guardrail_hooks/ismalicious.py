from pydantic import BaseModel, Field

from .base import GuardrailConfigModel


class IsMaliciousConfigModel(GuardrailConfigModel[BaseModel]):
    api_key: str | None = Field(default=None, description="Base64 encoding of the IsMalicious API key and secret pair")

    @staticmethod
    def ui_friendly_name() -> str:
        return "IsMalicious"
