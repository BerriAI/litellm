from typing import Optional

from pydantic import BaseModel, Field

from .base import GuardrailConfigModel


class LassoGuardrailConfigModelOptionalParams(BaseModel):
    user_id: Optional[str] = Field(
        default=None,
        description="The user ID for the Lasso guardrail. If not provided, the `LASSO_USER_ID` environment variable is checked.",
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="The conversation ID for the Lasso guardrail. If not provided, the `LASSO_CONVERSATION_ID` environment variable is checked.",
    )


class LassoGuardrailConfigModel(
    GuardrailConfigModel[LassoGuardrailConfigModelOptionalParams]
):
    api_key: Optional[str] = Field(
        default=None,
        description="The API key for the Lasso guardrail. If not provided, the `LASSO_API_KEY` environment variable is checked.",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="The API base for the Lasso guardrail. Default is https://server.lasso.security. Also checks if the `LASSO_API_BASE` environment variable is set.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Lasso Guardrail"
