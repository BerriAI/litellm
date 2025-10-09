from typing import Optional

from pydantic import BaseModel, Field

from .base import GuardrailConfigModel


class EnkryptAIGuardrailConfigModel(GuardrailConfigModel):
    api_key: Optional[str] = Field(
        default=None,
        description="The EnkryptAI API key. Reads from ENKRYPTAI_API_KEY env var if None.",
    )
    api_base: Optional[str] = Field(
        default=None,
        description="The EnkryptAI API base URL. Defaults to https://api.enkryptai.com. Also checks if the ENKRYPTAI_API_KEY env var is set.",
    )
    policy_name: Optional[str] = Field(
        default=None,
        description="The EnkryptAI policy name to use. Sent via x-enkrypt-policy header.",
    )
    deployment_name: Optional[str] = Field(
        default=None,
        description="The EnkryptAI deployment name to use. Sent via X-Enkrypt-Deployment header.",
    )
    detectors: Optional[dict] = Field(
        default=None,
        description="Dictionary of detector configurations (e.g., {'nsfw': {'enabled': True}, 'toxicity': {'enabled': True}}).",
    )
    block_on_violation: Optional[bool] = Field(
        default=True,
        description="Whether to block requests when violations are detected. Defaults to True.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "EnkryptAI"

