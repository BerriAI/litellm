from typing import Optional

from pydantic import BaseModel, Field

from .base import GuardrailConfigModel


class AkamaiFirewallForAIGuardrailOptionalParams(BaseModel):
    fai_configuration_id: Optional[str] = Field(
        default=None,
        description=(
            "The Firewall for AI configuration ID (path parameter `faiConfigurationId`). "
            "Reads from the AKAMAI_FIREWALL_CONFIGURATION_ID env var if None."
        ),
    )
    user_application_id: Optional[str] = Field(
        default=None,
        description=(
            "Identifies the application defined in your Firewall for AI configuration "
            "(request body `userApplicationId`). Reads from the "
            "AKAMAI_FIREWALL_USER_APPLICATION_ID env var if None."
        ),
    )
    max_detect_chars: Optional[int] = Field(
        default=None,
        description=(
            "Maximum number of characters sent in a single `llmInput`/`llmOutput`. Longer text is "
            "split into overlapping chunks that are scanned in parallel, because Firewall for AI "
            "answers an oversized field with an opaque HTTP 500. Defaults to 20000. Also checks the "
            "AKAMAI_FIREWALL_MAX_DETECT_CHARS env var."
        ),
    )


class AkamaiFirewallForAIGuardrailConfigModel(GuardrailConfigModel[AkamaiFirewallForAIGuardrailOptionalParams]):
    api_key: Optional[str] = Field(
        default=None,
        description=(
            "The Firewall for AI API key sent in the `Fai-Api-Key` header. "
            "Reads from the AKAMAI_FIREWALL_API_KEY env var if None."
        ),
    )
    api_base: Optional[str] = Field(
        default=None,
        description=(
            "The Firewall for AI API base URL. Defaults to https://aisec.akamai.com. "
            "Also checks the AKAMAI_FIREWALL_API_BASE env var."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Akamai Firewall for AI"
