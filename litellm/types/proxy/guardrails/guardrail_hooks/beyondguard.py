from pydantic import Field

from .base import GuardrailConfigModel


class BeyondGuardConfigModel(GuardrailConfigModel):
    api_key: str | None = Field(
        default=None,
        description=(
            "BeyondGuard endpoint token, sent as X-Guard-Token. "
            "If not provided, the BEYONDGUARD_API_KEY environment variable is used."
        ),
    )
    api_base: str | None = Field(
        default=None,
        description=(
            "BeyondGuard gateway base URL, e.g. https://<your-beyondguard-host>/bg-gateway. "
            "If not provided, the BEYONDGUARD_API_BASE environment variable is used."
        ),
    )
    beyondguard_endpoint_id: str | None = Field(
        default=None,
        description=(
            "ID of the BeyondGuard Context Guard endpoint that checks requests (pre_call / during_call). "
            "If not provided, the BEYONDGUARD_ENDPOINT_ID environment variable is used."
        ),
    )
    beyondguard_output_endpoint_id: str | None = Field(
        default=None,
        description=(
            "ID of the BeyondGuard Context Guard endpoint that checks model responses (post_call). "
            "Defaults to beyondguard_endpoint_id."
        ),
    )
    beyondguard_output_api_key: str | None = Field(
        default=None,
        description="Token of the output endpoint, if it differs from api_key.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "BeyondGuard"
