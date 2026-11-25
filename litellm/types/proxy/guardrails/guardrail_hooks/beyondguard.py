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
    beyondguard_application: str | None = Field(
        default=None,
        description=(
            "Application name sent as X-Application, so BeyondGuard can apply the per-model policy "
            "registered for this application."
        ),
    )
    beyondguard_destination_url: str | None = Field(
        default=None,
        description=("Model URL sent as X-Destination-Url, matched against the URL of BeyondGuard per-model policies."),
    )
    beyondguard_user_email_header: str | None = Field(
        default=None,
        description=(
            "Header that carries the LiteLLM user's email for BeyondGuard group-based policies. "
            "Must match the Email Header Key of the BeyondGuard organization. Defaults to X-User-Email."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "BeyondGuard"
