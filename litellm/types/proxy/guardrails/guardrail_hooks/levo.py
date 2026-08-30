from typing import Literal

from pydantic import BaseModel, Field

from .base import GuardrailConfigModel


class LevoGuardrailConfigModelOptionalParams(BaseModel):
    timeout: float | None = Field(
        default=None,
        description="Per-request timeout in seconds for calls to the Levo AI Gateway.",
    )

    unreachable_fallback: Literal["fail_open", "fail_closed"] | None = Field(
        default=None,
        description=(
            "Behaviour when the gateway cannot be reached. 'fail_closed' (default) "
            "rejects the LLM call; 'fail_open' lets it through unscanned, trading "
            "enforcement for availability."
        ),
    )

    extra_headers: list[str] | None = Field(
        default=None,
        description=(
            "Inbound header names whose values are forwarded to the gateway. "
            "Values of headers outside LiteLLM's default allowlist are replaced "
            "with a placeholder, so list anything the gateway must actually read "
            "— e.g. 'x-forwarded-for' for the real client IP, or a JWT-claims "
            "header used by identity policies."
        ),
    )

    buffer_streaming_until_moderated: bool | None = Field(
        default=None,
        description=(
            "Withhold streamed chunks until the assembled response has been "
            "moderated. Defaults to true, so a blocked response cannot reach the "
            "client after the fact. Set false to prioritise time-to-first-token, "
            "accepting that response-side findings arrive too late to stop output."
        ),
    )


class LevoGuardrailConfigModel(GuardrailConfigModel[LevoGuardrailConfigModelOptionalParams]):
    api_base: str = Field(
        min_length=1,
        description=(
            "Base URL of the Levo AI Gateway, e.g. http://levo-gateway:8080. "
            "The /beta/litellm_basic_guardrail_api path is appended automatically."
        ),
    )

    api_key: str | None = Field(
        default=None,
        description=(
            "Shared secret presented to the gateway as x-api-key. Must match "
            "LEVO_GUARDRAIL_API_KEY on the gateway, which serves the endpoint on "
            "its data-plane port and refuses to enable it until that is set. "
            "Env: LEVO_GUARDRAIL_API_KEY."
        ),
        json_schema_extra={"secret": True},
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Levo AI Gateway"
