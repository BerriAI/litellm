from typing import Final

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    field_validator,
)

from litellm._logging import verbose_proxy_logger

from .base import GuardrailConfigModel


class NomaGuardrailConfigModel(GuardrailConfigModel):
    use_v2: bool | None = Field(
        default=False,
        description="If True and guardrail='noma', route to the new Noma v2 implementation.",
    )
    api_key: str | None = Field(
        default=None,
        description="The Noma API key. Reads from NOMA_API_KEY env var if None.",
    )
    api_base: str | None = Field(
        default=None,
        description="The Noma API base URL. Defaults to https://api.noma.security. Also checks if the NOMA_API_KEY env var is set.",
    )
    application_id: str | None = Field(
        default=None,
        description="The Noma Application ID. Reads from NOMA_APPLICATION_ID env var if None.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Noma Security"


class NomaV2GuardrailConfigModel(GuardrailConfigModel):
    api_key: str | None = Field(
        default=None,
        description="The Noma API key. Reads from NOMA_API_KEY env var if None.",
    )
    api_base: str | None = Field(
        default=None,
        description="The Noma API base URL. Defaults to https://api.noma.security.",
    )
    application_id: str | None = Field(
        default=None,
        description="The Noma Application ID. Reads from NOMA_APPLICATION_ID env var if None.",
    )
    gateway_name: str | None = Field(
        default=None,
        description="Gateway name, used as the gateway_host label on Noma scans. Falls back to NOMA_GATEWAY_NAME.",
    )
    monitor_mode: bool | None = Field(
        default=None,
        description="When true, run guardrail checks in monitor mode.",
    )
    block_failures: bool | None = Field(
        default=None,
        description="When true, fail closed on Noma API errors.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Noma Security v2"


class NomaV2StreamingParams(BaseModel):
    """Streaming settings a Noma v2 guardrail reads from the top level of its litellm_params.

    A key left unset, or set to a value that does not validate, stays None, so the unified
    streaming hook keeps its own default for it and the guardrail still loads.
    """

    model_config = ConfigDict(frozen=True)

    streaming_buffer_until_moderated: bool | None = None
    streaming_end_of_stream_only: bool | None = None
    streaming_sampling_rate: int | None = Field(default=None, ge=1)
    streaming_buffer_release_on_scan: bool | None = None

    @field_validator("*", mode="wrap")
    @classmethod
    def _ignore_invalid_value(
        cls, value: object, handler: ValidatorFunctionWrapHandler, info: ValidationInfo
    ) -> object:
        try:
            validated: Final = handler(value)
        except ValidationError:
            verbose_proxy_logger.warning(
                "Noma v2 guardrail %s: ignoring %s=%.100r, which is not a valid value; the default applies",
                info.context.get("guardrail_name") if info.context else None,
                info.field_name,
                value,
            )
            return None
        return validated
