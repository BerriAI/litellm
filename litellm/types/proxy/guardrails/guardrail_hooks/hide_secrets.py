"""Types for the Hide Secrets guardrail."""

from pydantic import Field

from .base import GuardrailConfigModel


class HideSecretsGuardrailConfigModel(GuardrailConfigModel):
    """Configuration for the Hide Secrets guardrail. Detection runs in-process
    on the detect-secrets library; ``detect_secrets_config`` overrides the
    bundled plugin set."""

    detect_secrets_config: dict | None = Field(  # mutable-ok: UI type derivation maps dict to "object"
        default=None,
        description="Optional detect-secrets configuration (plugins_used, filters_used) overriding the bundled plugin set",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Hide Secrets"
