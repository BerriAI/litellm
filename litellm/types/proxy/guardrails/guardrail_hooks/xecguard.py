from typing import Final, Literal

from pydantic import Field

from .base import GuardrailConfigModel

XECGUARD_DEFAULT_POLICY_OPTIONS: Final = [
    "Default_Policy_SystemPromptEnforcement",
    "Default_Policy_GeneralPromptAttackProtection",
    "Default_Policy_ContentBiasProtection",
    "Default_Policy_HarmfulContentProtection",
    "Default_Policy_SkillsProtection",
    "Default_Policy_PIISensitiveDataProtection",
]


class XecGuardConfigModel(GuardrailConfigModel):
    api_key: str | None = Field(
        default=None,
        description=(
            "Service Token for XecGuard (prefix 'xgs_'). "
            "If not provided, the XECGUARD_API_KEY environment "
            "variable is used."
        ),
    )
    api_base: str | None = Field(
        default=None,
        description=(
            "XecGuard API base URL. "
            "Defaults to https://api-xecguard.cycraft.ai. "
            "Falls back to the XECGUARD_API_BASE env var."
        ),
    )
    xecguard_model: str | None = Field(
        default=None,
        description=("XecGuard scanning model identifier. Defaults to 'xecguard_v2'."),
    )
    policy_names: list[str] | None = Field(
        default=None,
        description=(
            "XecGuard policies to apply on each scan. Select one or more "
            "of the built-in default policies; if none are selected, "
            "the guardrail defaults to System Prompt Enforcement + "
            "Harmful Content Protection."
        ),
        json_schema_extra={
            "ui_type": "multiselect",
            "options": list(XECGUARD_DEFAULT_POLICY_OPTIONS),
        },
    )
    block_on_error: bool | None = Field(
        default=None,
        description=(
            "Whether to block requests when the XecGuard API is "
            "unreachable. Defaults to true (fail-closed). "
            "Falls back to the XECGUARD_BLOCK_ON_ERROR env var."
        ),
    )
    grounding_strictness: Literal["BALANCED", "STRICT"] | None = Field(
        default=None,
        description=(
            "Strictness level for XecGuard context-grounding "
            "validation. 'BALANCED' (default) treats INCOMPLETE "
            "answers as SAFE; 'STRICT' flags them as UNSAFE. "
            "Grounding only runs in post_call when "
            "`metadata.xecguard_grounding_documents` is provided."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "XecGuard"
