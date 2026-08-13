from typing import Final, Literal

from pydantic import Field, field_validator

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
    apply_to_aliases: str | list[str] | None = Field(  # mutable-ok: list sets UI type; Sequence ambiguous
        default=None,
        description=(
            "Allowlist of virtual-key aliases: only requests from keys whose "
            "alias is listed here are scanned by this guardrail. Leave empty to "
            "apply to all keys (subject to the exclude list below). Accepts a "
            "list or a comma-separated string."
        ),
    )
    except_aliases: str | list[str] | None = Field(  # mutable-ok: list sets UI type; Sequence ambiguous
        default=None,
        description=(
            "Exclude list of virtual-key aliases: requests from keys whose alias "
            "is listed here are NOT scanned by this guardrail (exempted), even "
            "when the allowlist is empty. Accepts a list or a comma-separated "
            "string."
        ),
    )
    send_meta: bool | None = Field(
        default=None,
        description=(
            "Forward caller context to XecGuard as the scan payload's `meta` "
            "object: `meta.virtualkey` is the calling virtual key's alias (its "
            "token hash when it has no alias) and `meta.data` is that key's own "
            "metadata from the Virtual Keys page. It takes no part in the "
            "verdict - XecGuard flattens it into the SIEM event (ctx_virtualkey, "
            "ctx_<field>) so scans can be traced back to the key that caused "
            "them. Defaults to false; falls back to the XECGUARD_SEND_META env "
            "var."
        ),
    )
    meta_data_fields: str | list[str] | None = Field(  # mutable-ok: list sets UI type; Sequence ambiguous
        default=None,
        description=(
            "Restrict which of the virtual key's metadata fields are forwarded "
            "in `meta.data`. Leave empty to send every field that fits the "
            "backend's contract (flat scalar values, at most 32 fields, 512 "
            "characters each), minus the proxy's own per-key control settings "
            "(rate limits, budget knobs, enforced params) which are skipped as "
            "SIEM noise - naming one here opts it back in. Callback credential "
            "slots are never forwarded either way. Accepts a list or a "
            "comma-separated string. Only used when `send_meta` is enabled."
        ),
    )
    # Named `meta_identity_format`, not `meta_virtualkey_format`: the proxy masks
    # any litellm_param whose name contains "key" before serving it back, so a
    # `virtualkey` in the name means the UI form prefills "ob****ct" and saving
    # the form writes that back - the plugin then falls through to the default and
    # the admin's choice is lost with no error. See the masking regression test.
    meta_identity_format: Literal["string", "object"] | None = Field(
        default=None,
        description=(
            "Wire shape of `meta.virtualkey`. 'string' (default) sends the alias "
            "as a bare string and is what current XecGuard backends accept. "
            "'object' sends `{alias, key_id}` so a scan stays attributable when "
            "the key has no alias or the alias was renamed or reused, and lifts "
            "the identifier-pattern restriction on aliases - it requires a "
            "backend that validates the object form, otherwise every scan is "
            "rejected with 400. Falls back to the "
            "XECGUARD_META_IDENTITY_FORMAT env var. Only used when `send_meta` "
            "is enabled."
        ),
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

    @field_validator("apply_to_aliases", "except_aliases", "meta_data_fields", mode="before")
    @classmethod
    def _normalize_alias_list(cls, v: object) -> object:
        """Accept either a list or a comma-separated string (the UI submits a
        plain text box as a string; YAML users may write a list) and normalize
        to a de-whitespaced, empties-dropped list of aliases / field names."""
        if v is None:
            return None
        items: Final = v.split(",") if isinstance(v, str) else v
        if isinstance(items, (list, tuple)):
            return [s.strip() for s in items if isinstance(s, str) and s.strip()]  # mutable-ok: tests assert this list
        return v

    @staticmethod
    def ui_friendly_name() -> str:
        return "XecGuard"
