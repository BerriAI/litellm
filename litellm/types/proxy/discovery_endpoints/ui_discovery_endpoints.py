from collections.abc import Sequence
from pydantic import (
    BaseModel,
    ConfigDict,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
)

from litellm.litellm_core_utils.native_oidc_validation import (
    has_control_characters,
    validate_issuer,
    validate_scope_tokens,
)
from litellm.types.proxy.control_plane_endpoints import WorkerRegistryEntry


class NativeOIDCConfig(BaseModel):
    """Public native OIDC bootstrap metadata.

    Only the issuer trust anchor, the public native client id, and the scopes to
    request. No client secret, signing material, claim mapping or team policy is
    ever published here.
    """

    issuer: str
    client_id: str
    scopes: tuple[str, ...]

    model_config = ConfigDict(extra="forbid")

    @field_validator("issuer")
    @classmethod
    def validate_issuer_value(cls, value: str) -> str:
        # Returned unchanged on purpose: the issuer is compared byte-for-byte
        # against the provider document, so normalizing it here would break the
        # trust anchor.
        return validate_issuer(value)

    @field_validator("client_id")
    @classmethod
    def validate_client_id(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("must be a non-blank client id without surrounding whitespace")
        if any(character.isspace() for character in value):
            raise ValueError("must not contain whitespace")
        if has_control_characters(value):
            raise ValueError("must not contain control characters")
        return value

    @field_validator("scopes")
    @classmethod
    def validate_scopes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return validate_scope_tokens(value)


class UiDiscoveryEndpoints(BaseModel):
    server_root_path: str
    proxy_base_url: str | None
    auto_redirect_to_sso: bool
    admin_ui_disabled: bool
    sso_configured: bool
    hide_default_credentials_hint: bool = False
    is_control_plane: bool = False
    workers: Sequence[WorkerRegistryEntry] = ()
    native_oidc: NativeOIDCConfig | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_native_oidc(self, handler: SerializerFunctionWrapHandler):
        """Drop `native_oidc` entirely when it is unset.

        Deliberately narrower than `exclude_none` / `response_model_exclude_none`:
        unrelated optional fields such as `proxy_base_url` must keep serializing
        as explicit nulls. Uses a model-level wrap serializer rather than
        `Field(exclude_if=...)` so the model stays compatible with the declared
        `pydantic>=2.10` floor.
        """
        serialized = handler(self)
        if isinstance(serialized, dict) and serialized.get("native_oidc") is None:
            serialized.pop("native_oidc", None)
        return serialized
