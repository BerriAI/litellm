"""Resolved SSO config object.

Reconciles the dedicated ``sso_config`` DB row (lowercase, per-value encrypted
keys) with the process environment (uppercase env vars) into a typed
``SSOConfig`` plus per-field provenance. This is the single source of truth for
the SSO field -> env-var mapping, used by both the read-back endpoint and the
save endpoint so the two can never drift.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper
from litellm.proxy.config_resolvers._descriptors import (
    FieldDescriptor,
    FieldSource,
    resolve_fields,
)
from litellm.types.proxy.management_endpoints.ui_sso import (
    RoleMappings,
    SSOConfig,
    TeamMappings,
)

SSO_DESCRIPTORS: tuple[FieldDescriptor, ...] = (
    FieldDescriptor("google_client_id", "google_client_id", "GOOGLE_CLIENT_ID"),
    FieldDescriptor("google_client_secret", "google_client_secret", "GOOGLE_CLIENT_SECRET", is_secret=True),
    FieldDescriptor("microsoft_client_id", "microsoft_client_id", "MICROSOFT_CLIENT_ID"),
    FieldDescriptor("microsoft_client_secret", "microsoft_client_secret", "MICROSOFT_CLIENT_SECRET", is_secret=True),
    FieldDescriptor("microsoft_tenant", "microsoft_tenant", "MICROSOFT_TENANT"),
    FieldDescriptor("generic_client_id", "generic_client_id", "GENERIC_CLIENT_ID"),
    FieldDescriptor("generic_client_secret", "generic_client_secret", "GENERIC_CLIENT_SECRET", is_secret=True),
    FieldDescriptor(
        "generic_authorization_endpoint", "generic_authorization_endpoint", "GENERIC_AUTHORIZATION_ENDPOINT"
    ),
    FieldDescriptor("generic_token_endpoint", "generic_token_endpoint", "GENERIC_TOKEN_ENDPOINT"),
    FieldDescriptor("generic_userinfo_endpoint", "generic_userinfo_endpoint", "GENERIC_USERINFO_ENDPOINT"),
    FieldDescriptor("generic_scope", "generic_scope", "GENERIC_SCOPE", default="openid email profile"),
    FieldDescriptor("proxy_base_url", "proxy_base_url", "PROXY_BASE_URL"),
)

# Derived from the descriptor table so read (masking) and the field->env mapping
# never diverge from the resolver.
SSO_SECRET_FIELDS: frozenset[str] = frozenset(d.field_name for d in SSO_DESCRIPTORS if d.is_secret)
SSO_FIELD_ENV_VARS: dict[str, str] = {d.field_name: d.env_var for d in SSO_DESCRIPTORS}

# Structured sub-objects stored on the SSO row that are not simple env-backed
# scalars; handled outside the descriptor resolution.
_STRUCTURED_KEYS = ("role_mappings", "team_mappings")


@dataclass(frozen=True, slots=True)
class ResolvedSSOConfig:
    config: SSOConfig
    provenance: dict[str, FieldSource]


def _decrypt(raw: Mapping[str, object]) -> dict[str, object]:
    return {
        key: (
            decrypt_value_helper(value=value, key=key, return_original_value=True) if isinstance(value, str) else value
        )
        for key, value in raw.items()
    }


def _parse_role_mappings(data: object) -> RoleMappings | None:
    # The stored row is JSON, so mappings arrive as a dict (or are absent).
    return RoleMappings(**data) if isinstance(data, dict) else None


def _parse_team_mappings(data: object) -> TeamMappings | None:
    return TeamMappings(**data) if isinstance(data, dict) else None


def resolve_sso_config(sso_db_settings: Mapping[str, object] | None, env: Mapping[str, str]) -> ResolvedSSOConfig:
    """Resolve the effective SSO config: stored row first, then process env.

    Decryption happens here, once, via the pure ``decrypt_value_helper``; this
    function never writes ``os.environ`` (unlike the legacy read path). Values
    are returned unmasked so the login path could consume them; the read-back
    endpoint is responsible for masking secrets before responding to the UI.
    """
    raw = dict(sso_db_settings) if sso_db_settings else {}
    decrypted = _decrypt({key: value for key, value in raw.items() if key not in _STRUCTURED_KEYS})
    values, provenance = resolve_fields(SSO_DESCRIPTORS, decrypted, env)
    structured = {
        "user_email": decrypted.get("user_email"),
        "ui_access_mode": decrypted.get("ui_access_mode"),
        "role_mappings": _parse_role_mappings(raw.get("role_mappings")),
        "team_mappings": _parse_team_mappings(raw.get("team_mappings")),
    }
    config = SSOConfig(**{**values, **structured})
    return ResolvedSSOConfig(config=config, provenance=provenance)
