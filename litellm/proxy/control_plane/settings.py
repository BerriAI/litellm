"""Process-level configuration for the control plane, read once at startup."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, TypeAlias

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, TypeAdapter, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from litellm.litellm_core_utils.oauth_token_exchange import TokenExchangeProfile
from litellm.types.proxy.control_plane.auth import PASSWORD_MAX_LENGTH, USERNAME_MAX_LENGTH

DEFAULT_SESSION_TTL_SECONDS: Final = 12 * 60 * 60
DEFAULT_DATAPLANE_TIMEOUT_SECONDS: Final = 10.0


@dataclass(frozen=True, slots=True)
class OidcSettings:
    """External IdP used to verify caller JWTs and, later, to exchange them for dataplane tokens."""

    issuer: str
    audience: str
    jwks_url: str | None
    roles_claim: str
    scope_claim: str
    dataplanes_claim: str
    email_claim: str
    leeway_seconds: int


class RoleSpec(BaseModel):
    """An operator-defined role, e.g. CONTROL_PLANE_ROLES='{"finops": {"scopes": ["usage:read", "audit:read"]}}'."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scopes: tuple[str, ...] = ()
    inherits: tuple[str, ...] = ()


class OidcExchangeSpec(BaseModel):
    """IdP token-exchange client the control plane uses to mint a token for this dataplane's audience."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    token_endpoint: str
    client_id: str
    client_secret: SecretStr
    profile: TokenExchangeProfile = "rfc8693"
    audience: str | None = None
    scope: str | None = None


class ServiceTokenSpec(BaseModel):
    """A dataplane virtual key holding the `impersonate` permission, the fallback when no IdP exchange is possible."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    token: SecretStr


class DataplaneSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str = Field(pattern=r"^https?://")
    name: str | None = None
    oidc_exchange: OidcExchangeSpec | None = None
    service_token: ServiceTokenSpec | None = None
    allow_insecure_http: bool = False

    @model_validator(mode="after")
    def _https_unless_opted_out(self) -> "DataplaneSpec":
        if self.url.startswith("http://") and not self.allow_insecure_http:
            raise ValueError(
                f"dataplane url {self.url!r} is plain http, the bearer credential would travel unencrypted. "
                "Use https or set allow_insecure_http: true for a local or in-cluster dataplane."
            )
        return self


DataplaneSpecs: TypeAlias = Mapping[str, DataplaneSpec]
_DATAPLANE_SPECS: Final = TypeAdapter(DataplaneSpecs)
_YAML_DOCUMENT: Final = TypeAdapter(dict[str, object])


class ControlPlaneSettings(BaseSettings):
    model_config = SettingsConfigDict(frozen=True, env_prefix="CONTROL_PLANE_", extra="ignore")

    master_key: SecretStr = Field(validation_alias="LITELLM_MASTER_KEY")
    ui_username: str | None = Field(default=None, max_length=USERNAME_MAX_LENGTH, validation_alias="UI_USERNAME")
    ui_password: SecretStr | None = Field(default=None, max_length=PASSWORD_MAX_LENGTH, validation_alias="UI_PASSWORD")
    session_ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS
    session_issuer: str = "litellm-control-plane"
    session_audience: str = "litellm-control-plane-api"
    roles: Mapping[str, RoleSpec] = Field(default_factory=dict)
    include_default_roles: bool = True
    oidc_issuer: str | None = None
    oidc_audience: str | None = None
    oidc_jwks_url: str | None = None
    oidc_roles_claim: str = "roles"
    oidc_scope_claim: str = "scope"
    oidc_dataplanes_claim: str = "litellm_dataplanes"
    oidc_email_claim: str = "email"
    oidc_leeway_seconds: int = 30
    dataplanes: DataplaneSpecs = Field(default_factory=dict)
    dataplanes_file: str | None = None
    dataplane_timeout_seconds: float = DEFAULT_DATAPLANE_TIMEOUT_SECONDS
    dataplane_retry_attempts: int = 3
    fanout_concurrency: int = 16
    audit_capacity: int = 10_000
    api_prefix: str = "/api/v1"

    @property
    def oidc(self) -> OidcSettings | None:
        if self.oidc_issuer is None or self.oidc_audience is None:
            return None
        return OidcSettings(
            issuer=self.oidc_issuer,
            audience=self.oidc_audience,
            jwks_url=self.oidc_jwks_url,
            roles_claim=self.oidc_roles_claim,
            scope_claim=self.oidc_scope_claim,
            dataplanes_claim=self.oidc_dataplanes_claim,
            email_claim=self.oidc_email_claim,
            leeway_seconds=self.oidc_leeway_seconds,
        )


def read_dataplanes_file(path: str) -> DataplaneSpecs:
    """A yaml file with a top-level `dataplanes:` map keyed by dataplane id, the same shape as the env JSON."""
    with open(path, encoding="utf-8") as handle:
        document: Final = _YAML_DOCUMENT.validate_python(yaml.safe_load(handle))
    return _DATAPLANE_SPECS.validate_python(document.get("dataplanes", {}))


def load_settings() -> ControlPlaneSettings:
    from_env: Final = ControlPlaneSettings()  # pyright: ignore[reportCallIssue]  # pydantic-settings fills fields from env
    if from_env.dataplanes_file is None:
        return from_env
    return from_env.model_copy(update={"dataplanes": read_dataplanes_file(from_env.dataplanes_file)})
