"""Process-level configuration for the control plane, read once at startup."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, TypeAdapter, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from litellm.litellm_core_utils.oauth_token_exchange import TokenExchangeProfile
from litellm.types.proxy.control_plane.auth import PASSWORD_MAX_LENGTH, USERNAME_MAX_LENGTH

DEFAULT_SESSION_TTL_SECONDS: Final = 12 * 60 * 60
AuditSinkKind: TypeAlias = Literal["log", "otlp", "s3"]
ObjectLockMode: TypeAlias = Literal["COMPLIANCE", "GOVERNANCE"]
DEFAULT_DATAPLANE_TIMEOUT_SECONDS: Final = 10.0


@dataclass(frozen=True, slots=True)
class S3SinkSettings:
    """Where the s3 audit sink writes. ``region`` is None until the container resolves it from the AWS environment."""

    bucket: str
    prefix: str
    region: str | None
    endpoint_url: str | None
    object_lock_mode: ObjectLockMode | None
    retention_days: int
    timeout_seconds: float


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
    otel_enabled: bool | None = None
    otel_service_name: str = "litellm-control-plane"
    otel_identity_attributes: bool = True
    audit_sinks: tuple[AuditSinkKind, ...] = ("log",)
    audit_queue_size: int = 10_000
    audit_batch_size: int = 100
    audit_strict: bool = False
    audit_s3_bucket: str | None = None
    audit_s3_prefix: str = "control-plane-audit"
    audit_s3_region: str | None = None
    audit_s3_endpoint_url: str | None = None
    audit_s3_allow_insecure_http: bool = False
    audit_s3_object_lock_mode: ObjectLockMode | None = None
    audit_s3_retention_days: int = Field(default=365, ge=1)
    audit_s3_timeout_seconds: float = 30.0

    @model_validator(mode="after")
    def _s3_sink_needs_a_bucket(self) -> "ControlPlaneSettings":
        if "s3" in self.audit_sinks and self.audit_s3_bucket is None:
            raise ValueError("CONTROL_PLANE_AUDIT_SINKS names s3 but CONTROL_PLANE_AUDIT_S3_BUCKET is not set.")
        return self

    @model_validator(mode="after")
    def _s3_endpoint_is_https_unless_opted_out(self) -> "ControlPlaneSettings":
        endpoint: Final = self.audit_s3_endpoint_url
        if endpoint is None or self.audit_s3_allow_insecure_http or urlsplit(endpoint).scheme.lower() == "https":
            return self
        raise ValueError(
            f"audit s3 endpoint {endpoint!r} is not https, audit records and the SigV4 authorization would travel "
            "unencrypted. Use https or set CONTROL_PLANE_AUDIT_S3_ALLOW_INSECURE_HTTP=true for an in-cluster store."
        )

    @property
    def audit_s3(self) -> S3SinkSettings | None:
        if self.audit_s3_bucket is None:
            return None
        return S3SinkSettings(
            bucket=self.audit_s3_bucket,
            prefix=self.audit_s3_prefix.strip("/"),
            region=self.audit_s3_region,
            endpoint_url=self.audit_s3_endpoint_url,
            object_lock_mode=self.audit_s3_object_lock_mode,
            retention_days=self.audit_s3_retention_days,
            timeout_seconds=self.audit_s3_timeout_seconds,
        )

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
