"""Process-level configuration for the control plane, read once at startup."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias
from urllib.parse import urlsplit

import yaml
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    TypeAdapter,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from litellm.litellm_core_utils.oauth_token_exchange import TokenExchangeProfile
from litellm.proxy.control_plane.audit.event import CONTROL_PLANE_ID
from litellm.proxy.control_plane.auth.rbac import ADMIN_ROLE
from litellm.types.proxy.control_plane.auth import PASSWORD_MAX_LENGTH, USERNAME_MAX_LENGTH

DEFAULT_SESSION_TTL_SECONDS: Final = 12 * 60 * 60
AuditSinkKind: TypeAlias = Literal["log", "otlp", "s3", "splunk_hec", "sentinel"]
ObjectLockMode: TypeAlias = Literal["COMPLIANCE", "GOVERNANCE"]
_SINK_URL_FIELDS: Final = (
    ("audit_s3_endpoint_url", "audit_s3_allow_insecure_http"),
    ("audit_splunk_url", "audit_splunk_allow_insecure_http"),
    ("audit_sentinel_endpoint", None),
    ("audit_sentinel_authority_host", None),
)


def _plain_http(url: object) -> bool:
    return isinstance(url, str) and urlsplit(url).scheme.lower() != "https"


_REQUIRED_SINK_FIELDS: Final = (
    ("s3", "audit_s3_bucket"),
    ("splunk_hec", "audit_splunk_url"),
    ("splunk_hec", "audit_splunk_token"),
    ("sentinel", "audit_sentinel_endpoint"),
    ("sentinel", "audit_sentinel_dcr_immutable_id"),
    ("sentinel", "audit_sentinel_tenant_id"),
    ("sentinel", "audit_sentinel_client_id"),
    ("sentinel", "audit_sentinel_client_secret"),
)
DEFAULT_DATAPLANE_TIMEOUT_SECONDS: Final = 10.0


@dataclass(frozen=True, slots=True)
class S3SinkSettings:
    bucket: str
    prefix: str
    region: str | None
    endpoint_url: str | None
    object_lock_mode: ObjectLockMode | None
    retention_days: int
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class SplunkHecSettings:
    url: str
    token: SecretStr
    index: str | None
    source: str
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class SentinelSettings:
    """Azure Monitor Logs Ingestion API target plus the Entra app that may write to it."""

    endpoint: str
    dcr_immutable_id: str
    stream_name: str
    tenant_id: str
    client_id: str
    client_secret: SecretStr
    authority_host: str
    token_scope: str
    timeout_seconds: float


def _sentinel_token_scope(authority_host: str, override: str | None) -> str:
    if override is not None:
        return override
    match urlsplit(authority_host).hostname:
        case "login.microsoftonline.us":
            return "https://monitor.azure.us/.default"
        case "login.chinacloudapi.cn":
            return "https://monitor.azure.cn/.default"
        case _:
            return "https://monitor.azure.com/.default"


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
        if "@" in self.url.partition("://")[2].partition("/")[0]:
            raise ValueError("dataplane url must not contain userinfo; credentials belong in a credential strategy.")
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
    database_url: SecretStr | None = None
    redis_url: SecretStr | None = None
    config_stream: str = Field(default="litellm:control_plane:config", min_length=1)
    config_stream_maxlen: int = Field(default=10_000, ge=1)
    salt_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("CONTROL_PLANE_SALT_KEY", "LITELLM_SALT_KEY")
    )
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
    fanout_concurrency: int = Field(default=16, ge=1)
    usage_max_window_days: int = Field(default=366, ge=1)
    audit_capacity: int = Field(default=10_000, ge=1)
    api_prefix: str = "/api/v1"
    otel_enabled: bool | None = None
    otel_service_name: str = "litellm-control-plane"
    otel_identity_attributes: bool = True
    audit_sinks: tuple[AuditSinkKind, ...] = ("log",)
    audit_queue_size: int = Field(default=10_000, ge=2)
    audit_batch_size: int = Field(default=100, ge=1)
    audit_drain_seconds: float = Field(default=5.0, gt=0)
    audit_strict: bool = False
    audit_s3_bucket: str | None = None
    audit_s3_prefix: str = "control-plane-audit"
    audit_s3_region: str | None = None
    audit_s3_endpoint_url: str | None = None
    audit_s3_allow_insecure_http: bool = False
    audit_s3_object_lock_mode: ObjectLockMode | None = None
    audit_s3_retention_days: int = Field(default=365, ge=1)
    audit_s3_timeout_seconds: float = 30.0
    audit_splunk_url: str | None = None
    audit_splunk_allow_insecure_http: bool = False
    audit_splunk_token: SecretStr | None = None
    audit_splunk_index: str | None = None
    audit_splunk_source: str = "litellm-control-plane"
    audit_splunk_timeout_seconds: float = 30.0
    audit_sentinel_endpoint: str | None = None
    audit_sentinel_dcr_immutable_id: str | None = None
    audit_sentinel_stream_name: str = "Custom-LiteLLMControlPlaneAudit_CL"
    audit_sentinel_tenant_id: str | None = None
    audit_sentinel_client_id: str | None = None
    audit_sentinel_client_secret: SecretStr | None = None
    audit_sentinel_authority_host: str = "https://login.microsoftonline.com"
    audit_sentinel_token_scope: str | None = None
    audit_sentinel_timeout_seconds: float = 30.0

    @model_validator(mode="after")
    def _runtime_store_settings(self) -> "ControlPlaneSettings":
        if (self.database_url is None) != (self.redis_url is None):
            raise ValueError("Set both CONTROL_PLANE_DATABASE_URL and CONTROL_PLANE_REDIS_URL, or neither.")
        if self.database_url is not None and (self.salt_key is None or not self.salt_key.get_secret_value().strip()):
            raise ValueError("Runtime config writes require CONTROL_PLANE_SALT_KEY or LITELLM_SALT_KEY.")
        return self

    @field_validator("audit_s3_bucket")
    @classmethod
    def _s3_bucket_is_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("CONTROL_PLANE_AUDIT_S3_BUCKET must not be empty or whitespace.")
        return value

    @field_validator(
        "audit_splunk_url",
        "audit_splunk_token",
        "audit_splunk_index",
        "audit_splunk_source",
        "audit_sentinel_endpoint",
        "audit_sentinel_dcr_immutable_id",
        "audit_sentinel_stream_name",
        "audit_sentinel_tenant_id",
        "audit_sentinel_client_id",
        "audit_sentinel_client_secret",
        "audit_sentinel_token_scope",
    )
    @classmethod
    def _required_sink_strings_are_not_blank(cls, value: str | SecretStr | None) -> str | SecretStr | None:
        text: Final = value.get_secret_value() if isinstance(value, SecretStr) else value
        if isinstance(text, str) and not text.strip():
            raise ValueError("Required audit sink settings must not be empty or whitespace.")
        return value

    @model_validator(mode="after")
    def _listed_sinks_have_their_settings(self) -> "ControlPlaneSettings":
        missing: Final = tuple(
            f"CONTROL_PLANE_{name.upper()}"
            for kind, name in _REQUIRED_SINK_FIELDS
            if kind in self.audit_sinks and getattr(self, name) is None
        )
        if missing:
            raise ValueError(
                f"CONTROL_PLANE_AUDIT_SINKS lists a sink whose settings are not set: {', '.join(missing)}."
            )
        return self

    @model_validator(mode="after")
    def _sink_urls_are_https_unless_opted_out(self) -> "ControlPlaneSettings":
        """Every sink URL carries a credential (SigV4, HEC token, bearer, client secret), so plain http is refused.
        Only the stores that can live in-cluster (S3 compatible, Splunk) have an opt-out."""
        insecure: Final = tuple(
            (name, flag)
            for name, flag in _SINK_URL_FIELDS
            if _plain_http(getattr(self, name)) and not (flag is not None and getattr(self, flag))
        )
        if not insecure:
            return self
        name, flag = insecure[0]
        remedy: Final = f" or set CONTROL_PLANE_{flag.upper()}=true for an in-cluster endpoint" if flag else ""
        raise ValueError(
            f"CONTROL_PLANE_{name.upper()} is not https, audit records and the sink credential would travel "
            f"unencrypted. Use https{remedy}."
        )

    @property
    def audit_splunk(self) -> SplunkHecSettings | None:
        if self.audit_splunk_url is None or self.audit_splunk_token is None:
            return None
        return SplunkHecSettings(
            url=self.audit_splunk_url,
            token=self.audit_splunk_token,
            index=self.audit_splunk_index,
            source=self.audit_splunk_source,
            timeout_seconds=self.audit_splunk_timeout_seconds,
        )

    @property
    def audit_sentinel(self) -> SentinelSettings | None:
        if (
            self.audit_sentinel_endpoint is None
            or self.audit_sentinel_dcr_immutable_id is None
            or self.audit_sentinel_tenant_id is None
            or self.audit_sentinel_client_id is None
            or self.audit_sentinel_client_secret is None
        ):
            return None
        authority_host: Final = self.audit_sentinel_authority_host.rstrip("/")
        return SentinelSettings(
            endpoint=self.audit_sentinel_endpoint,
            dcr_immutable_id=self.audit_sentinel_dcr_immutable_id,
            stream_name=self.audit_sentinel_stream_name,
            tenant_id=self.audit_sentinel_tenant_id,
            client_id=self.audit_sentinel_client_id,
            client_secret=self.audit_sentinel_client_secret,
            authority_host=authority_host,
            token_scope=_sentinel_token_scope(authority_host, self.audit_sentinel_token_scope),
            timeout_seconds=self.audit_sentinel_timeout_seconds,
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

    @model_validator(mode="after")
    def _strict_audit_requires_a_sink(self) -> "ControlPlaneSettings":
        if self.audit_strict and not self.audit_sinks:
            raise ValueError("CONTROL_PLANE_AUDIT_STRICT requires at least one audit sink.")
        return self

    @field_validator("master_key")
    @classmethod
    def _master_key_has_minimum_length(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value().encode("utf-8")) < 32:
            raise ValueError("LITELLM_MASTER_KEY must be at least 32 UTF-8 bytes.")
        return value

    @model_validator(mode="after")
    def _password_login_has_admin_role(self) -> "ControlPlaneSettings":
        if (
            self.ui_username is None
            or self.ui_password is None
            or self.include_default_roles
            or ADMIN_ROLE in self.roles
        ):
            return self
        raise ValueError("password login requires a custom admin role when default roles are disabled.")

    @model_validator(mode="after")
    def _dataplanes_exclude_control_plane_id(self) -> "ControlPlaneSettings":
        if CONTROL_PLANE_ID not in self.dataplanes:
            return self
        raise ValueError(f"dataplane id {CONTROL_PLANE_ID!r} is reserved.")

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
    if os.environ.get("CONTROL_PLANE_DATAPLANES") is not None:
        raise ValueError("CONTROL_PLANE_DATAPLANES and CONTROL_PLANE_DATAPLANES_FILE cannot both be set.")
    return ControlPlaneSettings.model_validate(
        {**from_env.model_dump(by_alias=True), "dataplanes": read_dataplanes_file(from_env.dataplanes_file)}
    )
