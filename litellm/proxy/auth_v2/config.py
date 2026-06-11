from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import AnyHttpUrl, BaseModel, Field, SecretStr, model_validator

from .models import SecuritySchemeType

DEFAULT_SAML_ATTRIBUTE_MAP = {
    "email": "email",
    "mail": "email",
    "givenName": "given_name",
    "surname": "family_name",
    "sn": "family_name",
    "displayName": "display_name",
    "userName": "user_name",
    "uid": "user_name",
    "groups": "groups",
    "roles": "roles",
}


class ApiKeySchemeConfig(BaseModel):
    header_name: str = "x-litellm-api-key"


class HttpBasicConfig(BaseModel):
    enabled: bool = False
    realm: str = "litellm"


class OIDCProviderConfig(BaseModel):
    issuer: str
    audience: List[str]
    jwks_uri: Optional[AnyHttpUrl] = None
    algorithms: List[str] = Field(default_factory=lambda: ["RS256"])
    require_at_jwt: bool = False
    client_id: Optional[str] = None
    client_secret: Optional[SecretStr] = None
    login_scopes: List[str] = Field(
        default_factory=lambda: ["openid", "email", "profile"]
    )


class OAuth2IntrospectionConfig(BaseModel):
    introspection_endpoint: AnyHttpUrl
    client_id: str
    client_secret: SecretStr
    subject_field: str = "sub"
    audience: List[str] = Field(default_factory=list)


class MutualTLSConfig(BaseModel):
    enabled: bool = False
    forwarded_subject_header: Optional[str] = None


class TrustedProxyConfig(BaseModel):
    use_forwarded_for: bool = False
    trusted_proxy_cidrs: List[str] = Field(default_factory=list)


class SessionConfig(BaseModel):
    cookie: str = "litellm_session"
    secure: bool = True
    ttl_seconds: int = 3600
    max_size: int = 10000
    default_redirect_path: str = "/"
    login_cookie: str = "litellm_oidc_txn"
    login_state_ttl: int = 300


class SAMLConfig(BaseModel):
    enabled: bool = False
    entity_id: str
    acs_url: str
    idp_metadata: str = ""
    sp_key_file: Optional[str] = None
    sp_cert_file: Optional[str] = None
    allow_unsolicited: bool = False
    xmlsec_binary: Optional[str] = None
    attribute_map: Dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_SAML_ATTRIBUTE_MAP)
    )

    @model_validator(mode="after")
    def _require_idp_metadata(self) -> "SAMLConfig":
        if self.enabled and not self.idp_metadata.strip():
            raise ValueError(
                "SAML enabled but idp_metadata is empty (inline XML, local path, or URL)"
            )
        return self


class AuthConfig(BaseModel):
    scheme_order: List[SecuritySchemeType] = Field(
        default_factory=lambda: [
            SecuritySchemeType.API_KEY,
            SecuritySchemeType.HTTP,
            SecuritySchemeType.OPENID_CONNECT,
            SecuritySchemeType.OAUTH2,
            SecuritySchemeType.MUTUAL_TLS,
        ]
    )
    api_key: Optional[ApiKeySchemeConfig] = Field(default_factory=ApiKeySchemeConfig)
    http_basic: HttpBasicConfig = Field(default_factory=HttpBasicConfig)
    oidc_providers: List[OIDCProviderConfig] = Field(default_factory=list)
    oauth2_introspection: Optional[OAuth2IntrospectionConfig] = None
    mutual_tls: MutualTLSConfig = Field(default_factory=MutualTLSConfig)
    network: TrustedProxyConfig = Field(default_factory=TrustedProxyConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    saml: Optional[SAMLConfig] = None
    casbin_policy_path: Optional[str] = None
