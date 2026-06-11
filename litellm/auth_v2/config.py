from __future__ import annotations

from typing import List, Optional

from pydantic import AnyHttpUrl, BaseModel, Field, SecretStr

from .models import SecuritySchemeType


class ApiKeySchemeConfig(BaseModel):
    header_name: str = "x-litellm-api-key"


class HttpBasicConfig(BaseModel):
    enabled: bool = False
    realm: str = "litellm"


class OidcProviderConfig(BaseModel):
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


class MutualTlsConfig(BaseModel):
    enabled: bool = False
    forwarded_subject_header: Optional[str] = None


class TrustedProxyConfig(BaseModel):
    use_forwarded_for: bool = False
    trusted_proxy_cidrs: List[str] = Field(default_factory=list)


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
    oidc_providers: List[OidcProviderConfig] = Field(default_factory=list)
    oauth2_introspection: Optional[OAuth2IntrospectionConfig] = None
    mutual_tls: MutualTlsConfig = Field(default_factory=MutualTlsConfig)
    network: TrustedProxyConfig = Field(default_factory=TrustedProxyConfig)
