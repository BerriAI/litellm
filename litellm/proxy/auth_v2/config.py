from typing import List, Optional

from pydantic import AnyHttpUrl, BaseModel, Field, SecretStr, field_validator

from .models import SecuritySchemeType, require_secure_url
from .oidc.config import OIDCProviderConfig
from .saml.config import SAMLConfig
from .session import SessionConfig


class ApiKeySchemeConfig(BaseModel):
    header_name: str = "x-litellm-api-key"


class HttpBasicConfig(BaseModel):
    enabled: bool = False
    realm: str = "litellm"


class OAuth2IntrospectionConfig(BaseModel):
    introspection_endpoint: AnyHttpUrl
    client_id: str
    client_secret: SecretStr
    subject_field: str = "sub"
    audience: List[str] = Field(default_factory=list)
    issuer: Optional[str] = None

    @field_validator("introspection_endpoint")
    @classmethod
    def _endpoint_https(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        require_secure_url(str(value))
        return value


class MutualTLSConfig(BaseModel):
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
    oidc_providers: List[OIDCProviderConfig] = Field(default_factory=list)
    oauth2_introspection: Optional[OAuth2IntrospectionConfig] = None
    mutual_tls: MutualTLSConfig = Field(default_factory=MutualTLSConfig)
    network: TrustedProxyConfig = Field(default_factory=TrustedProxyConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    saml: Optional[SAMLConfig] = None
    casbin_policy_path: Optional[str] = None
