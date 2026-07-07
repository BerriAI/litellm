from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from litellm.proxy.auth_v2.authorization import Role

if TYPE_CHECKING:
    from fastapi.security import SecurityScopes

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def require_secure_url(value: str) -> str:
    host = urlparse(value).hostname or ""
    if value.startswith("https://") or host in _LOOPBACK_HOSTS:
        return value
    raise ValueError(f"insecure URL, https is required (loopback excepted): {value}")


class SecuritySchemeType(str, Enum):
    API_KEY = "apiKey"
    HTTP = "http"
    OAUTH2 = "oauth2"
    OPENID_CONNECT = "openIdConnect"
    MUTUAL_TLS = "mutualTLS"


class AuthMethod(str, Enum):
    API_KEY = "api_key"
    HTTP_BASIC = "http_basic"
    BEARER_JWT = "bearer_jwt"
    OAUTH2_INTROSPECTION = "oauth2_introspection"
    OIDC = "oidc"
    SAML = "saml"
    MUTUAL_TLS = "mutual_tls"


class PrincipalType(str, Enum):
    HUMAN = "human"
    SERVICE_ACCOUNT = "service_account"


class TeamRole(str, Enum):
    ADMIN = "admin"
    MEMBER = "member"


class UserIdentity(BaseModel):
    id: str
    external_id: Optional[str] = None
    user_name: Optional[str] = None
    email: Optional[str] = None
    display_name: Optional[str] = None


class OrganizationIdentity(BaseModel):
    id: str
    name: Optional[str] = None


class TeamIdentity(BaseModel):
    id: str
    name: Optional[str] = None
    role: TeamRole = TeamRole.MEMBER


class ProjectIdentity(BaseModel):
    id: str
    name: Optional[str] = None


class EndUserIdentity(BaseModel):
    id: str


class CredentialRef(BaseModel):
    key_id: Optional[str] = None
    token_id: Optional[str] = None


class NetworkContext(BaseModel):
    client_ip: Optional[str] = None
    host: Optional[str] = None
    via_trusted_proxy: bool = False


class ClientCertificate(BaseModel):
    subject_dn: str
    issuer_dn: Optional[str] = None
    serial_number: Optional[str] = None


class Credential(BaseModel):
    """A verified credential, before identity resolution."""

    model_config = ConfigDict(frozen=True)

    scheme: SecuritySchemeType
    method: AuthMethod
    subject: str
    issuer: Optional[str] = None
    audience: List[str] = Field(default_factory=list)
    scopes: List[str] = Field(default_factory=list)
    claims: Dict[str, Any] = Field(default_factory=dict)
    credential_ref: CredentialRef = Field(default_factory=CredentialRef)
    client_certificate: Optional[ClientCertificate] = None

    # Raw bearer/access token as presented by the caller, retained so it can be
    # used as the subject_token for downstream token exchange (RFC 8693) when
    # calling LLM providers or MCP servers on the caller's behalf. None for
    # schemes without an exchangeable token (API key, HTTP basic, mTLS).
    subject_token: Optional[str] = None


class Principal(BaseModel):
    """Normalized caller identity. Identity only, no policy/budget state."""

    principal_type: PrincipalType
    subject: str
    issuer: Optional[str] = None
    audience: List[str] = Field(default_factory=list)

    user: Optional[UserIdentity] = None
    organization: Optional[OrganizationIdentity] = None
    teams: List[TeamIdentity] = Field(default_factory=list)
    project: Optional[ProjectIdentity] = None
    end_user: Optional[EndUserIdentity] = None

    roles: List[Role] = Field(default_factory=list)
    scopes: List[str] = Field(default_factory=list)

    auth_method: AuthMethod
    credential_ref: CredentialRef = Field(default_factory=CredentialRef)
    network: NetworkContext = Field(default_factory=NetworkContext)
    claims: Dict[str, Any] = Field(default_factory=dict)

    def has_required_scopes(self, security_scopes: SecurityScopes) -> bool:
        return set(security_scopes.scopes).issubset(self.scopes)
