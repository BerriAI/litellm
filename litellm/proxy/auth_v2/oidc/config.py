from typing import List, Optional

from pydantic import AnyHttpUrl, BaseModel, Field, SecretStr, field_validator

from ..models import require_secure_url


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
    allowed_roles: List[str] = Field(default_factory=list)
    allow_platform_roles: bool = False

    @field_validator("issuer")
    @classmethod
    def _issuer_https(cls, value: str) -> str:
        return require_secure_url(value)

    @field_validator("jwks_uri")
    @classmethod
    def _jwks_https(cls, value: Optional[AnyHttpUrl]) -> Optional[AnyHttpUrl]:
        if value is not None:
            require_secure_url(str(value))
        return value
