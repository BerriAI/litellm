import ipaddress
from typing import List, Optional
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

from litellm.types.proxy.control_plane_endpoints import WorkerRegistryEntry


def _has_control_characters(value: str) -> bool:
    return any(ord(character) < 32 or 127 <= ord(character) <= 159 for character in value)


class NativeOIDCConfig(BaseModel):
    discovery_url: str
    client_id: str
    scopes: List[str]

    model_config = ConfigDict(extra="forbid")

    @field_validator("discovery_url")
    @classmethod
    def validate_discovery_url(cls, value: str) -> str:
        try:
            parsed = urlsplit(value)
            _ = parsed.port
        except ValueError as error:
            raise ValueError("must contain a valid port") from error
        if (
            not value.strip()
            or _has_control_characters(value)
            or any(character.isspace() for character in value)
            or parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("must be an absolute OIDC discovery URL without credentials, query, or fragment")
        if parsed.scheme == "http":
            try:
                is_loopback = ipaddress.ip_address(parsed.hostname).is_loopback
            except ValueError:
                is_loopback = False
            if not is_loopback:
                raise ValueError("must use HTTPS unless the host is a loopback IP address")
        return value

    @field_validator("client_id")
    @classmethod
    def validate_client_id(cls, value: str) -> str:
        if not value.strip() or _has_control_characters(value):
            raise ValueError("must not be blank")
        return value

    @field_validator("scopes")
    @classmethod
    def validate_scopes(cls, value: List[str]) -> List[str]:
        if not value or any(not scope.strip() or _has_control_characters(scope) for scope in value):
            raise ValueError("must contain only non-blank scopes")
        return value


class UiDiscoveryEndpoints(BaseModel):
    server_root_path: str
    proxy_base_url: Optional[str]
    auto_redirect_to_sso: bool
    admin_ui_disabled: bool
    sso_configured: bool
    hide_default_credentials_hint: bool = False
    is_control_plane: bool = False
    workers: List[WorkerRegistryEntry] = []
    native_oidc: Optional[NativeOIDCConfig] = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
