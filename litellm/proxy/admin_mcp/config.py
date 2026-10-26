import ipaddress
import os
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlsplit

from .catalog import BY_NAME


def _loopback(hostname: str | None) -> bool:
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname or "").is_loopback
    except ValueError:
        return False


def gateway_origin(value: str) -> str:
    origin: Final = value.rstrip("/").removesuffix("/v1")
    parsed: Final = urlsplit(origin)
    if (
        not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path
        or parsed.query
        or parsed.fragment
        or any(c.isspace() for c in origin)
        or not (parsed.scheme == "https" or parsed.scheme == "http" and _loopback(parsed.hostname))
    ):
        raise ValueError(
            "LITELLM_BASE_URL must be an HTTPS gateway origin (optionally /v1); HTTP is allowed only on loopback."
        )
    try:
        _ = parsed.port
    except ValueError:
        raise ValueError("Invalid gateway port") from None
    return origin


def env_bool(name: str) -> bool:
    value: Final = os.getenv(name, "false").lower().strip()
    if value not in {"true", "false", "1", "0"}:
        raise ValueError(f"{name} must be true or false")
    return value in {"true", "1"}


@dataclass(frozen=True, slots=True)
class Config:
    base_url: str
    read_only: bool = False
    allowed_tools: frozenset[str] = frozenset()
    public_url: str = ""
    response_view: str = "compact"
    schema_mode: str = "full"

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", gateway_origin(self.base_url))
        if self.response_view not in {"full", "compact"}:
            raise ValueError("LITELLM_ADMIN_RESPONSE_VIEW must be full or compact")
        if self.schema_mode not in {"full", "discovery"}:
            raise ValueError("LITELLM_ADMIN_SCHEMA_MODE must be full or discovery")
        if self.allowed_tools - BY_NAME.keys():
            raise ValueError("LITELLM_ADMIN_TOOLS contains unknown tool names; inspect the published tool catalog.")
        if self.public_url:
            value: Final = gateway_origin(self.public_url)
            if value != self.public_url.rstrip("/") or not value.startswith("https://"):
                raise ValueError("LITELLM_MCP_PUBLIC_URL must be an HTTPS origin")
