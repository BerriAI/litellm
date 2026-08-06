import ipaddress
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit, urlunsplit


def _boolean(value: str | None) -> bool:
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


def _positive_integer(value: str | None, default: int) -> int:
    resolved = default if value is None else int(value)
    if resolved <= 0:
        raise ValueError("gateway limits must be positive")
    return resolved


def _loopback_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None or parsed.query or parsed.fragment:
        raise ValueError("HEADROOM_BASE_URL must be an HTTP(S) base URL without a query or fragment")
    hostname = parsed.hostname
    is_loopback = hostname == "localhost"
    if not is_loopback:
        try:
            is_loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            is_loopback = False
    if not is_loopback:
        raise ValueError("HEADROOM_BASE_URL must point to a loopback Headroom instance")
    return value.rstrip("/")


def _readiness_url(base_url: str, readiness_path: str) -> str:
    parsed = urlsplit(base_url)
    path = readiness_path if readiness_path.startswith("/") else f"/{readiness_path}"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


@dataclass(frozen=True, slots=True)
class GatewaySettings:
    local_key: str
    headroom_base_url: str = "http://127.0.0.1:8787/v1"
    readiness_path: str = "/livez"
    capture_enabled: bool = False
    trace_directory: Path = Path("~/.local/state/litellm-codex-gateway/traces")
    max_trace_bytes: int = 10 * 1024 * 1024
    max_trace_storage_bytes: int = 100 * 1024 * 1024
    trace_retention_seconds: int = 7 * 24 * 60 * 60

    def __post_init__(self) -> None:
        if not self.local_key:
            raise ValueError("LOCAL_CODEX_GATEWAY_KEY is required")
        object.__setattr__(self, "headroom_base_url", _loopback_base_url(self.headroom_base_url))
        object.__setattr__(self, "trace_directory", self.trace_directory.expanduser())

    @property
    def readiness_url(self) -> str:
        return _readiness_url(self.headroom_base_url, self.readiness_path)

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> "GatewaySettings":
        source = os.environ if environment is None else environment
        return cls(
            local_key=source.get("LOCAL_CODEX_GATEWAY_KEY", ""),
            headroom_base_url=source.get("HEADROOM_BASE_URL", "http://127.0.0.1:8787/v1"),
            readiness_path=source.get("HEADROOM_READINESS_PATH", "/livez"),
            capture_enabled=_boolean(source.get("CODEX_GATEWAY_CAPTURE")),
            trace_directory=Path(source.get("CODEX_GATEWAY_TRACE_DIR", "~/.local/state/litellm-codex-gateway/traces")),
            max_trace_bytes=_positive_integer(source.get("CODEX_GATEWAY_MAX_TRACE_BYTES"), 10 * 1024 * 1024),
            max_trace_storage_bytes=_positive_integer(
                source.get("CODEX_GATEWAY_MAX_TRACE_STORAGE_BYTES"), 100 * 1024 * 1024
            ),
            trace_retention_seconds=_positive_integer(
                source.get("CODEX_GATEWAY_TRACE_RETENTION_SECONDS"), 7 * 24 * 60 * 60
            ),
        )
