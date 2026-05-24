"""Environment-driven configuration for the LiteLLM provisioning MCP server.

Every setting is read once at startup from the process environment. Required
OAuth settings raise at load time so a misconfigured deployment fails fast
instead of accepting unauthenticated traffic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


class ConfigError(RuntimeError):
    """Raised when a required setting is missing or malformed."""


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(f"environment variable {name} is required")
    return value


def _optional(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"environment variable {name} must be an integer") from exc


@dataclass(frozen=True)
class Settings:
    # ---- OAuth 2.0 resource-server validation ----
    oauth_jwks_url: str
    oauth_issuer: str
    oauth_audience: str
    oauth_required_scope: str
    oauth_algorithms: tuple[str, ...]
    resource_server_url: str

    # ---- provisioning targets ----
    namespace: str
    chart_path: str
    default_image_registry: str
    release_prefix: str
    helm_binary: str
    kubectl_binary: str
    command_timeout: int

    # ---- HTTP server ----
    host: str
    port: int
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        algorithms = tuple(
            alg.strip()
            for alg in _optional("OAUTH_ALGORITHMS", "RS256").split(",")
            if alg.strip()
        )
        if not algorithms:
            raise ConfigError("OAUTH_ALGORITHMS must list at least one algorithm")

        return cls(
            oauth_jwks_url=_require("OAUTH_JWKS_URL"),
            oauth_issuer=_require("OAUTH_ISSUER"),
            oauth_audience=_require("OAUTH_AUDIENCE"),
            oauth_required_scope=_optional("OAUTH_REQUIRED_SCOPE", "litellm:provision"),
            oauth_algorithms=algorithms,
            resource_server_url=_require("MCP_RESOURCE_SERVER_URL"),
            namespace=_optional("LITELLM_NAMESPACE", "litellm"),
            chart_path=_optional("HELM_CHART_PATH", "/app/helm/litellm"),
            default_image_registry=_optional(
                "LITELLM_IMAGE_REGISTRY", "ghcr.io/berriai"
            ),
            release_prefix=_optional("LITELLM_RELEASE_PREFIX", "litellm-e2e"),
            helm_binary=_optional("HELM_BINARY", "helm"),
            kubectl_binary=_optional("KUBECTL_BINARY", "kubectl"),
            command_timeout=_int("COMMAND_TIMEOUT_SECONDS", 600),
            host=_optional("MCP_HOST", "0.0.0.0"),
            port=_int("MCP_PORT", 8080),
            log_level=_optional("MCP_LOG_LEVEL", "INFO").upper(),
        )
