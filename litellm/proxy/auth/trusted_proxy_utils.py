from typing import Any, Dict, Optional

from fastapi import Request

from litellm._logging import verbose_proxy_logger
from litellm.proxy.auth.network import (
    ip_in_networks,
    normalize_cidr_ranges,
    parse_trusted_proxy_ranges,
)

TRUSTED_PROXY_RANGES_KEY = "trusted_proxy_ranges"


def _get_proxy_general_settings() -> Dict[str, Any]:
    try:
        from litellm.proxy.proxy_server import general_settings

        return general_settings or {}
    except ImportError:
        return {}


def get_trusted_proxy_cidrs(
    general_settings: dict[str, Any] | None = None,
) -> list[str]:
    """Operator-configured trusted reverse-proxy CIDRs, normalized to strings.

    Empty when none are configured, in which case X-Forwarded-For must not be
    trusted and only the direct peer is authoritative.
    """
    if general_settings is None:
        general_settings = _get_proxy_general_settings()
    return normalize_cidr_ranges(
        general_settings.get(TRUSTED_PROXY_RANGES_KEY),
        setting_name=TRUSTED_PROXY_RANGES_KEY,
    )


def _get_direct_client_ip(request: Request) -> Optional[str]:
    client = getattr(request, "client", None)
    client_host = getattr(client, "host", None)
    if isinstance(client_host, str):
        return client_host
    return None


def require_trusted_proxy_request(
    *,
    request: Request,
    general_settings: Optional[Dict[str, Any]] = None,
    feature_name: str,
    setting_name: str = TRUSTED_PROXY_RANGES_KEY,
) -> None:
    """
    Fail closed unless the direct TCP peer is one of the configured
    trusted reverse proxies.

    Header-based auth paths must validate the direct peer, not
    X-Forwarded-For, because the direct peer is the actor supplying the
    identity headers.
    """
    if general_settings is None:
        general_settings = _get_proxy_general_settings()

    trusted_networks = parse_trusted_proxy_ranges(general_settings.get(setting_name), setting_name=setting_name)
    if not trusted_networks:
        raise ValueError(
            f"{feature_name} requires general_settings.{setting_name} before "
            "trusting identity headers from an upstream proxy."
        )

    direct_client_ip = _get_direct_client_ip(request)
    if not ip_in_networks(direct_client_ip, trusted_networks):
        verbose_proxy_logger.warning(
            "%s rejected identity headers from untrusted direct client IP %r",
            feature_name,
            direct_client_ip,
        )
        raise ValueError(
            f"{feature_name} only accepts identity headers from configured "
            f"trusted proxy ranges. Direct client IP {direct_client_ip!r} "
            "is not trusted."
        )
