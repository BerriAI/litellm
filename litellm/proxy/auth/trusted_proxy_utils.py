import ipaddress
from typing import Any, Dict, List, Optional, Union

from fastapi import Request

from litellm._logging import verbose_proxy_logger

TRUSTED_PROXY_RANGES_KEY = "trusted_proxy_ranges"
TrustedProxyNetwork = Union[ipaddress.IPv4Network, ipaddress.IPv6Network]


def _get_proxy_general_settings() -> Dict[str, Any]:
    try:
        from litellm.proxy.proxy_server import general_settings

        return general_settings or {}
    except ImportError:
        return {}


def _normalize_cidr_ranges(configured_ranges: Any, *, setting_name: str) -> List[str]:
    if not configured_ranges:
        return []
    if isinstance(configured_ranges, str):
        return [
            raw_range.strip()
            for raw_range in configured_ranges.split(",")
            if raw_range.strip()
        ]
    if isinstance(configured_ranges, (list, tuple, set)):
        return [
            str(raw_range).strip()
            for raw_range in configured_ranges
            if str(raw_range).strip()
        ]
    verbose_proxy_logger.warning(
        "Invalid %s value: expected a list of CIDR ranges, got %s",
        setting_name,
        type(configured_ranges).__name__,
    )
    return []


def parse_trusted_proxy_ranges(
    configured_ranges: Any,
    *,
    setting_name: str = TRUSTED_PROXY_RANGES_KEY,
) -> List[TrustedProxyNetwork]:
    networks: List[TrustedProxyNetwork] = []
    for cidr in _normalize_cidr_ranges(configured_ranges, setting_name=setting_name):
        try:
            networks.append(ipaddress.ip_network(cidr, strict=False))
        except ValueError:
            verbose_proxy_logger.warning(
                "Invalid CIDR in %s: %s, skipping", setting_name, cidr
            )
    return networks


def _get_direct_client_ip(request: Request) -> Optional[str]:
    client = getattr(request, "client", None)
    client_host = getattr(client, "host", None)
    if isinstance(client_host, str):
        return client_host
    return None


def _is_ip_in_networks(
    client_ip: Optional[str], networks: List[TrustedProxyNetwork]
) -> bool:
    if not client_ip or not networks:
        return False
    try:
        addr = ipaddress.ip_address(client_ip.strip())
    except ValueError:
        return False
    return any(addr in network for network in networks)


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

    trusted_networks = parse_trusted_proxy_ranges(
        general_settings.get(setting_name), setting_name=setting_name
    )
    if not trusted_networks:
        raise ValueError(
            f"{feature_name} requires general_settings.{setting_name} before "
            "trusting identity headers from an upstream proxy."
        )

    direct_client_ip = _get_direct_client_ip(request)
    if not _is_ip_in_networks(direct_client_ip, trusted_networks):
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
