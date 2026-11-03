from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import Final

from fastapi import Request

from litellm._logging import verbose_proxy_logger
from litellm.types.llms.base import LiteLLMBaseModel

TrustedProxyNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network


class NetworkContext(LiteLLMBaseModel):
    client_ip: str | None = None
    host: str | None = None
    via_trusted_proxy: bool = False


PROXY_HEADER_CONTEXT: Final = "litellm.proxy_header_context"


@dataclass(frozen=True, slots=True)
class ProxyHeaderContext:
    direct_client: tuple[str, int] | None
    direct_scheme: str
    configured: bool
    via_trusted_proxy: bool


def proxy_header_context(request: Request) -> ProxyHeaderContext | None:
    context: Final[object] = request.get(PROXY_HEADER_CONTEXT)
    return context if isinstance(context, ProxyHeaderContext) else None


def get_direct_client_ip(request: Request) -> str | None:
    context: Final = proxy_header_context(request)
    if context is not None:
        return context.direct_client[0] if context.direct_client else None
    return request.client.host if request.client else None


def get_direct_request(request: Request) -> Request:
    context: Final = proxy_header_context(request)
    if context is None:
        return request
    scope: Final = {key: value for key, value in request.scope.items() if key != PROXY_HEADER_CONTEXT}
    return Request({**scope, "client": context.direct_client, "scheme": context.direct_scheme})


def normalize_cidr_ranges(configured_ranges: object, *, setting_name: str = "trusted_proxy_cidrs") -> list[str]:
    if not configured_ranges:
        return []
    if isinstance(configured_ranges, str):
        return [r.strip() for r in configured_ranges.split(",") if r.strip()]
    if isinstance(configured_ranges, (list, tuple, set)):
        return [str(r).strip() for r in configured_ranges if str(r).strip()]
    verbose_proxy_logger.warning(
        "Invalid %s value: expected a list of CIDR ranges, got %s",
        setting_name,
        type(configured_ranges).__name__,
    )
    return []


def parse_trusted_proxy_ranges(
    configured_ranges: object, *, setting_name: str = "trusted_proxy_cidrs"
) -> list[TrustedProxyNetwork]:
    networks: Final[list[TrustedProxyNetwork]] = []
    for cidr in normalize_cidr_ranges(configured_ranges, setting_name=setting_name):
        try:
            networks.append(ipaddress.ip_network(cidr, strict=False))
        except ValueError:
            verbose_proxy_logger.warning("Invalid CIDR in %s: %s, skipping", setting_name, cidr)
    return networks


def _unmapped(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        return addr.ipv4_mapped
    return addr


def ip_in_networks(client_ip: str | None, networks: list[TrustedProxyNetwork]) -> bool:
    if not client_ip or not networks:
        return False
    try:
        addr: Final = ipaddress.ip_address(client_ip.strip())
    except ValueError:
        return False
    candidates: Final = (addr, _unmapped(addr))
    return any(candidate in network for candidate in candidates for network in networks)


def resolve_network_context(request: Request) -> NetworkContext:
    context: Final = proxy_header_context(request)
    return NetworkContext(
        client_ip=request.client.host if request.client else None,
        host=request.headers.get("host"),
        via_trusted_proxy=context.via_trusted_proxy if context else False,
    )
