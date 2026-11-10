from collections.abc import Sequence
from ipaddress import IPv4Address, IPv6Address, ip_address
from typing import Final

from litellm.proxy.auth.core.inputs import TrustedNetwork


def _unmapped(address: IPv4Address | IPv6Address) -> IPv4Address | IPv6Address:
    if isinstance(address, IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    return address


def ip_in_networks(client_ip: str | None, networks: Sequence[TrustedNetwork]) -> bool:
    if not client_ip or not networks:
        return False
    try:
        address: Final = ip_address(client_ip.strip())
    except ValueError:
        return False
    candidates: Final = (address, _unmapped(address))
    return any(
        candidate in network for network in networks for candidate in candidates
    )  # comprehension-ok: two fixed candidates per network
