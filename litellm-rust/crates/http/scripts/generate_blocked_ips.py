"""Regenerate generated/blocked_ips.json: what the Python SSRF policy says for each probe.

    uv run --no-sync python litellm-rust/crates/http/scripts/generate_blocked_ips.py \
        > litellm-rust/crates/http/generated/blocked_ips.json

Each row records whether ``litellm.litellm_core_utils.url_utils._is_blocked_ip`` blocks the
address under the interpreter named in the header. The verdicts depend on that interpreter's
``ipaddress`` tables, so the file is regenerated, never hand-edited.
"""

import json
import platform
import sys
from ipaddress import ip_network
from typing import Final

from litellm.litellm_core_utils.url_utils import _is_blocked_ip

# IANA special-purpose ranges plus the cloud-fabric exceptions, probed at both edges and one
# address past each edge so a prefix length that is off by one shows up.
NETWORKS: Final = (
    "0.0.0.0/8",
    "10.0.0.0/8",
    "100.64.0.0/10",
    "127.0.0.0/8",
    "168.63.129.16/32",
    "169.254.0.0/16",
    "172.16.0.0/12",
    "192.0.0.0/24",
    "192.0.0.9/32",
    "192.0.0.10/32",
    "192.0.0.170/31",
    "192.0.2.0/24",
    "192.31.196.0/24",
    "192.52.193.0/24",
    "192.88.99.0/24",
    "192.168.0.0/16",
    "192.175.48.0/24",
    "198.18.0.0/15",
    "198.51.100.0/24",
    "203.0.113.0/24",
    "224.0.0.0/4",
    "240.0.0.0/4",
    "255.255.255.255/32",
    "::/128",
    "::1/128",
    "::ffff:0:0/96",
    "64:ff9b::/96",
    "64:ff9b:1::/48",
    "100::/64",
    "2001::/23",
    "2001::/32",
    "2001:1::1/128",
    "2001:1::2/128",
    "2001:2::/48",
    "2001:3::/32",
    "2001:4:112::/48",
    "2001:10::/28",
    "2001:20::/28",
    "2001:30::/28",
    "2001:db8::/32",
    "2002::/16",
    "3fff::/20",
    "5f00::/16",
    "fc00::/7",
    "fe80::/10",
    "fec0::/10",
    "ff00::/8",
)

# Addresses inside the ranges that the module's own tests name, plus mapped and compat forms.
SPOT_CHECKS: Final = (
    "8.8.8.8",
    "1.1.1.1",
    "168.63.129.17",
    "100.128.0.1",
    "223.255.255.255",
    "2606:4700::1111",
    "2001:200::1",
    "2001:4860:4860::8888",
    "2002:c000:204::1",
    "2002:0a00:1::",
    "2002:808:808::1",
    "::ffff:127.0.0.1",
    "::ffff:168.63.129.16",
    "::ffff:10.0.0.1",
    "::ffff:8.8.8.8",
    "::ffff:169.254.169.254",
    "::127.0.0.1",
    "::8.8.8.8",
    "64:ff9b::808:808",
    "64:ff9b::a00:1",
)


def _edges(network: str) -> tuple[str, ...]:
    net: Final = ip_network(network)
    first: Final = net.network_address
    last: Final = net.broadcast_address
    below: Final = first - 1 if int(first) > 0 else None
    above: Final = last + 1 if int(last) < 2**net.max_prefixlen - 1 else None
    return tuple(str(address) for address in (below, first, last, above) if address is not None)


def _probes() -> tuple[str, ...]:
    edges: Final = tuple(address for network in NETWORKS for address in _edges(network))
    return tuple(dict.fromkeys(edges + SPOT_CHECKS))


def main() -> None:
    rows: Final = tuple({"address": address, "blocked": _is_blocked_ip(address)} for address in _probes())
    json.dump(
        {"python": platform.python_version(), "rows": rows},
        sys.stdout,
        indent=2,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
