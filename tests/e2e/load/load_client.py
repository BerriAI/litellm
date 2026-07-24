from __future__ import annotations

from dataclasses import dataclass

from proxy_client import ProxyClient


@dataclass(frozen=True, slots=True)
class LoadClient:
    proxy: ProxyClient


def build_client(proxy: ProxyClient) -> LoadClient:
    return LoadClient(proxy=proxy)
