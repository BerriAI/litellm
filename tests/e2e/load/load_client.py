from __future__ import annotations

from dataclasses import dataclass

from proxy_client import ProxyClient, build_proxy_client


@dataclass(frozen=True, slots=True)
class LoadClient:
    gateway: ProxyClient


def build_client() -> LoadClient:
    return LoadClient(gateway=build_proxy_client())
