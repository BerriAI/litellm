"""Client for the complexity auto-router e2e tests.

The suite drives the shared /chat/completions and spend-log reads on the ProxyClient,
so this client only carries the ProxyClient the shared lifecycle needs for cleanup.
"""

from __future__ import annotations

from dataclasses import dataclass

from proxy_client import ProxyClient


@dataclass(frozen=True, slots=True)
class ComplexityRouterClient:
    proxy: ProxyClient


def build_client(proxy: ProxyClient) -> ComplexityRouterClient:
    return ComplexityRouterClient(proxy=proxy)
