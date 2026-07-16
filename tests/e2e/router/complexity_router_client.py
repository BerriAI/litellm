"""Client for the complexity auto-router e2e tests.

The suite drives the shared /chat/completions and spend-log reads on the Gateway,
so this client only carries the Gateway the shared lifecycle needs for cleanup.
"""

from __future__ import annotations

from dataclasses import dataclass

from e2e_gateway import Gateway, build_gateway


@dataclass(frozen=True, slots=True)
class ComplexityRouterClient:
    gateway: Gateway


def build_client() -> ComplexityRouterClient:
    return ComplexityRouterClient(gateway=build_gateway())
