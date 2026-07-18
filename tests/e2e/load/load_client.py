from __future__ import annotations

from dataclasses import dataclass

from e2e_gateway import Gateway, build_gateway


@dataclass(frozen=True, slots=True)
class LoadClient:
    gateway: Gateway


def build_client() -> LoadClient:
    return LoadClient(gateway=build_gateway())
