"""Client for the throughput load test.

The suite drives raw /chat/completions traffic through Locust (its own HTTP
client, the one exception to the shared-transport rule, since a load generator
must measure request throughput itself) and only uses the shared Gateway to
create the key and the mock deployment it hammers, so this client just carries
the Gateway the shared lifecycle needs for cleanup.
"""

from __future__ import annotations

from dataclasses import dataclass

from e2e_gateway import Gateway, build_gateway


@dataclass(frozen=True, slots=True)
class LoadClient:
    gateway: Gateway


def build_client() -> LoadClient:
    return LoadClient(gateway=build_gateway())
