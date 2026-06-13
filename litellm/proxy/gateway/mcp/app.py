"""Composition root for MCP Gateway v2.

`build_gateway(deps)` is the single place the whole object graph is assembled
(Seemann's composition root). Everything else receives its dependencies by
constructor injection — no module-level singletons, no service locator.

This is an S0 stub: it wires nothing yet. Each subsequent section fills in a
real handler and registers it here exactly once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from starlette.applications import Starlette

    from litellm.proxy.gateway.mcp.foundation.deps import GatewayDeps


def build_gateway(deps: "GatewayDeps") -> "Starlette":
    """Assemble and return the gateway ASGI app from injected dependencies.

    The ONLY construction site. Each surface module exposes a single
    ``register(app)`` verb; this function calls each one once.
    """
    raise NotImplementedError("S0 stub — build_gateway is wired up starting in S1")
