"""foundation — MCP Gateway v2

S0 · chassis LEAF. Frozen types, Result/errors, GatewayDeps, naming. No I/O, no logic. Importable by anyone.
"""

from litellm.proxy.gateway.mcp.foundation.deps import (
    Cache,
    Clock,
    GatewayDeps,
    HttpxFactory,
    build_test_deps,
)
from litellm.proxy.gateway.mcp.foundation.errors import GatewayError, reason
from litellm.proxy.gateway.mcp.foundation.naming import (
    is_valid_name,
    namespace_tool,
    split_namespaced,
)
from litellm.proxy.gateway.mcp.foundation.result import (
    Error,
    GatewayResult,
    Ok,
    Result,
)
from litellm.proxy.gateway.mcp.foundation.types import Subject

__all__ = [
    "Cache",
    "Clock",
    "Error",
    "GatewayDeps",
    "GatewayError",
    "GatewayResult",
    "HttpxFactory",
    "Ok",
    "Result",
    "Subject",
    "build_test_deps",
    "is_valid_name",
    "namespace_tool",
    "reason",
    "split_namespaced",
]
