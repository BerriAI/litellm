"""Deterministic upstream MCP server for the mcp e2e suite.

A tiny FastMCP server exposing `add` and `multiply` over streamable-http so the
suite has a self-hosted, offline upstream to register and exercise. DNS-rebinding
protection is turned off because the litellm container reaches this over the
compose network by service name (`mcp-upstream:8090`), not localhost, and the
stack is an isolated throwaway. Bind host/port come from MCP_HOST/MCP_PORT.
"""

import os

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

mcp: FastMCP = FastMCP(
    "e2e-math",
    host=os.getenv("MCP_HOST", "0.0.0.0"),
    port=int(os.getenv("MCP_PORT", "8090")),
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers"""
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two integers"""
    return a * b


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
