# math_server.py
import argparse
import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCP math test server")
    parser.add_argument(
        "--transport",
        default=os.getenv("MCP_TRANSPORT", "stdio"),
        help="Transport to use (stdio or http)",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("MCP_HOST", "127.0.0.1"),
        help="Host to bind when serving over HTTP",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("MCP_PORT", "0")),
        help="Port to bind when serving over HTTP",
    )
    return parser.parse_args()


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


def main() -> None:
    args = _parse_args()
    transport = (args.transport or "stdio").lower()

    if transport == "stdio":
        mcp.run(transport="stdio")
        return

    if transport in {"http", "streamable_http", "streamable-http"}:
        if args.port <= 0:
            raise ValueError("HTTP transport requires a valid --port value")
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")
        return

    raise ValueError(f"Unsupported transport: {transport}")


if __name__ == "__main__":
    main()
