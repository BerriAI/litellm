import argparse
import json
import os
from typing import Final

from mcp.server.fastmcp import FastMCP

mcp: Final = FastMCP("places_api")


@mcp.tool()
def getPlaces(query: str) -> str:
    """Find places matching a query"""
    return json.dumps([{"name": f"Blue Bottle Coffee ({query})", "rating": 4.6}])


def _parse_args() -> argparse.Namespace:
    parser: Final = argparse.ArgumentParser(description="Docs golden-path MCP server")
    parser.add_argument("--host", default=os.getenv("MCP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MCP_PORT", "0")))
    return parser.parse_args()


def main() -> None:
    args: Final = _parse_args()
    if args.port <= 0:
        raise ValueError("HTTP transport requires a valid --port value")
    mcp.settings.host = args.host
    mcp.settings.port = args.port
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
