"""
CLI entry point for the LiteLLM MCP server.

Supports stdio (default) and HTTP transports.

Examples:
    # stdio transport (for Claude Desktop, Cursor, etc.)
    litellm-mcp-server

    # HTTP transport
    litellm-mcp-server --transport http --host 0.0.0.0 --port 8000
"""

import argparse
import asyncio
import logging
import sys


def _parse_args(argv: list | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LiteLLM MCP Server - Expose LiteLLM as MCP tools",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport type (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to for HTTP transport (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to for HTTP transport (default: 8000)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    return parser.parse_args(argv)


async def _run_stdio(server) -> None:
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


async def _run_http(server, host: str, port: int) -> None:
    from starlette.applications import Starlette
    from starlette.routing import Mount

    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    session_manager = StreamableHTTPSessionManager(
        app=server,
        event_store=None,
        json_response=False,
        stateless=True,
    )

    async with session_manager.run():
        app = Starlette(
            routes=[
                Mount("/mcp", app=session_manager.handle_request),
            ],
        )
        import uvicorn

        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        uv_server = uvicorn.Server(config)
        await uv_server.serve()


def main(argv: list | None = None) -> None:
    args = _parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    from litellm.litellm_mcp_server.server import create_mcp_server

    server = create_mcp_server()

    if args.transport == "stdio":
        asyncio.run(_run_stdio(server))
    else:
        asyncio.run(_run_http(server, args.host, args.port))


if __name__ == "__main__":
    main()
