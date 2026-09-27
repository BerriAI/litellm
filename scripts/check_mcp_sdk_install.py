import argparse
import importlib
import importlib.metadata
import sys
from typing import Final

MINIMUM_MCP_VERSION: Final[tuple[int, int, int]] = (2, 2, 0)

IMPORTED_MODULES: Final[tuple[str, ...]] = (
    "litellm",
    "litellm.experimental_mcp_client",
    "litellm.experimental_mcp_client.client",
    "litellm.proxy._experimental.mcp_server.server",
    "litellm.proxy._experimental.mcp_server.mcp_server_manager",
    "litellm.proxy._experimental.mcp_server.rest_endpoints",
)


def _version_tuple(distribution: str) -> tuple[int, ...]:
    return tuple(int(part) for part in importlib.metadata.version(distribution).split(".") if part.isdigit())


def main() -> int:
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("--extra", choices=("mcp", "proxy"), default="proxy")
    extra: Final = parser.parse_args().extra
    for module_name in IMPORTED_MODULES if extra == "proxy" else IMPORTED_MODULES[:3]:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            sys.stderr.write(f"failed to import {module_name}: {exc}\n")
            return 1

    mcp_version: Final = _version_tuple("mcp")
    if mcp_version < MINIMUM_MCP_VERSION:
        sys.stderr.write(f"mcp {importlib.metadata.version('mcp')} below floor 2.2.0\n")
        return 1

    from mcp_types.version import HANDSHAKE_PROTOCOL_VERSIONS

    for required in ("2024-11-05", "2025-06-18"):
        if required not in HANDSHAKE_PROTOCOL_VERSIONS:
            sys.stderr.write(f"HANDSHAKE_PROTOCOL_VERSIONS missing {required}\n")
            return 1

    if extra == "proxy":
        scope: Final = {
            "type": "http",
            "method": "POST",
            "path": "/mcp",
            "headers": [(b"mcp-protocol-version", b"2026-07-28")],
        }
        mcp_server: Final = sys.modules["litellm.proxy._experimental.mcp_server.server"]
        if mcp_server.unsupported_protocol_version(scope) != "2026-07-28":
            sys.stderr.write("unsupported_protocol_version accepted a modern-only version\n")
            return 1
        if (
            mcp_server.unsupported_protocol_version(dict(scope, headers=[(b"mcp-protocol-version", b"2025-06-18")]))
            is not None
        ):
            sys.stderr.write("unsupported_protocol_version rejected a handshake version\n")
            return 1

    sys.stdout.write(
        "python {} mcp {} httpx2 {} pydantic {} litellm {}\n".format(
            sys.version.split()[0],
            importlib.metadata.version("mcp"),
            importlib.metadata.version("httpx2"),
            importlib.metadata.version("pydantic"),
            importlib.metadata.version("litellm"),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
