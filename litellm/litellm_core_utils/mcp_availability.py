"""Availability guard for the MCP gateway (a proxy-extra dependency)."""

from litellm._logging import verbose_logger


def mcp_gateway_importable() -> bool:
    # The gateway's import chain needs the proxy-extra `mcp` package and dies at
    # `from mcp.types import ...` both when `mcp` is absent (base installs) and on
    # mcp 1.x (which has no CacheableResult); ordinary tool calls must skip it in
    # both cases, so probe the real import instead of just the package's presence.
    try:
        import litellm.responses.mcp.chat_completions_handler  # noqa: F401  # pyright: ignore[reportUnusedImport]  # probe only: the import succeeding IS the check
    except ImportError:
        verbose_logger.debug(
            "MCP gateway skipped: its import chain is unavailable (proxy-extra `mcp` missing or incompatible)"
        )
        return False
    return True
