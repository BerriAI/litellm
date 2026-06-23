"""Typed upstream-credential resolution for MCP servers.

This subpackage will house ``resolve_credentials`` and its per-mode typed configs.
Failures are modeled as values via :mod:`.result` (``Result[T, CredError]``) rather
than raised, so every seam is total. Nothing here is wired onto a live request path
yet; later PRs add the typed vocabulary, the resolver, and the v1 graft.
"""

from litellm.proxy._experimental.mcp_server.outbound_credentials.result import (
    Error,
    Ok,
    Result,
)

__all__ = ["Ok", "Error", "Result"]
