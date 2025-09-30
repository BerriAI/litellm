"""Mini-agent public surface plus a minimal event-loop fallback.

This module exposes the types needed by the deterministic mini-agent scenarios
and safeguards older harnesses that still rely on `asyncio.get_event_loop()`
without configuring a loop first. The fallback is intentionally minimal to
avoid the brittle layering that used to accumulate in this namespace.
"""

from __future__ import annotations

# Minimal namespace for the deterministic mini-agent harness.
# Ensure callers that reach for get_event_loop() on Python 3.12 still receive a loop.
import asyncio  # noqa: E402

try:  # pragma: no cover - safety belt for legacy harnesses
    asyncio.get_running_loop()
except RuntimeError:
    try:
        asyncio.set_event_loop(asyncio.new_event_loop())
    except Exception:
        pass

# Public exports from the mini-agent implementation
from .litellm_mcp_mini_agent import (
    AgentConfig,
    AgentRunResult,
    IterationRecord,
    MCPInvoker,
    EchoMCP,
    LocalMCPInvoker,
    arun_mcp_mini_agent,
    run_mcp_mini_agent,
)

try:  # optional dependency (httpx)
    from .http_tools_invoker import HttpToolsInvoker  # type: ignore
except Exception:  # pragma: no cover - optional export
    HttpToolsInvoker = None  # type: ignore

__all__ = [
    "AgentConfig",
    "AgentRunResult",
    "IterationRecord",
    "MCPInvoker",
    "EchoMCP",
    "LocalMCPInvoker",
    "HttpToolsInvoker",
    "arun_mcp_mini_agent",
    "run_mcp_mini_agent",
]
