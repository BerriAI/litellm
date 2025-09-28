"""Mini-agent public surface and event-loop guards.

This module exports the key mini-agent types and functions used in smokes and
lightweight integrations, while also ensuring an event loop exists in
environments that still call `asyncio.get_event_loop()` directly (e.g., some
test runners).
"""

from __future__ import annotations

# Minimal namespace for mini-agent helpers used in smokes.
# Ensure an event loop exists for tests that call asyncio.get_event_loop().run_until_complete(...)
import asyncio  # noqa: E402

# Normalize policy first (helps Python 3.12+ behavior)
try:
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
except Exception:
    pass

try:
    asyncio.get_running_loop()
except RuntimeError:
    try:
        _loop_pkg = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop_pkg)
    except Exception:
        # Best-effort; test runners may manage the loop differently
        pass

# Final guard for get_event_loop() callers
try:
    asyncio.get_event_loop()
except RuntimeError:
    try:
        _loop_pkg2 = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop_pkg2)
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
