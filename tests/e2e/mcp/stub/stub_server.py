"""Deterministic MCP upstream for the mcp e2e suite.

Serves the streamable-http MCP protocol with three tools. `echo` answers
immediately so auth tests can assert an exact round-trip. `slow_echo` holds the
request open for `sleep_seconds` while per-`marker` in-flight and max-in-flight
counters track how many calls the proxy let through simultaneously; that is the
observable a per-server `max_concurrent_requests` cap must bound. `stats` reads
those counters back, so tests observe upstream concurrency through the proxy
itself and the stub needs no side-channel port.

Counter updates are plain attribute mutations between awaits, so asyncio's
single-threaded scheduling makes them atomic; markers come from
`unique_marker()` so concurrent test runs never share a counter.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("e2e-stub", host="0.0.0.0", port=8765, stateless_http=True)


@dataclass
class _MarkerStats:
    in_flight: int = 0
    max_in_flight: int = 0
    completed: int = 0


_stats: dict[str, _MarkerStats] = {}


@mcp.tool()
def echo(text: str) -> str:
    """Return `text` unchanged."""
    return text


@mcp.tool()
async def slow_echo(text: str, marker: str, sleep_seconds: float) -> str:
    """Return `text` after `sleep_seconds`, recording concurrency under `marker`."""
    stats = _stats.setdefault(marker, _MarkerStats())
    stats.in_flight += 1
    stats.max_in_flight = max(stats.max_in_flight, stats.in_flight)
    try:
        await asyncio.sleep(sleep_seconds)
    finally:
        stats.in_flight -= 1
        stats.completed += 1
    return text


@mcp.tool()
def stats(marker: str) -> str:
    """Return the JSON stats recorded for `marker`."""
    recorded = _stats.get(marker, _MarkerStats())
    return json.dumps(
        {
            "marker": marker,
            "max_in_flight": recorded.max_in_flight,
            "completed": recorded.completed,
        }
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
