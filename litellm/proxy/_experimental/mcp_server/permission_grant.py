"""Parse raw ``mcp_servers`` permission lists into a tagged grant union.

This module is pure: it only knows about the sentinel strings via the
``SpecialMCPServerNames`` enum and never touches the server manager, the DB, or
the proxy server. Downstream resolvers turn an ``ExplicitServers`` set of
identifiers into concrete deployments.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Union

from litellm.proxy._types import SpecialMCPServerNames

MCP_GRANT_SENTINELS: frozenset[str] = frozenset(
    {
        SpecialMCPServerNames.no_mcp_servers.value,
        SpecialMCPServerNames.all_proxy_mcp_servers.value,
    }
)


@dataclass(frozen=True, slots=True)
class AllServers:
    """Grant every MCP server on the proxy, including ones added later."""


@dataclass(frozen=True, slots=True)
class NoServers:
    """Block all MCP servers."""


@dataclass(frozen=True, slots=True)
class ExplicitServers:
    """Grant a specific, unresolved set of server_ids / aliases / names."""

    identifiers: frozenset[str]


MCPServerGrant = Union[AllServers, NoServers, ExplicitServers]


def parse_mcp_server_grant(raw: Iterable[str]) -> MCPServerGrant:
    """Resolve precedence once, most-restrictive first.

    ``no-mcp-servers`` blocks everything and beats ``all-proxy-mcps``; absent any
    sentinel the entries are explicit identifiers.
    """
    values = frozenset(raw)
    if SpecialMCPServerNames.no_mcp_servers.value in values:
        return NoServers()
    if SpecialMCPServerNames.all_proxy_mcp_servers.value in values:
        return AllServers()
    return ExplicitServers(values)
