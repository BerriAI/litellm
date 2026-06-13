"""routes — pure routing parser for the transport edge.

No I/O. Maps a request path to a ``RouteTarget`` (or ``None`` when the path is
not an MCP endpoint). Tenant is deliberately NOT in the URL: it rides in on the
resolved ``Subject``. ``RouteTarget`` is a frozen pydantic model so later
sections can add fields (e.g. an explicit toolset) without breaking callers.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from litellm.proxy.gateway.mcp.foundation import is_valid_name


class RouteTarget(BaseModel):
    model_config = ConfigDict(frozen=True)

    server: str | None


def parse_route(path: str) -> RouteTarget | None:
    """Parse an ASGI path into a ``RouteTarget``.

    ``/mcp`` (aggregated) -> ``server=None``; ``/{server}/mcp`` -> that server.
    A trailing slash is accepted on both. Anything else returns ``None`` (not an
    MCP endpoint), so the transport can reply 404. The server segment must be a
    single SEP-986 name; a malformed alias is rejected at the edge rather than
    forwarded to the registry.
    """
    segments = [s for s in path.strip("/").split("/") if s != ""]
    if segments == ["mcp"]:
        return RouteTarget(server=None)
    if len(segments) == 2 and segments[1] == "mcp" and is_valid_name(segments[0]):
        return RouteTarget(server=segments[0])
    return None
