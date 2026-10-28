import importlib
import re
from collections.abc import Iterable
from typing import Final

LITELLM_PROXY_MCP_SERVER_URL: Final = "litellm_proxy"
LITELLM_PROXY_MCP_SERVER_URL_PREFIX: Final = f"{LITELLM_PROXY_MCP_SERVER_URL}/mcp/"

_PROXY_MCP_PATH_RE: Final = re.compile(r"^https?://.+/mcp/([^/]+)$")
_GATEWAY_REGISTRY_MODULE: Final = "litellm.proxy._experimental.mcp_server.mcp_server_manager"


def _mcp_server_url(tool: object) -> str | None:
    match tool:
        case {"type": "mcp", "server_url": str() as server_url} if isinstance(tool, dict):
            return server_url
        case _:
            return None


def names_gateway_explicitly(tool: object) -> bool:
    return (_mcp_server_url(tool) or "").startswith(LITELLM_PROXY_MCP_SERVER_URL)


def proxy_path_mcp_name(tool: object) -> str | None:
    server_url: Final = _mcp_server_url(tool)
    match: Final = None if server_url is None else _PROXY_MCP_PATH_RE.match(server_url)
    return None if match is None else match.group(1)


def _gateway_registry_loads() -> bool:
    try:
        importlib.import_module(_GATEWAY_REGISTRY_MODULE)
    except ImportError:
        return False
    return True


def may_name_litellm_mcp_gateway(tools: Iterable[object] | None) -> bool:
    items: Final = tuple(tools or ())
    if any(map(names_gateway_explicitly, items)):
        return True
    return any(proxy_path_mcp_name(tool) is not None for tool in items) and _gateway_registry_loads()
