"""naming — pure SEP-986 tool-name helpers.

Namespaces upstream tools as ``{server_alias}__{tool}`` and splits them back.
Pure leaf: no I/O, no dependencies beyond the gateway result vocabulary.
"""

from __future__ import annotations

import re

from litellm.proxy.gateway.mcp.foundation.errors import GatewayError
from litellm.proxy.gateway.mcp.foundation.result import Error, GatewayResult, Ok

_SEP = "__"
_SEP986 = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def is_valid_name(name: str) -> bool:
    return _SEP986.match(name) is not None


def namespace_tool(server_alias: str, tool: str) -> str:
    return f"{server_alias}{_SEP}{tool}"


def split_namespaced(name: str) -> GatewayResult[tuple[str, str]]:
    if not is_valid_name(name):
        return Error(GatewayError(invalid_input=f"invalid tool name: {name!r}"))
    parts = name.split(_SEP, 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return Error(GatewayError(invalid_input=f"not namespaced: {name!r}"))
    return Ok((parts[0], parts[1]))
