"""Management MCP e2e client: the official mcp SDK streamable-http transport
bound to the control-plane /litellm-management/mcp endpoint.

Raw httpx2 lives here alone (the SDK's streamable_http_client requires an
httpx2 AsyncClient object) and is allowlisted in
tests/code_coverage_tests/check_e2e_no_raw_requests.py; tests consume the typed
helpers below and never touch the HTTP layer.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Coroutine, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Final, TypeVar

import httpx2
import mcp.types as mcp_types
from e2e_config import CONTROL_PLANE_BASE_URL
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel, TypeAdapter

MANAGEMENT_MCP_PATH: Final = "/litellm-management/mcp"

_R = TypeVar("_R", bound=BaseModel)
_T = TypeVar("_T")

_STRUCTURED_ADAPTER: Final = TypeAdapter(dict[str, object])


@dataclass(frozen=True, slots=True)
class McpEndpoint:
    url: str
    key: str | None
    session_id: str | None


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    text: str
    structured: dict[str, object] | None
    is_error: bool


def management_mcp(
    path: str = MANAGEMENT_MCP_PATH, key: str | None = None, session_id: str | None = None
) -> McpEndpoint:
    return McpEndpoint(url=f"{CONTROL_PLANE_BASE_URL}{path}", key=key, session_id=session_id)


def run(coro: Coroutine[object, object, _T]) -> _T:
    return asyncio.run(coro)


def _headers(endpoint: McpEndpoint) -> dict[str, str]:
    headers: dict[str, str] = {}
    if endpoint.key is not None:
        headers["Authorization"] = f"Bearer {endpoint.key}"
    if endpoint.session_id is not None:
        headers["mcp-session-id"] = endpoint.session_id
    return headers


@asynccontextmanager
async def _client_session(endpoint: McpEndpoint) -> AsyncGenerator[ClientSession]:
    async with httpx2.AsyncClient(headers=_headers(endpoint), timeout=30.0) as http_client:
        async with streamable_http_client(endpoint.url, http_client=http_client) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


async def list_tool_names(endpoint: McpEndpoint) -> tuple[str, ...]:
    async with _client_session(endpoint) as session:
        listed: Final = await session.list_tools()
        return tuple(tool.name for tool in listed.tools)


async def call_tool(endpoint: McpEndpoint, name: str, arguments: Mapping[str, object]) -> ToolOutcome:
    async with _client_session(endpoint) as session:
        result: Final[mcp_types.CallToolResult] = await session.call_tool(name, dict(arguments))
        text: Final = "".join(content.text for content in result.content if isinstance(content, mcp_types.TextContent))
        raw_structured: Final[object] = result.structured_content  # pyright: ignore[reportAny]  # SDK declares structuredContent Any; validated into dict[str, object] right here
        structured: Final[dict[str, object] | None] = (
            _STRUCTURED_ADAPTER.validate_python(raw_structured) if raw_structured is not None else None
        )
        return ToolOutcome(text=text, structured=structured, is_error=bool(result.is_error))


async def call_tool_as(
    endpoint: McpEndpoint, name: str, arguments: Mapping[str, object], model: type[_R]
) -> tuple[ToolOutcome, _R | None]:
    outcome: Final = await call_tool(endpoint, name, arguments)
    parsed: Final[_R | None] = (
        TypeAdapter(model).validate_python(outcome.structured)
        if outcome.structured is not None and not outcome.is_error
        else None
    )
    return outcome, parsed


def initialize_status(endpoint: McpEndpoint) -> int:
    """HTTP status the endpoint answers a JSON-RPC initialize POST with."""

    async def _post() -> int:
        headers: Final = {"content-type": "application/json", "accept": "application/json, text/event-stream"}
        if endpoint.key is not None:
            headers["authorization"] = f"Bearer {endpoint.key}"
        if endpoint.session_id is not None:
            headers["mcp-session-id"] = endpoint.session_id
        async with httpx2.AsyncClient(timeout=30.0) as http:
            response: Final = await http.post(
                endpoint.url,
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "e2e", "version": "1"},
                    },
                },
            )
            return response.status_code

    return run(_post())
