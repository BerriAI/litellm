"""Client for the mcp e2e suite.

Management routes (/v1/mcp/server CRUD) go through the shared Gateway
transport. The MCP protocol itself (initialize, tools/list, tools/call over
streamable HTTP) goes through the official mcp SDK, the same client library
production MCP hosts use, aimed at the gateway's per-server URL namespace
{PROXY}/{alias}/mcp.

Every protocol method takes the request headers as a plain dict, built inside
the test body, so the exact wire format is visible where it is asserted. The gateway accepts the LiteLLM
virtual key as either `x-litellm-api-key: Bearer sk-...` or
`Authorization: Bearer sk-...` (both Bearer-prefixed on the MCP routes,
matching the docs).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Mapping, cast
from urllib.parse import parse_qsl, urljoin

import httpx
import pytest
from mcp import ClientSession
from mcp.client.auth import OAuthClientProvider
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel, RootModel

from e2e_config import PROXY_BASE_URL, REQUEST_TIMEOUT
from e2e_gateway import Gateway, build_gateway
from e2e_http import NoBody, unwrap
from models import McpServerCreateBody, McpServerInfo

ToolArguments = Mapping[str, str | float]


@dataclass(frozen=True, slots=True)
class McpToolNames:
    names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class McpDenied:
    """A tools/list attempt the gateway rejected before serving the session."""

    status_code: int | None
    message: str


ListToolsOutcome = McpToolNames | McpDenied


@dataclass(frozen=True, slots=True)
class McpToolText:
    """The text content of one tools/call result."""

    text: str
    is_error: bool


class StubToolStats(BaseModel):
    """JSON the stub's `stats` tool returns for one marker (see stub/stub_server.py)."""

    marker: str
    max_in_flight: int
    completed: int


class StubRecordedHeaders(RootModel[dict[str, str]]):
    """JSON the stub's `recorded_headers` tool returns: the (lowercased) header
    map of the most recent request its auth guard let through."""


def _mcp_url(alias: str) -> str:
    return f"{PROXY_BASE_URL}/{alias}/mcp"


def _first_text(result: CallToolResult) -> str:
    first = result.content[0] if result.content else None
    if isinstance(first, TextContent):
        return first.text
    return f"<non-text content: {type(first).__name__}>"


def _http_status(root: BaseException) -> int | None:
    """The HTTP status behind an SDK failure; the SDK wraps transport errors in
    (possibly nested) ExceptionGroups, so walk them without recursing."""
    pending: list[BaseException] = [root]  # mutable-ok: bounded worklist over an exception tree
    while pending:
        exc = pending.pop()
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code
        if isinstance(exc, BaseExceptionGroup):
            pending.extend(cast("tuple[BaseException, ...]", exc.exceptions))
    return None


def _http_client(headers: dict[str, str]) -> httpx.AsyncClient:
    return httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(REQUEST_TIMEOUT), follow_redirects=True)


async def _list_tool_names(url: str, headers: dict[str, str]) -> tuple[str, ...]:
    async with _http_client(headers) as http_client:
        async with streamable_http_client(url, http_client=http_client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                return tuple(sorted(tool.name for tool in listed.tools))


async def _call_tool(url: str, headers: dict[str, str], tool: str, arguments: ToolArguments) -> McpToolText:
    async with _http_client(headers) as http_client:
        async with streamable_http_client(url, http_client=http_client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool, dict(arguments))
                return McpToolText(text=_first_text(result), is_error=bool(result.isError))


async def _list_prompts(url: str, headers: dict[str, str]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """(name, argument names) per prompt, sorted by name."""
    async with _http_client(headers) as http_client:
        async with streamable_http_client(url, http_client=http_client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_prompts()
                return tuple(
                    sorted(
                        (prompt.name, tuple(arg.name for arg in (prompt.arguments or [])))
                        for prompt in listed.prompts
                    )
                )


async def _get_prompt(url: str, headers: dict[str, str], name: str, arguments: dict[str, str]) -> str:
    """The text of the rendered prompt's first message."""
    async with _http_client(headers) as http_client:
        async with streamable_http_client(url, http_client=http_client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.get_prompt(name, arguments)
                first = result.messages[0].content if result.messages else None
                if isinstance(first, TextContent):
                    return first.text
                return f"<non-text prompt content: {type(first).__name__}>"


async def _list_resources(url: str, headers: dict[str, str]) -> tuple[tuple[str, str], ...]:
    """(uri, name) per resource, sorted by uri."""
    async with _http_client(headers) as http_client:
        async with streamable_http_client(url, http_client=http_client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_resources()
                return tuple(sorted((str(resource.uri), resource.name) for resource in listed.resources))


async def _read_resource(url: str, headers: dict[str, str], uri: str) -> str:
    """The text of the resource's first content block."""
    from pydantic import AnyUrl

    async with _http_client(headers) as http_client:
        async with streamable_http_client(url, http_client=http_client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.read_resource(AnyUrl(uri))
                first = result.contents[0] if result.contents else None
                text = getattr(first, "text", None)
                if isinstance(text, str):
                    return text
                return f"<non-text resource content: {type(first).__name__}>"


# ---------- interactive (authorization_code) OAuth: the MCP-host side ----------

# Where the "browser" lands at the end of the authorize dance. Nothing listens
# here on purpose: the redirect chaser stops as soon as the chain points at this
# URL and reads the code/state off the query string, exactly like a desktop MCP
# host that intercepts its loopback redirect.
OAUTH_CLIENT_REDIRECT_URI = "http://127.0.0.1:53682/e2e/callback"


class InMemoryTokenStorage:
    """The mcp SDK's TokenStorage protocol, held in memory for one test: the
    DCR-registered client and the tokens the gateway minted for it."""

    def __init__(self) -> None:
        self._tokens: OAuthToken | None = None
        self._client_info: OAuthClientInformationFull | None = None

    async def get_tokens(self) -> OAuthToken | None:
        return self._tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self._tokens = tokens

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        return self._client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self._client_info = client_info


async def _follow_authorize_redirects(start_url: str) -> tuple[str, str | None]:
    """Play the browser's role in the authorize dance: chase the redirect chain
    (gateway authorize -> upstream IdP authorize -> gateway callback -> MCP
    host redirect_uri) with plain GETs, and return the code/state from the
    final hop's query string without dereferencing it."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT), follow_redirects=False) as browser:
        url = start_url
        for _ in range(10):
            if url.startswith(OAUTH_CLIENT_REDIRECT_URI):
                params = dict(parse_qsl(httpx.URL(url).query.decode()))
                assert "code" in params, f"authorize chain reached the client redirect_uri without a code: {url}"
                return params["code"], params.get("state")
            response = await browser.get(url)
            assert response.status_code in (302, 303, 307), (
                f"authorize chain broke at {url}: {response.status_code} {response.text[:300]}"
            )
            url = urljoin(url, response.headers["location"])
        raise AssertionError(f"authorize chain never reached {OAUTH_CLIENT_REDIRECT_URI}; last url: {url}")


def _oauth_provider(url: str, storage: InMemoryTokenStorage) -> OAuthClientProvider:
    """The SDK's real OAuth machinery (RFC 9728/8414 discovery, RFC 7591 DCR,
    PKCE, token exchange) with the browser leg replaced by the redirect chaser."""
    code_holder: dict[str, str | None] = {}  # mutable-ok: hand-off between the two SDK callbacks

    async def redirect_handler(authorize_url: str) -> None:
        code, state = await _follow_authorize_redirects(authorize_url)
        code_holder["code"] = code
        code_holder["state"] = state

    async def callback_handler() -> tuple[str, str | None]:
        code = code_holder.get("code")
        assert code is not None, "callback_handler ran before the authorize redirect completed"
        return code, code_holder.get("state")

    return OAuthClientProvider(
        server_url=url,
        client_metadata=OAuthClientMetadata.model_validate(
            {
                "redirect_uris": [OAUTH_CLIENT_REDIRECT_URI],
                "token_endpoint_auth_method": "none",
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "client_name": "e2e-mcp-host",
            }
        ),
        storage=storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )


class _HeaderInjectingTransport(httpx.AsyncBaseTransport):
    """Adds the caller's LiteLLM key header to every outgoing request,
    including the ones the SDK's OAuth machinery builds internally (discovery,
    DCR, token exchange). Those bypass httpx client-default headers, but the
    gateway resolves which user to store the upstream token for from the key
    on the token exchange, exactly like a production MCP host configured with
    a LiteLLM key header on the server URL."""

    def __init__(self, inner: httpx.AsyncBaseTransport, headers: dict[str, str]) -> None:
        self._inner = inner
        self._headers = headers

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        for name, value in self._headers.items():
            if name not in request.headers:
                request.headers[name] = value
        return await self._inner.handle_async_request(request)


def _oauth_http_client(headers: dict[str, str], auth: OAuthClientProvider) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers=headers,
        auth=auth,
        timeout=httpx.Timeout(REQUEST_TIMEOUT),
        follow_redirects=True,
        transport=_HeaderInjectingTransport(httpx.AsyncHTTPTransport(), headers),
    )


async def _oauth_list_tool_names(url: str, headers: dict[str, str], storage: InMemoryTokenStorage) -> tuple[str, ...]:
    async with _oauth_http_client(headers, _oauth_provider(url, storage)) as http_client:
        async with streamable_http_client(url, http_client=http_client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                return tuple(sorted(tool.name for tool in listed.tools))


async def _oauth_call_tool(
    url: str, headers: dict[str, str], storage: InMemoryTokenStorage, tool: str, arguments: ToolArguments
) -> McpToolText:
    async with _oauth_http_client(headers, _oauth_provider(url, storage)) as http_client:
        async with streamable_http_client(url, http_client=http_client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool, dict(arguments))
                return McpToolText(text=_first_text(result), is_error=bool(result.isError))


@dataclass(frozen=True, slots=True)
class McpClient:
    gateway: Gateway

    def create_server(self, body: McpServerCreateBody) -> McpServerInfo:
        return unwrap(
            self.gateway.transport.post(
                "/v1/mcp/server",
                headers=self.gateway.transport.master,
                json=body,
                response_type=McpServerInfo,
            )
        )

    def server_info(self, server_id: str) -> McpServerInfo:
        return unwrap(
            self.gateway.transport.get(
                f"/v1/mcp/server/{server_id}",
                headers=self.gateway.transport.master,
                params=NoBody(),
                response_type=McpServerInfo,
            )
        )

    def delete_server(self, server_id: str) -> None:
        _ = self.gateway.transport.delete(
            f"/v1/mcp/server/{server_id}",
            headers=self.gateway.transport.master,
            json=NoBody(),
            response_type=NoBody,
        )

    def list_tools_once(self, alias: str, headers: dict[str, str]) -> ListToolsOutcome:
        try:
            return McpToolNames(names=asyncio.run(_list_tool_names(_mcp_url(alias), headers)))
        except Exception as exc:  # noqa: BLE001 - the SDK raises ExceptionGroup-wrapped transport errors; modelled as a value
            return McpDenied(status_code=_http_status(exc), message=str(exc))

    def poll_tool_names(self, alias: str, headers: dict[str, str]) -> tuple[str, ...]:
        """tools/list to the shared deadline: a just-created server record
        propagates to the gateway asynchronously and a just-created key can lag
        the auth cache, so the first attempts may 401 or list nothing."""
        deadline = time.monotonic() + self.gateway.poll_timeout
        outcome: ListToolsOutcome = McpDenied(status_code=None, message="never attempted")
        while time.monotonic() < deadline:
            outcome = self.list_tools_once(alias, headers)
            match outcome:
                case McpToolNames(names=names) if names:
                    return names
                case _:
                    time.sleep(self.gateway.poll_interval)
        pytest.fail(
            f"MCP tools for {alias!r} never listed within {self.gateway.poll_timeout}s; last outcome: {outcome}"
        )

    def call_tool(self, alias: str, headers: dict[str, str], tool: str, arguments: ToolArguments) -> McpToolText:
        """One tools/call over its own fresh MCP session, so concurrent callers
        behave like independent clients."""
        return asyncio.run(_call_tool(_mcp_url(alias), headers, tool, arguments))

    def poll_oauth_tool_names(
        self, alias: str, headers: dict[str, str], storage: InMemoryTokenStorage
    ) -> tuple[str, ...]:
        """The full interactive flow (discovery, DCR, authorize dance, token
        exchange, then tools/list) retried to the shared deadline, since the
        just-created server record and key propagate asynchronously. Tokens
        land in `storage`, so later calls skip the dance like a real host."""
        deadline = time.monotonic() + self.gateway.poll_timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                return asyncio.run(_oauth_list_tool_names(_mcp_url(alias), headers, storage))
            except Exception as exc:  # noqa: BLE001 - retried to the deadline; the last error is surfaced below
                last_error = exc
                time.sleep(self.gateway.poll_interval)
        pytest.fail(
            f"interactive OAuth flow for {alias!r} never completed within "
            f"{self.gateway.poll_timeout}s; last error: {last_error!r}"
        )

    def oauth_call_tool(
        self, alias: str, headers: dict[str, str], storage: InMemoryTokenStorage, tool: str, arguments: ToolArguments
    ) -> McpToolText:
        """One tools/call authenticated by the tokens in `storage` (minted by a
        prior poll_oauth_tool_names dance), over its own fresh MCP session."""
        return asyncio.run(_oauth_call_tool(_mcp_url(alias), headers, storage, tool, arguments))

    def list_prompts(self, alias: str, headers: dict[str, str]) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """prompts/list over its own fresh MCP session: (name, argument names) sorted by name."""
        return asyncio.run(_list_prompts(_mcp_url(alias), headers))

    def get_prompt(self, alias: str, headers: dict[str, str], name: str, arguments: dict[str, str]) -> str:
        """prompts/get over its own fresh MCP session: the rendered first message's text."""
        return asyncio.run(_get_prompt(_mcp_url(alias), headers, name, arguments))

    def list_resources(self, alias: str, headers: dict[str, str]) -> tuple[tuple[str, str], ...]:
        """resources/list over its own fresh MCP session: (uri, name) sorted by uri."""
        return asyncio.run(_list_resources(_mcp_url(alias), headers))

    def read_resource(self, alias: str, headers: dict[str, str], uri: str) -> str:
        """resources/read over its own fresh MCP session: the first content block's text."""
        return asyncio.run(_read_resource(_mcp_url(alias), headers, uri))

    def stub_stats(self, alias: str, headers: dict[str, str], stats_tool: str, marker: str) -> StubToolStats:
        outcome = self.call_tool(alias, headers, stats_tool, {"marker": marker})
        return StubToolStats.model_validate_json(outcome.text)

    def stub_recorded_headers(self, alias: str, headers: dict[str, str], headers_tool: str) -> dict[str, str]:
        """The headers the stub's auth guard recorded for its most recent
        authorized request: by construction, the tools/call this method just
        made, i.e. exactly what the gateway sent upstream on the caller's behalf."""
        outcome = self.call_tool(alias, headers, headers_tool, {})
        assert outcome.is_error is False, f"recorded_headers call errored: {outcome.text[:300]}"
        return StubRecordedHeaders.model_validate_json(outcome.text).root


def build_client() -> McpClient:
    return McpClient(gateway=build_gateway())
