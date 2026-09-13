"""Client for the mcp chat-completion OAuth e2e suite.

Registers a gateway-managed OAuth (authorization_code) MCP server, seeds the
per-user upstream token by driving the interactive authorize dance with the
official mcp SDK's OAuthClientProvider (the browser leg is a headless Chromium
primed with a human's saved Linear session), then exercises the server through
/chat/completions, where the gateway lists and executes its tools with the
stored per-user token.

Management routes (/v1/mcp/server CRUD, /chat/completions) go through the
shared ProxyClient transport. The MCP protocol used to seed the token goes through
the mcp SDK, the same library production MCP hosts run.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import parse_qsl

import httpx
import pytest
from mcp import ClientSession
from mcp.client.auth import OAuthClientProvider
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

from e2e_config import PROXY_BASE_URL, REQUEST_TIMEOUT
from proxy_client import ProxyClient
from e2e_http import AuthHeaders, NoBody, unwrap
from models import ChatBody, ChatResponse, McpServerCreateBody, McpServerInfo

if TYPE_CHECKING:
    from playwright.async_api import Route

# Where the "browser" lands at the end of the authorize dance. Nothing listens
# here: the route interceptor short-circuits the final redirect and reads the
# code/state off its query string, exactly like a desktop MCP host intercepting
# its loopback redirect.
OAUTH_CLIENT_REDIRECT_URI = "http://127.0.0.1:53682/e2e/callback"
BROWSER_CONSENT_TIMEOUT = 60.0


def _mcp_url(alias: str) -> str:
    return f"{PROXY_BASE_URL}/{alias}/mcp"


class InMemoryTokenStorage:
    """The mcp SDK's TokenStorage protocol, in memory for one dance: the
    DCR-registered client and the gateway tokens minted for it."""

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


async def _browser_follow_authorize(start_url: str, storage_state_path: str) -> tuple[str, str | None]:
    """Play the browser's role for a real upstream whose authorize endpoint
    serves an interactive consent page (Linear). A headless Chromium primed
    with a human's saved Linear session opens the gateway authorize URL and
    clicks through Linear's consent screens (the mcp.linear.app Approve form,
    then the linear.app workspace-selection page), riding the rest of the chain
    (Linear -> gateway callback -> host redirect_uri). The final hop is
    intercepted and short-circuited, since nothing listens there, and its
    code/state are read off the query string."""
    from playwright.async_api import async_playwright

    captured: dict[str, str] = {}  # mutable-ok: hand-off from the request listener
    trail: list[str] = []  # mutable-ok: navigation diagnostics for a failed dance

    def _note_request(request: object) -> None:
        url = getattr(request, "url", "")
        if url.startswith(OAUTH_CLIENT_REDIRECT_URI) and "url" not in captured:
            captured["url"] = url

    async def _swallow_redirect(route: "Route") -> None:
        await route.fulfill(status=200, content_type="text/plain", body="ok")

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(storage_state=storage_state_path)
        await context.route(re.compile(re.escape(OAUTH_CLIENT_REDIRECT_URI) + r".*"), _swallow_redirect)
        page = await context.new_page()
        page.on("request", _note_request)
        page.on("framenavigated", lambda frame: trail.append(frame.url.split("?", 1)[0]))
        await page.goto(start_url, wait_until="domcontentloaded")
        deadline = time.monotonic() + BROWSER_CONSENT_TIMEOUT
        while "url" not in captured and time.monotonic() < deadline:
            try:
                await page.wait_for_load_state("networkidle", timeout=8000)
            except Exception:  # noqa: BLE001 - a busy consent page never idles; fall through and try to advance it
                pass
            if "url" in captured:
                break
            control = page.locator(
                'button[name="action"][value="approve"], button:has-text("Authorize"), '
                'button:has-text("Allow"), button:has-text("@"), a:has-text("@")'
            ).first
            try:
                await control.click(timeout=5000)
            except Exception:  # noqa: BLE001 - nothing to advance yet; loop and re-check
                await asyncio.sleep(0.5)
        final_url = page.url
        await browser.close()

    landing = captured.get("url")
    assert landing is not None, (
        f"consent flow never reached {OAUTH_CLIENT_REDIRECT_URI}; "
        f"final={final_url.split('?', 1)[0]!r}; trail={trail[-6:]}"
    )
    params = dict(parse_qsl(httpx.URL(landing).query.decode()))
    assert "code" in params, f"client redirect_uri carried no code: {landing}"
    return params["code"], params.get("state")


def _oauth_provider(url: str, storage: InMemoryTokenStorage, storage_state_path: str) -> OAuthClientProvider:
    """The SDK's real OAuth machinery (RFC 9728/8414 discovery, RFC 7591 DCR,
    PKCE, token exchange) with the browser leg driven by Playwright against the
    upstream's consent screen."""
    code_holder: dict[str, str | None] = {}  # mutable-ok: hand-off between the two SDK callbacks

    async def redirect_handler(authorize_url: str) -> None:
        code, state = await _browser_follow_authorize(authorize_url, storage_state_path)
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
    """Adds the caller's LiteLLM key header to every outgoing SDK request
    (discovery, DCR, token exchange), so the gateway resolves which user to
    store the upstream token for from the key on the token exchange, exactly
    like a production MCP host configured with a LiteLLM key header."""

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


async def _seed_via_dance(
    url: str, headers: dict[str, str], storage: InMemoryTokenStorage, storage_state_path: str
) -> tuple[str, ...]:
    async with _oauth_http_client(headers, _oauth_provider(url, storage, storage_state_path)) as http_client:
        async with streamable_http_client(url, http_client=http_client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                return tuple(sorted(tool.name for tool in listed.tools))


@dataclass(frozen=True, slots=True)
class ChatMcpClient:
    proxy: ProxyClient

    def create_server(self, body: McpServerCreateBody) -> McpServerInfo:
        return unwrap(
            self.proxy.transport.post(
                "/v1/mcp/server",
                headers=self.proxy.transport.master,
                json=body,
                response_type=McpServerInfo,
            )
        )

    def server_info(self, server_id: str) -> McpServerInfo:
        return unwrap(
            self.proxy.transport.get(
                f"/v1/mcp/server/{server_id}",
                headers=self.proxy.transport.master,
                params=NoBody(),
                response_type=McpServerInfo,
            )
        )

    def delete_server(self, server_id: str) -> None:
        _ = self.proxy.transport.delete(
            f"/v1/mcp/server/{server_id}",
            headers=self.proxy.transport.master,
            json=NoBody(),
            response_type=NoBody,
        )

    def seed_user_token(self, alias: str, key: str, storage_state_path: str) -> tuple[str, ...]:
        """Drive the interactive authorize dance for `key`'s user so the gateway
        stores their upstream token, retried to the shared deadline since the
        just-created server and key propagate asynchronously. The LiteLLM key
        rides x-litellm-api-key so the gateway binds the token to that user.
        Returns the upstream tool names the dance listed, proof the token works."""
        headers = {"x-litellm-api-key": f"Bearer {key}"}
        storage = InMemoryTokenStorage()
        deadline = time.monotonic() + self.proxy.poll_timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                return asyncio.run(_seed_via_dance(_mcp_url(alias), headers, storage, storage_state_path))
            except Exception as exc:  # noqa: BLE001 - retried to the deadline; the last error surfaces below
                last_error = exc
                time.sleep(self.proxy.poll_interval)
        pytest.fail(
            f"authorize dance for {alias!r} never completed within {self.proxy.poll_timeout}s; "
            f"last error: {last_error!r}"
        )

    def chat_with_mcp(self, headers: AuthHeaders, body: ChatBody) -> ChatResponse:
        """POST /chat/completions carrying the LiteLLM key in `headers` (either
        ingress form) with an MCP server attached in `body.tools`. The gateway
        resolves the user from the key and lists/executes the server's tools
        with that user's stored upstream token."""
        return unwrap(
            self.proxy.transport.post(
                "/chat/completions",
                headers=headers,
                json=body,
                response_type=ChatResponse,
            )
        )


def build_chat_client(proxy: ProxyClient) -> ChatMcpClient:
    return ChatMcpClient(proxy=proxy)
