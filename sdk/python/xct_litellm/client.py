"""XctClient — thin wrapper around the OpenAI-compatible proxy + XCT extras.

Design choices:

- Composition over inheritance: we expose `xct.chat`, `xct.agents`, `xct.mcp`,
  `xct.skills`, `xct.capabilities` as nested objects. Each is a tiny class
  holding a reference back to the parent client (sharing the http session).
- Plain `httpx` — no SDK-of-SDK indirection. Async path uses `httpx.AsyncClient`,
  sync path uses `httpx.Client`. Both share the same wire-level code path via
  `_request`.
- `chat.completions.create()` is the most-used method; mirror the OpenAI shape
  ({model, messages, stream, ...}) so anyone familiar with the OpenAI SDK can
  use this without re-learning.
- App-tenancy: `app_id` is sent as the `x-xct-app-id` header when constructed
  with one. The proxy's S4-04 auth uses it only as a fallback (token-baked
  app_id wins).
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import httpx

from .errors import from_response


class _Resource:
    def __init__(self, client: "XctClient"):
        self._client = client


class CapabilitiesResource(_Resource):
    async def alist(self) -> Dict[str, Any]:
        return await self._client._arequest("GET", "/v1/capabilities")

    def list(self) -> Dict[str, Any]:
        return self._client._request("GET", "/v1/capabilities")


class AgentsResource(_Resource):
    async def alist(
        self,
        *,
        q: Optional[str] = None,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        params = _prune(
            {"q": q, "category": category, "tag": tag, "cursor": cursor, "limit": limit}
        )
        return await self._client._arequest("GET", "/v1/agents", params=params)

    def list(self, **kwargs) -> List[Dict[str, Any]]:
        params = _prune(kwargs)
        return self._client._request("GET", "/v1/agents", params=params)

    async def aget(self, agent_id: str) -> Dict[str, Any]:
        return await self._client._arequest("GET", f"/v1/agents/{agent_id}")

    def get(self, agent_id: str) -> Dict[str, Any]:
        return self._client._request("GET", f"/v1/agents/{agent_id}")

    async def ainvoke(
        self,
        agent_id: str,
        *,
        message: Dict[str, Any],
        request_id: str = "1",
        stream: bool = False,
    ) -> Any:
        """Invoke an A2A agent via JSON-RPC 2.0.

        When ``stream=True`` the return value is an async iterator yielding the
        parsed SSE / NDJSON events. Falls back to NDJSON if the server doesn't
        honor SSE.
        """
        body = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "message/stream" if stream else "message/send",
            "params": {"message": message},
        }
        if stream:
            return self._client._astream(
                "POST",
                f"/v1/a2a/{agent_id}/message/send",
                json=body,
                accept_sse=True,
            )
        return await self._client._arequest(
            "POST", f"/v1/a2a/{agent_id}/message/send", json=body
        )


class McpResource(_Resource):
    async def alist_tools(self) -> List[Dict[str, Any]]:
        return await self._client._arequest("GET", "/v1/mcp/tools")

    def list_tools(self) -> List[Dict[str, Any]]:
        return self._client._request("GET", "/v1/mcp/tools")


class SkillsResource(_Resource):
    async def alist(self, **kwargs) -> Dict[str, Any]:
        return await self._client._arequest(
            "GET", "/v1/xct-skills", params=_prune(kwargs)
        )

    def list(self, **kwargs) -> Dict[str, Any]:
        return self._client._request("GET", "/v1/xct-skills", params=_prune(kwargs))


class _ChatCompletionsResource(_Resource):
    async def acreate(
        self, **payload
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        if payload.get("stream"):
            return self._client._astream(
                "POST", "/v1/chat/completions", json=payload, accept_sse=True
            )
        return await self._client._arequest(
            "POST", "/v1/chat/completions", json=payload
        )

    def create(self, **payload) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        if payload.get("stream"):
            return self._client._stream(
                "POST", "/v1/chat/completions", json=payload, accept_sse=True
            )
        return self._client._request("POST", "/v1/chat/completions", json=payload)


class ChatResource(_Resource):
    def __init__(self, client: "XctClient"):
        super().__init__(client)
        self.completions = _ChatCompletionsResource(client)


class XctClient:
    """Entry point.

    Args:
        base_url: proxy origin (e.g. "https://api.xct.test").
        access_token: Bearer token (OAuth-issued or admin virtual key).
        app_id: optional; sent as ``x-xct-app-id``. Used by S4-04 as a
                fallback when the access_token doesn't already carry an
                app_id (e.g. service-account keys).
        timeout: per-request timeout in seconds (default 60).
    """

    def __init__(
        self,
        base_url: str,
        access_token: Optional[str] = None,
        *,
        app_id: Optional[str] = None,
        timeout: float = 60.0,
        http_client: Optional[httpx.Client] = None,
        async_http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.access_token = access_token
        self.app_id = app_id
        self.timeout = timeout
        self._http = http_client
        self._ahttp = async_http_client

        self.capabilities = CapabilitiesResource(self)
        self.agents = AgentsResource(self)
        self.mcp = McpResource(self)
        self.skills = SkillsResource(self)
        self.chat = ChatResource(self)

    # ---- header / url helpers --------------------------------------------

    def _headers(self) -> Dict[str, str]:
        h = {"Accept": "application/json"}
        if self.access_token:
            h["Authorization"] = f"Bearer {self.access_token}"
        if self.app_id:
            h["x-xct-app-id"] = self.app_id
        return h

    def _full_url(self, path: str) -> str:
        return (
            f"{self.base_url}{path}"
            if path.startswith("/")
            else f"{self.base_url}/{path}"
        )

    # ---- sync ------------------------------------------------------------

    def _client_sync(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(timeout=self.timeout)
        return self._http

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> Any:
        client = self._client_sync()
        resp = client.request(
            method,
            self._full_url(path),
            params=params,
            json=json,
            headers=self._headers(),
        )
        return _handle(resp)

    def _stream(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        accept_sse: bool = False,
    ) -> Iterator[Dict[str, Any]]:
        client = self._client_sync()
        headers = self._headers()
        if accept_sse:
            headers["Accept"] = "text/event-stream"
        with client.stream(
            method, self._full_url(path), json=json, headers=headers
        ) as resp:
            if resp.status_code >= 400:
                resp.read()
                _handle(resp)
            yield from _iter_events(resp)

    # ---- async -----------------------------------------------------------

    def _aclient(self) -> httpx.AsyncClient:
        if self._ahttp is None:
            self._ahttp = httpx.AsyncClient(timeout=self.timeout)
        return self._ahttp

    async def _arequest(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> Any:
        client = self._aclient()
        resp = await client.request(
            method,
            self._full_url(path),
            params=params,
            json=json,
            headers=self._headers(),
        )
        return _handle(resp)

    async def _astream(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        accept_sse: bool = False,
    ) -> AsyncIterator[Dict[str, Any]]:
        client = self._aclient()
        headers = self._headers()
        if accept_sse:
            headers["Accept"] = "text/event-stream"
        async with client.stream(
            method, self._full_url(path), json=json, headers=headers
        ) as resp:
            if resp.status_code >= 400:
                await resp.aread()
                _handle(resp)
            async for event in _aiter_events(resp):
                yield event

    # ---- shutdown --------------------------------------------------------

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None

    async def aclose(self) -> None:
        if self._ahttp is not None:
            await self._ahttp.aclose()
            self._ahttp = None


# ============================================================================
# helpers
# ============================================================================


def _prune(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def _handle(resp: httpx.Response) -> Any:
    if 200 <= resp.status_code < 300:
        if not resp.content:
            return None
        ct = resp.headers.get("content-type", "")
        if "application/json" in ct:
            return resp.json()
        return resp.text
    try:
        body = resp.json()
    except Exception:
        body = resp.text
    raise from_response(resp.status_code, body)


def _iter_events(resp: httpx.Response) -> Iterator[Dict[str, Any]]:
    """Yield events from either SSE (text/event-stream) or NDJSON."""
    import json

    ct = resp.headers.get("content-type", "")
    if "text/event-stream" in ct:
        for event in _sse_event_iter_sync(resp):
            if event is not None:
                yield event
    else:
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


async def _aiter_events(resp: httpx.Response) -> AsyncIterator[Dict[str, Any]]:
    import json

    ct = resp.headers.get("content-type", "")
    if "text/event-stream" in ct:
        async for event in _sse_event_iter_async(resp):
            if event is not None:
                yield event
    else:
        async for line in resp.aiter_lines():
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _sse_event_iter_sync(resp: httpx.Response):
    """Minimal SSE parser: emits the parsed `data:` JSON per event."""
    import json

    buffer: list[str] = []
    for line in resp.iter_lines():
        if line == "":
            if not buffer:
                continue
            yield _coalesce_sse_event(buffer)
            buffer = []
            continue
        buffer.append(line)
    if buffer:
        yield _coalesce_sse_event(buffer)


async def _sse_event_iter_async(resp: httpx.Response):
    import json

    buffer: list[str] = []
    async for line in resp.aiter_lines():
        if line == "":
            if not buffer:
                continue
            yield _coalesce_sse_event(buffer)
            buffer = []
            continue
        buffer.append(line)
    if buffer:
        yield _coalesce_sse_event(buffer)


def _coalesce_sse_event(lines: list[str]):
    """Take raw SSE event lines, return the parsed `data` payload (or None)."""
    import json

    data_parts = [ln[len("data:") :].lstrip() for ln in lines if ln.startswith("data:")]
    if not data_parts:
        return None
    raw = "\n".join(data_parts).strip()
    if raw == "" or raw == "[DONE]":
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}
