"""
Rust sidecar transport for httpx.

Drop-in replacement for LiteLLMAiohttpTransport that forwards HTTP requests
through the Rust sidecar binary. All LiteLLM provider transformations,
logging, callbacks, and retry logic still run in Python — only the TCP
connection pooling and HTTP round-trip move to Rust.
"""

import typing
from typing import Optional

import aiohttp
import httpx


class SidecarResponseStream(httpx.AsyncByteStream):
    """Wraps an aiohttp response as an httpx async byte stream."""

    CHUNK_SIZE = 1024 * 16

    def __init__(self, response: aiohttp.ClientResponse) -> None:
        self._response = response

    async def __aiter__(self) -> typing.AsyncIterator[bytes]:
        async for chunk in self._response.content.iter_chunked(self.CHUNK_SIZE):
            yield chunk

    async def aclose(self) -> None:
        self._response.close()


class LiteLLMSidecarTransport(httpx.AsyncBaseTransport):
    """
    httpx transport that forwards requests through the Rust sidecar.

    The sidecar handles connection pooling, timeout enforcement, and
    metrics aggregation in Rust — eliminating GIL contention for the
    HTTP I/O path under high concurrency.
    """

    def __init__(self, sidecar_url: str = "http://127.0.0.1:8787"):
        self._sidecar_url = sidecar_url
        self._session: Optional[aiohttp.ClientSession] = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(
                    limit=0,
                    keepalive_timeout=90,
                    enable_cleanup_closed=True,
                ),
                timeout=aiohttp.ClientTimeout(total=None, connect=5),
            )
        return self._session

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # Extract the provider host from the full URL for the sidecar headers
        parsed = request.url
        provider_base = f"{parsed.scheme}://{parsed.host}"
        if parsed.port and parsed.port not in (80, 443):
            provider_base += f":{parsed.port}"
        path = parsed.raw_path.decode("ascii") if isinstance(parsed.raw_path, bytes) else str(parsed.raw_path)

        # Extract auth header if present
        api_key = ""
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            api_key = auth[7:]

        # Determine if streaming from content-type or accept headers
        is_stream = "text/event-stream" in request.headers.get("accept", "")

        # Get timeout from request extensions
        timeout_config = request.extensions.get("timeout", {})
        timeout_secs = timeout_config.get("read", 300) or 300
        if isinstance(timeout_secs, (int, float)):
            timeout_secs = int(timeout_secs)
        else:
            timeout_secs = 300

        try:
            body = request.content
        except httpx.RequestNotRead:
            body = b""

        headers = {
            "X-LiteLLM-Provider-URL": provider_base,
            "X-LiteLLM-API-Key": api_key,
            "X-LiteLLM-Timeout": str(timeout_secs),
            "X-LiteLLM-Stream": "true" if is_stream else "false",
            "X-LiteLLM-Path": path,
            "Content-Type": request.headers.get("content-type", "application/json"),
        }

        # Copy provider-specific headers the sidecar should forward
        for key in ("x-api-key", "anthropic-version", "x-goog-api-key"):
            val = request.headers.get(key)
            if val:
                headers[f"X-LiteLLM-Fwd-{key}"] = val

        session = self._get_session()
        resp = await session.post(
            f"{self._sidecar_url}/forward",
            data=body,
            headers=headers,
        )

        return httpx.Response(
            status_code=resp.status,
            headers=dict(resp.headers),
            stream=SidecarResponseStream(resp),
            request=request,
        )

    async def aclose(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
