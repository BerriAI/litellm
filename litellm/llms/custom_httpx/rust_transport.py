"""
PyO3 Rust-native transport for httpx.

Replaces the HTTP round-trip with an in-process Rust call via PyO3.
Unlike the sidecar approach, this eliminates the localhost network hop —
Python calls directly into Rust, which releases the GIL and performs the
HTTP request using reqwest/tokio. Multiple requests run truly in parallel.
"""

import asyncio
import typing
import httpx


try:
    import litellm_pyext

    _has_rust_ext = True
except ImportError:
    _has_rust_ext = False


def is_available() -> bool:
    return _has_rust_ext


class RustResponseStream(httpx.AsyncByteStream):
    """Wraps pre-fetched bytes as a single-chunk async stream."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    async def __aiter__(self) -> typing.AsyncIterator[bytes]:
        yield self._data

    async def aclose(self) -> None:
        pass


class LiteLLMRustTransport(httpx.AsyncBaseTransport):
    """
    httpx async transport that forwards requests through the in-process
    Rust extension. The GIL is released during the HTTP round-trip, so
    the Python event loop can handle other coroutines concurrently.
    """

    def __init__(self, max_workers: int = 8):
        from concurrent.futures import ThreadPoolExecutor

        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="rust-http"
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        method = request.method

        headers: dict[str, str] = {}
        for key, val in request.headers.items():
            lower = key.lower()
            if lower not in ("host", "content-length", "transfer-encoding"):
                headers[key] = val

        try:
            body = request.content
        except httpx.RequestNotRead:
            body = b""

        timeout_config = request.extensions.get("timeout", {})
        timeout_secs = timeout_config.get("read", 300) or 300
        if isinstance(timeout_secs, (int, float)):
            timeout_secs = int(timeout_secs)
        else:
            timeout_secs = 300

        loop = asyncio.get_event_loop()
        status, resp_headers, resp_body = await loop.run_in_executor(
            self._executor,
            litellm_pyext.forward_request,
            url,
            method,
            headers,
            bytes(body),
            timeout_secs,
        )

        return httpx.Response(
            status_code=status,
            headers=resp_headers,
            stream=RustResponseStream(bytes(resp_body)),
            request=request,
        )

    async def aclose(self) -> None:
        self._executor.shutdown(wait=False)
