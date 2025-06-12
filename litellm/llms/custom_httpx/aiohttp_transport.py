import asyncio
import contextlib
import typing
import weakref
from typing import Callable, Dict, Union

import aiohttp
import aiohttp.client_exceptions
import aiohttp.http_exceptions
import httpx
from aiohttp.client import ClientResponse, ClientSession

from litellm._logging import verbose_logger

AIOHTTP_EXC_MAP: Dict = {
    # Order matters here, most specific exception first
    # Timeout related exceptions
    aiohttp.ServerTimeoutError: httpx.TimeoutException,
    aiohttp.ConnectionTimeoutError: httpx.ConnectTimeout,
    aiohttp.SocketTimeoutError: httpx.ReadTimeout,
    # Proxy related exceptions
    aiohttp.ClientProxyConnectionError: httpx.ProxyError,
    # SSL related exceptions
    aiohttp.ClientConnectorCertificateError: httpx.ProtocolError,
    aiohttp.ClientSSLError: httpx.ProtocolError,
    aiohttp.ServerFingerprintMismatch: httpx.ProtocolError,
    # Network related exceptions
    aiohttp.ClientConnectorError: httpx.ConnectError,
    aiohttp.ClientOSError: httpx.ConnectError,
    aiohttp.ClientPayloadError: httpx.ReadError,
    # Connection disconnection exceptions
    aiohttp.ServerDisconnectedError: httpx.ReadError,
    # Response related exceptions
    aiohttp.ClientConnectionError: httpx.NetworkError,
    aiohttp.ClientPayloadError: httpx.ReadError,
    aiohttp.ContentTypeError: httpx.ReadError,
    aiohttp.TooManyRedirects: httpx.TooManyRedirects,
    # URL related exceptions
    aiohttp.InvalidURL: httpx.InvalidURL,
    # Base exceptions
    aiohttp.ClientError: httpx.RequestError,
}

# Add client_exceptions module exceptions
try:
    import aiohttp.client_exceptions

    AIOHTTP_EXC_MAP[aiohttp.client_exceptions.ClientPayloadError] = httpx.ReadError
except ImportError:
    pass


@contextlib.contextmanager
def map_aiohttp_exceptions() -> typing.Iterator[None]:
    try:
        yield
    except Exception as exc:
        mapped_exc = None

        for from_exc, to_exc in AIOHTTP_EXC_MAP.items():
            if not isinstance(exc, from_exc):  # type: ignore
                continue
            if mapped_exc is None or issubclass(to_exc, mapped_exc):
                mapped_exc = to_exc

        if mapped_exc is None:  # pragma: no cover
            raise

        message = str(exc)
        raise mapped_exc(message) from exc


class AiohttpResponseStream(httpx.AsyncByteStream):
    CHUNK_SIZE = 1024 * 16

    def __init__(self, aiohttp_response: ClientResponse) -> None:
        self._aiohttp_response = aiohttp_response

    async def __aiter__(self) -> typing.AsyncIterator[bytes]:
        try:
            async for chunk in self._aiohttp_response.content.iter_chunked(
                self.CHUNK_SIZE
            ):
                yield chunk
        except (
            aiohttp.ClientPayloadError,
            aiohttp.client_exceptions.ClientPayloadError,
        ) as e:
            # Handle incomplete transfers more gracefully
            # Log the error but don't re-raise if we've already yielded some data
            verbose_logger.debug(f"Transfer incomplete, but continuing: {e}")
            # If the error is due to incomplete transfer encoding, we can still
            # return what we've received so far, similar to how httpx handles it
            return
        except aiohttp.http_exceptions.TransferEncodingError as e:
            # Handle transfer encoding errors gracefully
            verbose_logger.debug(f"Transfer encoding error, but continuing: {e}")
            return
        except Exception:
            # For other exceptions, use the normal mapping
            with map_aiohttp_exceptions():
                raise

    async def aclose(self) -> None:
        with map_aiohttp_exceptions():
            await self._aiohttp_response.__aexit__(None, None, None)


class AiohttpTransport(httpx.AsyncBaseTransport):
    def __init__(
        self, client: Union[ClientSession, Callable[[], ClientSession]]
    ) -> None:
        self.client = client

    async def aclose(self) -> None:
        if isinstance(self.client, ClientSession):
            await self.client.close()


class LiteLLMAiohttpTransport(AiohttpTransport):
    """
    LiteLLM wrapper around AiohttpTransport to handle %-encodings in URLs
    and event loop lifecycle issues in CI/CD environments

    Credit to: https://github.com/karpetrosyan/httpx-aiohttp for this implementation
    """

    def __init__(self, client: Union[ClientSession, Callable[[], ClientSession]]):
        self.client = client
        super().__init__(client=client)
        # Store the client factory for recreating sessions when needed
        if callable(client):
            self._client_factory = client
        
        # Performance optimization: Cache session validation to avoid expensive checks
        self._session_validated = False
        self._cached_loop_id = None
        self._session_error_count = 0
        self._max_session_errors = 3  # Recreate session after this many errors

    def _get_valid_client_session(self) -> ClientSession:
        """
        Optimized helper to get a valid ClientSession for the current event loop.
        
        This version minimizes expensive operations and caches validation results
        to improve performance.
        """
        from aiohttp.client import ClientSession

        # Fast path: If we have a valid session and it's been validated recently, use it
        if (isinstance(self.client, ClientSession) and 
            self._session_validated and 
            self._session_error_count < self._max_session_errors):
            try:
                # Quick check: is the session still open?
                if not self.client.closed:
                    return self.client
            except Exception:
                # If any error occurs, fall through to recreation
                pass

        # Create or recreate session
        self._create_new_session()
        
        # Ensure we always return a ClientSession
        if isinstance(self.client, ClientSession):
            return self.client
        else:
            # This should not happen after _create_new_session, but handle it safely
            self.client = ClientSession()
            return self.client

    def _create_new_session(self) -> None:
        """Create a new session and update validation cache."""
        current_loop = None
        current_loop_id = None
        
        try:
            # Get current event loop ID for caching
            try:
                current_loop = asyncio.get_running_loop()
                current_loop_id = id(current_loop)
            except RuntimeError:
                current_loop_id = None

            # Close existing session if it exists and is different loop
            if (isinstance(self.client, ClientSession) and 
                not self.client.closed and 
                self._cached_loop_id != current_loop_id):
                try:
                    # Schedule closure without waiting since it might be from different loop
                    if current_loop and hasattr(self.client, '_loop') and self.client._loop != current_loop:
                        # Don't await close() if it's from a different loop
                        pass
                    else:
                        # Same loop, can close properly
                        asyncio.create_task(self.client.close())
                except Exception as e:
                    verbose_logger.debug(f"Error closing old session: {e}")

            # Create new session
            if hasattr(self, "_client_factory") and callable(self._client_factory):
                self.client = self._client_factory()
            else:
                self.client = ClientSession()

            # Update cache
            self._session_validated = True
            self._cached_loop_id = current_loop_id
            self._session_error_count = 0

        except Exception as e:
            verbose_logger.debug(f"Error creating new session: {e}")
            # Fallback: create basic session
            self.client = ClientSession()
            self._session_validated = True
            self._session_error_count = 0

    def _handle_session_error(self) -> None:
        """Handle session errors by incrementing error count and invalidating cache."""
        self._session_error_count += 1
        if self._session_error_count >= self._max_session_errors:
            self._session_validated = False

    async def handle_async_request(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        from aiohttp import ClientTimeout
        from yarl import URL as YarlURL

        timeout = request.extensions.get("timeout", {})
        sni_hostname = request.extensions.get("sni_hostname")

        # Use optimized helper to get session (minimal overhead)
        client_session = self._get_valid_client_session()

        try:
            with map_aiohttp_exceptions():
                try:
                    data = request.content
                except httpx.RequestNotRead:
                    data = request.stream  # type: ignore
                    request.headers.pop("transfer-encoding", None)  # handled by aiohttp

                response = await client_session.request(
                    method=request.method,
                    url=YarlURL(str(request.url), encoded=True),
                    headers=request.headers,
                    data=data,
                    allow_redirects=False,
                    auto_decompress=False,
                    timeout=ClientTimeout(
                        sock_connect=timeout.get("connect"),
                        sock_read=timeout.get("read"),
                        connect=timeout.get("pool"),
                    ),
                    server_hostname=sni_hostname,
                ).__aenter__()

            return httpx.Response(
                status_code=response.status,
                headers=response.headers,
                content=AiohttpResponseStream(response),
                request=request,
            )
        except Exception as e:
            # Handle session errors
            self._handle_session_error()
            raise
