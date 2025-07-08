import asyncio
import contextlib
import os
import typing
import urllib.request
from typing import Callable, Dict, Union

import aiohttp
import aiohttp.client_exceptions
import aiohttp.http_exceptions
import httpx
from aiohttp.client import ClientResponse, ClientSession

import litellm
from litellm._logging import verbose_logger
from litellm.secret_managers.main import str_to_bool

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

    def _get_valid_client_session(self) -> ClientSession:
        """
        Helper to get a valid ClientSession for the current event loop.

        This handles the case where the session was created in a different
        event loop that may have been closed (common in CI/CD environments).
        """
        from aiohttp.client import ClientSession

        # If we don't have a client or it's not a ClientSession, create one
        if not isinstance(self.client, ClientSession):
            if hasattr(self, "_client_factory") and callable(self._client_factory):
                self.client = self._client_factory()
            else:
                self.client = ClientSession()
            return self.client

        # Check if the existing session is still valid for the current event loop
        try:
            session_loop = getattr(self.client, "_loop", None)
            current_loop = asyncio.get_running_loop()

            # If session is from a different or closed loop, recreate it
            if (
                session_loop is None
                or session_loop != current_loop
                or session_loop.is_closed()
            ):
                # Clean up the old session
                try:
                    # Note: not awaiting close() here as it might be from a different loop
                    # The session will be garbage collected
                    pass
                except Exception as e:
                    verbose_logger.debug(f"Error closing old session: {e}")
                    pass

                # Create a new session in the current event loop
                if hasattr(self, "_client_factory") and callable(self._client_factory):
                    self.client = self._client_factory()
                else:
                    self.client = ClientSession()

        except (RuntimeError, AttributeError):
            # If we can't check the loop or session is invalid, recreate it
            if hasattr(self, "_client_factory") and callable(self._client_factory):
                self.client = self._client_factory()
            else:
                self.client = ClientSession()

        return self.client
    
    async def handle_async_request(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        from aiohttp import ClientTimeout
        from yarl import URL as YarlURL

        timeout = request.extensions.get("timeout", {})
        sni_hostname = request.extensions.get("sni_hostname")

        # Use helper to ensure we have a valid session for the current event loop
        client_session = self._get_valid_client_session()

        # Resolve proxy settings from environment variables
        proxy = await self._get_proxy_settings(request)

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
                proxy=proxy,
                server_hostname=sni_hostname,
            ).__aenter__()

        return httpx.Response(
            status_code=response.status,
            headers=response.headers,
            content=AiohttpResponseStream(response),
            request=request,
        )
    

    async def _get_proxy_settings(self, request: httpx.Request):
        proxy = None
        if not (
            litellm.disable_aiohttp_trust_env
            or str_to_bool(os.getenv("DISABLE_AIOHTTP_TRUST_ENV", "False"))
        ):
            try:
                proxy = self._proxy_from_env(request.url)
            except Exception as e:  # pragma: no cover - best effort
                verbose_logger.debug(f"Error reading proxy env: {e}")

        return proxy
    

    def _proxy_from_env(self, url: httpx.URL) -> typing.Optional[str]:
        """Return proxy URL from env for the given request URL."""
        proxies = urllib.request.getproxies()
        if urllib.request.proxy_bypass(url.host):
            return None

        proxy = proxies.get(url.scheme) or proxies.get("all")
        if proxy and "://" not in proxy:
            proxy = f"http://{proxy}"
        return proxy
