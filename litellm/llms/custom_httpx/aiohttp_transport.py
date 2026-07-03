import asyncio
import concurrent.futures
import contextlib
import os
import ssl
import typing
import urllib.request
from typing import Any, Callable, ClassVar, Dict, Optional, Union

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
    asyncio.TimeoutError: httpx.TimeoutException,
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
            async for chunk in self._aiohttp_response.content.iter_chunked(self.CHUNK_SIZE):
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
        except RuntimeError as e:
            # Some providers (e.g., SSE streams) may close the connection
            # causing aiohttp StreamReader to raise a generic RuntimeError
            # with message "Connection closed.". Treat this as a graceful
            # end-of-stream so downstream consumers don't error.
            if "Connection closed" in str(e):
                verbose_logger.debug("Upstream closed streaming connection; ending iterator gracefully")
                return
            raise
        except aiohttp.http_exceptions.TransferEncodingError as e:
            # Handle transfer encoding errors gracefully
            verbose_logger.debug(f"Transfer encoding error, but continuing: {e}")
            return
        except Exception:
            # For other exceptions, use the normal mapping
            with map_aiohttp_exceptions():
                raise
        finally:
            # Release the aiohttp connection when iteration ends for any
            # reason (read timeout, cancellation from a client disconnect,
            # GeneratorExit). Without this, abnormally terminated streams
            # permanently hold a slot in the TCPConnector pool; once the
            # pool is exhausted every request to that host times out (408)
            # until the proxy is restarted, even after the backend recovers.
            # On a fully-read response the connection was already released
            # at EOF and close() is a no-op.
            self._aiohttp_response.close()

    async def aclose(self) -> None:
        with map_aiohttp_exceptions():
            await self._aiohttp_response.__aexit__(None, None, None)


class AiohttpTransport(httpx.AsyncBaseTransport):
    def __init__(
        self,
        client: Union[ClientSession, Callable[[], ClientSession]],
        owns_session: bool = True,
    ) -> None:
        self.client = client
        self._owns_session = owns_session

        #########################################################
        # Class variables for proxy settings
        #########################################################
        self.proxy_cache: Dict[str, Optional[str]] = {}

    async def aclose(self) -> None:
        if self._owns_session and isinstance(self.client, ClientSession):
            await self.client.close()


class LiteLLMAiohttpTransport(AiohttpTransport):
    """
    LiteLLM wrapper around AiohttpTransport to handle %-encodings in URLs
    and event loop lifecycle issues in CI/CD environments

    Credit to: https://github.com/karpetrosyan/httpx-aiohttp for this implementation
    """

    # Strong references to scheduled session-close tasks. A bare
    # asyncio.create_task() result may be garbage-collected before it runs,
    # leaving the recycled session unclosed ("Unclosed client session").
    _background_close_tasks: ClassVar[set["asyncio.Task[None]"]] = set()

    def __init__(
        self,
        client: Union[ClientSession, Callable[[], ClientSession]],
        ssl_verify: Optional[Union[bool, ssl.SSLContext]] = None,
        owns_session: bool = True,
    ):
        self.client = client
        self._ssl_verify = ssl_verify  # Store for per-request SSL override
        super().__init__(client=client, owns_session=owns_session)
        # Store the client factory for recreating sessions when needed
        if callable(client):
            self._client_factory = client

    @classmethod
    def _on_close_task_done(cls, task: "asyncio.Task[None]") -> None:
        cls._background_close_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            verbose_logger.debug("Error closing recycled aiohttp session: %s", exc)

    @staticmethod
    def _on_threadsafe_close_done(future: "concurrent.futures.Future[None]") -> None:
        exc = future.exception()
        if exc is not None:
            verbose_logger.debug("Error closing recycled aiohttp session on its own loop: %s", exc)

    @staticmethod
    def _mark_connector_closed(session: ClientSession) -> None:
        """Synchronously dispose a session whose event loop is gone.

        An async close can no longer run on a closed loop. BaseConnector._close
        is the same synchronous teardown aiohttp's own finalizer (__del__)
        uses: it is guarded for closed loops, releases pooled connections, and
        flips the flags that ClientSession.closed / BaseConnector.closed read -
        so no "Unclosed client session" / "Unclosed connector" warnings reach
        the event-loop exception handler at garbage collection.
        """
        connector = getattr(session, "_connector", None)
        close_sync = getattr(connector, "_close", None)
        if not callable(close_sync):
            return
        try:
            close_sync()
        except (RuntimeError, AttributeError, OSError) as e:
            verbose_logger.debug("Best-effort connector close failed: %s", e)

    def _close_recycled_session(self, session: ClientSession) -> None:
        """Deterministically dispose a ClientSession this transport is replacing.

        Covers the three lifecycles a recycled session can be in:
        - its loop is the current running loop: schedule an async close and keep
          a strong reference to the task until it completes;
        - its loop is still running elsewhere (e.g. another thread): hand the
          close to that loop thread-safely;
        - its loop is stopped or closed, or there is no running loop: fall
          back to the synchronous finalizer-safe teardown.
        """
        if session.closed:
            return

        session_loop = getattr(session, "_loop", None)
        try:
            current_loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if session_loop is not None and session_loop is not current_loop:
            if not session_loop.is_closed() and session_loop.is_running():
                # The session's loop is running somewhere else (e.g. another
                # thread): closing from here would touch that loop's internals
                # unsafely; hand the close to its own loop.
                try:
                    future = asyncio.run_coroutine_threadsafe(session.close(), session_loop)
                except RuntimeError as e:  # loop shut down between the checks
                    verbose_logger.debug("Threadsafe session close failed: %s", e)
                    self._mark_connector_closed(session)
                else:
                    future.add_done_callback(self._on_threadsafe_close_done)
                return

            # Foreign loop that is stopped or closed: an async close can no
            # longer run there, and running it on the current loop would touch
            # another loop's internals. Dispose synchronously instead.
            self._mark_connector_closed(session)
            return

        if current_loop is None:
            self._mark_connector_closed(session)
            return

        task = current_loop.create_task(session.close())
        cls = type(self)
        cls._background_close_tasks.add(task)
        task.add_done_callback(cls._on_close_task_done)

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
            # Don't return yet - check if the newly created session is valid

        # Check if the session itself is closed
        if self.client.closed:
            verbose_logger.debug("Session is closed, creating new session")
            # Create a new session
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
            if session_loop is None or session_loop != current_loop or session_loop.is_closed():
                # Close old session to prevent leaks
                old_session = self.client
                try:
                    self._close_recycled_session(old_session)
                except Exception as e:
                    verbose_logger.debug(f"Error closing old session: {e}")

                # Create a new session in the current event loop
                if hasattr(self, "_client_factory") and callable(self._client_factory):
                    self.client = self._client_factory()
                else:
                    self.client = ClientSession()

        except (RuntimeError, AttributeError) as e:
            # If we can't check the loop or session is invalid, recreate it,
            # but still dispose of the session being replaced.
            old_session = self.client
            try:
                self._close_recycled_session(old_session)
            except (RuntimeError, AttributeError, OSError) as close_error:
                verbose_logger.debug(f"Error closing old session: {close_error}")
            if hasattr(self, "_client_factory") and callable(self._client_factory):
                self.client = self._client_factory()
            else:
                self.client = ClientSession()
            verbose_logger.debug(f"Error checking session loop, created new session: {e}")

        return self.client

    async def _make_aiohttp_request(
        self,
        client_session: ClientSession,
        request: httpx.Request,
        timeout: dict,
        proxy: Optional[str],
        sni_hostname: Optional[str],
        ssl_verify: Optional[Union[bool, ssl.SSLContext]] = None,
    ) -> ClientResponse:
        """
        Helper function to make an aiohttp request with the given parameters.

        Args:
            client_session: The aiohttp ClientSession to use
            request: The httpx Request to send
            timeout: Timeout settings dict with 'connect', 'read', 'pool' keys
            proxy: Optional proxy URL
            sni_hostname: Optional SNI hostname for SSL
            ssl_verify: Optional SSL verification setting (False to disable, SSLContext for custom)

        Returns:
            ClientResponse from aiohttp
        """
        from aiohttp import ClientTimeout
        from yarl import URL as YarlURL

        try:
            # Coerce an empty body to None so aiohttp does not attach a
            # `Content-Type: application/octet-stream` header for bodyless
            # requests (e.g. DELETE /responses/{id}), which upstream APIs reject.
            data = request.content or None
        except httpx.RequestNotRead:
            data = request.stream  # type: ignore
            request.headers.pop("transfer-encoding", None)  # handled by aiohttp

        # Only pass ssl kwarg when explicitly configured, to avoid
        # overriding the session/connector defaults with None (which is
        # not a valid value for aiohttp's ssl parameter).
        request_kwargs: Dict[str, Any] = {
            "method": request.method,
            "url": YarlURL(str(request.url), encoded=True),
            "headers": request.headers,
            "data": data,
            "allow_redirects": False,
            "auto_decompress": False,
            "timeout": ClientTimeout(
                sock_connect=timeout.get("connect"),
                sock_read=timeout.get("read"),
                connect=timeout.get("pool"),
            ),
            "proxy": proxy,
            "server_hostname": sni_hostname,
        }
        if ssl_verify is not None:
            request_kwargs["ssl"] = ssl_verify

        response = await client_session.request(**request_kwargs).__aenter__()

        return response

    async def handle_async_request(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        timeout = request.extensions.get("timeout", {})
        sni_hostname = request.extensions.get("sni_hostname")

        # Use helper to ensure we have a valid session for the current event loop
        client_session = self._get_valid_client_session()

        # Resolve proxy settings from environment variables
        proxy = await self._get_proxy_settings(request)

        # Use stored SSL configuration for per-request override
        ssl_config = self._ssl_verify

        try:
            with map_aiohttp_exceptions():
                response = await self._make_aiohttp_request(
                    client_session=client_session,
                    request=request,
                    timeout=timeout,
                    proxy=proxy,
                    sni_hostname=sni_hostname,
                    ssl_verify=ssl_config,
                )
        except RuntimeError as e:
            # Handle the case where session was closed between our check and actual use
            if "Session is closed" in str(e):
                verbose_logger.debug(f"Session closed during request, retrying with new session: {e}")
                # Dispose of the session that actually faulted. Do NOT read
                # self.client here: a concurrent task may already have
                # replaced it with a healthy session that must stay open.
                # Guarded by isinstance: factory-injected sessions may be
                # duck-typed test doubles without a close() coroutine.
                if isinstance(client_session, ClientSession):
                    self._close_recycled_session(client_session)
                if hasattr(self, "_client_factory") and callable(self._client_factory):
                    self.client = self._client_factory()
                else:
                    self.client = ClientSession()
                client_session = self.client

                # Retry the request with the new session
                with map_aiohttp_exceptions():
                    response = await self._make_aiohttp_request(
                        client_session=client_session,
                        request=request,
                        timeout=timeout,
                        proxy=proxy,
                        sni_hostname=sni_hostname,
                        ssl_verify=ssl_config,
                    )
            else:
                # Re-raise if it's a different RuntimeError
                raise

        return httpx.Response(
            status_code=response.status,
            headers=response.headers,
            stream=AiohttpResponseStream(response),
            request=request,
        )

    async def _get_proxy_settings(self, request: httpx.Request):
        proxy = None
        if not (litellm.disable_aiohttp_trust_env or str_to_bool(os.getenv("DISABLE_AIOHTTP_TRUST_ENV", "False"))):
            try:
                proxy = self._proxy_from_env(request.url)
            except Exception as e:  # pragma: no cover - best effort
                verbose_logger.debug(f"Error reading proxy env: {e}")

        return proxy

    def _proxy_from_env(self, url: httpx.URL) -> typing.Optional[str]:
        """
        Return proxy URL from env for the given request URL

        Only check the proxy env settings once, this is a costly operation for CPU % usage

        ."""
        #########################################################
        # Check if we've already checked the proxy env settings
        #########################################################
        proxy_cache_key = url.host

        if proxy_cache_key in self.proxy_cache:
            return self.proxy_cache[proxy_cache_key]

        proxies = urllib.request.getproxies()
        if urllib.request.proxy_bypass(url.host):
            proxy_url = None
        else:
            proxy = proxies.get(url.scheme) or proxies.get("all")
            if proxy and "://" not in proxy:
                proxy = f"http://{proxy}"
            proxy_url = proxy

        self.proxy_cache[proxy_cache_key] = proxy_url

        return proxy_url
